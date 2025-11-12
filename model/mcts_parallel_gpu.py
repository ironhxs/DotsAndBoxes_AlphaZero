# -*- coding: utf-8 -*-
"""
GPU 加速的多进程并行 MCTS
使用共享 GPU 推理队列，提升 GPU 利用率
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
from queue import Empty
import math


class SharedGPUInferenceServer:
    """
    共享 GPU 推理服务器（在主进程中运行）
    收集来自多个 worker 的推理请求，批量处理
    """
    
    def __init__(self, nnet, args, request_queue, result_dict, stop_event):
        self.nnet = nnet
        self.args = args
        self.request_queue = request_queue
        self.result_dict = result_dict
        self.stop_event = stop_event
        self.batch_size = args.get('mcts_batch_size', 32)
        self.timeout = 0.01  # 10ms 超时
        
    def run(self):
        """推理服务器主循环"""
        print(f"  🔥 GPU 推理服务器启动 (batch_size={self.batch_size})")
        
        device = next(self.nnet.parameters()).device
        self.nnet.eval()
        
        while not self.stop_event.is_set():
            try:
                # 收集一批请求
                batch_requests = []
                start_time = time.time()
                
                # 等待第一个请求
                try:
                    first_req = self.request_queue.get(timeout=self.timeout)
                    batch_requests.append(first_req)
                except Empty:
                    continue
                
                # 快速收集更多请求
                while len(batch_requests) < self.batch_size:
                    if time.time() - start_time > self.timeout:
                        break
                    try:
                        req = self.request_queue.get_nowait()
                        batch_requests.append(req)
                    except Empty:
                        break
                
                if not batch_requests:
                    continue
                
                # 批量推理
                obs_list = [req['obs'] for req in batch_requests]
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
                
                with torch.no_grad():
                    log_pi_batch, v_batch = self.nnet(obs_tensor)
                
                pi_batch = torch.exp(log_pi_batch).cpu().numpy()
                v_batch = v_batch.cpu().numpy().flatten()
                
                # 返回结果
                for idx, req in enumerate(batch_requests):
                    self.result_dict[req['id']] = (pi_batch[idx], float(v_batch[idx]))
                
            except Exception as e:
                print(f"  ⚠️ GPU 推理错误: {e}")
                time.sleep(0.01)
        
        print(f"  ✓ GPU 推理服务器停止")


def worker_play_game_gpu(args_tuple):
    """
    Worker 函数：使用共享 GPU 推理服务执行游戏
    
    Args:
        args_tuple: (game_args, mcts_args, request_queue, result_dict, seed)
    """
    from model.game import DotsAndBoxesGame
    
    game_args, mcts_args, request_queue, result_dict, seed = args_tuple
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化游戏
    game = DotsAndBoxesGame(**game_args)
    
    # 初始化 MCTS（不需要网络）
    mcts_trees = {}  # {state_str: {Ps, Ns, Qsa, Nsa}}
    
    # 执行一局游戏
    examples = []
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    request_counter = seed * 10000
    
    while True:
        # MCTS 搜索
        canonical_board = game.get_observation(state)
        
        # 执行 num_simulations 次模拟
        for _ in range(mcts_args['num_simulations']):
            _mcts_search(state, game, mcts_trees, mcts_args, 
                        request_queue, result_dict, request_counter)
            request_counter += 1
        
        # 计算动作概率
        s = str(state)
        counts = [mcts_trees[s]['Nsa'].get(a, 0) for a in range(game.get_action_size())]
        
        temp = int(episode_step < mcts_args['temp_threshold'])
        if temp == 0:
            best_actions = np.where(counts == np.max(counts))[0]
            pi = np.zeros(len(counts), dtype=np.float32)
            pi[best_actions] = 1.0 / len(best_actions)
        else:
            counts_temp = np.array([x ** (1.0 / temp) for x in counts], dtype=np.float32)
            counts_sum = float(np.sum(counts_temp))
            if counts_sum > 0:
                pi = counts_temp / counts_sum
            else:
                valids = game.get_valid_moves(state)
                pi = valids / np.sum(valids)
        
        # 添加探索噪声
        if episode_step <= 30:
            noise = np.random.dirichlet([mcts_args['dirichlet_alpha']] * len(pi))
            pi = (1 - mcts_args['dirichlet_epsilon']) * pi + mcts_args['dirichlet_epsilon'] * noise
            valids = game.get_valid_moves(state)
            pi = pi * valids
            if np.sum(pi) > 0:
                pi = pi / np.sum(pi)
            else:
                pi = valids / np.sum(valids)
        
        # 记录样本
        examples.append([canonical_board, cur_player, pi, None])
        
        # 执行动作
        action = np.random.choice(len(pi), p=pi)
        state = game.get_next_state(state, action)
        episode_step += 1
        
        # 检查游戏是否结束
        r = game.get_game_result(state, cur_player)
        
        if r != 0:
            # 游戏结束
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in examples]
        
        # 更新当前玩家
        cur_player = game.get_current_player(state)


def _mcts_search(state, game, trees, args, request_queue, result_dict, request_id):
    """执行一次 MCTS 搜索"""
    path = []
    current_state = state.clone()
    
    while True:
        s = str(current_state)
        
        # 终止状态
        if current_state.is_terminal():
            returns = current_state.returns()
            if returns[0] > returns[1]:
                value = 1.0
            elif returns[0] < returns[1]:
                value = -1.0
            else:
                value = 0.0
            _backpropagate(trees, path, value)
            return
        
        # 叶子节点：需要网络评估
        if s not in trees or 'Ps' not in trees[s]:
            # 请求 GPU 推理
            obs = game.get_observation(current_state)
            request_queue.put({'id': request_id, 'obs': obs})
            
            # 等待结果
            max_wait = 100  # 最多等待 100 次
            for _ in range(max_wait):
                if request_id in result_dict:
                    pi, v = result_dict.pop(request_id)
                    
                    # 应用合法动作掩码
                    valids = game.get_valid_moves(current_state)
                    pi = pi * valids
                    if np.sum(pi) > 0:
                        pi = pi / np.sum(pi)
                    else:
                        pi = valids / np.sum(valids)
                    
                    # 初始化节点
                    if s not in trees:
                        trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
                    else:
                        trees[s]['Ps'] = pi
                        trees[s]['Ns'] = 0
                    
                    _backpropagate(trees, path, v)
                    return
                time.sleep(0.0001)  # 100us
            
            # 超时，使用随机策略
            valids = game.get_valid_moves(current_state)
            pi = valids / np.sum(valids)
            if s not in trees:
                trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
            _backpropagate(trees, path, 0.0)
            return
        
        # 选择动作
        valids = game.get_valid_moves(current_state)
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(game.get_action_size()):
            if not valids[a]:
                continue
            
            if a in trees[s]['Qsa']:
                u = trees[s]['Qsa'][a] + args['cpuct'] * trees[s]['Ps'][a] * \
                    math.sqrt(trees[s]['Ns']) / (1 + trees[s]['Nsa'][a])
            else:
                u = args['cpuct'] * trees[s]['Ps'][a] * math.sqrt(trees[s]['Ns'] + 1e-8)
            
            if u > cur_best:
                cur_best = u
                best_act = a
        
        if best_act == -1:
            legal_actions = np.where(valids > 0)[0]
            if len(legal_actions) == 0:
                _backpropagate(trees, path, 0.0)
                return
            best_act = np.random.choice(legal_actions)
        
        path.append((s, best_act))
        current_state = game.get_next_state(current_state, best_act)


def _backpropagate(trees, path, value):
    """回传价值"""
    for s, a in reversed(path):
        if s not in trees:
            trees[s] = {'Ps': None, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
        
        if a in trees[s]['Qsa']:
            trees[s]['Qsa'][a] = (trees[s]['Nsa'][a] * trees[s]['Qsa'][a] + value) / (trees[s]['Nsa'][a] + 1)
            trees[s]['Nsa'][a] += 1
        else:
            trees[s]['Qsa'][a] = value
            trees[s]['Nsa'][a] = 1
        
        trees[s]['Ns'] += 1
        value = -value


class GPUAcceleratedSelfPlay:
    """
    GPU 加速的多进程自我对弈
    主进程运行 GPU 推理服务器，worker 进程执行游戏逻辑
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        self.game_args = {
            'num_rows': args['num_rows'],
            'num_cols': args['num_cols']
        }
        
        self.mcts_args = {
            'num_simulations': args['num_simulations'],
            'cpuct': args['cpuct'],
            'dirichlet_alpha': args.get('dirichlet_alpha', 0.3),
            'dirichlet_epsilon': args.get('dirichlet_epsilon', 0.25),
            'temp_threshold': args['temp_threshold'],
            'mcts_batch_size': args.get('mcts_batch_size', 32)
        }
    
    def execute_episodes_parallel(self, num_episodes):
        """使用多进程 + GPU 批量推理执行游戏"""
        num_workers = self.args.get('num_workers', 8)
        
        print(f'  🚀 启动 GPU 加速并行自我对弈')
        print(f'     CPU 进程: {num_workers}')
        print(f'     GPU 批量: {self.mcts_args["mcts_batch_size"]}')
        print(f'     总局数: {num_episodes}')
        
        # 创建进程间通信
        mp_ctx = mp.get_context('spawn')
        request_queue = mp_ctx.Queue(maxsize=1000)
        manager = mp_ctx.Manager()
        result_dict = manager.dict()
        stop_event = mp_ctx.Event()
        
        # 启动 GPU 推理服务器
        server = SharedGPUInferenceServer(self.nnet, self.args, request_queue, result_dict, stop_event)
        server_process = mp_ctx.Process(target=server.run, daemon=True)
        server_process.start()
        
        time.sleep(1)  # 等待服务器启动
        
        # 准备 worker 参数
        base_seed = int(time.time()) % (2**16)
        worker_args = [
            (self.game_args, self.mcts_args, request_queue, result_dict,
             (base_seed * 10000 + i) % (2**32 - 1))
            for i in range(num_episodes)
        ]
        
        start_time = time.time()
        
        # 启动 worker 进程池
        with mp_ctx.Pool(processes=num_workers) as pool:
            results = pool.map(worker_play_game_gpu, worker_args)
        
        # 停止推理服务器
        stop_event.set()
        server_process.join(timeout=2)
        if server_process.is_alive():
            server_process.terminate()
        
        # 合并结果
        all_examples = []
        for game_examples in results:
            all_examples.extend(game_examples)
        
        elapsed_time = time.time() - start_time
        
        print(f'  ✅ GPU 加速自我对弈完成')
        print(f'    耗时: {elapsed_time:.2f}s')
        print(f'    速度: {num_episodes / elapsed_time:.2f} 局/秒')
        print(f'    样本数: {len(all_examples):,}')
        
        return all_examples
