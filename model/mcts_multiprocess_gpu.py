# -*- coding: utf-8 -*-
"""
真正的多进程并行 + GPU 批量推理
使用 torch.multiprocessing 实现高效的进程间通信
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import math
import queue
from typing import List, Tuple


def mcts_worker(worker_id, game_args, mcts_args, request_queue, result_queues, num_games, seed_base):
    """
    Worker 进程：执行 MCTS 搜索，通过队列请求 GPU 推理
    
    Args:
        worker_id: Worker ID
        game_args: 游戏参数
        mcts_args: MCTS 参数
        request_queue: 推理请求队列（发送给主进程）
        result_queues: 结果队列列表（每个 worker 一个）
        num_games: 此 worker 负责的游戏数
        seed_base: 随机种子基数
    """
    from model.game import DotsAndBoxesGame
    
    # 设置随机种子
    np.random.seed(seed_base + worker_id)
    
    # 初始化游戏
    game = DotsAndBoxesGame(**game_args)
    result_queue = result_queues[worker_id]
    
    all_examples = []
    
    # 执行多局游戏
    for game_idx in range(num_games):
        examples = _play_one_game(
            game, mcts_args, request_queue, result_queue, 
            worker_id, game_idx
        )
        all_examples.extend(examples)
    
    return all_examples


def _play_one_game(game, args, request_queue, result_queue, worker_id, game_idx):
    """执行一局游戏"""
    trees = {}  # MCTS 树
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    examples = []
    request_counter = 0
    
    while True:
        # 执行 MCTS
        for _ in range(args['num_simulations']):
            _mcts_search_one_sim(
                state, game, trees, args,
                request_queue, result_queue,
                worker_id, request_counter
            )
            request_counter += 1
        
        # 计算动作概率
        s = str(state)
        if s not in trees or 'Nsa' not in trees[s]:
            valids = game.get_valid_moves(state)
            pi = valids / np.sum(valids)
        else:
            counts = [trees[s]['Nsa'].get(a, 0) for a in range(game.get_action_size())]
            temp = int(episode_step < args['temp_threshold'])
            
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
            noise = np.random.dirichlet([args['dirichlet_alpha']] * len(pi))
            pi = (1 - args['dirichlet_epsilon']) * pi + args['dirichlet_epsilon'] * noise
            valids = game.get_valid_moves(state)
            pi = pi * valids
            if np.sum(pi) > 0:
                pi = pi / np.sum(pi)
        
        # 记录样本
        canonical_board = game.get_observation(state)
        examples.append([canonical_board, cur_player, pi, None])
        
        # 执行动作
        action = np.random.choice(len(pi), p=pi)
        state = game.get_next_state(state, action)
        episode_step += 1
        
        # 检查游戏是否结束
        r = game.get_game_result(state, cur_player)
        if r != 0:
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in examples]
        
        cur_player = game.get_current_player(state)


def _mcts_search_one_sim(state, game, trees, args, request_queue, result_queue, worker_id, request_id):
    """执行一次 MCTS 模拟"""
    current_state = state.clone()
    path = []
    
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
        
        # 叶子节点：请求 GPU 推理
        if s not in trees or 'Ps' not in trees[s]:
            obs = game.get_observation(current_state)
            valids = game.get_valid_moves(current_state)
            
            # 发送推理请求
            request_queue.put({
                'worker_id': worker_id,
                'request_id': request_id,
                'obs': obs,
                'valids': valids,
                'state_str': s
            })
            
            # 等待结果
            try:
                result = result_queue.get(timeout=5.0)
                pi = result['pi']
                v = result['v']
                
                # 初始化节点
                if s not in trees:
                    trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
                
                _backpropagate(trees, path, v)
                return
            except queue.Empty:
                # 超时，使用随机策略
                pi = valids / np.sum(valids)
                if s not in trees:
                    trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
                _backpropagate(trees, path, 0.0)
                return
        
        # 内部节点：UCB 选择
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


class MultiProcessGPUSelfPlay:
    """
    多进程 + GPU 批量推理
    
    架构：
    - 主进程：运行神经网络（GPU），批量处理推理请求
    - Worker 进程：并行执行 MCTS（CPU）
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.device = next(nnet.parameters()).device
        
        self.game_args = {
            'num_rows': args['num_rows'],
            'num_cols': args['num_cols']
        }
        
        self.mcts_args = {
            'num_simulations': args['num_simulations'],
            'cpuct': args['cpuct'],
            'dirichlet_alpha': args.get('dirichlet_alpha', 0.3),
            'dirichlet_epsilon': args.get('dirichlet_epsilon', 0.25),
            'temp_threshold': args['temp_threshold']
        }
    
    def execute_episodes_parallel(self, num_episodes):
        """多进程并行执行游戏"""
        num_workers = self.args.get('num_workers', 8)
        batch_size = self.args.get('mcts_batch_size', 32)
        
        print(f'  🚀 启动多进程 + GPU 批量推理')
        print(f'     Worker 进程: {num_workers}')
        print(f'     GPU 批量: {batch_size}')
        print(f'     总局数: {num_episodes}')
        
        # 创建队列
        mp_ctx = mp.get_context('spawn')
        request_queue = mp_ctx.Queue(maxsize=1000)
        result_queues = [mp_ctx.Queue() for _ in range(num_workers)]
        
        # 分配游戏到各个 worker
        games_per_worker = [num_episodes // num_workers] * num_workers
        for i in range(num_episodes % num_workers):
            games_per_worker[i] += 1
        
        # 启动 GPU 推理服务器（在主进程）
        stop_event = mp_ctx.Event()
        server_thread = mp.Process(
            target=self._gpu_inference_server,
            args=(request_queue, result_queues, stop_event, batch_size)
        )
        server_thread.start()
        
        time.sleep(1)  # 等待服务器启动
        
        # 启动 worker 进程
        start_time = time.time()
        seed_base = int(time.time()) % 10000
        
        with mp_ctx.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                mcts_worker,
                [(i, self.game_args, self.mcts_args, request_queue, result_queues, 
                  games_per_worker[i], seed_base) for i in range(num_workers)]
            )
        
        # 停止推理服务器
        stop_event.set()
        server_thread.join(timeout=2)
        if server_thread.is_alive():
            server_thread.terminate()
        
        # 合并结果
        all_examples = []
        for worker_examples in results:
            all_examples.extend(worker_examples)
        
        elapsed = time.time() - start_time
        
        print(f'  ✅ 多进程训练完成')
        print(f'    耗时: {elapsed:.2f}s')
        print(f'    速度: {num_episodes / elapsed:.2f} 局/秒')
        print(f'    样本数: {len(all_examples):,}')
        
        return all_examples
    
    def _gpu_inference_server(self, request_queue, result_queues, stop_event, batch_size):
        """GPU 推理服务器（主进程运行）"""
        print(f'  🔥 GPU 推理服务器启动')
        
        self.nnet.eval()
        timeout = 0.01  # 10ms
        
        while not stop_event.is_set():
            try:
                # 收集一批请求
                batch_requests = []
                start_time = time.time()
                
                # 等待第一个请求
                try:
                    first_req = request_queue.get(timeout=timeout)
                    batch_requests.append(first_req)
                except queue.Empty:
                    continue
                
                # 快速收集更多请求
                while len(batch_requests) < batch_size:
                    if time.time() - start_time > timeout:
                        break
                    try:
                        req = request_queue.get_nowait()
                        batch_requests.append(req)
                    except queue.Empty:
                        break
                
                if not batch_requests:
                    continue
                
                # 批量推理
                obs_list = [req['obs'] for req in batch_requests]
                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
                
                with torch.no_grad():
                    log_pi_batch, v_batch = self.nnet(obs_tensor)
                
                pi_batch = torch.exp(log_pi_batch).cpu().numpy()
                v_batch = v_batch.cpu().numpy().flatten()
                
                # 返回结果
                for idx, req in enumerate(batch_requests):
                    pi = pi_batch[idx] * req['valids']
                    if np.sum(pi) > 0:
                        pi = pi / np.sum(pi)
                    else:
                        pi = req['valids'] / np.sum(req['valids'])
                    
                    result_queues[req['worker_id']].put({
                        'request_id': req['request_id'],
                        'pi': pi,
                        'v': float(v_batch[idx])
                    })
                
            except Exception as e:
                print(f'  ⚠️ GPU 推理错误: {e}')
                time.sleep(0.01)
        
        print(f'  ✓ GPU 推理服务器停止')
