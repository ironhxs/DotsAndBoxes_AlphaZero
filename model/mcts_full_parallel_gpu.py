# -*- coding: utf-8 -*-
"""
多 GPU 副本并行训练
每个 worker 进程独立运行神经网络（GPU），完全并行，GPU 利用率最高
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import time
import math
from multiprocessing import Manager
from tqdm import tqdm


def worker_self_play_with_gpu(worker_id, game_args, mcts_args, nnet_state_dict, num_games, seed_base, progress_dict):
    """
    Worker 进程：独立运行神经网络（GPU），执行自我对弈
    
    每个进程：
    1. 加载神经网络到 GPU
    2. 独立执行 MCTS + GPU 推理
    3. 完全并行，无通信开销
    
    Args:
        worker_id: Worker ID
        game_args: 游戏参数
        mcts_args: MCTS 参数
        nnet_state_dict: 神经网络权重（共享内存）
        num_games: 此 worker 负责的游戏数
        seed_base: 随机种子基数
        progress_dict: 共享进度字典
    """
    from model.game import DotsAndBoxesGame
    from model.model import DotsAndBoxesNet
    
    # 设置随机种子
    np.random.seed(seed_base + worker_id)
    torch.manual_seed(seed_base + worker_id)
    
    # 初始化游戏
    game = DotsAndBoxesGame(**game_args)
    
    # 初始化神经网络（每个 worker 独立的 GPU 副本）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet = DotsAndBoxesNet(
        game=game,
        args=mcts_args
    ).to(device)
    
    # 加载权重，允许部分匹配（因为模型可能有动态创建的参数）
    nnet.load_state_dict(nnet_state_dict, strict=False)
    nnet.eval()
    
    # 执行多局游戏
    all_examples = []
    for game_idx in range(num_games):
        try:
            examples = _play_one_game_gpu(game, nnet, mcts_args, device)
            all_examples.extend(examples)
            # 更新进度
            progress_dict[worker_id] = game_idx + 1
        except Exception as e:
            print(f'  ⚠️  Worker {worker_id} 游戏 {game_idx+1} 失败: {e}')
            continue
    
    # 清理 worker 的 GPU 显存
    del nnet, game
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_examples


def _play_one_game_gpu(game, nnet, args, device):
    """执行一局游戏（GPU 推理）"""
    trees = {}  # 每局游戏独立的 MCTS 树
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    examples = []
    max_steps = 120  # 防止无限循环
    
    while episode_step < max_steps:
        # 确保状态不被污染：传入新克隆的状态
        state_for_mcts = state.clone()
        
        # 执行 MCTS（GPU 批量推理）
        _mcts_with_batching(state_for_mcts, game, trees, nnet, args, device)
        
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
        
        # 执行动作（增加安全检查）
        valids = game.get_valid_moves(state)
        action = np.random.choice(len(pi), p=pi)
        
        # 验证动作合法性
        if not valids[action]:
            legal_actions = np.where(valids > 0)[0]
            if len(legal_actions) == 0:
                break
            action = np.random.choice(legal_actions)
        
        state = game.get_next_state(state, action)
        episode_step += 1
        
        # 检查游戏是否结束
        r = game.get_game_result(state, cur_player)
        if r != 0:
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in examples]
        
        cur_player = game.get_current_player(state)


def _mcts_with_batching(state, game, trees, nnet, args, device):
    """
    执行 MCTS（支持批量 GPU 推理）
    
    策略：
    1. 收集多个模拟的叶子节点
    2. 批量 GPU 推理
    3. 回传价值
    """
    batch_size = args.get('mcts_batch_size', 32)
    pending_evals = []  # [(state, state_str, path)]
    
    for sim_idx in range(args['num_simulations']):
        current_state = state.clone()
        path = []
        leaf_eval = None
        
        # 搜索到叶子节点
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
                break
            
            # 叶子节点
            if s not in trees or 'Ps' not in trees[s]:
                leaf_eval = (current_state.clone(), s, path[:])
                break
            
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
                    break
                best_act = np.random.choice(legal_actions)
            
            path.append((s, best_act))
            current_state = game.get_next_state(current_state, best_act)
        
        # 收集叶子节点
        if leaf_eval:
            pending_evals.append(leaf_eval)
            
            # 达到批量大小或最后一次模拟，立即评估
            if len(pending_evals) >= batch_size or sim_idx == args['num_simulations'] - 1:
                _batch_evaluate_gpu(game, trees, pending_evals, nnet, device)
                pending_evals = []


def _batch_evaluate_gpu(game, trees, pending_evals, nnet, device):
    """批量 GPU 推理"""
    if not pending_evals:
        return
    
    # 准备批量输入
    observations = []
    valid_masks = []
    
    for state, s, path in pending_evals:
        observations.append(game.get_observation(state))
        valid_masks.append(game.get_valid_moves(state))
    
    # GPU 批量推理
    obs_tensor = torch.FloatTensor(np.array(observations)).to(device)
    
    with torch.no_grad():
        log_pi_batch, v_batch = nnet(obs_tensor)
    
    pi_batch = torch.exp(log_pi_batch).cpu().numpy()
    v_batch = v_batch.cpu().numpy().flatten()
    
    # 处理结果并回传
    for idx, (state, s, path) in enumerate(pending_evals):
        pi = pi_batch[idx] * valid_masks[idx]
        if np.sum(pi) > 0:
            pi = pi / np.sum(pi)
        else:
            pi = valid_masks[idx] / np.sum(valid_masks[idx])
        
        # 初始化节点
        if s not in trees:
            trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
        else:
            trees[s]['Ps'] = pi
        
        # 回传价值
        _backpropagate(trees, path, float(v_batch[idx]))


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


class FullParallelGPUSelfPlay:
    """
    完全并行 GPU 训练
    
    架构：每个 worker 进程独立运行神经网络（GPU）
    优势：
    1. 完全并行，无进程通信
    2. GPU 利用率最高（多个进程同时用 GPU）
    3. 显存充足时性能最佳
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # 共享神经网络权重（使用共享内存）
        self.nnet_state_dict = {k: v.cpu().share_memory_() for k, v in nnet.state_dict().items()}
        
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
            'mcts_batch_size': args.get('mcts_batch_size', 32),
            'num_res_blocks': args['num_res_blocks'],
            'num_filters': args['num_filters'],
            'num_heads': args['num_heads']
        }
    
    def execute_episodes_parallel(self, num_episodes, current_iteration=None, total_iterations=None):
        """
        完全并行执行游戏（每个进程独立 GPU）
        
        Args:
            num_episodes: 对弈局数
            current_iteration: 当前迭代轮次（可选，用于进度条显示）
            total_iterations: 总迭代轮次（可选，用于进度条显示）
        """
        num_workers = self.args.get('num_workers', 8)
        
        # 关键：同步最新训练的模型权重到共享内存
        # 主进程的 self.nnet 已经通过训练更新了，现在同步给 workers
        self.nnet_state_dict = {k: v.cpu().share_memory_() for k, v in self.nnet.state_dict().items()}
        
        # 分配游戏到各个 worker
        games_per_worker = [num_episodes // num_workers] * num_workers
        for i in range(num_episodes % num_workers):
            games_per_worker[i] += 1
        
        start_time = time.time()
        seed_base = int(time.time()) % 10000
        
        # 创建共享进度字典
        manager = Manager()
        progress_dict = manager.dict({i: 0 for i in range(num_workers)})
        
        # 启动多进程（每个进程独立 GPU 网络）
        mp_ctx = mp.get_context('spawn')
        
        # 使用 apply_async 以便监控进度
        pool = mp_ctx.Pool(processes=num_workers)
        results_async = []
        
        for i in range(num_workers):
            result = pool.apply_async(
                worker_self_play_with_gpu,
                (i, self.game_args, self.mcts_args, self.nnet_state_dict,
                 games_per_worker[i], seed_base, progress_dict)
            )
            results_async.append(result)
        
        # 监控进度
        if current_iteration and total_iterations:
            # 固定宽度30字符，确保与Train和Arena对齐
            epoch_str = f'Epoch {current_iteration}/{total_iterations}'
            desc = f'({epoch_str:<15})SelfPlay'
        else:
            desc = 'SelfPlay'
        pbar = tqdm(total=num_episodes, desc=desc, unit='局')
        
        try:
            while True:
                completed = sum(progress_dict.values())
                pbar.n = completed
                pbar.refresh()
                
                if completed >= num_episodes:
                    break
                
                # 检查是否所有任务完成
                if all(r.ready() for r in results_async):
                    break
                    
                time.sleep(0.5)
        except KeyboardInterrupt:
            pbar.close()
            print('\n  ⚠️  训练被中断')
            pool.terminate()
            pool.join()
            raise
        
        pbar.close()
        
        # 收集结果
        results = [r.get() for r in results_async]
        pool.close()
        pool.join()
        
        # 合并结果
        all_examples = []
        for worker_examples in results:
            all_examples.extend(worker_examples)
        
        # 清理主进程的 GPU 缓存（workers 已经清理完毕）
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        return all_examples
