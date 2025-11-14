# -*- coding: utf-8 -*-
"""
真正的多进程并行 MCTS + 批量推理
使用 multiprocessing.Pool 实现 CPU 并行，BatchInferenceServer 实现 GPU 批量推理
"""

import multiprocessing as mp
import numpy as np
import torch
import time
from queue import Queue
from typing import List, Tuple


def worker_play_game(args_tuple):
    """
    Worker 函数：执行单局游戏
    
    Args:
        args_tuple: (game_class, game_args, nnet_state_dict, mcts_args, seed)
    
    Returns:
        训练样本列表
    """
    from model.game import DotsAndBoxesGame
    from model.model import DotsAndBoxesNet
    from model.mcts import MCTS
    
    game_args, nnet_state_dict, mcts_args, seed = args_tuple
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化游戏
    game = DotsAndBoxesGame(**game_args)
    
    # 初始化网络（每个进程独立）
    device = torch.device('cpu')  # Worker 用 CPU
    nnet = DotsAndBoxesNet(
        game=game,
        args=mcts_args
    ).to(device)
    nnet.load_state_dict(nnet_state_dict)
    nnet.eval()
    
    # 初始化 MCTS
    mcts = MCTS(game, nnet, mcts_args)
    
    # 执行一局游戏
    examples = []
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    
    while True:
        # MCTS 搜索
        canonical_board = game.get_observation(state)
        temp = int(episode_step < mcts_args['temp_threshold'])
        pi = mcts.get_action_prob(state, temp=temp)
        
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
            # 游戏结束，分配奖励
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in examples]
        
        # 更新当前玩家
        cur_player = game.get_current_player(state)


class MultiProcessSelfPlay:
    """
    真正的多进程自我对弈
    
    改进：
    1. 使用 multiprocessing.Pool 实现真并行
    2. 每个进程独立执行游戏（CPU 并行）
    3. （可选）集成 BatchInferenceServer（GPU 批量推理）
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # 提取网络权重（用于传递给子进程）
        self.nnet_state_dict = {k: v.cpu() for k, v in nnet.state_dict().items()}
        
        # 游戏参数
        self.game_args = {
            'num_rows': args['num_rows'],
            'num_cols': args['num_cols']
        }
        
        # MCTS 参数
        self.mcts_args = {
            'num_simulations': args['num_simulations'],
            'cpuct': args['cpuct'],
            'dirichlet_alpha': args.get('dirichlet_alpha', 0.3),
            'dirichlet_epsilon': args.get('dirichlet_epsilon', 0.25),
            'temp_threshold': args['temp_threshold'],
            'num_res_blocks': args['num_res_blocks'],
            'num_filters': args['num_filters'],
            'num_heads': args['num_heads']
        }
    
    def execute_episodes_parallel(self, num_episodes):
        """
        使用多进程并行执行游戏
        
        Args:
            num_episodes: 游戏局数
        
        Returns:
            训练样本列表
        """
        num_workers = self.args.get('num_workers', 8)
        
        print(f'  🚀 启动多进程并行自我对弈')
        print(f'     进程数: {num_workers}')
        print(f'     总局数: {num_episodes}')
        
        # 准备参数（每局游戏不同的随机种子）
        base_seed = int(time.time()) % (2**16)  # 限制种子范围
        worker_args = [
            (self.game_args, self.nnet_state_dict, self.mcts_args, 
             (base_seed * 10000 + i) % (2**32 - 1))  # 确保种子在有效范围内
            for i in range(num_episodes)
        ]
        
        start_time = time.time()
        
        # 使用进程池并行执行
        mp_context = mp.get_context('spawn')
        with mp_context.Pool(processes=num_workers) as pool:
            results = pool.map(worker_play_game, worker_args)
        
        # 合并所有样本
        all_examples = []
        for game_examples in results:
            all_examples.extend(game_examples)
        
        elapsed_time = time.time() - start_time
        
        print(f'  ✓ 多进程自我对弈完成')
        print(f'    耗时: {elapsed_time:.2f}s')
        print(f'    速度: {num_episodes / elapsed_time:.1f} 局/秒')
        print(f'    样本数: {len(all_examples):,}')
        
        return all_examples
