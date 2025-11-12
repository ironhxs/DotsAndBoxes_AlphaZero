# -*- coding: utf-8 -*-
"""
并行化 Coach - 提升训练效率

主要改进：
1. 完全并行 GPU：每个进程独立 GPU 网络（最高 GPU 利用率）
2. 多进程共享 GPU：主进程推理服务器（节省显存）
3. 单进程并发：简单稳定（调试用）
"""

import numpy as np
import time
from tqdm import tqdm

from .mcts_full_parallel_gpu import FullParallelGPUSelfPlay
from .mcts_multiprocess_gpu import MultiProcessGPUSelfPlay
from .mcts_concurrent_gpu import ConcurrentGames
from .base_coach import BaseCoach


class ParallelCoach(BaseCoach):
    """
    并行化训练 Coach
    
    优化策略（按性能排序）：
    1. 完全并行模式：每个 worker 独立 GPU 网络（GPU+显存充足时最快）
    2. 共享 GPU 模式：多 worker + 主进程推理服务器（节省显存）
    3. 单进程模式：单进程并发（简单稳定，调试用）
    
    继承自 BaseCoach，只需实现并行自我对弈逻辑
    """
    
    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        
        # 选择并行模式
        parallel_mode = args.get('parallel_mode', 'full')  # 'full', 'shared', 'single'
        
        if parallel_mode == 'full':
            self.parallel_engine = FullParallelGPUSelfPlay(game, nnet, args)
        elif parallel_mode == 'shared':
            self.parallel_engine = MultiProcessGPUSelfPlay(game, nnet, args)
        else:
            self.parallel_engine = ConcurrentGames(game, nnet, args)
    
    def execute_episode(self):
        """
        单局自我对弈（为了兼容性保留，但不推荐使用）
        
        建议：使用 collect_self_play_data() 进行批量并行对弈
        """
        raise NotImplementedError(
            "ParallelCoach 应该使用 collect_self_play_data() 进行并行对弈，"
            "而不是单局的 execute_episode()"
        )
    
    def collect_self_play_data(self):
        """
        并行执行多局游戏
        
        Returns:
            训练样本列表
        """
        start_time = time.time()
        
        iteration_train_examples = self.parallel_engine.execute_episodes_parallel(
            self.args['num_episodes']
        )
        
        elapsed_time = time.time() - start_time
        num_games = self.args["num_episodes"]
        num_samples = len(iteration_train_examples)
        avg_steps = num_samples / num_games if num_games > 0 else 0
        
        print(f'  ✓ 完成: {elapsed_time:.1f}s, {num_games / elapsed_time:.2f} 局/秒')
        print(f'  ✓ 样本: {num_samples} 个 ({num_games}局 × 平均{avg_steps:.1f}步)')
        
        return iteration_train_examples
