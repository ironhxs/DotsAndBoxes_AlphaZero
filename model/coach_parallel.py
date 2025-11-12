# -*- coding: utf-8 -*-
"""
并行化 Coach - 提升训练效率

主要改进：
1. 多进程自我对弈
2. 批量 MCTS 推理
3. GPU 利用率优化
"""

import numpy as np
import time
from tqdm import tqdm

from .mcts_batch import ParallelSelfPlay
from .base_coach import BaseCoach


class ParallelCoach(BaseCoach):
    """
    并行化训练 Coach
    
    优化策略：
    1. 多进程自我对弈（CPU 并行）
    2. 批量神经网络推理（GPU 批处理）
    3. 异步数据收集
    
    继承自 BaseCoach，只需实现并行自我对弈逻辑
    """
    
    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        
        # 并行自我对弈
        self.parallel_self_play = ParallelSelfPlay(game, nnet, args)
    
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
        批量执行多局游戏（并行）
        
        重写基类方法以实现并行版本
        
        Returns:
            训练样本列表
        """
        print(f'  并行自我对弈：{self.args["num_episodes"]} 局')
        print(f'  并行进程数：{self.args.get("num_workers", 8)}')
        print(f'  MCTS 批量大小：{self.args.get("mcts_batch_size", 32)}')
        
        start_time = time.time()
        
        iteration_train_examples = self.parallel_self_play.execute_episodes_parallel(
            self.args['num_episodes']
        )
        
        elapsed_time = time.time() - start_time
        
        print(f'  ✓ 并行自我对弈完成')
        print(f'    耗时: {elapsed_time:.2f}s')
        print(f'    速度: {len(iteration_train_examples) / elapsed_time:.1f} 样本/秒')
        
        return iteration_train_examples
