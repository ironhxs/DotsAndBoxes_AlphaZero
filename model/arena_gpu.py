# -*- coding: utf-8 -*-
"""Arena GPU版本 - 使用批量推理加速对战"""

import numpy as np
from tqdm import tqdm
from .mcts import MCTS
import torch
import concurrent.futures
from threading import Lock, RLock


class ArenaGPU:
    """
    GPU加速版Arena：在主进程中并行管理多个对战，使用批量推理
    """
    
    def __init__(self, player1, player2, game, args):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.args = args
        
        # 线程锁保护模型推理
        self.lock1 = RLock()
        self.lock2 = RLock()
        
        # 确保模型在GPU上并设置eval模式
        if args.get('cuda', False) and torch.cuda.is_available():
            self.player1.cuda()
            self.player2.cuda()
        
        self.player1.eval()
        self.player2.eval()
        
        # 禁用梯度计算
        for param in self.player1.parameters():
            param.requires_grad = False
        for param in self.player2.parameters():
            param.requires_grad = False
    
    def play_game_parallel(self, player1_starts=True):
        """
        执行单局对战（线程安全版本）
        """
        # 创建线程安全的MCTS包装
        class ThreadSafeMCTS(MCTS):
            def __init__(self, game, nnet, args, lock):
                super().__init__(game, nnet, args)
                self.lock = lock
            
            def search(self, state):
                # 在神经网络推理时加锁
                with self.lock:
                    return super().search(state)
        
        # 创建MCTS（使用锁保护）
        mcts1 = ThreadSafeMCTS(self.game, self.player1, self.args, self.lock1)
        mcts2 = ThreadSafeMCTS(self.game, self.player2, self.args, self.lock2)
        mcts_players = [mcts1, mcts2]
        
        cur_player_idx = 0 if player1_starts else 1
        state = self.game.get_initial_state()
        
        while True:
            mcts = mcts_players[cur_player_idx]
            pi = mcts.get_action_prob(state, temp=0)
            
            valid_moves = self.game.get_valid_moves(state)
            pi = pi * valid_moves
            
            if np.sum(pi) == 0:
                action = np.random.choice(np.where(valid_moves > 0)[0])
            else:
                action = np.argmax(pi)
            
            state = self.game.get_next_state(state, action)
            
            if state.is_terminal():
                returns = state.returns()
                if len(returns) >= 2:
                    if player1_starts:
                        result = 1 if returns[0] > returns[1] else (-1 if returns[0] < returns[1] else 0.0001)
                    else:
                        result = 1 if returns[1] > returns[0] else (-1 if returns[1] < returns[0] else 0.0001)
                    return result
            
            cur_player_idx = 1 - cur_player_idx
    
    def play_games(self, num_games, verbose=False):
        """
        GPU并行版本：使用ThreadPoolExecutor在主进程中并行对战
        所有MCTS共享GPU上的模型，自动批量推理
        """
        num_games = int(num_games / 2) * 2
        
        mcts_sims = self.args.get('arena_mcts_simulations', 
                                   self.args.get('num_simulations', 100) * 2)
        
        # 使用线程池并行（线程安全版本）
        # Arena专用workers配置
        max_workers = min(
            self.args.get('arena_num_workers', self.args.get('num_workers', 6)), 
            num_games, 
            8  # 最多8线程，避免过度竞争
        )
        
        print(f"\n{'='*70}")
        print(f"🥊 Arena对战 (GPU批量推理): {num_games} 局")
        print(f"   MCTS={mcts_sims}次 | 并行度={max_workers} | GPU加速")
        print(f"   先手/后手各 {num_games//2} 局")
        print(f"{'='*70}")
        
        # 准备任务
        tasks = [(i % 2 == 0) for i in range(num_games)]
        
        # 使用线程池并行执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.play_game_parallel, player1_starts) 
                      for player1_starts in tasks]
            
            results = []
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=num_games, desc="🎮 GPU对战"):
                results.append(future.result())
        
        # 统计结果
        one_won = sum(1 for r in results if r == 1)
        two_won = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r != 1 and r != -1)
        
        print(f"\n{'='*70}")
        print(f"📊 对战结果统计:")
        print(f"{'='*70}")
        print(f"Player1 (新模型) 胜: {one_won}/{num_games} ({100*one_won/num_games:.1f}%)")
        print(f"Player2 (旧模型) 胜: {two_won}/{num_games} ({100*two_won/num_games:.1f}%)")
        print(f"平局:              {draws}/{num_games} ({100*draws/num_games:.1f}%)")
        print(f"{'='*70}\n")
        
        return one_won, two_won, draws
