# -*- coding: utf-8 -*-
"""Arena - 批量推理版本 (生产者-消费者模式)"""

import numpy as np
import torch
from tqdm import tqdm
from .mcts_batch import MCTSBatchInference
from multiprocessing import cpu_count
from threading import Thread
from queue import Queue, Empty
import time


class ArenaBatchInference:
    """
    高效Arena实现：
    - 多个线程并行执行MCTS搜索（CPU密集）
    - 单个GPU线程批量推理（共享模型，节省显存）
    - 生产者-消费者队列通信
    """
    
    def __init__(self, player1, player2, game, args):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.args = args
        
    def play_game(self, player1_starts=True):
        """执行一局对战"""
        players = [self.player1, self.player2]
        cur_player_idx = 0 if player1_starts else 1
        
        # 为每个玩家创建MCTS（会使用批量推理）
        mcts1 = MCTSBatchInference(self.game, self.player1, self.args)
        mcts2 = MCTSBatchInference(self.game, self.player2, self.args)
        mcts_players = [mcts1, mcts2]
        
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
    
    def play_games(self, num_games):
        """
        ⚠️ Python GIL限制：多线程无法利用多核CPU
        
        暂时fallback到原始多进程Arena实现
        """
        num_games = int(num_games / 2) * 2
        mcts_sims = self.args.get('arena_mcts_simulations', 
                                   self.args.get('num_simulations', 100) * 2)
        
        num_workers = min(
            self.args.get('arena_num_workers', self.args.get('num_workers', 3)), 
            num_games,
            cpu_count() - 1
        )
        
        print(f"\n{'='*70}")
        print(f"🥊 Arena对战: {num_games} 局 (MCTS={mcts_sims}次)")
        print(f"⚠️  批量推理受GIL限制，使用多进程模式")
        print(f"   架构: {num_workers} 个CPU进程（各自GPU推理）")
        print(f"   显存: {num_workers}个×2模型 (~{num_workers*400}MB)")
        print(f"{'='*70}")
        
        # Fallback到多进程实现
        from .arena import Arena
        fallback_arena = Arena(self.player1, self.player2, self.game, self.args)
        return fallback_arena.play_games(num_games)
