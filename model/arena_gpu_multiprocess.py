# -*- coding: utf-8 -*-
"""Arena GPU 多进程版本 - 真正的并行"""

import numpy as np
from tqdm import tqdm
from .mcts import MCTS
import torch
from multiprocessing import Pool, cpu_count, get_context


def _init_arena_worker_cuda():
    """子进程初始化 - Arena 专用"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.current_device()
        torch.backends.cudnn.benchmark = False


def _arena_single_game_worker(args_tuple):
    """
    Arena 单局对战（多进程版本）
    
    关键：每个进程独立加载模型和使用 GPU
    就像自我对弈那样！
    """
    game, p1_state, p2_state, model_args, game_args, player1_starts, seed = args_tuple
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 重建两个模型（就像自我对弈那样）
    from .model_transformer import DotsAndBoxesTransformer
    player1 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    player1.load_state_dict(p1_state)
    player1.eval()
    
    player2 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    player2.load_state_dict(p2_state)
    player2.eval()
    
    # ⚡ 使用 GPU（每个进程独立）
    if game_args.get('cuda', False) and torch.cuda.is_available():
        player1 = player1.cuda()
        player2 = player2.cuda()
    
    # 创建 MCTS
    mcts1 = MCTS(game, player1, game_args)
    mcts2 = MCTS(game, player2, game_args)
    mcts_players = [mcts1, mcts2]
    
    # 执行对战
    cur_player_idx = 0 if player1_starts else 1
    state = game.get_initial_state()
    
    while True:
        mcts = mcts_players[cur_player_idx]
        pi = mcts.get_action_prob(state, temp=0)
        
        valid_moves = game.get_valid_moves(state)
        pi = pi * valid_moves
        
        if np.sum(pi) == 0:
            action = np.random.choice(np.where(valid_moves > 0)[0])
        else:
            action = np.argmax(pi)
        
        state = game.get_next_state(state, action)
        
        if state.is_terminal():
            returns = state.returns()
            if len(returns) >= 2:
                if player1_starts:
                    result = 1 if returns[0] > returns[1] else (-1 if returns[0] < returns[1] else 0.0001)
                else:
                    result = 1 if returns[1] > returns[0] else (-1 if returns[1] < returns[0] else 0.0001)
                
                # 🔥 彻底清理显存（关键！）
                del player1, player2, mcts1, mcts2, state, mcts_players
                if game_args.get('cuda', False) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                return result
        
        cur_player_idx = 1 - cur_player_idx


class ArenaGPUMultiProcess:
    """
    Arena GPU 多进程版本 - 真正的并行
    
    使用方式：
    - 每个进程独立加载模型
    - 每个进程独立使用 GPU
    - 真正的多核并行（不受 GIL 限制）
    
    就像自我对弈那样！
    """
    
    def __init__(self, player1, player2, game, args):
        # 保存 state_dict（用于传递给子进程）
        self.p1_state_dict = {k: v.cpu() for k, v in player1.state_dict().items()}
        self.p2_state_dict = {k: v.cpu() for k, v in player2.state_dict().items()}
        self.game = game
        self.args = args
    
    def play_games(self, num_games, verbose=False):
        """
        多进程并行对战
        
        完全模仿自我对弈的实现！
        """
        num_games = int(num_games / 2) * 2
        
        mcts_sims = self.args.get('arena_mcts_simulations', 
                                   self.args.get('num_simulations', 100) * 2)
        
        # 进程数配置（与自我对弈一样）
        num_workers = min(
            self.args.get('arena_num_workers', self.args.get('num_workers', 4)), 
            num_games, 
            cpu_count() - 1
        )
        
        print(f"\n{'='*70}")
        print(f"🥊 Arena对战 (GPU多进程): {num_games} 局")
        print(f"   MCTS={mcts_sims}次 | 并行={num_workers}进程 | GPU加速")
        print(f"   先手/后手各 {num_games//2} 局")
        print(f"   ⚡ 真正的多核并行（与自我对弈同样方式）")
        print(f"{'='*70}")
        
        # 准备参数
        model_args = {
            'num_filters': self.args['num_filters'],
            'num_res_blocks': self.args['num_res_blocks'],
            'num_heads': self.args['num_heads']
        }
        
        # Arena 专用配置
        game_args = self.args.copy()
        game_args['num_simulations'] = mcts_sims
        
        tasks = [
            (self.game, self.p1_state_dict, self.p2_state_dict, model_args, game_args, 
             (i % 2 == 0), np.random.randint(0, 1000000))
            for i in range(num_games)
        ]
        
        # 使用 spawn 模式（与自我对弈一样）
        ctx = get_context('spawn')
        with ctx.Pool(processes=num_workers, initializer=_init_arena_worker_cuda) as pool:
            results = list(tqdm(
                pool.imap(_arena_single_game_worker, tasks),
                total=num_games,
                desc=f"🎮 Arena({num_workers}进程)"
            ))
        
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
