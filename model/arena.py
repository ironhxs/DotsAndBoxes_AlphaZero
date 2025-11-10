# -*- coding: utf-8 -*-
"""Arena - 模型对战评估器 (AlphaZero核心组件)"""

import numpy as np
from tqdm import tqdm
from .mcts import MCTS
from multiprocessing import Pool, cpu_count, get_context
import torch


def _init_worker_cuda():
    """子进程初始化函数 - 设置 CUDA 环境"""
    import torch
    import os
    
    if torch.cuda.is_available():
        try:
            # 设置每个进程使用独立的 CUDA 设备
            # 或者禁用 CUDA，只用 CPU（更稳定）
            # torch.cuda.set_device(os.getpid() % torch.cuda.device_count())
            
            # 触发 CUDA 初始化
            device = torch.cuda.current_device()
            torch.cuda.set_device(device)
            
            # 禁用 cudnn benchmark (多进程环境下更稳定)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True
            
            # 预热 cuDNN
            dummy = torch.zeros(1, 1, 1, 1).cuda()
            _ = dummy + dummy
            del dummy
            torch.cuda.synchronize()
            
        except Exception as e:
            print(f"⚠️ Worker CUDA初始化失败: {e}, 将使用CPU模式")
            torch.cuda.is_available = lambda: False


def _play_single_game(args_tuple):
    """全局函数用于多进程并行对战"""
    game, p1_state_dict, p2_state_dict, model_args, game_args, player1_starts, seed = args_tuple
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 🔧 多进程环境：每个子进程独立使用 GPU
    # 注意：需要足够的显存（每个进程加载2个模型）
    use_cuda = game_args.get('cuda', False) and torch.cuda.is_available()
    
    # 重建两个模型
    from .model_transformer import DotsAndBoxesTransformer
    player1 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    player1.load_state_dict(p1_state_dict)
    player1.eval()
    
    player2 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    player2.load_state_dict(p2_state_dict)
    player2.eval()
    
    # ⚡ Arena推理：多进程环境下使用CPU（更稳定）
    # 如需GPU加速，请使用 arena_batch_inference 模式
    if use_cuda:
        try:
            player1 = player1.cuda()
            player2 = player2.cuda()
        except Exception as e:
            print(f"⚠️ GPU初始化失败，降级为CPU: {e}")
            use_cuda = False
    
    # 创建MCTS
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
                
                # 🔥 显式释放显存
                del player1, player2, mcts1, mcts2
                if use_cuda:
                    torch.cuda.empty_cache()
                
                return result
        
        cur_player_idx = 1 - cur_player_idx


class Arena:
    """
    Arena类用于评估两个模型的对战胜率
    这是AlphaZero中判断新模型是否优于旧模型的关键机制
    """
    
    def __init__(self, p1_state_dict, p2_state_dict, game, args):
        """
        p1_state_dict: 第一个玩家的模型权重 (state_dict)
        p2_state_dict: 第二个玩家的模型权重 (state_dict)
        game: 游戏实例
        args: MCTS参数
        """
        self.p1_state_dict = p1_state_dict
        self.p2_state_dict = p2_state_dict
        self.game = game
        self.args = args
    
    def play_game(self, player1_starts=True, verbose=False):
        """
        执行一局完整对战
        
        Args:
            player1_starts: player1是否先手
            verbose: 是否打印详细信息
        
        Returns:
            1: player1胜
            -1: player1负 (player2胜)
            0.0001: 平局 (避免完全0值影响统计)
        """
        # 🔥 重建模型（避免跨进程传递GPU tensor）
        from .model_transformer import DotsAndBoxesTransformer
        
        player1 = DotsAndBoxesTransformer(
            self.game,
            num_filters=self.args['num_filters'],
            num_blocks=self.args['num_res_blocks'],
            num_heads=self.args['num_heads']
        )
        player1.load_state_dict(self.p1_state_dict)
        player1.eval()
        
        player2 = DotsAndBoxesTransformer(
            self.game,
            num_filters=self.args['num_filters'],
            num_blocks=self.args['num_res_blocks'],
            num_heads=self.args['num_heads']
        )
        player2.load_state_dict(self.p2_state_dict)
        player2.eval()
        
        if self.args.get('cuda', False) and torch.cuda.is_available():
            player1 = player1.cuda()
            player2 = player2.cuda()
        
        players = [player1, player2]
        cur_player_idx = 0 if player1_starts else 1
        
        # Arena对战使用更少的MCTS次数加速验证
        arena_args = self.args.copy()
        arena_args['num_simulations'] = self.args.get('arena_mcts_simulations', 
                                                       self.args.get('num_simulations', 25))
        
        # 为每个玩家创建独立的MCTS
        mcts1 = MCTS(self.game, player1, arena_args)
        mcts2 = MCTS(self.game, player2, arena_args)
        mcts_players = [mcts1, mcts2]
        
        state = self.game.get_initial_state()
        it = 0
        
        while True:
            it += 1
            if verbose:
                print(f"Turn {it}, Player {cur_player_idx + 1}")
            
            # 获取当前玩家的MCTS
            mcts = mcts_players[cur_player_idx]
            
            # 使用MCTS获取最佳动作 (temperature=0, 选择最优)
            pi = mcts.get_action_prob(state, temp=0)
            
            # 选择概率最高的动作
            valid_moves = self.game.get_valid_moves(state)
            pi = pi * valid_moves  # 确保只选择合法动作
            
            if np.sum(pi) == 0:
                # 如果没有合法动作，选择任意合法动作
                action = np.random.choice(np.where(valid_moves > 0)[0])
            else:
                action = np.argmax(pi)
            
            if verbose:
                print(f"  Action: {action}")
            
            # 执行动作
            state = self.game.get_next_state(state, action)
            
            # 检查游戏是否结束
            if state.is_terminal():
                returns = state.returns()
                if len(returns) >= 2:
                    # player1视角的结果
                    if player1_starts:
                        result = 1 if returns[0] > returns[1] else (-1 if returns[0] < returns[1] else 0.0001)
                    else:
                        result = 1 if returns[1] > returns[0] else (-1 if returns[1] < returns[0] else 0.0001)
                    
                    if verbose:
                        print(f"Game over. Returns: {returns}, Result: {result}")
                    
                    # 🔥 释放GPU显存
                    del player1, player2, mcts1, mcts2
                    if self.args.get('cuda', False) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return result
            
            # 切换玩家
            new_player = self.game.get_current_player(state)
            # OpenSpiel中player可能不变(如吃子再走一步)，这里简化处理
            cur_player_idx = 1 - cur_player_idx
    
    def play_games(self, num_games, verbose=False):
        """
        进行多局对战并统计胜率 (支持多进程并行)
        
        Args:
            num_games: 对战局数 (必须是偶数，确保公平)
            verbose: 是否显示详细信息
        
        Returns:
            (wins, losses, draws): player1的胜/负/平局数
        """
        num_games = int(num_games / 2) * 2  # 确保是偶数
        
        # Arena使用更多MCTS确保评估准确性（2倍于训练）
        # Arena使用更多MCTS确保评估准确性（2倍于训练）
        mcts_sims = self.args.get('arena_mcts_simulations', 
                                   self.args.get('num_simulations', 100) * 2)
        # CPU多进程版本使用arena_num_workers或num_workers
        num_workers = min(
            self.args.get('arena_num_workers', self.args.get('num_workers', 4)), 
            num_games, 
            cpu_count() - 1
        )
        use_parallel = self.args.get('use_parallel', True) and num_workers > 1
        
        print(f"\n{'='*70}")
        print(f"🥊 Arena对战: {num_games} 局 (MCTS={mcts_sims}次, 高精度评估)")
        print(f"   并行: {num_workers} 进程 | 先手/后手各 {num_games//2} 局")
        print(f"{'='*70}")
        
        if use_parallel:
            # 准备参数（已经是state_dict）
            p1_state = self.p1_state_dict
            p2_state = self.p2_state_dict
            
            model_args = {
                'num_filters': self.args['num_filters'],
                'num_res_blocks': self.args['num_res_blocks'],
                'num_heads': self.args['num_heads']
            }
            
            # Arena专用配置：更多MCTS
            game_args = self.args.copy()
            game_args['num_simulations'] = mcts_sims
            
            tasks = [
                (self.game, p1_state, p2_state, model_args, game_args, 
                 (i % 2 == 0), np.random.randint(0, 1000000))
                for i in range(num_games)
            ]
            
            # 并行执行对战（使用spawn模式支持CUDA）
            ctx = get_context('spawn')
            with ctx.Pool(processes=num_workers, initializer=_init_worker_cuda) as pool:
                results = list(tqdm(
                    pool.imap(_play_single_game, tasks),
                    total=num_games,
                    desc=f"🎮 对战({num_workers}进程)"
                ))
        else:
            # 串行版本（保留用于调试）
            results = []
            for i in tqdm(range(num_games), desc="对战进度"):
                player1_starts = (i % 2 == 0)
                result = self.play_game(player1_starts=player1_starts, verbose=verbose)
                results.append(result)
        
        # 统计结果
        one_won = sum(1 for r in results if r == 1)
        two_won = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r != 1 and r != -1)
        
        # 输出统计结果
        print(f"\n{'='*70}")
        print(f"📊 对战结果统计:")
        print(f"{'='*70}")
        print(f"Player1 (新模型) 胜: {one_won}/{num_games} ({100*one_won/num_games:.1f}%)")
        print(f"Player2 (旧模型) 胜: {two_won}/{num_games} ({100*two_won/num_games:.1f}%)")
        print(f"平局:              {draws}/{num_games} ({100*draws/num_games:.1f}%)")
        print(f"{'='*70}\n")
        
        return one_won, two_won, draws
