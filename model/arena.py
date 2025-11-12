# -*- coding: utf-8 -*-
"""
Arena - AlphaZero模型对战系统
新模型 vs 旧模型对战，只有胜率>阈值才接受新模型
"""

import numpy as np
from tqdm import tqdm
import torch
from .mcts import MCTS


class Arena:
    """
    Arena对战系统
    
    让两个玩家对战多局，统计胜率
    玩家可以是：神经网络+MCTS、纯MCTS、随机等
    """
    
    def __init__(self, game, player1, player2, args):
        """
        初始化Arena
        
        Args:
            game: 游戏环境
            player1: 玩家1 (通常是新模型)
            player2: 玩家2 (通常是旧模型)
            args: 配置参数
        """
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.args = args
        
    def play_game(self, player1_starts=True):
        """
        进行一局游戏
        
        Args:
            player1_starts: player1是否先手
            
        Returns:
            1: player1赢
            -1: player2赢
            0: 平局
        """
        player_mapping = {
            0: self.player1 if player1_starts else self.player2,
            1: self.player2 if player1_starts else self.player1,
        }
        player1_actual_id = 0 if player1_starts else 1

        state = self.game.get_initial_state()
        move_count = 0
        max_moves = self.args.get('arena_max_moves', 300)

        while not self.game.is_terminal(state) and move_count < max_moves:
            move_count += 1

            current_player_id = self.game.get_current_player(state)
            if current_player_id not in player_mapping:
                raise RuntimeError(f"Arena 遇到未知的玩家编号: {current_player_id}")

            current_player = player_mapping[current_player_id]
            action = current_player(state)

            valid_moves = self.game.get_valid_moves(state)
            if action < 0 or action >= len(valid_moves) or valid_moves[action] == 0:
                # 非法动作，当前玩家直接判负
                return -1 if current_player_id == player1_actual_id else 1

            state = self.game.get_next_state(state, action)

        if self.game.is_terminal(state):
            return self.game.get_game_result(state, player1_actual_id)

        return 0
    
    def play_games(self, num_games, num_workers=1, random_start=False):
        """
        进行多局对战（支持并行）
        
        Args:
            num_games: 对战局数
            num_workers: 并行进程数（1=串行，>1=并行）
            random_start: 是否随机先手（True=随机，False=交替）
            
        Returns:
            (player1_wins, player2_wins, draws)
        """
        num_games = max(1, num_games)  # 至少1局
        
        if num_workers <= 1:
            # 串行执行
            return self._play_games_serial(num_games, random_start)
        else:
            # 并行执行
            return self._play_games_parallel(num_games, num_workers, random_start)
    
    def _play_games_serial(self, num_games, random_start=False):
        """串行执行多局游戏"""
        import random
        
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        for i in tqdm(range(num_games), desc="Arena对战"):
            # 决定先手
            if random_start:
                player1_starts = random.random() < 0.5
            else:
                player1_starts = (i % 2 == 0)
            
            result = self.play_game(player1_starts=player1_starts)
            
            if result == 1:
                player1_wins += 1
            elif result == -1:
                player2_wins += 1
            else:
                draws += 1
        
        return player1_wins, player2_wins, draws
    
    def _play_games_parallel(self, num_games, num_workers, random_start=False):
        """并行执行多局游戏"""
        import multiprocessing as mp
        import random
        
        # ⚠️ CUDA 多进程必须使用 spawn 模式
        mp_context = mp.get_context('spawn')
        
        # 生成先手分配
        if random_start:
            starts = [random.random() < 0.5 for _ in range(num_games)]
        else:
            starts = [i % 2 == 0 for i in range(num_games)]
        
        # 创建任务列表
        tasks = [(i, starts[i]) for i in range(num_games)]
        
        # 并行执行
        with mp_context.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._play_game_worker, tasks),
                total=num_games,
                desc=f"Arena对战({num_workers}进程)"
            ))
        
        # 统计结果
        player1_wins = sum(1 for r in results if r == 1)
        player2_wins = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r == 0)
        
        return player1_wins, player2_wins, draws
    
    def _play_game_worker(self, task):
        """并行工作函数（用于多进程）"""
        game_idx, player1_starts = task
        return self.play_game(player1_starts=player1_starts)


class NeuralNetPlayer:
    """神经网络玩家 (使用MCTS)"""
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(game, nnet, args)
        self.args = args
        
    def __call__(self, state):
        """
        根据当前状态选择动作
        
        Args:
            state: 游戏状态
            
        Returns:
            action: 选择的动作
        """
        # 使用MCTS获取动作概率 (temp=0表示贪心选择)
        probs = self.mcts.get_action_prob(state, temp=0)
        
        # 选择概率最高的合法动作
        valid_moves = self.game.get_valid_moves(state)
        probs = probs * valid_moves  # 只考虑合法动作
        
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
            action = np.argmax(probs)
        else:
            # 如果没有合法动作，随机选一个
            action = np.random.choice(np.where(valid_moves > 0)[0])
        
        return action


class RandomPlayer:
    """随机玩家 (baseline)"""
    
    def __init__(self, game):
        self.game = game
        
    def __call__(self, state):
        """随机选择合法动作"""
        valid_moves = self.game.get_valid_moves(state)
        valid_actions = np.where(valid_moves > 0)[0]
        return np.random.choice(valid_actions)


def compare_models(game, new_nnet, old_nnet, args):
    """
    比较新旧模型
    
    Args:
        game: 游戏环境
        new_nnet: 新模型
        old_nnet: 旧模型 (如果为None，则与随机玩家比较)
        args: 配置参数
        
    Returns:
        (win_rate, should_accept): 胜率和是否接受新模型
    """
    # 创建MCTS参数 (Arena用更多模拟次数)
    arena_args = args.copy()
    arena_args['num_simulations'] = args.get('arena_mcts_simulations', 200)
    
    # 创建玩家
    new_player = NeuralNetPlayer(game, new_nnet, arena_args)
    
    if old_nnet is not None:
        old_player = NeuralNetPlayer(game, old_nnet, arena_args)
    else:
        # 如果没有旧模型，与随机玩家比较
        old_player = RandomPlayer(game)
    
    # 创建Arena
    arena = Arena(game, new_player, old_player, arena_args)
    
    # 进行对战
    num_games = args.get('arena_compare', 40)
    new_wins, old_wins, draws = arena.play_games(num_games)
    
    # 计算胜率
    total_decisive = new_wins + old_wins
    if total_decisive > 0:
        win_rate = new_wins / total_decisive
    else:
        win_rate = 0.5  # 全平局，算50%
    
    # 判断是否接受新模型
    threshold = args.get('update_threshold', 0.55)
    should_accept = win_rate >= threshold
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Arena对战结果:")
    print(f"  新模型: {new_wins}胜 ({win_rate*100:.1f}%)")
    print(f"  旧模型: {old_wins}胜")
    print(f"  平局: {draws}")
    print(f"  阈值: {threshold*100:.1f}%")
    print(f"  决定: {'✅ 接受新模型' if should_accept else '❌ 拒绝新模型，保留旧模型'}")
    print(f"{'='*60}\n")
    
    return win_rate, should_accept
