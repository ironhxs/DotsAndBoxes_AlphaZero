# -*- coding: utf-8 -*-
"""Arena - 模型对战评估器 (AlphaZero核心组件)"""

import numpy as np
from tqdm import tqdm
from .mcts import MCTS


class Arena:
    """
    Arena类用于评估两个模型的对战胜率
    这是AlphaZero中判断新模型是否优于旧模型的关键机制
    """
    
    def __init__(self, player1, player2, game, args):
        """
        player1: 第一个玩家的神经网络
        player2: 第二个玩家的神经网络
        game: 游戏实例
        args: MCTS参数
        """
        self.player1 = player1
        self.player2 = player2
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
        players = [self.player1, self.player2]
        cur_player_idx = 0 if player1_starts else 1
        
        # 为每个玩家创建独立的MCTS
        mcts1 = MCTS(self.game, self.player1, self.args)
        mcts2 = MCTS(self.game, self.player2, self.args)
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
                    
                    return result
            
            # 切换玩家
            new_player = self.game.get_current_player(state)
            # OpenSpiel中player可能不变(如吃子再走一步)，这里简化处理
            cur_player_idx = 1 - cur_player_idx
    
    def play_games(self, num_games, verbose=False):
        """
        进行多局对战并统计胜率
        
        Args:
            num_games: 对战局数 (必须是偶数，确保公平)
            verbose: 是否显示详细信息
        
        Returns:
            (wins, losses, draws): player1的胜/负/平局数
        """
        num_games = int(num_games / 2) * 2  # 确保是偶数
        
        one_won = 0
        two_won = 0
        draws = 0
        
        print(f"\n{'='*70}")
        print(f"🥊 Arena对战: {num_games} 局 (player1 先手{num_games//2}局, 后手{num_games//2}局)")
        print(f"{'='*70}")
        
        for i in tqdm(range(num_games), desc="对战进度"):
            # 交替先后手，确保公平
            player1_starts = (i % 2 == 0)
            
            game_result = self.play_game(player1_starts=player1_starts, verbose=verbose)
            
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1
        
        # 输出统计结果
        print(f"\n{'='*70}")
        print(f"📊 对战结果统计:")
        print(f"{'='*70}")
        print(f"Player1 (新模型) 胜: {one_won}/{num_games} ({100*one_won/num_games:.1f}%)")
        print(f"Player2 (旧模型) 胜: {two_won}/{num_games} ({100*two_won/num_games:.1f}%)")
        print(f"平局:              {draws}/{num_games} ({100*draws/num_games:.1f}%)")
        print(f"{'='*70}\n")
        
        return one_won, two_won, draws
