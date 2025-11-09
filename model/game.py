# -*- coding: utf-8 -*-
"""点格棋游戏封装 - 基于 OpenSpiel"""

import pyspiel
import numpy as np


class DotsAndBoxesGame:
    """封装 OpenSpiel 的点格棋环境"""
    
    def __init__(self, num_rows=5, num_cols=5):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.game = pyspiel.load_game(f"dots_and_boxes(num_rows={num_rows},num_cols={num_cols})")
        self.action_size = self.game.num_distinct_actions()
    
    def get_initial_state(self):
        return self.game.new_initial_state()
    
    def get_action_size(self):
        """返回动作空间大小"""
        return self.action_size
    
    def get_observation(self, state):
        """获取状态的张量表示 (9, 6, 6)"""
        obs = np.array(state.observation_tensor()).reshape(9, self.num_rows + 1, self.num_cols + 1)
        return obs.astype(np.float32)
    
    def get_valid_moves(self, state):
        """返回合法动作掩码"""
        valid = np.zeros(self.action_size, dtype=np.float32)
        for action in state.legal_actions():
            valid[action] = 1.0
        return valid
    
    def get_next_state(self, state, action):
        """执行动作，返回新状态"""
        new_state = state.clone()
        new_state.apply_action(action)
        return new_state
    
    def get_current_player(self, state):
        """获取当前玩家 ID"""
        return state.current_player()
    
    def is_terminal(self, state):
        """判断游戏是否结束"""
        return state.is_terminal()
    
    def get_game_result(self, state, player):
        """获取游戏结果 (1=赢, -1=输, 0=未结束)
        
        Args:
            state: 游戏状态
            player: 玩家 ID (0 或 1)
            
        Returns:
            1 表示该玩家赢，-1 表示输，0 表示平局或未结束
        """
        if not state.is_terminal():
            return 0
        
        returns = state.returns()
        
        # OpenSpiel 的 dots_and_boxes 返回列表 [玩家0得分, 玩家1得分]
        # 得分是相对差值，正数表示赢，负数表示输
        if player >= 0 and player < len(returns):
            score = returns[player]
            if score > 0:
                return 1
            elif score < 0:
                return -1
            else:
                return 0
        
        # 如果 player 无效（如游戏结束时的 -4），返回 0
        return 0
    
    def display(self, state):
        """打印游戏状态"""
        print(state)
