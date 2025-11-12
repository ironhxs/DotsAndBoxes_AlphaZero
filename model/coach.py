# -*- coding: utf-8 -*-
"""AlphaZero 训练教练 - 单进程版本"""

import numpy as np
from .mcts import MCTS
from .base_coach import BaseCoach


class Coach(BaseCoach):
    """
    单进程版本的 AlphaZero Coach
    
    继承自 BaseCoach，只需实现自我对弈逻辑
    """
    
    def __init__(self, game, nnet, args):
        super().__init__(game, nnet, args)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        
    def execute_episode(self):
        """
        执行一局自我对弈
        
        Returns:
            训练样本列表 [(observation, policy, value), ...]
        """
        train_examples = []
        state, cur_player, episode_step = self.game.get_initial_state(), 0, 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_observation(state)
            temp = int(episode_step < self.args['temp_threshold'])
            pi = self.mcts.get_action_prob(state, temp=temp)
            
            if episode_step <= 30:
                noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(pi))
                pi = (1 - self.args['dirichlet_epsilon']) * pi + self.args['dirichlet_epsilon'] * noise
                pi = pi * self.game.get_valid_moves(state)
                pi = pi / np.sum(pi)
            
            train_examples.append([canonical_board, cur_player, pi, None])
            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(state, action)
            r = self.game.get_game_result(state, cur_player)
            
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
            
            new_player = self.game.get_current_player(state)
            if new_player != cur_player:
                cur_player = new_player
