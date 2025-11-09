# -*- coding: utf-8 -*-
"""蒙特卡洛树搜索"""

import numpy as np
import math
import torch


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
    
    def get_action_prob(self, state, temp=1):
        for _ in range(self.args['num_simulations']):
            self.search(state.clone())
        
        s = str(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 
                  for a in range(self.game.get_action_size())]
        
        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return np.array(probs, dtype=np.float32)
        
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return np.array(probs, dtype=np.float32)
    
    def search(self, state):
        s = str(state)
        
        if state.is_terminal():
            # 游戏结束时，从当前玩家视角返回结果
            # 使用 returns() 直接获取最终得分
            returns = state.returns()
            if len(returns) >= 2:
                # 返回从先手玩家(0)视角的结果
                result = 1 if returns[0] > returns[1] else (-1 if returns[0] < returns[1] else 0)
                return -result  # 返回负值因为是从对手视角
        
        if s not in self.Ps:
            obs = self.game.get_observation(state)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if next(self.nnet.parameters()).is_cuda:
                obs_tensor = obs_tensor.cuda()
            
            self.nnet.eval()
            with torch.no_grad():
                pi, v = self.nnet(obs_tensor)
            
            pi = torch.exp(pi).cpu().numpy()[0]
            valids = self.game.get_valid_moves(state)
            pi = pi * valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = valids / np.sum(valids)
            
            self.Ps[s] = pi
            self.Ns[s] = 0
            return -v.item()
        
        valids = self.game.get_valid_moves(state)
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        next_state = self.game.get_next_state(state, a)
        v = self.search(next_state)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v
