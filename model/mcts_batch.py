# -*- coding: utf-8 -*-
"""MCTS - 支持批量推理的版本"""

import numpy as np
import torch


class MCTSBatchInference:
    """
    MCTS with Batch Inference Support
    使用全局BatchInferenceServer进行推理
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.num_simulations = args.get('num_simulations', 100)
        self.cpuct = args.get('cpuct', 1.0)
        
        # MCTS搜索树
        self.Qsa = {}  # Q值
        self.Nsa = {}  # 访问次数
        self.Ns = {}   # 状态访问次数
        self.Ps = {}   # 先验概率
        
        # 批量推理服务器（全局单例，由Arena创建）
        self.inference_server = None
        self.model_idx = None
    
    def set_inference_server(self, server, model_idx):
        """设置批量推理服务器"""
        self.inference_server = server
        self.model_idx = model_idx
    
    def get_action_prob(self, state, temp=1):
        """获取动作概率分布"""
        for _ in range(self.num_simulations):
            self.search(state.clone())
        
        s = str(state)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]
        
        if temp == 0:
            # 贪婪选择
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = np.zeros(len(counts))
            probs[bestA] = 1
            return probs
        
        counts = np.array(counts, dtype=np.float64)
        counts = counts ** (1. / temp)
        probs = counts / np.sum(counts)
        return probs
    
    def search(self, state):
        """MCTS搜索（递归）"""
        s = str(state)
        
        if state.is_terminal():
            returns = state.returns()
            if len(returns) >= 2:
                result = 1 if returns[0] > returns[1] else (-1 if returns[0] < returns[1] else 0)
                return -result
        
        if s not in self.Ps:
            # 叶子节点：需要神经网络推理
            obs = self.game.get_observation(state)
            
            # 使用批量推理服务（如果可用）
            if self.inference_server is not None:
                pi, v = self.inference_server.predict(self.model_idx, obs)
            else:
                # Fallback: 直接推理（单进程模式）
                import torch
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                if next(self.nnet.parameters()).is_cuda:
                    obs_tensor = obs_tensor.cuda()
                
                self.nnet.eval()
                with torch.no_grad():
                    pi_tensor, v_tensor = self.nnet(obs_tensor)
                
                pi = torch.exp(pi_tensor).cpu().numpy()[0]
                v = v_tensor.cpu().numpy()[0][0]
            
            valids = self.game.get_valid_moves(state)
            pi = pi * valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = valids / np.sum(valids)
            
            self.Ps[s] = pi
            self.Ns[s] = 0
            return -v
        
        # 内部节点：选择最佳动作
        valids = self.game.get_valid_moves(state)
        cur_best = -float('inf')
        best_act = -1
        
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)
                
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
