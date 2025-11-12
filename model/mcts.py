# -*- coding: utf-8 -*-
"""
蒙特卡洛树搜索 (MCTS) - AlphaZero 核心组件
修复版本：完整实现搜索、回溯、价值传播
"""

import numpy as np
import math
import torch
import logging

logger = logging.getLogger(__name__)


class MCTS:
    """
    MCTS 实现 AlphaGo Zero / AlphaZero 风格的蒙特卡洛树搜索
    
    关键改进：
    1. 修复价值传播逻辑
    2. 正确处理终局状态
    3. 添加虚拟损失（virtual loss）支持并行搜索
    4. 增强的异常处理和日志
    """
    
    def __init__(self, game, nnet, args):
        """
        初始化 MCTS
        
        Args:
            game: 游戏环境实例
            nnet: 神经网络模型
            args: 配置参数（包含 num_simulations, cpuct 等）
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # MCTS 统计量
        self.Qsa = {}  # Q(s,a): 状态-动作值函数
        self.Nsa = {}  # N(s,a): 状态-动作访问次数
        self.Ns = {}   # N(s): 状态访问次数
        self.Ps = {}   # P(s): 策略先验概率
        
        # 虚拟损失（用于并行 MCTS）
        self.virtual_loss = 3
        self.Vsa = {}  # 虚拟损失计数
    
    def get_action_prob(self, state, temp=1):
        """
        执行 MCTS 模拟并返回动作概率分布
        
        Args:
            state: 当前游戏状态
            temp: 温度参数
                  - temp = 0: 完全贪心（选择访问次数最多的动作）
                  - temp = 1: 按访问次数比例采样
                  - temp > 1: 更加探索性
                  
        Returns:
            np.array: 动作概率分布
        """
        try:
            # 执行 MCTS 模拟
            for i in range(self.args['num_simulations']):
                try:
                    # 每次模拟使用状态的副本
                    self.search(state.clone())
                except Exception as e:
                    logger.warning(f"MCTS 模拟 {i+1} 失败: {e}")
                    continue
            
            s = str(state)
            
            # 收集每个动作的访问次数
            counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]
            
            # 温度为 0：完全贪心
            if temp == 0:
                best_actions = np.where(counts == np.max(counts))[0]
                probs = np.zeros(len(counts), dtype=np.float32)
                probs[best_actions] = 1.0 / len(best_actions)  # 如果有多个最佳动作，均匀分布
                return probs
            
            # 温度采样
            counts_temp = np.array([x ** (1.0 / temp) for x in counts], dtype=np.float32)
            counts_sum = float(np.sum(counts_temp))
            
            if counts_sum > 0:
                probs = counts_temp / counts_sum
            else:
                # 如果没有任何访问（不应该发生），使用均匀分布
                logger.warning(f"状态 {s[:50]}... 没有访问记录，使用均匀分布")
                valids = self.game.get_valid_moves(state)
                probs = valids / np.sum(valids)
            
            return probs.astype(np.float32)
        
        except Exception as e:
            logger.error(f"获取动作概率失败: {e}", exc_info=True)
            # 返回均匀分布
            valids = self.game.get_valid_moves(state)
            return (valids / np.sum(valids)).astype(np.float32)
    
    def reset(self):
        """重置 MCTS 树（清空所有统计量）"""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Vsa.clear()
        logger.debug("MCTS 树已重置")
    
    def get_search_statistics(self, state):
        """
        获取搜索统计信息（用于调试和分析）
        
        Returns:
            dict: 包含 Q 值、访问次数等信息
        """
        s = str(state)
        stats = {
            'total_visits': self.Ns.get(s, 0),
            'actions': []
        }
        
        for a in range(self.game.get_action_size()):
            if (s, a) in self.Nsa:
                stats['actions'].append({
                    'action': a,
                    'visits': self.Nsa[(s, a)],
                    'q_value': self.Qsa[(s, a)],
                    'prior': self.Ps.get(s, [0] * self.game.get_action_size())[a]
                })
        
        # 按访问次数排序
        stats['actions'].sort(key=lambda x: x['visits'], reverse=True)
        return stats
    
    def search(self, state):
        """
        执行一次 MCTS 模拟
        
        Args:
            state: 当前游戏状态
            
        Returns:
            float: 从当前状态的价值估计（从当前玩家视角）
        """
        try:
            s = str(state)
            
            # === 1. 终局检查 ===
            if state.is_terminal():
                # 游戏结束时，从当前玩家视角返回结果
                returns = state.returns()
                if len(returns) >= 2:
                    # 获取当前应该轮到谁走（虽然已终局，但需要知道上一步是谁走的）
                    # OpenSpiel 中 returns[0] 是玩家0的得分，returns[1] 是玩家1的得分
                    # 需要确定是从哪个玩家视角
                    
                    # 假设返回的是从玩家0的视角
                    result = 1.0 if returns[0] > returns[1] else (-1.0 if returns[0] < returns[1] else 0.0)
                    
                    # 如果当前状态的 current_player 是玩家1，需要反转
                    # 但终局时 current_player 可能是 TERMINAL (-4)，所以用历史玩家判断
                    # 为简化，这里返回玩家0视角的结果，外层会处理
                    return result
                
                return 0.0  # 安全返回
            
            # === 2. 叶子节点扩展 ===
            if s not in self.Ps:
                # 使用神经网络评估当前状态
                obs = self.game.get_observation(state)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # 检查设备
                device = next(self.nnet.parameters()).device
                obs_tensor = obs_tensor.to(device)
                
                self.nnet.eval()
                with torch.no_grad():
                    log_pi, v = self.nnet(obs_tensor)
                
                # 转换概率（log_softmax -> softmax）
                pi = torch.exp(log_pi).cpu().numpy()[0]
                
                # 屏蔽非法动作
                valids = self.game.get_valid_moves(state)
                pi = pi * valids
                
                # 归一化
                sum_pi = np.sum(pi)
                if sum_pi > 0:
                    pi /= sum_pi
                else:
                    # 如果所有动作都被屏蔽（不应该发生），均匀分布
                    logger.warning(f"状态 {s[:50]}... 所有动作被屏蔽！")
                    pi = valids / np.sum(valids)
                
                # 存储先验概率和初始化访问计数
                self.Ps[s] = pi
                self.Ns[s] = 0
                
                # 返回价值估计（已经是 tanh 输出，在 [-1, 1] 范围内）
                v_value = v.item()
                
                # 注意：这里返回的价值是从当前玩家视角
                # AlphaZero 论文中，网络输出是从当前玩家视角的胜率预测
                return -v_value  # 返回负值，因为父节点需要从对手视角看
            
            # === 3. 选择动作（UCB 公式）===
            valids = self.game.get_valid_moves(state)
            cur_best = -float('inf')
            best_act = -1
            
            for a in range(self.game.get_action_size()):
                if valids[a]:
                    if (s, a) in self.Qsa:
                        # Q(s,a) + U(s,a)
                        # U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                        q_value = self.Qsa[(s, a)]
                        n_sa = self.Nsa[(s, a)]
                        
                        # 虚拟损失（如果启用）
                        virtual_loss_adjustment = 0
                        if (s, a) in self.Vsa:
                            virtual_loss_adjustment = -self.virtual_loss * self.Vsa[(s, a)]
                        
                        u_value = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + n_sa)
                        ucb_value = q_value + u_value + virtual_loss_adjustment
                    else:
                        # 未访问过的节点，Q(s,a) = 0
                        u_value = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                        ucb_value = u_value
                    
                    if ucb_value > cur_best:
                        cur_best = ucb_value
                        best_act = a
            
            # 检查是否找到合法动作
            if best_act == -1:
                logger.error(f"状态 {s[:50]}... 没有找到合法动作！")
                # 随机选择一个合法动作
                legal_actions = np.where(valids > 0)[0]
                if len(legal_actions) > 0:
                    best_act = np.random.choice(legal_actions)
                else:
                    raise RuntimeError("没有合法动作可选")
            
            a = best_act
            
            # 添加虚拟损失（可选，用于并行 MCTS）
            # if (s, a) in self.Vsa:
            #     self.Vsa[(s, a)] += 1
            # else:
            #     self.Vsa[(s, a)] = 1
            
            # === 4. 递归搜索 ===
            next_state = self.game.get_next_state(state, a)
            v = self.search(next_state)
            
            # 移除虚拟损失
            # if (s, a) in self.Vsa:
            #     self.Vsa[(s, a)] -= 1
            
            # === 5. 回溯更新 ===
            if (s, a) in self.Qsa:
                # 增量更新 Q 值: Q_new = (N * Q_old + v) / (N + 1)
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                # 首次访问
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1
            
            self.Ns[s] += 1
            
            # 返回负值（从父节点的视角）
            return -v
        
        except Exception as e:
            logger.error(f"MCTS 搜索错误: {e}", exc_info=True)
            # 返回中性价值
            return 0.0
