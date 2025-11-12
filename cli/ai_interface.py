#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单易用的AI接口 - 用于对战和获取建议
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.mcts import MCTS


class DotsAndBoxesAI:
    """
    点格棋AI - 简单易用的接口
    
    使用方法：
        # 1. 创建AI
        ai = DotsAndBoxesAI(checkpoint_path='results/test_4060/best_2.pth')
        
        # 2. 获取推荐动作
        state = game.get_initial_state()
        action = ai.get_action(state)
        
        # 3. 获取动作概率分布
        probs = ai.get_action_probs(state)
        
        # 4. 评估局面
        value = ai.evaluate_position(state)
    """
    
    def __init__(self, checkpoint_path=None, num_simulations=100, use_cuda=True):
        """
        初始化AI
        
        Args:
            checkpoint_path: 模型checkpoint路径，如果为None则使用随机初始化的模型
            num_simulations: MCTS模拟次数 (越多越强但越慢)
            use_cuda: 是否使用GPU
        """
        self.game = DotsAndBoxesGame()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        # 初始化模型
        self.model = DotsAndBoxesTransformer(
            game=self.game,
            num_filters=64,
            num_blocks=4,
            num_heads=4,
            input_channels=9
        ).to(self.device)
        
        # 加载checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"加载模型: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # 兼容多种checkpoint格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 宽松加载：忽略不匹配的键（动态层会在forward中创建）
            self.model.load_state_dict(state_dict, strict=False)
            print("✓ 模型加载成功")
        else:
            print("⚠️ 未加载checkpoint，使用随机初始化的模型")
        
        self.model.eval()
        
        # MCTS配置
        self.args = {
            'num_simulations': num_simulations,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.0,  # 对战时不加噪声
            'temp_threshold': 0,
            'cuda': self.use_cuda,
        }
        
        self.mcts = MCTS(self.game, self.model, self.args)
    
    def get_action(self, state, temperature=0):
        """
        获取推荐的动作
        
        Args:
            state: 游戏状态
            temperature: 温度参数 (0=贪心选择最佳, 1=按概率采样)
            
        Returns:
            action: 推荐的动作编号
        """
        probs = self.mcts.get_action_prob(state, temp=temperature)
        
        if temperature == 0:
            # 贪心选择
            action = np.argmax(probs)
        else:
            # 按概率采样
            action = np.random.choice(len(probs), p=probs)
        
        return action
    
    def get_action_probs(self, state, temperature=1):
        """
        获取所有动作的概率分布
        
        Args:
            state: 游戏状态
            temperature: 温度参数
            
        Returns:
            probs: 动作概率数组 [action_size]
        """
        return self.mcts.get_action_prob(state, temp=temperature)
    
    def get_top_k_actions(self, state, k=5):
        """
        获取前K个最佳动作
        
        Args:
            state: 游戏状态
            k: 返回前k个动作
            
        Returns:
            [(action, prob), ...] 按概率降序排列
        """
        probs = self.get_action_probs(state, temperature=0)
        
        # 获取top-k
        top_k_indices = np.argsort(probs)[-k:][::-1]
        top_k_actions = [(int(idx), float(probs[idx])) for idx in top_k_indices]
        
        return top_k_actions
    
    def evaluate_position(self, state):
        """
        评估当前局面价值
        
        Args:
            state: 游戏状态
            
        Returns:
            value: 局面价值 (-1到1, 正数表示对当前玩家有利)
        """
        obs = self.game.get_observation(state)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, v = self.model(obs_tensor)
            value = v.item()
        
        return value
    
    def explain_action(self, state, action=None):
        """
        解释推荐的动作
        
        Args:
            state: 游戏状态
            action: 要解释的动作，如果为None则解释推荐动作
            
        Returns:
            dict: 包含动作信息和分析
        """
        if action is None:
            action = self.get_action(state)
        
        probs = self.get_action_probs(state, temperature=0)
        value = self.evaluate_position(state)
        valid_moves = self.game.get_valid_moves(state)
        
        # 计算动作排名
        valid_probs = probs * valid_moves
        sorted_actions = np.argsort(valid_probs)[::-1]
        rank = np.where(sorted_actions == action)[0][0] + 1
        
        return {
            'action': action,
            'probability': float(probs[action]),
            'rank': int(rank),
            'position_value': float(value),
            'is_valid': bool(valid_moves[action]),
            'explanation': self._generate_explanation(value, probs[action], rank)
        }
    
    def _generate_explanation(self, value, prob, rank):
        """生成人类可读的解释"""
        explanation = []
        
        # 局面评估
        if value > 0.5:
            explanation.append("局面非常有利")
        elif value > 0.2:
            explanation.append("局面略有优势")
        elif value > -0.2:
            explanation.append("局面均势")
        elif value > -0.5:
            explanation.append("局面略显劣势")
        else:
            explanation.append("局面较为不利")
        
        # 动作评价
        if rank == 1:
            explanation.append("这是最佳选择")
        elif rank <= 3:
            explanation.append(f"这是第{rank}好的选择")
        else:
            explanation.append(f"这不是最优选择（排名第{rank}）")
        
        if prob > 0.5:
            explanation.append("AI非常确信这步棋")
        elif prob > 0.3:
            explanation.append("AI较为确信这步棋")
        else:
            explanation.append("AI对这步棋不太确定")
        
        return "，".join(explanation)
    
    def reset_mcts(self):
        """重置MCTS搜索树（新游戏时调用）"""
        self.mcts = MCTS(self.game, self.model, self.args)


def demo():
    """演示如何使用AI"""
    print("=" * 60)
    print("点格棋AI使用演示")
    print("=" * 60)
    
    # 创建AI
    ai = DotsAndBoxesAI(
        checkpoint_path='results/test_4060/latest.pth',  # 使用训练的模型
        num_simulations=50,  # MCTS模拟次数
    )
    
    # 创建游戏
    game = DotsAndBoxesGame()
    state = game.get_initial_state()
    
    print("\n1. 获取推荐动作:")
    action = ai.get_action(state)
    print(f"   推荐动作: {action}")
    
    print("\n2. 获取前5个最佳动作:")
    top5 = ai.get_top_k_actions(state, k=5)
    for i, (act, prob) in enumerate(top5, 1):
        print(f"   {i}. 动作 {act}: {prob*100:.1f}%")
    
    print("\n3. 评估局面:")
    value = ai.evaluate_position(state)
    print(f"   局面价值: {value:.3f} ({'有利' if value > 0 else '不利'})")
    
    print("\n4. 解释推荐动作:")
    explanation = ai.explain_action(state)
    print(f"   动作: {explanation['action']}")
    print(f"   概率: {explanation['probability']*100:.1f}%")
    print(f"   排名: 第{explanation['rank']}")
    print(f"   解释: {explanation['explanation']}")
    
    print("\n5. 模拟一步:")
    state = game.get_next_state(state, action)
    game.display(state)
    
    print("\n" + "=" * 60)
    print("✓ 演示完成")
    print("=" * 60)


if __name__ == '__main__':
    demo()
