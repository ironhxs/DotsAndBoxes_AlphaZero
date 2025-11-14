#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试早停机制
"""
import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

import numpy as np
import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import TransformerNet
from model.base_coach import Coach
import yaml

def test_early_stopping():
    """测试早停是否正常工作"""
    print("=" * 60)
    print("测试早停机制")
    print("=" * 60)
    
    # 加载配置
    with open('/root/DotsAndBoxes_AlphaZero/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建游戏和模型
    game = DotsAndBoxesGame()
    nnet = TransformerNet(game, config)
    
    # 修改配置用于快速测试
    config['epochs'] = 100  # 足够多的epochs来触发早停
    config['patience'] = 5  # 快速测试: 5个epoch不改进就停止
    config['min_delta'] = 0.0001
    config['batch_size'] = 128
    config['num_self_play_games'] = 10
    config['cuda'] = torch.cuda.is_available()
    
    # 创建coach
    coach = Coach(game, nnet, config)
    
    # 生成一些虚拟训练数据
    print("\n生成虚拟训练数据...")
    examples = []
    for _ in range(1000):
        # 随机状态
        state = np.random.randn(9, 6, 6).astype(np.float32)
        # 随机策略
        policy = np.random.dirichlet([1.0] * 60)
        # 随机价值
        value = np.random.choice([-1, 1])
        examples.append((state, policy, value))
    
    print(f"生成了 {len(examples)} 个训练样本")
    
    # 开始训练并观察早停
    print("\n开始训练(应该会触发早停机制)...")
    print("-" * 60)
    
    try:
        stats = coach.train(examples)
        print("\n训练完成!")
        print(f"实际训练的epoch数: {len(stats.get('total_losses', []))}")
        print(f"配置的最大epoch数: {config['epochs']}")
        
        if len(stats.get('total_losses', [])) < config['epochs']:
            print("✅ 早停机制成功触发!")
        else:
            print("❌ 早停机制未触发(可能是损失持续下降)")
            
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_early_stopping()
