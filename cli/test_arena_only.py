#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""超快速测试 - 跳过自我对弈，直接测试 Arena"""

import warnings
import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
import numpy as np
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu import ArenaGPU


def ultra_fast_arena_test():
    """
    超快速 Arena 测试 - 直接创建模型并对战
    
    目的：验证 Arena GPU 模式在实际训练环境中是否正常
    """
    
    print("="*70)
    print("⚡ 超快速 Arena 测试")
    print("="*70)
    print("策略：跳过自我对弈和训练，直接测试 Arena GPU 对战")
    print("="*70)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 创建游戏（使用与训练相同的配置）
    print("\n1️⃣ 创建游戏环境（5x5棋盘）...")
    game = DotsAndBoxesGame(num_rows=5, num_cols=5)
    print("✓ 游戏创建成功")
    
    # 创建两个模型（模拟训练中的新旧模型）
    print("\n2️⃣ 创建两个模型（模拟新旧模型对战）...")
    model_args = {
        'num_filters': 128,
        'num_res_blocks': 8,
        'num_heads': 8
    }
    
    print("   创建模型1（新模型）...")
    model1 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    
    print("   创建模型2（旧模型）...")
    model2 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    
    # 随机初始化参数（模拟训练后的参数差异）
    print("   随机初始化参数差异...")
    for p in model1.parameters():
        p.data.add_(torch.randn_like(p) * 0.01)
    
    print("✓ 模型创建成功")
    
    # 配置 Arena（使用与训练相同的配置）
    print("\n3️⃣ 配置 Arena GPU 模式...")
    arena_args = {
        # MCTS 配置
        'num_simulations': 100,
        'arena_mcts_simulations': 200,  # 与训练配置一致
        'cpuct': 1.0,
        
        # Arena 配置
        'arena_num_workers': 6,  # 6个线程
        
        # 硬件
        'cuda': True,
    }
    
    print(f"   Arena模式: GPU多线程")
    print(f"   MCTS次数: {arena_args['arena_mcts_simulations']}")
    print(f"   并行度: {arena_args['arena_num_workers']} 线程")
    
    # 创建 Arena
    print("\n4️⃣ 创建 ArenaGPU...")
    try:
        arena = ArenaGPU(model1, model2, game, arena_args)
        print("✓ ArenaGPU 创建成功")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return False
    
    # 执行对战测试
    print("\n5️⃣ 执行Arena对战测试...")
    print("="*70)
    print("🎯 测试配置：20局对战（与训练一致）")
    print("="*70)
    
    try:
        # 执行 20 局对战（与训练配置一致）
        new_wins, old_wins, draws = arena.play_games(num_games=20)
        
        print("\n" + "="*70)
        print("✅ Arena 对战测试成功！")
        print("="*70)
        print(f"📊 对战结果:")
        print(f"   新模型胜: {new_wins}")
        print(f"   旧模型胜: {old_wins}")
        print(f"   平局: {draws}")
        print(f"   新模型胜率: {(new_wins + 0.5*draws)/20*100:.1f}%")
        print("="*70)
        print("🎉 Arena GPU 模式正常工作，未出现 CUDA 错误！")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ Arena 对战失败！")
        print("="*70)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧪 AlphaZero Arena GPU 超快速测试")
    print("="*70)
    print("目的：验证训练环境中 Arena GPU 模式的稳定性")
    print("配置：与实际训练完全一致（5x5棋盘，200次MCTS）")
    print("="*70 + "\n")
    
    success = ultra_fast_arena_test()
    
    if success:
        print("\n💡 测试结论：")
        print("   ✅ Arena GPU 模式正常工作")
        print("   ✅ 未出现 CUDNN_STATUS_NOT_INITIALIZED 错误")
        print("   ✅ 可以安全地开始正式训练")
        print("\n🚀 下一步：运行完整训练")
        print("   python cli/train_alphazero.py")
    else:
        print("\n⚠️  Arena GPU 模式测试失败")
        print("   请检查上方错误信息")
    
    sys.exit(0 if success else 1)
