#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Arena GPU 多线程版本"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu import ArenaGPU


def test_arena_gpu():
    """测试 Arena GPU 多线程版本"""
    print("="*70)
    print("🧪 测试 Arena GPU 多线程版本")
    print("="*70)
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法测试 GPU 版本")
        return False
    
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 创建游戏
    game = DotsAndBoxesGame(num_rows=3, num_cols=3)
    
    # 创建两个模型
    model_args = {
        'num_filters': 64,
        'num_res_blocks': 4,
        'num_heads': 4
    }
    
    print("\n1️⃣ 创建模型...")
    model1 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    
    model2 = DotsAndBoxesTransformer(
        game,
        num_filters=model_args['num_filters'],
        num_blocks=model_args['num_res_blocks'],
        num_heads=model_args['num_heads']
    )
    
    print("✓ 模型创建成功")
    
    # 创建 ArenaGPU
    print("\n2️⃣ 创建 ArenaGPU...")
    args = {
        'num_simulations': 25,  # 减少模拟次数加快测试
        'arena_mcts_simulations': 25,
        'cpuct': 1.0,
        'cuda': True,
        'num_filters': 64,
        'num_res_blocks': 4,
        'num_heads': 4,
        'arena_num_workers': 2,  # 使用2个线程
    }
    
    arena = ArenaGPU(model1, model2, game, args)
    print("✓ ArenaGPU 创建成功")
    
    # 运行测试对战
    print("\n3️⃣ 运行 GPU 加速对战...")
    print("🚀 使用 GPU + 多线程，速度快且显存占用低")
    
    try:
        # 测试4局
        one_won, two_won, draws = arena.play_games(num_games=4)
        
        print("\n✅ 测试成功！")
        print(f"   Player1 胜: {one_won}")
        print(f"   Player2 胜: {two_won}")
        print(f"   平局: {draws}")
        print("\n🎉 Arena GPU 多线程版本正常工作！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_arena_gpu()
    sys.exit(0 if success else 1)
