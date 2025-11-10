#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 Arena 多进程修复是否生效"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena import Arena


def test_arena_multiprocess():
    """测试 Arena 多进程对战（修复后应该正常运行）"""
    print("="*70)
    print("🧪 测试 Arena 多进程修复")
    print("="*70)
    
    # 创建游戏（使用正确的参数名）
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
    
    # 获取 state_dict（移到CPU）
    state1 = {k: v.cpu() for k, v in model1.state_dict().items()}
    state2 = {k: v.cpu() for k, v in model2.state_dict().items()}
    
    print("✓ 模型创建成功")
    
    # 创建 Arena
    print("\n2️⃣ 创建 Arena...")
    args = {
        'num_simulations': 25,  # 减少模拟次数加快测试
        'arena_mcts_simulations': 25,  # Arena MCTS 次数
        'cpuct': 1.0,
        'cuda': True,  # 虽然设置为True，但会被强制为CPU
        'num_filters': 64,
        'num_res_blocks': 4,
        'num_heads': 4,
        'arena_num_workers': 2,  # 使用2个进程
        'use_parallel': True  # 启用并行
    }
    
    arena = Arena(state1, state2, game, args)
    print("✓ Arena 创建成功")
    
    # 运行少量对战测试
    print("\n3️⃣ 运行测试对战...")
    print("⚠️  使用CPU模式，速度较慢但稳定")
    
    try:
        # 只测试4局
        one_won, two_won, draws = arena.play_games(num_games=4)
        
        print("\n✅ 测试成功！")
        print(f"   Player1 胜: {one_won}")
        print(f"   Player2 胜: {two_won}")
        print(f"   平局: {draws}")
        print("\n🎉 Arena 多进程修复验证通过！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_arena_multiprocess()
    sys.exit(0 if success else 1)
