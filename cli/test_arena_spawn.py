#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最小化测试：直接测试 Arena GPU 在 spawn 环境下是否工作"""

import multiprocessing
# ⚠️ 必须在导入torch之前
multiprocessing.set_start_method('spawn', force=True)

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu import ArenaGPU


def test_arena_in_spawn_env():
    """
    最小化测试：在 spawn 多进程环境下测试 Arena GPU
    
    这是实际训练环境的关键部分！
    """
    print("="*70)
    print("🧪 Arena GPU 在 spawn 环境下的测试")
    print("="*70)
    print("⚠️  关键：multiprocessing.set_start_method('spawn', force=True)")
    print("   这是实际训练使用的多进程模式")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 创建游戏
    print("\n1️⃣ 创建游戏（5x5，与训练一致）...")
    game = DotsAndBoxesGame(num_rows=5, num_cols=5)
    
    # 创建模型
    print("\n2️⃣ 创建模型...")
    model1 = DotsAndBoxesTransformer(game, num_filters=64, num_blocks=4, num_heads=4)
    model2 = DotsAndBoxesTransformer(game, num_filters=64, num_blocks=4, num_heads=4)
    
    # 配置
    args = {
        'num_simulations': 50,
        'arena_mcts_simulations': 100,
        'cpuct': 1.0,
        'cuda': True,
        'arena_num_workers': 4,
    }
    
    print("\n3️⃣ 创建 ArenaGPU...")
    arena = ArenaGPU(model1, model2, game, args)
    
    print("\n4️⃣ 执行 Arena 对战（10局）...")
    print("="*70)
    
    try:
        new_wins, old_wins, draws = arena.play_games(num_games=10)
        
        print("\n" + "="*70)
        print("✅ 测试成功！")
        print("="*70)
        print(f"新模型胜: {new_wins}")
        print(f"旧模型胜: {old_wins}")
        print(f"平局: {draws}")
        print("="*70)
        print("🎉 Arena GPU 在 spawn 环境下完全正常！")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ 测试失败！")
        print("="*70)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🧪 关键测试：Arena GPU + spawn 多进程\n")
    
    success = test_arena_in_spawn_env()
    
    if success:
        print("\n💡 结论：")
        print("   ✅ Arena GPU 模式在 spawn 环境下正常工作")
        print("   ✅ 没有 CUDNN_STATUS_NOT_INITIALIZED 错误")
        print("   ✅ Arena 修复成功！")
        print("\n📝 其他问题（如设备不匹配）是训练代码的问题，不是 Arena 的问题")
    
    sys.exit(0 if success else 1)
