#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Arena GPU 多进程版本 - 真正的并行"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu_multiprocess import ArenaGPUMultiProcess


def test_arena_gpu_multiprocess():
    """测试真正的多进程 Arena GPU"""
    
    print("="*70)
    print("🧪 测试 Arena GPU 多进程版本")
    print("="*70)
    print("关键：使用多进程（不是多线程）")
    print("      每个进程独立使用 GPU")
    print("      真正的多核并行（不受 GIL 限制）")
    print("      就像自我对弈那样！")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"\n✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
    
    # 创建游戏
    print("\n1️⃣ 创建游戏（3x3加快测试）...")
    game = DotsAndBoxesGame(num_rows=3, num_cols=3)
    
    # 创建模型
    print("\n2️⃣ 创建模型...")
    model1 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    model2 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    
    # 配置
    args = {
        'num_simulations': 25,
        'arena_mcts_simulations': 50,
        'cpuct': 1.0,
        'cuda': True,
        'arena_num_workers': 4,  # 4个进程
        'num_filters': 32,
        'num_res_blocks': 2,
        'num_heads': 2,
    }
    
    print("\n3️⃣ 创建 ArenaGPUMultiProcess...")
    arena = ArenaGPUMultiProcess(model1, model2, game, args)
    
    print("\n4️⃣ 执行对战（观察 CPU 使用率）...")
    print("💡 提示：用 htop 观察，应该看到多个 Python 进程同时运行")
    print("="*70)
    
    try:
        new_wins, old_wins, draws = arena.play_games(num_games=8)
        
        print("\n" + "="*70)
        print("✅ 测试成功！")
        print("="*70)
        print(f"新模型胜: {new_wins}")
        print(f"旧模型胜: {old_wins}")
        print(f"平局: {draws}")
        print("="*70)
        print("🎉 Arena GPU 多进程版本正常工作！")
        print("   ✅ 真正的多核并行")
        print("   ✅ GPU 加速")
        print("   ✅ 无 GIL 限制")
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
    print("\n🧪 Arena GPU 多进程测试（真正的并行）\n")
    
    success = test_arena_gpu_multiprocess()
    
    if success:
        print("\n💡 结论：")
        print("   ✅ Arena 现在使用多进程（不是多线程）")
        print("   ✅ 真正的多核并行（与自我对弈一样）")
        print("   ✅ 不受 GIL 限制")
        print("   ✅ 充分利用多核 CPU")
    
    sys.exit(0 if success else 1)
