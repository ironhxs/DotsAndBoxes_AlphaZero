#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""性能对比：多线程 vs 多进程"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
import os
import time
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu import ArenaGPU
from model.arena_gpu_multiprocess import ArenaGPUMultiProcess


def benchmark():
    """性能对比"""
    
    print("="*70)
    print("⚡ Arena 性能对比：多线程 vs 多进程")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return
    
    # 创建游戏和模型
    game = DotsAndBoxesGame(num_rows=3, num_cols=3)
    model1 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    model2 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    
    test_games = 8
    
    # 测试1: 多线程
    print("\n" + "="*70)
    print("🧪 测试1: GPU 多线程（ThreadPoolExecutor）")
    print("="*70)
    
    args_thread = {
        'num_simulations': 25,
        'arena_mcts_simulations': 50,
        'cpuct': 1.0,
        'cuda': True,
        'arena_num_workers': 4,
    }
    
    arena_thread = ArenaGPU(model1, model2, game, args_thread)
    start = time.time()
    arena_thread.play_games(num_games=test_games)
    thread_time = time.time() - start
    
    print(f"\n⏱️  多线程耗时: {thread_time:.2f}秒")
    
    # 测试2: 多进程
    print("\n" + "="*70)
    print("🧪 测试2: GPU 多进程（ProcessPoolExecutor）")
    print("="*70)
    
    args_process = {
        'num_simulations': 25,
        'arena_mcts_simulations': 50,
        'cpuct': 1.0,
        'cuda': True,
        'arena_num_workers': 4,
        'num_filters': 32,
        'num_res_blocks': 2,
        'num_heads': 2,
    }
    
    arena_process = ArenaGPUMultiProcess(model1, model2, game, args_process)
    start = time.time()
    arena_process.play_games(num_games=test_games)
    process_time = time.time() - start
    
    print(f"\n⏱️  多进程耗时: {process_time:.2f}秒")
    
    # 对比
    print("\n" + "="*70)
    print("📊 性能对比")
    print("="*70)
    speedup = thread_time / process_time
    print(f"\n多线程: {thread_time:.2f}秒")
    print(f"多进程: {process_time:.2f}秒")
    print(f"速度提升: {speedup:.2f}x")
    
    print("\n" + "="*70)
    print("💡 关键区别:")
    print("="*70)
    print("\n多线程（ThreadPoolExecutor）:")
    print("   ❌ 受 Python GIL 限制")
    print("   ❌ CPU 密集型任务无法并行")
    print("   ❌ 实际只用 1-2 个 CPU 核心")
    
    print("\n多进程（ProcessPoolExecutor）:")
    print("   ✅ 不受 GIL 限制")
    print("   ✅ 真正的多核并行")
    print("   ✅ 充分利用所有 CPU 核心")
    print("   ✅ 与自我对弈同样方式")
    
    print("\n" + "="*70)
    print(f"🎯 结论：多进程比多线程快 {speedup:.2f}x")
    print("="*70)


if __name__ == '__main__':
    benchmark()
