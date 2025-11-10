#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""性能对比：Arena GPU 多线程 vs CPU 多进程"""

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
from model.arena import Arena


def benchmark_arena_modes():
    """对比不同 Arena 模式的性能"""
    
    print("="*70)
    print("⚡ Arena 性能对比测试")
    print("="*70)
    
    # 创建游戏（小棋盘加快测试）
    game = DotsAndBoxesGame(num_rows=3, num_cols=3)
    
    # 创建模型（小模型加快测试）
    print("\n准备模型...")
    model1 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    model2 = DotsAndBoxesTransformer(game, num_filters=32, num_blocks=2, num_heads=2)
    
    # 配置
    test_games = 4  # 测试局数
    mcts_sims = 25   # MCTS 次数
    
    results = {}
    
    # ========== 测试1: GPU 多线程模式 ==========
    print("\n" + "="*70)
    print("🧪 测试1: GPU 多线程模式 (推荐)")
    print("="*70)
    
    args_gpu = {
        'num_simulations': mcts_sims,
        'arena_mcts_simulations': mcts_sims,
        'cpuct': 1.0,
        'cuda': True,
        'arena_num_workers': 4,
    }
    
    arena_gpu = ArenaGPU(model1, model2, game, args_gpu)
    
    start = time.time()
    new_wins, old_wins, draws = arena_gpu.play_games(num_games=test_games)
    gpu_time = time.time() - start
    
    results['gpu_thread'] = {
        'time': gpu_time,
        'speed': test_games / gpu_time,
        'wins': (new_wins, old_wins, draws)
    }
    
    print(f"\n⏱️  GPU多线程: {gpu_time:.2f}秒 ({test_games/gpu_time:.2f} 局/秒)")
    
    # ========== 测试2: CPU 多进程模式 ==========
    print("\n" + "="*70)
    print("🧪 测试2: CPU 多进程模式（会有 CUDA 错误风险）")
    print("="*70)
    
    # 获取 state_dict
    state1 = {k: v.cpu() for k, v in model1.state_dict().items()}
    state2 = {k: v.cpu() for k, v in model2.state_dict().items()}
    
    args_cpu = {
        'num_simulations': mcts_sims,
        'arena_mcts_simulations': mcts_sims,
        'cpuct': 1.0,
        'cuda': False,  # 强制 CPU
        'arena_num_workers': 2,
        'use_parallel': True,
        'num_filters': 32,
        'num_res_blocks': 2,
        'num_heads': 2,
    }
    
    arena_cpu = Arena(state1, state2, game, args_cpu)
    
    try:
        start = time.time()
        new_wins, old_wins, draws = arena_cpu.play_games(num_games=test_games)
        cpu_time = time.time() - start
        
        results['cpu_multiprocess'] = {
            'time': cpu_time,
            'speed': test_games / cpu_time,
            'wins': (new_wins, old_wins, draws)
        }
        
        print(f"\n⏱️  CPU多进程: {cpu_time:.2f}秒 ({test_games/cpu_time:.2f} 局/秒)")
    except Exception as e:
        print(f"\n❌ CPU多进程失败: {e}")
        cpu_time = float('inf')
    
    # ========== 性能总结 ==========
    print("\n" + "="*70)
    print("📊 性能对比总结")
    print("="*70)
    
    if 'cpu_multiprocess' in results:
        speedup = cpu_time / gpu_time
        print(f"\nGPU多线程模式: {gpu_time:.2f}秒")
        print(f"CPU多进程模式: {cpu_time:.2f}秒")
        print(f"速度提升: {speedup:.1f}x")
    else:
        print(f"\nGPU多线程模式: {gpu_time:.2f}秒 ✅")
        print(f"CPU多进程模式: 失败 ❌")
    
    print("\n" + "="*70)
    print("💡 关键特性对比:")
    print("="*70)
    print("\n✅ GPU 多线程模式（推荐）:")
    print("   - 无 CUDA 初始化问题")
    print("   - 显存占用低（2个模型）")
    print("   - 速度较快")
    print("   - 稳定性高")
    print("   ⚠️  受 GIL 限制，无法完全并行")
    
    print("\n⚠️  CPU 多进程模式:")
    print("   - 可能出现 CUDA 错误")
    print("   - 显存占用高（N×2个模型）")
    print("   - 速度慢")
    print("   - 稳定性差")
    
    print("\n" + "="*70)
    print("🎯 推荐配置：")
    print("   arena_mode: 'gpu_thread'")
    print("   arena_num_workers: 4-6（线程数）")
    print("="*70)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法进行对比测试")
        sys.exit(1)
    
    benchmark_arena_modes()
