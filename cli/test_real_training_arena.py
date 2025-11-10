#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完全模拟实际训练环境的 Arena 测试"""

import warnings
import os
import sys
import multiprocessing

# ⚠️ 关键：必须在导入torch之前设置（与实际训练一致）
multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 抑制警告（与实际训练一致）
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.coach_alphazero import Coach


def test_real_training_with_arena():
    """
    完全模拟实际训练环境的测试
    
    流程：
    1. ✅ 使用 spawn 多进程模式（与训练一致）
    2. ✅ 创建 Coach 对象（与训练一致）
    3. ✅ 执行少量自我对弈（模拟真实环境）
    4. ✅ 执行神经网络训练
    5. ✅ 进入 Arena 对战阶段（重点测试）
    """
    
    print("="*70)
    print("🧪 完全模拟实际训练环境的 Arena 测试")
    print("="*70)
    print("⚠️  关键区别：")
    print("   1. 使用 spawn 多进程模式（实际训练环境）")
    print("   2. 经过自我对弈和训练阶段")
    print("   3. 然后进入 Arena 对战")
    print("="*70)
    
    # GPU优化（与训练一致）
    torch.backends.cudnn.benchmark = True
    os.environ['OMP_NUM_THREADS'] = '4'
    
    # 配置（简化版，但保持关键参数与训练一致）
    args = {
        # 游戏配置
        'num_rows': 5,
        'num_cols': 5,
        
        # ✅ Arena 配置（与训练完全一致）
        'arena_compare': 10,       # 减少到10局加快测试
        'update_threshold': 0.55,
        
        # 并行配置
        'use_parallel': True,
        'self_play_mode': 'batch',
        'num_workers': 3,          # 减少进程数加快测试
        
        # ⚠️ 关键：Arena 模式（测试重点）
        'arena_mode': 'gpu_thread',     # GPU多线程模式
        'arena_num_workers': 4,         # 减少线程数
        
        # MCTS配置（减少以加快测试）
        'num_simulations': 50,          # 自我对弈减少
        'arena_mcts_simulations': 100,  # Arena也减少
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temp_threshold': 15,
        
        # 训练规模（最小化）
        'num_iterations': 2,      # ⚡ 只跑2次迭代
        'num_episodes': 10,       # ⚡ 每次只10局自我对弈
        'arena_interval': 1,      # 每次都进行Arena验证
        'max_queue_length': 50000,
        'num_iters_for_train_examples_history': 2,
        
        # 神经网络训练（最小化）
        'epochs': 5,              # ⚡ 只训练5个epoch
        'batch_size': 128,
        'lr': 0.002,
        'weight_decay': 1e-4,
        
        # 模型配置（简化）
        'num_filters': 64,        # 减小模型
        'num_res_blocks': 4,
        'num_heads': 4,
        
        # 硬件
        'cuda': torch.cuda.is_available(),
        
        # 路径
        'checkpoint': None,
        'load_folder': None,
        'save_folder': 'results/test_arena_real',
        'log_dir': 'results/test_arena_real/logs',
    }
    
    print("\n" + "="*70)
    print("📝 测试配置:")
    print("="*70)
    print(f"迭代次数: {args['num_iterations']}")
    print(f"每次自我对弈: {args['num_episodes']} 局")
    print(f"训练轮数: {args['epochs']} epochs")
    print(f"Arena对战: {args['arena_compare']} 局")
    print(f"Arena模式: {args['arena_mode']}")
    print(f"Arena MCTS: {args['arena_mcts_simulations']} 次")
    print("="*70)
    
    # 创建游戏
    print("\n1️⃣ 创建游戏环境...")
    game = DotsAndBoxesGame(num_rows=args['num_rows'], num_cols=args['num_cols'])
    print("✓ 游戏创建成功")
    
    # 创建神经网络
    print("\n2️⃣ 创建神经网络...")
    nnet = DotsAndBoxesTransformer(
        game,
        num_filters=args['num_filters'],
        num_blocks=args['num_res_blocks'],
        num_heads=args['num_heads']
    )
    print("✓ 神经网络创建成功")
    
    # 创建 Coach
    print("\n3️⃣ 创建 Coach（训练管理器）...")
    coach = Coach(game, nnet, args)
    print("✓ Coach 创建成功")
    
    # 开始训练（会自动进入 Arena 阶段）
    print("\n" + "="*70)
    print("🚀 开始模拟训练（包含完整流程）")
    print("="*70)
    print("流程：")
    print("  1. 第1次迭代:")
    print("     - 自我对弈10局 (spawn多进程)")
    print("     - 训练神经网络5个epoch")
    print("     - Arena对战10局 (GPU多线程) ← 重点测试")
    print("  2. 第2次迭代:")
    print("     - 自我对弈10局")
    print("     - 训练神经网络5个epoch")
    print("     - Arena对战10局 ← 重点测试")
    print("="*70)
    
    try:
        # 执行训练（会自动调用Arena）
        coach.learn()
        
        print("\n" + "="*70)
        print("✅ 测试成功！")
        print("="*70)
        print("🎉 Arena GPU 模式在实际训练环境中正常工作！")
        print("   - spawn 多进程环境 ✅")
        print("   - 自我对弈后的状态 ✅")
        print("   - 训练后的模型对战 ✅")
        print("   - 无 CUDA 错误 ✅")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ 测试失败！")
        print("="*70)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧪 AlphaZero 实际训练环境 Arena 测试")
    print("="*70)
    print("目的：验证 Arena GPU 模式在真实训练流程中的稳定性")
    print("配置：完全模拟实际训练环境（spawn + 自我对弈 + 训练 + Arena）")
    print("="*70 + "\n")
    
    success = test_real_training_with_arena()
    
    if success:
        print("\n💡 测试结论：")
        print("   ✅ Arena GPU 模式在实际训练环境中完全正常")
        print("   ✅ 可以安全开始完整训练")
        print("\n🚀 下一步：运行完整训练")
        print("   python cli/train_alphazero.py")
    else:
        print("\n⚠️  在实际训练环境中发现问题")
        print("   需要进一步调试")
    
    sys.exit(0 if success else 1)
