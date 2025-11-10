#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试训练流程 - 验证 Arena GPU 模式"""

import warnings
import os
import sys
import multiprocessing

# 设置多进程启动方法为 spawn (CUDA 兼容)
multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 抑制多进程警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer as DotsAndBoxesNet
from model.coach_alphazero import Coach


def quick_test_train():
    """
    快速测试训练流程 - 专注验证 Arena GPU 模式
    
    策略：
    1. 极少的自我对弈局数（快速生成数据）
    2. 极少的训练轮数（快速更新模型）
    3. 快速进入 Arena 对战验证
    """
    
    print("="*70)
    print("🧪 快速测试：训练 + Arena GPU 验证")
    print("="*70)
    print("⚡ 使用最小化配置，快速进入 Arena 对战阶段")
    print("="*70)
    
    # GPU优化
    torch.backends.cudnn.benchmark = True
    os.environ['OMP_NUM_THREADS'] = '4'
    
    args = {
        # 游戏配置（使用小棋盘加速）
        'num_rows': 3,  # 3x3 小棋盘（对战更快）
        'num_cols': 3,
        
        # ✅ AlphaZero核心: Arena对战配置
        'arena_compare': 6,        # ⚡ 只测试6局（快速验证）
        'update_threshold': 0.55,
        
        # 并行配置
        'use_parallel': True,
        'self_play_mode': 'batch',
        'num_workers': 2,          # ⚡ 2个进程（更快启动）
        
        # 🎯 Arena配置 - 关键测试点
        'arena_mode': 'gpu_thread',     # 🚀 GPU多线程模式
        'arena_num_workers': 4,         # 4个线程
        
        # MCTS配置 - ⚡ 大幅减少搜索次数
        'num_simulations': 10,         # 自我对弈：10次（加速测试）
        'arena_mcts_simulations': 20,  # Arena：20次（加速测试）
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temp_threshold': 10,
        
        # 训练规模 - ⚡ 最小化配置
        'num_iterations': 2,       # ⚡ 只跑2次迭代
        'num_episodes': 20,        # ⚡ 每次20局自我对弈（确保足够数据）
        'arena_interval': 1,       # 每次迭代都进行Arena验证
        'max_queue_length': 5000,
        'num_iters_for_train_examples_history': 2,
        
        # 神经网络训练 - ⚡ 快速训练
        'epochs': 10,              # ⚡ 只训练10轮（原300轮）
        'batch_size': 128,         # ⚡ 小批量（确保有足够batch）
        'lr': 0.002,
        'weight_decay': 1e-4,
        
        # 模型配置（小模型加速）
        'num_filters': 64,         # 64通道（原128）
        'num_res_blocks': 4,       # 4个残差块（原8）
        'num_heads': 4,            # 4个注意力头（原8）
        
        # 保存配置
        'checkpoint': './results/test_arena/',
        'load_model': False,
        
        # 硬件
        'cuda': torch.cuda.is_available(),
        'use_amp': True,
        
        # 日志
        'verbose': True,
    }
    
    print(f"\n📋 测试配置:")
    print(f"   棋盘: {args['num_rows']}x{args['num_cols']}")
    print(f"   迭代次数: {args['num_iterations']}")
    print(f"   每次自我对弈: {args['num_episodes']}局")
    print(f"   训练轮数: {args['epochs']}")
    print(f"   Arena对战: {args['arena_compare']}局")
    print(f"   Arena模式: {args['arena_mode']} (关键测试点)")
    print(f"   Arena MCTS: {args['arena_mcts_simulations']}次")
    print(f"   CUDA: {args['cuda']}")
    print("="*70)
    
    # 创建游戏
    game = DotsAndBoxesGame(
        num_rows=args['num_rows'],
        num_cols=args['num_cols']
    )
    
    # 创建神经网络
    nnet = DotsAndBoxesNet(
        game,
        num_filters=args['num_filters'],
        num_blocks=args['num_res_blocks'],
        num_heads=args['num_heads']
    )
    
    if args['cuda']:
        nnet.cuda()
        print("✓ 模型已加载到 GPU")
    
    # 创建 Coach
    coach = Coach(game, nnet, args)
    
    print("\n🚀 开始快速测试训练...")
    print("="*70)
    
    try:
        # 开始训练（会快速进入 Arena 阶段）
        coach.learn()
        
        print("\n" + "="*70)
        print("✅ 测试成功完成！")
        print("="*70)
        print("🎉 Arena GPU 模式正常工作，未出现 CUDA 错误！")
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
    print("\n" + "="*70)
    print("🧪 AlphaZero 快速测试 - Arena GPU 验证")
    print("="*70)
    print("目标：快速进入 Arena 阶段，验证 GPU 多线程模式")
    print("预期：Arena 对战使用 GPU 加速，无 CUDA 错误")
    print("="*70 + "\n")
    
    success = quick_test_train()
    
    if success:
        print("\n🎊 恭喜！Arena GPU 修复验证成功！")
        print("💡 现在可以使用完整配置进行正式训练了。")
    else:
        print("\n⚠️  测试失败，请检查错误信息。")
    
    sys.exit(0 if success else 1)
