#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""极限GPU优化 + 多进程并行训练"""

import warnings
import os
import sys
import multiprocessing

# 设置多进程启动方法为 spawn (CUDA 兼容)
multiprocessing.set_start_method('spawn', force=True)

# ===== 抑制多进程导入时的警告 =====
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from model.game import DotsAndBoxesGame
from model.model import DotsAndBoxesNet
from model.coach_parallel import Coach


def extreme_train():
    """极限优化训练 - 最大化GPU和CPU利用率"""
    
    # 🔥 启用所有优化
    torch.backends.cudnn.benchmark = True
    os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP 线程数
    
    args = {
        # 游戏配置
        'num_rows': 5, 'num_cols': 5,
        
        # 🔥 CPU优化: 多进程并行
        'use_parallel': True,
        'num_workers': 6,  # 6个并行进程 (根据CPU核心数调整)
        
        # 🔥 MCTS配置: 极致精简
        'num_simulations': 12,  # 进一步减少 (15→12)
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temp_threshold': 15,
        
        # 🔥 训练规模: 超大数据量
        'num_iterations': 2,
        'num_episodes': 80,  # 更多游戏 (60→80)
        'max_queue_length': 20000,
        'num_iters_for_train_examples_history': 20,
        
        # 🔥 GPU优化: 超大批量和深度训练
        'epochs': 100,  # 充分训练（数据珍贵，多训练）
        'batch_size': 2048,  # 超超大批量 (1024→2048)
        'lr': 0.004,  # 适配大batch的学习率
        'weight_decay': 1e-4,
        
        # 🔥 模型配置: 超大模型
        'num_filters': 256,  # 巨大模型 (128→256)
        'num_res_blocks': 15,  # 超深网络 (10→15)
        
        # 其他
        'cuda': torch.cuda.is_available(),
        'checkpoint': './checkpoints',
        'checkpoint_interval': 1,
    }
    
    print("=" * 80)
    print("🚀🚀🚀 极限GPU+CPU优化 - AlphaZero点格棋训练 🚀🚀🚀")
    print("=" * 80)
    print(f"🎮 游戏: {args['num_rows']}x{args['num_cols']} 点格棋")
    print(f"🔥 GPU: {torch.cuda.get_device_name(0) if args['cuda'] else 'CPU模式'}")
    print(f"💻 CPU: {args['num_workers']}进程并行 (自我对弈加速{args['num_workers']}x)")
    print()
    print(f"📊 训练规模:")
    print(f"   ├─ 迭代次数: {args['num_iterations']}")
    print(f"   ├─ 每轮游戏: {args['num_episodes']}局")
    print(f"   ├─ MCTS模拟: {args['num_simulations']}次/步")
    print(f"   └─ 预计样本: ~{args['num_episodes'] * 30}个/迭代")
    print()
    print(f"🧠 网络架构:")
    print(f"   ├─ 通道数: {args['num_filters']}")
    print(f"   ├─ 残差层: {args['num_res_blocks']}")
    print(f"   └─ 参数量: 预计1000万+")
    print()
    print(f"⚡ 训练强度:")
    print(f"   ├─ 训练轮数: {args['epochs']} epochs")
    print(f"   ├─ 批量大小: {args['batch_size']}")
    print(f"   ├─ 学习率: {args['lr']}")
    print(f"   └─ 预计显存: 8-12 GB")
    print()
    print("=" * 80)
    print("🎯 优化策略总览:")
    print("=" * 80)
    print("✅ CPU优化:")
    print(f"   • {args['num_workers']}进程并行自我对弈 → 自我对弈加速{args['num_workers']}x")
    print(f"   • MCTS模拟降至{args['num_simulations']}次 → 单局速度提升50%")
    print(f"   • 预计自我对弈时间: ~{args['num_episodes'] * 2.5 / args['num_workers']:.0f}秒")
    print()
    print("✅ GPU优化:")
    print(f"   • 超大模型({args['num_filters']}通道×{args['num_res_blocks']}层) → 显存占用8GB+")
    print(f"   • 超大批量({args['batch_size']}) → GPU利用率最大化")
    print(f"   • 超长训练({args['epochs']}轮) → GPU持续高负载")
    print(f"   • 预计GPU训练时间: ~{args['epochs'] * 5:.0f}秒")
    print()
    print("✅ 时间分配预估:")
    est_selfplay = args['num_episodes'] * 2.5 / args['num_workers']
    est_training = args['epochs'] * 5
    total_time = est_selfplay + est_training
    print(f"   • 自我对弈: {est_selfplay:.0f}秒 ({est_selfplay/total_time*100:.0f}%)")
    print(f"   • GPU训练: {est_training:.0f}秒 ({est_training/total_time*100:.0f}%) ← 占主导")
    print(f"   • 总计: {total_time:.0f}秒 = {total_time/60:.1f}分钟/迭代")
    print("=" * 80)
    
    # 显存检查
    if args['cuda']:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n📊 GPU显存: {gpu_mem:.1f} GB")
        if gpu_mem < 16:
            print("⚠️  警告: 显存可能不足，建议降低 batch_size 或 num_filters")
            response = input("是否继续? (y/n): ")
            if response.lower() != 'y':
                return
    
    print("\n🚀 开始训练...")
    print("💡 提示: 可在另一终端运行 './quick_monitor.sh' 监控GPU状态")
    print()
    
    game = DotsAndBoxesGame(args['num_rows'], args['num_cols'])
    nnet = DotsAndBoxesNet(game, args['num_filters'], args['num_res_blocks'])
    
    if args['cuda']:
        nnet.cuda()
    
    param_count = sum(p.numel() for p in nnet.parameters())
    print(f"✓ 模型已创建: {param_count:,} 参数 ({param_count/1e6:.1f}M)")
    
    coach = Coach(game, nnet, args)
    coach.learn()
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"💾 模型保存位置: {args['checkpoint']}/")
    print(f"📈 下一步:")
    print(f"   1. 运行 'python play.py' 测试模型")
    print(f"   2. 继续训练: 修改 num_iterations 后重新运行")
    print(f"   3. 评估性能: 使用 evaluate.py")


if __name__ == '__main__':
    # 多进程需要
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    extreme_train()
