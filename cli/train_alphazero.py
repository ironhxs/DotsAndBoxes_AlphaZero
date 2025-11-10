#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真正的 AlphaZero 训练 - 包含 Arena 对战验证"""

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
# 使用现代 Transformer 架构
from model.model_transformer import DotsAndBoxesTransformer as DotsAndBoxesNet
from model.coach_alphazero import Coach


def true_alphazero_train():
    """
    真正的AlphaZero训练流程:
    1. 自我对弈生成训练数据
    2. 训练神经网络
    3. 新模型 vs 旧模型 Arena对战
    4. 只有胜率>55%才接受新模型
    """
    
    # GPU优化
    torch.backends.cudnn.benchmark = True
    os.environ['OMP_NUM_THREADS'] = '4'
    
    args = {
        # 游戏配置
        'num_rows': 5,
        'num_cols': 5,
        
        # ✅ AlphaZero核心: Arena对战配置
        'arena_compare': 20,       # Arena对战局数 (偶数，建议10-40局)
        'update_threshold': 0.55,  # 新模型必须>55%胜率才接受
        
        # 并行配置
        'use_parallel': True,
        'self_play_mode': 'batch',  # ⚠️ 暂时fallback到multiprocess（GIL限制）
        'num_workers': 10,          # 自我对弈CPU进程数
        
        # Arena配置
        'arena_mode': 'gpu_multiprocess',  # 🚀 GPU多进程模式（推荐：真正的多核并行）
        'arena_num_workers': 2,            # Arena进程数（⚠️ 每个进程2个模型，避免OOM）
        
        # MCTS配置 - ⚡ 提高搜索质量
        'num_simulations': 100,        # 自我对弈MCTS (探索+质量平衡)
        'arena_mcts_simulations': 200, # Arena评估MCTS (2倍，确保准确)
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25,
        'temp_threshold': 15,
        
        # 训练规模 - 高质量数据策略
        'num_iterations': 600,     # 600次迭代
        'num_episodes': 80,        
        'arena_interval': 1,      # 每20次迭代进行一次Arena验证
        'max_queue_length': 50000,
        'num_iters_for_train_examples_history': 20,
        
        # 神经网络训练
        'epochs': 300,            
        'batch_size': 2048,        
        'lr': 0.002,
        'weight_decay': 1e-4,
        
        # 简化模型配置 (更快推理 + 100次MCTS)
        'num_filters': 256,      # 384→256 (减少33%参数)
        'num_res_blocks': 12,    # 18→12 (减少6层)
        'num_heads': 8,          # 12→8 (减少注意力头)
        
        # 其他
        'cuda': torch.cuda.is_available(),
        'checkpoint': './results/checkpoints',  # 保存在 results/checkpoints/
        'checkpoint_interval': 20,  # 每20次迭代保存一次 checkpoint
    }
    
    print("=" * 80)
    print("🧠 AlphaZero 训练系统 - 高质量搜索优化版")
    print("=" * 80)
    print(f"训练迭代: {args['num_iterations']} 次")
    print(f"每次迭代: {args['num_episodes']} 局自我对弈")
    print(f"Arena验证: 每 {args['arena_interval']} 次迭代 ({args['arena_compare']} 局)")
    print(f"更新阈值: {args['update_threshold']*100}% 胜率")
    print(f"⚙️  自我对弈: {args['num_workers']} CPU进程（各自GPU） | MCTS={args['num_simulations']}次")
    print(f"⚙️  Arena对战: {args['arena_num_workers']} CPU进程（各自GPU） | MCTS={args['arena_mcts_simulations']}次")
    print(f"⚠️  注意: Python GIL限制，批量推理模式暂不可用")
    print(f"神经网络: Transformer + ConvNeXt ({args['num_filters']}d × {args['num_res_blocks']} blocks)")
    print(f"注意力机制: {args['num_heads']}-head Self-Attention")
    print(f"训练规模: Batch={args['batch_size']}, Epochs={args['epochs']}")
    print(f"GPU加速: {'✅ CUDA可用' if args['cuda'] else '❌ 仅CPU'}")
    print("=" * 80)
    print()
    
    # 初始化现代模型
    game = DotsAndBoxesGame(args['num_rows'], args['num_cols'])
    nnet = DotsAndBoxesNet(
        game, 
        num_filters=args['num_filters'], 
        num_blocks=args['num_res_blocks'],
        num_heads=args['num_heads']
    )
    
    if args['cuda']:
        nnet.cuda()
        print("✓ 模型已转移至GPU\n")
    
    # 开始训练
    coach = Coach(game, nnet, args)
    
    print("🚀 开始 AlphaZero 训练...")
    print("   每次迭代包含: 自我对弈 → 训练 → Arena对战 → 模型筛选\n")
    
    coach.learn()
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"最佳模型保存在: {args['checkpoint']}/best_*.pth")
    print(f"最新模型保存在: {args['checkpoint']}/latest.pth")
    print("\n使用以下命令验证模型:")
    print(f"  python play.py")
    print(f"  python evaluate_model.py")
    print("=" * 80)


if __name__ == '__main__':
    # 必须在主块中再次设置 (确保生效)
    multiprocessing.set_start_method('spawn', force=True)
    true_alphazero_train()
