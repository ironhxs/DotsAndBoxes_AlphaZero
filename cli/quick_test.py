#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""4060 快速测试脚本 - 验证代码能否跑通"""

import warnings
import os
import sys
import multiprocessing

# 设置多进程启动方法为 spawn (CUDA 兼容)
multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 抑制警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer as DotsAndBoxesNet
from model.coach import Coach


def quick_test():
    """快速测试代码是否能跑通"""
    print("=" * 60)
    print("4060 Laptop 快速测试")
    print("=" * 60)
    
    # 1. 检查CUDA
    print("\n[1/5] 检查CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("✗ CUDA不可用，将使用CPU")
    
    # 2. 检查OpenSpiel
    print("\n[2/5] 检查OpenSpiel...")
    try:
        import pyspiel
        print(f"✓ OpenSpiel已安装")
    except ImportError as e:
        print(f"✗ OpenSpiel导入失败: {e}")
        return False
    
    # 3. 测试游戏环境
    print("\n[3/5] 测试游戏环境...")
    try:
        game = DotsAndBoxesGame()
        print(f"✓ 游戏环境初始化成功")
        print(f"  棋盘大小: {game.num_rows}x{game.num_cols}")
        print(f"  动作空间: {game.get_action_size()}")
    except Exception as e:
        print(f"✗ 游戏环境初始化失败: {e}")
        return False
    
    # 4. 测试模型
    print("\n[4/5] 测试模型...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用测试配置 - 只使用DotsAndBoxesTransformer支持的参数
        model = DotsAndBoxesNet(
            game=game,
            num_filters=64,    # 简化 (256→64)
            num_blocks=4,      # 简化 (12→4)
            num_heads=4,       # 简化 (8→4)
            input_channels=9   # OpenSpiel标准
        ).to(device)
        
        # 测试前向传播
        board_h = game.num_rows + 1  # 6
        board_w = game.num_cols + 1  # 6
        test_input = torch.randn(2, 9, board_h, board_w).to(device)
        pi, v = model(test_input)
        
        print(f"✓ 模型初始化成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        print(f"  策略输出形状: {pi.shape}")
        print(f"  价值输出形状: {v.shape}")
        
        # 显存占用
        if torch.cuda.is_available():
            print(f"  显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试Coach
    print("\n[5/5] 测试Coach初始化...")
    try:
        # Coach需要args字典
        test_args = {
            'num_iterations': 5,
            'num_episodes': 10,
            'temp_threshold': 10,
            'num_simulations': 50,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'max_queue_length': 5000,
            'num_iters_for_train_examples_history': 20,
            'checkpoint_interval': 2,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'epochs': 2,
            'batch_size': 64,
        }
        
        coach = Coach(
            game=game,
            nnet=model,
            args=test_args
        )
        print(f"✓ Coach初始化成功")
    except Exception as e:
        print(f"✗ Coach初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！代码可以正常运行")
    print("=" * 60)
    print("\n现在可以运行完整训练:")
    print("  python cli/test_train_4060.py")
    return True


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
