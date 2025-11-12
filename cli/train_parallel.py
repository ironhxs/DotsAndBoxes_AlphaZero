#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行化训练脚本 - GPU 优化版

主要优化：
1. 批量 MCTS 推理（batch_size=32）
2. 并行自我对弈（8 局同时进行）
3. 提升 GPU 利用率（减少 CPU-GPU 传输）

预期提升：
- GPU 利用率: 15% → 60%+
- 训练速度: 2-3倍提升
"""

import warnings
import os
import sys
import multiprocessing
import torch
import yaml

# 设置多进程启动方法为 spawn (CUDA 兼容)
multiprocessing.set_start_method('spawn', force=True)

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 抑制多进程警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer as DotsAndBoxesNet
from model.coach_parallel import ParallelCoach


def load_config_from_yaml(config_path='config/config.yaml'):
    """
    从 YAML 配置文件加载所有配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 完整的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 手动加载子配置文件（支持 Hydra defaults 结构）
    config_dir = os.path.dirname(config_path)
    
    # 从 defaults 中提取子配置名称，如果没有则使用默认值
    defaults = config.get('defaults', [])
    game_name = 'dots_and_boxes'
    model_name = 'transformer'
    trainer_name = 'alphazero'
    
    for item in defaults:
        if isinstance(item, dict):
            if 'game' in item:
                game_name = item['game']
            elif 'model' in item:
                model_name = item['model']
            elif 'trainer' in item:
                trainer_name = item['trainer']
    
    # 加载子配置文件
    with open(os.path.join(config_dir, 'game', f'{game_name}.yaml'), 'r', encoding='utf-8') as f:
        game_config = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, 'model', f'{model_name}.yaml'), 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, 'trainer', f'{trainer_name}.yaml'), 'r', encoding='utf-8') as f:
        trainer_config = yaml.safe_load(f)
    
    # 构建统一的 args 字典
    args = {
        # 游戏配置
        'num_rows': game_config['num_rows'],
        'num_cols': game_config['num_cols'],
        
        # 训练基础参数
        'num_iterations': config.get('num_iterations', trainer_config['num_iterations']),
        'num_episodes': config.get('num_self_play_games', trainer_config['num_episodes']),
        'temp_threshold': config.get('temperature_threshold', trainer_config['temp_threshold']),
        'update_threshold': trainer_config.get('update_threshold', 0.55),
        'max_queue_length': config.get('replay_buffer_size', 200000),
        'num_iters_for_train_examples_history': trainer_config.get('num_iters_for_train_examples_history', 20),
        
        # Arena 配置
        'arena_compare': config.get('arena_compare', 40),
        'arena_num_workers': config.get('arena_num_workers', 8),
        'arena_random_start': True,
        'arena_mcts_simulations': config.get('arena_mcts_simulations', trainer_config['num_simulations'] * 2),
        'arena_mode': config.get('arena_mode', 'serial'),  # 添加 arena_mode 配置
        'update_threshold': config.get('update_threshold', config.get('arena_threshold', 0.55)),
        
        # MCTS 参数
        'num_simulations': config.get('num_simulations', trainer_config['num_simulations']),
        'cpuct': config.get('cpuct', trainer_config['cpuct']),
        'dirichlet_alpha': config.get('dirichlet_alpha', trainer_config.get('dirichlet_alpha', 0.3)),
        'dirichlet_epsilon': config.get('dirichlet_epsilon', trainer_config.get('dirichlet_epsilon', 0.25)),
        
        # 训练参数
        'epochs': config.get('train_epochs', trainer_config['epochs']),
        'batch_size': config.get('batch_size', trainer_config['batch_size']),
        'lr': float(config.get('learning_rate', trainer_config['lr'])),
        'weight_decay': config.get('weight_decay', 1e-4),
        'grad_clip': trainer_config.get('max_grad_norm', 5.0),
        
        # 并行优化参数
        'use_parallel': True,
        'parallel_mode': config.get('parallel_mode', 'full'),
        'use_multiprocess': config.get('use_multiprocess', True),
        'self_play_mode': 'batch',
        'num_workers': config.get('num_parallel_games', 8),
        'mcts_batch_size': config.get('mcts_batch_size', 32),
        'parallel_games': config.get('num_parallel_games', 8),
        'use_gpu_inference': config.get('use_gpu_inference', True),
        
        # GPU 配置
        'cuda': config.get('cuda', True) and torch.cuda.is_available(),
        'use_amp': config.get('use_amp', True),
        
        # 模型配置
        'num_filters': model_config['num_filters'],
        'num_res_blocks': model_config['num_blocks'],
        'num_heads': model_config['num_heads'],
        
        # 检查点
        'checkpoint': './results/checkpoints',
        'checkpoint_interval': config.get('checkpoint_interval', 10),
    }
    
    return args, config


def main():
    """主函数"""
    # 加载配置
    args, config = load_config_from_yaml()
    
    # 设置设备
    device = torch.device('cuda' if args['cuda'] and torch.cuda.is_available() else 'cpu')
    args['device'] = device
    
    # 简洁的设备信息
    if args['cuda'] and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        print("⚠ 使用CPU训练")
    
    # 初始化游戏和模型
    game = DotsAndBoxesGame(**{'num_rows': args['num_rows'], 'num_cols': args['num_cols']})
    
    nnet = DotsAndBoxesNet(
        game=game,
        num_blocks=args['num_res_blocks'],
        num_filters=args['num_filters'],
        num_heads=args['num_heads']
    ).to(device)
    
    total_params = sum(p.numel() for p in nnet.parameters())
    print(f"模型参数: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 打印简洁的配置信息
    print("\n" + "=" * 70)
    print("🚀 AlphaZero 训练")
    print("=" * 70)
    print(f"迭代: {args['num_iterations']} | 自我对弈: {args['num_episodes']}局/次 | MCTS: {args['num_simulations']}次")
    print(f"训练: Batch={args['batch_size']}, Epochs={args['epochs']}, LR={args['lr']}")
    print(f"模型: {args['num_filters']}d×{args['num_res_blocks']}块 | 参数: {total_params/1e6:.1f}M")
    print(f"并行: {args['num_workers']} workers | GPU批量: {args['mcts_batch_size']}")
    print("=" * 70)
    
    # 检查点目录
    os.makedirs(args['checkpoint'], exist_ok=True)
    
    # 初始化并行 Coach
    coach = ParallelCoach(game, nnet, args)
    
    # 开始训练
    print("\n")
    
    try:
        coach.learn()
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        coach.save_checkpoint(filename='interrupted.pth')
        print(f"✓ 模型已保存到 {args['checkpoint']}/interrupted.pth")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        coach.save_checkpoint(filename='error.pth')
        print(f"✓ 模型已保存到 {args['checkpoint']}/error.pth")
    
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"最佳模型保存在: {args['checkpoint']}/best_*.pth")
    print(f"最新模型保存在: {args['checkpoint']}/latest.pth")
    print("\n使用以下命令验证模型:")
    print(f"  python cli/play_ultimate.py --checkpoint {args['checkpoint']}/best_*.pth")
    print(f"  python cli/evaluate_model.py --checkpoint {args['checkpoint']}/best_*.pth")
    print("=" * 80)


if __name__ == '__main__':
    # 必须在主块中再次设置 (确保生效)
    multiprocessing.set_start_method('spawn', force=True)
    main()
