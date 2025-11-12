#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Arena 配置"""

import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

from cli.train_parallel import load_config_from_yaml

args, config = load_config_from_yaml()

print("Arena 配置检查:")
print(f"  arena_mode: {args.get('arena_mode', 'NOT SET')}")
print(f"  arena_compare: {args.get('arena_compare', 'NOT SET')}")
print(f"  arena_num_workers: {args.get('arena_num_workers', 'NOT SET')}")
print(f"  arena_mcts_simulations: {args.get('arena_mcts_simulations', 'NOT SET')}")
print(f"  cuda: {args.get('cuda', 'NOT SET')}")
print(f"  update_threshold: {args.get('update_threshold', 'NOT SET')}")

# 模拟判断逻辑
arena_mode = args.get('arena_mode', 'serial')
cuda_enabled = args.get('cuda', False)

print(f"\n判断结果:")
print(f"  arena_mode == 'gpu_parallel': {arena_mode == 'gpu_parallel'}")
print(f"  cuda == True: {cuda_enabled}")
print(f"  会使用 GPU 并行: {arena_mode == 'gpu_parallel' and cuda_enabled}")
