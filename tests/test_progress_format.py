#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试进度条格式和 Arena GPU 并行
"""

import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

from tqdm import tqdm
import time

# 测试进度条格式
print("测试新的进度条格式:")
print("预期格式: 训练 [145/300]: 48%|███| [01:41<01:46, Loss=4.5568]\n")

progress_bar = tqdm(
    total=300,
    desc='  训练',
    bar_format='{desc} [{n_fmt}/{total_fmt}]: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {postfix}]'
)

for epoch in range(300):
    time.sleep(0.01)  # 模拟训练
    loss = 5.0 - epoch * 0.01  # 模拟损失下降
    progress_bar.update(1)
    progress_bar.set_postfix_str(f'Loss={loss:.4f}')

progress_bar.close()

print("\n✅ 进度条格式测试完成!")
print("\n现在测试 Arena GPU 并行...")

# 加载配置
from cli.train_parallel import load_config_from_yaml
args, config = load_config_from_yaml()

print(f"\nArena 配置:")
print(f"  模式: {args.get('arena_mode', 'serial')}")
print(f"  对战局数: {args.get('arena_compare', 40)}")
print(f"  并行进程数: {args.get('arena_num_workers', 1)}")
print(f"  MCTS 模拟次数: {args.get('arena_mcts_simulations', 100)}")

print("\n✅ Arena 配置加载成功!")
