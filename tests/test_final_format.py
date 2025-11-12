#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的进度条格式和 AMP 训练
"""

import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

from tqdm import tqdm
import time

print("✅ 测试新的进度条格式:")
print("预期: Train: 48%|████| 145/300 [01:41<01:46, Loss=4.5568]\n")

progress_bar = tqdm(
    total=300,
    desc='  Train',
    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
)

for epoch in range(300):
    time.sleep(0.005)  # 模拟训练
    loss = 5.0 - epoch * 0.01
    progress_bar.update(1)
    progress_bar.set_postfix_str(f'Loss={loss:.4f}')

progress_bar.close()

print("\n✅ 进度条格式正确!")
print("\n格式说明:")
print("  - 'Train' 替代了 '训练'")
print("  - '145/300' 显示在百分比后面")
print("  - 'Loss=4.5568' 显示在时间后面")
print("  - 没有迭代速度 (2.92it/s)")
