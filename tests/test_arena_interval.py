#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Arena 间隔逻辑"""

import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

from cli.train_parallel import load_config_from_yaml

args, config = load_config_from_yaml()

print("Arena 间隔配置检查:")
print(f"  config.yaml 中的 eval_interval: {config.get('eval_interval', 'NOT SET')}")
print(f"  args 中的 arena_interval: {args.get('arena_interval', 'NOT SET')}")

print("\n模拟迭代逻辑:")
arena_interval = args.get('arena_interval', 1)
for iteration in range(1, 16):
    should_arena = (iteration % arena_interval == 0)
    status = "执行 Arena" if should_arena else "跳过 Arena"
    print(f"  迭代 {iteration:2d}: {status}")

print(f"\n结论: 每 {arena_interval} 次迭代执行一次 Arena 对战")
print(f"前 15 次迭代中，Arena 执行 {sum(1 for i in range(1, 16) if i % arena_interval == 0)} 次")
