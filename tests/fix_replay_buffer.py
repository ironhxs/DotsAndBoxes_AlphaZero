#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复并验证 replay_buffer_size 配置
"""

import yaml

print("="*80)
print("🔧 修复 replay_buffer_size 配置问题")
print("="*80)

# 问题诊断
print("\n📋 问题诊断:")
print("-"*80)
print("""
你遇到的问题:
  设置: replay_buffer_size: 30000
  预期: maxlen=1 (保留1次迭代)
  实际: 输出显示 (2/20) - 说明 maxlen=20

可能原因:
  1. 配置没有传递到 base_coach.py (使用了默认值 360000)
  2. trainer/alphazero.yaml 中的 num_iters_for_train_examples_history=20 被使用
  3. 从checkpoint恢复时,deque已经是20了
""")

# 解决方案
print("\n✅ 解决方案:")
print("-"*80)
print("""
方案1: 确认配置生效 (推荐)
──────────────────────────────────
1. 删除旧的 checkpoint:
   rm -rf results/checkpoints/*.pth

2. 重新启动训练,查看第一行输出:
   ✓ 经验池配置: 保留最近 N 次迭代
   
3. 如果 N=1, 配置生效 ✅
   如果 N=20, 配置未生效,继续方案2

方案2: 修改 trainer 配置
──────────────────────────────────
编辑 config/trainer/alphazero.yaml:

# 删除或注释掉这一行:
# num_iters_for_train_examples_history: 20

或者改为:
num_iters_for_train_examples_history: 1

方案3: 使用正确的配置值
──────────────────────────────────
如果你想保留1次迭代:
  replay_buffer_size: 18000  (300 × 60 × 1)

如果你想保留2次迭代:
  replay_buffer_size: 36000  (300 × 60 × 2)

如果你想保留20次迭代:
  replay_buffer_size: 360000 (300 × 60 × 20)
""")

# 验证当前配置
print("\n🔍 当前配置验证:")
print("-"*80)

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

replay_buffer = config.get('replay_buffer_size')
num_games = config.get('num_self_play_games')
samples_per_iter = num_games * 60
max_iters = max(1, replay_buffer // samples_per_iter)

print(f"config.yaml:")
print(f"  replay_buffer_size:  {replay_buffer:,}")
print(f"  num_self_play_games: {num_games}")
print(f"  → 计算结果: maxlen={max_iters}")

with open('config/trainer/alphazero.yaml', 'r', encoding='utf-8') as f:
    trainer = yaml.safe_load(f)

num_iters_history = trainer.get('num_iters_for_train_examples_history')
print(f"\ntrainer/alphazero.yaml:")
print(f"  num_iters_for_train_examples_history: {num_iters_history}")

if num_iters_history and num_iters_history != max_iters:
    print(f"\n  ⚠️  冲突! trainer 配置 ({num_iters_history}) ≠ 计算结果 ({max_iters})")
    print(f"  建议: 删除 trainer 配置中的 num_iters_for_train_examples_history")

# 生成修复脚本
print("\n" + "="*80)
print("🛠️  快速修复脚本:")
print("="*80)
print("""
# 1. 删除旧checkpoint
rm -rf results/checkpoints/*.pth

# 2. 启动训练
python cli/train_parallel.py

# 3. 查看第一行输出,确认经验池配置
""")
