#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 replay_buffer_size 配置问题
"""

import sys
import yaml

print("="*80)
print("🔍 诊断 replay_buffer_size 配置")
print("="*80)

# 1. 读取配置文件
print("\n1️⃣ 读取 config.yaml:")
print("-"*80)

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

replay_buffer = config.get('replay_buffer_size')
num_games = config.get('num_self_play_games')

print(f"   num_self_play_games: {num_games}")
print(f"   replay_buffer_size:  {replay_buffer}")

# 2. 计算预期值
print("\n2️⃣ 计算预期值:")
print("-"*80)

samples_per_iter = num_games * 60
max_iters = max(1, replay_buffer // samples_per_iter)

print(f"   samples_per_iter = {num_games} × 60 = {samples_per_iter}")
print(f"   max_iters = max(1, {replay_buffer} ÷ {samples_per_iter}) = {max_iters}")

# 3. 读取 trainer 配置
print("\n3️⃣ 检查 trainer 配置:")
print("-"*80)

with open('config/trainer/alphazero.yaml', 'r', encoding='utf-8') as f:
    trainer_config = yaml.safe_load(f)

num_iters_history = trainer_config.get('num_iters_for_train_examples_history')
print(f"   num_iters_for_train_examples_history: {num_iters_history}")

if num_iters_history:
    print(f"   ⚠️  警告: trainer 配置中有硬编码的 {num_iters_history}!")
    print(f"   这可能会覆盖 replay_buffer_size 的计算")

# 4. 模拟实际加载
print("\n4️⃣ 模拟 train_parallel.py 加载逻辑:")
print("-"*80)

args = {
    'replay_buffer_size': config.get('replay_buffer_size', 360000),
    'num_self_play_games': config.get('num_self_play_games', 300),
    'num_iters_for_train_examples_history': trainer_config.get('num_iters_for_train_examples_history', 20),
}

print(f"   args['replay_buffer_size'] = {args['replay_buffer_size']}")
print(f"   args['num_self_play_games'] = {args['num_self_play_games']}")
print(f"   args['num_iters_for_train_examples_history'] = {args['num_iters_for_train_examples_history']}")

# 5. 模拟 base_coach.py 计算
print("\n5️⃣ 模拟 base_coach.py 计算:")
print("-"*80)

samples = args['num_self_play_games'] * 60
maxlen = max(1, args['replay_buffer_size'] // samples)

print(f"   samples_per_iter = {args['num_self_play_games']} × 60 = {samples}")
print(f"   max_iters = max(1, {args['replay_buffer_size']} ÷ {samples}) = {maxlen}")
print(f"   deque(maxlen={maxlen})")

# 6. 结论
print("\n" + "="*80)
print("🎯 结论:")
print("="*80)

if maxlen == max_iters:
    print(f"✅ 配置正确: 应该保留 {maxlen} 次迭代")
else:
    print(f"❌ 配置冲突!")
    print(f"   从 replay_buffer_size 计算: {max_iters} 次迭代")
    print(f"   从 trainer 配置: {num_iters_history} 次迭代")

print(f"\n如果训练输出显示 (N/20), 说明:")
print(f"   1. 配置没有正确传递到 base_coach.py")
print(f"   2. 或者 base_coach.py 使用了默认值 360000")
print(f"   3. 或者 trainer 配置覆盖了 replay_buffer_size")

print("\n💡 建议:")
print(f"   在训练开始时查看这一行:")
print(f"   '✓ 经验池配置: 保留最近 N 次迭代'")
print(f"   如果 N={maxlen}, 说明配置生效")
print(f"   如果 N=20, 说明使用了默认值,配置未生效")
print("="*80)
