#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Arena GPU 并行修复"""

import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer

print("测试 1: 模型动态参数初始化")
game = DotsAndBoxesGame()
model = DotsAndBoxesTransformer(game, 256, 12, 8)

print(f"  初始状态: pos_embedding = {model.pos_embedding}")
print(f"  初始状态: policy_fc1 = {model.policy_fc1}")
print(f"  初始状态: value_fc1 = {model.value_fc1}")

# 做一次前向传播
dummy_input = torch.randn(1, 9, 6, 6)
output_policy, output_value = model(dummy_input)

print(f"  前向后: pos_embedding shape = {model.pos_embedding.shape if model.pos_embedding is not None else None}")
print(f"  前向后: policy_fc1 = {model.policy_fc1}")
print(f"  前向后: value_fc1 = {model.value_fc1}")

print("\n测试 2: 状态字典复制")
state_dict = model.state_dict()
print(f"  状态字典包含 {len(state_dict)} 个参数")
print(f"  包含 pos_embedding: {'pos_embedding' in state_dict}")
print(f"  包含 policy_fc1.weight: {'policy_fc1.weight' in state_dict}")
print(f"  包含 value_fc1.weight: {'value_fc1.weight' in state_dict}")

print("\n测试 3: 加载状态字典（strict=False）")
model2 = DotsAndBoxesTransformer(game, 256, 12, 8)
try:
    model2.load_state_dict(state_dict, strict=False)
    print("  ✅ 加载成功（strict=False）")
except Exception as e:
    print(f"  ❌ 加载失败: {e}")

print("\n测试 4: 检查state_dict判断逻辑")
is_transformer = 'transformer_blocks.0.norm1.weight' in state_dict
print(f"  判断为 Transformer: {is_transformer}")

print("\n✅ 所有测试完成!")
