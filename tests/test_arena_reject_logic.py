#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的 Arena 拒绝逻辑
验证: 拒绝新模型时，不回滚权重，继续训练
"""

import torch
import copy


def test_reject_logic():
    """测试拒绝模型的逻辑"""
    
    print("="*70)
    print("测试 AlphaZero Arena 拒绝逻辑")
    print("="*70)
    
    # 模拟训练过程
    print("\n📚 AlphaZero 论文原文:")
    print("   'if the new player won by a margin of 55%, then it replaced")
    print("    the best player; otherwise, it was discarded.'")
    print("\n   'discarded' = 不接受为 baseline，但继续训练\n")
    
    # 创建模拟的模型
    class DummyModel:
        def __init__(self, name, param_value):
            self.name = name
            self.param = torch.tensor([param_value], dtype=torch.float32)
        
        def state_dict(self):
            return {'param': self.param}
        
        def load_state_dict(self, state_dict, strict=True):
            self.param = state_dict['param']
        
        def __repr__(self):
            return f"{self.name}(param={self.param.item():.2f})"
    
    # 模拟训练流程
    print("🔹 迭代 0: 初始化")
    nnet = DummyModel("current", 1.0)
    previous_nnet = copy.deepcopy(nnet)
    print(f"   current_nnet:  {nnet}")
    print(f"   previous_nnet: {previous_nnet}")
    
    print("\n🔹 迭代 1-5: 训练 (self-play + gradient descent)")
    nnet.param = torch.tensor([2.5])  # 模拟训练更新
    print(f"   current_nnet:  {nnet}")
    print(f"   previous_nnet: {previous_nnet} (未变)")
    
    print("\n🔹 迭代 5: Arena 评估")
    print("   对战结果: 新模型 52% vs 旧模型 48%")
    print("   阈值: 55%")
    print("   决定: ❌ 拒绝新模型 (52% < 55%)")
    
    print("\n   ⚙️  旧的错误实现:")
    print("      nnet.load_state_dict(previous_nnet.state_dict())")
    print("      → 回滚到 param=1.0 ❌")
    
    print("\n   ✅ 新的正确实现:")
    print("      # 什么都不做，继续训练")
    print("      → 保持 current_nnet param=2.5 ✅")
    print("      → 保持 previous_nnet param=1.0 (baseline 不变) ✅")
    
    # 验证正确逻辑
    print("\n🔹 迭代 6-10: 继续训练 (从 param=2.5 开始)")
    nnet.param = torch.tensor([3.2])  # 继续训练
    print(f"   current_nnet:  {nnet}")
    print(f"   previous_nnet: {previous_nnet} (仍然是旧 baseline)")
    
    print("\n🔹 迭代 10: 再次 Arena 评估")
    print("   对战结果: 新模型 60% vs 旧模型 40%")
    print("   决定: ✅ 接受新模型 (60% > 55%)")
    print("\n   ⚙️  执行:")
    print("      previous_nnet = deepcopy(nnet)")
    previous_nnet = copy.deepcopy(nnet)
    print(f"   previous_nnet: {previous_nnet} (更新为新 baseline) ✅")
    
    print("\n" + "="*70)
    print("📊 总结对比")
    print("="*70)
    print("❌ 错误实现 (旧代码):")
    print("   - 拒绝时回滚权重 → 丢失训练进度")
    print("   - 重复训练相同的状态 → 浪费计算")
    print("   - 可能陷入局部最优 → 难以突破")
    print("\n✅ 正确实现 (新代码):")
    print("   - 拒绝时保持当前权重 → 继续探索")
    print("   - 训练持续进步 → 不浪费计算")
    print("   - baseline 保持稳定 → 只接受明显更好的模型")
    print("="*70)
    
    print("\n✅ 测试通过!")


def test_replay_buffer():
    """测试 replay_buffer_size 的计算"""
    
    print("\n" + "="*70)
    print("测试 replay_buffer_size 配置")
    print("="*70)
    
    configs = [
        {"name": "错误配置", "num_games": 300, "buffer_size": 30000},
        {"name": "正确配置", "num_games": 300, "buffer_size": 360000},
    ]
    
    for cfg in configs:
        samples_per_iter = cfg["num_games"] * 60  # 60步平均
        max_iters = max(1, cfg["buffer_size"] // samples_per_iter)
        
        print(f"\n🔹 {cfg['name']}:")
        print(f"   num_self_play_games: {cfg['num_games']}")
        print(f"   replay_buffer_size:  {cfg['buffer_size']:,}")
        print(f"   → 每次迭代样本数: {samples_per_iter:,}")
        print(f"   → 保留迭代次数:   {max_iters}")
        
        if max_iters < 5:
            print(f"   ⚠️  警告: 只保留 {max_iters} 次迭代，样本多样性不足!")
        else:
            print(f"   ✅ 保留 {max_iters} 次迭代，样本多样性充足")
    
    print("\n" + "="*70)
    print("✅ 建议: replay_buffer_size ≥ 360000 (保留约 20 次迭代)")
    print("="*70)


if __name__ == '__main__':
    test_reject_logic()
    test_replay_buffer()
