#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极端测试: replay_buffer_size 各种边界情况
测试目的: 验证经验池在极端配置下的行为
"""

from collections import deque


def test_extreme_cases():
    """测试各种极端配置"""
    
    print("="*70)
    print("极端测试: replay_buffer_size 边界情况")
    print("="*70)
    
    test_cases = [
        ("超小配置", 1000, 300),      # 0.05 次迭代 → 0 (会被修正为 1)
        ("极小配置", 18000, 300),     # 1 次迭代
        ("你的测试配置", 36000, 300),  # 2 次迭代
        ("偏小配置", 90000, 300),     # 5 次迭代
        ("正常配置", 360000, 300),    # 20 次迭代
        ("大型配置", 900000, 300),    # 50 次迭代
        ("超大配置", 1800000, 300),   # 100 次迭代
    ]
    
    for name, buffer_size, num_games in test_cases:
        samples_per_iter = num_games * 60
        max_iters = max(1, buffer_size // samples_per_iter)  # ⚠️ max(1, ...) 防止为 0
        
        print(f"\n{'='*70}")
        print(f"🔹 {name}")
        print(f"{'='*70}")
        print(f"配置: replay_buffer_size = {buffer_size:,}")
        print(f"计算: max_iters = max(1, {buffer_size:,} // {samples_per_iter:,}) = {max_iters}")
        print(f"结果: 保留最近 {max_iters} 次迭代")
        
        # 评估
        if max_iters == 1:
            print("⚠️  警告: 只保留 1 次迭代!")
            print("   - 没有历史数据")
            print("   - 每次训练只用最新的数据")
            print("   - 等同于没有经验回放")
        elif max_iters < 5:
            print("❌ 极差: 只保留 {} 次迭代".format(max_iters))
            print("   - 样本多样性严重不足")
            print("   - 容易过拟合到最近策略")
            print("   - 训练非常不稳定")
        elif max_iters < 10:
            print("⚠️  偏小: 保留 {} 次迭代".format(max_iters))
            print("   - 样本多样性不足")
            print("   - 可能影响训练稳定性")
        elif max_iters <= 30:
            print("✅ 合适: 保留 {} 次迭代".format(max_iters))
            print("   - 样本多样性充足")
            print("   - 训练稳定")
            print("   - AlphaZero 推荐范围")
        else:
            print("✅ 很大: 保留 {} 次迭代".format(max_iters))
            print("   - 样本多样性非常好")
            print("   - 适合长期训练")
            print("   - 需要更多内存")
        
        # 模拟内存使用
        # 假设每个样本 200 bytes (state + policy + value)
        bytes_per_sample = 200
        total_bytes = max_iters * samples_per_iter * bytes_per_sample
        mb = total_bytes / (1024 * 1024)
        gb = mb / 1024
        
        print(f"预估内存: {mb:.1f} MB ({gb:.2f} GB)")


def test_overflow_behavior():
    """测试溢出行为"""
    
    print("\n" + "="*70)
    print("测试: 经验池溢出行为")
    print("="*70)
    
    # 模拟 maxlen=3 的情况
    history = deque(maxlen=3)
    
    print("\n配置: deque(maxlen=3)")
    print("\n添加过程:\n")
    
    for i in range(1, 8):
        history.append(f"iter{i}")
        content = list(history)
        
        print(f"添加 iter{i}:")
        print(f"  → deque 内容: {content}")
        print(f"  → 长度: {len(history)}")
        
        if len(history) >= 3:
            print(f"  → 状态: ✅ 已满 (删除最老的)")
        else:
            print(f"  → 状态: ⬆️ 增长中")
        print()
    
    print("="*70)
    print("结论:")
    print("="*70)
    print("✅ deque 自动管理大小:")
    print("   - 达到 maxlen 后,每次 append 自动删除最左边(最老)的元素")
    print("   - 永远不会超过 maxlen")
    print("   - 不需要手动删除")
    print("   - 不会抛出异常")
    print("\n✅ 对于训练:")
    print("   - 迭代 1-3: 经验池增长")
    print("   - 迭代 4+: 自动删除最老的迭代")
    print("   - 始终保持最近 N 次迭代的数据")
    print("="*70)


def test_memory_comparison():
    """内存使用对比"""
    
    print("\n" + "="*70)
    print("内存使用对比 (假设每样本 200 bytes)")
    print("="*70)
    
    configs = [
        (1, 18000),    # 极小
        (2, 36000),    # 你的测试
        (5, 90000),    # 偏小
        (20, 360000),  # 正常
        (50, 900000),  # 大型
    ]
    
    print(f"\n{'保留迭代':<12} {'总样本数':<12} {'内存 (MB)':<12} {'内存 (GB)':<12}")
    print("-" * 70)
    
    for max_iters, total_samples in configs:
        bytes_used = total_samples * 200
        mb = bytes_used / (1024 * 1024)
        gb = mb / 1024
        
        print(f"{max_iters:<12} {total_samples:>11,} {mb:>11.1f} {gb:>11.2f}")
    
    print("\n说明:")
    print("  - 20 次迭代 (360,000 样本) 约需 69 MB 内存 ✅")
    print("  - 这对现代 GPU 来说微不足道")
    print("  - 瓶颈通常是训练速度,不是内存")


if __name__ == '__main__':
    test_extreme_cases()
    test_overflow_behavior()
    test_memory_comparison()
