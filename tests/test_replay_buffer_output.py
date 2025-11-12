#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 replay_buffer 输出信息
验证经验池增长过程的显示
"""

from collections import deque


def simulate_training_output():
    """模拟训练输出，展示经验池状态"""
    
    print("="*70)
    print("模拟训练输出 - 经验池状态显示")
    print("="*70)
    
    # 配置
    num_games = 300
    replay_buffer_size = 360000
    samples_per_iter = num_games * 60  # 18000
    max_iters = max(1, replay_buffer_size // samples_per_iter)  # 20
    
    print(f"\n📋 配置:")
    print(f"   num_self_play_games: {num_games}")
    print(f"   replay_buffer_size: {replay_buffer_size:,}")
    print(f"   → 保留最近 {max_iters} 次迭代\n")
    
    # 模拟训练
    history = deque(maxlen=max_iters)
    
    print("🔹 训练过程输出:\n")
    
    # 显示关键迭代
    show_iterations = [1, 2, 5, 10, 15, 19, 20, 21, 22, 50]
    
    for i in range(1, 51):
        # 添加新样本
        history.append([f'sample_{i}'] * samples_per_iter)
        
        # 统计
        current_iters = len(history)
        total_samples = sum(len(examples) for examples in history)
        is_full = current_iters >= max_iters
        
        if i in show_iterations:
            # 模拟训练输出格式
            print(f"======================================================================")
            print(f"迭代 {i}/1000")
            print(f"======================================================================")
            
            status = "✅ 已满" if is_full else f"⬆️ 增长中 ({current_iters}/{max_iters})"
            print(f"[1/3] 自我对弈...")
            print(f"  ✓ 训练集: {total_samples:,} 样本 (保留 {current_iters} 次迭代) {status}")
            print()
    
    print("="*70)
    print("📊 说明:")
    print("="*70)
    print("✅ 前 20 次迭代: 经验池逐渐增长 (这是正常的!)")
    print("   - 样本数从 18,000 增长到 360,000")
    print("   - 状态显示 '⬆️ 增长中 (N/20)'")
    print()
    print("✅ 第 20 次迭代后: 经验池保持稳定")
    print("   - 样本数稳定在 360,000")
    print("   - 状态显示 '✅ 已满'")
    print("   - deque 自动丢弃最老的数据")
    print()
    print("✅ 这种行为完全符合 AlphaZero 的设计!")
    print("   - 保持样本多样性 (20 次迭代)")
    print("   - 控制内存使用 (不会无限增长)")
    print("="*70)


def compare_configs():
    """对比不同配置的效果"""
    
    print("\n" + "="*70)
    print("配置对比")
    print("="*70)
    
    configs = [
        ("原配置 (错误)", 30000, 300),
        ("新配置 (正确)", 360000, 300),
        ("大型配置", 900000, 300),
    ]
    
    for name, buffer_size, num_games in configs:
        samples_per_iter = num_games * 60
        max_iters = max(1, buffer_size // samples_per_iter)
        
        print(f"\n🔹 {name}:")
        print(f"   replay_buffer_size: {buffer_size:,}")
        print(f"   → 保留迭代数: {max_iters}")
        
        if max_iters < 5:
            print(f"   ❌ 太小! 样本多样性不足")
        elif max_iters < 10:
            print(f"   ⚠️  偏小，建议增加到 20+")
        elif max_iters <= 30:
            print(f"   ✅ 合适! (推荐 20-30)")
        else:
            print(f"   ✅ 很大，适合长期训练")
        
        # 模拟增长
        print(f"   增长过程:")
        history = deque(maxlen=max_iters)
        for i in [1, 5, 10, max_iters, max_iters + 5]:
            if i <= max_iters + 5:
                history.append([0] * samples_per_iter)
                total = sum(len(x) for x in history)
                current = len(history)
                status = "✅" if current >= max_iters else "⬆️"
                print(f"      迭代 {i:2d}: {total:>7,} 样本 {status}")


if __name__ == '__main__':
    simulate_training_output()
    compare_configs()
