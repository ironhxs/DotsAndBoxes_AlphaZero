#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断训练问题 - 分析Loss无法下降的原因
"""
import sys
sys.path.append('/root/DotsAndBoxes_AlphaZero')

import torch
import numpy as np
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
import yaml

def diagnose_training_issues():
    """诊断训练中的潜在问题"""
    print("=" * 70)
    print("训练诊断工具")
    print("=" * 70)
    
    # 加载配置 - 正确加载所有子配置
    import os
    config_dir = '/root/DotsAndBoxes_AlphaZero/config'
    
    with open(os.path.join(config_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, 'game/dots_and_boxes.yaml'), 'r') as f:
        game_config = yaml.safe_load(f)
    
    with open(os.path.join(config_dir, 'model/transformer.yaml'), 'r') as f:
        model_config = yaml.safe_load(f)
    
    # 合并配置
    config.update(game_config)
    config.update(model_config)
    
    # 添加必需的参数
    config['lr'] = config.get('learning_rate', 0.001)
    config['epochs'] = config.get('train_epochs', 300)
    
    game = DotsAndBoxesGame()
    
    # 正确创建模型 - 传递单独的参数而不是config字典
    num_filters = model_config.get('num_filters', 256)
    num_blocks = model_config.get('num_blocks', 12)
    num_heads = model_config.get('num_heads', 8)
    
    nnet = DotsAndBoxesTransformer(
        game=game,
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_heads=num_heads
    )
    
    # 尝试加载最新模型
    checkpoint_path = '/root/DotsAndBoxes_AlphaZero/results/checkpoints/latest.pth'
    model_loaded = False
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        nnet.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"✓ 已加载模型: {checkpoint_path}")
        print(f"  (部分权重不匹配,已忽略)\n")
        model_loaded = True
    except Exception as e:
        print(f"⚠️  无法加载模型,使用随机初始化权重")
        print(f"   原因: {str(e)[:100]}\n")
        model_loaded = False
    
    # ========== 1. 检查模型梯度 ==========
    print("1️⃣  检查模型梯度流")
    print("-" * 70)
    
    nnet.eval()
    if torch.cuda.is_available():
        nnet.cuda()
    
    # 生成随机输入
    dummy_input = torch.randn(8, 9, 6, 6)
    dummy_policy = torch.randn(8, 60)
    dummy_value = torch.randn(8)
    
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        dummy_policy = dummy_policy.cuda()
        dummy_value = dummy_value.cuda()
    
    nnet.train()
    out_pi, out_v = nnet(dummy_input)
    
    # 计算损失
    l_pi = -torch.sum(dummy_policy * out_pi) / dummy_policy.size(0)
    l_v = torch.sum((dummy_value - out_v.view(-1)) ** 2) / dummy_value.size(0)
    total_loss = l_pi + l_v
    
    total_loss.backward()
    
    # 统计梯度
    grad_stats = []
    for name, param in nnet.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats.append({
                'name': name,
                'shape': list(param.shape),
                'grad_norm': grad_norm,
                'grad_mean': param.grad.mean().item(),
                'grad_std': param.grad.std().item()
            })
    
    # 找出梯度异常的层
    print("梯度统计:")
    zero_grad_layers = []
    large_grad_layers = []
    
    for stat in grad_stats[:10]:  # 只显示前10层
        print(f"  {stat['name'][:50]:50s} | Norm: {stat['grad_norm']:.6f} | Mean: {stat['grad_mean']:.6f}")
        
        if stat['grad_norm'] < 1e-7:
            zero_grad_layers.append(stat['name'])
        elif stat['grad_norm'] > 100:
            large_grad_layers.append(stat['name'])
    
    if zero_grad_layers:
        print(f"\n⚠️  发现 {len(zero_grad_layers)} 层梯度接近0 (梯度消失)")
        for layer in zero_grad_layers[:5]:
            print(f"    - {layer}")
    
    if large_grad_layers:
        print(f"\n⚠️  发现 {len(large_grad_layers)} 层梯度过大 (梯度爆炸)")
        for layer in large_grad_layers[:5]:
            print(f"    - {layer}")
    
    if not zero_grad_layers and not large_grad_layers:
        print("✓ 梯度流正常")
    
    # ========== 2. 检查学习率 ==========
    print("\n2️⃣  检查学习率配置")
    print("-" * 70)
    
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    
    print(f"当前学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    
    if lr > 0.01:
        print("⚠️  学习率可能过大,建议<0.01")
    elif lr < 0.0001:
        print("⚠️  学习率可能过小,训练会很慢")
    else:
        print("✓ 学习率在合理范围")
    
    # ========== 3. 检查数据分布 ==========
    print("\n3️⃣  检查训练数据特征")
    print("-" * 70)
    
    print(f"策略输出范围: [{out_pi.min().item():.4f}, {out_pi.max().item():.4f}]")
    print(f"价值输出范围: [{out_v.min().item():.4f}, {out_v.max().item():.4f}]")
    print(f"策略损失: {l_pi.item():.4f}")
    print(f"价值损失: {l_v.item():.4f}")
    
    # ========== 4. 推荐改进措施 ==========
    print("\n4️⃣  改进建议")
    print("-" * 70)
    
    recommendations = []
    
    if l_pi.item() > 1.0:
        recommendations.append("策略损失较高,可能的原因:")
        recommendations.append("  - 增加MCTS模拟次数(num_simulations)")
        recommendations.append("  - 减小学习率,让模型更稳定地学习")
        recommendations.append("  - 增大replay_buffer保留更多高质量数据")
    
    if l_v.item() < 0.1:
        recommendations.append("价值损失很低,模型可能:")
        recommendations.append("  - 过度拟合价值函数")
        recommendations.append("  - 策略学习不够(策略损失>>价值损失)")
    
    if config.get('batch_size', 1024) > 4096:
        recommendations.append("Batch size过大可能导致:")
        recommendations.append("  - 泛化能力下降")
        recommendations.append("  - 建议降低到2048-4096")
    
    if not recommendations:
        recommendations.append("✓ 未发现明显问题")
        recommendations.append("Loss停滞可能是正常的收敛现象")
        recommendations.append("建议:")
        recommendations.append("  1. 继续训练,观察Arena胜率")
        recommendations.append("  2. 降低学习率到0.0005")
        recommendations.append("  3. 增加exploration(温度参数)")
    
    for rec in recommendations:
        print(rec)
    
    # ========== 5. 快速修复配置 ==========
    print("\n5️⃣  推荐配置调整")
    print("-" * 70)
    
    print("修改 config/config.yaml:")
    print("""
# 如果Loss停滞:
learning_rate: 0.0005        # 从0.001降低到0.0005
batch_size: 4096             # 从7200降低到4096
num_simulations: 1200        # 从800增加到1200 (提高MCTS质量)
replay_buffer_size: 100000   # 从60000增加到100000 (更多样化数据)

# 如果想加速收敛:
patience: 15                 # 从10增加到15 (给更多机会)
min_delta: 0.00005           # 从0.0001降低 (更敏感)
    """)

if __name__ == '__main__':
    diagnose_training_issues()
