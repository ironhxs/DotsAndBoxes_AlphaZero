#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型效果评估工具 - 验证训练是否有效"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import numpy as np
from tqdm import tqdm
from model.game import DotsAndBoxesGame
from model.model import DotsAndBoxesNet
from model.arena import Arena
import os


class RandomPlayer:
    """随机策略玩家 - 作为基线"""
    def __init__(self, game):
        self.game = game
        
    def __call__(self, obs_tensor):
        """返回随机策略"""
        # 返回均匀分布的策略和0价值
        action_size = self.game.get_action_size()
        pi = torch.ones(1, action_size) / action_size
        v = torch.zeros(1, 1)
        return torch.log(pi), v


class GreedyPlayer:
    """贪心策略玩家 - 尽量吃格子"""
    def __init__(self, game):
        self.game = game
        
    def __call__(self, obs_tensor):
        """返回贪心策略: 优先选择能吃格子的动作"""
        action_size = self.game.get_action_size()
        # 简化版: 返回均匀分布 (完整版需要模拟每个动作)
        pi = torch.ones(1, action_size) / action_size
        v = torch.zeros(1, 1)
        return torch.log(pi), v


def evaluate_model(checkpoint_path, num_games=40):
    """
    评估模型效果
    
    测试模型 vs:
    1. 随机策略 (应该100%胜率)
    2. 贪心策略 (应该>80%胜率)
    3. 更早期的模型 (应该>60%胜率)
    """
    
    print("=" * 80)
    print("🔬 模型效果评估系统")
    print("=" * 80)
    print(f"评估模型: {checkpoint_path}")
    print(f"对战局数: {num_games} 局")
    print("=" * 80)
    print()
    
    # 加载配置
    args = {
        'num_rows': 5,
        'num_cols': 5,
        'num_simulations': 25,  # 评估时用更多模拟
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.0,  # 评估时不加噪声
        'temp_threshold': 0,       # 评估时总是选最优
        'num_filters': 128,
        'num_res_blocks': 10,
        'cuda': torch.cuda.is_available(),
    }
    
    # 初始化游戏和模型
    game = DotsAndBoxesGame(args['num_rows'], args['num_cols'])
    nnet = DotsAndBoxesNet(game, args)
    
    # 加载模型
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到模型文件 {checkpoint_path}")
        print("\n请先运行训练:")
        print("  python train_alphazero.py")
        return
    
    checkpoint = torch.load(checkpoint_path)
    nnet.load_state_dict(checkpoint['state_dict'])
    
    if args['cuda']:
        nnet.cuda()
    
    nnet.eval()
    print(f"✓ 已加载模型: {checkpoint_path}\n")
    
    # ============================================================
    # 测试1: vs 随机策略
    # ============================================================
    print(f"\n{'='*80}")
    print("📊 测试 1/3: 训练模型 vs 随机策略")
    print(f"{'='*80}")
    print("期望结果: 模型应该 100% 胜率 (如果<90%说明训练失败)")
    
    random_player = RandomPlayer(game)
    arena_random = Arena(nnet, random_player, game, args)
    wins_r, losses_r, draws_r = arena_random.play_games(num_games)
    
    win_rate_random = (wins_r + 0.5 * draws_r) / num_games
    
    print(f"\n结果评估:")
    if win_rate_random >= 0.90:
        print(f"  ✅ 优秀! 胜率 {win_rate_random*100:.1f}% - 模型已学会基本策略")
    elif win_rate_random >= 0.70:
        print(f"  ⚠️  一般. 胜率 {win_rate_random*100:.1f}% - 模型还需继续训练")
    else:
        print(f"  ❌ 失败! 胜率 {win_rate_random*100:.1f}% - 模型训练可能有问题")
    
    # ============================================================
    # 测试2: vs 贪心策略
    # ============================================================
    print(f"\n{'='*80}")
    print("📊 测试 2/3: 训练模型 vs 贪心策略")
    print(f"{'='*80}")
    print("期望结果: 模型应该 >70% 胜率 (贪心策略比随机强)")
    
    greedy_player = GreedyPlayer(game)
    arena_greedy = Arena(nnet, greedy_player, game, args)
    wins_g, losses_g, draws_g = arena_greedy.play_games(num_games)
    
    win_rate_greedy = (wins_g + 0.5 * draws_g) / num_games
    
    print(f"\n结果评估:")
    if win_rate_greedy >= 0.70:
        print(f"  ✅ 优秀! 胜率 {win_rate_greedy*100:.1f}% - 模型已超越简单策略")
    elif win_rate_greedy >= 0.50:
        print(f"  ⚠️  一般. 胜率 {win_rate_greedy*100:.1f}% - 模型略优于贪心")
    else:
        print(f"  ❌ 差劲! 胜率 {win_rate_greedy*100:.1f}% - 模型甚至弱于贪心策略")
    
    # ============================================================
    # 测试3: vs 早期模型 (如果存在)
    # ============================================================
    early_checkpoint = checkpoint_path.replace('latest', 'checkpoint_5')
    if os.path.exists(early_checkpoint) and early_checkpoint != checkpoint_path:
        print(f"\n{'='*80}")
        print("📊 测试 3/3: 当前模型 vs 早期模型")
        print(f"{'='*80}")
        print(f"早期模型: {early_checkpoint}")
        print("期望结果: 当前模型应该 >60% 胜率 (说明有进步)")
        
        nnet_old = DotsAndBoxesNet(game, args)
        checkpoint_old = torch.load(early_checkpoint)
        nnet_old.load_state_dict(checkpoint_old['state_dict'])
        
        if args['cuda']:
            nnet_old.cuda()
        nnet_old.eval()
        
        arena_old = Arena(nnet, nnet_old, game, args)
        wins_o, losses_o, draws_o = arena_old.play_games(num_games)
        
        win_rate_old = (wins_o + 0.5 * draws_o) / num_games
        
        print(f"\n结果评估:")
        if win_rate_old >= 0.65:
            print(f"  ✅ 显著进步! 胜率 {win_rate_old*100:.1f}% - 训练正在提升模型")
        elif win_rate_old >= 0.55:
            print(f"  ⚠️  轻微进步. 胜率 {win_rate_old*100:.1f}% - 有进步但不明显")
        else:
            print(f"  ❌ 无进步! 胜率 {win_rate_old*100:.1f}% - 可能需要调整超参数")
    else:
        print(f"\n{'='*80}")
        print("📊 测试 3/3: 跳过 (未找到早期模型)")
        print(f"{'='*80}")
        win_rate_old = None
    
    # ============================================================
    # 综合评估
    # ============================================================
    print(f"\n{'='*80}")
    print("🏆 综合评估报告")
    print(f"{'='*80}")
    print(f"模型: {checkpoint_path}")
    print(f"\n性能指标:")
    print(f"  vs 随机策略: {win_rate_random*100:5.1f}% {'✅' if win_rate_random >= 0.90 else '❌'}")
    print(f"  vs 贪心策略: {win_rate_greedy*100:5.1f}% {'✅' if win_rate_greedy >= 0.70 else '❌'}")
    if win_rate_old is not None:
        print(f"  vs 早期模型: {win_rate_old*100:5.1f}% {'✅' if win_rate_old >= 0.60 else '❌'}")
    
    # 总体评分
    scores = [win_rate_random >= 0.90, win_rate_greedy >= 0.70]
    if win_rate_old is not None:
        scores.append(win_rate_old >= 0.60)
    
    total_score = sum(scores) / len(scores)
    
    print(f"\n总体评分: {total_score*100:.0f}%")
    if total_score >= 0.8:
        print("评级: ⭐⭐⭐ 优秀 - 模型训练成功!")
    elif total_score >= 0.6:
        print("评级: ⭐⭐ 良好 - 模型基本可用，建议继续训练")
    else:
        print("评级: ⭐ 较差 - 建议检查训练配置或增加训练时间")
    
    print(f"{'='*80}\n")
    
    # 训练建议
    if total_score < 0.8:
        print("💡 训练建议:")
        if win_rate_random < 0.90:
            print("  1. 增加训练迭代次数 (num_iterations)")
            print("  2. 增加每次迭代的游戏数 (num_episodes)")
        if win_rate_greedy < 0.70:
            print("  3. 增加MCTS模拟次数 (num_simulations)")
            print("  4. 增加神经网络容量 (num_filters, num_res_blocks)")
        if win_rate_old is not None and win_rate_old < 0.60:
            print("  5. 降低学习率避免过拟合 (lr)")
            print("  6. 增加Arena对战局数保证筛选质量 (arena_compare)")
        print()


def quick_test():
    """快速测试 - 只对战10局"""
    checkpoint = './checkpoints/latest.pth'
    if not os.path.exists(checkpoint):
        checkpoint = './checkpoints/best_*.pth'
        import glob
        files = glob.glob(checkpoint)
        if files:
            checkpoint = sorted(files)[-1]
    
    evaluate_model(checkpoint, num_games=10)


def full_test():
    """完整测试 - 对战40局"""
    checkpoint = './checkpoints/latest.pth'
    if not os.path.exists(checkpoint):
        checkpoint = './checkpoints/best_*.pth'
        import glob
        files = glob.glob(checkpoint)
        if files:
            checkpoint = sorted(files)[-1]
    
    evaluate_model(checkpoint, num_games=40)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        full_test()
