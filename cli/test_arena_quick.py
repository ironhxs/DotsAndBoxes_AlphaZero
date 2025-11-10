#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""极速测试 - 减少 MCTS 次数，快速验证 Arena GPU"""

import warnings
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

import torch
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.arena_gpu import ArenaGPU


def quick_test():
    """极速测试 - 少量 MCTS，快速验证"""
    
    print("="*70)
    print("⚡ Arena GPU 极速验证测试")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # 小棋盘
    print("\n创建 3x3 小棋盘...")
    game = DotsAndBoxesGame(num_rows=3, num_cols=3)
    
    # 小模型
    print("创建小模型...")
    model1 = DotsAndBoxesTransformer(game, num_filters=64, num_blocks=2, num_heads=4)
    model2 = DotsAndBoxesTransformer(game, num_filters=64, num_blocks=2, num_heads=4)
    
    # 极少 MCTS
    print("\n配置 Arena（低 MCTS 快速测试）...")
    args = {
        'num_simulations': 10,           # 极少 MCTS
        'arena_mcts_simulations': 10,    # 极少 MCTS
        'cpuct': 1.0,
        'arena_num_workers': 4,
        'cuda': True,
    }
    
    print(f"   MCTS: 仅 {args['arena_mcts_simulations']} 次（极速模式）")
    print(f"   线程: {args['arena_num_workers']}")
    
    print("\n创建 ArenaGPU...")
    arena = ArenaGPU(model1, model2, game, args)
    
    print("\n开始 4 局快速对战...")
    print("="*70)
    
    import time
    start = time.time()
    
    try:
        new_wins, old_wins, draws = arena.play_games(num_games=4)
        elapsed = time.time() - start
        
        print("\n" + "="*70)
        print("✅ 测试成功！")
        print("="*70)
        print(f"⏱️  耗时: {elapsed:.2f} 秒")
        print(f"📊 结果: {new_wins}胜 {draws}平 {old_wins}负")
        print("="*70)
        print("🎉 Arena GPU 模式工作正常！")
        print("\n💡 性能分析:")
        print(f"   - 4局对战用时: {elapsed:.2f}秒")
        print(f"   - 平均每局: {elapsed/4:.2f}秒")
        print(f"   - MCTS次数: 仅10次（实际训练200次会慢20倍）")
        print("\n⚠️  为什么 GPU 利用率低？")
        print("   - MCTS 是 CPU 密集型（树搜索）")
        print("   - GPU 只在推理时使用（占总时间 5-10%）")
        print("   - 每次只推理 1 个样本（没有批量化）")
        print("\n🚀 如何提升 GPU 利用率？")
        print("   - 使用真正的批量推理服务器（batch_inference_server.py）")
        print("   - 收集多个推理请求，批量处理")
        print("   - 但实现复杂，目前方案已经够用")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
