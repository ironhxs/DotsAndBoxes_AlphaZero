#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 测试点格棋 AlphaZero 项目

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """测试基础导入"""
    print("\n" + "=" * 60)
    print("点格棋 AlphaZero 项目测试")
    print("=" * 60)
    
    # 测试 PyTorch
    print("\n步骤 1: 检查 PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch 已安装: {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        print(f"❌ PyTorch 未安装: {e}")
        return False
    
    # 测试 OpenSpiel
    print("\n步骤 2: 检查 OpenSpiel...")
    try:
        import pyspiel
        print("✓ OpenSpiel 已安装")
        # 尝试加载点格棋
        game = pyspiel.load_game("dots_and_boxes(num_rows=5,num_cols=5)")
        print(f"  - 成功加载点格棋游戏")
        print(f"  - 动作空间大小: {game.num_distinct_actions()}")
    except ImportError:
        print("⚠️  OpenSpiel 未安装，正在尝试安装...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "install", "open_spiel"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ OpenSpiel 安装成功")
            import pyspiel
        else:
            print(f"❌ OpenSpiel 安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ OpenSpiel 加载失败: {e}")
        return False
    
    return True

def test_game_module():
    """测试游戏模块"""
    print("\n步骤 3: 测试游戏模块...")
    try:
        from game import DotsAndBoxesGame
        
        game = DotsAndBoxesGame(5, 5)
        state = game.get_initial_state()
        
        print("✓ 游戏模块正常")
        print(f"  - 格子数: {game.num_rows}x{game.num_cols}")
        print(f"  - 点阵: {game.num_rows+1}x{game.num_cols+1}")
        print(f"  - 动作数: {game.get_action_size()}")
        print(f"  - 总边数: {game.get_action_size()} (水平 + 垂直)")
        
        # 测试观察
        obs = game.get_observation(state)
        print(f"  - 观察张量形状: {obs.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 游戏模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_module():
    """测试模型模块"""
    print("\n步骤 4: 测试模型模块...")
    try:
        import torch
        from game import DotsAndBoxesGame
        from model import DotsAndBoxesNet
        
        game = DotsAndBoxesGame(5, 5)
        model = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✓ 模型模块正常")
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")
        
        # 测试前向传播
        state = game.get_initial_state()
        obs = game.get_observation(state)
        
        if torch.cuda.is_available():
            model = model.cuda()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).cuda()
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            pi, v = model(obs_tensor)
        
        print(f"  - 策略输出形状: {pi.shape}")
        print(f"  - 价值输出形状: {v.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 模型模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcts_module():
    """测试 MCTS 模块"""
    print("\n步骤 5: 测试 MCTS 模块...")
    try:
        import torch
        from game import DotsAndBoxesGame
        from model import DotsAndBoxesNet
        from mcts import MCTS
        
        game = DotsAndBoxesGame(5, 5)
        model = DotsAndBoxesNet(game, num_filters=32, num_res_blocks=2)
        
        if torch.cuda.is_available():
            model.cuda()
        
        args = {
            'num_simulations': 10,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'temp_threshold': 15,
        }
        
        mcts = MCTS(game, model, args)
        state = game.get_initial_state()
        
        print("✓ MCTS 模块正常")
        
        # 测试搜索
        print("  - 执行 MCTS 搜索测试...")
        pi = mcts.get_action_prob(state, temp=1)
        print(f"  - 策略分布形状: {len(pi)}")
        print(f"  - 策略和: {sum(pi):.4f}")
        
        return True
    except Exception as e:
        print(f"❌ MCTS 模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_self_play():
    """测试自我对弈"""
    print("\n步骤 6: 快速测试自我对弈...")
    try:
        import torch
        import numpy as np
        from game import DotsAndBoxesGame
        from model import DotsAndBoxesNet
        from mcts import MCTS
        
        game = DotsAndBoxesGame(5, 5)
        model = DotsAndBoxesNet(game, num_filters=32, num_res_blocks=2)
        
        if torch.cuda.is_available():
            model.cuda()
        
        args = {
            'num_simulations': 5,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'temp_threshold': 5,
        }
        
        mcts = MCTS(game, model, args)
        state = game.get_initial_state()
        
        moves = 0
        max_moves = 10
        
        while not game.is_terminal(state) and moves < max_moves:
            pi = mcts.get_action_prob(state, temp=1)
            action = np.random.choice(len(pi), p=pi)
            state = game.get_next_state(state, action)
            moves += 1
        
        print(f"✓ 自我对弈测试正常")
        print(f"  - 执行步数: {moves}/{max_moves}")
        print(f"  - 游戏结束: {game.is_terminal(state)}")
        
        return True
    except Exception as e:
        print(f"❌ 自我对弈测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    tests = [
        ("基础导入", test_imports),
        ("游戏模块", test_game_module),
        ("模型模块", test_model_module),
        ("MCTS模块", test_mcts_module),
        ("自我对弈", test_self_play),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}测试出错: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n可以开始训练：")
        print("  python main.py")
        print("\n或进行人机对战：")
        print("  python play.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        return 1
    
    print("=" * 60 + "\n")
    return 0

if __name__ == '__main__':
    sys.exit(main())
