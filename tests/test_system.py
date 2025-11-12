# -*- coding: utf-8 -*-
"""
系统集成测试 - 验证所有改进是否正常工作
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_config_system():
    """测试配置系统"""
    print("\n" + "="*60)
    print("测试1: 配置系统")
    print("="*60)
    
    try:
        config_path = project_root / "config" / "config.yaml"
        assert config_path.exists(), "配置文件不存在"

        with config_path.open('r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        assert 'game' in config, "配置中缺少 game 部分"
        assert 'model' in config, "配置中缺少 model 部分"
        assert 'trainer' in config, "配置中缺少 trainer 部分"

        game_cfg = config['game']
        trainer_cfg = config['trainer']
        model_cfg = config['model']

        print(f"✅ 成功加载配置: {config_path}")
        print(f"  - 游戏尺寸: {game_cfg['num_rows']}x{game_cfg['num_cols']}")
        print(f"  - 训练批量: {trainer_cfg['batch_size']}")
        print(f"  - 模型层数: {model_cfg['num_blocks']}")

        return True
    
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game():
    """测试游戏环境"""
    print("\n" + "="*60)
    print("测试2: 游戏环境")
    print("="*60)
    
    try:
        from model.game import DotsAndBoxesGame
        
        # 创建游戏
        game = DotsAndBoxesGame(num_rows=5, num_cols=5)
        print(f"✅ 创建游戏成功")
        
        # 测试初始状态
        state = game.get_initial_state()
        assert not state.is_terminal(), "初始状态不应是终局"
        print(f"✅ 初始状态正常")
        
        # 测试观察
        obs = game.get_observation(state)
        assert obs.shape == (9, 6, 6), f"观察形状应为 (9, 6, 6)，实际为 {obs.shape}"
        print(f"✅ 观察形状正确: {obs.shape}")
        
        # 测试合法动作
        valid_moves = game.get_valid_moves(state)
        assert valid_moves.sum() > 0, "初始状态应有合法动作"
        print(f"✅ 合法动作数: {valid_moves.sum()}")
        
        return True
    
    except Exception as e:
        print(f"❌ 游戏环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试神经网络模型"""
    print("\n" + "="*60)
    print("测试3: 神经网络模型")
    print("="*60)
    
    try:
        from model.game import DotsAndBoxesGame
        from model.model import DotsAndBoxesNet
        
        # 创建游戏和模型
        game = DotsAndBoxesGame(num_rows=5, num_cols=5)
        model = DotsAndBoxesNet(
            game,
            num_filters=128,
            num_res_blocks=10,
            dropout=0.3,
            use_se=True
        )
        print(f"✅ 创建模型成功")
        
        # 测试前向传播
        state = game.get_initial_state()
        obs = game.get_observation(state)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            log_pi, v = model(obs_tensor)
        
        assert log_pi.shape == (1, game.get_action_size()), f"策略输出形状错误"
        assert v.shape == (1, 1), f"价值输出形状错误"
        assert -1 <= v.item() <= 1, f"价值应在 [-1, 1] 范围内"
        print(f"✅ 前向传播成功")
        print(f"  - 策略输出形状: {log_pi.shape}")
        print(f"  - 价值输出: {v.item():.4f}")
        
        # 测试参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 总参数量: {total_params:,}")
        
        # 测试 Dropout 模式
        model.train()
        with torch.no_grad():
            log_pi_train, v_train = model(obs_tensor)
        print(f"✅ Train 模式正常")
        
        return True
    
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcts():
    """测试 MCTS"""
    print("\n" + "="*60)
    print("测试4: MCTS")
    print("="*60)
    
    try:
        from model.game import DotsAndBoxesGame
        from model.model import DotsAndBoxesNet
        from model.mcts import MCTS
        
        # 创建游戏、模型和 MCTS
        game = DotsAndBoxesGame(num_rows=5, num_cols=5)
        model = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)
        
        args = {
            'num_simulations': 50,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25
        }
        
        mcts = MCTS(game, model, args)
        print(f"✅ 创建 MCTS 成功")
        
        # 测试动作概率
        state = game.get_initial_state()
        pi = mcts.get_action_prob(state, temp=1)
        
        assert pi.shape == (game.get_action_size(),), f"动作概率形状错误"
        assert np.abs(pi.sum() - 1.0) < 1e-5, f"动作概率之和应为 1.0"
        print(f"✅ MCTS 搜索成功")
        print(f"  - 动作概率形状: {pi.shape}")
        print(f"  - 概率之和: {pi.sum():.6f}")
        
        # 测试贪心模式
        pi_greedy = mcts.get_action_prob(state, temp=0)
        assert np.max(pi_greedy) == 1.0, "贪心模式应有一个动作概率为 1.0"
        print(f"✅ 贪心模式正常")
        
        # 测试搜索统计
        stats = mcts.get_search_statistics(state)
        print(f"✅ 搜索统计: 总访问 {stats['total_visits']} 次")
        
        return True
    
    except Exception as e:
        print(f"❌ MCTS 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference():
    """测试批量推理服务器"""
    print("\n" + "="*60)
    print("测试5: 批量推理服务器")
    print("="*60)
    
    try:
        from model.game import DotsAndBoxesGame
        from model.model import DotsAndBoxesNet
        from model.batch_inference_server import BatchInferenceServer
        import time
        import threading
        
        # 创建游戏和模型
        game = DotsAndBoxesGame(num_rows=5, num_cols=5)
        model = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)
        
        # 创建批量推理服务器
        server = BatchInferenceServer(model, batch_size=16, timeout=0.05)
        server.start()
        print(f"✅ 批量推理服务器启动成功")
        
        # 测试单次推理
        state = game.get_initial_state()
        obs = game.get_observation(state)
        
        pi, v = server.predict(obs)
        assert pi.shape == (game.get_action_size(),), "策略输出形状错误"
        assert -1 <= v <= 1, "价值应在 [-1, 1] 范围内"
        print(f"✅ 单次推理成功")
        
        # 测试并发推理
        results = []
        
        def worker():
            for _ in range(10):
                pi, v = server.predict(obs)
                results.append((pi, v))
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        start_time = time.time()
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        total_requests = len(results)
        throughput = total_requests / elapsed
        
        print(f"✅ 并发推理成功")
        print(f"  - 总请求数: {total_requests}")
        print(f"  - 耗时: {elapsed:.2f} 秒")
        print(f"  - 吞吐量: {throughput:.1f} requests/sec")
        
        # 停止服务器
        server.stop()
        print(f"✅ 批量推理服务器停止成功")
        
        return True
    
    except Exception as e:
        print(f"❌ 批量推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """测试训练循环"""
    print("\n" + "="*60)
    print("测试6: 训练循环")
    print("="*60)
    
    try:
        from model.game import DotsAndBoxesGame
        from model.model import DotsAndBoxesNet
        from model.coach import Coach
        
        # 创建游戏和模型
        game = DotsAndBoxesGame(num_rows=5, num_cols=5)
        model = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)
        
        args = {
            'num_simulations': 25,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
            'num_iterations': 1,
            'num_episodes': 5,
            'lr': 0.001,
            'batch_size': 32,
            'epochs': 2,
            'weight_decay': 1e-4,
            'max_queue_length': 1000,
            'num_iters_for_train_examples_history': 5,
            'checkpoint_interval': 1,
            'checkpoint': 'test_checkpoints',
            'cuda': False,  # 测试时使用 CPU
            'temp_threshold': 15,
            'use_amp': False
        }
        
        coach = Coach(game, model, args)
        print(f"✅ 创建 Coach 成功")
        
        # 测试单局对弈
        examples = coach.execute_episode()
        assert len(examples) > 0, "应该生成训练样本"
        print(f"✅ 执行一局对弈成功，生成 {len(examples)} 个样本")
        
        # 测试训练
        if len(examples) >= 32:  # 确保有足够的样本
            train_stats = coach.train(examples)
            assert 'pi_losses' in train_stats, "应返回策略损失"
            assert 'v_losses' in train_stats, "应返回价值损失"
            print(f"✅ 训练循环成功")
            print(f"  - 策略损失: {train_stats['pi_losses'][-1]:.4f}")
            print(f"  - 价值损失: {train_stats['v_losses'][-1]:.4f}")
        else:
            print(f"⚠️ 样本不足（{len(examples)}），跳过训练测试")
        
        return True
    
    except Exception as e:
        print(f"❌ 训练循环测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DotsAndBoxes AlphaZero 系统测试")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = []
    
    # 运行测试
    results.append(("配置系统", test_config_system()))
    results.append(("游戏环境", test_game()))
    results.append(("神经网络模型", test_model()))
    results.append(("MCTS", test_mcts()))
    results.append(("批量推理服务器", test_batch_inference()))
    results.append(("训练循环", test_training()))
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统工作正常。")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())
