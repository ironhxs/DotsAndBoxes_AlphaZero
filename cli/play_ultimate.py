#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dots and Boxes - 终极对战版本
合并所有功能：人机对战、AI对战、坐标输入、记录导出
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from pathlib import Path
from datetime import datetime
from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.mcts import MCTS
import argparse


class GameVisualizer:
    """游戏记录可视化和导出"""
    def __init__(self, game):
        self.game = game
        self.move_history = []
    
    def record_move(self, state, action, player):
        self.move_history.append({
            'move_number': len(self.move_history) + 1,
            'player': int(player),
            'action': int(action)
        })
    
    def export_to_json(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_record_{timestamp}.json"
        
        filepath = Path("results") / "games" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        game_data = {
            'game_info': {
                'game': 'Dots and Boxes',
                'rows': int(self.game.num_rows),
                'cols': int(self.game.num_cols),
                'date': datetime.now().isoformat(),
                'total_moves': len(self.move_history)
            },
            'moves': self.move_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(game_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 对局记录已保存: {filepath}")
        return filepath


class Player:
    """玩家基类"""
    def get_action(self, state, valid_moves):
        raise NotImplementedError
    
    def reset(self):
        pass


class HumanPlayer(Player):
    """人类玩家 - 支持坐标输入"""
    def __init__(self, game, ai_helper=None):
        self.game = game
        self.ai_helper = ai_helper
    
    def coord_to_action(self, edge_type, row, col):
        """
        坐标转换为动作编号
        
        Args:
            edge_type: 'h' 或 'v' (横边或竖边)
            row: 行号
            col: 列号
        
        Returns:
            动作编号，如果无效返回 None
        """
        num_rows = self.game.num_rows
        num_cols = self.game.num_cols
        num_horizontal = (num_rows + 1) * num_cols
        
        if edge_type.lower() == 'h':
            # 横边
            if 0 <= row <= num_rows and 0 <= col < num_cols:
                return row * num_cols + col
        elif edge_type.lower() == 'v':
            # 竖边
            if 0 <= row < num_rows + 1 and 0 <= col <= num_cols:
                return num_horizontal + col * (num_rows + 1) + row
        
        return None
    
    def action_to_coord(self, action):
        """动作编号转换为坐标描述"""
        num_rows = self.game.num_rows
        num_cols = self.game.num_cols
        num_horizontal = (num_rows + 1) * num_cols
        
        if action < num_horizontal:
            # 横边
            row = action // num_cols
            col = action % num_cols
            return f"横边 h {row} {col} (点({row},{col})到点({row},{col+1}))", "h", row, col
        else:
            # 竖边
            vertical_idx = action - num_horizontal
            col = vertical_idx // (num_rows + 1)
            row = vertical_idx % (num_rows + 1)
            return f"竖边 v {row} {col} (点({row},{col})到点({row+1},{col}))", "v", row, col
    
    def print_valid_moves_with_coords(self, state):
        """显示所有合法动作及其坐标"""
        valid = self.game.get_valid_moves(state)
        valid_actions = [i for i, v in enumerate(valid) if v > 0]
        
        print(f"\n可选择的边 (共 {len(valid_actions)} 条):")
        print("=" * 80)
        
        h_moves = []
        v_moves = []
        
        for action in valid_actions:
            desc, edge_type, row, col = self.action_to_coord(action)
            if edge_type == "h":
                h_moves.append((action, row, col, desc))
            else:
                v_moves.append((action, row, col, desc))
        
        if h_moves:
            print("\n横边 (输入格式: h 行 列):")
            for action, row, col, desc in h_moves[:30]:
                print(f"  {action:3d}: h {row} {col}  ->  {desc}")
            if len(h_moves) > 30:
                print(f"  ... 还有 {len(h_moves) - 30} 条横边")
        
        if v_moves:
            print("\n竖边 (输入格式: v 行 列):")
            for action, row, col, desc in v_moves[:30]:
                print(f"  {action:3d}: v {row} {col}  ->  {desc}")
            if len(v_moves) > 30:
                print(f"  ... 还有 {len(v_moves) - 30} 条竖边")
        
        print("=" * 80)
    
    def get_action(self, state, valid_moves):
        """获取人类玩家输入"""
        valid_actions = [i for i, v in enumerate(valid_moves) if v > 0]
        
        print("\n" + "=" * 80)
        print("你的回合！")
        print("=" * 80)
        
        print("\n可用命令:")
        print("  • 动作编号: 直接输入数字 (如: 5)")
        print("  • 坐标输入: h 行 列 (横边) 或 v 行 列 (竖边)")
        print("    例如: h 1 0  表示第1行第0列的横边")
        print("    例如: v 2 1  表示第2行第1列的竖边")
        print("  • moves  - 显示所有可选的边")
        print("  • hint   - AI 建议最佳动作")
        print("  • top5   - 显示前5个推荐动作")
        print("  • eval   - 评估当前局面优势")
        print("  • quit   - 退出游戏")
        
        while True:
            try:
                user_input = input("\n请输入动作: ").strip().lower()
                
                if user_input == 'quit':
                    print("游戏结束")
                    return None
                
                elif user_input == 'moves':
                    self.print_valid_moves_with_coords(state)
                    continue
                
                elif user_input == 'hint' and self.ai_helper:
                    print("\n💡 AI 正在分析...")
                    action = self.ai_helper.get_action(state, valid_moves)
                    desc, edge_type, row, col = self.action_to_coord(action)
                    probs = self.ai_helper.mcts.get_action_prob(state, temp=0)
                    print(f"   推荐: 动作 {action} = {edge_type} {row} {col}")
                    print(f"   说明: {desc}")
                    print(f"   胜率: {probs[action]*100:.1f}%")
                    continue
                
                elif user_input == 'top5' and self.ai_helper:
                    print("\n🏆 前5个推荐动作:")
                    probs = self.ai_helper.mcts.get_action_prob(state, temp=0)
                    sorted_actions = torch.argsort(torch.tensor(probs), descending=True)
                    
                    count = 0
                    for action in sorted_actions:
                        action = action.item()
                        if probs[action] > 0 and action in valid_actions:
                            desc, edge_type, row, col = self.action_to_coord(action)
                            print(f"   {count+1}. 动作 {action:3d} = {edge_type} {row} {col}  ({probs[action]*100:.1f}%)")
                            print(f"      -> {desc}")
                            count += 1
                            if count >= 5:
                                break
                    continue
                
                elif user_input == 'eval' and self.ai_helper:
                    # 简单评估
                    probs = self.ai_helper.mcts.get_action_prob(state, temp=0)
                    with torch.no_grad():
                        # 获取模型所在的设备
                        device = next(self.ai_helper.nnet.parameters()).device
                        policy, value = self.ai_helper.nnet(
                            torch.FloatTensor(self.game.get_observation(state)).unsqueeze(0).to(device)
                        )
                    value = value.item()
                    print(f"\n📊 局面评估: {value:.3f}")
                    if value > 0.1:
                        print(f"   当前局面对你有利 ✓")
                    elif value < -0.1:
                        print(f"   当前局面对 AI 有利 ✗")
                    else:
                        print(f"   局面均势 ⚖")
                    continue
                
                # 尝试解析输入
                parts = user_input.split()
                
                if len(parts) == 3 and parts[0] in ['h', 'v']:
                    # 坐标输入: h/v 行 列
                    edge_type = parts[0]
                    row = int(parts[1])
                    col = int(parts[2])
                    action = self.coord_to_action(edge_type, row, col)
                    
                    if action is None:
                        print(f"❌ 坐标超出范围！")
                        print(f"   横边范围: h [0-{self.game.num_rows}] [0-{self.game.num_cols-1}]")
                        print(f"   竖边范围: v [0-{self.game.num_rows}] [0-{self.game.num_cols}]")
                        continue
                    
                    if action not in valid_actions:
                        desc, _, _, _ = self.action_to_coord(action)
                        print(f"❌ 该边不可选 (可能已被占用): {desc}")
                        continue
                    
                    desc, _, _, _ = self.action_to_coord(action)
                    print(f"✓ 执行: {desc}")
                    return action
                
                elif len(parts) == 1:
                    # 数字输入
                    action = int(parts[0])
                    
                    if action not in valid_actions:
                        print(f"❌ 动作 {action} 不合法！")
                        continue
                    
                    desc, edge_type, row, col = self.action_to_coord(action)
                    print(f"✓ 执行: 动作 {action} = {edge_type} {row} {col}")
                    print(f"   {desc}")
                    return action
                
                else:
                    print("❌ 无效输入！请输入:")
                    print("   - 数字 (如: 5)")
                    print("   - 坐标 (如: h 1 0 或 v 2 1)")
                    print("   - 命令 (如: hint, moves, quit)")
            
            except (ValueError, IndexError):
                print("❌ 输入格式错误！")
                continue
            except KeyboardInterrupt:
                print("\n游戏中断")
                return None


class AIPlayer(Player):
    """AI 玩家"""
    def __init__(self, game, nnet, mcts_args, name="AI", verbose=False):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(game, nnet, mcts_args)
        self.name = name
        self.verbose = verbose
        self.total_thinking_time = 0
        self.move_count = 0
    
    def get_action(self, state, valid_moves):
        import time
        start_time = time.time()
        
        probs = self.mcts.get_action_prob(state, temp=0)
        action = torch.argmax(torch.tensor(probs)).item()
        
        thinking_time = time.time() - start_time
        self.total_thinking_time += thinking_time
        self.move_count += 1
        
        if self.verbose:
            # 显示前3个候选动作
            sorted_actions = torch.argsort(torch.tensor(probs), descending=True)
            print(f"\n{self.name} 思考 ({thinking_time:.2f}s):")
            for i, a in enumerate(sorted_actions[:3]):
                a = a.item()
                if probs[a] > 0:
                    print(f"  {i+1}. 动作 {a:3d}: {probs[a]*100:.1f}%")
            print(f"  选择: 动作 {action}")
        
        return action
    
    def reset(self):
        self.mcts = MCTS(self.game, self.nnet, self.mcts.args)
        self.total_thinking_time = 0
        self.move_count = 0
    
    def get_avg_thinking_time(self):
        return self.total_thinking_time / self.move_count if self.move_count > 0 else 0


def play_game(player1, player2, game, mode="human_vs_ai", display=True, export_record=False):
    """
    游戏主循环
    
    Args:
        player1, player2: 玩家对象
        game: 游戏对象
        mode: 游戏模式 (human_vs_ai, ai_vs_ai)
        display: 是否显示详细信息
        export_record: 是否导出记录
    
    Returns:
        游戏结果 (从 player1 视角: 1=赢, -1=输, 0=平局)
    """
    state = game.get_initial_state()
    current_player = 0
    move_count = 0
    
    player1.reset()
    player2.reset()
    
    visualizer = GameVisualizer(game) if export_record else None
    
    if display:
        print("\n" + "=" * 80)
        print("游戏开始！")
        print("=" * 80)
        game.display(state)
    
    while not game.is_terminal(state):
        move_count += 1
        
        if display:
            print(f"\n{'='*80}")
            print(f"第 {move_count} 步 - 当前玩家: {current_player} ({player1.name if current_player == 0 else player2.name})")
            print('='*80)
        
        player = player1 if current_player == 0 else player2
        valid_moves = game.get_valid_moves(state)
        
        action = player.get_action(state, valid_moves)
        
        if action is None:
            return None
        
        if visualizer:
            visualizer.record_move(state, action, current_player)
        
        state = game.get_next_state(state, action)
        
        # 更新当前玩家 - 使用游戏状态的玩家信息
        # Dots and Boxes 规则：完成盒子后继续下棋
        if not game.is_terminal(state):
            current_player = game.get_current_player(state)
        
        if display:
            game.display(state)
    
    result = game.get_game_result(state, 0)
    
    if display:
        print("\n" + "=" * 80)
        print("游戏结束！")
        print("=" * 80)
        
        if result > 0:
            print(f"🏆 {player1.name} 获胜！")
        elif result < 0:
            print(f"🏆 {player2.name} 获胜！")
        else:
            print("🤝 平局！")
        
        print(f"总步数: {move_count}")
        print("=" * 80)
    
    if visualizer and export_record:
        visualizer.export_to_json()
    
    return result


def load_model(checkpoint_path, game, device):
    """加载模型"""
    nnet = DotsAndBoxesTransformer(
        game=game,
        num_filters=64,
        num_blocks=4,
        num_heads=4,
        input_channels=9
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        nnet.load_state_dict(state_dict, strict=False)
        print(f"✓ 模型已加载: {checkpoint_path}")
    else:
        print(f"⚠️  未找到模型: {checkpoint_path}")
        print("   使用随机初始化模型...")
    
    nnet.eval()
    return nnet


def main():
    parser = argparse.ArgumentParser(description='Dots and Boxes - 终极对战版')
    parser.add_argument('--mode', type=str, default='human',
                       choices=['human', 'ai', 'dual-ai'],
                       help='游戏模式: human (人机对战), ai (AI自我对弈), dual-ai (双AI对战)')
    parser.add_argument('--checkpoint', type=str, default='results/test_4060/latest.pth',
                       help='AI1 模型路径')
    parser.add_argument('--checkpoint2', type=str, default=None,
                       help='AI2 模型路径 (仅用于 dual-ai 模式)')
    parser.add_argument('--simulations', type=int, default=100,
                       help='AI1 的 MCTS 模拟次数')
    parser.add_argument('--simulations2', type=int, default=None,
                       help='AI2 的 MCTS 模拟次数 (仅用于 dual-ai 模式)')
    parser.add_argument('--human-first', action='store_true', default=True,
                       help='人类先手 (默认)')
    parser.add_argument('--ai-first', dest='human_first', action='store_false',
                       help='AI 先手')
    parser.add_argument('--num-games', type=int, default=1,
                       help='AI 对战局数')
    parser.add_argument('--export', action='store_true',
                       help='导出对局记录')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Dots and Boxes - 终极对战版")
    print("=" * 80)
    
    # 创建游戏
    game = DotsAndBoxesGame()
    
    print(f"\n游戏配置:")
    print(f"  棋盘: {game.num_rows}x{game.num_cols}")
    print(f"  模式: {'人机对战' if args.mode == 'human' else 'AI 自我对弈'}")
    print(f"  MCTS: {args.simulations} 次模拟")
    
    # 加载模型
    print("\n加载 AI 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nnet = load_model(args.checkpoint, game, device)
    
    if torch.cuda.is_available():
        print("✓ GPU 加速已启用")
    
    # MCTS 配置
    mcts_args = {
        'num_simulations': args.simulations,
        'cpuct': 1.0,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.0,
        'temp_threshold': 0,
        'cuda': torch.cuda.is_available(),
    }
    
    if args.mode == 'human':
        # 人机对战
        ai_player = AIPlayer(game, nnet, mcts_args, name="AI")
        human_player = HumanPlayer(game, ai_helper=ai_player)
        
        human_player.name = "你"
        
        if args.human_first:
            player1, player2 = human_player, ai_player
            print(f"\n你是先手 (玩家1)")
        else:
            player1, player2 = ai_player, human_player
            print(f"\nAI 先手，你是后手 (玩家2)")
        
        input("\n按回车开始游戏...")
        
        result = play_game(player1, player2, game, mode="human_vs_ai", 
                          display=True, export_record=args.export)
        
        if result is None:
            print("游戏被中断")
    
    elif args.mode == 'dual-ai':
        # 双 AI 对战 - 不同模型或不同参数
        print(f"\n双 AI 对战模式 ({args.num_games} 局)...")
        
        # 加载 AI2
        checkpoint2 = args.checkpoint2 if args.checkpoint2 else args.checkpoint
        simulations2 = args.simulations2 if args.simulations2 else args.simulations
        
        print(f"\nAI-1 配置:")
        print(f"  模型: {args.checkpoint}")
        print(f"  MCTS: {args.simulations} 次模拟")
        
        print(f"\nAI-2 配置:")
        print(f"  模型: {checkpoint2}")
        print(f"  MCTS: {simulations2} 次模拟")
        
        # 创建 AI2 的模型和 MCTS
        if checkpoint2 != args.checkpoint:
            print("\n加载 AI-2 模型...")
            nnet2 = load_model(checkpoint2, game, device)
        else:
            nnet2 = nnet  # 使用相同模型
        
        mcts_args2 = {
            'num_simulations': simulations2,
            'cpuct': 1.0,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.0,
            'temp_threshold': 0,
            'cuda': torch.cuda.is_available(),
        }
        
        # 启用详细模式用于诊断（仅单局时）
        verbose = (args.num_games == 1)
        ai1 = AIPlayer(game, nnet, mcts_args, name="AI-1", verbose=verbose)
        ai2 = AIPlayer(game, nnet2, mcts_args2, name="AI-2", verbose=verbose)
        
        wins = 0
        losses = 0
        draws = 0
        total_ai1_time = 0
        total_ai2_time = 0
        
        import random
        
        for i in range(args.num_games):
            print(f"\n{'='*80}")
            print(f"第 {i+1}/{args.num_games} 局")
            print('='*80)
            
            # 随机决定先手 - 公平对战
            if random.random() < 0.5:
                # AI-1 先手
                player1, player2 = ai1, ai2
                print("AI-1 先手")
                result = play_game(player1, player2, game, mode="ai_vs_ai",
                                  display=(args.num_games == 1),
                                  export_record=args.export)
                
                if result > 0:
                    wins += 1
                    print("结果: AI-1 获胜")
                elif result < 0:
                    losses += 1
                    print("结果: AI-2 获胜")
                else:
                    draws += 1
                    print("结果: 平局")
            else:
                # AI-2 先手
                player1, player2 = ai2, ai1
                print("AI-2 先手")
                result = play_game(player1, player2, game, mode="ai_vs_ai",
                                  display=(args.num_games == 1),
                                  export_record=args.export)
                
                # 注意：结果是从 player1 视角，所以要反转
                if result > 0:
                    losses += 1
                    print("结果: AI-2 获胜")
                elif result < 0:
                    wins += 1
                    print("结果: AI-1 获胜")
                else:
                    draws += 1
                    print("结果: 平局")
            
            # 累计统计
            if ai1.move_count > 0:
                print(f"  AI-1 平均思考: {ai1.get_avg_thinking_time():.3f}s/步")
            if ai2.move_count > 0:
                print(f"  AI-2 平均思考: {ai2.get_avg_thinking_time():.3f}s/步")
            
            total_ai1_time += ai1.total_thinking_time
            total_ai2_time += ai2.total_thinking_time
        
        # 统计
        print(f"\n{'='*80}")
        print("对战统计（随机先手，公平对战）")
        print('='*80)
        print(f"总局数: {args.num_games}")
        print(f"AI-1 胜: {wins} ({wins/args.num_games*100:.1f}%)")
        print(f"AI-2 胜: {losses} ({losses/args.num_games*100:.1f}%)")
        print(f"平局: {draws} ({draws/args.num_games*100:.1f}%)")
        print('='*80)
    
    else:
        # AI vs AI (自我对弈)
        print(f"\n开始 AI 自我对弈 ({args.num_games} 局)...")
        
        ai1 = AIPlayer(game, nnet, mcts_args, name="AI-1")
        ai2 = AIPlayer(game, nnet, mcts_args, name="AI-2")
        
        wins = 0
        losses = 0
        draws = 0
        
        import random
        
        for i in range(args.num_games):
            print(f"\n{'='*80}")
            print(f"第 {i+1}/{args.num_games} 局")
            print('='*80)
            
            # 随机决定先手 - 公平对战
            if random.random() < 0.5:
                # AI-1 先手
                player1, player2 = ai1, ai2
                print("AI-1 先手")
                result = play_game(player1, player2, game, mode="ai_vs_ai",
                                  display=(args.num_games == 1),
                                  export_record=args.export)
                
                if result > 0:
                    wins += 1
                    print("结果: AI-1 获胜")
                elif result < 0:
                    losses += 1
                    print("结果: AI-2 获胜")
                else:
                    draws += 1
                    print("结果: 平局")
            else:
                # AI-2 先手
                player1, player2 = ai2, ai1
                print("AI-2 先手")
                result = play_game(player1, player2, game, mode="ai_vs_ai",
                                  display=(args.num_games == 1),
                                  export_record=args.export)
                
                # 结果是从 player1 视角，所以要反转
                if result > 0:
                    losses += 1
                    print("结果: AI-2 获胜")
                elif result < 0:
                    wins += 1
                    print("结果: AI-1 获胜")
                else:
                    draws += 1
                    print("结果: 平局")
        
        # 统计
        print(f"\n{'='*80}")
        print("对战统计（随机先手，公平对战）")
        print('='*80)
        print(f"总局数: {args.num_games}")
        print(f"AI-1 胜: {wins} ({wins/args.num_games*100:.1f}%)")
        print(f"AI-2 胜: {losses} ({losses/args.num_games*100:.1f}%)")
        print(f"平局: {draws} ({draws/args.num_games*100:.1f}%)")
        print('='*80)


if __name__ == "__main__":
    main()
