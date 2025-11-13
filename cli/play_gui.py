#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dots and Boxes - GUI 终极版本
集成 play_ultimate.py 的功能。
面向用户的界面文本/按钮/弹窗使用英文 (ASCII)，代码注释使用中文。
"""

import sys  
import os
import argparse
import tkinter as tk
from tkinter import messagebox
import threading
import time
import torch
import numpy as np

# --- 编码兼容性提醒 (适用于 WSL/Linux) ---
# 注意：这里不再包含任何 sys.stdout 编码修改代码。
# 如果终端仍有乱码，请使用：PYTHONIOENCODING=utf-8 python cli/play_gui.py

# --- 关键导入 ---
# 将父目录添加到 sys.path 以导入 'model'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.game import DotsAndBoxesGame
from model.model_transformer import DotsAndBoxesTransformer
from model.mcts import MCTS


# ======================================================================
# Part 1: 核心逻辑 (AIPlayer 和 load_model)
# ======================================================================

class AIPlayer:
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
        # 使用 temp=0 进行贪婪选择
        probs = self.mcts.get_action_prob(state.clone(), temp=0)
        action = torch.argmax(torch.tensor(probs)).item()
        return action
    
    def reset(self):
        # 重新创建 MCTS 实例，而不是仅仅清空
        self.mcts = MCTS(self.game, self.nnet, self.mcts.args)
        self.total_thinking_time = 0
        self.move_count = 0
    
    def get_avg_thinking_time(self):
        return self.total_thinking_time / self.move_count if self.move_count > 0 else 0

def load_model(checkpoint_path, game, device):
    """加载模型 (来自 play_ultimate.py)"""
    nnet = DotsAndBoxesTransformer(
        game=game,
        num_filters=256,
        num_blocks=12,
        num_heads=8,
        input_channels=9
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            nnet.load_state_dict(state_dict, strict=False)
            print(f"✓ Model loaded: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}. Using randomly initialized model...")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("   Using randomly initialized model...")
    
    nnet.eval()
    return nnet


# ======================================================================
# Part 2: GUI 应用类
# ======================================================================

class DotsAndBoxesGUI:
    """点格棋 Tkinter GUI 界面"""

    def __init__(self, master, game, ai_player, ai_first=False):
        self.master = master
        self.game = game
        self.ai_player = ai_player
        self.device = next(self.ai_player.nnet.parameters()).device
        
        # --- 颜色和玩家配置 (面向 GUI 的英文文本) ---
        self.HUMAN_PLAYER_NUM = 0
        self.AI_PLAYER_NUM = 1
        self.HUMAN_NAME = "You (Red)" 
        self.AI_NAME = "AI (Blue)"
        self.HUMAN_COLOR = "#D40000"  # 深红色
        self.AI_COLOR = "#0040A0"     # 深蓝色
        self.EDGE_UNPLAYED_COLOR = "#CCCCCC"  # 可走边 (浅灰色)
        self.EDGE_HOVER_COLOR = "#00FF7F"    # 悬停颜色 (中性绿色)
        
        # 游戏状态
        self.state = None
        self.game_over = False
        self.human_score = 0
        self.ai_score = 0
        self.ai_first = ai_first
        self.edge_owner = {}

        # 绘图常量
        self.CELL_SIZE = 90
        self.DOT_RADIUS = 6
        self.PADDING = 50
        
        # 棋盘尺寸
        self.rows = self.game.num_rows
        self.cols = self.game.num_cols
        
        # 画布尺寸
        self.canvas_width = self.cols * self.CELL_SIZE + 2 * self.PADDING
        self.canvas_height = self.rows * self.CELL_SIZE + 2 * self.PADDING

        # 窗口标题 (英文)
        self.master.title(f"Dots and Boxes Human-AI Duel (MCTS Simulations: {self.ai_player.mcts.args['num_simulations']})")
        
        # --- 创建控件 ---
        
        # 1. 状态栏 (英文)
        self.status_text = tk.StringVar()
        self.status_label = tk.Label(master, textvariable=self.status_text, 
                                     font=("Arial", 16, "bold"), 
                                     relief=tk.SUNKEN, anchor=tk.W, 
                                     bg="#E0E0E0", padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 2. 控制面板 (按钮使用英文)
        self.control_frame = tk.Frame(master, padx=10, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.new_game_btn = tk.Button(self.control_frame, text="New Game", font=("Arial", 12), command=self.start_new_game)
        self.new_game_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.hint_btn = tk.Button(self.control_frame, text="AI Hint", font=("Arial", 12), command=self.on_hint_click)
        self.hint_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.top5_btn = tk.Button(self.control_frame, text="Top 5 Moves", font=("Arial", 12), command=self.on_top5_click)
        self.top5_btn.pack(side=tk.LEFT, padx=10, pady=5)

        self.eval_btn = tk.Button(self.control_frame, text="Board Eval (Value)", font=("Arial", 12), command=self.on_eval_click)
        self.eval_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 3. 游戏画布
        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.BOTTOM, padx=20, pady=20)

        # 启动游戏
        self.start_new_game()

    def start_new_game(self):
        """开始一个新游戏"""
        self.game_over = False
        self.state = self.game.get_initial_state()
        self.ai_player.reset()
        self.human_score = 0
        self.ai_score = 0
        self.edge_owner = {}
        
        self.draw_board()
        self.update_status()
        
        # 状态栏信息使用英文
        if self.ai_first:
            self.status_text.set(f"New Game Started! {self.AI_NAME} goes first...")
            self.check_ai_turn()
        else:
            self.status_text.set(f"New Game Started! {self.HUMAN_NAME} goes first.")

    def update_status(self):
        """更新状态栏和分数"""
        if self.game_over:
            return

        current_player = self.game.get_current_player(self.state)
        
        score_str = f"Score: [{self.HUMAN_NAME}] {self.human_score} - {self.ai_score} [{self.AI_NAME}]"
        
        # 启用/禁用分析按钮
        is_human_turn = (current_player == self.HUMAN_PLAYER_NUM)
        state_config = tk.NORMAL if is_human_turn else tk.DISABLED
        
        self.hint_btn.config(state=state_config)
        self.top5_btn.config(state=state_config)
        self.eval_btn.config(state=state_config)
        
        # 状态栏信息使用英文
        if current_player == self.HUMAN_PLAYER_NUM:
            self.status_label.config(fg=self.HUMAN_COLOR)
            self.status_text.set(f"Current Turn: {self.HUMAN_NAME} | {score_str}")
            self.canvas.config(cursor="hand2")
        else:
            self.status_label.config(fg=self.AI_COLOR)
            self.status_text.set(f"AI Thinking: {self.AI_NAME} | {score_str}")
            self.canvas.config(cursor="watch")

    def draw_board(self):
        """(重新)绘制整个棋盘"""
        self.canvas.delete("all")
        valid_moves = self.game.get_valid_moves(self.state)
        total_actions = self.game.get_action_size()

        # 1. 绘制边
        for action in range(total_actions):
            desc, edge_type, r, c = self.action_to_coord_info(action)
            if desc is None:
                continue 

            if edge_type == 'h':
                x1, y1 = self.get_canvas_coords(r, c)
                x2, y2 = x1 + self.CELL_SIZE, y1
                rect = (x1 + 8, y1 - 6, x2 - 8, y2 + 6)
            elif edge_type == 'v':
                x1, y1 = self.get_canvas_coords(r, c)
                x2, y2 = x1, y1 + self.CELL_SIZE
                rect = (x1 - 6, y1 + 8, x2 + 6, y2 - 8)
            else:
                continue

            if action in self.edge_owner:
                owner = self.edge_owner[action]
                color = self.HUMAN_COLOR if owner == self.HUMAN_PLAYER_NUM else self.AI_COLOR
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=6, capstyle=tk.BUTT)
            elif valid_moves[action] > 0:
                edge_id = self.canvas.create_rectangle(rect, fill=self.EDGE_UNPLAYED_COLOR, outline="", tags="edge")
                self.canvas.tag_bind(edge_id, "<Button-1>", lambda e, a=action: self.on_edge_click(a))
                self.canvas.tag_bind(edge_id, "<Enter>", lambda e, eid=edge_id: self.canvas.itemconfig(eid, fill=self.EDGE_HOVER_COLOR))
                self.canvas.tag_bind(edge_id, "<Leave>", lambda e, eid=edge_id: self.canvas.itemconfig(eid, fill=self.EDGE_UNPLAYED_COLOR))

        # 2. 绘制点
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                x, y = self.get_canvas_coords(r, c)
                self.canvas.create_oval(x - self.DOT_RADIUS, y - self.DOT_RADIUS, 
                                        x + self.DOT_RADIUS, y + self.DOT_RADIUS, 
                                        fill="black", outline="black")
    
    def on_edge_click(self, action):
        """人类玩家点击一条边"""
        if self.game_over or self.game.get_current_player(self.state) != self.HUMAN_PLAYER_NUM:
            return
            
        valid_moves = self.game.get_valid_moves(self.state)
        if valid_moves[action] == 0:
            return
        
        old_player = self.game.get_current_player(self.state)
        self.edge_owner[action] = self.HUMAN_PLAYER_NUM
        self.state = self.game.get_next_state(self.state, action)
        
        if self.game.is_terminal(self.state):
            self.draw_board()
            self.show_game_over()
            return

        new_player = self.game.get_current_player(self.state)
        
        if old_player == self.HUMAN_PLAYER_NUM and new_player == self.HUMAN_PLAYER_NUM:
            self.human_score += 1
        
        self.draw_board()
        self.update_status()
        self.check_ai_turn()

    def check_ai_turn(self):
        """检查是否轮到 AI，如果是则启动 AI 线程"""
        if self.game_over:
            return

        if self.game.get_current_player(self.state) == self.AI_PLAYER_NUM:
            self.update_status()
            threading.Thread(target=self.run_ai_move, daemon=True).start()

    def run_ai_move(self):
        """[线程] AI 思考并获取动作"""
        start_time = time.time()
        valid_moves = self.game.get_valid_moves(self.state)
        action = self.ai_player.get_action(self.state, valid_moves)
        thinking_time = time.time() - start_time
        # 控制台输出使用英文
        print(f"AI thought for {thinking_time:.2f}s, chose action {action}")
        self.master.after(0, self.apply_ai_move, action)

    def apply_ai_move(self, action):
        """[主线程] 应用 AI 的动作并更新 GUI"""
        if self.game_over:
            return
        
        old_player = self.game.get_current_player(self.state)
        self.edge_owner[action] = self.AI_PLAYER_NUM
        self.state = self.game.get_next_state(self.state, action)
        
        if self.game.is_terminal(self.state):
            self.draw_board()
            self.show_game_over()
            return
            
        new_player = self.game.get_current_player(self.state)
        
        if old_player == self.AI_PLAYER_NUM and new_player == self.AI_PLAYER_NUM:
            self.ai_score += 1

        self.draw_board()
        self.update_status()
        self.check_ai_turn()

    def show_game_over(self):
        """显示游戏结束信息 (英文)"""
        self.game_over = True
        self.hint_btn.config(state=tk.DISABLED)
        self.top5_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        self.canvas.config(cursor="")
        
        result = self.game.get_game_result(self.state, self.HUMAN_PLAYER_NUM)
        
        # GUI 弹窗文本使用英文 (ASCII)
        if result > 0:
            msg_text = f"🏆 Congratulations, {self.HUMAN_NAME} Wins!"
        elif result < 0:
            msg_text = f"🤖 {self.AI_NAME} Wins!"
        else:
            if self.human_score > self.ai_score:
                msg_text = f"🏆 Congratulations, {self.HUMAN_NAME} Wins! ({self.human_score}:{self.ai_score})"
            elif self.ai_score > self.human_score:
                msg_text = f"🤖 {self.AI_NAME} Wins! ({self.ai_score}:{self.human_score})"
            else:
                msg_text = f"🤝 Draw! ({self.human_score}:{self.ai_score})"
        
        self.status_text.set(f"Game Over! {msg_text}")
        messagebox.showinfo("Game Over", msg_text) # 弹窗标题和内容为英文

    def on_hint_click(self):
        """[线程] 启动 AI 提示"""
        self.status_text.set("AI Analyzing (Hint)...")
        self.hint_btn.config(state=tk.DISABLED)
        self.top5_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_ai_analysis, args=("hint",), daemon=True).start()

    def on_top5_click(self):
        """[线程] 启动 Top 5 分析"""
        self.status_text.set("AI Analyzing (Top 5)...")
        self.hint_btn.config(state=tk.DISABLED)
        self.top5_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_ai_analysis, args=("top5",), daemon=True).start()

    def on_eval_click(self):
        """[线程] 启动 AI 评估"""
        self.status_text.set("AI Analyzing (Eval)...")
        self.hint_btn.config(state=tk.DISABLED)
        self.top5_btn.config(state=tk.DISABLED)
        self.eval_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_ai_analysis, args=("eval",), daemon=True).start()

    def run_ai_analysis(self, mode):
        """[线程] 运行 MCTS 分析 (提示, Top5, 评估)"""
        try:
            mcts = self.ai_player.mcts
            nnet = self.ai_player.nnet
            cloned_state = self.state.clone()
            
            # 始终运行 MCTS 获取概率
            probs = mcts.get_action_prob(cloned_state, temp=0)
            valid_actions = [i for i, p in enumerate(probs) if p > 0]
            
            # 获取 NNet 价值评估
            obs = self.game.get_observation(self.state)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = nnet(obs_tensor)
            value = value.item()
            
            # GUI 弹窗文本使用英文 (ASCII)
            if mode == "hint":
                best_action = torch.argmax(torch.tensor(probs)).item()
                desc, edge_type, row, col = self.action_to_coord_info(best_action)
                prob = probs[best_action]
                
                msg = (f"💡 AI Recommended Action: {best_action}\n"
                       f"   Type/Coords: {edge_type} {row} {col}\n"
                       f"   Description: {desc}\n"
                       f"   MCTS Predicted Probability: {prob*100:.1f}%\n"
                       f"   NNet Board Value (Red): {value:.3f}")
                self.master.after(0, messagebox.showinfo, "AI Hint", msg) # 弹窗标题和内容为英文
            
            elif mode == "top5":
                top_actions = torch.argsort(torch.tensor(probs), descending=True)
                
                msg = "🏆 Top 5 Recommended Actions:\n"
                count = 0
                for action in top_actions:
                    action = action.item()
                    if probs[action] > 0 and action in valid_actions:
                        desc, edge_type, row, col = self.action_to_coord_info(action)
                        msg += f" {count+1}. Action {action:3d} = {edge_type} {row} {col} ({probs[action]*100:.1f}%)\n"
                        msg += f"    -> {desc}\n"
                        count += 1
                        if count >= 5:
                            break
                
                msg += f"\nNNet Board Value (Red): {value:.3f}"
                self.master.after(0, messagebox.showinfo, "Top 5 Recommended Moves", msg) # 弹窗标题和内容为英文
            
            elif mode == "eval":
                value_for_human = value 
                value_for_ai = -value 

                msg = f"📊 NNet Board Value (Red/{self.HUMAN_NAME}): {value_for_human:.3f}\n"
                msg += f"📊 NNet Board Value (Blue/{self.AI_NAME}): {value_for_ai:.3f}\n\n"
                
                if value_for_human > 0.1:
                    msg += f"   (Board is highly favorable to {self.HUMAN_NAME} ✓)"
                elif value_for_human < -0.1:
                    msg += f"   (Board is highly favorable to {self.AI_NAME} ✗)"
                else:
                    msg += "   (Board is balanced ⚖)"
                self.master.after(0, messagebox.showinfo, "Board Evaluation", msg) # 弹窗标题和内容为英文

        except Exception as e:
            # GUI 弹窗文本使用英文 (ASCII)
            self.master.after(0, messagebox.showerror, "Error", f"Analysis failed: {e}") # 弹窗标题和内容为英文
        
        self.master.after(0, self.update_status)

    def get_canvas_coords(self, r, c):
        """获取画布坐标"""
        x = self.PADDING + c * self.CELL_SIZE
        y = self.PADDING + r * self.CELL_SIZE
        return x, y

    def action_to_coord_info(self, action):
        """动作编号转换为坐标描述"""
        num_rows = self.game.num_rows
        num_cols = self.game.num_cols
        num_horizontal = (num_rows + 1) * num_cols
        
        if action < num_horizontal:
            # 横边
            row = action // num_cols
            col = action % num_cols
            return f"Horizontal Edge h {row} {col} (Dot ({row},{col}) to Dot ({row},{col+1}))", "h", row, col
        else:
            # 竖边
            vertical_idx = action - num_horizontal
            row = vertical_idx // (num_cols + 1)
            col = vertical_idx % (num_cols + 1)
            return f"Vertical Edge v {row} {col} (Dot ({row},{col}) to Dot ({row+1},{col}))", "v", row, col
        
        return None, None, -1, -1


# ======================================================================
# Part 3: 主函数 (程序入口)
# ======================================================================

def main():
    # 描述信息使用英文
    parser = argparse.ArgumentParser(description='Dots and Boxes - GUI Player (Console output is English)')
    
    parser.add_argument('--checkpoint', type=str, default='results/test_4060/latest.pth',
                       help='AI 模型检查点路径.')
    parser.add_argument('--simulations', type=int, default=100,
                       help='MCTS 模拟次数 (AI 难度).')
    parser.add_argument('--ai-first', action='store_true',
                       help='如果设置，AI 先手.')
    
    args = parser.parse_args()
    
    # 控制台输出使用英文
    print("\n" + "=" * 80)
    print("Dots and Boxes - GUI Player")
    print("USING ENGLISH CONSOLE OUTPUT FOR WSL/LINUX STABILITY")
    print("=" * 80)
    
    try:
        game = DotsAndBoxesGame()
    except Exception as e:
        print(f"FATAL ERROR: Could not load DotsAndBoxesGame: {e}")
        return

    print(f"\nGame Configuration:")
    print(f"  Board: {game.num_rows}x{game.num_cols}")
    print(f"  AI Difficulty (MCTS): {args.simulations} simulations")

    print("\nLoading AI Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        nnet = load_model(args.checkpoint, game, device)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize Neural Network: {e}")
        return
        
    if torch.cuda.is_available():
        print("✓ GPU acceleration is enabled")
    else:
        print("✓ Using CPU")

    mcts_args = {
        'num_simulations': args.simulations,
        'cpuct': 1.0,
    }
    
    try:
        ai_player = AIPlayer(game, nnet, mcts_args, name="AI", verbose=True)
    except Exception as e:
        print(f"FATAL ERROR: Could not create MCTS Player: {e}")
        return

    print("\nStarting GUI...")
    root = tk.Tk()
    app = DotsAndBoxesGUI(root, game=game, ai_player=ai_player, ai_first=args.ai_first)
    
    def on_closing():
        # GUI 弹窗使用英文 (ASCII)
        if messagebox.askokcancel("Exit", "Are you sure you want to quit the game?"):
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()


if __name__ == "__main__":
    main()