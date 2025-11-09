# -*- coding: utf-8 -*-
"""AlphaZero 完整训练教练 - 包含Arena对战验证"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import time
import copy
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
import torch.optim as optim
from .mcts import MCTS
from .arena import Arena
from multiprocessing import Pool, cpu_count


class Coach:
    """
    真正的AlphaZero训练流程:
    1. 自我对弈收集数据
    2. 训练神经网络得到新模型
    3. Arena对战: 新模型 vs 旧模型
    4. 只有新模型胜率 > 阈值(55%) 才接受更新
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.train_examples_history = []
        
        # 保存最佳模型
        self.best_nnet = copy.deepcopy(nnet)
        
    def execute_episode(self, _=None):
        """单局游戏执行 (支持多进程调用)"""
        mcts = MCTS(self.game, self.nnet, self.args)
        train_examples = []
        state = self.game.get_initial_state()
        cur_player = 0
        episode_step = 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_observation(state)
            temp = int(episode_step < self.args['temp_threshold'])
            pi = mcts.get_action_prob(state, temp=temp)
            
            # 前30步添加Dirichlet噪声增加探索
            if episode_step <= 30:
                noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(pi))
                pi = (1 - self.args['dirichlet_epsilon']) * pi + self.args['dirichlet_epsilon'] * noise
                pi = pi * self.game.get_valid_moves(state)
                pi = pi / np.sum(pi)
            
            train_examples.append([canonical_board, cur_player, pi, None])
            action = np.random.choice(len(pi), p=pi)
            state = self.game.get_next_state(state, action)
            r = self.game.get_game_result(state, cur_player)
            
            if r != 0:
                # 游戏结束，为所有训练样本分配奖励
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
            
            new_player = self.game.get_current_player(state)
            if new_player != cur_player:
                cur_player = new_player
    
    def learn(self):
        """
        AlphaZero完整训练循环:
        每次迭代 = 自我对弈 → 训练 → Arena对战 → 模型更新判断
        """
        # 确定并行进程数
        num_workers = self.args.get('num_workers', min(cpu_count() - 1, 8))
        use_parallel = self.args.get('use_parallel', True) and num_workers > 1
        
        if use_parallel:
            print(f"🚀 启用多进程并行: {num_workers} 个工作进程")
        
        for i in range(1, self.args['num_iterations'] + 1):
            print(f'\n{"=" * 70}')
            print(f'📍 AlphaZero 迭代 {i}/{self.args["num_iterations"]}')
            print(f'{"=" * 70}\n')
            
            # ============================================================
            # 阶段1: 自我对弈收集训练数据
            # ============================================================
            iteration_train_examples = deque([], maxlen=self.args['max_queue_length'])
            
            if use_parallel:
                with Pool(processes=num_workers) as pool:
                    results = list(tqdm(
                        pool.imap(self.execute_episode, range(self.args['num_episodes'])),
                        total=self.args['num_episodes'],
                        desc=f"🎮 自我对弈({num_workers}进程)"
                    ))
                    for result in results:
                        iteration_train_examples += result
            else:
                for _ in tqdm(range(self.args['num_episodes']), desc="🎮 自我对弈"):
                    iteration_train_examples += self.execute_episode()
            
            self.train_examples_history.append(iteration_train_examples)
            if len(self.train_examples_history) > self.args['num_iters_for_train_examples_history']:
                self.train_examples_history.pop(0)
            
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            
            print(f"✓ 收集到 {len(train_examples)} 个训练样本\n")
            
            # ============================================================
            # 阶段2: 训练神经网络
            # ============================================================
            # 保存当前模型作为"旧模型"
            temp_nnet = copy.deepcopy(self.nnet)
            
            self.train(train_examples)
            
            # ============================================================
            # 阶段3: Arena对战 - 新模型 vs 旧模型 (每N次迭代进行一次)
            # ============================================================
            arena_interval = self.args.get('arena_interval', 1)  # 默认每次都验证
            should_arena = (i % arena_interval == 0) or (i == self.args['num_iterations'])
            
            if should_arena:
                print(f"\n🥊 Arena对战验证 (迭代 {i}): 新模型 vs 旧模型")
                
                arena = Arena(self.nnet, temp_nnet, self.game, self.args)
                new_wins, old_wins, draws = arena.play_games(self.args['arena_compare'])
                
                # 计算新模型胜率
                total_games = new_wins + old_wins + draws
                new_win_rate = (new_wins + 0.5 * draws) / total_games
                
                print(f"\n📊 新模型胜率: {new_win_rate*100:.1f}% ({new_wins}胜 {draws}平 {old_wins}负)")
                
                # ============================================================
                # 阶段4: 模型更新判断
                # ============================================================
                threshold = self.args.get('update_threshold', 0.55)
                
                if new_win_rate >= threshold:
                    print(f'✅ 新模型胜率 {new_win_rate*100:.1f}% >= {threshold*100:.1f}% → 接受更新!')
                    self.best_nnet = copy.deepcopy(self.nnet)
                    self.save_checkpoint(filename=f'best_{i}.pth')
                else:
                    print(f'❌ 新模型胜率 {new_win_rate*100:.1f}% < {threshold*100:.1f}% → 拒绝更新!')
                    print(f'   恢复使用旧模型继续训练...')
                    self.nnet = copy.deepcopy(temp_nnet)
            else:
                print(f"\n⏭️  跳过Arena验证 (下次验证: 迭代 {(i // arena_interval + 1) * arena_interval})")
                # 不验证时，直接接受新模型
                self.best_nnet = copy.deepcopy(self.nnet)
            
            # 保存检查点
            if i % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(filename=f'checkpoint_{i}.pth')
            self.save_checkpoint(filename='latest.pth')
    
    def train(self, examples):
        """训练神经网络"""
        optimizer = optim.Adam(
            self.nnet.parameters(), 
            lr=self.args['lr'], 
            weight_decay=self.args['weight_decay']
        )
        self.nnet.train()
        
        # 预先打乱数据
        np.random.shuffle(examples)
        num_batches = len(examples) // self.args['batch_size']
        all_indices = np.arange(len(examples))
        
        print(f"\n🧠 训练神经网络: {self.args['epochs']} epochs, {num_batches} batches/epoch")
        
        for epoch in range(self.args['epochs']):
            np.random.shuffle(all_indices)
            
            epoch_pi_loss = 0
            epoch_v_loss = 0
            epoch_start = time.time()
            
            batch_iter = tqdm(range(num_batches), desc=f'  Epoch {epoch+1}/{self.args["epochs"]}', leave=False)
            
            for batch_idx in batch_iter:
                start_idx = batch_idx * self.args['batch_size']
                end_idx = start_idx + self.args['batch_size']
                batch_indices = all_indices[start_idx:end_idx]
                
                boards, pis, vs = list(zip(*[examples[i] for i in batch_indices]))
                
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))
                
                if self.args['cuda']:
                    boards = boards.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()
                
                # 前向传播
                out_pi, out_v = self.nnet(boards)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                total_loss = l_pi + l_v
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_pi_loss += l_pi.item()
                epoch_v_loss += l_v.item()
            
            # 输出损失
            avg_pi_loss = epoch_pi_loss / num_batches
            avg_v_loss = epoch_v_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.args['epochs']:
                print(f'  Epoch {epoch+1:2d}: Loss π={avg_pi_loss:.3f} v={avg_v_loss:.3f} '
                      f'total={avg_pi_loss+avg_v_loss:.3f} ({num_batches/epoch_time:.1f} batch/s)')
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """保存模型检查点"""
        os.makedirs(self.args['checkpoint'], exist_ok=True)
        filepath = os.path.join(self.args['checkpoint'], filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)
        if 'best' in filename:
            print(f'🏆 最佳模型已保存: {filepath}')
        else:
            print(f'💾 检查点已保存: {filepath}')
