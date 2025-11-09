# -*- coding: utf-8 -*-
"""AlphaZero 训练教练 - 多进程并行版本"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import time
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
import torch.optim as optim
from .mcts import MCTS
from multiprocessing import Pool, cpu_count
import pickle


class Coach:
    def __init__(self, game, nnet, args):
        self.game, self.nnet, self.args = game, nnet, args
        self.train_examples_history = []
        
    def execute_episode(self, _=None):
        """单局游戏执行 (支持多进程调用)"""
        # 每个进程创建自己的 MCTS
        mcts = MCTS(self.game, self.nnet, self.args)
        train_examples = []
        state, cur_player, episode_step = self.game.get_initial_state(), 0, 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_observation(state)
            temp = int(episode_step < self.args['temp_threshold'])
            pi = mcts.get_action_prob(state, temp=temp)
            
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
                return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
            
            new_player = self.game.get_current_player(state)
            if new_player != cur_player:
                cur_player = new_player
    
    def learn(self):
        """训练循环 - 支持并行自我对弈"""
        # 确定并行进程数
        num_workers = self.args.get('num_workers', min(cpu_count() - 1, 8))
        use_parallel = self.args.get('use_parallel', True) and num_workers > 1
        
        if use_parallel:
            print(f"🚀 启用多进程并行: {num_workers} 个工作进程")
        
        for i in range(1, self.args['num_iterations'] + 1):
            print(f'\n{"=" * 70}')
            print(f'迭代 {i}/{self.args["num_iterations"]}')
            print(f'{"=" * 70}')
            print(f'📌 当前模型参数: {sum(p.numel() for p in self.nnet.parameters())/1e6:.2f}M')
            print(f'📊 历史训练样本: {sum(len(e) for e in self.train_examples_history)} 个')
            print()
            
            iteration_train_examples = deque([], maxlen=self.args['max_queue_length'])
            
            # 🔥 多进程并行自我对弈
            if use_parallel:
                with Pool(processes=num_workers) as pool:
                    # 使用 imap 显示进度条
                    results = list(tqdm(
                        pool.imap(self.execute_episode, range(self.args['num_episodes'])),
                        total=self.args['num_episodes'],
                        desc=f"并行自我对弈({num_workers}进程)"
                    ))
                    for result in results:
                        iteration_train_examples += result
            else:
                # 单进程模式
                for _ in tqdm(range(self.args['num_episodes']), desc="自我对弈"):
                    iteration_train_examples += self.execute_episode()
            
            self.train_examples_history.append(iteration_train_examples)
            if len(self.train_examples_history) > self.args['num_iters_for_train_examples_history']:
                oldest = self.train_examples_history.pop(0)
                print(f"   (丢弃最早的 {len(oldest)} 个样本)")
            
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            
            print(f"\n{'─'*70}")
            print(f"📦 阶段1完成: 自我对弈")
            print(f"   • 本次收集: {len(iteration_train_examples)} 个新样本")
            print(f"   • 历史保留: {len(self.train_examples_history)} 次迭代的数据")
            print(f"   • 总训练集: {len(train_examples)} 个样本")
            print(f"{'─'*70}\n")
            
            print(f"🧠 阶段2开始: 训练神经网络")
            print(f"   • 训练轮数: {self.args['epochs']} epochs")
            print(f"   • 批大小: {self.args['batch_size']}")
            print(f"   • 优化器: Adam (lr={self.args['lr']})")
            print()
            
            self.train(train_examples)
            
            if i % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(filename=f'checkpoint_{i}.pth')
            self.save_checkpoint(filename='latest.pth')
    
    def train(self, examples):
        """训练神经网络 - 优化版"""
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.nnet.train()
        
        # 🔥 预先打乱数据，避免每个epoch重复
        np.random.shuffle(examples)
        num_batches = len(examples) // self.args['batch_size']
        
        # 🔥 预先创建所有batch索引，避免重复计算
        all_indices = np.arange(len(examples))
        
        total_pi_loss = 0
        total_v_loss = 0
        
        for epoch in range(self.args['epochs']):
            # 🔥 每个epoch只打乱一次索引（不是整个数据）
            np.random.shuffle(all_indices)
            
            epoch_pi_loss = 0
            epoch_v_loss = 0
            
            # 🔥 显示epoch进度
            epoch_start = time.time()
            
            for batch_idx in tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{self.args["epochs"]}', leave=False):
                # 🔥 使用切片而不是随机采样，更高效
                start_idx = batch_idx * self.args['batch_size']
                end_idx = start_idx + self.args['batch_size']
                batch_indices = all_indices[start_idx:end_idx]
                
                boards, pis, vs = list(zip(*[examples[i] for i in batch_indices]))
                
                # 🔥 直接转换为GPU tensor，减少CPU-GPU传输
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))
                
                if self.args['cuda']:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()
                
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
            
            # 计算平均损失
            avg_pi_loss = epoch_pi_loss / num_batches
            avg_v_loss = epoch_v_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            # 🔥 更简洁的输出，每10个epoch显示一次
            if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == self.args['epochs']:
                print(f'Epoch {epoch+1:2d}/{self.args["epochs"]}: '
                      f'Loss π={avg_pi_loss:.4f} v={avg_v_loss:.4f} '
                      f'total={avg_pi_loss+avg_v_loss:.4f} '
                      f'({epoch_time:.1f}s, {num_batches/epoch_time:.1f} batch/s)')
            
            total_pi_loss += avg_pi_loss
            total_v_loss += avg_v_loss
        
        print(f"\n{'─'*70}")
        print(f"📦 阶段2完成: 神经网络训练")
        print(f"   • 平均Policy损失: {total_pi_loss/self.args['epochs']:.4f}")
        print(f"   • 平均Value损失: {total_v_loss/self.args['epochs']:.4f}")
        print(f"{'─'*70}\n")
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        os.makedirs(self.args['checkpoint'], exist_ok=True)
        filepath = os.path.join(self.args['checkpoint'], filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)
        print(f'💾 模型已保存: {filepath}')
