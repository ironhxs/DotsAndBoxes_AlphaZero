# -*- coding: utf-8 -*-
"""AlphaZero 训练教练"""

import os
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
import torch.optim as optim
from .mcts import MCTS


class Coach:
    def __init__(self, game, nnet, args):
        self.game, self.nnet, self.args = game, nnet, args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.train_examples_history = []
        
    def execute_episode(self):
        train_examples = []
        state, cur_player, episode_step = self.game.get_initial_state(), 0, 0
        
        while True:
            episode_step += 1
            canonical_board = self.game.get_observation(state)
            temp = int(episode_step < self.args['temp_threshold'])
            pi = self.mcts.get_action_prob(state, temp=temp)
            
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
        for i in range(1, self.args['num_iterations'] + 1):
            print(f'\n{"=" * 60}\n迭代 {i}/{self.args["num_iterations"]}\n{"=" * 60}')
            
            iteration_train_examples = deque([], maxlen=self.args['max_queue_length'])
            for _ in tqdm(range(self.args['num_episodes']), desc="自我对弈"):
                self.mcts = MCTS(self.game, self.nnet, self.args)
                iteration_train_examples += self.execute_episode()
            
            self.train_examples_history.append(iteration_train_examples)
            if len(self.train_examples_history) > self.args['num_iters_for_train_examples_history']:
                self.train_examples_history.pop(0)
            
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)
            
            self.train(train_examples)
            
            if i % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(filename=f'checkpoint_{i}.pth')
            self.save_checkpoint(filename='latest.pth')
    
    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
        self.nnet.train()
        
        for epoch in range(self.args['epochs']):
            print(f'Epoch {epoch + 1}/{self.args["epochs"]}')
            np.random.shuffle(examples)
            num_batches = len(examples) // self.args['batch_size']
            
            for _ in tqdm(range(num_batches), desc='Training'):
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                
                boards = torch.FloatTensor(np.array(boards))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs))
                
                if self.args['cuda']:
                    boards, target_pis, target_vs = boards.cuda(), target_pis.cuda(), target_vs.cuda()
                
                out_pi, out_v = self.nnet(boards)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                
                optimizer.zero_grad()
                (l_pi + l_v).backward()
                optimizer.step()
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        os.makedirs(self.args['checkpoint'], exist_ok=True)
        filepath = os.path.join(self.args['checkpoint'], filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)
        print(f'模型已保存: {filepath}')
