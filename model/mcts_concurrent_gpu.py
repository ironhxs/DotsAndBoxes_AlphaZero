# -*- coding: utf-8 -*-
"""
单进程并发多游戏 + GPU 批量推理
最高效的 GPU 利用方案
"""

import numpy as np
import torch
import time
import math
from typing import List, Dict, Tuple


class ConcurrentGames:
    """
    并发执行多局游戏，自动批量 GPU 推理
    
    关键优势：
    1. 单进程内并发多局游戏，无进程间通信开销
    2. 自然地批量收集推理请求
    3. GPU 利用率最高
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.device = next(nnet.parameters()).device
        
    def execute_episodes_concurrent(self, num_episodes):
        """
        并发执行多局游戏
        
        Args:
            num_episodes: 游戏总局数
        
        Returns:
            训练样本列表
        """
        concurrent_games = min(self.args.get('parallel_games', 16), num_episodes)
        
        print(f'  🚀 启动并发 GPU 训练')
        print(f'     并发游戏数: {concurrent_games}')
        print(f'     MCTS 模拟: {self.args["num_simulations"]}')
        print(f'     总局数: {num_episodes}')
        
        all_examples = []
        start_time = time.time()
        games_completed = 0
        
        # 分批执行
        for batch_start in range(0, num_episodes, concurrent_games):
            batch_size = min(concurrent_games, num_episodes - batch_start)
            batch_examples = self._run_concurrent_batch(batch_size)
            all_examples.extend(batch_examples)
            games_completed += batch_size
            
            elapsed = time.time() - start_time
            speed = games_completed / elapsed if elapsed > 0 else 0
            print(f'     进度: {games_completed}/{num_episodes} ({speed:.1f} 局/秒)')
        
        elapsed_time = time.time() - start_time
        
        print(f'  ✅ 并发训练完成')
        print(f'    总耗时: {elapsed_time:.2f}s')
        print(f'    平均速度: {num_episodes / elapsed_time:.2f} 局/秒')
        print(f'    样本数: {len(all_examples):,}')
        
        return all_examples
    
    def _run_concurrent_batch(self, num_games):
        """并发运行一批游戏"""
        # 初始化所有游戏状态
        games = []
        for i in range(num_games):
            games.append({
                'state': self.game.get_initial_state(),
                'cur_player': 0,
                'episode_step': 0,
                'examples': [],
                'mcts_trees': {},  # {state_str: {Ps, Ns, Qsa, Nsa}}
                'finished': False
            })
        
        # 游戏主循环
        while any(not g['finished'] for g in games):
            active_games = [g for g in games if not g['finished']]
            if not active_games:
                break
            
            # 每个活跃游戏执行 MCTS
            for game_info in active_games:
                self._execute_mcts_concurrent(game_info)
                
                # 计算动作概率
                state = game_info['state']
                s = str(state)
                trees = game_info['mcts_trees']
                
                if s not in trees or 'Nsa' not in trees[s]:
                    # 应该不会到这里，但以防万一
                    valids = self.game.get_valid_moves(state)
                    pi = valids / np.sum(valids)
                else:
                    counts = [trees[s]['Nsa'].get(a, 0) for a in range(self.game.get_action_size())]
                    
                    temp = int(game_info['episode_step'] < self.args['temp_threshold'])
                    if temp == 0:
                        best_actions = np.where(counts == np.max(counts))[0]
                        pi = np.zeros(len(counts), dtype=np.float32)
                        pi[best_actions] = 1.0 / len(best_actions)
                    else:
                        counts_temp = np.array([x ** (1.0 / temp) for x in counts], dtype=np.float32)
                        counts_sum = float(np.sum(counts_temp))
                        if counts_sum > 0:
                            pi = counts_temp / counts_sum
                        else:
                            valids = self.game.get_valid_moves(state)
                            pi = valids / np.sum(valids)
                
                # 添加探索噪声
                if game_info['episode_step'] <= 30:
                    noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(pi))
                    pi = (1 - self.args['dirichlet_epsilon']) * pi + self.args['dirichlet_epsilon'] * noise
                    valids = self.game.get_valid_moves(state)
                    pi = pi * valids
                    if np.sum(pi) > 0:
                        pi = pi / np.sum(pi)
                    else:
                        pi = valids / np.sum(valids)
                
                # 记录样本
                canonical_board = self.game.get_observation(state)
                game_info['examples'].append([canonical_board, game_info['cur_player'], pi, None])
                
                # 执行动作
                action = np.random.choice(len(pi), p=pi)
                next_state = self.game.get_next_state(state, action)
                game_info['state'] = next_state
                game_info['episode_step'] += 1
                
                # 检查游戏是否结束
                r = self.game.get_game_result(next_state, game_info['cur_player'])
                
                if r != 0:
                    game_info['finished'] = True
                else:
                    game_info['cur_player'] = self.game.get_current_player(next_state)
        
        # 收集所有样本
        batch_examples = []
        for game_info in games:
            if game_info['examples']:
                # 获取最终结果
                final_state = game_info['state']
                final_player = game_info['cur_player']
                r = self.game.get_game_result(final_state, final_player)
                
                # 分配奖励
                final_examples = [
                    (x[0], x[2], r * ((-1) ** (x[1] != final_player)))
                    for x in game_info['examples']
                ]
                batch_examples.extend(final_examples)
        
        return batch_examples
    
    def _execute_mcts_concurrent(self, game_info):
        """为一个游戏执行完整的 MCTS（所有模拟）"""
        state = game_info['state']
        trees = game_info['mcts_trees']
        
        # 批量大小：收集多少个叶子节点后立即评估
        eval_batch_size = self.args.get('mcts_batch_size', 32)
        pending_evaluations = []
        
        for sim_idx in range(self.args['num_simulations']):
            # 每次模拟从根状态开始
            current_state = state.clone()
            path = []
            leaf_to_evaluate = None
            
            # 搜索到叶子节点或终止状态
            while True:
                s = str(current_state)
                
                # 终止状态：直接回传
                if current_state.is_terminal():
                    returns = current_state.returns()
                    if returns[0] > returns[1]:
                        value = 1.0
                    elif returns[0] < returns[1]:
                        value = -1.0
                    else:
                        value = 0.0
                    self._backpropagate(trees, path, value)
                    break
                
                # 叶子节点：需要评估
                if s not in trees or 'Ps' not in trees[s]:
                    leaf_to_evaluate = (current_state.clone(), s, path[:])
                    break
                
                # 内部节点：选择动作（UCB）
                valids = self.game.get_valid_moves(current_state)
                cur_best = -float('inf')
                best_act = -1
                
                for a in range(self.game.get_action_size()):
                    if not valids[a]:
                        continue
                    
                    if a in trees[s]['Qsa']:
                        u = trees[s]['Qsa'][a] + self.args['cpuct'] * trees[s]['Ps'][a] * \
                            math.sqrt(trees[s]['Ns']) / (1 + trees[s]['Nsa'][a])
                    else:
                        u = self.args['cpuct'] * trees[s]['Ps'][a] * math.sqrt(trees[s]['Ns'] + 1e-8)
                    
                    if u > cur_best:
                        cur_best = u
                        best_act = a
                
                if best_act == -1:
                    legal_actions = np.where(valids > 0)[0]
                    if len(legal_actions) == 0:
                        self._backpropagate(trees, path, 0.0)
                        break
                    best_act = np.random.choice(legal_actions)
                
                path.append((s, best_act))
                next_state = self.game.get_next_state(current_state, best_act)
                current_state = next_state
            
            # 收集叶子节点，达到批量大小时立即评估
            if leaf_to_evaluate:
                pending_evaluations.append(leaf_to_evaluate)
                
                # 达到批量大小或最后一次模拟，立即评估
                if len(pending_evaluations) >= eval_batch_size or sim_idx == self.args['num_simulations'] - 1:
                    self._batch_evaluate(trees, pending_evaluations)
                    pending_evaluations = []  # 清空
    
    def _batch_evaluate(self, trees, pending_evaluations):
        """批量评估所有待评估的叶子节点"""
        if not pending_evaluations:
            return
        
        # 准备批量输入
        observations = []
        valid_masks = []
        
        for state, s, path in pending_evaluations:
            observations.append(self.game.get_observation(state))
            valid_masks.append(self.game.get_valid_moves(state))
        
        # GPU 批量推理
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        
        self.nnet.eval()
        with torch.no_grad():
            log_pi_batch, v_batch = self.nnet(obs_tensor)
        
        pi_batch = torch.exp(log_pi_batch).cpu().numpy()
        v_batch = v_batch.cpu().numpy().flatten()
        
        # 处理结果并回传
        for idx, (state, s, path) in enumerate(pending_evaluations):
            pi = pi_batch[idx]
            v = float(v_batch[idx])
            valids = valid_masks[idx]
            
            # 应用合法动作掩码
            pi = pi * valids
            if np.sum(pi) > 0:
                pi = pi / np.sum(pi)
            else:
                pi = valids / np.sum(valids)
            
            # 初始化节点
            if s not in trees:
                trees[s] = {'Ps': pi, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
            else:
                trees[s]['Ps'] = pi
                trees[s]['Ns'] = 0
                trees[s]['Qsa'] = {}
                trees[s]['Nsa'] = {}
            
            # 回传价值
            self._backpropagate(trees, path, v)
    
    def _backpropagate(self, trees, path, value):
        """回传价值"""
        for s, a in reversed(path):
            if s not in trees:
                trees[s] = {'Ps': None, 'Ns': 0, 'Qsa': {}, 'Nsa': {}}
            
            if a in trees[s]['Qsa']:
                trees[s]['Qsa'][a] = (trees[s]['Nsa'][a] * trees[s]['Qsa'][a] + value) / (trees[s]['Nsa'][a] + 1)
                trees[s]['Nsa'][a] += 1
            else:
                trees[s]['Qsa'][a] = value
                trees[s]['Nsa'][a] = 1
            
            trees[s]['Ns'] += 1
            value = -value
