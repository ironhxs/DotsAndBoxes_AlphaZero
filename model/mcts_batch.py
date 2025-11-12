# -*- coding: utf-8 -*-
"""
批量 MCTS - 提升 GPU 利用率
通过批量神经网络推理减少 CPU-GPU 传输开销
"""

import math
import numpy as np
import torch


class BatchMCTS:
    """
    批量 MCTS 实现
    
    关键改进：
    1. 收集多个状态，批量推理
    2. 减少 CPU-GPU 数据传输
    3. 提升 GPU 利用率
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        
        # MCTS 统计量（每个树独立）
        self.trees = {}  # tree_id -> {Qsa, Nsa, Ns, Ps}
        
        # 批量推理相关
        self.batch_size = args.get('mcts_batch_size', 32)
        
    def create_tree(self, tree_id):
        """为新的搜索树创建存储空间"""
        if tree_id not in self.trees:
            self.trees[tree_id] = {
                'Qsa': {},
                'Nsa': {},
                'Ns': {},
                'Ps': {}
            }
    
    def get_action_prob(self, state, tree_id=0, temp=1):
        """
        执行 MCTS 并返回动作概率
        
        Args:
            state: 游戏状态
            tree_id: 树的 ID（用于并行搜索）
            temp: 温度参数
        """
        self.create_tree(tree_id)
        
        pending_states = []
        pending_paths = []
        
        for i in range(self.args['num_simulations']):
            state_copy = state.clone()
            path, value, pending = self._search_collect(state_copy, tree_id)

            if pending is not None:
                pending_states.append(pending)
                pending_paths.append(path)
            else:
                self._backpropagate(tree_id, path, value)

            if pending_states and (len(pending_states) >= self.batch_size or i == self.args['num_simulations'] - 1):
                evaluated_values = self._batch_evaluate(pending_states, tree_id)
                for path_info, value in zip(pending_paths, evaluated_values):
                    self._backpropagate(tree_id, path_info, value)
                pending_states = []
                pending_paths = []
        
        # 计算动作概率
        tree = self.trees[tree_id]
        s = str(state)
        counts = [tree['Nsa'].get((s, a), 0) for a in range(self.game.get_action_size())]
        
        if temp == 0:
            best_actions = np.where(counts == np.max(counts))[0]
            probs = np.zeros(len(counts), dtype=np.float32)
            probs[best_actions] = 1.0 / len(best_actions)
            return probs
        
        counts_temp = np.array([x ** (1.0 / temp) for x in counts], dtype=np.float32)
        counts_sum = float(np.sum(counts_temp))
        
        if counts_sum > 0:
            probs = counts_temp / counts_sum
        else:
            valids = self.game.get_valid_moves(state)
            probs = valids / np.sum(valids)
        
        return probs.astype(np.float32)
    
    def _search_collect(self, state, tree_id):
        """执行一次 MCTS 搜索，返回路径和待评估的叶子"""
        tree = self.trees[tree_id]
        path = []

        while True:
            s = str(state)

            if state.is_terminal():
                returns = state.returns()
                if returns[0] > returns[1]:
                    value = 1.0
                elif returns[0] < returns[1]:
                    value = -1.0
                else:
                    value = 0.0
                return path, value, None

            if s not in tree['Ps']:
                return path, None, (state.clone(), s, tree_id)

            if s not in tree['Ns']:
                tree['Ns'][s] = 0

            valids = self.game.get_valid_moves(state)
            cur_best = -float('inf')
            best_act = -1

            for a in range(self.game.get_action_size()):
                if not valids[a]:
                    continue

                if (s, a) in tree['Qsa']:
                    u = tree['Qsa'][(s, a)] + self.args['cpuct'] * tree['Ps'][s][a] * \
                        math.sqrt(tree['Ns'][s]) / (1 + tree['Nsa'][(s, a)])
                else:
                    u = self.args['cpuct'] * tree['Ps'][s][a] * math.sqrt(tree['Ns'][s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

            if best_act == -1:
                legal_actions = np.where(valids > 0)[0]
                if len(legal_actions) == 0:
                    return path, 0.0, None
                best_act = np.random.choice(legal_actions)

            path.append((s, best_act))
            state = self.game.get_next_state(state, best_act)
    
    def _batch_evaluate(self, states_to_evaluate, tree_id):
        """批量评估多个状态并返回对应价值"""
        if not states_to_evaluate:
            return []

        tree = self.trees[tree_id]

        observations = []
        valid_masks = []
        state_strs = []

        for state, s, _ in states_to_evaluate:
            observations.append(self.game.get_observation(state))
            valid_masks.append(self.game.get_valid_moves(state))
            state_strs.append(s)

        obs_tensor = torch.FloatTensor(np.array(observations))
        device = next(self.nnet.parameters()).device
        obs_tensor = obs_tensor.to(device)

        self.nnet.eval()
        with torch.no_grad():
            log_pi_batch, v_batch = self.nnet(obs_tensor)

        pi_batch = torch.exp(log_pi_batch).cpu().numpy()
        v_batch = v_batch.cpu().numpy().flatten()

        values = []

        for idx, s in enumerate(state_strs):
            pi = pi_batch[idx]
            v = float(v_batch[idx])
            valids = valid_masks[idx]

            pi = pi * valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                pi = valids / np.sum(valids)

            tree['Ps'][s] = pi
            tree['Ns'][s] = 0
            values.append(v)

        return values

    def _backpropagate(self, tree_id, path, leaf_value):
        """根据网络评估结果回传价值"""
        if not path:
            return

        tree = self.trees[tree_id]
        value = leaf_value

        for state_str, action in reversed(path):
            key = (state_str, action)
            if key in tree['Qsa']:
                tree['Qsa'][key] = (tree['Nsa'][key] * tree['Qsa'][key] + value) / (tree['Nsa'][key] + 1)
                tree['Nsa'][key] += 1
            else:
                tree['Qsa'][key] = value
                tree['Nsa'][key] = 1

            tree['Ns'][state_str] = tree['Ns'].get(state_str, 0) + 1
            value = -value
    
    def reset_tree(self, tree_id):
        """重置指定树"""
        if tree_id in self.trees:
            del self.trees[tree_id]


class ParallelSelfPlay:
    """
    并行自我对弈
    
    改进：
    1. 多个对局同时进行
    2. 批量 MCTS 推理
    3. 提升 GPU 利用率
    """
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.batch_mcts = BatchMCTS(game, nnet, args)
    
    def execute_episodes_parallel(self, num_episodes):
        """
        并行执行多局游戏
        
        Args:
            num_episodes: 游戏局数
        
        Returns:
            训练样本列表
        """
        all_examples = []
        
        # 分批并行
        parallel_games = self.args.get('parallel_games', 8)
        
        for batch_start in range(0, num_episodes, parallel_games):
            batch_size = min(parallel_games, num_episodes - batch_start)
            
            # 初始化多局游戏
            games_state = []
            for i in range(batch_size):
                games_state.append({
                    'state': self.game.get_initial_state(),
                    'examples': [],
                    'cur_player': 0,
                    'episode_step': 0,
                    'finished': False,
                    'tree_id': batch_start + i
                })
            
            # 并行执行游戏
            while any(not g['finished'] for g in games_state):
                # 收集所有活跃游戏的状态
                active_games = [g for g in games_state if not g['finished']]
                
                if not active_games:
                    break
                
                # 批量 MCTS
                for game_info in active_games:
                    state = game_info['state']
                    tree_id = game_info['tree_id']
                    episode_step = game_info['episode_step']
                    
                    # 获取动作概率
                    canonical_board = self.game.get_observation(state)
                    temp = int(episode_step < self.args['temp_threshold'])
                    pi = self.batch_mcts.get_action_prob(state, tree_id=tree_id, temp=temp)
                    
                    # 添加探索噪声
                    if episode_step <= 30:
                        noise = np.random.dirichlet([self.args['dirichlet_alpha']] * len(pi))
                        pi = (1 - self.args['dirichlet_epsilon']) * pi + self.args['dirichlet_epsilon'] * noise
                        pi = pi * self.game.get_valid_moves(state)
                        pi = pi / np.sum(pi)
                    
                    # 记录样本
                    game_info['examples'].append([canonical_board, game_info['cur_player'], pi, None])
                    
                    # 执行动作
                    action = np.random.choice(len(pi), p=pi)
                    next_state = self.game.get_next_state(state, action)
                    game_info['state'] = next_state
                    game_info['episode_step'] += 1
                    
                    # 检查游戏是否结束
                    r = self.game.get_game_result(next_state, game_info['cur_player'])
                    
                    if r != 0:
                        # 游戏结束，分配奖励
                        final_examples = [(x[0], x[2], r * ((-1) ** (x[1] != game_info['cur_player']))) 
                                         for x in game_info['examples']]
                        all_examples.extend(final_examples)
                        game_info['finished'] = True
                        
                        # 清理树
                        self.batch_mcts.reset_tree(tree_id)
                    else:
                        # 更新当前玩家
                        new_player = self.game.get_current_player(next_state)
                        if new_player != game_info['cur_player']:
                            game_info['cur_player'] = new_player

        return all_examples
