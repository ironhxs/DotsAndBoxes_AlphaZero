# -*- coding: utf-8 -*-
"""
Arena - AlphaZero模型对战系统
新模型 vs 旧模型对战，只有胜率>阈值才接受新模型
"""

import numpy as np
from tqdm import tqdm
import torch
from .mcts import MCTS


class Arena:
    """
    Arena对战系统
    
    让两个玩家对战多局，统计胜率
    玩家可以是：神经网络+MCTS、纯MCTS、随机等
    """
    
    def __init__(self, game, player1, player2, args):
        """
        初始化Arena
        
        Args:
            game: 游戏环境
            player1: 玩家1 (通常是新模型)
            player2: 玩家2 (通常是旧模型)
            args: 配置参数
        """
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.args = args
        
    def play_game(self, player1_starts=True):
        """
        进行一局游戏
        
        Args:
            player1_starts: player1是否先手
            
        Returns:
            1: player1赢
            -1: player2赢
            0: 平局
        """
        player_mapping = {
            0: self.player1 if player1_starts else self.player2,
            1: self.player2 if player1_starts else self.player1,
        }
        player1_actual_id = 0 if player1_starts else 1

        state = self.game.get_initial_state()
        move_count = 0
        max_moves = self.args.get('arena_max_moves', 300)

        while not self.game.is_terminal(state) and move_count < max_moves:
            move_count += 1

            current_player_id = self.game.get_current_player(state)
            if current_player_id not in player_mapping:
                raise RuntimeError(f"Arena 遇到未知的玩家编号: {current_player_id}")

            current_player = player_mapping[current_player_id]
            action = current_player(state)

            valid_moves = self.game.get_valid_moves(state)
            if action < 0 or action >= len(valid_moves) or valid_moves[action] == 0:
                # 非法动作，当前玩家直接判负
                return -1 if current_player_id == player1_actual_id else 1

            state = self.game.get_next_state(state, action)

        if self.game.is_terminal(state):
            return self.game.get_game_result(state, player1_actual_id)

        return 0
    
    def play_games(self, num_games, num_workers=1, random_start=False, current_iteration=None, total_iterations=None):
        """
        进行多局对战（支持并行）
        
        Args:
            num_games: 对战局数
            num_workers: 并行进程数（1=串行，>1=并行）
            random_start: 是否随机先手（True=随机，False=交替）
            current_iteration: 当前迭代轮次（可选，用于进度条显示）
            total_iterations: 总迭代轮次（可选，用于进度条显示）
            
        Returns:
            (player1_wins, player2_wins, draws)
        """
        num_games = max(1, num_games)  # 至少1局
        
        if num_workers <= 1:
            # 串行执行
            return self._play_games_serial(num_games, random_start, current_iteration, total_iterations)
        else:
            # 并行执行
            return self._play_games_parallel(num_games, num_workers, random_start, current_iteration, total_iterations)
    
    def _play_games_serial(self, num_games, random_start=False, current_iteration=None, total_iterations=None):
        """串行执行多局游戏"""
        import random
        
        player1_wins = 0
        player2_wins = 0
        draws = 0
        
        # 固定宽度30字符，确保与SelfPlay和Train对齐
        desc = f'{"Arena":<30}'
        
        for i in tqdm(range(num_games), desc=desc):
            # 决定先手
            if random_start:
                player1_starts = random.random() < 0.5
            else:
                player1_starts = (i % 2 == 0)
            
            result = self.play_game(player1_starts=player1_starts)
            
            if result == 1:
                player1_wins += 1
            elif result == -1:
                player2_wins += 1
            else:
                draws += 1
        
        return player1_wins, player2_wins, draws
    
    def _play_games_parallel(self, num_games, num_workers, random_start=False, current_iteration=None, total_iterations=None):
        """并行执行多局游戏"""
        import multiprocessing as mp
        import random
        
        # ⚠️ CUDA 多进程必须使用 spawn 模式
        mp_context = mp.get_context('spawn')
        
        # 生成先手分配
        if random_start:
            starts = [random.random() < 0.5 for _ in range(num_games)]
        else:
            starts = [i % 2 == 0 for i in range(num_games)]
        
        # 创建任务列表
        tasks = [(i, starts[i]) for i in range(num_games)]
        
        # 固定宽度30字符，确保与SelfPlay和Train对齐
        desc = f"Arena({num_workers}进程)"
        desc = f'{desc:<30}'
        
        # 并行执行
        with mp_context.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(self._play_game_worker, tasks),
                total=num_games,
                desc=desc
            ))
        
        # 统计结果
        player1_wins = sum(1 for r in results if r == 1)
        player2_wins = sum(1 for r in results if r == -1)
        draws = sum(1 for r in results if r == 0)
        
        return player1_wins, player2_wins, draws
    
    def _play_game_worker(self, task):
        """并行工作函数（用于多进程）"""
        game_idx, player1_starts = task
        return self.play_game(player1_starts=player1_starts)


class NeuralNetPlayer:
    """神经网络玩家 (使用MCTS)"""
    
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.mcts = MCTS(game, nnet, args)
        self.args = args
        
    def __call__(self, state):
        """
        根据当前状态选择动作
        
        Args:
            state: 游戏状态
            
        Returns:
            action: 选择的动作
        """
        # 使用MCTS获取动作概率 (temp=0表示贪心选择)
        probs = self.mcts.get_action_prob(state, temp=0)
        
        # 选择概率最高的合法动作
        valid_moves = self.game.get_valid_moves(state)
        probs = probs * valid_moves  # 只考虑合法动作
        
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
            action = np.argmax(probs)
        else:
            # 如果没有合法动作，随机选一个
            action = np.random.choice(np.where(valid_moves > 0)[0])
        
        return action


class RandomPlayer:
    """随机玩家 (baseline)"""
    
    def __init__(self, game):
        self.game = game
        
    def __call__(self, state):
        """随机选择合法动作"""
        valid_moves = self.game.get_valid_moves(state)
        valid_actions = np.where(valid_moves > 0)[0]
        return np.random.choice(valid_actions)


def compare_models(game, new_nnet, old_nnet, args, current_iteration=None, total_iterations=None):
    """
    比较新旧模型（支持GPU并行）
    
    Args:
        game: 游戏环境
        new_nnet: 新模型
        old_nnet: 旧模型 (如果为None，则与随机玩家比较)
        args: 配置参数
        current_iteration: 当前迭代轮次（可选，用于进度条显示）
        total_iterations: 总迭代轮次（可选，用于进度条显示）
        
    Returns:
        (win_rate, should_accept): 胜率和是否接受新模型
    """
    arena_mode = args.get('arena_mode', 'serial')
    cuda_enabled = args.get('cuda', False)
    
    # Arena 模式信息已在 learn() 开始时输出，此处不重复
    
    if arena_mode == 'gpu_parallel' and cuda_enabled:
        # GPU 多进程并行模式
        return _compare_models_gpu_parallel(game, new_nnet, old_nnet, args, current_iteration, total_iterations)
    else:
        # 串行模式（原始实现）
        if arena_mode == 'gpu_parallel' and not cuda_enabled:
            print(f"  ⚠️  GPU并行模式需要启用CUDA，降级到串行模式")
        return _compare_models_serial(game, new_nnet, old_nnet, args, current_iteration, total_iterations)


def _compare_models_serial(game, new_nnet, old_nnet, args, current_iteration=None, total_iterations=None):
    """串行比较模型（原始实现）"""
    # 创建MCTS参数 (Arena用更多模拟次数)
    arena_args = args.copy()
    arena_args['num_simulations'] = args.get('arena_mcts_simulations', 200)
    
    # 创建玩家
    new_player = NeuralNetPlayer(game, new_nnet, arena_args)
    
    if old_nnet is not None:
        old_player = NeuralNetPlayer(game, old_nnet, arena_args)
    else:
        # 如果没有旧模型，与随机玩家比较
        old_player = RandomPlayer(game)
    
    # 创建Arena
    arena = Arena(game, new_player, old_player, arena_args)
    
    # 进行对战
    num_games = args.get('arena_compare', 40)
    new_wins, old_wins, draws = arena.play_games(
        num_games, num_workers=1, 
        current_iteration=current_iteration, 
        total_iterations=total_iterations
    )
    
    # 计算胜率
    total_decisive = new_wins + old_wins
    if total_decisive > 0:
        win_rate = new_wins / total_decisive
    else:
        win_rate = 0.5  # 全平局，算50%
    
    # 判断是否接受新模型
    threshold = args.get('update_threshold', 0.55)
    should_accept = win_rate >= threshold
    
    # 简洁输出结果（一行）
    decision = '✅ 接受' if should_accept else '❌ 拒绝'
    print(f"Arena: 新模型 {new_wins}胜 vs 旧模型 {old_wins}胜 (平{draws}局) | 胜率 {win_rate*100:.1f}% | {decision}")
    
    return win_rate, should_accept


def _compare_models_gpu_parallel(game, new_nnet, old_nnet, args, current_iteration=None, total_iterations=None):
    """GPU 多进程并行比较模型"""
    import multiprocessing as mp
    from multiprocessing import Manager
    import random
    
    mp.set_start_method('spawn', force=True)
    
    num_games = args.get('arena_compare', 40)
    num_workers = min(args.get('arena_num_workers', 10), num_games)
    
    # 生成先手分配（交替先手）
    starts = [i % 2 == 0 for i in range(num_games)]
    
    # 准备共享的模型状态
    new_state_dict = new_nnet.state_dict()
    old_state_dict = old_nnet.state_dict() if old_nnet is not None else None
    
    # 创建任务列表
    tasks = [(i, starts[i], new_state_dict, old_state_dict, args) for i in range(num_games)]
    
    # 固定宽度30字符，确保与SelfPlay和Train对齐
    desc = f"Arena(GPU×{num_workers})"
    desc = f'{desc:<22}'
    
    # 并行执行 (简洁输出: 使用 tqdm 进度条显示进度)
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_arena_worker_gpu, tasks),
            total=num_games,
            desc=desc
        ))
    
    # 统计结果
    new_wins = sum(1 for r in results if r == 1)
    old_wins = sum(1 for r in results if r == -1)
    draws = sum(1 for r in results if r == 0)
    
    # 计算胜率
    total_decisive = new_wins + old_wins
    if total_decisive > 0:
        win_rate = new_wins / total_decisive
    else:
        win_rate = 0.5
    
    # 判断是否接受
    threshold = args.get('update_threshold', 0.55)
    should_accept = win_rate >= threshold
    
    # 简洁输出结果（一行）
    decision = '✅ 接受' if should_accept else '❌ 拒绝'
    print(f"Arena: 新模型 {new_wins}胜 vs 旧模型 {old_wins}胜 (平{draws}局) | 胜率 {win_rate*100:.1f}% | {decision}")
    
    return win_rate, should_accept


def _arena_worker_gpu(task):
    """GPU 并行 Arena 工作进程"""
    import torch
    import gc
    from .game import DotsAndBoxesGame
    from .mcts import MCTS
    
    game_idx, player1_starts, new_state_dict, old_state_dict, args = task
    
    try:
        # 设置 CUDA
        if args.get('cuda', False):
            torch.cuda.set_device(0)
        
        # 创建游戏环境
        game = DotsAndBoxesGame()
        
        # 创建神经网络（统一使用DotsAndBoxesNet）
        from .model import DotsAndBoxesNet
        
        # 创建模型实例
        new_nnet = DotsAndBoxesNet(game, args)
        old_nnet = DotsAndBoxesNet(game, args) if old_state_dict else None
        
        # 加载模型权重
        new_nnet.load_state_dict(new_state_dict, strict=False)
        if old_nnet and old_state_dict:
            old_nnet.load_state_dict(old_state_dict, strict=False)
        
        # 移到 GPU
        if args.get('cuda', False):
            new_nnet.cuda()
            if old_nnet:
                old_nnet.cuda()
        
        new_nnet.eval()
        if old_nnet:
            old_nnet.eval()
        
        # 创建 MCTS 参数
        arena_args = args.copy()
        arena_args['num_simulations'] = args.get('arena_mcts_simulations', 100)
        
        # 创建 MCTS
        new_mcts = MCTS(game, new_nnet, arena_args)
        old_mcts = MCTS(game, old_nnet, arena_args) if old_nnet else None
        
        # 创建玩家映射
        player_mapping = {
            0: new_mcts if player1_starts else old_mcts,
            1: old_mcts if player1_starts else new_mcts,
        }
        player1_actual_id = 0 if player1_starts else 1
        
        # 进行对战
        state = game.get_initial_state()
        move_count = 0
        max_moves = arena_args.get('arena_max_moves', 300)
        
        while not game.is_terminal(state) and move_count < max_moves:
            move_count += 1
            
            current_player_id = game.get_current_player(state)
            current_mcts = player_mapping[current_player_id]
            
            # 如果是随机玩家
            if current_mcts is None:
                valid_moves = game.get_valid_moves(state)
                action = np.random.choice(np.where(valid_moves > 0)[0])
            else:
                # 使用 MCTS (temp=0 贪心选择)
                probs = current_mcts.get_action_prob(state, temp=0)
                valid_moves = game.get_valid_moves(state)
                probs = probs * valid_moves
                
                if np.sum(probs) > 0:
                    action = np.argmax(probs)
                else:
                    action = np.random.choice(np.where(valid_moves > 0)[0])
            
            # 执行动作
            state = game.get_next_state(state, action)
        
        # 获取结果
        if game.is_terminal(state):
            result = game.get_game_result(state, player1_actual_id)
        else:
            result = 0
        
        # 清理内存
        del new_nnet, old_nnet, new_mcts, old_mcts, game
        gc.collect()
        if args.get('cuda', False):
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        print(f"\n❌ Arena worker {game_idx} 出错: {e}")
        import traceback
        traceback.print_exc()
        return 0
