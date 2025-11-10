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
from torch.utils.tensorboard import SummaryWriter
from .mcts import MCTS
from .arena import Arena
from multiprocessing import Pool, cpu_count


def _init_worker_cuda():
    """子进程初始化函数 - 设置 CUDA 环境"""
    import torch
    if torch.cuda.is_available():
        # 触发 CUDA 初始化
        torch.cuda.current_device()
        # 禁用 cudnn benchmark (多进程环境下更稳定)
        torch.backends.cudnn.benchmark = False


def _execute_episode_worker(args_tuple):
    """全局函数用于多进程 - 避免序列化 self"""
    game, nnet_state_dict, args, seed = args_tuple
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 重建模型 (每个进程独立)
    from .model_transformer import DotsAndBoxesTransformer
    nnet = DotsAndBoxesTransformer(
        game,
        num_filters=args['num_filters'],
        num_blocks=args['num_res_blocks'],
        num_heads=args['num_heads']
    )
    nnet.load_state_dict(nnet_state_dict)
    
    # ⚡ 每个进程独立使用GPU推理（必须用GPU，否则Transformer太慢）
    # 注意：进程数不能太多，否则会OOM
    if args.get('cuda', False) and torch.cuda.is_available():
        nnet = nnet.cuda()
    
    nnet.eval()
    
    # 执行一局游戏
    mcts = MCTS(game, nnet, args)
    train_examples = []
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    
    while True:
        episode_step += 1
        canonical_board = game.get_observation(state)
        temp = int(episode_step < args['temp_threshold'])
        pi = mcts.get_action_prob(state, temp=temp)
        
        # 前30步添加Dirichlet噪声
        if episode_step <= 30:
            noise = np.random.dirichlet([args.get('dirichlet_alpha', 0.3)] * len(pi))
            pi = (1 - args.get('dirichlet_epsilon', 0.25)) * pi + args.get('dirichlet_epsilon', 0.25) * noise
            pi = pi * game.get_valid_moves(state)
            pi = pi / np.sum(pi)
        
        train_examples.append([canonical_board, cur_player, pi, None])
        action = np.random.choice(len(pi), p=pi)
        state = game.get_next_state(state, action)
        r = game.get_game_result(state, cur_player)
        
        if r != 0:
            result = [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
            # 🔥 清理显存
            del nnet, mcts, state, train_examples
            if args.get('cuda', False) and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return result
        
        new_player = game.get_current_player(state)
        if new_player != cur_player:
            cur_player = new_player


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
        
        # 🔧 确保模型在正确的设备上
        if args.get('cuda', False) and torch.cuda.is_available():
            self.nnet = self.nnet.cuda()
        
        # TensorBoard - 延迟初始化，避免多进程序列化问题
        self.log_dir = os.path.join('results', 'logs', 'tensorboard')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = None  # 在 learn() 中初始化
        
        # 保存最佳模型（也要在同一设备上）
        self.best_nnet = copy.deepcopy(nnet)
        if args.get('cuda', False) and torch.cuda.is_available():
            self.best_nnet = self.best_nnet.cuda()
    
    def learn(self):
        """
        AlphaZero完整训练循环:
        每次迭代 = 自我对弈 → 训练 → Arena对战 → 模型更新判断
        """
        # 初始化 TensorBoard (仅在主进程)
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"📊 TensorBoard 日志: {self.log_dir}")
            print(f"   启动命令: tensorboard --logdir={self.log_dir}\n")
        
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
            
            # 选择自我对弈模式
            self_play_mode = self.args.get('self_play_mode', 'batch')  # 'batch' or 'multiprocess'
            
            if use_parallel and self_play_mode == 'batch':
                # 🚀 批量推理模式（最优）：多线程对局 + 单GPU批量推理
                from .self_play_batch import self_play_parallel_batch
                train_examples = self_play_parallel_batch(self.game, self.nnet, self.args)
                iteration_train_examples += train_examples
                
            elif use_parallel:
                # 多进程模式（每个进程独立GPU，显存占用高）
                # 🔥 移到CPU避免跨进程传递GPU tensor
                nnet_state = {k: v.cpu() for k, v in self.nnet.state_dict().items()}
                tasks = [
                    (self.game, nnet_state, self.args, np.random.randint(0, 1000000))
                    for _ in range(self.args['num_episodes'])
                ]
                
                with Pool(processes=num_workers, initializer=_init_worker_cuda) as pool:
                    results = list(tqdm(
                        pool.imap(_execute_episode_worker, tasks),
                        total=self.args['num_episodes'],
                        desc=f"🎮 自我对弈({num_workers}进程)"
                    ))
                    for result in results:
                        iteration_train_examples += result
                
                # 🔥 自我对弈后清理显存
                if self.args.get('cuda', False) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                # 单进程模式 (直接使用现有 MCTS)
                for _ in tqdm(range(self.args['num_episodes']), desc="🎮 自我对弈"):
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
                        
                        train_examples.append([canonical_board, cur_player, pi, None])
                        action = np.random.choice(len(pi), p=pi)
                        state = self.game.get_next_state(state, action)
                        r = self.game.get_game_result(state, cur_player)
                        
                        if r != 0:
                            iteration_train_examples += [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
                            break
                        
                        new_player = self.game.get_current_player(state)
                        if new_player != cur_player:
                            cur_player = new_player
            
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
            self.train(train_examples, iteration=i)
            
            # ============================================================
            # 阶段3: Arena对战 - 新模型 vs 历史最好模型 (每N次迭代进行一次)
            # ============================================================
            arena_interval = self.args.get('arena_interval', 1)  # 默认每次都验证
            should_arena = (i % arena_interval == 0) or (i == self.args['num_iterations'])
            
            if should_arena:
                # 🔥 Arena 前清理显存
                if self.args.get('cuda', False) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print(f"\n🥊 Arena对战验证 (迭代 {i}): 新训练模型 vs 历史最好模型")
                
                # 🔥 使用历史最好模型作为对手 (移到CPU避免跨进程传递GPU tensor)
                best_nnet_state = {k: v.cpu() for k, v in self.best_nnet.state_dict().items()}
                
                # 当前模型state_dict (移到CPU)
                current_state = {k: v.cpu() for k, v in self.nnet.state_dict().items()}
                
                # 选择Arena实现
                # 🎯 默认使用 gpu_multiprocess 模式：真正的多进程并行
                arena_mode = self.args.get('arena_mode', 'gpu_multiprocess')  # 'gpu_multiprocess', 'gpu_thread', 'multiprocess'
                
                if arena_mode == 'gpu_multiprocess':
                    # 🚀 GPU多进程模式（推荐）：真正的多核并行 + GPU加速
                    # 优点：充分利用多核CPU、GPU加速、与自我对弈同样方式
                    from .arena_gpu_multiprocess import ArenaGPUMultiProcess
                    arena = ArenaGPUMultiProcess(self.nnet, self.best_nnet, self.game, self.args)
                elif arena_mode == 'gpu_thread':
                    # GPU多线程模式：受GIL限制，实际只用1-2个核心
                    from .arena_gpu import ArenaGPU
                    arena = ArenaGPU(self.nnet, self.best_nnet, self.game, self.args)
                elif arena_mode == 'batch':
                    # 批量推理模式（实验性）：目前fallback到多进程
                    from .arena_batch_inference import ArenaBatchInference
                    arena = ArenaBatchInference(self.nnet, self.best_nnet, self.game, self.args)
                else:
                    # CPU多进程模式（慢但稳定）：用于调试或无GPU环境
                    arena = Arena(current_state, best_nnet_state, self.game, self.args)
                
                new_wins, old_wins, draws = arena.play_games(self.args['arena_compare'])
                
                # 🔥 立即释放 Arena 和显存
                del arena
                if self.args.get('cuda', False) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # 强制同步，确保显存真正释放
                    torch.cuda.synchronize()
                
                # 计算新模型胜率
                total_games = new_wins + old_wins + draws
                new_win_rate = (new_wins + 0.5 * draws) / total_games
                
                print(f"\n📊 新模型胜率: {new_win_rate*100:.1f}% ({new_wins}胜 {draws}平 {old_wins}负)")
                
                # 记录到 TensorBoard
                self.writer.add_scalar('Arena/win_rate', new_win_rate, i)
                self.writer.add_scalar('Arena/new_wins', new_wins, i)
                self.writer.add_scalar('Arena/old_wins', old_wins, i)
                self.writer.add_scalar('Arena/draws', draws, i)
                
                # ============================================================
                # 阶段4: 模型更新判断
                # ============================================================
                threshold = self.args.get('update_threshold', 0.55)
                
                if new_win_rate >= threshold:
                    print(f'✅ 新模型胜率 {new_win_rate*100:.1f}% >= {threshold*100:.1f}% → 接受更新!')
                    self.best_nnet = copy.deepcopy(self.nnet)
                    self.save_checkpoint(filename=f'best_{i}.pth')
                    self.writer.add_scalar('Arena/model_accepted', 1, i)
                else:
                    print(f'❌ 新模型胜率 {new_win_rate*100:.1f}% < {threshold*100:.1f}% → best_nnet保持不变')
                    print(f'   继续用新训练的模型进行下一轮自我对弈...')
                    self.writer.add_scalar('Arena/model_accepted', 0, i)
            else:
                print(f"\n⏭️  跳过Arena验证 (下次验证: 迭代 {(i // arena_interval + 1) * arena_interval})")
                # 不验证时，直接接受新模型
                self.best_nnet = copy.deepcopy(self.nnet)
            
            # 保存检查点
            if i % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(filename=f'checkpoint_{i}.pth')
            self.save_checkpoint(filename='latest.pth')
        
        # 训练结束，关闭 TensorBoard writer
        if self.writer is not None:
            self.writer.close()
            print("\n📊 TensorBoard 日志已保存")
    
    def train(self, examples, iteration=0):
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
        
        # 使用 tqdm 包装整个 epoch 循环
        epoch_iter = tqdm(range(self.args['epochs']), desc='🎯 训练进度', unit='epoch')
        
        for epoch in epoch_iter:
            np.random.shuffle(all_indices)
            
            epoch_pi_loss = 0
            epoch_v_loss = 0
            epoch_start = time.time()
            
            batch_iter = range(num_batches)  # 不显示 batch 进度，只显示 epoch
            
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
            
            # 计算并更新损失到 tqdm
            avg_pi_loss = epoch_pi_loss / num_batches
            avg_v_loss = epoch_v_loss / num_batches
            total_loss = avg_pi_loss + avg_v_loss
            epoch_time = time.time() - epoch_start
            
            # 记录到 TensorBoard
            global_step = iteration * self.args['epochs'] + epoch
            self.writer.add_scalar('Loss/policy', avg_pi_loss, global_step)
            self.writer.add_scalar('Loss/value', avg_v_loss, global_step)
            self.writer.add_scalar('Loss/total', total_loss, global_step)
            self.writer.add_scalar('Training/speed_batches_per_sec', num_batches/epoch_time, global_step)
            
            # 更新 tqdm 显示
            epoch_iter.set_postfix({
                'π_loss': f'{avg_pi_loss:.3f}',
                'v_loss': f'{avg_v_loss:.3f}',
                'total': f'{total_loss:.3f}',
                'speed': f'{num_batches/epoch_time:.1f}b/s'
            })
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """保存模型检查点"""
        os.makedirs(self.args['checkpoint'], exist_ok=True)
        filepath = os.path.join(self.args['checkpoint'], filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)
        if 'best' in filename:
            print(f'🏆 最佳模型已保存: {filepath}')
        else:
            print(f'💾 检查点已保存: {filepath}')
