# -*- coding: utf-8 -*-
"""
BaseCoach - AlphaZero 训练基类

设计理念：
1. 提取 Coach 和 ParallelCoach 的公共逻辑
2. 简化模型更新机制（previous_net vs current_net）
3. 统一错误处理和日志记录
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm
from collections import deque
from abc import ABC, abstractmethod
import copy
from torch.utils.tensorboard import SummaryWriter


class BaseCoach(ABC):
    """
    AlphaZero 训练基类
    
    子类需要实现：
    - execute_episode() 或 execute_episode_batch(): 自我对弈逻辑
    """
    
    def __init__(self, game, nnet, args):
        """
        初始化教练
        
        Args:
            game: 游戏环境
            nnet: 神经网络模型
            args: 配置参数字典
        """
        self.game = game
        self.nnet = nnet  # 当前模型 (current_net)
        self.args = args
        
        # 训练历史 - 使用 replay_buffer_size 或默认保留 20 次迭代
        # 假设每次迭代约 18000 样本，根据 replay_buffer_size 计算保留次数
        samples_per_iter = args.get('num_self_play_games', 300) * 60  # 300局×60步
        max_iters = max(1, args.get('replay_buffer_size', 360000) // samples_per_iter)
        
        self.train_examples_history = deque(maxlen=max_iters)
        print(f"经验池: 保留 {max_iters} 次迭代（约 {max_iters * samples_per_iter:,} 样本）")
        
        # 前一个模型（用于 Arena 对战）
        self.previous_nnet = None
        
        # TensorBoard
        self.writer = None
        if args.get('tensorboard', False):  # 使用 'tensorboard' 而不是 'use_tensorboard'
            log_dir = args.get('log_dir', 'results/logs')
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"✓ TensorBoard 日志目录: {log_dir}")
        
        # 初始化时，将当前模型作为 previous_net
        self._initialize_previous_net()
    
    def _initialize_previous_net(self):
        """初始化 previous_net 为当前模型的副本"""
        # 对于有动态参数的模型（如Transformer），确保在拷贝前已经初始化
        if hasattr(self.nnet, 'pos_embedding') and self.nnet.pos_embedding is None:
            # 强制初始化动态参数：做一次前向传播
            with torch.no_grad():
                dummy_input = torch.randn(1, 9, 6, 6)
                if self.args.get('cuda', False):
                    dummy_input = dummy_input.cuda()
                    self.nnet.cuda()
                _ = self.nnet(dummy_input)
        
        self.previous_nnet = copy.deepcopy(self.nnet)
        # print("✓ 初始化 previous_net = current_net")
    
    @abstractmethod
    def execute_episode(self):
        """
        执行一局自我对弈（单进程版本）
        
        Returns:
            训练样本列表 [(observation, policy, value), ...]
        """
        pass
    
    def collect_self_play_data(self):
        """
        收集自我对弈数据
        
        注意：子类可以重写此方法以实现并行版本
        
        Returns:
            训练样本列表
        """
        iteration_train_examples = []
        
        for _ in tqdm(range(self.args['num_episodes']), desc="自我对弈"):
            iteration_train_examples.extend(self.execute_episode())
        
        return iteration_train_examples
    
    def learn(self):
        """
        AlphaZero 主训练循环
        
        流程：
        1. 自我对弈生成训练数据
        2. 训练当前模型
        3. Arena 对战（current_net vs previous_net）
        4. 根据胜率决定是否更新 previous_net
        5. 保存检查点
        """
        # 在训练开始前输出 Arena 模式信息（仅一次）
        arena_mode = self.args.get('arena_mode', 'serial')
        cuda_enabled = self.args.get('cuda', False)
        print(f"Arena 模式: {arena_mode}, CUDA: {cuda_enabled}")
        
        for iteration in range(1, self.args['num_iterations'] + 1):
            # 设置当前迭代号（用于 TensorBoard 记录）
            self._current_iteration = iteration
            
            # 不单独打印迭代号，而是在进度条描述中显示 Epoch
            # print(f'迭代 {iteration}/{self.args["num_iterations"]}')
            
            # ========== 1. 自我对弈 ==========
            # print(f'[1/3] 自我对弈...')
            iteration_train_examples = self.collect_self_play_data(iteration)
            
            # 添加到历史
            self.train_examples_history.append(iteration_train_examples)
            
            # 合并所有历史数据
            train_examples = []
            for examples in self.train_examples_history:
                train_examples.extend(examples)
            
            # 显示经验池状态
            max_iters = self.train_examples_history.maxlen
            current_iters = len(self.train_examples_history)
            is_full = current_iters >= max_iters
            
            status = "✅ 已满" if is_full else f"⬆️ 增长中 ({current_iters}/{max_iters})"
            # print(f'  ✓ 训练集: {len(train_examples):,} 样本 (保留 {current_iters} 次迭代) {status}')
            
            # TensorBoard 记录数据集大小
            if self.writer is not None:
                self.writer.add_scalar('Data/IterationSamples', len(iteration_train_examples), iteration)
                self.writer.add_scalar('Data/TotalSamples', len(train_examples), iteration)
            
            # ========== 2. 训练神经网络 ==========
            # print(f'[2/3] 训练神经网络...')
            try:
                train_stats = self.train(train_examples)
                
                # 关键：训练完成后，将主进程的模型移到 CPU，释放 GPU 显存
                if self.args.get('cuda', False):
                    self.nnet.cpu()
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    # print(f'  ✓ 已将模型移至 CPU，释放 GPU 显存')
            except Exception as e:
                print(f'  ❌ 训练出错: {e}')
                import traceback
                traceback.print_exc()
                continue
            
            # ========== 3. Arena 对战 ==========
            arena_interval = self.args.get('arena_interval', 1)
            # ========== 3. Arena 评估 (每 N 次迭代) ==========
            if iteration % arena_interval == 0:
                # print(f'[3/3] Arena 对战...')
                try:
                    should_accept = self._arena_compare()
                    
                    if should_accept:
                        # print('  ✅ 接受新模型 → 更新 baseline (previous_nnet)')
                        self._accept_new_model()
                        self.save_checkpoint(filename=f'best_{iteration}.pth')
                    else:
                        # print('  ❌ 拒绝新模型 → 保持旧 baseline，但继续训练当前模型')
                        self._reject_new_model()
                
                except Exception as e:
                    print(f'  ❌ Arena 出错: {e}')
                    import traceback
                    traceback.print_exc()
                    self._reject_new_model()
            else:
                # 跳过 Arena 时，暂时接受新模型（等下次 Arena 再验证）
                # print(f'[3/3] 跳过 Arena (每 {arena_interval} 次执行一次)')
                self._accept_new_model()
                # print('  ⚠️  新模型暂时接受，将在下次 Arena 中验证')
            
            # ========== 5. 保存检查点 ==========
            if iteration % self.args.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(filename=f'checkpoint_{iteration}.pth')
            
            self.save_checkpoint(filename='latest.pth')
    
    def _arena_compare(self):
        """
        Arena 对战：current_net vs previous_net
        
        Returns:
            bool: 是否应该接受新模型
        """
        from .arena import compare_models
        
        win_rate, should_accept = compare_models(
            self.game,
            self.nnet,  # current_net
            self.previous_nnet,  # previous_net
            self.args,
            current_iteration=self._current_iteration,
            total_iterations=self.args['num_iterations']
        )
        
        # TensorBoard 记录 Arena 结果
        if self.writer is not None and hasattr(self, '_current_iteration'):
            self.writer.add_scalar('Arena/WinRate', win_rate, self._current_iteration)
            self.writer.add_scalar('Arena/Accepted', 1 if should_accept else 0, self._current_iteration)
        
        return should_accept
    
    def _accept_new_model(self):
        """接受新模型：用 current_net 更新 previous_net"""
        # 深拷贝当前模型到 previous_nnet
        # 对于有动态参数的模型（如Transformer），确保在拷贝前已经初始化
        if hasattr(self.nnet, 'pos_embedding') and self.nnet.pos_embedding is None:
            # 强制初始化动态参数：做一次前向传播
            with torch.no_grad():
                dummy_input = torch.randn(1, 9, 6, 6)
                if self.args.get('cuda', False):
                    dummy_input = dummy_input.cuda()
                _ = self.nnet(dummy_input)
        
        self.previous_nnet = copy.deepcopy(self.nnet)
    
    def _reject_new_model(self):
        """
        拒绝新模型：不更新 previous_nnet，但继续从当前模型训练
        
        根据 AlphaZero 论文 (Science 2018) 和 AlphaGo Zero (Nature 2017):
        "if the new player won by a margin of 55%, then it replaced the best player; 
         otherwise, it was discarded."
        
        "discarded" 意思是：不接受为新 baseline，但继续从当前模型训练。
        这样允许训练持续探索，而不是回滚到旧状态。
        """
        # ✅ 不回滚权重 - 保持 self.nnet 继续训练
        # ✅ 不更新 previous_nnet - 保持旧的 baseline
        # 已在 Arena 输出中显示决策，此处不重复输出
        pass
    
    def _load_temp_checkpoint(self):
        """加载临时检查点"""
        temp_path = os.path.join(self.args['checkpoint'], 'temp.pth')
        if os.path.exists(temp_path):
            checkpoint = torch.load(temp_path)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            print(f'✓ 已恢复到训练前的模型')
    
    def train(self, examples):
        """
        训练神经网络
        
        Args:
            examples: 训练样本列表 [(observation, policy, value), ...]
        
        Returns:
            dict: 训练统计信息
        """
        # 创建优化器
        weight_decay = self.args.get('weight_decay', 1e-4)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
        
        optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=self.args['lr'],
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args['epochs'],
            eta_min=self.args['lr'] * 0.01
        )
        
        # 混合精度训练
        use_amp = self.args.get('use_amp', False) and self.args.get('cuda', False)
        scaler = amp.GradScaler(enabled=use_amp) if use_amp else None
        
        # 确保模型在正确的设备上
        if self.args.get('cuda', False):
            self.nnet.cuda()
        
        self.nnet.train()
        
        pi_losses = []
        v_losses = []
        total_losses = []
        
        # 使用 epoch 作为进度（而不是 batch）
        total_epochs = self.args['epochs']
        
        # 创建进度条 - 格式: (Loss=5.1234)        Train: 100%|███| 10/10 [00:07<00:00, pi=4.0951, v=0.9491]
        # 固定宽度确保与SelfPlay和Arena对齐
        progress_bar = tqdm(
            total=total_epochs,
            desc='Train',
            bar_format='({postfix[0]:<15})' + '{desc}:    {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix[1]}]',
            postfix=['Loss=0.0000', 'pi=0.0000, v=0.0000']
        )
        
        # 计算每个 epoch 的批次数
        num_batches_per_epoch = max(1, len(examples) // self.args['batch_size'])
        
        for epoch in range(self.args['epochs']):
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打乱数据
            np.random.shuffle(examples)
            
            epoch_pi_loss = 0
            epoch_v_loss = 0
            epoch_total_loss = 0
            
            for batch_idx in range(num_batches_per_epoch):
                try:
                    # 采样 batch
                    sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                    
                    # 转换为 Tensor
                    boards = torch.FloatTensor(np.array(boards))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs))
                    
                    if self.args.get('cuda', False):
                        boards = boards.cuda()
                        target_pis = target_pis.cuda()
                        target_vs = target_vs.cuda()
                    
                    # 前向传播
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with amp.autocast(enabled=True):
                            out_pi, out_v = self.nnet(boards)
                            
                            # 策略损失（交叉熵）
                            l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                            
                            # 价值损失（MSE）
                            l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size(0)
                            
                            # 总损失
                            total_loss = l_pi + l_v
                        
                        # 反向传播（混合精度）
                        scaler.scale(total_loss).backward()
                        
                        # 梯度裁剪
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.nnet.parameters(),
                            self.args.get('grad_clip', 5.0)
                        )
                        
                        # 优化器步进
                        scaler.step(optimizer)
                        scaler.update()
                        
                        # 立即删除中间变量释放显存
                        del out_pi, out_v, boards, target_pis, target_vs
                    else:
                        # 标准训练
                        out_pi, out_v = self.nnet(boards)
                        
                        # 策略损失（交叉熵）
                        l_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                        
                        # 价值损失（MSE）
                        l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size(0)
                        
                        # 总损失
                        total_loss = l_pi + l_v
                        
                        # 反向传播
                        total_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.nnet.parameters(),
                            self.args.get('grad_clip', 5.0)
                        )
                        
                        # 优化器步进
                        optimizer.step()
                        
                        # 立即删除中间变量释放显存
                        del out_pi, out_v, boards, target_pis, target_vs
                    
                    # 记录损失
                    epoch_pi_loss += l_pi.item()
                    epoch_v_loss += l_v.item()
                    epoch_total_loss += total_loss.item()
                
                except Exception as e:
                    print(f'\n❌ 训练批次 {batch_idx} 出错: {e}')
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 学习率调度
            scheduler.step()
            
            # 记录 epoch 平均损失
            if num_batches_per_epoch > 0:
                avg_pi_loss = epoch_pi_loss / num_batches_per_epoch
                avg_v_loss = epoch_v_loss / num_batches_per_epoch
                avg_total_loss = epoch_total_loss / num_batches_per_epoch
                
                pi_losses.append(avg_pi_loss)
                v_losses.append(avg_v_loss)
                total_losses.append(avg_total_loss)
                
                # 更新进度条 - 格式: (Loss=5.1234)Train: 100%|███| 10/10 [00:07<00:00, pi=4.0951, v=0.9491]
                progress_bar.update(1)
                progress_bar.postfix[0] = f'Loss={avg_total_loss:.4f}'
                progress_bar.postfix[1] = f'pi={avg_pi_loss:.4f}, v={avg_v_loss:.4f}'
                progress_bar.refresh()
                
                # TensorBoard 记录
                if self.writer is not None:
                    global_step = epoch + 1  # 需要从 learn() 传入 iteration
                    if hasattr(self, '_current_iteration'):
                        global_step = self._current_iteration * self.args['epochs'] + epoch + 1
                    self.writer.add_scalar('Loss/Policy', avg_pi_loss, global_step)
                    self.writer.add_scalar('Loss/Value', avg_v_loss, global_step)
                    self.writer.add_scalar('Loss/Total', avg_total_loss, global_step)
                    self.writer.add_scalar('Training/LearningRate', current_lr, global_step)
        
        progress_bar.close()
        
        # 不再单独打印最终统计，已在进度条中显示
        
        # 清理显存 - 删除所有训练相关的变量
        del examples  # 只删除 examples，不删除传入的 train_examples
        if self.args.get('cuda', False):
            # 清理优化器状态
            del optimizer, scheduler
            if scaler is not None:
                del scaler
            # 强制垃圾回收
            import gc
            gc.collect()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
        
        # 设置为评估模式
        self.nnet.eval()
        
        # 返回训练统计
        return {
            'pi_losses': pi_losses,
            'v_losses': v_losses,
            'total_losses': total_losses
        }
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """
        保存模型检查点
        
        Args:
            filename: 文件名
        """
        checkpoint_dir = self.args.get('checkpoint', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filepath = os.path.join(checkpoint_dir, filename)
        
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'args': self.args
        }, filepath)
        
        # 静默保存，不打印信息
        # if 'best' in filename or 'checkpoint' in filename:
        #     print(f'💾 模型已保存: {filepath}')
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """
        加载模型检查点
        
        Args:
            filename: 文件名
        """
        checkpoint_dir = self.args.get('checkpoint', './checkpoints')
        filepath = os.path.join(checkpoint_dir, filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            self.nnet.load_state_dict(checkpoint['state_dict'])
            print(f'✓ 已加载模型: {filepath}')
            return True
        else:
            print(f'⚠️  检查点不存在: {filepath}')
            return False
    
    def __del__(self):
        """析构函数：关闭 TensorBoard writer"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
