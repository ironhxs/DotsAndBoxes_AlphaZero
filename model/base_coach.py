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
        
        # 训练历史
        self.train_examples_history = deque(
            maxlen=args.get('num_iters_for_train_examples_history', 20)
        )
        
        # 前一个模型（用于 Arena 对战）
        self.previous_nnet = None
        
        # 初始化时，将当前模型作为 previous_net
        self._initialize_previous_net()
    
    def _initialize_previous_net(self):
        """初始化 previous_net 为当前模型的副本"""
        self.previous_nnet = copy.deepcopy(self.nnet)
        print("✓ 初始化 previous_net = current_net")
    
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
        for iteration in range(1, self.args['num_iterations'] + 1):
            print(f'\n{"=" * 60}')
            print(f'迭代 {iteration}/{self.args["num_iterations"]}')
            print(f'{"=" * 60}')
            
            # ========== 1. 自我对弈 ==========
            print(f'\n[1/4] 自我对弈...')
            iteration_train_examples = self.collect_self_play_data()
            
            print(f'✓ 生成 {len(iteration_train_examples)} 个训练样本')
            
            # 添加到历史
            self.train_examples_history.append(iteration_train_examples)
            
            # 合并所有历史数据
            train_examples = []
            for examples in self.train_examples_history:
                train_examples.extend(examples)
            
            print(f'✓ 训练集大小: {len(train_examples)} 样本')
            
            # ========== 2. 保存训练前的模型 ==========
            print(f'\n[2/4] 保存当前模型...')
            self.save_checkpoint(filename='temp.pth')
            
            # ========== 3. 训练神经网络 ==========
            print(f'\n[3/4] 训练神经网络...')
            try:
                train_stats = self.train(train_examples)
                print(f'✓ 训练完成')
                if train_stats:
                    print(f'  平均策略损失: {np.mean(train_stats["pi_losses"]):.4f}')
                    print(f'  平均价值损失: {np.mean(train_stats["v_losses"]):.4f}')
            except Exception as e:
                print(f'❌ 训练出错: {e}')
                import traceback
                traceback.print_exc()
                # 恢复到训练前的模型
                self._load_temp_checkpoint()
                continue
            
            # ========== 4. Arena 对战 ==========
            arena_interval = self.args.get('arena_interval', 1)
            
            if iteration % arena_interval == 0:
                print(f'\n[4/4] Arena 对战验证...')
                try:
                    should_accept = self._arena_compare()
                    
                    if should_accept:
                        print('✅ 新模型表现更好，更新 previous_net')
                        self._accept_new_model()
                        
                        # 保存最佳模型
                        self.save_checkpoint(filename=f'best_{iteration}.pth')
                    else:
                        print('❌ 新模型表现不佳，保持 previous_net，回退 current_net')
                        self._reject_new_model()
                
                except Exception as e:
                    print(f'❌ Arena 对战出错: {e}')
                    import traceback
                    traceback.print_exc()
                    # 出错时保守策略：不接受新模型
                    self._reject_new_model()
            else:
                print(f'\n[4/4] 跳过 Arena 验证 (每 {arena_interval} 次迭代验证一次)')
                # 不验证时，保守策略：接受新模型
                self._accept_new_model()
            
            # ========== 5. 保存检查点 ==========
            if iteration % self.args.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(filename=f'checkpoint_{iteration}.pth')
            
            self.save_checkpoint(filename='latest.pth')
            
            print(f'\n{"=" * 60}')
            print(f'迭代 {iteration} 完成')
            print(f'{"=" * 60}\n')
    
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
            self.args
        )
        
        print(f'  Arena 胜率: {win_rate * 100:.1f}%')
        
        return should_accept
    
    def _accept_new_model(self):
        """接受新模型：用 current_net 更新 previous_net"""
        self.previous_nnet = copy.deepcopy(self.nnet)
    
    def _reject_new_model(self):
        """拒绝新模型：用 previous_net 恢复 current_net"""
        if self.previous_nnet is not None:
            self.nnet.load_state_dict(self.previous_nnet.state_dict())
    
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
        optimizer = optim.Adam(
            self.nnet.parameters(),
            lr=self.args['lr'],
            weight_decay=self.args.get('weight_decay', 1e-4)
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
        
        self.nnet.train()
        
        pi_losses = []
        v_losses = []
        total_losses = []
        
        for epoch in range(self.args['epochs']):
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{self.args["epochs"]} - LR: {current_lr:.6f}')
            
            # 打乱数据
            np.random.shuffle(examples)
            num_batches = max(1, len(examples) // self.args['batch_size'])
            
            epoch_pi_loss = 0
            epoch_v_loss = 0
            epoch_total_loss = 0
            
            batch_iterator = tqdm(range(num_batches), desc='Training')
            
            for batch_idx in batch_iterator:
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
                    
                    # 记录损失
                    epoch_pi_loss += l_pi.item()
                    epoch_v_loss += l_v.item()
                    epoch_total_loss += total_loss.item()
                    
                    # 更新进度条
                    batch_iterator.set_postfix({
                        'pi_loss': f'{l_pi.item():.4f}',
                        'v_loss': f'{l_v.item():.4f}',
                        'total': f'{total_loss.item():.4f}'
                    })
                
                except Exception as e:
                    print(f'\n❌ 训练批次 {batch_idx} 出错: {e}')
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 学习率调度
            scheduler.step()
            
            # 记录 epoch 平均损失
            if num_batches > 0:
                avg_pi_loss = epoch_pi_loss / num_batches
                avg_v_loss = epoch_v_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches
                
                pi_losses.append(avg_pi_loss)
                v_losses.append(avg_v_loss)
                total_losses.append(avg_total_loss)
                
                print(f'Epoch {epoch + 1} 平均损失 - '
                      f'Policy: {avg_pi_loss:.4f}, '
                      f'Value: {avg_v_loss:.4f}, '
                      f'Total: {avg_total_loss:.4f}')
        
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
        
        # 只在保存重要检查点时打印
        if 'best' in filename or 'checkpoint' in filename:
            print(f'💾 模型已保存: {filepath}')
    
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
