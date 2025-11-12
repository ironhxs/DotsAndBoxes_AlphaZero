# -*- coding: utf-8 -*-
"""
神经网络模型 - AlphaZero 风格
改进版本：增加 Dropout、更大容量、改进的残差块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """
    改进的残差块
    - 使用 GroupNorm 代替 BatchNorm（多进程更稳定）
    - 添加 Dropout 正则化
    - SE (Squeeze-and-Excitation) 注意力机制
    """
    def __init__(self, num_filters, dropout=0.3, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, num_filters // 4), num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, num_filters // 4), num_filters)
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        # SE Block (Squeeze-and-Excitation)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(num_filters, reduction=16)
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
        
        out += residual
        out = F.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - 通道注意力"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class DotsAndBoxesNet(nn.Module):
    """
    AlphaZero 风格的神经网络
    
    改进：
    1. 更大的网络容量（默认 128 个滤波器，10 个残差块）
    2. 使用 GroupNorm 替代 BatchNorm（多进程环境更稳定）
    3. 添加 Dropout 正则化
    4. 添加 SE 注意力机制
    5. 改进的初始化策略
    """
    def __init__(self, game, num_filters=128, num_res_blocks=10, dropout=0.3, use_se=True):
        super(DotsAndBoxesNet, self).__init__()
        
        self.action_size = game.get_action_size()
        self.board_size = (game.num_rows + 1) * (game.num_cols + 1)
        self.input_channels = 9  # OpenSpiel dots_and_boxes 的观察通道数
        
        logger.info(f"初始化 DotsAndBoxesNet: "
                   f"filters={num_filters}, blocks={num_res_blocks}, "
                   f"dropout={dropout}, board_size={self.board_size}, "
                   f"actions={self.action_size}")
        
        # === 特征提取 ===
        self.conv1 = nn.Conv2d(self.input_channels, num_filters, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, num_filters // 4), num_filters)
        
        # === 残差塔 ===
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, dropout=dropout, use_se=use_se) 
            for _ in range(num_res_blocks)
        ])
        
        # === 策略头 (Policy Head) ===
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_gn = nn.GroupNorm(8, 32)
        self.policy_fc = nn.Linear(32 * self.board_size, self.action_size)
        self.policy_dropout = nn.Dropout(dropout)
        
        # === 价值头 (Value Head) ===
        self.value_conv = nn.Conv2d(num_filters, 16, 1, bias=False)
        self.value_gn = nn.GroupNorm(4, 16)
        self.value_fc1 = nn.Linear(16 * self.board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (batch_size, 9, H, W) 输入观察
            
        Returns:
            policy: (batch_size, action_size) log 概率
            value: (batch_size, 1) 价值估计 [-1, 1]
        """
        # 特征提取
        x = F.relu(self.gn1(self.conv1(x)))
        
        # 残差塔
        for block in self.res_blocks:
            x = block(x)
        
        # === 策略头 ===
        policy = F.relu(self.policy_gn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # === 价值头 ===
        value = F.relu(self.value_gn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, obs):
        """
        单个观察的预测（用于 MCTS）
        
        Args:
            obs: (C, H, W) numpy array
            
        Returns:
            pi: (action_size,) 策略概率
            v: float 价值估计
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            device = next(self.parameters()).device
            obs_tensor = obs_tensor.to(device)
            
            log_pi, v = self.forward(obs_tensor)
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.item()
        
        return pi, v
    
    def save_checkpoint(self, filepath):
        """保存模型检查点"""
        torch.save({
            'state_dict': self.state_dict(),
            'action_size': self.action_size,
            'board_size': self.board_size
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        logger.info(f"模型已从 {filepath} 加载")
        return self


def create_model(game, config):
    """
    模型工厂函数 - 根据配置创建模型
    
    Args:
        game: 游戏实例
        config: ModelConfig 配置对象
        
    Returns:
        模型实例 (DotsAndBoxesNet 或 DotsAndBoxesTransformer)
    """
    architecture = getattr(config, 'architecture', 'resnet').lower()
    
    if architecture == 'transformer':
        logger.info("创建 Transformer 模型...")
        from .model_transformer import DotsAndBoxesTransformer
        return DotsAndBoxesTransformer(
            game=game,
            num_filters=config.num_filters,
            num_blocks=config.num_res_blocks,
            num_heads=config.num_heads
        )
    else:  # 默认 resnet
        logger.info("创建 ResNet 模型...")
        return DotsAndBoxesNet(
            game=game,
            num_filters=config.num_filters,
            num_res_blocks=config.num_res_blocks,
            dropout=config.dropout,
            use_se=True
        )
