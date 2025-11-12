# -*- coding: utf-8 -*-
"""现代神经网络模型 - 基于 Transformer + Self-Attention"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention 机制"""
    def __init__(self, dim, num_heads=8):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with Attention + MLP"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    """现代卷积块 - Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


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
        return x * y.expand_as(x)


class ModernResBlock(nn.Module):
    """现代残差块 - ConvNeXt style"""
    def __init__(self, dim, drop_path=0.):
        super(ModernResBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.se = SEBlock(dim)
        
    def forward(self, x):
        residual = x
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = self.se(x)
        x = residual + x
        return x


class DotsAndBoxesTransformer(nn.Module):
    """
    现代 AlphaZero 模型架构:
    1. ConvNeXt-style 卷积提取局部特征
    2. Transformer 捕获全局关系
    3. SE 通道注意力增强重要特征
    """
    def __init__(self, game, num_filters=256, num_blocks=12, num_heads=8, input_channels=9):
        super(DotsAndBoxesTransformer, self).__init__()
        self.action_size = game.get_action_size()
        self.board_h = game.num_rows + 1
        self.board_w = game.num_cols + 1
        self.board_size = self.board_h * self.board_w
        self.input_channels = input_channels
        
        # Stem: 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, num_filters // 2, 3, padding=1),
            nn.BatchNorm2d(num_filters // 2),
            nn.GELU(),
            nn.Conv2d(num_filters // 2, num_filters, 3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.GELU()
        )
        
        # Modern ConvNeXt-style blocks
        self.conv_blocks = nn.ModuleList([
            ModernResBlock(num_filters) for _ in range(num_blocks // 2)
        ])
        
        # Transformer blocks for global context
        self.use_transformer = True
        if self.use_transformer:
            self.to_patches = nn.Conv2d(num_filters, num_filters, 1)
            # 位置编码将在forward中动态生成,以适应不同的输入大小
            self.register_buffer('pos_embedding_initialized', torch.tensor(False))
            self.pos_embedding = None
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(num_filters, num_heads) 
                for _ in range(num_blocks // 2)
            ])
            self.from_patches = nn.Linear(num_filters, num_filters)
        
        # Policy Head - 预测动作概率
        self.policy_conv = nn.Sequential(
            ConvBlock(num_filters, num_filters // 2),
            nn.Conv2d(num_filters // 2, 32, 1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # 动态线性层将在第一次forward时初始化
        self.policy_fc1 = None
        self.policy_fc2 = nn.Linear(512, self.action_size)
        self.policy_dropout = nn.Dropout(0.3)
        
        # Value Head - 预测局面价值
        self.value_conv = nn.Sequential(
            ConvBlock(num_filters, num_filters // 4),
            nn.Conv2d(num_filters // 4, 16, 1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        # 动态线性层将在第一次forward时初始化
        self.value_fc1 = None
        self.value_fc2 = nn.Linear(256, 64)
        self.value_fc3 = nn.Linear(64, 1)
        self.value_dropout = nn.Dropout(0.3)
        
        self._init_weights()
        
    def _init_weights(self):
        """Xavier/Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # ConvNeXt blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Transformer for global attention
        if self.use_transformer:
            # Save for skip connection
            conv_out = x
            
            # Reshape to patches
            B, C, H, W = x.shape
            x = self.to_patches(x)
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
            
            # 动态初始化位置编码
            if self.pos_embedding is None or self.pos_embedding.shape[1] != H * W:
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, H * W, C, device=x.device)
                )
            
            # Add positional embedding
            x = x + self.pos_embedding
            
            # Transformer blocks
            for block in self.transformer_blocks:
                x = block(x)
            
            # Reshape back
            x = self.from_patches(x)
            x = x.transpose(1, 2).reshape(B, C, H, W)
            
            # Residual connection
            x = x + conv_out
        
        # Policy head
        policy = self.policy_conv(x)
        B, C_p, H_p, W_p = policy.shape
        policy = policy.flatten(1)
        
        # 动态初始化policy fc1
        if self.policy_fc1 is None or self.policy_fc1.in_features != policy.shape[1]:
            self.policy_fc1 = nn.Linear(policy.shape[1], 512).to(x.device)
            # 初始化权重
            nn.init.kaiming_normal_(self.policy_fc1.weight, mode='fan_out', nonlinearity='relu')
            if self.policy_fc1.bias is not None:
                nn.init.constant_(self.policy_fc1.bias, 0)
        
        policy = self.policy_fc1(policy)
        policy = F.gelu(policy)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc2(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = self.value_conv(x)
        B, C_v, H_v, W_v = value.shape
        value = value.flatten(1)
        
        # 动态初始化value fc1
        if self.value_fc1 is None or self.value_fc1.in_features != value.shape[1]:
            self.value_fc1 = nn.Linear(value.shape[1], 256).to(x.device)
            # 初始化权重
            nn.init.kaiming_normal_(self.value_fc1.weight, mode='fan_out', nonlinearity='relu')
            if self.value_fc1.bias is not None:
                nn.init.constant_(self.value_fc1.bias, 0)
        
        value = self.value_fc1(value)
        value = F.gelu(value)
        value = self.value_dropout(value)
        value = self.value_fc2(value)
        value = F.gelu(value)
        value = self.value_fc3(value)
        value = torch.tanh(value)
        
        return policy, value


# 为了兼容性，保留原接口
DotsAndBoxesNet = DotsAndBoxesTransformer
