# -*- coding: utf-8 -*-
# @Time : 2025/11/8
# @Author : ironhxs
# @File : __init__.py

"""
点格棋 AlphaZero 训练项目
基于 OpenSpiel 和 PyTorch 实现的轻量级 AlphaZero
"""

__version__ = "1.0.0"
__author__ = "ironhxs"

from . import config
from .game import DotsAndBoxesGame
from .model import DotsAndBoxesNNet
from .mcts import MCTS
from .train import Trainer

__all__ = [
    'config',
    'DotsAndBoxesGame',
    'DotsAndBoxesNNet',
    'MCTS',
    'Trainer',
]
