# -*- coding: utf-8 -*-
"""
AlphaZero Model Package
=======================

Core Components:
- DotsAndBoxesGame: OpenSpiel game wrapper with 9-channel observation
- DotsAndBoxesNet: ResNet with SE attention blocks
- MCTS: Monte Carlo Tree Search with neural network guidance
- Arena: Model evaluation through self-play tournaments
- Coach/ParallelCoach: Training orchestration

Parallel Self-Play Engines:
- mcts_full_parallel_gpu: Independent GPU model per worker (highest GPU utilization)
- mcts_multiprocess_gpu: Shared GPU via inference server (saves VRAM)
- mcts_concurrent_gpu: Single-process async games (debugging)
"""

from .game import DotsAndBoxesGame
from .model import DotsAndBoxesNet
from .mcts import MCTS
from .arena import Arena
from .base_coach import BaseCoach
from .coach import Coach
from .coach_parallel import ParallelCoach

__all__ = [
    'DotsAndBoxesGame',
    'DotsAndBoxesNet',
    'MCTS',
    'Arena',
    'BaseCoach',
    'Coach',
    'ParallelCoach',
]
