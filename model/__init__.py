# -*- coding: utf-8 -*-
"""AlphaZero 模型包"""

from .base_coach import BaseCoach
from .coach import Coach
from .coach_parallel import ParallelCoach
from .game import DotsAndBoxesGame
from .mcts import MCTS
from .arena import Arena

__all__ = [
    'BaseCoach',
    'Coach',
    'ParallelCoach',
    'DotsAndBoxesGame',
    'MCTS',
    'Arena',
]
