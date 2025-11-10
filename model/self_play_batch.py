# -*- coding: utf-8 -*-
"""自我对弈 - 批量推理版本（生产者-消费者模式）"""

import numpy as np
import torch
from tqdm import tqdm
from .mcts_batch import MCTSBatchInference
from .batch_inference_server import BatchInferenceServer
from concurrent.futures import ThreadPoolExecutor, as_completed


def execute_episode_batch(game, nnet, args, inference_server, model_idx=0):
    """
    执行一局自我对弈（使用批量推理）
    
    Args:
        game: 游戏实例
        nnet: 神经网络（仅用于获取参数信息，不直接推理）
        args: 配置参数
        inference_server: 批量推理服务器
        model_idx: 模型索引（总是0，因为自我对弈只用一个模型）
    
    Returns:
        训练样本列表
    """
    mcts = MCTSBatchInference(game, nnet, args)
    mcts.set_inference_server(inference_server, model_idx)
    
    train_examples = []
    state = game.get_initial_state()
    cur_player = 0
    episode_step = 0
    
    while True:
        episode_step += 1
        canonical_board = game.get_observation(state)
        temp = int(episode_step < args['temp_threshold'])
        pi = mcts.get_action_prob(state, temp=temp)
        
        # 前30步添加Dirichlet噪声增加探索
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
            # 游戏结束，返回带标签的训练样本
            return [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples]
        
        new_player = game.get_current_player(state)
        if new_player != cur_player:
            cur_player = new_player


def self_play_parallel_batch(game, nnet, args):
    """
    并行自我对弈（生产者-消费者模式）
    
    ⚠️ 问题：Python GIL导致多线程无法利用多核CPU
    解决：改回多进程，但每个进程共享GPU推理服务
    
    暂时fallback到原始多进程版本
    """
    num_episodes = args['num_episodes']
    num_workers = args.get('num_workers', 10)
    
    print(f"\n⚠️  批量推理模式受GIL限制，自动切换到多进程模式")
    print(f"   架构: {num_workers} 个CPU进程（各自GPU推理）")
    print(f"   显存: {num_workers}个模型 (~{num_workers*200}MB)")
    
    # Fallback到原始多进程实现
    from multiprocessing import Pool
    from .coach_alphazero import _execute_episode_worker
    
    nnet_state = nnet.state_dict()
    tasks = [
        (game, nnet_state, args, np.random.randint(0, 1000000))
        for _ in range(num_episodes)
    ]
    
    all_train_examples = []
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(_execute_episode_worker, tasks),
            total=num_episodes,
            desc=f"🎮 自我对弈({num_workers}进程)"
        ))
        for result in results:
            all_train_examples.extend(result)
    
    print(f"✓ 收集到 {len(all_train_examples)} 个训练样本")
    
    return all_train_examples
