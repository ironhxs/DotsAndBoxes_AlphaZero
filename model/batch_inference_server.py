# -*- coding: utf-8 -*-
"""批量推理服务器 - GPU高效推理的核心"""

import torch
import numpy as np
from queue import Queue, Empty
from threading import Thread, Event
import time


class BatchInferenceServer:
    """
    批量推理服务器（生产者-消费者模式）
    
    工作流程：
    1. 多个MCTS线程提交推理请求到队列（生产者）
    2. GPU线程批量处理请求并返回结果（消费者）
    3. 自动批处理，提高GPU利用率
    """
    
    def __init__(self, models, use_cuda=True, batch_size=32, timeout=0.01):
        """
        Args:
            models: 模型列表 [player1_model, player2_model]
            use_cuda: 是否使用GPU
            batch_size: 最大批量大小
            timeout: 批处理超时时间（秒）
        """
        self.models = models
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.batch_size = batch_size
        self.timeout = timeout
        
        # 为每个模型创建请求队列
        self.request_queues = [Queue() for _ in models]
        
        # GPU线程
        self.inference_threads = []
        self.stop_event = Event()
        
        # 将模型移到GPU
        if self.use_cuda:
            for model in self.models:
                model.cuda()
                model.eval()
    
    def start(self):
        """启动推理服务"""
        for model_idx, model in enumerate(self.models):
            thread = Thread(
                target=self._inference_worker,
                args=(model_idx, model),
                daemon=True
            )
            thread.start()
            self.inference_threads.append(thread)
    
    def stop(self):
        """停止推理服务"""
        self.stop_event.set()
        for thread in self.inference_threads:
            thread.join(timeout=1.0)
    
    def predict(self, model_idx, obs):
        """
        提交推理请求（由MCTS线程调用）
        
        Args:
            model_idx: 模型索引 (0=player1, 1=player2)
            obs: 观察张量 (已经是numpy array)
        
        Returns:
            (policy, value): 推理结果
        """
        # 创建结果容器
        result_queue = Queue(maxsize=1)
        
        # 提交请求
        self.request_queues[model_idx].put((obs, result_queue))
        
        # 等待结果
        policy, value = result_queue.get()
        return policy, value
    
    def _inference_worker(self, model_idx, model):
        """
        GPU推理工作线程
        批量处理请求，提高GPU利用率
        """
        while not self.stop_event.is_set():
            # 收集一批请求
            batch_requests = []
            deadline = time.time() + self.timeout
            
            while len(batch_requests) < self.batch_size:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    break
                
                try:
                    request = self.request_queues[model_idx].get(timeout=remaining_time)
                    batch_requests.append(request)
                except Empty:
                    break
            
            if not batch_requests:
                time.sleep(0.001)  # 短暂休眠避免空转
                continue
            
            # 批量推理
            obs_list = [req[0] for req in batch_requests]
            result_queues = [req[1] for req in batch_requests]
            
            try:
                # 堆叠成批量
                obs_batch = np.stack(obs_list, axis=0)
                obs_tensor = torch.FloatTensor(obs_batch)
                
                if self.use_cuda:
                    obs_tensor = obs_tensor.cuda()
                
                # GPU推理
                with torch.no_grad():
                    pi_batch, v_batch = model(obs_tensor)
                
                # 转换回numpy
                pi_batch = torch.exp(pi_batch).cpu().numpy()
                v_batch = v_batch.cpu().numpy()
                
                # 分发结果
                for i, result_queue in enumerate(result_queues):
                    result_queue.put((pi_batch[i], v_batch[i][0]))
                    
            except Exception as e:
                print(f"⚠️  批量推理错误: {e}")
                # 错误情况下返回默认值
                for result_queue in result_queues:
                    result_queue.put((np.zeros(60), 0.0))
