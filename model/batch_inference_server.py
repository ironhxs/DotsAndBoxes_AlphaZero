# -*- coding: utf-8 -*-
"""
批量推理服务器 - 解决 GPU 利用率低的问题
使用生产者-消费者模式，批量处理多个 MCTS 的推理请求
"""

import torch
import torch.multiprocessing as mp
from queue import Queue, Empty
from threading import Thread, Lock
import time
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: int
    observation: np.ndarray
    result_queue: Queue


class BatchInferenceServer:
    """
    批量推理服务器
    
    解决问题：
    1. MCTS 每次调用神经网络时，batch_size=1，GPU 利用率极低
    2. 自我对弈时有多个游戏并行，可以批量推理
    
    工作原理：
    1. 多个 MCTS 将推理请求放入队列
    2. 服务器收集一批请求（或超时）
    3. 批量前向传播
    4. 将结果返回给各个 MCTS
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 32,
        timeout: float = 0.01,  # 10ms 超时
        device: torch.device = None
    ):
        """
        初始化批量推理服务器
        
        Args:
            model: 神经网络模型
            batch_size: 最大批量大小
            timeout: 收集请求的超时时间（秒）
            device: 计算设备
        """
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备
        self.model.to(self.device)
        self.model.eval()
        
        # 请求队列
        self.request_queue = Queue()
        
        # 服务器状态
        self.running = False
        self.server_thread = None
        self.lock = Lock()
        
        # 统计信息
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0
        
        logger.info(f"批量推理服务器初始化: batch_size={batch_size}, "
                   f"timeout={timeout}, device={self.device}")
    
    def start(self):
        """启动服务器"""
        with self.lock:
            if self.running:
                logger.warning("服务器已经在运行")
                return
            
            self.running = True
            self.server_thread = Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            logger.info("批量推理服务器已启动")
    
    def stop(self):
        """停止服务器"""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            
            logger.info("批量推理服务器已停止")
            self._print_statistics()
    
    def _server_loop(self):
        """服务器主循环"""
        logger.info("服务器主循环开始")
        
        while self.running:
            try:
                # 收集一批请求
                batch_requests = self._collect_batch()
                
                if len(batch_requests) == 0:
                    time.sleep(0.001)  # 短暂休眠
                    continue
                
                # 批量推理
                self._process_batch(batch_requests)
            
            except Exception as e:
                logger.error(f"服务器循环错误: {e}", exc_info=True)
                time.sleep(0.01)
        
        logger.info("服务器主循环结束")
    
    def _collect_batch(self) -> List[InferenceRequest]:
        """
        收集一批推理请求
        
        策略：
        1. 等待第一个请求（阻塞）
        2. 快速收集更多请求直到 batch_size 或超时
        """
        batch = []
        start_time = time.time()
        
        # 等待第一个请求（最多等待 timeout）
        try:
            first_request = self.request_queue.get(timeout=self.timeout)
            batch.append(first_request)
        except Empty:
            return batch
        
        # 快速收集更多请求
        while len(batch) < self.batch_size:
            elapsed = time.time() - start_time
            remaining = self.timeout - elapsed
            
            if remaining <= 0:
                break
            
            try:
                request = self.request_queue.get(timeout=remaining)
                batch.append(request)
            except Empty:
                break
        
        return batch
    
    def _process_batch(self, batch_requests: List[InferenceRequest]):
        """
        批量处理推理请求
        
        Args:
            batch_requests: 推理请求列表
        """
        try:
            batch_size = len(batch_requests)
            
            # 准备批量输入
            observations = np.array([req.observation for req in batch_requests])
            obs_tensor = torch.FloatTensor(observations).to(self.device)
            
            # 批量推理
            start_time = time.time()
            with torch.no_grad():
                log_pi_batch, v_batch = self.model(obs_tensor)
            
            inference_time = time.time() - start_time
            
            # 转换结果
            pi_batch = torch.exp(log_pi_batch).cpu().numpy()
            v_batch = v_batch.cpu().numpy().flatten()
            
            # 分发结果
            for i, request in enumerate(batch_requests):
                try:
                    request.result_queue.put((pi_batch[i], v_batch[i]))
                except Exception as e:
                    logger.error(f"分发结果失败: {e}")
            
            # 更新统计
            self.total_requests += batch_size
            self.total_batches += 1
            self.total_inference_time += inference_time
            
            # 定期打印统计
            if self.total_batches % 100 == 0:
                avg_batch_size = self.total_requests / self.total_batches
                avg_inference_time = self.total_inference_time / self.total_batches * 1000
                logger.debug(f"批量推理统计: batches={self.total_batches}, "
                            f"avg_batch_size={avg_batch_size:.1f}, "
                            f"avg_time={avg_inference_time:.2f}ms")
        
        except Exception as e:
            logger.error(f"批量推理错误: {e}", exc_info=True)
            # 向所有请求返回错误
            for request in batch_requests:
                try:
                    request.result_queue.put(None)
                except:
                    pass
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        推理接口（供 MCTS 调用）
        
        Args:
            observation: 观察 (C, H, W)
            
        Returns:
            pi: 策略概率 (action_size,)
            v: 价值估计 (float)
        """
        if not self.running:
            raise RuntimeError("批量推理服务器未运行")
        
        # 创建结果队列
        result_queue = Queue(maxsize=1)
        
        # 创建请求
        request = InferenceRequest(
            request_id=self.total_requests,
            observation=observation,
            result_queue=result_queue
        )
        
        # 提交请求
        self.request_queue.put(request)
        
        # 等待结果（最多等待 1 秒）
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:
                raise RuntimeError("推理失败")
            return result
        except Empty:
            raise TimeoutError("推理超时")
    
    def update_model(self, new_state_dict: dict):
        """
        更新模型参数（在训练迭代之间）
        
        Args:
            new_state_dict: 新的模型权重
        """
        with self.lock:
            self.model.load_state_dict(new_state_dict)
            logger.info("模型参数已更新")
    
    def _print_statistics(self):
        """打印统计信息"""
        if self.total_batches > 0:
            avg_batch_size = self.total_requests / self.total_batches
            avg_inference_time = self.total_inference_time / self.total_batches * 1000
            throughput = self.total_requests / max(self.total_inference_time, 1e-6)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"批量推理统计:")
            logger.info(f"  总请求数: {self.total_requests}")
            logger.info(f"  总批次数: {self.total_batches}")
            logger.info(f"  平均批量大小: {avg_batch_size:.2f}")
            logger.info(f"  平均推理时间: {avg_inference_time:.2f} ms/batch")
            logger.info(f"  吞吐量: {throughput:.1f} requests/sec")
            logger.info(f"{'='*60}\n")
    
    def __enter__(self):
        """上下文管理器支持"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.stop()


class BatchMCTS:
    """
    支持批量推理的 MCTS
    
    用法：
        server = BatchInferenceServer(model)
        server.start()
        
        mcts = BatchMCTS(game, server, args)
        pi = mcts.get_action_prob(state)
        
        server.stop()
    """
    
    def __init__(self, game, inference_server: BatchInferenceServer, args):
        """
        初始化 MCTS
        
        Args:
            game: 游戏环境
            inference_server: 批量推理服务器
            args: MCTS 参数
        """
        from .mcts import MCTS
        
        # 使用标准 MCTS，但替换推理函数
        self.mcts = MCTS(game, None, args)  # nnet=None
        self.inference_server = inference_server
        self.game = game
        
        # 替换 MCTS 的推理逻辑
        self._patch_mcts()
    
    def _patch_mcts(self):
        """修补 MCTS 使用批量推理服务器"""
        original_search = self.mcts.search
        
        def patched_search(state):
            # 大部分逻辑与原始 MCTS 相同
            # 只在需要神经网络时调用批量推理服务器
            s = str(state)
            
            if state.is_terminal():
                returns = state.returns()
                if len(returns) >= 2:
                    result = 1.0 if returns[0] > returns[1] else (-1.0 if returns[0] < returns[1] else 0.0)
                    return result
                return 0.0
            
            if s not in self.mcts.Ps:
                # 使用批量推理服务器
                obs = self.game.get_observation(state)
                pi, v = self.inference_server.predict(obs)
                
                # 屏蔽非法动作
                valids = self.game.get_valid_moves(state)
                pi = pi * valids
                sum_pi = np.sum(pi)
                if sum_pi > 0:
                    pi /= sum_pi
                else:
                    pi = valids / np.sum(valids)
                
                self.mcts.Ps[s] = pi
                self.mcts.Ns[s] = 0
                return -v
            
            # 其余逻辑与标准 MCTS 相同...
            return original_search(state)
        
        self.mcts.search = patched_search
    
    def get_action_prob(self, state, temp=1):
        """获取动作概率（委托给内部 MCTS）"""
        return self.mcts.get_action_prob(state, temp)
    
    def reset(self):
        """重置 MCTS"""
        self.mcts.reset()


if __name__ == "__main__":
    # 测试批量推理服务器
    import sys
    sys.path.append('..')
    from model.game import DotsAndBoxesGame
    from model.model import DotsAndBoxesNet
    
    # 创建游戏和模型
    game = DotsAndBoxesGame(num_rows=5, num_cols=5)
    model = DotsAndBoxesNet(game, num_filters=64, num_res_blocks=5)
    
    # 创建批量推理服务器
    server = BatchInferenceServer(model, batch_size=16, timeout=0.05)
    
    # 测试
    print("启动服务器...")
    server.start()
    
    # 模拟多个 MCTS 并发请求
    import threading
    
    def test_worker(worker_id):
        state = game.get_initial_state()
        obs = game.get_observation(state)
        
        for i in range(10):
            pi, v = server.predict(obs)
            print(f"Worker {worker_id}, request {i}: v={v:.3f}")
            time.sleep(0.01)
    
    # 启动多个工作线程
    threads = [threading.Thread(target=test_worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # 停止服务器
    print("\n停止服务器...")
    server.stop()
