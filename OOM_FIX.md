## 🔥 显存不足（OOM）问题分析与修复

### 📊 显存占用计算

#### 单个模型显存（5x5棋盘）
```
模型配置：
- num_filters: 128
- num_res_blocks: 8  
- num_heads: 8

单个模型大小：约 200MB
```

#### 自我对弈显存占用
```
自我对弈: 10个进程 × 1个模型 = 10个模型
显存占用: 10 × 200MB = 2GB
```

#### Arena 显存占用（原配置）
```
Arena: 4个进程 × 2个模型 = 8个模型
显存占用: 8 × 200MB = 1.6GB
```

#### 总显存需求
```
自我对弈 + Arena（如果同时存在）:
2GB + 1.6GB = 3.6GB

但实际上还有：
- 主进程的 self.nnet 和 self.best_nnet: 2 × 200MB = 400MB
- 训练时的梯度和优化器状态: ~500MB
- PyTorch 缓存: ~500MB

总计: 约 5GB
```

### ❌ 为什么测试没问题，训练出错？

#### 测试环境
- 只运行 Arena
- Arena: 4进程 × 2模型 = 8个模型 = 1.6GB
- ✅ 显存充足

#### 训练环境  
- 刚完成自我对弈（可能有残留显存）
- 刚完成训练（梯度、优化器状态未释放）
- 然后进入 Arena
- 主进程还有 2 个模型常驻
- ❌ 显存不足！

### ✅ 已实施的修复

#### 1. 减少 Arena 进程数
```python
# 从 4 减少到 2
'arena_num_workers': 2  # 2进程 × 2模型 = 4个模型 = 800MB
```

#### 2. 自我对弈后清理显存
```python
# coach_alphazero.py
# 自我对弈后
if self.args.get('cuda', False) and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### 3. Arena 前清理显存
```python
# Arena 前
if self.args.get('cuda', False) and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### 4. Arena 后清理显存
```python
# Arena 后
del arena
if self.args.get('cuda', False) and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

#### 5. 子进程结束时清理
```python
# _execute_episode_worker 和 _arena_single_game_worker
del nnet, mcts, ...
if args.get('cuda', False) and torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 📊 修复后的显存占用

```
峰值显存（Arena阶段）:
- 主进程: 2个模型 = 400MB
- Arena: 2进程 × 2模型 = 800MB
- PyTorch缓存: ~500MB
总计: ~1.7GB

相比原来的 5GB，降低了 66%！
```

### 🎯 进一步优化建议

#### 如果仍然 OOM

1. **减少进程数到 1**
   ```python
   'arena_num_workers': 1  # 最保守
   ```

2. **减小模型**
   ```python
   'num_filters': 64,      # 从 128 降到 64
   'num_res_blocks': 4,    # 从 8 降到 4
   ```

3. **使用 CPU Arena**（最后手段）
   ```python
   'arena_mode': 'multiprocess',  # CPU多进程
   'cuda': False,  # Arena不用GPU
   ```

4. **减少自我对弈进程数**
   ```python
   'num_workers': 5,  # 从 10 降到 5
   ```

### 📝 显存监控命令

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看详细信息
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### ✅ 验证修复

运行训练，观察是否还有 OOM：
```bash
python cli/train_alphazero.py
```

如果仍有 OOM，进一步减少 `arena_num_workers` 到 1。
