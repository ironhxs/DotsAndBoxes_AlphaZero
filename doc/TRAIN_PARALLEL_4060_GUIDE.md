# RTX 4060 并行训练指南

## 📋 脚本说明

专为 RTX 4060 (8GB VRAM) 优化的并行训练脚本。

---

## 🎯 三个版本对比

| 脚本 | 用途 | 迭代次数 | MCTS | 进程数 | 时间 | 显存 |
|------|------|---------|------|--------|------|------|
| `test_parallel_quick.py` | 快速验证 | 3 | 25 | 2 | ~5分钟 | <3GB |
| `train_parallel_4060.py` | 完整训练 | 50 | 50 | 4 | ~3小时 | <6GB |
| `train_parallel.py` | 高性能训练 | 600 | 100 | 10 | ~20小时 | 需要更多显存 |

---

## 🚀 快速开始

### 1. 快速测试（推荐先运行）

验证并行训练是否正常工作（约 5 分钟）：

```bash
cd /mnt/d/DotsAndBoxes_AlphaZero
conda activate alphazero
python cli/test_parallel_quick.py
```

**预期输出：**
- ✓ 3 次迭代顺利完成
- ✓ 每次迭代约 1-2 分钟
- ✓ 显存占用 < 3GB
- ✓ GPU 利用率 30-50%

### 2. 4060 优化训练

通过测试后，运行完整训练（约 3 小时）：

```bash
python cli/train_parallel_4060.py
```

**预期结果：**
- 50 次迭代
- 每次约 3-4 分钟
- 显存占用 < 6GB
- GPU 利用率 40-60%

---

## ⚙️ 配置详解

### train_parallel_4060.py 核心配置

```python
# 训练规模
'num_iterations': 50,        # 50 次迭代（适中）
'num_episodes': 40,          # 每次 40 局
'num_simulations': 50,       # MCTS 50 次（平衡）

# 并行配置
'num_workers': 4,            # 4 个自我对弈进程
'arena_num_workers': 4,      # 4 个 Arena 进程
'mcts_batch_size': 16,       # 批量推理 16

# 模型配置
'num_filters': 64,           # 64 通道（轻量）
'num_res_blocks': 4,         # 4 个残差块
'num_heads': 4,              # 4 个注意力头

# 训练参数
'epochs': 5,                 # 5 个 epoch
'batch_size': 128,           # batch=128（省显存）
'use_amp': True,             # 混合精度
```

### 资源占用

| 资源 | 占用量 | 说明 |
|------|--------|------|
| 显存 | < 6GB | 混合精度 + 小模型 |
| CPU | 4-8 核 | 4 个自我对弈进程 + 4 个 Arena 进程 |
| RAM | < 8GB | 训练数据缓存 |
| 磁盘 | ~100MB | 检查点文件 |

---

## 🎮 训练后使用

### 1. 与 AI 对战

```bash
python cli/play_ultimate.py \
    --checkpoint ./results/checkpoints_4060/best_*.pth \
    --mode human \
    --mcts-sims 100
```

### 2. AI 自我对战

```bash
python cli/play_ultimate.py \
    --checkpoint ./results/checkpoints_4060/best_*.pth \
    --mode ai \
    --num-games 10
```

### 3. 双 AI 对战

```bash
python cli/play_ultimate.py \
    --checkpoint1 ./results/checkpoints_4060/best_*.pth \
    --checkpoint2 ./results/checkpoints_4060/latest.pth \
    --mode dual-ai \
    --num-games 20
```

---

## 📊 性能对比

### 串行 vs 并行（4060）

| 指标 | 串行训练 | 并行训练 (4060优化) | 提升 |
|------|---------|-------------------|------|
| GPU 利用率 | 10-20% | 40-60% | **3-4倍** |
| 每次迭代时间 | 8-10 分钟 | 3-4 分钟 | **2-3倍** |
| 50 次迭代总时间 | ~8 小时 | ~3 小时 | **2.5倍** |
| 显存占用 | 2-3GB | 4-6GB | 2倍 |

### 不同配置对比

| 配置 | 进程数 | MCTS | Batch | GPU 利用率 | 显存 | 速度 |
|------|--------|------|-------|-----------|------|------|
| 最小 (快速测试) | 2 | 25 | 8 | 30% | 2GB | 基准 |
| 4060 优化 | 4 | 50 | 16 | 50% | 5GB | 2倍 |
| 高性能 | 10 | 100 | 32 | 70% | >8GB | 4倍（需更大显存）|

---

## ⚠️ 注意事项

### 1. 显存管理

**如果遇到 OOM (Out of Memory)：**

```python
# 在 train_parallel_4060.py 中降低这些参数：

'num_workers': 4 → 2              # 减少进程数
'arena_num_workers': 4 → 2        # 减少 Arena 进程
'mcts_batch_size': 16 → 8         # 减少批量大小
'batch_size': 128 → 64            # 减少训练批量
```

### 2. CPU 占用

如果 CPU 过载（温度过高）：

```python
'num_workers': 4 → 2              # 减少并行进程
os.environ['OMP_NUM_THREADS'] = '4' → '2'  # 减少线程数
```

### 3. 监控工具

**实时监控 GPU：**
```bash
# 终端 1: 运行训练
python cli/train_parallel_4060.py

# 终端 2: 监控 GPU
watch -n 1 nvidia-smi
```

**实时监控 CPU：**
```bash
htop  # 或 top
```

---

## 🔧 进阶调优

### 提升 GPU 利用率

如果 GPU 利用率 < 40%，可以增加：

```python
'num_workers': 4 → 6              # 更多进程
'mcts_batch_size': 16 → 24        # 更大批量
'parallel_games': 4 → 6           # 更多并行游戏
```

### 提升训练质量

如果想要更强的模型：

```python
'num_simulations': 50 → 100       # 更多 MCTS
'arena_mcts_simulations': 100 → 200  # 更准确的评估
'epochs': 5 → 10                  # 更多训练轮数
```

### 加快训练速度

如果只是快速实验：

```python
'num_episodes': 40 → 20           # 更少对局
'num_simulations': 50 → 25        # 更少 MCTS
'epochs': 5 → 2                   # 更少 epoch
```

---

## 🐛 常见问题

### Q1: RuntimeError: CUDA out of memory

**解决方法：**
1. 降低 `num_workers` 和 `arena_num_workers`
2. 降低 `mcts_batch_size` 和 `batch_size`
3. 关闭其他占用显存的程序
4. 使用 `test_parallel_quick.py` 验证配置

### Q2: 进程卡住不动

**解决方法：**
1. 检查是否有僵尸进程：`ps aux | grep python`
2. 杀掉所有 Python 进程：`pkill -9 python`
3. 重新运行

### Q3: GPU 利用率很低

**可能原因：**
1. CPU 成为瓶颈 → 增加 `num_workers`
2. MCTS 模拟次数太少 → 增加 `num_simulations`
3. 批量大小太小 → 增加 `mcts_batch_size`

### Q4: 训练速度比预期慢

**检查项：**
1. 是否启用了混合精度：`use_amp: True`
2. 是否启用了 cudnn benchmark：`torch.backends.cudnn.benchmark = True`
3. 是否有其他程序占用 GPU
4. MCTS 模拟次数是否过高

---

## 📈 预期训练曲线

### 正常训练表现

| 迭代 | 平均回合长度 | 策略损失 | 价值损失 | Arena 胜率 |
|------|------------|---------|---------|-----------|
| 1-10 | 40-50 | 3.0-4.0 | 1.5-2.0 | - |
| 11-20 | 35-45 | 2.5-3.5 | 1.0-1.5 | 50-60% |
| 21-30 | 30-40 | 2.0-3.0 | 0.8-1.2 | 55-65% |
| 31-50 | 25-35 | 1.5-2.5 | 0.6-1.0 | 60-70% |

### 异常情况

- ❌ 损失不下降 → 学习率太高或太低
- ❌ 回合长度不变 → MCTS 探索不足
- ❌ Arena 胜率 < 50% → 模型没有进步，可能过拟合

---

## 🎯 下一步

训练完成后：

1. **评估模型强度**
   ```bash
   python cli/evaluate_model.py --checkpoint ./results/checkpoints_4060/best_*.pth
   ```

2. **与人类对战**
   ```bash
   python cli/play_ultimate.py --mode human
   ```

3. **继续训练**
   - 修改 `num_iterations`: 50 → 100
   - 或使用更大的 MCTS: 50 → 100

4. **升级到高性能版**
   - 如果有更大显存的 GPU，使用 `train_parallel.py`

---

## 📝 总结

- ✅ **test_parallel_quick.py**: 5 分钟快速验证
- ✅ **train_parallel_4060.py**: 3 小时完整训练（推荐）
- ✅ **train_parallel.py**: 20 小时高性能训练（需更大显存）

**推荐流程：**
1. 先运行 `test_parallel_quick.py` 验证环境
2. 通过后运行 `train_parallel_4060.py` 完整训练
3. 根据结果调整参数或升级配置

Happy Training! 🎉
