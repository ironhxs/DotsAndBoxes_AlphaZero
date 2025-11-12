# TensorBoard 使用指南

## 启动 TensorBoard

训练过程会自动记录到 TensorBoard（已在 `config/config.yaml` 中启用）。

### 方法 1：使用启动脚本（推荐）
```bash
./cli/tensorboard.sh
```

### 方法 2：手动启动
```bash
tensorboard --logdir results/logs --port 6006
```

然后在浏览器打开：http://localhost:6006

## 监控指标

### 1. Loss（损失函数）
- **Loss/Policy**: 策略网络损失（交叉熵）
- **Loss/Value**: 价值网络损失（MSE）
- **Loss/Total**: 总损失

### 2. Training（训练过程）
- **Training/LearningRate**: 学习率变化（余弦退火）

### 3. Data（数据统计）
- **Data/IterationSamples**: 每次迭代生成的样本数
- **Data/TotalSamples**: 训练集总样本数（滑动窗口，最多保留20次迭代）

### 4. Arena（模型对战）
- **Arena/WinRate**: 新模型 vs 旧模型的胜率
- **Arena/Accepted**: 新模型是否被接受（1=接受，0=拒绝）

## 配置选项

在 `config/config.yaml` 中：

```yaml
# 启用/禁用 TensorBoard
tensorboard: true

# 日志保存目录
log_dir: results/logs
```

## 查看历史训练

如果要查看之前的训练记录：

```bash
tensorboard --logdir results/logs --port 6006
```

TensorBoard 会自动加载目录下的所有事件文件。

## 清除日志

如果要重新开始记录（删除旧日志）：

```bash
rm -rf results/logs/*
```

## 多次训练对比

TensorBoard 支持在同一个图表中显示多次训练的曲线，方便对比不同超参数的效果。
