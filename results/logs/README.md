# 训练日志目录

## TensorBoard 可视化

### 启动 TensorBoard

```bash
# 方法1: 使用脚本
./cli/tensorboard.sh

# 方法2: 直接命令
tensorboard --logdir=results/logs/tensorboard --host=0.0.0.0 --port=6006
```

### 访问地址

- **本地**: http://localhost:6006
- **远程**: http://YOUR_SERVER_IP:6006

### 记录的指标

#### 1. **Loss (损失)**
- `Loss/policy` - 策略网络损失 (π)
        echo "" (v)
- `Loss/total` - 总损失

#### 2. Brain-Tumor-Segmentation DotsAndBoxes_AlphaZero download multirun outputs results )**
- `Arena/win_rate` - 新模型胜率
- `Arena/new_wins` - 新模型获胜局数
- `Arena/old_wins` - 旧模型获胜局数
- `Arena/draws` - 平局数
- `Arena/model_accepted` - 模型是否被接受 (1=接受, 0=拒绝)

#### 3. **Training (训练)**
- `Training/speed_batches_per_sec` - 训练速度 (批次/秒)

### 使用技巧

1. **对比多次训练**: TensorBoard 会自动识别不同的运行
2. **平滑曲线**: 
3. **选择指标**: 左侧可以选择显示/隐藏特定指标
4. **缩放**: 点击图表可以放大查看细节

### 目录结构

```
results/logs/
 tensorboard/        # TensorBoard 事件文件
   └── events.out.tfevents.*
 README.md          # 本说明文档
```

### 清理旧日志

```bash
# 删除旧的 TensorBoard 日志
rm -rf results/logs/tensorboard/*
```

### 问题排查

**端口被占用**:
```bash
# 查找占用端口的进程
lsof -i:6006

# 使用其他端口
tensorboard --logdir=results/logs/tensorboard --port=6007
```

**无法访问**:
- 检查防火墙设置
- 确保使用 `--host=0.0.0.0` 允许远程访问
- 检查服务器 IP 和端口是否正确
