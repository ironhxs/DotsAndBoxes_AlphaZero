#!/bin/bash
# 启动 TensorBoard 监控训练进度

echo "启动 TensorBoard..."
echo "访问地址: http://localhost:6006"
echo ""

tensorboard --logdir results/logs --port 6006
