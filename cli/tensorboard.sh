#!/bin/bash
# 启动 TensorBoard 查看训练日志

cd /HFUT_002/DotsAndBoxes_AlphaZero

LOG_DIR="results/logs/tensorboard"

echo "========================================="
echo "  TensorBoard 可视化"
echo "========================================="
echo ""
echo "日志目录: $LOG_DIR"
echo ""

# 检查日志目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo "⚠️  日志目录不存在，请先开始训练"
    exit 1
fi

# 检查是否有日志文件
if [ -z "$(ls -A $LOG_DIR)" ]; then
    echo "⚠️  日志目录为空，请先开始训练"
    exit 1
fi

echo "🚀 启动 TensorBoard..."
echo ""
echo "访问地址:"
echo "  本地: http://localhost:6006"
echo "  远程: http://$(hostname -I | awk '{print $1}'):6006"
echo ""
echo "按 Ctrl+C 停止"
echo "========================================="
echo ""

# 启动 TensorBoard
tensorboard --logdir=$LOG_DIR --host=0.0.0.0 --port=6006
