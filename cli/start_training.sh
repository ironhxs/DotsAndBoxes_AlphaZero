#!/bin/bash
# AlphaZero 点格棋'SCRIPT_EOF' - 一键启动脚本

PYTHON="/root/miniconda3/envs/gmd/bin/python"
PROJECT_DIR="/HFUT_002/DotsAndBoxes_AlphaZero"

cd $PROJECT_DIR

echo "========================================"
echo "  🧠 AlphaZero 点格棋训练系统"
echo "========================================"
echo ""
echo "请选择训练模式:"
echo ""
echo "  1) ✅ AlphaZero 完整训练 (推荐)"
echo "     - 包含 Arena 对战验证"
echo "     - 新模型>55%胜率才接受"
echo "     - 20次迭代, 40局/次"
echo "     - 每次迭代约15分钟"
echo ""
echo "  2) ⚡ 快速训练 (无)"
echo "     - 纯自我对弈+训练"
echo "     - 无Arena对战"
echo "     - 速度快但可能过拟合"
echo ""
echo "  "
echo "     - vs 随机策略"
echo "     - vs 贪心策略"
echo "     - vs 早期模型"
echo ""
echo "  4) 🎮 人机对战"
echo "     - 与训练好的AI下棋"
echo ""
echo "  5) 📊 GPU监控"
echo ""
echo "  0) 退出"
echo ""
echo "========================================"
read -p "请输入选项 [0-5]: " choice

case $choice in
    1)
        echo ""
        echo "✅ 启动 AlphaZero 完整训练..."
        echo ""
        echo "训练流程:"
scipy:              要求 1.7. → 训练网络 → Arena对战 → 模型筛选"
        echo ""
        read -p "确认开始? [Y/n]: " confirm
        if [[ -z $confirm || $confirm == [yY] ]]; then
            echo ""
            cd .. && $PYTHON cli/train_alphazero.py
        else
            echo "已取消"
        fi
        ;;
    2)
        echo ""
        echo "⚡ 启动快'SCRIPT_EOF' (无验证)..."
        echo ""
        echo "⚠️  警告: 此模式无Arena验证，可能导致模型退化"
        echo "   建议使用选项1的完'SCRIPT_EOF'"
        echo ""
        read -p "确认使用快速模式? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            cd .. && $PYTHON cli/extreme_train.py
        else
            echo "已取消，请选择选项1"
        fi
        ;;
    3)
        echo ""
        ..."
        echo ""
        echo "测试项目:"
        echo "  1. vs 随机策略 (应>90%胜率)"
        echo "  2. vs 贪心策略 (应>70%胜率)"
        echo "  3. vs 早期模型 (应>60%胜率)"
        echo ""
        read -p "完整测试(40局)还是快速测试(10局)? [F/q]: " test_mode
        if [[ $test_mode == [qQ] ]]; then
            cd .. && $PYTHON cli/evaluate_model.py quick
        else
            cd .. && $PYTHON cli/evaluate_model.py
        fi
        ;;
    4)
        echo ""
        echo "🎮 启动人机对战..."
        if [ ! -f "checkpoints/latest.pth" ]; then
            echo "❌ 错误: 未找到训练好的模型"
            'SCRIPT_EOF'"
        else
            cd .. && $PYTHON cli/play.py
        fi
        ;;
    5)
        echo ""
        echo "📊 启动GPU监控..."
        echo "   (按 Ctrl+C 退出)"
        echo ""
        ./monitor_gpu.sh
        ;;
    0)
        echo "再见!"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac
