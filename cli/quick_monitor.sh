#!/bin/bash
# 快速GPU监控 - 单次输出

echo "================================================"
echo "   GPU 状态监控 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader | while IFS=', ' read name gpu_util mem_util mem_used mem_total temp power; do
    
    echo "🎮 GPU: $name"
    echo "   ├─ GPU利用率: ${gpu_util}%"
    echo "   ├─ 显存利用率: ${mem_util}%"
    echo "   ├─ 显存使用: ${mem_used} / ${mem_total}"
    echo "   ├─ 温度: ${temp}°C"
    echo "   └─ 功耗: ${power}W"
done

echo ""
echo "📊 运行中的训练进程:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
    awk -F', ' '{printf "   PID %s: %s (显存: %s MB)\n", $1, $2, $3}' || echo "   无GPU进程"

echo "================================================"
