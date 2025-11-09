#!/bin/bash
# GPU 监控脚本

echo "🔍 实时监控 GPU 利用率（每秒刷新）"
echo "按 Ctrl+C 停止"
echo ""

while true; do
    clear
    echo "========================================"
    echo "  GPU 利用率监控 - $(date +%H:%M:%S)"
    echo "========================================"
    
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "🎮 GPU %s: %s\n", $1, $2
        printf "   GPU 利用率: %3s%% ", $3
        if ($3 < 20) printf "❌ 太低\n"
        else if ($3 < 50) printf "⚠️  偏低\n"
        else if ($3 < 80) printf "✅ 良好\n"
        else printf "🔥 高负载\n"
        
        printf "   显存利用率: %3s%% ", $4
        if ($4 < 20) printf "❌ 太低\n"
        else if ($4 < 50) printf "⚠️  偏低\n"
        else if ($4 < 80) printf "✅ 良好\n"
        else printf "🔥 高负载\n"
        
        printf "   显存使用: %s / %s MB\n", $5, $6
        printf "   温度: %s°C | 功耗: %s W\n", $7, $8
    }'
    
    echo ""
    echo "========================================"
    echo "正在运行的进程:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
    awk -F', ' '{printf "  PID %s: %s (%s MB)\n", $1, $2, $3}' || echo "  无GPU进程运行"
    
    sleep 1
done
