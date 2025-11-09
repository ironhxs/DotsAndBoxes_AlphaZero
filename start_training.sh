#!/bin/bash
# AlphaZero Dots and Boxes Training Script

PYTHON="/root/miniconda3/envs/gmd/bin/python"
PROJECT_DIR="/HFUT_002/DotsAndBoxes_AlphaZero"

cd "$PROJECT_DIR" || exit 1

echo "========================================"
echo "  AlphaZero Training System"
echo "========================================"
echo ""
echo "Options:"
echo ""
echo "  1) AlphaZero Full Training (Recommended)"
echo "     - Arena validation every 20 iterations"
echo "     - New model accepted if >55% win rate"
echo "     - 600 iterations x 80 games x 300 epochs"
echo "     - About 3-5 min/iter, long-term training"
echo ""
echo "  2) Extreme Fast Training (Experimental)"
echo "     - No Arena validation, faster"
echo "     - 6 parallel processes, GPU 70%+"
echo "     - May cause model degradation"
echo ""
echo "  3) Evaluate Model"
echo "     - vs Random (expect >90%)"
echo "     - vs Greedy (expect >70%)"
echo "     - vs Old Model (expect >60%)"
echo ""
echo "  4) Play Against AI"
echo ""
echo "  5) GPU Monitor"
echo ""
echo "  6) Environment Test"
echo ""
echo "  0) Exit"
echo ""
echo "========================================"
read -p "Enter option [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "========================================"
        echo "  AlphaZero Full Training"
        echo "========================================"
        echo "Configuration:"
        echo "  - Iterations: 600 (long-term training)"
        echo "  - Games per iteration: 80 (improved GPU usage)"
        echo "  - Training epochs: 300 (deep learning)"
        echo "  - Arena validation: every 20 iterations"
        echo "  - Model selection: win rate >55%"
        echo ""
        echo "Estimated time: 3-5 min/iter (Ctrl+C to stop anytime)"
        echo "========================================"
        echo ""
        read -p "Confirm start? [Y/n]: " confirm
        if [[ -z $confirm || $confirm == [yY] ]]; then
            echo ""
            echo "Starting training..."
            echo ""
            $PYTHON cli/train_alphazero.py
        else
            echo "Cancelled"
        fi
        ;;
    2)
        echo ""
        echo "========================================"
        echo "  Extreme Fast Training"
        echo "========================================"
        echo "Warning: No Arena validation"
        echo "  - Fast training (3 min/iter)"
        echo "  - May cause model degradation"
        echo "  - For experiments only"
        echo ""
        echo "Use option 1 for serious training!"
        echo "========================================"
        echo ""
        read -p "Continue anyway? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            $PYTHON cli/extreme_train.py
        else
            echo "Cancelled. Use option 1 instead"
        fi
        ;;
    3)
        echo ""
        echo "========================================"
        echo "  Model Evaluation"
        echo "========================================"
        if [ ! -f "results/checkpoints/latest.pth" ]; then
            echo "Error: No trained model found"
            echo "Please run option 1 first"
        else
            echo "Test items:"
            echo "  1. vs Random (expect >90% win rate)"
            echo "  2. vs Greedy (expect >70% win rate)"
            echo "  3. vs Old Model (expect >60% win rate)"
            echo ""
            read -p "Full test (40 games) or Quick test (10 games)? [F/q]: " test_mode
            echo ""
            if [[ $test_mode == [qQ] ]]; then
                $PYTHON cli/evaluate_model.py quick
            else
                $PYTHON cli/evaluate_model.py
            fi
        fi
        ;;
    4)
        echo ""
        echo "========================================"
        echo "  Play Against AI"
        echo "========================================"
        if [ ! -f "results/checkpoints/latest.pth" ]; then
            echo "Error: No trained model found"
            echo "Please run option 1 first"
        else
            $PYTHON cli/play.py
        fi
        ;;
    5)
        echo ""
        echo "========================================"
        echo "  GPU Monitor"
        echo "========================================"
        echo "Select monitor mode:"
        echo "  1) Detailed real-time (Recommended)"
        echo "  2) Quick view"
        echo ""
        read -p "Choose [1/2]: " monitor_choice
        echo ""
        if [[ $monitor_choice == "2" ]]; then
            ./cli/quick_monitor.sh
        else
            ./cli/monitor_gpu.sh
        fi
        ;;
    6)
        echo ""
        echo "========================================"
        echo "  Environment Test"
        echo "========================================"
        $PYTHON cli/test_project.py
        ;;
    0)
        echo ""
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
