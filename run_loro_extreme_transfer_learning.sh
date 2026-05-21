#!/bin/bash
# LORO Extreme Transfer Learning Experiment
#
# Flow:
# 1. Use LORO_Extreme pretrained model (trained on extreme-filtered 11 receptors)
# 2. Split extreme test data (30% fine-tune, 70% test)
# 3. Fine-tune pretrained model on target receptor
#
# Data:   datasets/cnnscore_database/loro_extreme/fold_X/test.csv
# Model:  result/LORO_Extreme/fold_X/best_model_epoch_*.pth
# Output: result/LORO_Extreme_Transfer/fold_X/
#
# Usage:
#   ./run_loro_extreme_transfer_learning.sh <fold> <gpu_id> [finetune_ratio]
#   ./run_loro_extreme_transfer_learning.sh all <gpu_id> [finetune_ratio]
#
# Examples:
#   ./run_loro_extreme_transfer_learning.sh fold_1 0 0.3
#   ./run_loro_extreme_transfer_learning.sh all 0 0.3

set -e

FOLD=${1:-fold_1}
GPU_ID=${2:-0}
FINETUNE_RATIO=${3:-0.3}

run_single_fold() {
    local fold=$1
    local gpu=$2
    local ratio=$3

    echo "============================================================"
    echo "LORO Extreme Transfer Learning - $fold"
    echo "============================================================"
    echo "GPU: $gpu"
    echo "Fine-tune ratio: $ratio"
    echo "============================================================"

    # Step 1: Prepare transfer learning data from extreme LORO
    echo ""
    echo "[Step 1] Preparing transfer learning data from extreme LORO..."
    python prepare_loro_transfer_learning.py \
        --fold $fold \
        --data_dir ./datasets/cnnscore_database/loro_extreme \
        --output_dir ./datasets/cnnscore_database/loro_extreme_transfer/$fold \
        --finetune_ratio $ratio \
        --seed 42

    # Step 2: Find the best pretrained model from LORO_Extreme
    PRETRAINED_MODEL=$(ls result/LORO_Extreme/$fold/best_model_epoch_*.pth 2>/dev/null | head -1)

    if [ -z "$PRETRAINED_MODEL" ]; then
        echo "ERROR: No pretrained model found in result/LORO_Extreme/$fold/"
        echo "Please run LORO Extreme training first:"
        echo "  ./run_loro_extreme.sh $fold $gpu"
        exit 1
    fi

    echo ""
    echo "[Step 2] Found pretrained model: $PRETRAINED_MODEL"

    # Step 3: Create config with pretrained model path
    CONFIG_TEMPLATE="configs/DrugBAN_BiLSTM_CNNScore_Multitask_Extreme_Transfer.yaml"
    CONFIG_TEMP="configs/DrugBAN_BiLSTM_CNNScore_Multitask_Extreme_Transfer_${fold}.yaml"

    cp $CONFIG_TEMPLATE $CONFIG_TEMP
    sed -i "s|PRETRAINED_MODEL: \"\"|PRETRAINED_MODEL: \"$PRETRAINED_MODEL\"|g" $CONFIG_TEMP

    echo "[Step 3] Created config: $CONFIG_TEMP"

    # Step 4: Run transfer learning
    echo ""
    echo "[Step 4] Running transfer learning..."
    echo "Command: CUDA_VISIBLE_DEVICES=$gpu python main.py --cfg $CONFIG_TEMP --data cnnscore_database/loro_extreme_transfer --split $fold"
    echo ""

    CUDA_VISIBLE_DEVICES=$gpu python main.py \
        --cfg $CONFIG_TEMP \
        --data cnnscore_database/loro_extreme_transfer \
        --split $fold

    echo ""
    echo "============================================================"
    echo "Extreme Transfer learning completed for $fold"
    echo "Results saved to: ./result/LORO_Extreme_Transfer/$fold"
    echo "============================================================"
}

if [ "$FOLD" = "all" ]; then
    for i in $(seq 0 11); do
        fold="fold_$i"
        echo ""
        echo ">>> Starting $fold on GPU $GPU_ID"
        run_single_fold $fold $GPU_ID $FINETUNE_RATIO
    done

    echo ""
    echo "============================================"
    echo "All 12 folds completed!"
    echo "Results in: ./result/LORO_Extreme_Transfer/"
    echo "============================================"
else
    run_single_fold $FOLD $GPU_ID $FINETUNE_RATIO
fi
