#!/bin/bash
# LORO 3-Class Classification Experiment
#
# 3-Class labeling:
#   Y=0: Z < 0.4     (Inactive)
#   Y=1: 0.4 <= Z <= 0.9  (Intermediate)
#   Y=2: Z > 0.9     (Highly Active)
#
# Usage:
#   ./run_loro_3class.sh <fold> <gpu_id>
#   ./run_loro_3class.sh all <gpu_id>
#
# Examples:
#   ./run_loro_3class.sh fold_0 0
#   ./run_loro_3class.sh all 0

set -e

FOLD=${1:-fold_0}
GPU_ID=${2:-0}

CONFIG="configs/DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml"
DATA_DIR="cnnscore_database/loro_3class"

run_single_fold() {
    local fold=$1
    local gpu=$2

    echo "============================================================"
    echo "LORO 3-Class Classification - $fold on GPU $gpu"
    echo "  Y=0: Z<0.4 (Inactive)"
    echo "  Y=1: 0.4<=Z<=0.9 (Intermediate)"
    echo "  Y=2: Z>0.9 (Highly Active)"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$gpu python main.py \
        --cfg $CONFIG \
        --data $DATA_DIR \
        --split $fold

    echo ""
    echo "Done: $fold -> result/LORO_3Class/$fold"
    echo "============================================================"
}

if [ "$FOLD" = "all" ]; then
    for i in $(seq 0 11); do
        fold="fold_$i"
        echo ""
        echo ">>> Starting $fold on GPU $GPU_ID"
        run_single_fold $fold $GPU_ID
    done

    echo ""
    echo "============================================"
    echo "All 12 folds completed!"
    echo "Results in: ./result/LORO_3Class/"
    echo "============================================"
else
    run_single_fold $FOLD $GPU_ID
fi
