#!/bin/bash
# LORO Transfer Learning Experiment
#
# This script:
# 1. Prepares transfer learning data for each fold
# 2. Creates a config with the correct pretrained model path
# 3. Runs fine-tuning on target receptor data
#
# Usage:
#   ./run_loro_transfer_learning.sh <fold> <gpu_id> [finetune_ratio]
#
# Example:
#   ./run_loro_transfer_learning.sh fold_1 0 0.3

set -e

FOLD=${1:-fold_1}
GPU_ID=${2:-0}
FINETUNE_RATIO=${3:-0.3}

echo "============================================================"
echo "LORO Transfer Learning Experiment"
echo "============================================================"
echo "Fold: $FOLD"
echo "GPU: $GPU_ID"
echo "Fine-tune ratio: $FINETUNE_RATIO"
echo "============================================================"

# Step 1: Prepare transfer learning data
echo ""
echo "[Step 1] Preparing transfer learning data..."
python prepare_loro_transfer_learning.py \
    --fold $FOLD \
    --finetune_ratio $FINETUNE_RATIO \
    --seed 42

# Step 2: Find the best pretrained model for this fold
PRETRAINED_MODEL=$(ls result/LORO/$FOLD/best_model_epoch_*.pth 2>/dev/null | head -1)

if [ -z "$PRETRAINED_MODEL" ]; then
    echo "ERROR: No pretrained model found for $FOLD"
    echo "Please run LORO training first:"
    echo "  CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --cfg configs/DrugBAN_BiLSTM_CNNScore_Multitask.yaml --data cnnscore_database/loro --split $FOLD"
    exit 1
fi

echo ""
echo "[Step 2] Found pretrained model: $PRETRAINED_MODEL"

# Step 3: Create a temporary config with the pretrained model path
CONFIG_TEMPLATE="configs/DrugBAN_BiLSTM_CNNScore_Multitask_Transfer.yaml"
CONFIG_TEMP="configs/DrugBAN_BiLSTM_CNNScore_Multitask_Transfer_${FOLD}.yaml"

# Copy template and update PRETRAINED_MODEL
cp $CONFIG_TEMPLATE $CONFIG_TEMP
sed -i "s|PRETRAINED_MODEL: \"\"|PRETRAINED_MODEL: \"$PRETRAINED_MODEL\"|g" $CONFIG_TEMP

# Note: main.py now auto-appends fold_X to OUTPUT_DIR, so keep base path only

echo "[Step 3] Created config: $CONFIG_TEMP"

# Step 4: Run transfer learning
echo ""
echo "[Step 4] Running transfer learning..."
echo "Command: CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --cfg $CONFIG_TEMP --data cnnscore_database/loro_transfer --split $FOLD"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    --cfg $CONFIG_TEMP \
    --data cnnscore_database/loro_transfer \
    --split $FOLD

echo ""
echo "============================================================"
echo "Transfer learning completed for $FOLD"
echo "Results saved to: ./result/LORO_Transfer/$FOLD"
echo "============================================================"
