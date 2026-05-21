#!/bin/bash
#
# LORO (Leave-One-Receptor-Out) Pipeline
#
# 
# ./run_loro_experiment.sh # row12 folds
# ./run_loro_experiment.sh --quick # testrow fold 0-2
# ./run_loro_experiment.sh --fold 5 # row fold 5
#

set -e # error

# color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# config
CONFIG="configs/DrugBAN_BiLSTM_CNNScore_Multitask.yaml"
DATA_DIR="datasets/cnnscore_database/random"
LORO_DIR="datasets/cnnscore_database/loro"
OUTPUT_DIR="result/LORO"
EPOCHS=30 # LORO epochstraining

# parameter
QUICK_MODE=false
SINGLE_FOLD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --fold)
            SINGLE_FOLD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Run quick test (folds 0-2 only)"
            echo "  --fold N      Run single fold N (0-11)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}          LORO (Leave-One-Receptor-Out) Experiment Pipeline${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Config:     $CONFIG"
echo "Data:       $DATA_DIR"
echo "LORO dir:   $LORO_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Epochs:     $EPOCHS"
echo ""

# Step 1: create LORO datasplit
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 1: Creating LORO Data Splits${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

if [ -d "$LORO_DIR" ]; then
    echo -e "${YELLOW}⚠️  LORO directory already exists: $LORO_DIR${NC}"
    read -p "Recreate splits? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$LORO_DIR"
        python scripts/create_loro_splits.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$LORO_DIR" \
            --val-ratio 0.1
    else
        echo "Using existing splits."
    fi
else
    python scripts/create_loro_splits.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$LORO_DIR" \
        --val-ratio 0.1
fi

echo ""
echo -e "${GREEN}✅ LORO splits created!${NC}"
echo ""

# Step 2: training folds
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 2: Training LORO Folds${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# training folds
if [ -n "$SINGLE_FOLD" ]; then
    FOLDS=($SINGLE_FOLD)
    echo "Training single fold: $SINGLE_FOLD"
elif [ "$QUICK_MODE" = true ]; then
    FOLDS=(0 1 2)
    echo "Quick mode: Training folds 0-2 only"
else
    FOLDS=(0 1 2 3 4 5 6 7 8 9 10 11)
    echo "Training all 12 folds"
fi

echo ""

# trainingeach fold
FAILED_FOLDS=()
COMPLETED_FOLDS=()

for fold in "${FOLDS[@]}"; do
    echo ""
    echo -e "${BLUE}--------------------------------------------------------------------------------${NC}"
    echo -e "${BLUE}Training Fold $fold (Test on Receptor $fold)${NC}"
    echo -e "${BLUE}--------------------------------------------------------------------------------${NC}"
    echo ""

 # checktraining
    FOLD_OUTPUT="$OUTPUT_DIR/fold_$fold"
    if [ -f "$FOLD_OUTPUT/best_model.pth" ]; then
        echo -e "${YELLOW}⚠️  Fold $fold already trained: $FOLD_OUTPUT/best_model.pth${NC}"
        read -p "Retrain? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping fold $fold"
            COMPLETED_FOLDS+=($fold)
            continue
        fi
    fi

 # training fold
 # use main.pydatapath LORO fold
    FOLD_DATA_DIR="$LORO_DIR/fold_$fold"

    echo "Data directory: $FOLD_DATA_DIR"
    echo "Output directory: $FOLD_OUTPUT"
    echo ""

 # rowtraining
    python main.py \
        --cfg "$CONFIG" \
        --data cnnscore_database \
        --split loro \
        --loro-fold "$fold" \
        --loro-dir "$LORO_DIR" \
        --output-dir "$FOLD_OUTPUT" \
        --epochs "$EPOCHS" \
        2>&1 | tee "$FOLD_OUTPUT/training.log"

 # checksuccess
    if [ ${PIPESTATUS[0]} -eq 0 ] && [ -f "$FOLD_OUTPUT/best_model.pth" ]; then
        echo -e "${GREEN}✅ Fold $fold training completed!${NC}"
        COMPLETED_FOLDS+=($fold)
    else
        echo -e "${RED}❌ Fold $fold training failed!${NC}"
        FAILED_FOLDS+=($fold)
    fi
done

echo ""
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Training Summary${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Completed folds: ${COMPLETED_FOLDS[@]}"
if [ ${#FAILED_FOLDS[@]} -gt 0 ]; then
    echo -e "${RED}Failed folds: ${FAILED_FOLDS[@]}${NC}"
fi
echo ""

# fold
if [ ${#FAILED_FOLDS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Some folds failed. Continue with evaluation? (y/N)${NC}"
    read -p ": " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Stopping."
        exit 1
    fi
fi

# Step 3: folds generatereport
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 3: Evaluating LORO Results${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python scripts/evaluate_loro_results.py \
    --loro-dir "$LORO_DIR" \
    --result-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/analysis"

echo ""
echo -e "${GREEN}✅ Evaluation complete!${NC}"
echo ""

# Step 4: generatevisualization
echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Step 4: Generating Visualizations${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python scripts/plot_loro_results.py \
    --input "$OUTPUT_DIR/analysis/loro_results_summary.csv" \
    --output-dir "$OUTPUT_DIR/analysis/figures"

echo ""
echo -e "${GREEN}✅ Visualizations generated!${NC}"
echo ""

# done
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}          ✅ LORO Experiment Complete!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo "Results directory: $OUTPUT_DIR/analysis"
echo ""
echo "Key files:"
echo "  - loro_results_summary.csv: Performance metrics for all folds"
echo "  - figures/loro_performance_boxplot.png: Boxplot of AUROC/AUPRC/R² across folds"
echo "  - figures/loro_per_fold_performance.png: Bar chart for each fold"
echo "  - loro_analysis_report.txt: Detailed statistical analysis"
echo ""
echo "Next steps:"
echo "1. Review the results:"
echo "   cat $OUTPUT_DIR/analysis/loro_analysis_report.txt"
echo "2. Include figures in your paper (Figure X: Generalization Analysis)"
echo "3. Report in paper:"
echo "   'Leave-one-receptor-out cross-validation achieved AUROC of X.XX ± Y.YY,"
echo "   demonstrating strong generalization to unseen receptors.'"
echo ""
