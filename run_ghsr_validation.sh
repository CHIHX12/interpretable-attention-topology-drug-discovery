#!/bin/bash
################################################################################
# GHSR pocketvalidate
# Complete GHSR Binding Pocket Validation Workflow
#
# DrugBAN_BiLSTM modelpocketamino acid
#
# step：
# 1. prediction 1,539 GHSR drugextractattentionscore
# 2. consensusanalysisfrequencyresidue
# 3. comparisonprediction vs. pocket (PDB 8JSR)
# 4. generatevalidatereportvisualization
#
# usemethod：
#   bash run_ghsr_validation.sh [--model MODEL_PATH] [--config CONFIG_PATH]
#
# option：
#   --model    modelweightpath (default: result/DrugBAN_BiLSTM/best_model.pth)
#   --config   configfilepath (default: configs/DrugBAN_BiLSTM.yaml)
#   --batch-size  batchsize (default: 32)
#   --device   device (default: cuda)
################################################################################

set -e  # Exit on error

# color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# defaultparameter
MODEL_PATH="result/DrugBAN_BiLSTM/best_model.pth"
CONFIG_PATH="configs/DrugBAN_BiLSTM.yaml"
DATA_FILE="datasets/GPCR_resarch/GHSR_training_data.csv"
TRUE_POCKET="datasets/GPCR_resarch/validation_results/GHSR_true_pocket.txt"
BATCH_SIZE=32
DEVICE="cuda"
PROTEIN_LENGTH=523

# outputdirectory
ATTENTION_DIR="datasets/GPCR_resarch/attention_results"
CONSENSUS_DIR="datasets/GPCR_resarch/consensus_results"
VALIDATION_DIR="datasets/GPCR_resarch/validation_results"

# rowparameter
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
 echo -e "${RED}parameter: $1${NC}"
            echo "usemethod: $0 [--model MODEL_PATH] [--config CONFIG_PATH] [--batch-size N] [--device DEVICE]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🧬 GHSR pocketvalidate${NC}"
echo -e "${BLUE}   Growth Hormone Secretagogue Receptor Binding Pocket Validation${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# checkfile
echo -e "${YELLOW}📋 checkfile...${NC}"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
 echo -e "${RED}❌ modelfile: $MODEL_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ model: $MODEL_PATH${NC}"

if [ ! -f "$CONFIG_PATH" ]; then
 echo -e "${RED}❌ configfile: $CONFIG_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ config: $CONFIG_PATH${NC}"

if [ ! -f "$DATA_FILE" ]; then
 echo -e "${RED}❌ datafile: $DATA_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ data: $DATA_FILE (GHSR training data)${NC}"

if [ ! -f "$TRUE_POCKET" ]; then
 echo -e "${RED}❌ pocketfile: $TRUE_POCKET${NC}"
 echo -e "${YELLOW}⚠️ row: python validate_ghsr_pocket.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pocket: $TRUE_POCKET${NC}"

echo ""
echo -e "${YELLOW}⚙️  config:${NC}"
echo "   model: $MODEL_PATH"
echo "   config: $CONFIG_PATH"
echo "   data: $DATA_FILE"
echo "   batchsize: $BATCH_SIZE"
echo "   device: $DEVICE"
echo "   proteinlength: $PROTEIN_LENGTH aa"
echo ""

# createoutputdirectory
mkdir -p "$ATTENTION_DIR"
mkdir -p "$CONSENSUS_DIR"
mkdir -p "$VALIDATION_DIR"

# ============================================================================
# step 1: predictionextractattentionscore
# ============================================================================

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}step 1/3: prediction GHSR drugextractattentionscore${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python batch_predict_ghsr.py \
    --config "$CONFIG_PATH" \
    --data_file "$DATA_FILE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$ATTENTION_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
 echo -e "${RED}❌ prediction${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ step 1 doneattentionscoreextract${NC}"
echo -e "${GREEN}   output: $ATTENTION_DIR/GHSR_attention_scores.npz${NC}"
echo ""

# ============================================================================
# step 2: consensusanalysis
# ============================================================================

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}step 2/3: consensusanalysis - frequencyresidue${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python consensus_analysis_ghsr.py \
    --attention_file "$ATTENTION_DIR/GHSR_attention_scores.npz" \
    --output_dir "$CONSENSUS_DIR" \
    --protein_length "$PROTEIN_LENGTH" \
    --threshold_method "mean_std" \
    --min_frequency 0.25

if [ $? -ne 0 ]; then
 echo -e "${RED}❌ consensusanalysis${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ step 2 doneconsensusresidue${NC}"
echo -e "${GREEN}   output: $CONSENSUS_DIR/GHSR_consensus_residues.csv${NC}"
echo ""

# ============================================================================
# step 3: comparisonprediction vs. pocket
# ============================================================================

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}step 3/3: comparisonmodelprediction vs. pocket (PDB 8JSR)${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

python compare_prediction_vs_truth_ghsr.py \
    --predicted "$CONSENSUS_DIR/GHSR_consensus_residues.csv" \
    --true_pocket "$TRUE_POCKET" \
    --output_dir "$VALIDATION_DIR" \
    --protein_length "$PROTEIN_LENGTH" \
    --top_k 10 20 30 50

if [ $? -ne 0 ]; then
 echo -e "${RED}❌ comparisonanalysis${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ step 3 donevalidateresultgenerate${NC}"
echo -e "${GREEN}   output: $VALIDATION_DIR/GHSR_validation_report.txt${NC}"
echo ""

# ============================================================================
# donesummary
# ============================================================================

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}🎉 GHSR validatedone${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

echo -e "${GREEN}📁 outputfile:${NC}"
echo ""
echo -e "${YELLOW}1. attentionscore:${NC}"
echo "   $ATTENTION_DIR/GHSR_attention_scores.npz"
echo "   $ATTENTION_DIR/GHSR_predictions.csv"
echo "   $ATTENTION_DIR/GHSR_prediction_stats.json"
echo ""

echo -e "${YELLOW}2. consensusanalysis:${NC}"
echo "   $CONSENSUS_DIR/GHSR_consensus_residues.csv"
echo "   $CONSENSUS_DIR/GHSR_consensus_frequencies.txt"
echo "   $CONSENSUS_DIR/GHSR_consensus_heatmap.png"
echo "   $CONSENSUS_DIR/GHSR_top_consensus_residues.png"
echo "   $CONSENSUS_DIR/visualize_GHSR_consensus.pml"
echo ""

echo -e "${YELLOW}3. validateresult:${NC}"
echo " $VALIDATION_DIR/GHSR_validation_report.txt ${GREEN}← report${NC}"
echo "   $VALIDATION_DIR/GHSR_validation_metrics.csv"
echo "   $VALIDATION_DIR/GHSR_performance_vs_k.png"
echo "   $VALIDATION_DIR/GHSR_overlap_top30.png"
echo "   $VALIDATION_DIR/compare_prediction_vs_truth.pml"
echo ""

echo -e "${GREEN}📊 result:${NC}"
echo ""
echo " # validatereport"
echo "   cat $VALIDATION_DIR/GHSR_validation_report.txt"
echo ""
echo " # consensusresidue"
echo "   head -20 $CONSENSUS_DIR/GHSR_consensus_residues.csv"
echo ""
echo " # PyMOL visualizationpocket"
echo "   pymol $VALIDATION_DIR/visualize_GHSR_pocket.pml"
echo ""
echo " # PyMOL visualizationprediction vs. "
echo "   pymol $VALIDATION_DIR/compare_prediction_vs_truth.pml"
echo ""
echo "   # PyMOL visualization（consensusresidue）"
echo "   pymol $CONSENSUS_DIR/visualize_GHSR_consensus.pml"
echo ""

echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}✅ analysisdone${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# displayvalidatereport
if [ -f "$VALIDATION_DIR/GHSR_validation_report.txt" ]; then
 echo -e "${YELLOW}📝 validatereport:${NC}"
    echo ""
 # extractpartial
 sed -n '/3. /,/4. comparison/p' "$VALIDATION_DIR/GHSR_validation_report.txt" | head -n -1
    echo ""
fi

exit 0
