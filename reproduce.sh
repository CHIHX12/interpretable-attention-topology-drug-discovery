#!/usr/bin/env bash
# reproduce.sh — end-to-end reproducibility script
#
# Reproduces the key results from:
#   "An Interpretable Attention-Topology Framework Decouples Affinity and
#    Efficacy in Structure-Free Drug Discovery"
#
# Usage:
#   bash reproduce.sh          # use provided fine-tuned model (fast, ~1 min)
#   bash reproduce.sh --retrain  # re-run fine-tuning from pretrained weights (~30 min on GPU)
#
# Requirements:
#   conda activate drugban   (or equivalent env with requirements.txt installed)
#   CUDA GPU recommended; CPU is supported but slow

set -euo pipefail

RETRAIN=false
if [[ "${1:-}" == "--retrain" ]]; then
  RETRAIN=true
fi

CFG="configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml"
DATA_DIR="datasets/GPCR_resarch"
FINETUNED_MODEL="models/finetuned/DrugBAN_BiLSTM_GHSR_epoch36.pth"
ATTENTION_DIR="${DATA_DIR}/attention_results_reproduce"
ANALYSIS_DIR="result/class_diff_reproduce"
CONSENSUS_DIR="${DATA_DIR}/consensus_results_reproduce"

echo "============================================================"
echo " Interpretable Attention-Topology Framework — Reproduce"
echo "============================================================"
echo ""

# ── Step 0: sanity checks ────────────────────────────────────────
echo "[0/4] Checking prerequisites..."

if [[ ! -f "${DATA_DIR}/GHSR_training_data.csv" ]]; then
  echo "ERROR: ${DATA_DIR}/GHSR_training_data.csv not found."
  echo "       Make sure you cloned the full repository."
  exit 1
fi

if [[ ! -d "${DATA_DIR}/random" ]]; then
  echo "      Splitting GHSR data into train/val/test..."
  python split_ghsr_data.py \
    --input "${DATA_DIR}/GHSR_training_data.csv" \
    --output "${DATA_DIR}/random" \
    --seed 42
fi

echo "      OK — data ready ($(wc -l < ${DATA_DIR}/GHSR_training_data.csv) rows)"

# ── Step 1: fine-tuning (optional) ──────────────────────────────
if [[ "${RETRAIN}" == "true" ]]; then
  echo ""
  echo "[1/4] Fine-tuning from pretrained BindingDB weights..."
  echo "      Config : ${CFG}"
  echo "      Output : result/DrugBAN_BiLSTM_GHSR_Reproduce/"
  python main.py \
    --cfg "${CFG}" \
    --data GPCR_resarch \
    --split random
  # Use the freshly trained best model
  FINETUNED_MODEL="result/DrugBAN_BiLSTM_GHSR_Reproduce/best_model_epoch_36.pth"
  echo "      Fine-tuning done."
else
  echo ""
  echo "[1/4] Skipping fine-tuning — using provided model:"
  echo "      ${FINETUNED_MODEL}"
  if [[ ! -f "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: Model file not found: ${FINETUNED_MODEL}"
    echo "       Run with --retrain, or check that models/finetuned/ is present."
    exit 1
  fi
fi

# ── Step 2: batch prediction + attention extraction ──────────────
echo ""
echo "[2/4] Running batch prediction and extracting attention..."
echo "      Model  : ${FINETUNED_MODEL}"
echo "      Output : ${ATTENTION_DIR}/"
python batch_predict_ghsr.py \
  --config "${CFG}" \
  --data_file "${DATA_DIR}/GHSR_training_data.csv" \
  --model_path "${FINETUNED_MODEL}" \
  --output_dir "${ATTENTION_DIR}" \
  --device cuda

AUROC=$(python -c "
import json
with open('${ATTENTION_DIR}/GHSR_prediction_stats.json') as f:
    d = json.load(f)
print(f\"{d['performance']['auroc']:.4f}\")
")
echo "      AUROC: ${AUROC}  (paper reports 0.9621 for best_model_epoch_36)"

# ── Step 3: class-differential attention analysis ────────────────
echo ""
echo "[3/4] Running class-differential attention analysis..."
echo "      Output : ${ANALYSIS_DIR}/"
python analyze_class_attention_difference_en.py \
  --config "${CFG}" \
  --model_path "${FINETUNED_MODEL}" \
  --data_file "${DATA_DIR}/GHSR_training_data.csv" \
  --output_dir "${ANALYSIS_DIR}"

echo "      Key residues saved to ${ANALYSIS_DIR}/"

# ── Step 4: consensus residue analysis ──────────────────────────
echo ""
echo "[4/4] Running consensus residue analysis..."
echo "      Output : ${CONSENSUS_DIR}/"
python consensus_analysis_ghsr.py \
  --attention_file "${ATTENTION_DIR}/GHSR_attention_scores.npz" \
  --output_dir "${CONSENSUS_DIR}" \
  --protein_length 523

echo ""
echo "============================================================"
echo " Done."
echo ""
echo " Outputs:"
echo "   Attention scores : ${ATTENTION_DIR}/GHSR_attention_scores.npz"
echo "   Predictions      : ${ATTENTION_DIR}/GHSR_predictions.csv"
echo "   Class diff CSV   : ${ANALYSIS_DIR}/"
echo "   Consensus CSV    : ${CONSENSUS_DIR}/GHSR_consensus_residues.csv"
echo "   PyMOL script     : ${CONSENSUS_DIR}/visualize_GHSR_consensus.pml"
echo ""
echo " To visualize in PyMOL:"
echo "   pymol ${CONSENSUS_DIR}/visualize_GHSR_consensus.pml"
echo "============================================================"
