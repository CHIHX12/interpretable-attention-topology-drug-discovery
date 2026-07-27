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
# Training strategy (--retrain):
#   - Runs for MAX_EPOCH=50 epochs (no early stopping)
#   - Tracks best checkpoint by validation loss (multitask) or AUROC (single-task)
#   - Saves best_model_epoch_N.pth (best val) and model_epoch_50.pth (last)
#   - Best epoch was 36 (val AUROC = 0.9621) in the paper run
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
# Publication network-overlap figures (600 dpi) read/write here:
FIGURE_DIR="result/class_attention_analysis_pdb"

echo "============================================================"
echo " Interpretable Attention-Topology Framework — Reproduce"
echo "============================================================"
echo ""

# ── Step 0: sanity checks ────────────────────────────────────────
echo "[0/5] Checking prerequisites..."

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
  echo "[1/5] Fine-tuning from pretrained BindingDB weights..."
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
  echo "[1/5] Skipping fine-tuning — using provided model:"
  echo "      ${FINETUNED_MODEL}"
  if [[ ! -f "${FINETUNED_MODEL}" ]]; then
    echo "ERROR: Model parameters not found: ${FINETUNED_MODEL}"
    echo ""
    echo "       Trained parameters are NOT distributed in this repository."
    echo "       They are available on request for noncommercial use — see"
    echo "       MODEL-WEIGHTS-TERMS.md for the terms and how to request access."
    echo ""
    echo "       Once granted, place the file at:"
    echo "         ${FINETUNED_MODEL}"
    echo ""
    echo "       Alternatively, train from scratch with:  bash reproduce.sh --retrain"
    exit 1
  fi
fi

# ── Step 2: batch prediction + attention extraction ──────────────
echo ""
echo "[2/5] Running batch prediction and extracting attention..."
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
echo "      AUROC: ${AUROC}  (on full 1,539-sample set; val AUROC during training = 0.9621)"

# ── Step 3: class-differential attention analysis ────────────────
echo ""
echo "[3/5] Running class-differential attention analysis..."
echo "      Output : ${ANALYSIS_DIR}/"
python analyze_class_attention_difference_en.py \
  --config "${CFG}" \
  --model_path "${FINETUNED_MODEL}" \
  --data_file "${DATA_DIR}/GHSR_training_data.csv" \
  --output_dir "${ANALYSIS_DIR}"

echo "      Key residues saved to ${ANALYSIS_DIR}/"

# ── Step 4: consensus residue analysis ──────────────────────────
echo ""
echo "[4/5] Running consensus residue analysis..."
echo "      Output : ${CONSENSUS_DIR}/"
python consensus_analysis_ghsr.py \
  --attention_file "${ATTENTION_DIR}/GHSR_attention_scores.npz" \
  --output_dir "${CONSENSUS_DIR}" \
  --protein_length 523

# ── Step 5: publication network-overlap figures (600 dpi) ───────
echo ""
echo "[5/5] Generating publication network-overlap figures (600 dpi)..."
echo "      Output : ${FIGURE_DIR}/"

# The figure scripts read the user-specified 25-pair tables. These are
# provided in Important_Analysis/{Active,Inactive}/; stage them into the
# figure directory if they are not already present.
mkdir -p "${FIGURE_DIR}"
for cls in Active Inactive; do
  src="Important_Analysis/${cls}/exact_five_${cls,,}_25pairs.csv"
  dst="${FIGURE_DIR}/exact_five_${cls,,}_25pairs.csv"
  if [[ ! -f "${dst}" && -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
done

if [[ -f "${FIGURE_DIR}/exact_five_active_25pairs.csv" ]]; then
  # Sharing matrix / overlap diagram / functional regions (Active & Inactive)
  python analyze_network_overlap_active.py
  python analyze_network_overlap_inactive.py
  # 5x5 ΔImp heatmaps + pairing networks (Active & Inactive)
  python create_exact_five_figures.py
  # Inactive 3-way sharing diagram
  python analyze_network_overlap_inactive_3way.py

  # Refresh the curated copies under Important_Analysis/
  for ext in png pdf; do
    cp "${FIGURE_DIR}"/{target_constitutive_sharing_matrix,network_overlap_diagram,functional_regions_3d}_active.${ext} Important_Analysis/Active/ 2>/dev/null || true
    cp "${FIGURE_DIR}"/exact_five_active_{heatmap,network}.${ext} Important_Analysis/Active/ 2>/dev/null || true
    cp "${FIGURE_DIR}"/{target_constitutive_sharing_matrix,network_overlap_diagram,functional_regions_3d}_inactive.${ext} Important_Analysis/Inactive/ 2>/dev/null || true
    cp "${FIGURE_DIR}"/exact_five_inactive_{heatmap,network}.${ext} Important_Analysis/Inactive/ 2>/dev/null || true
    cp "${FIGURE_DIR}"/network_overlap_diagram_inactive_3way.${ext} Important_Analysis/Inactive/ 2>/dev/null || true
  done
  echo "      Figures regenerated at 600 dpi and copied to Important_Analysis/"
else
  echo "      SKIP — exact_five_*_25pairs.csv not found; figures left unchanged."
fi

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
echo "   Figures (600dpi) : ${FIGURE_DIR}/  +  Important_Analysis/{Active,Inactive}/"
echo ""
echo " To visualize in PyMOL:"
echo "   pymol ${CONSENSUS_DIR}/visualize_GHSR_consensus.pml"
echo "============================================================"
