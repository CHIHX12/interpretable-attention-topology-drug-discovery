#!/usr/bin/env bash
# run_rev22_featoff.sh — main-model runs for the v22 revision: training and
# attention read-out both without the four physicochemical channels, so that
# the analysed model is the model that classifies.
# Usage: bash revision/run_rev22_featoff.sh GPU_ID PART NPARTS
set -uo pipefail
GPU="${1:-0}"; PART="${2:-0}"; NPARTS="${3:-1}"
PY="${PY:-/home/cycheng/miniforge3/envs/drugban/bin/python}"
mkdir -p result/rev22/logs
JOBS=()
for s in 42 43 44 45 46 47 48 49 50 51; do JOBS+=("random pretrained $s"); done
for s in 45 46; do JOBS+=("random scratch $s"); done
for k in 0 1 2 3 4; do
  JOBS+=("rev22_compound_s${k} pretrained 42" "rev22_compound_s${k} scratch 42")
  JOBS+=("rev22_scaffold_s${k} pretrained 42" "rev22_scaffold_s${k} scratch 42")
done
for i in "${!JOBS[@]}"; do
  (( i % NPARTS == PART )) || continue
  read -r split init seed <<< "${JOBS[$i]}"
  out="result/rev22/${split}/${init}_featoff_seed${seed}"
  [[ -f "${out}/test_markdowntable.txt" ]] && { echo "skip ${split} ${init} ${seed}"; continue; }
  echo "$(date '+%F %T') start ${split} ${init} seed${seed}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u revision/train_rev22.py --split "${split}" \
    --init "${init}" --seed "${seed}" --no_protein_features \
    > "result/rev22/logs/featoff_${split}_${init}_seed${seed}.log" 2>&1 \
    || echo "FAILED ${split} ${init} ${seed}"
done
echo "$(date '+%F %T') featoff part ${PART}/${NPARTS} finished"
