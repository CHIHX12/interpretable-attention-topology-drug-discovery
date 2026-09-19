#!/usr/bin/env bash
# run_rev22_queue.sh — all training runs for the v22 revision.
#
# Each run writes to result/rev22/<split>/<init>_seed<seed>/ and is skipped if
# that folder already holds test_markdowntable.txt, so the queue can be
# restarted safely and never overwrites earlier results.
#
# Usage:
#   bash revision/run_rev22_queue.sh GPU_ID [PART NPARTS]
#   e.g. two GPUs in parallel:
#     bash revision/run_rev22_queue.sh 0 0 2 &
#     bash revision/run_rev22_queue.sh 1 1 2 &
set -uo pipefail

GPU="${1:-0}"
PART="${2:-0}"
NPARTS="${3:-1}"
# environment used for all GHSR runs (torch 2.2.1, dgl 2.1.0+cu121)
PY="${PY:-/home/cycheng/miniforge3/envs/drugban/bin/python}"
LOG_DIR="result/rev22/logs"
mkdir -p "${LOG_DIR}"

JOBS=()
# 1. deposited split, ten fine-tuning seeds (performance + attention stability)
for s in 42 43 44 45 46 47 48 49 50 51; do JOBS+=("random pretrained $s"); done
# 2. compound-disjoint and scaffold splits, with and without BindingDB pre-training
for k in 0 1 2 3 4; do
  JOBS+=("rev22_compound_s${k} pretrained 42" "rev22_compound_s${k} scratch 42")
  JOBS+=("rev22_scaffold_s${k} pretrained 42" "rev22_scaffold_s${k} scratch 42")
done
# 3. from-scratch control on the deposited split
for s in 42 43 44 45 46; do JOBS+=("random scratch $s"); done
# 4. sensitivity: dual-endpoint ligands removed
for k in 0 1 2 3 4; do JOBS+=("rev22_nodual_s${k} pretrained 42"); done

for i in "${!JOBS[@]}"; do
  (( i % NPARTS == PART )) || continue
  read -r split init seed <<< "${JOBS[$i]}"
  log="${LOG_DIR}/${split}_${init}_seed${seed}.log"
  if [[ -f "result/rev22/${split}/${init}_seed${seed}/test_markdowntable.txt" ]]; then
    echo "skip ${split} ${init} ${seed}"
    continue
  fi
  echo "$(date '+%F %T') start ${split} ${init} seed${seed}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -u revision/train_rev22.py \
    --split "${split}" --init "${init}" --seed "${seed}" > "${log}" 2>&1 \
    || echo "FAILED ${split} ${init} ${seed} (see ${log})"
  echo "$(date '+%F %T') end   ${split} ${init} seed${seed}"
done
echo "$(date '+%F %T') part ${PART}/${NPARTS} finished"
