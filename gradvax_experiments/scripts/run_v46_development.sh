#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
WORKERS="${WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_dev() {
  local method="$1"
  local run_id="$2"
  shift 2
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[skip completed] ${run_id}"
    return
  fi
  local resume=()
  if [[ -f "${run_dir}/checkpoint_last.pt" ]]; then
    resume=(--resume auto)
  fi
  echo "[single-gpu run] ${run_id} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  python -u train.py \
    --protocol P-C100-100 \
    --method "$method" \
    --mode development \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$run_id" \
    --checkpoint-every 500 \
    --snapshot-every 50000 \
    "${resume[@]}" \
    "$@"
}

# Ordered deliberately: establish the exact paired baseline first, then isolate
# output-row restriction, classwise anchors, and finally the correction budget.
run_dev fixmatch v46-dev-c100-100-fixmatch-observer-seed0 --tailrow-observer
run_dev tangs v46-dev-c100-100-legacy-pcgrad-seed0 --tangs-tau inf
run_dev tailrow-group v46-dev-c100-100-tailrow-group-seed0
run_dev tailrow-classwise v46-dev-c100-100-tailrow-classwise-seed0 \
  --tangs-correction-rho inf
run_dev tangs-v46 v46-dev-c100-100-tangs-rho1-seed0 \
  --tangs-correction-rho 1

python scripts/analyze_v46_development.py \
  --results "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/v46-development-selection.json"
