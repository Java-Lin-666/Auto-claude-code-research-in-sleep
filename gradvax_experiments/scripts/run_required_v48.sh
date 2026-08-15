#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
HOLDOUT_GATE="${V48_HOLDOUT_GATE:-${OUTPUT_ROOT}/v48-holdout-development-gate.json}"
WORKERS="${WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$HOLDOUT_GATE" ]]; then
  echo "[blocked] Missing v4.8 seed-1 holdout gate: ${HOLDOUT_GATE}" >&2
  exit 4
fi
readarray -t GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_confirmatory_matrix"])' "$HOLDOUT_GATE"
)
if [[ "${GATE[0]}" != "PASS" || "${GATE[1]}" != "True" ]]; then
  echo "[blocked] v4.8 holdout gate is ${GATE[0]}; 250k matrix is forbidden." >&2
  exit 4
fi

run_cell() {
  local protocol="$1"
  local run_id="$2"
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[skip completed] ${run_id}"
    return
  fi
  local resume=()
  if [[ -f "${run_dir}/checkpoint_last.pt" ]]; then
    resume=(--resume auto)
  fi
  echo "[single-gpu v4.8 confirmatory, 250000 steps] ${run_id}"
  python -u train.py \
    --protocol "$protocol" \
    --method tangs-v48 \
    --mode confirmatory \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$run_id" \
    --score-uniform-la-alpha 0.80 \
    --score-base-alpha 0.70 \
    --score-extra-tail-alpha 0.40 \
    --score-anchor-threshold 0.775 \
    --score-anchor-margin 0.05 \
    --score-confidence-ceiling 0.90 \
    --checkpoint-every 500 \
    --snapshot-every 50000 \
    "${resume[@]}"
}

C100_ID="v48-confirm-c100-100-seed0"
run_cell P-C100-100 "$C100_ID"
python scripts/analyze_v48_confirmatory.py \
  --run-dir "${OUTPUT_ROOT}/${C100_ID}" \
  --holdout-gate "$HOLDOUT_GATE" \
  --output "${OUTPUT_ROOT}/v48-confirmatory-c100-gate.json"

readarray -t C100_GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_cross_dataset_runs"])' "${OUTPUT_ROOT}/v48-confirmatory-c100-gate.json"
)
if [[ "${C100_GATE[0]}" != "PASS" || "${C100_GATE[1]}" != "True" ]]; then
  echo "[stop-loss] C100 v4.8 gate is ${C100_GATE[0]}; C10/STL10 are not launched." >&2
  exit 5
fi

C10_ID="v48-confirm-c10-100-seed0"
STL20_ID="v48-confirm-stl10-20-seed0"
run_cell P-C10-100 "$C10_ID"
run_cell P-STL10-20 "$STL20_ID"
python scripts/analyze_v48_matrix.py \
  --c100-run-dir "${OUTPUT_ROOT}/${C100_ID}" \
  --c10-run-dir "${OUTPUT_ROOT}/${C10_ID}" \
  --stl20-run-dir "${OUTPUT_ROOT}/${STL20_ID}" \
  --output "${OUTPUT_ROOT}/v48-required-matrix.json"
