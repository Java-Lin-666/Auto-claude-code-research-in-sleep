#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
DEVELOPMENT_GATE="${V47_DEVELOPMENT_GATE:-${OUTPUT_ROOT}/v47-integrated-development-gate.json}"
WORKERS="${WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$DEVELOPMENT_GATE" ]]; then
  echo "[blocked] Missing fresh integrated 50k gate: ${DEVELOPMENT_GATE}" >&2
  exit 4
fi
readarray -t GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_confirmatory_matrix"])' "$DEVELOPMENT_GATE"
)
if [[ "${GATE[0]}" != "PASS" || "${GATE[1]}" != "True" ]]; then
  echo "[blocked] Fresh integrated 50k gate is ${GATE[0]}; 250k matrix is forbidden." >&2
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

  echo "[single-gpu confirmatory run, 250000 steps] ${run_id}"
  python -u train.py \
    --protocol "$protocol" \
    --method tangs-v47 \
    --mode confirmatory \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$run_id" \
    --score-uniform-la-alpha 0.85 \
    --score-base-alpha 0.65 \
    --score-extra-tail-alpha 0.25 \
    --score-anchor-threshold 0.75 \
    --checkpoint-every 500 \
    --snapshot-every 50000 \
    "${resume[@]}"
}

# C100 is the core stop-loss gate, not the complete paper matrix.
C100_ID="v47-confirm-c100-100-seed0"
run_cell P-C100-100 "$C100_ID"
python scripts/analyze_v47_confirmatory.py \
  --run-dir "${OUTPUT_ROOT}/${C100_ID}" \
  --output "${OUTPUT_ROOT}/v47-confirmatory-c100-gate.json"

readarray -t C100_GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_cross_dataset_runs"])' "${OUTPUT_ROOT}/v47-confirmatory-c100-gate.json"
)
if [[ "${C100_GATE[0]}" != "PASS" || "${C100_GATE[1]}" != "True" ]]; then
  echo "[stop-loss] C100 confirmatory gate is ${C100_GATE[0]}; C10/STL10 are not launched." >&2
  exit 5
fi

# Required cross-dataset transfer with exactly the C100-frozen constants.
C10_ID="v47-confirm-c10-100-seed0"
STL20_ID="v47-confirm-stl10-20-seed0"
run_cell P-C10-100 "$C10_ID"
run_cell P-STL10-20 "$STL20_ID"

python scripts/analyze_v47_matrix.py \
  --c100-run-dir "${OUTPUT_ROOT}/${C100_ID}" \
  --c10-run-dir "${OUTPUT_ROOT}/${C10_ID}" \
  --stl20-run-dir "${OUTPUT_ROOT}/${STL20_ID}" \
  --output "${OUTPUT_ROOT}/v47-required-matrix.json"
