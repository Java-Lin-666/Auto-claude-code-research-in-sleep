#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
SELECTION="${V46_SELECTION:-${OUTPUT_ROOT}/v46-development-selection.json}"
WORKERS="${WORKERS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$SELECTION" ]]; then
  echo "[blocked] Missing v4.6 development gate: ${SELECTION}" >&2
  exit 4
fi
STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$SELECTION")"
ALLOW="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["allow_confirmatory_250k"])' "$SELECTION")"
if [[ "$STATUS" != "PASS" || "$ALLOW" != "True" ]]; then
  echo "[blocked] v4.6 development gate is ${STATUS}; 250k runs are forbidden." >&2
  exit 4
fi

run_cell() {
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
  echo "[single-gpu confirmatory run] ${run_id}"
  python -u train.py \
    --protocol P-C100-100 \
    --method "$method" \
    --mode confirmatory \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$run_id" \
    "${resume[@]}" \
    "$@"
}

# Paired local baseline is mandatory; paper-reported numbers are context only.
run_cell fixmatch v46-c100-100-fixmatch-seed0 --tailrow-observer
run_cell tangs-v46 v46-c100-100-tangs-rho1-seed0 --tangs-correction-rho 1
