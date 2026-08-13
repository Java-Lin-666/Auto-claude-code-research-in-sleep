#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
RUN_ID="${SMOKE_RUN_ID:-u0-smoke-tangs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"

if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip completed] ${RUN_ID}"
  exit 0
fi

resume=()
if [[ -f "${RUN_DIR}/checkpoint_last.pt" ]]; then
  resume=(--resume auto)
elif [[ -d "${RUN_DIR}" ]]; then
  echo "[blocked] Existing smoke directory has no resumable checkpoint: ${RUN_DIR}" >&2
  echo "Set SMOKE_RUN_ID to a new value after inspecting the failed directory." >&2
  exit 4
fi

python -u train.py \
  --protocol P-C100-100 \
  --method tangs \
  --mode smoke \
  --manual-seed 0 \
  --data-root "${DATA_ROOT:-./data/cache}" \
  --output-root "$OUTPUT_ROOT" \
  --run-id "$RUN_ID" \
  "${resume[@]}" \
  --diagnostic-level deep \
  --log-every 1 \
  --checkpoint-every 5 \
  --snapshot-every 5
