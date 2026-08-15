#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
WORKERS="${WORKERS:-4}"
SELECTION="${V48_SELECTION:-${OUTPUT_ROOT}/third_try_2026-08-15/v48-retrospective-selection.json}"
SMOKE_RUN_ID="${V48_SMOKE_RUN_ID:-v48-server-smoke-c100-seed1}"
RUN_ID="${V48_HOLDOUT_RUN_ID:-v48-holdout-c100-100-seed1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$SELECTION" ]]; then
  echo "[blocked] Missing v4.8 retrospective selection: ${SELECTION}" >&2
  exit 4
fi
readarray -t GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_seed1_holdout_50k"]); print(d["allow_confirmatory_250k"])' "$SELECTION"
)
if [[ "${GATE[0]}" != "PROVISIONAL" || "${GATE[1]}" != "True" || "${GATE[2]}" != "False" ]]; then
  echo "[blocked] v4.8 selection does not authorize only the seed-1 holdout." >&2
  exit 4
fi

SMOKE_DIR="${OUTPUT_ROOT}/${SMOKE_RUN_ID}"
if [[ ! -f "${SMOKE_DIR}/summary.json" ]]; then
  echo "[blocked] Run scripts/run_v48_smoke.sh first." >&2
  exit 4
fi
python - "$SMOKE_DIR" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
diagnostics = json.loads((run / "gradient_diagnostics.json").read_text(encoding="utf-8"))
scientific = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))["scientific_config"]
expected = {
    "method": "tangs-v48",
    "mode": "smoke",
    "protocol_id": "P-C100-100",
    "manual_seed": 1,
    "score_uniform_la_alpha": 0.80,
    "score_base_alpha": 0.70,
    "score_extra_tail_alpha": 0.40,
    "score_anchor_threshold": 0.775,
    "score_anchor_margin": 0.05,
    "score_confidence_ceiling": 0.90,
}
if summary.get("final_step") != 5:
    raise SystemExit("[blocked] v4.8 server smoke is incomplete")
if any(scientific.get(key) != value for key, value in expected.items()):
    raise SystemExit("[blocked] v4.8 server smoke configuration mismatch")
if diagnostics["counts"].get("nonzero_gradient_modification_steps", 0) != 0:
    raise SystemExit("[blocked] v4.8 server smoke modified gradients")
print(f"[verified] v4.8 target-server smoke: {run}")
PY

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip completed] ${RUN_ID}"
else
  RESUME=()
  if [[ -f "${RUN_DIR}/checkpoint_last.pt" ]]; then
    RESUME=(--resume auto)
  fi
  echo "[single-gpu v4.8 seed-1 holdout, 50000 steps] ${RUN_ID}"
  python -u train.py \
    --protocol P-C100-100 \
    --method tangs-v48 \
    --mode development \
    --manual-seed 1 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$RUN_ID" \
    --score-uniform-la-alpha 0.80 \
    --score-base-alpha 0.70 \
    --score-extra-tail-alpha 0.40 \
    --score-anchor-threshold 0.775 \
    --score-anchor-margin 0.05 \
    --score-confidence-ceiling 0.90 \
    --checkpoint-every 500 \
    --snapshot-every 50000 \
    "${RESUME[@]}"
fi

python scripts/analyze_v48_holdout.py \
  --run-dir "$RUN_DIR" \
  --selection "$SELECTION" \
  --output "${OUTPUT_ROOT}/v48-holdout-development-gate.json"
