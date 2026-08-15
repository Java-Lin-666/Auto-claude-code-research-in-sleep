#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
WORKERS="${WORKERS:-4}"
SCREEN="${V47_RETROSPECTIVE_SCREEN:-${OUTPUT_ROOT}/second_try_2026-08-15/v47-development-selection.json}"
SMOKE_RUN_ID="${V47_SMOKE_RUN_ID:-v47-server-smoke-c100-seed0}"
RUN_ID="${V47_DEVELOPMENT_RUN_ID:-v47-dev-c100-100-integrated-seed0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$SCREEN" ]]; then
  echo "[blocked] Missing retrospective v4.7 candidate screen: ${SCREEN}" >&2
  exit 4
fi
readarray -t SCREEN_GATE < <(
  python -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); print(d["status"]); print(d["allow_integrated_50k_validation"])' "$SCREEN"
)
if [[ "${SCREEN_GATE[0]}" != "PROVISIONAL" || "${SCREEN_GATE[1]}" != "True" ]]; then
  echo "[blocked] Retrospective screen is ${SCREEN_GATE[0]}; integrated 50k validation is forbidden." >&2
  exit 4
fi

SMOKE_DIR="${OUTPUT_ROOT}/${SMOKE_RUN_ID}"
python - "$SMOKE_DIR" <<'PY'
import json
import math
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
try:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (run / "gradient_diagnostics.json").read_text(encoding="utf-8")
    )
    resolved = json.loads(
        (run / "resolved_config.json").read_text(encoding="utf-8")
    )["scientific_config"]
except FileNotFoundError as exc:
    raise SystemExit(f"[blocked] Run scripts/run_v47_smoke.sh first: {exc}")
if summary.get("final_step") != 5:
    raise SystemExit("[blocked] Target-server v4.7 smoke is incomplete")
if (
    resolved.get("method") != "tangs-v47"
    or resolved.get("mode") != "smoke"
    or resolved.get("protocol_id") != "P-C100-100"
):
    raise SystemExit("[blocked] Target-server smoke has the wrong method/mode/protocol")
expected = {
    "score_uniform_la_alpha": 0.85,
    "score_base_alpha": 0.65,
    "score_extra_tail_alpha": 0.25,
    "score_anchor_threshold": 0.75,
}
if any(resolved.get(key) != value for key, value in expected.items()):
    raise SystemExit("[blocked] Target-server smoke did not use frozen v4.7 constants")
if diagnostics["counts"].get("nonzero_gradient_modification_steps", 0) != 0:
    raise SystemExit("[blocked] Target-server v4.7 smoke modified gradients")
for block in ("performance", "raw_performance", "uniform_logit_adjustment_performance"):
    if not math.isfinite(float(summary[block]["balanced_accuracy"])):
        raise SystemExit(f"[blocked] Target-server smoke has non-finite {block}")
print(f"[verified] target-server v4.7 smoke: {run}")
PY

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -f "${RUN_DIR}/summary.json" ]]; then
  echo "[skip completed] ${RUN_ID}"
else
  RESUME=()
  if [[ -f "${RUN_DIR}/checkpoint_last.pt" ]]; then
    RESUME=(--resume auto)
  fi

  echo "[single-gpu integrated development run, 50000 steps] ${RUN_ID}"
  python -u train.py \
    --protocol P-C100-100 \
    --method tangs-v47 \
    --mode development \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$RUN_ID" \
    --score-uniform-la-alpha 0.85 \
    --score-base-alpha 0.65 \
    --score-extra-tail-alpha 0.25 \
    --score-anchor-threshold 0.75 \
    --checkpoint-every 500 \
    --snapshot-every 50000 \
    "${RESUME[@]}"
fi

python scripts/analyze_v47_integrated_development.py \
  --run-dir "$RUN_DIR" \
  --output "${OUTPUT_ROOT}/v47-integrated-development-gate.json"
