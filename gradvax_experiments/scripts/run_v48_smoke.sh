#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
WORKERS="${WORKERS:-4}"
RUN_ID="${V48_SMOKE_RUN_ID:-v48-server-smoke-c100-seed1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m pytest -q

if [[ ! -f "${OUTPUT_ROOT}/${RUN_ID}/summary.json" ]]; then
  python -u train.py \
    --protocol P-C100-100 \
    --method tangs-v48 \
    --mode smoke \
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
    --log-every 1 \
    --checkpoint-every 5 \
    --snapshot-every 5
else
  echo "[skip completed] ${RUN_ID}"
fi

python - "$OUTPUT_ROOT/$RUN_ID" <<'PY'
import json
import math
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
if any(scientific.get(key) != value for key, value in expected.items()):
    raise SystemExit("v4.8 smoke scientific configuration mismatch")
if summary["final_step"] != 5:
    raise SystemExit("v4.8 smoke did not finish five steps")
if diagnostics["counts"].get("nonzero_gradient_modification_steps", 0) != 0:
    raise SystemExit("v4.8 must never modify a training gradient")
for block in ("performance", "raw_performance", "uniform_logit_adjustment_performance"):
    for key in ("balanced_accuracy", "head_accuracy", "tail_accuracy"):
        if not math.isfinite(float(summary[block][key])):
            raise SystemExit(f"non-finite {block}.{key}")
print(f"[pass] v4.8 CUDA smoke: {run}")
PY
