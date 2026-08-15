#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
WORKERS="${WORKERS:-4}"
RUN_ID="${V47_SMOKE_RUN_ID:-v47-server-smoke-c100-seed0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m pytest -q

if [[ ! -f "${OUTPUT_ROOT}/${RUN_ID}/summary.json" ]]; then
  python -u train.py \
    --protocol P-C100-100 \
    --method tangs-v47 \
    --mode smoke \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --run-id "$RUN_ID" \
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
diagnostics = json.loads(
    (run / "gradient_diagnostics.json").read_text(encoding="utf-8")
)
if summary["final_step"] != 5:
    raise SystemExit("v4.7 smoke did not finish five steps")
if diagnostics["counts"].get("nonzero_gradient_modification_steps", 0) != 0:
    raise SystemExit("v4.7 must never modify a training gradient")
for block in (
    "performance",
    "raw_performance",
    "uniform_logit_adjustment_performance",
):
    for key in ("balanced_accuracy", "head_accuracy", "tail_accuracy"):
        if not math.isfinite(float(summary[block][key])):
            raise SystemExit(f"non-finite {block}.{key}")
if summary.get("adjusted_pseudo_labels") is None:
    raise SystemExit("missing adjusted pseudo-label diagnostics")
print(f"[pass] v4.7 CUDA smoke: {run}")
PY
