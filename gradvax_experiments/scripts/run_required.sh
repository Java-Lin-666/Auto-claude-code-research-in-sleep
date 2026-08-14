#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_V45_REQUIRED:-0}" != "1" ]]; then
  echo "[blocked] This is the archived v4.5 matrix and may resume obsolete runs." >&2
  echo "Use scripts/run_v46_development.sh, then scripts/run_required_v46.sh." >&2
  echo "Set ALLOW_LEGACY_V45_REQUIRED=1 only for an explicit audit reproduction." >&2
  exit 4
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
PILOT_GATE="${PILOT_GATE:-${OUTPUT_ROOT}/minimal-gate-c10-100-20000step.json}"
DEV_SELECTION="${DEV_SELECTION:-${OUTPUT_ROOT}/development-selection.json}"

if [[ ! -f "$PILOT_GATE" ]]; then
  echo "[blocked] Missing minimal-pilot gate report: ${PILOT_GATE}" >&2
  echo "Run first: python scripts/run_minimal_gate.py" >&2
  exit 4
fi
GATE_STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$PILOT_GATE")"
if [[ "$GATE_STATUS" != "PASS" ]]; then
  echo "[blocked] Minimal-pilot gate is ${GATE_STATUS}, not PASS: ${PILOT_GATE}" >&2
  exit 4
fi

if [[ ! -f "$DEV_SELECTION" ]]; then
  echo "[blocked] Missing 50k development freeze: ${DEV_SELECTION}" >&2
  echo "Run first: bash scripts/run_development.sh" >&2
  exit 4
fi
DEV_STATUS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$DEV_SELECTION")"
if [[ "$DEV_STATUS" != "PASS" ]]; then
  echo "[blocked] Development freeze is ${DEV_STATUS}, not PASS: ${DEV_SELECTION}" >&2
  exit 4
fi
FROZEN_TAU="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["frozen"]["tangs_tau"])' "$DEV_SELECTION")"
FROZEN_HEAD_WEIGHT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["frozen"]["head_weight"])' "$DEV_SELECTION")"
TANGS_TAU="${TANGS_TAU:-$FROZEN_TAU}"
HEAD_WEIGHT="${HEAD_WEIGHT:-$FROZEN_HEAD_WEIGHT}"
python - "$TANGS_TAU" "$FROZEN_TAU" "$HEAD_WEIGHT" "$FROZEN_HEAD_WEIGHT" <<'PY'
import math
import sys


def same(left: str, right: str) -> bool:
    a, b = float(left), float(right)
    return (math.isinf(a) and math.isinf(b)) or math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)


if not same(sys.argv[1], sys.argv[2]):
    raise SystemExit(f"TANGS_TAU={sys.argv[1]} differs from frozen value {sys.argv[2]}")
if not same(sys.argv[3], sys.argv[4]):
    raise SystemExit(f"HEAD_WEIGHT={sys.argv[3]} differs from frozen value {sys.argv[4]}")
PY
echo "[frozen] TANGS_TAU=${TANGS_TAU}; HEAD_WEIGHT=${HEAD_WEIGHT}"

run_cell() {
  local protocol="$1"
  local method="$2"
  local run_id="$3"
  shift 3
  local run_dir="${OUTPUT_ROOT}/${run_id}"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[skip completed] ${run_id}"
    return
  fi
  local resume=()
  if [[ -f "${run_dir}/checkpoint_last.pt" ]]; then
    resume=(--resume auto)
  fi
  echo "[run] ${run_id}"
  python -u train.py \
    --protocol "$protocol" \
    --method "$method" \
    --mode confirmatory \
    --manual-seed 0 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    --tangs-tau "$TANGS_TAU" \
    "${resume[@]}" \
    "$@"
}

# C1: three local TANGS rows; FixMatch-family baselines are paper-reported.
run_cell P-C10-100 tangs c1-c10-100-tangs-seed0
run_cell P-C100-100 tangs c1-c100-100-tangs-seed0 --diagnostic-level deep
run_cell P-STL10-20 tangs c1-stl10-20-tangs-seed0

# B1: target-only oracle pair.
run_cell P-C100-100 oracle-fixmatch b1-c100-100-oracle-fixmatch-seed0
run_cell P-C100-100 oracle-tangs b1-c100-100-oracle-tangs-seed0

# C2: direct magnitude/direction controls.
run_cell P-C100-100 head-downweight c2-c100-100-head-downweight-seed0 \
  --head-weight "$HEAD_WEIGHT"
run_cell P-C100-100 head-clip c2-c100-100-head-clip-seed0
run_cell P-C100-100 pcgrad c2-c100-100-pcgrad-seed0

# D1: no-cap is the same locked operator as C2-PCG and reuses that artifact.
run_cell P-C100-100 no-projection d1-c100-100-no-projection-seed0
run_cell P-C100-100 instant-anchor d1-c100-100-instant-anchor-seed0
