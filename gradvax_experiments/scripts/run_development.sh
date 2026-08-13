#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-./data/cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
PILOT_GATE="${PILOT_GATE:-${OUTPUT_ROOT}/minimal-gate-c10-100-20000step.json}"

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
  python -u train.py \
    --protocol P-C10-100 \
    --method "$method" \
    --mode development \
    --manual-seed 0 \
    --max-steps 50000 \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --run-id "$run_id" \
    "${resume[@]}" \
    "$@"
}

# The 50k FixMatch row is a development-only balance reference. It never enters
# the CDMAD-style paper table.
run_dev fixmatch "dev-c10-100-fixmatch-seed0"

# P0 showed an excessive Head/Overall trade-off at tau=5. Evaluate the less
# aggressive candidates first for operational feedback, but complete the full
# registered screen before freezing a value.
for tau in 10 inf 5 2; do
  run_dev tangs "dev-c10-100-tangs-tau${tau}-seed0" --tangs-tau "$tau"
done

# Stop here when no tau retains enough Head/Overall performance. This avoids
# spending three additional 50k runs on controls for a method that cannot yet
# enter the confirmatory matrix.
python scripts/analyze_development.py \
  --results "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/development-tangs-screen.json" \
  --tangs-only

for weight in 0.25 0.5 0.75; do
  run_dev head-downweight "dev-c10-100-head-weight${weight}-seed0" \
    --head-weight "$weight"
done

python scripts/analyze_development.py \
  --results "$OUTPUT_ROOT" \
  --output "${OUTPUT_ROOT}/development-selection.json"
