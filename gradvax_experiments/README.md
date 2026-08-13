# TANGS experiments (protocol v4.5)

This directory is a self-contained server package for the paper-facing **TANGS**
experiments. The training substrate is derived from CDMAD commit
`7cd732b4615b9d94934a9197e69c6775496fb5ee`; it does not import the sibling
`CDMAD/` checkout at runtime.

## Scientific lock

- `manualSeed=0`, one run per minimum-plan cell.
- Exact CDMAD WRN-28-2, including the unused rotation head and executed
  BatchNorm defaults.
- Adam at constant `1.5e-3`, no scheduler, FP32 only.
- Labeled batch 32, unlabeled batch 64, 500 steps per epoch, 250,000 steps.
- Two independent strong views using RandAugment(3,4) and Cutout(16).
- Evaluation-model EMA 0.999 with the released `WeightEMA` behavior preserved.
- The local TANGS FixMatch substrate uses detached hard weak-view argmax targets and confidence 0.95.
- CDMAD white-image logit subtraction is disabled in TANGS and its local controls.
- Confirmatory evidence uses the final EMA checkpoint only.

The research method is not changed: TANGS replaces only the exact accepted
predicted-head contribution to `model.output` weight and bias. The backbone and
the rest of the base classifier gradient remain untouched.

## Environment

Use Python 3.11. Install the CUDA build of PyTorch that matches the server, then
the small project dependency set:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

For another CUDA runtime, change only the PyTorch wheel index. Record that
environment in the run manifest; do not enable AMP.

The runnable TANGS implementation vendors the required WRN, augmentations, data
logic, and EMA behavior under `tangs/`; it does not import the sibling `CDMAD/`
checkout at runtime. The sibling checkout is needed only for the source-parity
audit. Because it is currently a nested Git repository, a parent-repository
clone does not include it unless it is transferred separately. To reconstruct
the audit checkout on the server, run these commands from
`gradvax_experiments/`:

```bash
git clone https://github.com/LeeHyuck/CDMAD.git ../CDMAD
git -C ../CDMAD checkout 7cd732b4615b9d94934a9197e69c6775496fb5ee
```

Training and smoke tests remain functional without that optional audit checkout.

For the complete three-dataset matrix, provision at least 16 GiB system RAM and
about 10 GiB free disk before accounting for long-term logs. The local dataset
cache occupies about 6.0 GiB with compressed archives retained. A full matrix's
rolling plus five 50k snapshots is estimated at about 1.4 GiB for ten runs.
STL-10 is the highest host-memory path because the released protocol combines
the 100,000-image pool with appended labeled images; its v4.5 smoke reached
914.74 MiB peak CUDA allocation with the locked batch sizes.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test.sh
```

The smoke run is explicitly non-confirmatory and uses five optimizer steps. It
skips a completed smoke run and resumes an interrupted one from its rolling
checkpoint. Set `SMOKE_RUN_ID` when a separate fresh smoke artifact is needed.

## Minimal idea gate

Before the 50k development sweep or any 250k run, launch the paired 20k pilot.
The Python launcher is the canonical Windows/Linux entry point and uses the
currently active interpreter:

```bash
python scripts/run_minimal_gate.py --dry-run
python scripts/run_minimal_gate.py
```

On this Windows workstation, run it from PowerShell with the verified virtual
environment:

```powershell
Set-Location D:\Auto-claude-code-research-in-sleep\gradvax_experiments
$env:CUDA_VISIBLE_DEVICES = "0"
& C:\lintao\envs\fixmatch\python.exe .\scripts\run_minimal_gate.py --dry-run
& C:\lintao\envs\fixmatch\python.exe .\scripts\run_minimal_gate.py
```

Linux servers may use the same Python command after activating the server
environment. `bash scripts/run_minimal_gate.sh` remains as a thin wrapper.

This runs FixMatch and TANGS (`tau=5`) on the identical DEV-C10-100 seed-0
split, using the held-out development validation set rather than the test set.
It costs 40k optimizer steps in total, or 0.16 of one 250k run. The analyzer
writes `results/minimal-gate-c10-100-20000step.json` with one of three states:

- `PASS`: mechanism is active, validation bACC does not collapse, and early
  bACC or tail accuracy has a positive signal; the 50k freeze becomes eligible
  only after every remaining U0 row passes and G0 is marked DONE.
- `HOLD`: mechanism is active and stable but the early performance signal is
  flat; extend only the paired comparison to 50k.
- `STOP`: pairing, mechanism, or non-collapse checks failed; inspect before
  spending on the sweep or full matrix.

The pilot FixMatch run is a development safeguard only. It is not a main-table
row and does not change the CDMAD-style citation policy. `run_required.sh`
refuses to launch unless the default gate report is `PASS`; set `PILOT_GATE`
only when the report is stored elsewhere.

The canonical PASS report
`results/minimal-gate-c10-100-20000step.json` is intentionally allowed through
the repository ignore rules so a clean server clone does not silently lose the
completed P0 gate. Large run directories, logs, and checkpoints remain ignored.

## Development freeze

Tune only on the registered CIFAR-10 development split:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_development.sh
```

Run this only after the minimal gate passes **and G0 is DONE**. The launcher
checks the machine-readable P0 report but does not parse the Markdown U0 ledger,
so the operator must verify G0 before launch. It launches the four registered
cap candidates, three head-downweight candidates, and one paired 50k FixMatch
development reference. The reference exists only to measure the TANGS
Head/Overall trade-off; it is not a paper baseline row. The final command writes
`results/development-selection.json`.

The freeze passes only when at least one TANGS candidate improves validation
bACC by at least 0.25 pp while keeping GM within 0.25 pp, Head within 5 pp, and
Overall within 3 pp of the paired FixMatch reference. Among passing candidates,
the selector maximizes bACC and then uses GM, Head retention, Overall retention,
and the less aggressive `tau` as tie-breakers. These thresholds are frozen
before the 50k sweep and must not be changed after seeing its results.

The launcher writes `development-tangs-screen.json` immediately after the four
`tau` runs. A failed screen stops the script before the three head-downweight
controls, saving 150k unnecessary optimizer steps. A passing screen continues
to the controls and writes the final `development-selection.json` freeze.

## Required matrix

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_required.sh
```

The script reads `TANGS_TAU` and `HEAD_WEIGHT` from the PASS development freeze,
rejects conflicting environment overrides, runs the 10 registered jobs
sequentially, and resumes an interrupted job from its rolling checkpoint. Use
only one copy of this unmodified script for a shared output directory. Parallel
execution requires an explicitly partitioned, non-overlapping run list; two
copies of the full script can race on the same run directories.
The three plain FixMatch main-table rows are taken from the audited CDMAD paper
tables and are therefore not launched by this script.
The instrumented P-C100-100 TANGS row also emits synchronized
`extra_gradient_surgery_fraction`, allocated/reserved memory, and
`known_tangs_tensor_fraction_of_peak` fields for E1 without another full run.



Run one cell directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --protocol P-C100-100 \
  --method tangs \
  --manual-seed 0 \
  --data-root ./data/cache \
  --output-root ./results \
  --tangs-tau 5 \
  --diagnostic-level deep
```

Available methods are:

`fixmatch`, `tangs`, `oracle-fixmatch`, `oracle-tangs`,
`head-downweight`, `head-clip`, `pcgrad`, `no-cap`,
`no-projection`, and `instant-anchor`.


`pcgrad` and `no-cap` are retained as CLI aliases but implement the same locked
v4.5 operation. The required matrix launches `pcgrad` once and reuses that
artifact for the D1 no-cap ablation.

## Artifacts and monitoring

Every run directory contains:

- `resolved_config.json` and a SHA-256 configuration hash;
- `split_manifest.json`, full split indices, overlap counts, and hashes;
- `run_manifest.json`, including Git/environment/GPU provenance;
- `train_metrics.jsonl` and `status.json`;
- rolling and 50k-step checkpoints;
- `gradient_diagnostics.json`;
- final `summary.json`.

Inspect server progress without importing PyTorch:

```bash
python scripts/status.py --results ./results
python scripts/status.py --results ./results --watch 30
```

Evaluate the predeclared 50k-step EMA snapshots only after a run is complete:

```bash
python scripts/evaluate_checkpoints.py \
  --run-dir ./results/c1-c100-100-tangs-seed0 \
  --all-snapshots --data-root ./data/cache
```

Resume a single interrupted run with the same command plus `--resume auto`.
Model, optimizer, EMA, anchor, and RNG states are restored. PyTorch DataLoader
worker/prefetch state is not serializable, so a resumed process is operationally
safe but not bit-exact at the batch-stream boundary. Prefer uninterrupted
confirmatory runs; every resume is recorded in `events.jsonl`, `summary.json`,
and the run manifest's `resume_history`.
