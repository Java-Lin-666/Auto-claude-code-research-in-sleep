# TANGS v4.8 Handoff — 2026-08-15

## Current decision

- v4.5, v4.6, and v4.7 are STOP and remain reproducible under their original
  method identifiers.
- v4.8 is a separately registered selective score-calibration method.
- Two retrospective 50k realizations support v4.8 only as PROVISIONAL.
- Only target-server smoke plus one fresh seed-1 C100 50k holdout is currently
  authorized.
- No 250k run is authorized until the seed-1 holdout machine gate passes.

## Frozen method

Training is exact FixMatch plus a classwise supervised tail-anchor observer;
the controller must report zero training-gradient modifications. At inference:

1. apply labeled-prior logit adjustment with base alpha 0.70;
2. find the two most similar valid tail-anchor feature directions;
3. require best cosine >= 0.775 and best-minus-second cosine >= 0.05;
4. require raw maximum softmax confidence <= 0.90;
5. add `0.40 * [-log(prior_c)]` only to the best eligible tail class.

The paired uniform-LA control uses alpha 0.80.

## Why this changes v4.7

The genuine v4.7 server 50k run reached 39.80 bACC and 16.42 Tail but failed
Head (-2.30 vs raw) and bACC-versus-LA (+0.38) guards. Its absolute anchor gate
made approximately 67.56% of examples eligible. v4.8 adds anchor-identity
separation and a confidence ceiling, reducing eligibility to about 12% and
prediction flips to about 5%.

The frozen v4.8 scorer reaches 39.86/39.84 bACC on the fresh/archived seed-0
checkpoints, with Head drops -1.82/-1.58 and +0.64 bACC over head-compliant
LA-0.80 on both. Twenty-one bounded-grid candidates pass all original gates on
both realizations. These are retrospective development values, not paper
results and not a validation PASS.

## Implemented files

- method/config: `tangs/calibration.py`, `tangs/config.py`, `tangs/trainer.py`;
- robust search: `scripts/analyze_v48_logit_gap.py`;
- production retrospective freeze: `scripts/analyze_v48_retrospective.py`;
- seed-1 holdout gate: `scripts/analyze_v48_holdout.py`;
- target smoke: `scripts/run_v48_smoke.sh`;
- seed-1 holdout runner: `scripts/run_v48_holdout.sh`;
- confirmatory gate/matrix: `scripts/analyze_v48_confirmatory.py`,
  `scripts/analyze_v48_matrix.py`, `scripts/run_required_v48.sh`;
- PROVISIONAL artifact:
  `results/third_try_2026-08-15/v48-retrospective-selection.json`.

Local verification: 61/61 tests pass. Five-step RTX 4060 CUDA smoke
`results/v48-local-smoke-c100-seed1` completed with config
`a5e4e6bffb726936cb095950fd961b0d08a83dee766197399c273c993fea4405`,
five observer records, zero gradient modifications, and finite raw/LA/v4.8
summaries.

## Target-server procedure

Deploy to a new isolated directory; do not overwrite the archived v4.6 or
v4.7 server worktrees. Reuse only the verified environment and data cache.
Ensure the PROVISIONAL v4.8 selection artifact is present at its default path,
then run:

```bash
cd /path/to/new-v48-deployment/gradvax_experiments
source /root/rivermind-data/tangs/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v48_smoke.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v48_holdout.sh
```

The holdout run ID is `v48-holdout-c100-100-seed1`; it is 50,000 steps and is
resumable. The analyzer writes `results/v48-holdout-development-gate.json`.
PASS requires seed 1, an unseen split hash, all frozen parameters, complete
diagnostics, zero gradient modifications, and every unchanged efficacy rule.

If STOP, preserve artifacts and do not tune or launch 250k. If PASS, report
the result before launching `scripts/run_required_v48.sh`. That later script
runs C100 seed 0 first, then the required C10-100 and STL10-20 seed-0 cells
only after the C100 gate passes. P-STL10-10 remains optional.
