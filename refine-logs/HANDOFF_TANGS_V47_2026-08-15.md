# TANGS v4.7 Handoff — 2026-08-15

## Current decision

- v4.5 and v4.6 gradient surgery are permanently STOP.
- TANGS now means **Tail-ANchor-Gated Scores**.
- The old-checkpoint v4.7 C100 rescoring result is PROVISIONAL, not PASS.
- After target-server v4.7 tests and CUDA smoke pass, only one fresh integrated
  P-C100-100 50k development run is authorized.
- No 250k run is authorized until that fresh 50k gate passes.
- The required 250k matrix is C100 first, then C10-100 and STL10-20 after the
  C100 confirmatory gate passes. C100-first is a single-GPU stop-loss rule, not
  permission to omit C10 or STL10.

## Why v4.6 failed

The FixMatch observer recorded conflict on all 1,589,275 eligible tail rows
(mean cosine -0.827). Full v4.6 changed 47,414/50,000 steps but still lost
0.82 bACC point to FixMatch. This conflict is the normal sign structure of
cross-entropy target and non-target rows, not a selective harmful event. Tail
pseudo-label precision was already 98.08%, so more projection/capping is not a
valid repair.

## Frozen v4.7 method

Training remains exact FixMatch. The observer maintains each supervised tail
self-row EMA anchor. At scoring:

1. subtract `0.65 * log(labeled_prior)` from every logit;
2. take the negative weight coordinates of each tail gradient anchor as a
   normalized feature direction;
3. find the single most compatible tail direction;
4. if cosine >= 0.75, add `0.25 * [-log(prior_c)]` to that one tail logit.

Uniform-LA control: `alpha=0.85`. All constants are frozen.

## Retrospective candidate result

Same step-50k observer checkpoint, disjoint balanced C100 development set:

| Scoring | bACC | Head | Medium | Tail | GM | Dead |
|---|---:|---:|---:|---:|---:|---:|
| raw FixMatch | 36.70 | 69.39 | 34.12 | 6.67 | 5.65 | 22 |
| uniform LA | 39.42 | 67.58 | 39.18 | 11.52 | 16.11 | 7 |
| TANGS v4.7 | 40.04 | 67.45 | 36.94 | 15.82 | 20.84 | 4 |

Formal artifact:
`gradvax_experiments/results/second_try_2026-08-15/v47-development-selection.json`.
Its correct status is `PROVISIONAL`; it allows the integrated 50k validation
only and forbids confirmatory work.

Important limitation: unconstrained LA reached 41.30 bACC at alpha=1.55 but
Head fell to 63.27. The supported claim is improvement over LA under the same
two-point Head-loss guard. v4.7 is 2.24 points worse on Medium than the frozen
LA control; disclose this.

## Implemented and verified

- scorer: `gradvax_experiments/tangs/calibration.py`;
- integrated method/config: `tangs-v47`, config version 4.7;
- final summary contains raw FixMatch, uniform LA, and v4.7 performance from
  the same checkpoint;
- retrospective candidate screen: `scripts/analyze_v47_development.py`;
- fresh integrated 50k gate: `scripts/analyze_v47_integrated_development.py`;
- confirmatory gate: `scripts/analyze_v47_confirmatory.py`;
- server smoke runner: `scripts/run_v47_smoke.sh`;
- integrated 50k runner: `scripts/run_v47_development.sh`;
- serial three-dataset confirmatory runner: `scripts/run_required_v47.sh`;
- final matrix collector: `scripts/analyze_v47_matrix.py`;
- local tests: 58/58 PASS in 4.88 s;
- local CUDA smoke: `results/v47-local-smoke-v3-c100-seed0`, config
  `d0bfeac13b185468083c5d352c47b2ac6f0b435048081d5b7882f8b61d111e75`;
  zero gradient modifications, raw/LA/v4.7 and pseudo-label summaries finite;
- both shell scripts pass Git-Bash `bash -n`.

## Target-server procedure

Use a new isolated deployment directory; do not alter or delete the archived
v4.6 worktree `/root/rivermind-data/tangs/repo-v46-c5f7418` or its results.
Record the new deployment commit and environment in the tracker.

After syncing the v4.7 repository and reusing only the verified dataset cache:

```bash
cd /path/to/new-v47-deployment/gradvax_experiments
source /root/rivermind-data/tangs/venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v47_smoke.sh
```

The smoke must report 58 passing tests, five completed steps, zero gradient
modifications, and finite raw/LA/v4.7 summaries. If deployment or environment
state differs materially, also repeat the controlled checkpoint/resume check.

Ensure the retrospective screen exists at
`results/second_try_2026-08-15/v47-development-selection.json`, or set
`V47_RETROSPECTIVE_SCREEN` to its exact deployed path. It must say
`PROVISIONAL`, not PASS. Then launch the fresh 50k validation:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_v47_development.sh
```

This launches one resumable `v47-dev-c100-100-integrated-seed0` run and writes
`results/v47-integrated-development-gate.json`. Observe the complete 50,000
steps and inspect the gate. Only if it is PASS, launch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_required_v47.sh
```

The required runner executes serially: C100 250k, checks
`v47-confirmatory-c100-gate.json`, then—only after C100 PASS—C10-100 250k and
STL10-20 250k. It finishes with `v47-required-matrix.json`. Never launch
separate raw FixMatch or uniform-LA training runs. P-STL10-10 is an optional
extension after this required matrix.

## Development and confirmatory decisions

The real development PASS uses the fresh integrated C100 50k run and the same
efficacy thresholds below, plus method/mode/step/split/frozen-parameter and
zero-gradient-write integrity checks. A development STOP forbids all 250k
runs.

PASS requires:

- v4.7 vs raw: bACC >= +1.0, Tail >= +2.0, Head >= -2.0, GM non-decreasing,
  dead classes non-increasing;
- v4.7 vs uniform LA: bACC >= +0.5 and Tail >= +2.0;
- seed 0, 250k steps, exact frozen parameters/hashes, final EMA only, and zero
  training-gradient modifications.

If C100 confirmatory is STOP, do not tune on the official test set and do not
run C10/STL. If it is PASS, the already-frozen C10-100 and STL10-20 transfers
are required and run one at a time on the single GPU. The final matrix status
`COMPLETE` means artifacts passed integrity checks; it is deliberately not
called an efficacy PASS and still requires an independent claims audit.
