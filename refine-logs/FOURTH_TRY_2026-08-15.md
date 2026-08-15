# TANGS v4.8 Fourth Try — Fresh Seed-1 Holdout Preservation Record

*Recorded: 2026-08-15*
*Disposition: normal 50,000-step completion; machine-gate STOP; no 250k experiment launched.*

## Scope

This record closes the one authorized fresh TANGS v4.8 C100 development
holdout. It preserves the target-server smoke, the full seed-1 run, raw/LA/v4.8
comparison, machine gate, logs, checkpoints, and the retrospective-selection
authorization that preceded it. The official test set was not read.

## Execution provenance

- Deployment commit: `ccecb736ac44067b0fd8a2e571411443ee3880e8`.
- Server worktree: `/root/rivermind-data/tangs/repo-v48-ccecb73-bundle/gradvax_experiments`.
- Target environment: RTX 4090, CUDA 12.1, PyTorch 2.3.1+cu121, torchvision
  0.18.1+cu121.
- Target smoke: 61/61 tests in 4.86 s and five finite CUDA steps, with zero
  gradient modifications. Smoke config hash:
  `fbf0c3dab4cb5ac02e7a9b39dc7a02a67fb6fab39f51cac612d6fd707a56ade6`.
- Holdout run: `v48-holdout-c100-100-seed1`, `P-C100-100`, development mode,
  `manual_seed=1`, 50,000/50,000 steps, final EMA evaluation over 5,000
  development examples.
- Holdout config hash:
  `4b75e3813c7c6328f96c08652738ace92fd70f34bd8247520c02e7e789030f87`.
- Holdout split hash:
  `5d0e47e53ac3342156acfa336ac7eb1eb86a163d9c6424ceabfd6c31995034ae`.

The worktree reported only the required untracked `CDMAD` and data-cache
symlinks; the tracked source commit above is exact. No W&B run was required or
used; JSON, checkpoints, manifests, and logs are the audit evidence.

## Machine-gate outcome

Every integrity check is true: seed 1; unseen selection split; configured and
final 50,000 steps; full diagnostics and observer coverage; 5,000-example
development evaluation; development role; disjoint active-unlabeled validation;
all six scoring constants frozen; provisional selection allowed this holdout;
and zero training-gradient modifications.

| Comparator / scorer | bACC | Head | Medium | Tail | GM | Dead |
|---|---:|---:|---:|---:|---:|---:|
| Raw FixMatch | 36.32 | 69.52 | 34.35 | 5.15 | 8.47 | 14 |
| Uniform LA-0.80 | 38.64 | 67.82 | 38.94 | 9.15 | 15.35 | 7 |
| TANGS v4.8 | **38.74** | 66.97 | 37.29 | **12.00** | **18.26** | **5** |

TANGS v4.8 improves on raw by +2.42 bACC, +6.85 Tail, +9.79 GM, and nine
fewer dead classes. It improves on LA by +0.10 bACC, +2.85 Tail, +2.91 GM,
and two fewer dead classes. The frozen gate is nevertheless `STOP` because:

- bACC versus uniform LA is +0.10 pp, below the required +0.50 pp;
- Head versus raw is -2.55 pp, beyond the allowed -2.00 pp.

The gate sets `allow_confirmatory_matrix=false`. This is a valid negative
development result, not an infrastructure failure or an interrupted run.

## Preserved artifacts and verification

The server-side, non-overwriting archive is:

`/root/rivermind-data/tangs/archives/tangs-v48-fourth-try-20260815T094207Z.tar.gz`

It contains 27 members, has 88,386,257 bytes, and SHA-256:

`6d3ce7cdef8cac318487e55e37d859bcf64b97d66ec64182d7c6c05150d052fc`

The identically hashed local archive is retained at:

`tmp/tangs-v48-fourth-try-20260815T094207Z.tar.gz`

It was extracted without overwriting into:

`gradvax_experiments/results/fourth_try_2026-08-15/`

The local snapshot contains 25 files, including both final checkpoints,
`development_metrics.jsonl`, `events.jsonl`, `gradient_diagnostics.json`,
resolved/run/split manifests, status and summary JSON, the target smoke run,
both server logs, the gate JSON, and the frozen v4.8 selection artifact.

Primary local raw artifact:

`gradvax_experiments/results/fourth_try_2026-08-15/v48-holdout-c100-100-seed1/`

Machine gate:

`gradvax_experiments/results/fourth_try_2026-08-15/v48-holdout-development-gate.json`

## Next state

No v4.8 250k C100, C10, or STL10-20 run is authorized, and no replacement
holdout is authorized by this record. The monitoring heartbeat was retired
after normal completion. Any further assessment must be an independent audit
by a different model family over the code and raw paths above; it may assess
integrity and the negative outcome, but cannot turn this frozen STOP into a
PASS or launch the blocked matrix.
