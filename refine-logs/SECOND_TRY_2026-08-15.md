# TANGS v4.6 C100 Development — Second Try

Status: completed serially on 2026-08-15. This is a development-only execution
record, not paper evidence or an independent integrity/claims review. The
pre-registered machine-readable gate returned **STOP**; no 250k experiment was
started.

## Stop state

At preservation time there was no queue, training, or analysis process on the
server, and the RTX 4090 reported 0% utilization and 0 MiB allocated. No stop
signal was required or sent. The queue had exited after completing all five
registered 50k cells and writing its selection artifact. No NaN, OOM,
traceback, or killed-process event was present in the final queue log.

## Local raw-artifact snapshot

- Local archive directory:
  `gradvax_experiments/results/second_try_2026-08-15/`
- Source worktree:
  `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/`
  at deployment commit `c5f74189d81a00402cd183a49cc4dc74d3ddf5b0`.
- Included: all five run directories (including `summary.json`,
  `development_metrics.jsonl`, `train_metrics.jsonl`, and
  `checkpoint_last.pt`), `v46-development-selection.json`, and
  `v46-development-queue-resume-20260814T170718Z.log`.
- Archive size: 57 files, 265,000,393 bytes.
- Integrity: SHA-256 matched the server for all 22 primary files: five
  summaries, five development JSONLs, five train JSONLs, five last
  checkpoints, the selection JSON, and the queue log.

## Machine-verifiable run state

All cells used `P-C100-100`, seed 0, 50,000 steps, and the same development
split. Every row completed and retains a final checkpoint locally and remotely.

| Run ID | Config hash | bACC | Head | Medium | Tail | GM | Worst | Dead classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `v46-dev-c100-100-fixmatch-observer-seed0` | `7a15365bf8b73ffa88f6164652f521a3efa28cb32b02d4bd8469e95bb2c23948` | 36.70 | 69.39 | 34.12 | 6.67 | 5.65 | 0 | 22 |
| `v46-dev-c100-100-legacy-pcgrad-seed0` | `b8963e78cd3e5eb384fa5e3e86f31cbf38b97585796cc58094a64ce894a03bd7` | 35.96 | 67.94 | 32.00 | 8.06 | 8.61 | 0 | 14 |
| `v46-dev-c100-100-tailrow-group-seed0` | `ca52cbc06f03578d6c226ecfe30b10c56bd0cb9eca2270e510004128554a94f3` | 36.16 | 68.42 | 33.88 | 6.24 | 6.13 | 0 | 20 |
| `v46-dev-c100-100-tailrow-classwise-seed0` | `86b6d5c766d999766bcb853de4a5a394742b266f1ceeeb034d8fd134f781dc40` | 35.40 | 66.85 | 31.94 | 7.52 | 8.24 | 0 | 15 |
| `v46-dev-c100-100-tangs-rho1-seed0` | `99a83cc295ad35f1de4e9fcc96a14d527282c90f664e4e84e9be79061bc57d3f` | 35.88 | 67.88 | 33.35 | 6.48 | 6.81 | 0 | 18 |

All five summaries record split hash
`38fc04c6d1d0dd9109879bba0445b40dd264f85b7ef198c640a68c45dac57ca9`
and partition hash
`01b501e58a2ee7b38669a235098585d0e55ee3f84d952129aeff0a2a7e344bd0`.

## Gate outcome

`v46-development-selection.json` records `status: STOP` and
`allow_confirmatory_250k: false`.

- G1 stopped: full v4.6 versus FixMatch changed bACC by -0.82 pp and tail by
  -0.18 pp, below the required +1.0 pp and +2.0 pp. The head (-1.52 pp) and
  GM (+1.15 pp) guards held.
- G2 passed: 18 dead classes versus the paired FixMatch's 22.
- G3 stopped: full v4.6 bACC was -0.08 pp versus legacy `tau=inf`, rather
  than at least +0.5 pp.
- G4 passed: rho=1 was +0.48 pp bACC versus unbounded classwise, exceeding
  +0.25 pp.
- G5 passed: the recorded split and partition hashes agree.

The STOP is an efficacy decision under the registered rules, not an
infrastructure failure. Do not start 250k, resume v4.5 STL, loosen the gate,
or relabel this development result as a paper claim. A different model family
must independently audit the raw local archive and code before any separately
versioned method-revision decision.

## Post-STOP diagnosis and v4.7 reuse (2026-08-15)

No v4.6 result above was changed. Additional read-only analysis of the saved
observer artifact established:

- observer conflict was 1,589,275 / 1,589,275 eligible tail rows (100%), with
  mean cosine -0.827;
- full v4.6 changed the classifier gradient on 47,414 / 50,000 steps;
- raw tail pseudo-label precision/recall/coverage were 98.08/32.59/47.92%.

The first two facts show that the conflict predicate detects ordinary
cross-entropy target/non-target competition rather than a selective harmful
event. The third rules out widespread tail pseudo-label corruption as the
primary bottleneck. This closes v4.6 permanently; no beta/tau/rho repair is
authorized.

The unchanged FixMatch observer checkpoint was then reused for a v4.7
development-only score-calibration study. The official test set was not read.

| Scoring | bACC | Head | Medium | Tail | GM | Dead |
|---|---:|---:|---:|---:|---:|---:|
| raw | 36.70 | 69.39 | 34.12 | 6.67 | 5.65 | 22 |
| uniform LA, alpha=0.85 | 39.42 | 67.58 | 39.18 | 11.52 | 16.11 | 7 |
| TANGS v4.7, 0.65/0.25/0.75 | **40.04** | 67.45 | 36.94 | **15.82** | **20.84** | **4** |

The retrospective artifact
`gradvax_experiments/results/second_try_2026-08-15/v47-development-selection.json`
is **PROVISIONAL**, not PASS. It authorizes only one fresh target-server
P-C100-100 integrated v4.7 development run of 50,000 steps after the v4.7 CUDA
smoke; it explicitly forbids every 250k launch. Only the new integrated 50k
artifact may issue a real development PASS/STOP. If it passes, the single-GPU
confirmatory schedule is C100 250k first as a stop-loss gate, followed by the
required C10-100 and STL10-20 250k runs after C100 PASS. This candidate signal
does not reverse or soften the v4.6 STOP.
