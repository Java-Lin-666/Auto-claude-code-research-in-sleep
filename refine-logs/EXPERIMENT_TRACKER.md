# TANGS Experiment Tracker

*Specification version: 4.8 — selective score-correction ledger*
*Last updated: 2026-08-15*
*Status: v4.5/v4.6/v4.7 are archived STOP results. v4.8 is PROVISIONAL after passing all original gates on two archived 50k realizations. Local tests are 61/61 and local CUDA smoke is DONE. Only the fresh seed-1 C100 50k holdout is pending; all 250k work is BLOCKED.*
*Aligned files: FINAL_PROPOSAL.md v4.8, EXPERIMENT_PLAN.md v4.8, reported_results_from_papers.md v4.8 policy*
*Legacy working name: GradVax. TANGS v4.8 means selective Tail-ANchor-Gated Scores.*

---

## 0B. v4.8 Dashboard (Controlling)

### 0B.1 Failure-driven change

| Item | Observation | Decision |
|---|---:|---|
| Genuine v4.7 50k | bACC 39.80; Head -2.30 vs raw; bACC +0.38 vs LA | STOP v4.7 |
| v4.7 broad anchor gate | about 67.56% eligible on fresh checkpoint | add uniqueness and confidence filters |
| v4.8 eligibility / flips | 12.36% / 5.06% fresh; 11.70% / 4.86% archive | selective intervention restored |
| v4.8 robust grid | 110 fresh passes; 21 pass both realizations | freeze one rule; require unseen holdout |

### 0B.2 Frozen retrospective ledger

| Realization | Raw bACC/Head | LA-0.80 bACC/Head/Tail | v4.8 bACC/Head/Medium/Tail/GM | Dead | Status |
|---|---:|---:|---:|---:|---|
| Fresh integrated seed-0 | 36.14/68.61 | 39.22/66.85/12.30 | **39.86**/66.79/37.18/**15.70**/20.49 | 3 | SIGNAL |
| Archived observer seed-0 | 36.70/69.39 | 39.20/67.70/11.09 | **39.84**/67.82/37.18/**14.61**/20.40 | 4 | SIGNAL |

Frozen parameters: uniform LA 0.80, base 0.70, extra 0.40, anchor threshold
0.775, best-vs-second anchor margin 0.05, raw-confidence ceiling 0.90.
Both rows pass the unchanged raw and LA efficacy thresholds. They remain
retrospective and only authorize a new seed-1 holdout.

Artifacts:

- robust grid: `results/third_try_2026-08-15/v48-logit-gap-robust-search.json`;
- production-scorer freeze: `results/third_try_2026-08-15/v48-retrospective-selection.json`;
- implementation: `tangs/calibration.py`, method `tangs-v48`;
- local smoke: `results/v48-local-smoke-c100-seed1`, config `a5e4e6bffb726936cb095950fd961b0d08a83dee766197399c273c993fea4405`.

### 0B.3 Integrity and single-GPU queue

| ID | Item | Status | Evidence / condition |
|---|---|---|---|
| V48-CODE | isolated method/config/scorer/gates | DONE | v4.5-v4.7 identifiers retained |
| V48-LOCAL-TEST | full suite | DONE | 61/61 |
| V48-LOCAL-SMOKE | five-step CUDA, seed 1 | DONE | zero gradient modifications; all summaries finite |
| V48-SERVER-SMOKE | target tests and five-step CUDA | TODO | must precede paid holdout |
| V48-HOLDOUT | `v48-holdout-c100-100-seed1`, 50k | BLOCKED-UNTIL-SMOKE | only paid run currently authorized |
| V48-C100-250K | seed-0 confirmatory | BLOCKED | only after holdout PASS |
| V48-C10-250K | seed-0 transfer | BLOCKED | required after C100 PASS |
| V48-STL20-250K | seed-0 transfer | BLOCKED | required after C100 PASS |
| V48-INDEPENDENT-AUDIT | different model family | TODO | required before paper claim |

The holdout analyzer rejects seed 0, reused split hashes, incomplete
diagnostics, any gradient modification, parameter drift, and any original
efficacy failure. A holdout STOP closes v4.8; a PASS only unlocks the serial
C100 → C10 → STL10-20 confirmatory script.

---

## 0A. v4.7 Dashboard (Historical)

**Historical:** the fresh integrated v4.7 50k gate is STOP. This dashboard no
longer authorizes execution.

Every v4.5/v4.6 status below remains historical. This dashboard alone
authorizes new execution.

### 0A.1 Root-cause disposition

| Finding | Observation | Decision |
|---|---:|---|
| v4.6 full efficacy | 35.88 bACC vs 36.70 paired FixMatch; Tail 6.48 vs 6.67 | STOP gradient surgery |
| observer tail-row conflict | 1,589,275 / 1,589,275 eligible rows; mean cosine -0.827 | conflict trigger is structurally degenerate |
| v4.6 intervention burden | 47,414 / 50,000 steps modified | not an under-activation problem |
| raw tail pseudo labels | 98.08 precision / 32.59 recall / 47.92 coverage | exploit reliable tail evidence; do not suppress normal head negatives |
| dominant recoverable error | prior correction raises bACC by several points without retraining | pivot to score calibration |

### 0A.2 Retrospective candidate-scoring ledger

All three rows reuse the exact same **v4.6** FixMatch observer checkpoint and
split. These checks establish a positive candidate signal only; they do not
constitute a fresh integrated v4.7 training PASS.

| ID | Scoring rule | Parameters | bACC | H/M/T | GM | Dead | Status |
|---|---|---|---:|---:|---:|---:|---|
| V47-RAW | raw EMA logits | none | 36.70 | 69.39/34.12/6.67 | 5.65 | 22 | REFERENCE |
| V47-LA | uniform logit adjustment | alpha=0.85 | 39.42 | 67.58/39.18/11.52 | 16.11 | 7 | FROZEN-CONTROL |
| V47-TANGS | Tail-ANchor-Gated Scores candidate | base=0.65, extra=0.25, cosine=0.75 | **40.04** | 67.45/36.94/**15.82** | **20.84** | **4** | **PROVISIONAL** |

| Gate | Pass rule | Observed | Status |
|---|---|---:|---|
| V47-R1 | candidate vs raw bACC >= +1.0 pp | +3.34 | SIGNAL |
| V47-R2 | candidate vs raw Tail >= +2.0 pp | +9.15 | SIGNAL |
| V47-R3 | candidate vs raw Head >= -2.0 pp; GM >= 0; dead non-increasing | -1.94 / +15.19 / -18 | SIGNAL |
| V47-R4 | candidate vs head-constrained LA bACC >= +0.5 pp | +0.62 | SIGNAL |
| V47-R5 | candidate vs head-constrained LA Tail >= +2.0 pp | +4.30 | SIGNAL |

Required disclosure: v4.7 is -2.24 pp on Medium versus the frozen LA control.
The unconstrained LA sweep reached 41.30 bACC at alpha=1.55 but Head fell to
63.27, outside the two-point preservation guard. The v4.7 claim is therefore
head-constrained Pareto improvement, not unconditional LA dominance.

Artifacts:

- retrospective screen: `gradvax_experiments/results/second_try_2026-08-15/v47-development-selection.json` (`PROVISIONAL`, 250k forbidden);
- exploratory sweeps: observer run's `exploratory_prior_shift.json` and
  `exploratory_anchor_gated_prior.json`;
- implementation: `gradvax_experiments/tangs/calibration.py`;
- local smoke: `gradvax_experiments/results/v47-local-smoke-v3-c100-seed0`.

### 0A.3 Code and integrity

| ID | Check | Status | Evidence / next action |
|---|---|---|---|
| V47-CODE | frozen scorer, config, trainer integration, raw/adjusted reporting | DONE | `calibration.py`, `metrics.py`, `trainer.py`, config version 4.7 |
| V47-LOCAL-TEST | complete suite | DONE | 58/58 PASS in 4.56 s; RTX 4060 environment |
| V47-LOCAL-SMOKE | five-step CUDA integration | DONE | config `d0bfeac1...111e75`; zero modifications; raw/LA/adjusted summaries finite |
| V47-SERVER-TEST | repeat suite on target | DONE | 58/58 PASS in 4.51 s on target RTX 4090 |
| V47-SERVER-SMOKE | five-step CUDA smoke on deployed v4.7 | DONE | `v47-server-smoke-c100-seed0`, config `5d242224...f9f3e9`; finite and observer-only |
| V47-INTEGRATED-50K | fresh C100 v4.7 development run and machine gate | STOP | config `c057da1f...6824b11a`; normal 50k completion, all integrity checks PASS, two efficacy checks fail |
| V47-INDEPENDENT-AUDIT | different model family reads code and raw artifacts | TODO | required before paper claim |

### 0A.4 Single-GPU run ledger

| Order | Run ID | Protocol | Steps | Status | Authorization |
|---:|---|---|---:|---|---|
| 1 | `v47-dev-c100-100-integrated-seed0` | P-C100-100 development | 50,000 | STOP | complete third-try artifact; bACC-vs-LA and head-vs-raw guards fail |
| 2 | `v47-confirm-c100-100-seed0` | P-C100-100 | 250,000 | BLOCKED | only after fresh integrated 50k PASS |
| 3 | `v47-confirm-c10-100-seed0` | P-C10-100 | 250,000 | BLOCKED | required after C100 confirmatory PASS |
| 4 | `v47-confirm-stl10-20-seed0` | P-STL10-20 | 250,000 | BLOCKED | required after C100 confirmatory PASS |

No separate FixMatch or LA training run is permitted. The fresh 50k gate uses
the same numerical thresholds as retrospective signals R1-R5 and additionally
checks integrated code, split role, 50k completion, frozen parameters, and
zero gradient writes. The C100 final gate repeats those efficacy checks on the
official test result and adds 250k/hash/config checks. Parameters are
immutable. A failed 50k gate stops rows 2-4; a failed C100 confirmatory gate
stops rows 3-4. C100 is only the stop-loss-first dataset. The required paper
matrix still includes C10 and STL10-20; P-STL10-10 is optional afterward.

### 0A.5 Third try — fresh server 50k result (2026-08-15)

The only fresh integrated v4.7 development run completed on the target server
without runtime failure. It used the 5,000-example balanced, disjoint
development split; the official test set was not read. The machine gate is
authoritative and is **STOP**.

| Scoring row | bACC | Head / Medium / Tail | GM | Dead |
|---|---:|---:|---:|---:|
| Raw FixMatch | 36.12 | 68.61 / 33.24 / 6.61 | 5.72 | 21 |
| Uniform LA (`alpha=0.85`) | 39.42 | 66.55 / 38.94 / 12.79 | 16.00 | 7 |
| TANGS v4.7 | 39.80 | 66.30 / 36.76 / 16.42 | 20.34 | 4 |

TANGS v4.7 passes raw bACC (+3.68 pp), raw tail (+9.82 pp), GM, and
dead-class guards, and it passes tail versus uniform LA (+3.64 pp). It fails
the bACC-versus-LA guard (+0.38 pp, required >= +0.50 pp) and the
head-versus-raw guard (-2.30 pp, permitted >= -2.00 pp). All integrity fields
in `v47-integrated-development-gate.json` are true, including frozen
constants, zero gradient modifications, 50,000-step diagnostics, and split
disjointness. Therefore `allow_single_c100_confirmatory_run=false` and
`allow_confirmatory_matrix=false`; no 250k run is authorized.

Primary raw artifact: `gradvax_experiments/results/third_try_2026-08-15/v47-dev-c100-100-integrated-seed0/`.
Gate: `gradvax_experiments/results/third_try_2026-08-15/v47-integrated-development-gate.json`.
Preservation record: `refine-logs/THIRD_TRY_2026-08-15.md`.
An independent reviewer of a different model family remains required before
any paper-facing conclusion.

---

## 0. v4.6 Recovery Dashboard (Historical)

Conflicting v4.5 rows later in this tracker are historical and cannot
authorize execution. `tangs` remains the archived method identity; current
experiments use explicit v4.6 names.

### 0.1 First-attempt disposition

| Item | Observed | Disposition | Artifact |
|---|---:|---|---|
| v4.5 C10 full | bACC 79.59; GM 78.66; Tail 69.93 | ARCHIVED positive single-seed result; not evidence for v4.6 | `refine-logs/FIRST_TRY_2026-08-14.md` |
| v4.5 C100 full | bACC 36.53; GM 7.55; Tail 5.61; Worst 0; 15 dead classes | STOP for v4.5 efficacy | archived `c1-c100-100-tangs-seed0/summary.json` |
| v4.5 C100 geometry | eligible 99.23%; conflict/eligible 93.99%; anchor update 25.42%; invalid updates 2 | rules out an availability-only repair | archived `gradient_diagnostics.json` |
| v4.5 development freeze | `tau=inf`; +1.77 bACC on DEV-C10 | exposes identity with PCGrad/no-cap | archived `development-selection.json` |
| interrupted STL v4.5 | step 131,981 | DO NOT RESUME for current paper evidence | archived checkpoint only |

### 0.2 v4.6 code and integrity

| ID | Check | Status | Evidence / next action |
|---|---|---|---|
| V46-CODE | classwise tail-row controller, analytical rows, correction budget, diagnostics | DONE | `gradvax_experiments/tangs/surgery.py`, `trainer.py`, config v4.6 |
| V46-DATA | balanced unused-train C100 development split | DONE-SERVER | `tangs/data.py`; persistent CIFAR cache read successfully on target server |
| V46-QUEUE | single-GPU resumable development and gated confirmatory scripts | DONE | `scripts/run_v46_development.sh`, `run_required_v46.sh` |
| V46-LOCAL-TEST | full pytest suite in registered local FixMatch environment | DONE | 53/53 passed in 5.24 s; Python 3.10.19, torch 2.5.1, torchvision 0.20.1, CUDA 11.8, RTX 4060 |
| V46-TORCH-PARITY | analytical rows equal autograd, row-only mutation, correction cap, observer no-op | DONE-LOCAL | `test_tailrow_v46.py`; repeat on target server before execution |
| V46-CUDA-SMOKE | observer/full five-step CUDA, resume, finite logs and memory | DONE | target artifacts and config hashes recorded in §0.2.1; 53 server tests PASS |
| V46-INDEPENDENT-AUDIT | different model family audits code and raw artifacts | TODO | five completed development artifacts and the STOP gate are ready for independent review |

### 0.2.1 Target server audit — DONE (2026-08-14)

Target: `jupyter-4fdh8ohtracfw7gp`, one idle NVIDIA GeForce RTX 4090 (24,564 MiB), driver 590.48.01. Persistent volume `/root/rivermind-data/tangs` contains the C10/C100/STL caches, archive, old clean v4.5 repo, and venv. The v4.6 work was isolated in `/root/rivermind-data/tangs/repo-v46-c5f7418` at deployment commit `c5f74189d81a00402cd183a49cc4dc74d3ddf5b0`; no old artifact was deleted and no v4.5 STL checkpoint was resumed. Environment: Python 3.11.10, torch 2.3.1+cu121, torchvision 0.18.1+cu121, CUDA 12.1. `python -m pytest -q`: **53 passed in 4.01 s**.

| Smoke item | Status | Artifact | Config hash / verified result |
|---|---|---|---|
| Observer (FixMatch, 5 steps) | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-fixmatch-observer-seed0` | `7f8c3ce7443e0162d1b5299ef454f35b10d7e80c6b9ae91c233f04b18ce14ebd`; completed, all finite; `observer-only` ×5; no nonzero gradient modification |
| Full (`tangs-v46`, rho=1, 5 steps) | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-seed0` | `98408fa20e4f3a0aa7b34ae2a30b29b768283429bcbc836d396eabf0bf2a754c`; completed and finite; warm-up-only smoke |
| Controlled interrupt + `--resume auto` | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-v2-seed0` | `cc52d447219a3802b0518afceb2d32f0ec91a7907c5305718e2dbc02d99d91c2`; step-2 checkpoint, controller/RNG/model/optimizer/EMA saved, resumed to completed step 5; finite, strictly increasing JSONL |
| Interrupt sent after completion | RETAINED-INVALID | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-seed0` | `89dc6007160bf217dd4ec036cc23c9b19aaf99cd0503ab45fbb0e4eaf28f7040`; keep for audit, not accepted as resume validation |

The valid resume event records the expected limitation that DataLoader worker/prefetch state is not serializable; this is not a false bit-exact claim. The new smoke metrics are infrastructure-only and have not been entered as development or paper results. After the original host killed the first development process, the queue resumed from its retained checkpoint on replacement host `jupyter-3fclyh0mynm24s7b` (RTX 4090, driver 580.119.02) at the same worktree and deployment commit. The replacement-host suite passed **53 tests in 4.17 s**. The five 50k cells subsequently completed serially; the machine-readable gate below is STOP and continues to control all 250k work.

### 0.3 Single-GPU development ledger

All rows use P-C100-100, seed 0, 50,000 steps, and the same 5,000-image
balanced/disjoint training-source validation set. They run sequentially.

| Order | Run ID | Method | Main role | Status | Config hash | Result |
|---:|---|---|---|---|---|---|
| 1 | `v46-dev-c100-100-fixmatch-observer-seed0` | FixMatch + observer | exact paired baseline | DONE | `7a15365bf8b73ffa88f6164652f521a3efa28cb32b02d4bd8469e95bb2c23948` | bACC 36.70; H/M/T 69.39/34.12/6.67; GM 5.65; worst 0; dead 22 |
| 2 | `v46-dev-c100-100-legacy-pcgrad-seed0` | archived `tangs`, tau=inf | legacy comparison | DONE | `b8963e78cd3e5eb384fa5e3e86f31cbf38b97585796cc58094a64ce894a03bd7` | bACC 35.96; H/M/T 67.94/32.00/8.06; GM 8.61; worst 0; dead 14 |
| 3 | `v46-dev-c100-100-tailrow-group-seed0` | tailrow-group | isolate row restriction | DONE | `ca52cbc06f03578d6c226ecfe30b10c56bd0cb9eca2270e510004128554a94f3` | bACC 36.16; H/M/T 68.42/33.88/6.24; GM 6.13; worst 0; dead 20 |
| 4 | `v46-dev-c100-100-tailrow-classwise-seed0` | tailrow-classwise | isolate per-class anchors | DONE | `86b6d5c766d999766bcb853de4a5a394742b266f1ceeeb034d8fd134f781dc40` | bACC 35.40; H/M/T 66.85/31.94/7.52; GM 8.24; worst 0; dead 15 |
| 5 | `v46-dev-c100-100-tangs-rho1-seed0` | tangs-v46, rho=1 | full method | DONE | `99a83cc295ad35f1de4e9fcc96a14d527282c90f664e4e84e9be79061bc57d3f` | bACC 35.88; H/M/T 67.88/33.35/6.48; GM 6.81; worst 0; dead 18 |

Machine-readable gate: `results/v46-development-selection.json`.

| Gate | Pass rule | Status |
|---|---|---|
| V46-G1 | full vs FixMatch: bACC >= +1.0 pp; Tail >= +2.0 pp; Head >= -2.0 pp; GM >= 0.0 pp | STOP — bACC −0.82 pp and Tail −0.18 pp; Head −1.52 pp and GM +1.15 pp meet guards |
| V46-G2 | dead-class count no worse than paired FixMatch | PASS — 18 vs 22 |
| V46-G3 | full bACC >= legacy `tau=inf` +0.5 pp | STOP — −0.08 pp vs legacy |
| V46-G4 | rho=1 bACC >= unbounded classwise +0.25 pp | PASS — +0.48 pp |
| V46-G5 | all split/partition hashes match; validation disjoint; 5,000 examples | PASS — shared split/partition hashes recorded in selection artifact |

Gate artifact: `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-development-selection.json` (`status: STOP`, `allow_confirmatory_250k: false`). All five raw run directories contain `summary.json`, `development_metrics.jsonl`, and `checkpoint_last.pt`; the shared split hash is `38fc04c6d1d0dd9109879bba0445b40dd264f85b7ef198c640a68c45dac57ca9` and partition hash is `01b501e58a2ee7b38669a235098585d0e55ee3f84d952129aeff0a2a7e344bd0`.

The complete local second-try snapshot is `gradvax_experiments/results/second_try_2026-08-15/`; its preservation and SHA-256 verification record is `refine-logs/SECOND_TRY_2026-08-15.md`.

Only all-PASS sets `allow_confirmatory_250k=true`. This gate is STOP, so no
250k run, cross-dataset run, oracle, or extra control is authorized. The
completed evidence must next receive an independent-reviewer audit; it does
not establish a v4.6 efficacy claim.

---

## 1. Status Vocabulary

| Status | Meaning |
|---|---|
| BLOCKED | prerequisite not complete; do not run |
| TODO | ready to schedule |
| QUEUED | resources assigned |
| RUNNING | process active |
| DONE | artifacts exist and checks pass |
| FAILED | run ended without valid result; reason required |
| STOP | go/no-go gate failed; downstream claims blocked |
| OPTIONAL | not required for the minimum paper |
| CITED | no local run; audited primary-paper value used as a citation-marked benchmark row |

No row may be marked DONE without a raw artifact path and config hash.

---

## 2. Document and Venue Lock

| Item | Required value | Status | Evidence |
|---|---|---|---|
| Proposal version | 4.7 | DONE | refine-logs/FINAL_PROPOSAL.md |
| Plan version | 4.7 | DONE | refine-logs/EXPERIMENT_PLAN.md |
| Tracker version | 4.7 | DONE | this file |
| Reported-results recovery policy | 4.7 | DONE | gradvax_experiments/reported_results_from_papers.md |
| Paper-facing method name | TANGS | DONE | all v4.5 files |
| Upstream implementation substrate | clean official `LeeHyuck/CDMAD` snapshot | DONE | commit `7cd732b4615b9d94934a9197e69c6775496fb5ee` |
| Exact venue and year | TBD | BLOCKED | venue URL required |
| Page/anonymity policy | TBD | BLOCKED | official venue URL required |
| Submission deadline/timezone | TBD | BLOCKED | official venue URL required |

---

## 3. Protocol Lock

### 3.1 Data Protocols

| Protocol ID | Dataset | \(N_1\) | \(M_1\) / upstream argument | Imbalance | `manualSeed` | v4.5 role | Status | Split artifact |
|---|---|---:|---:|---|---|---|---|---|
| DEV-C10-100 | CIFAR-10-LT | 1500 | 3000 | 100/100; 20% labeled validation | 0 | short development only | TODO | P0 gate split `f98af602...f4c20`; full manifests regenerated per run |
| DEV-C100-100-V46 | CIFAR-100-LT | 150 | 300 | 100/100; 50 unused train images/class for validation | 0 | v4.6 recovery development | TODO | generated and hash-checked per run |
| P-C10-100 | CIFAR-10-LT | 1500 | 3000 | 100/100 | 0 | required local generalization | BLOCKED | TBD |
| P-C100-100 | CIFAR-100-LT | 150 | 300 | 100/100 | 0 | required local core | BLOCKED | TBD |
| P-STL10-10 | STL-10-LT | 450 | original 100k | 10/N/A | 0 | reported context; optional local | OPTIONAL | TBD |
| P-STL10-20 | STL-10-LT | 450 | original 100k | 20/N/A | 0 | required local generalization | BLOCKED | TBD |

Released-loader semantics:

- CIFAR unlabeled-loader indices use the prefix through \(n_c+m_c\), including the labeled prefix;
- CIFAR-10 and STL-10 use fixed class-order prefixes for every seed;
- CIFAR-100 uses fixed prefixes for seed 0 and within-class shuffling for nonzero seeds.
- STL-10 appends the selected labeled images to the official 100k unlabeled split, so the actual unlabeled-loader cardinality is `100000 + |L|`;

The v4.5 minimum matrix locks `manualSeed=0` for every local run. Any nonzero-seed robustness study requires a v4.6+ protocol and separate tracker rows; on CIFAR-100 it also changes the sampled split.

Local data readiness (2026-08-13): CIFAR-10 archive MD5 `c58f30108f718f92721af3b95e74349a` and torchvision sizes 50,000/10,000 verified; CIFAR-100 torchvision sizes 50,000/10,000 verified; STL-10 archive MD5 `91f7769df0f17e558f3565bffb0c7dfb` and split sizes 5,000/100,000/8,000 verified. This is storage readiness, not completion of the protocol rows above.

For every protocol/seed, save:

- integer per-class counts;
- labeled and unlabeled-loader index arrays;
- overlap count and fraction;
- generation code version;
- SHA256;
- deterministic head/medium/tail partition and SHA256.

### 3.2 Base Configuration

| Item | Locked value | Status | Evidence |
|---|---|---|---|
| Architecture | exact `CDMAD/wrn.py` WRN-28-2; LeakyReLU 0.1; unused rotation head retained; executed BatchNorm defaults `eps=1e-5`, `momentum=0.1`; classifier is `model.output` | DONE | CDMAD/wrn.py |
| Image size | 32×32 | DONE | CDMAD/dataset/fix_*.py |
| Normalization | released CIFAR-10/STL-10 and CIFAR-100 mean/std tuples | DONE | CDMAD/dataset/fix_*.py |
| Precision | FP32; AMP disabled | DONE | CDMAD/fixmatchcdmad.py |
| Optimizer | Adam defaults: betas 0.9/0.999, eps 1e-8, weight_decay 0, amsgrad False | DONE | `optim.Adam(params, lr=args.lr)` |
| Learning rate | constant 1.5e-3; no scheduler | DONE | CDMAD/fixmatchcdmad.py |
| Labeled/unlabeled batch | 32/64 | DONE | CDMAD/fixmatchcdmad.py |
| Unlabeled ratio \(\mu\) | 2 | DONE | CDMAD/fixmatchcdmad.py |
| Train loaders | shuffle True, drop_last True, 4 persistent workers, iterator restart without respawn | DONE | tangs/data.py; perf-v2 P0 |
| Test loader | batch 200, shuffle False, drop_last False, 4 persistent workers | DONE | tangs/data.py; perf-v2 P0 |
| Total steps | 500 epochs × 500 steps = 250,000 | DONE | CDMAD/fixmatchcdmad.py |
| EMA mechanics | EMA-to-online initial copy; state-dictionary update after Adam; floating entries also receive `1-wd*lr` | DONE | CDMAD/fixmatchcdmad.py `WeightEMA` |
| Eval-model EMA | 0.999 | DONE | CDMAD/fixmatchcdmad.py |
| Confidence threshold | 0.95; renamed derived argument | DONE | gradvax_experiments/tangs/config.py; trainer.py |
| Pseudo-label | detached hard weak-view argmax | DONE | gradvax_experiments/tangs/trainer.py |
| CDMAD white-image subtraction | disabled in TANGS and local controls | DONE | gradvax_experiments/tangs/trainer.py; no subtraction branch |
| Loss weights | `loss=Lx+Lu`; \(\lambda_u=1.0\); no auxiliary loss | DONE | CDMAD/fixmatchcdmad.py |
| Weak augmentation | exact released resize/crop/flip/tensor/normalize ordering | DONE | CDMAD/dataset/fix_*.py |
| Strong augmentation | two exact released RandAugment(3,4)+crop/flip/normalize+Cutout(16) views | DONE | CDMAD/dataset/fix_*.py |
| Custom decay coefficients | CIFAR-10/CIFAR-100/STL-10 = 0.04/0.08/0.01 | DONE | CDMAD/fixmatchcdmad.py `WeightEMA.step()` |
| Test transform | exact released CIFAR/STL test transforms | DONE | CDMAD/dataset/fix_*.py |
| Progress/logging | upstream per-step progress; structured metrics every 50 steps for all methods | DONE | gradvax_experiments/tangs/trainer.py |
| Checkpoint cadence | rolling every epoch; snapshots every 100 epochs/50,000 steps plus final | DONE | CDMAD/fixmatchcdmad.py |
| Confirmatory checkpoint | final EMA at 250k | DONE | plan v4.5 |
| Test selection | prohibited; upstream per-epoch test cannot select | DONE | plan v4.5 |
| cuDNN | deterministic=True, benchmark=False | DONE | gradvax_experiments/tangs/trainer.py `seed_everything()` |

### 3.3 TANGS Defaults

Current v4.6 values:

| Item | Locked value | Status |
|---|---|---|
| Anchor | per-tail-class EMA of supervised self-row contribution | DONE-CODE |
| Norm statistic | per-tail-class EMA of supervised self-row norm | DONE-CODE |
| Correction budget rho | 1.0 | FROZEN-DEV-CANDIDATE |
| Beta / warm-up / epsilon | 0.99 / 2500 / 1e-12 | DONE-CODE |
| Scope | only tail classifier weight+bias rows | DONE-CODE |
| Minimum class support | 1 labeled example for that class | DONE-CODE |
| Full update | exact base gradient with only each conflicting tail-row contribution replaced | DONE-CODE |

Archived v4.5 values below are retained for artifact interpretation only:

| Item | Locked value | Status |
|---|---|---|
| \(\beta\) | 0.99 | DONE |
| \(\tau\) | initial candidate 5.0; choose once from {2,5,10,infinity} on DEV | BLOCKED |
| Warm-up | 2500 steps | DONE |
| Minimum tail support | 2 | DONE |
| Epsilon | 1e-12 | DONE |
| Scope | classifier weight+bias jointly | DONE |
| Geometry dtype | FP32; AMP disabled | DONE |
| Anchor initialization | first valid current-tail contribution | DONE |
| Domination trigger | post-projection ratio \(>\tau\) | DONE |
| Final update | \(g_{\mathrm{base}}-g_h+g_h^{\mathrm{mod}}\) | DONE |

---

## 4. Blocking Integrity Suite

Completed local code-readiness checks:

| Validation ID | Check | Status | Artifact | Config/source hash |
|---|---|---|---|---|
| U0-LOCAL-TESTS | 42-test integrity suite including closed-form head-gradient parity, backbone no-op equivalence, graph release, loader cycling, gate states, v4.5 identity, resume rollback, and CUDA-mapped RNG restoration | DONE | gradvax_experiments/validation/U0_LOCAL_VALIDATION.md | `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0-LOCAL-SMOKE | real CUDA five-step P-C100-100 TANGS smoke under v4.5 | DONE | gradvax_experiments/results/u0-server-preflight-v45-final-20260813; durable record above | `b2fa6bd9e9017c8b0845ca5c281d13270e9f1e52338c5fb5885a0efca338fedb` |
| U0-LOCAL-RESUME | controlled CUDA interruption at rolling step 4 followed by successful resume through step 20 | DONE | gradvax_experiments/results/u0-resume-midrun-v45-20260813; durable record above | `972e9483d739a0598e614477fa7c59cb77f36abf2e25f4c0afdfdf8d89e05877` |
| U0-LOCAL-STL-SMOKE | real CUDA five-step P-STL10-20 smoke including 100k-pool append path and full 8k test evaluation | DONE | gradvax_experiments/results/u0-stl10-20-server-preflight-v45-20260813; durable record above | `1e9b9ee0193bfee04b9ffcb422239023b439376911ed600b542ffcdf307f930e` |

These checks show that the implementation imports, trains, checkpoints, and emits the required artifacts on a real GPU. They are not performance evidence and do not automatically satisfy U0 criteria that still lack a dedicated independent test.


| ID | Test | Pass criterion | Status | Artifact |
|---|---|---|---|---|
| U0a | exact head contribution | independent and implementation gradients agree within relative 1e-6 | DONE | test_noop_optimizer.py closed-form linear-gradient check; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0b | no-op classifier equivalence | gradient and optimizer step agree within relative 1e-6 | DONE | test_noop_optimizer.py; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0c | no-op backbone equivalence | gradient agrees within relative 1e-6 | DONE | test_noop_optimizer.py backbone-gradient/full-Adam check; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0d | projection conflict removal | post dot ≥ −1e-6 normalized tolerance | DONE | test_torch_integrity.py; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0e | non-amplifying cap | modified norm ≤ projected norm | DONE | test_torch_integrity.py; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0f | pre/post-ratio edge case | no scale-up when projection drops ratio below \(\tau\) | DONE | test_torch_integrity.py projection-below-cap boundary; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0g | empty-mask bypass | exact base update | DONE | test_torch_integrity.py valid-anchor/empty-head check; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0h | invalid-anchor bypass | exact base update plus warning | DONE | test_torch_integrity.py invalid-anchor no-op; trainer emits one structured post-warm-up warning; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0i | first-anchor initialization | anchor equals first valid \(g_t\) | DONE | test_torch_integrity.py first-anchor equality; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0j | warm-up boundary | base update before boundary; TANGS after | DONE | test_torch_integrity.py boundary-step check; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0k | precision guard | FP32 active and AMP-enabled configurations rejected | DONE | test_protocol.py; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0l | graph lifetime | no cross-step retained graph or memory growth | DONE | test_torch_integrity.py repeated weak-reference graph-release check plus CUDA smoke; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0m | optimizer-state logging | effective update diagnostics emitted | DONE | u0-server-preflight-v45-final-20260813/train_metrics.jsonl; config `b2fa6bd9e9017c8b0845ca5c281d13270e9f1e52338c5fb5885a0efca338fedb` |
| U0n | artifact trace | config→run→metrics→checkpoint chain resolves | DONE | results/u0-server-preflight-v45-final-20260813; config `b2fa6bd9e9017c8b0845ca5c281d13270e9f1e52338c5fb5885a0efca338fedb` |
| U0o | substrate config parity | model/precision/Adam/LR/loaders/transforms/loss/EMA/checkpoint values equal Section 3.2 | DONE | test_protocol.py + test_torch_integrity.py + test_data_parity.py + v4.5 CUDA smoke; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0p | `WeightEMA` one-step parity | online, EMA, floating buffers, and decay match pinned code | DONE | test_torch_integrity.py; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0q | loader parity | indices/cardinalities/overlap/batch shapes/restart behavior match pinned loaders | DONE | test_data_parity.py + P0, CIFAR-100, and STL-10 v4.5 CUDA split manifests; source SHA256 `5eba7ed1e6ec7e9f31047907ef06ff55fa7466e0e202ce94adb6441d8614d2fd` |
| U0r | E1 instrumentation | synchronized timing and memory-footprint fields are finite across 5 CUDA steps | DONE | u0-server-preflight-v45-final-20260813/train_metrics.jsonl |

**G0 status: DONE (2026-08-13).** Every U0 row has direct test/code/runtime evidence. The completed P0 pair remains development-only evidence; with G0 DONE and P0 PASS, the registered 50k development freeze is now authorized. No 250k confirmatory run is authorized before the final development freeze is PASS.

---

## 5. Development/Tuning Ledger

The controlling development ledger is Section 0.3. The DEV-C10 ledger below
documents the completed v4.5 selection that chose `tau=inf`; it is closed and
must not be rerun or used to authorize v4.6 confirmation.

DEV-C10-100 validation indices must be removed from both the labeled and overlapping unlabeled training loaders. Save the three role manifests (train-labeled, train-unlabeled, validation) and verify that validation is disjoint from both training loaders before any tuning run; the two training loaders may retain their released overlap.

### P0 Minimal Idea Gate

| Pilot ID | Method | Protocol | Steps | Seed | Status | Artifact |
|---|---|---|---:|---:|---|---|
| P0-FM-20K | FixMatch | DEV-C10-100 | 20,000 | 0 | DONE | `results/pilot-c10-100-fixmatch-20000step-seed0-perf-v2`; config `6d16f478ef4fa18ecb54832a1c059f113d6afa99479f8705f72cffc38cb85c13` |
| P0-TG-20K | TANGS, `tau=5` | DEV-C10-100 | 20,000 | 0 | DONE | `results/pilot-c10-100-tangs-tau5-20000step-seed0-perf-v2`; config `c3ab3925b4fa7bb28c7999164e1824f425799d937735de01dcae0a6a7e9174de` |

| P0 check | Pass rule | Observed | Status |
|---|---|---|---|
| pairing | same split/partition hashes, development-validation role, seed 0, final step 20k | split `f98af602...f4c20`; partition `48472189...c3aa`; both match | PASS |
| mechanism | eligible ≥100; anchor coverage ≥90%; conflict-or-domination ≥10%; sampled nonzero modification exists | eligible 19,927; coverage 99.94%; either 94.73%; nonzero modifications 16,568 | PASS |
| non-collapse | TANGS validation bACC drop versus paired FixMatch ≤0.5 pp | 69.49 vs 68.20; +1.29 pp | PASS |
| positive signal | bACC gain ≥0.25 pp OR tail gain ≥0.5 pp | bACC +1.29 pp; tail +13.33 pp | PASS |
| **P0 decision** | PASS makes 50k tuning eligible after G0; HOLD permits only a paired 50k extension after G0; STOP blocks further performance runs | `PASS`; Overall -5.91 pp, Head -10.32 pp, Medium +0.96 pp, Tail +13.33 pp | **PASS** |

P0 uses the local FixMatch row only as a development spending safeguard. It never enters the CDMAD-style paper main table. Machine-readable output: `results/minimal-gate-c10-100-20000step.json`.

Runtime audit: the first unversioned FixMatch attempt was stopped at 11k and retained as `INVALID-RUNTIME` after Windows DataLoader workers were observed respawning whenever a cycling loader exhausted. The `perf-v2` pair enabled persistent workers, passed 28 unit/integrity tests, kept worker PIDs stable across loader cycles, and is the only pair used by the gate. No 50k run was launched after PASS.

Worker persistence preserves dataset membership and the loss definition but can alter augmentation RNG consumption relative to respawning workers. It is therefore locked for all subsequent local methods and recorded in `split_manifest.json`; no result may be paired across worker-policy versions.

Global maximum: 12 completed development runs total, each stopped at 50,000 steps. This is at most 2.4 full-run equivalents. Add one row for every attempted configuration, including completed poor configurations; failed executions are recorded with failure reasons and may be replaced only within the same global cap.

| Allocation | Maximum short runs | Selection role |
|---|---:|---|
| paired FixMatch development reference | 1 | quantify bACC/GM/Head/Overall deltas; never a C1 or main-table row |
| TANGS \(\tau\in\{2,5,10,\mathrm{infinity}\}\) | 4 | apply the locked balance gate, then choose by bACC/GM/Head/Overall/larger-\(\tau\) ordering |
| head-downweight coefficient | 3 | choose once by DEV validation bACC, then GM and Head |
| documented tie-break/repair reserve | 4 | use only with written reason |
| **Global cap** | **12** | never reset per method |

| Dev run ID | Method | Config hash | Main changed values | Validation bACC | Status | Artifact |
|---|---|---|---|---:|---|---|
| DEV-FM-REF | FixMatch | TBD | 50k paired development reference only | TBD | TODO | TBD |
| DEV-TG-T2 | TANGS | TBD | \(\tau=2\), 50k steps | TBD | TODO | TBD |
| DEV-TG-T5 | TANGS | TBD | \(\tau=5\), 50k steps | TBD | TODO | TBD |
| DEV-TG-T10 | TANGS | TBD | \(\tau=10\), 50k steps | TBD | TODO | TBD |
| DEV-TG-TINF | TANGS | TBD | \(\tau=\mathrm{infinity}\), 50k steps | TBD | TODO | TBD |
| DEV-HDW-025 | head downweighting | TBD | coefficient 0.25, 50k steps | TBD | BLOCKED | TBD |
| DEV-HDW-050 | head downweighting | TBD | coefficient 0.50, 50k steps | TBD | BLOCKED | TBD |
| DEV-HDW-075 | head downweighting | TBD | coefficient 0.75, 50k steps | TBD | BLOCKED | TBD |

The intermediate `results/development-tangs-screen.json` is written after DEV-FM-REF and the four \(\tau\) rows; failure stops before the three head-downweight rows. The final machine-readable freeze is `results/development-selection.json`. It is PASS only when every registered run above is complete with matching split/partition hashes and at least one TANGS candidate has bACC delta >= +0.25 pp, GM delta >= -0.25 pp, Head delta >= -5 pp, and Overall delta >= -3 pp versus DEV-FM-REF. No 250k command is authorized without this PASS artifact; `scripts/run_required.sh` enforces the frozen `TANGS_TAU` and `HEAD_WEIGHT` values.

Frozen winner config hashes:

| Method | Winning config hash | Frozen date | Status |
|---|---|---|---|
| TANGS | TBD | TBD | BLOCKED |
| head downweighting | TBD | TBD | BLOCKED |
| matched clipping | TBD | TBD | BLOCKED |
| PCGrad-adapted | TBD | TBD | BLOCKED |
| cited methods | published configurations | N/A | CITED |


---

## 6. Phase A — Diagnostics

| ID | Experiment | Protocol | Seeds | Required outputs | Status | Artifact |
|---|---|---|---|---|---|---|
| A1 | occurrence/eligibility | P-C100-100 | 0 pilot, then confirm | eligible/conflict/domination/either fractions; raw/post ratios | BLOCKED | TBD |
| A2 | true-class decomposition | P-C100-100 | same A1 stream | true H/M/T counts, error, cosine, norm | BLOCKED | TBD |
| A3 | anchor quality | P-C100-100 | same A1 stream | age, norm, availability, cancellation, per-class cosine | BLOCKED | TBD |
| A4 | temporal association | DEV-C10-100 or locked checkpoints | optional | detrended/lagged association with autocorrelation-aware CI | OPTIONAL | TBD |

Decision fields:

| Quantity | Threshold | Observed | Status |
|---|---:|---:|---|
| either-trigger / eligible steps | ≥15% | TBD | BLOCKED |
| conflict / eligible steps | ≥5% OR domination gate passes | TBD | BLOCKED |
| domination / eligible steps | ≥10% OR conflict gate passes | TBD | BLOCKED |
| valid anchor / post-warm-up steps | ≥90% | TBD | BLOCKED |
| median within-tail cancellation ratio | ≥0.10 | TBD | BLOCKED |

**G1 occurrence:** BLOCKED

**G2 anchor availability:** BLOCKED

**G3 anchor cancellation:** BLOCKED

If G1, G2, or G3 fails, mark downstream TANGS confirmatory rows STOP and create a new proposal version before redesigning the anchor or trigger.

---

## 7. Phase B — Mechanism Controls

| ID | Method pair | Protocol | Seeds | Isolation rule | Status | Artifact |
|---|---|---|---|---|---|---|
| B1 | FixMatch-oracle-target vs. +TANGS | P-C100-100 | 0 | same ordinary mask/membership; target only replaced | BLOCKED | TBD |
| B2 | full-oracle upper bound | P-C100-100 | optional | all unlabeled GT; upper bound only | OPTIONAL | TBD |
`C1b-TG` supplies the ordinary-TANGS quantity for B1. The two new B1 jobs are the oracle-target FixMatch and oracle-target TANGS pair shown above.

| B3 | locked dynamics | P-C100-100 | TANGS and registered controls | fixed checkpoint grid; no selection | BLOCKED | TBD |

Mechanism calculation:

\[
\mathrm{shrinkage}
=1-
\frac{Q_{\mathrm{oracle\mbox{-}TANGS}}}
{Q_{\mathrm{ordinary\mbox{-}TANGS}}},
\]

where \(Q\) is mean `gradient_delta_norm` over eligible steps.

| Gate | Threshold | Observed | Status |
|---|---:|---:|---|
| G7 oracle mechanism | oracle-target TANGS reduces mean intervention-gradient norm by ≥50% relative to ordinary TANGS | TBD | BLOCKED |

Compute both quantities with the same eligibility and logging definitions. The paired oracle-target bACC result remains supporting evidence for the mechanism analysis.

---

## 8. Phase C — Main Comparisons

### C1. Cross-Dataset Matrix

Each minimum-plan matrix row is one locked `manualSeed=0` result; cited rows retain the uncertainty reported by their source papers.

| Matrix ID | Protocol | Method | Seeds | Planned runs | Status | Aggregate artifact |
|---|---|---|---|---:|---|---|
| C1a-TG | P-C10-100 | FixMatch+TANGS | 0 | 1 | BLOCKED | TBD |
| C1b-TG | P-C100-100 | FixMatch+TANGS | 0 | 1 | BLOCKED | TBD |
| C1d-TG | P-STL10-20 | FixMatch+TANGS | 0 | 1 | BLOCKED | TBD |

The local C1 matrix contains 3 TANGS full runs. Audited published methods supply the remaining main-table rows with zero local runs:

| Reported-baseline ID | Scope | Methods | Local runs | Status | Evidence |
|---|---|---|---:|---|---|
| R1-C10 | P-C10-100 | FixMatch, DARP/cRT, CReST/LA, ABC, CoSSL, CDMAD | 0 | CITED | CDMAD-2024 Table 1; original method sources retained |
| R1-C100 | P-C100-100 | FixMatch, DARP/cRT, CReST/LA, ABC, CoSSL, UDAL, CDMAD | 0 | CITED | CDMAD-2024 Table 4 |
| R1-STL10 | P-STL10-10 | FixMatch, DARP/cRT, ABC, CDMAD | 0 | CITED | CDMAD-2024 Table 2 |
| R1-STL20 | P-STL10-20 | FixMatch, DARP/cRT, ABC, CDMAD | 0 | CITED | CDMAD-2024 Table 2 |
| R1-LCGC | matching settings where available | LCGC and reproduced baselines | 0 | CITED | LCGC-2025 tables |
| R1-SIMPRO | all | SimPro | 0 | BLOCKED | remains quarantined until `[VERIFY]` fields are resolved |

CITED rows and the local TANGS row appear in the same CDMAD-style main table. Citation markers preserve provenance, published uncertainty is retained, and the proposed row is labeled `TANGS (ours)`.

### C2. Direct Debiasing/Gradient/Optimization Baselines

| Matrix ID | Protocol | Method | Seeds | Planned runs | Status | Artifact |
|---|---|---|---|---:|---|---|
| C2-HDW | P-C100-100 | head downweighting | 0 | 1 | BLOCKED | TBD |
| C2-CLIP | P-C100-100 | matched head-only clipping | 0 | 1 | BLOCKED | TBD |
| C2-PCG | P-C100-100 | PCGrad-adapted | 0 | 1 | BLOCKED | TBD |

| Gate | Threshold | Observed | Status |
|---|---:|---:|---|
| G8 specificity | seed-0 TANGS exceeds head downweighting and matched clipping by ≥0.5 pp bACC | TBD | BLOCKED |

### C3. Optional CDMAD Complementarity

| Matrix ID | Protocol | Method | Seeds | Planned runs | Status | Artifact |
|---|---|---|---|---:|---|---|
| C3-CD | P-C100-100 | CDMAD | 0 | 1 | OPTIONAL | TBD |
| C3-CD-TG | P-C100-100 | CDMAD+TANGS | 0 | 1 | OPTIONAL | TBD |

| Gate | Threshold | Observed | Status |
|---|---:|---:|---|
| G11 optional complementarity | seed-0 CDMAD+TANGS exceeds matched CDMAD by ≥0.5 pp bACC without head drop below −1.0 pp | TBD | OPTIONAL |

### C4. Optional Base-SSL Transfer

| Matrix ID | Protocol | Method | Seeds | Planned runs | Status | Artifact |
|---|---|---|---|---:|---|---|
| C4-RMM | P-C100-100 | ReMixMatch | 0 | 1 | OPTIONAL | TBD |
| C4-RMM-TG | P-C100-100 | ReMixMatch+TANGS | 0 | 1 | OPTIONAL | TBD |

Seed-0 C4 provides an optional ReMixMatch backbone-transfer result on the evaluated benchmark.

---

## 9. Phase D — Ablations

| Matrix ID | Protocol | Variant | Seeds | Planned runs | Status | Artifact |
|---|---|---|---|---:|---|---|
| D1-FULL | P-C100-100 | full TANGS | reuse C1b-TG seed 0 | 0 new | BLOCKED | TBD |
| D1-NOCAP | P-C100-100 | no norm cap | reuse C2-PCG seed 0 | 0 new | BLOCKED | same artifact as C2-PCG |
| D1-NOPROJ | P-C100-100 | no projection | 0 | 1 | BLOCKED | TBD |
| D1-INSTANT | P-C100-100 | instantaneous anchor | 0 | 1 | BLOCKED | TBD |
| D2-CBANCHOR | P-C100-100 | class-balanced anchor redesign | TBD | TBD | OPTIONAL | gated by A3 |
| D2-PCANCHOR | P-C100-100 | per-class EMA redesign | TBD | TBD | OPTIONAL | gated by A3 |

`D1-NOCAP` and `C2-PCG` are the same locked v4.5 operator, protocol, and seed. The D1 row is an evidence alias and must never launch a duplicate run.

| Gate | Threshold | Observed | Status |
|---|---:|---:|---|
| G9 components | seed-0 full TANGS exceeds both no-cap and no-projection variants | TBD | BLOCKED |

### D3 Sensitivity

| ID | Scalar | Values | Protocol | Max runs | Status | Artifact |
|---|---|---|---|---:|---|---|
| D3-TAU | \(\tau\) | 2,5,10,infinity | DEV-C10-100, 50k steps | 4 within global cap | BLOCKED | TBD |
| D3-BETA | \(\beta\) | 0.95,0.99,0.999 | separate optional budget | 3 | OPTIONAL | TBD |
| D3-WARM | warm-up | 0,2500,5000 | separate optional budget | 3 | OPTIONAL | TBD |

---

## 10. Phase E — Cost

| ID | Comparison | Protocol | Repeats/seeds | Required outputs | Status | Artifact |
|---|---|---|---|---|---|---|
| E1a | extra-gradient/surgery time fraction | reuse C1b-TG | 5 synchronized windows | section time / total step time; 0 new runs | BLOCKED | TBD |
| E1b | known TANGS tensor-memory fraction | reuse C1b-TG | 1 instrumented run | gradient/anchor tensor bytes / peak allocated memory; 0 new runs | BLOCKED | TBD |
| E1c | eligible/intervention microbenchmark | reuse C1b-TG seed 0 | 5 windows | per-step overhead; 0 new runs | BLOCKED | TBD |

| Gate | Threshold | Observed | Status |
|---|---:|---:|---|
| G10 extra-gradient/surgery time fraction | ≤20% | TBD | BLOCKED |
| G10 known gradient/anchor tensor fraction of peak allocated memory | ≤15% | TBD | BLOCKED |

---

## 11. Core Efficacy and Specificity Gates

Source rows: C1b-TG, cited CDMAD Table 4 FixMatch, and the local C2 mechanism controls.

| Gate | Pass rule | Observed | Status |
|---|---|---|---|
| G4 core efficacy | seed-0 TANGS exceeds CDMAD-reported FixMatch bACC by ≥1.0 pp on P-C100-100 | TBD | BLOCKED |
| G5 cross-dataset consistency | seed-0 TANGS exceeds cited FixMatch bACC on at least 2 of 3 required settings | TBD | BLOCKED |
| G6 tail specificity | seed-0 TANGS tail accuracy exceeds both head downweighting and matched clipping by ≥0.5 pp | TBD | BLOCKED |

Statistics artifact must include:

- cited FixMatch value, uncertainty type, and table locator;
- raw seed-0 TANGS value;
- direct percentage-point difference from cited FixMatch;
- TANGS-versus-control tail differences for G6;
- explicit statement that no local variance, confidence interval, or significance test is available.

---

## 12. Go/No-Go Dashboard

| Gate | Pass rule | Status | Downstream effect if failed |
|---|---|---|---|
| G0 | all U0 tests pass | DONE | every post-P0 run remains subject to its next registered gate |
| P0 | paired 20k mechanism, non-collapse, and positive-signal checks pass | PASS | 50k development freeze is permitted only after remaining G0 prerequisites are DONE |
| G1 | either-trigger fraction ≥15% of eligible steps and at least one of conflict ≥5% or domination ≥10% | BLOCKED | stop method story |
| G2 | valid anchor on ≥90% of post-warm-up steps | BLOCKED | redesign before confirmation |
| G3 | median cancellation ratio ≥0.10 | BLOCKED | redesign before confirmation |
| G4 | seed-0 TANGS exceeds cited core FixMatch bACC by ≥1.0 pp | BLOCKED | no submission as efficacy method |
| G5 | seed-0 TANGS exceeds cited FixMatch on at least 2 of 3 settings | BLOCKED | narrow cross-dataset claim |
| G6 | TANGS tail accuracy exceeds both direct magnitude controls by ≥0.5 pp | BLOCKED | narrow tail-specific claim |
| G7 | oracle-target TANGS reduces mean intervention-gradient norm by ≥50% | BLOCKED | remove pseudo-label-noise mechanism claim |
| G8 | seed-0 TANGS exceeds head downweighting and matched clipping by ≥0.5 pp bACC | BLOCKED | reduce novelty to magnitude control |
| G9 | seed-0 full TANGS exceeds both no-cap and no-projection variants | BLOCKED | remove unsupported component |
| G10 | extra-gradient/surgery time ≤20% and known gradient/anchor tensor fraction of peak allocated memory ≤15% | BLOCKED | remove lightweight claim |
| G11 optional | seed-0 CDMAD+TANGS exceeds matched CDMAD by ≥0.5 pp bACC without head drop below −1.0 pp | OPTIONAL | no complementary/plug-in claim beyond the fixed setting |

These v4.5 thresholds are frozen before the first confirmatory launch. Any later change requires v4.6+ with a timestamped rationale.

---

## 13. Claim-to-Evidence Map

| Intended paper statement | Required rows/gates | Current permission |
|---|---|---|
| interference is measurable | A1,A2,A3 + G1–G3 | NOT ALLOWED |
| TANGS outperforms reported FixMatch and shows tail-specific value beyond magnitude controls | C1,C2 + G4–G6 | NOT ALLOWED |
| pseudo-label error contributes to the effect | B1 + G7 | NOT ALLOWED |
| directional surgery beats simple magnitude control | C2 + G8 | NOT ALLOWED |
| both projection and cap matter | D1 + G9 | NOT ALLOWED |
| TANGS complements CDMAD | optional C3 + G11 | NOT ALLOWED |
| TANGS transfers to ReMixMatch on the fixed seed-0 setting | C4 matched seed-0 pair | NOT ALLOWED |
| TANGS is lightweight | E1 + G10 | NOT ALLOWED |
| results are reproducible | completed run ledger + independent audit | NOT ALLOWED |

---

## 14. Master Run Ledger

Every concrete run generated from a matrix row must be appended here before launch.

| Run ID | Matrix ID | Method | Protocol | Data seed | Train seed | Config hash | Git commit | GPU | Status | Metrics artifact | Checkpoint | Log | GPU hours | Failure reason |
|---|---|---|---|---:|---:|---|---|---|---|---|---|---|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | BLOCKED | TBD | TBD | TBD | TBD | |

Run-ID format:

\[
\text{METHOD-PROTOCOL-Dxx-Txx-RNNN}
\]

where Dxx is the zero-padded data seed, Txx the training seed, and RNNN a monotonic run number.

Rules:

- never overwrite a failed run;
- reruns receive a new run ID;
- aggregate artifacts list exact source run IDs;
- a metrics file without a manifest is invalid;
- a locally measured paper number without a DONE run ID is prohibited;
- a cited paper number without a verified registry row, citation, table/row locator, metric, uncertainty type, and benchmark setting is prohibited.

---

## 15. Compute Budget

Controlling v4.6 budget: one GPU, never parallel. First spend is 5 x 50k = one
full-run equivalent. After and only after PASS, spend 2 x 250k on the paired
C100 baseline/full method. The prior 10-full-run v4.5 matrix below is archived
and no longer approved.

Minimum v4.5 seed-0 matrix count before optional A4/B2/C3/C4/D2/beta-warm-up sensitivity/P-STL10-10 local runs:

| Block | Planned new full runs |
|---|---:|
| C1 local TANGS seed-0 rows | 3; FixMatch-family rows are cited |
| B1 oracle-target seed-0 pair | 2 |
| C2 seed-0 mechanism controls | 3 |
| D1 seed-0 new ablations | 2; no-cap reuses C2-PCG |
| E1 timing/memory | 0; reuse instrumented C1b-TG run |
| **Required full-run subtotal** | **10** |
| P0 spending gate | 2 × 20k steps = 0.16 full-run equivalents |
| Development | at most 12 × 50k-step runs = 2.4 full-run equivalents |

Any E1 replacement caused by invalid instrumentation must record the reason and increases the budget explicitly. After the first valid full pilot:

| Item | Value |
|---|---:|
| pilot GPU hours/run | TBD |
| available parallel GPUs | TBD |
| utilization assumption | TBD |
| estimated required GPU hours | TBD |
| estimated calendar days | TBD |
| maximum approved full runs | 10 required; optional budget separate |
| budget approval | BLOCKED until pilot timing and user approval |

---

## 16. Final Audit

Before paper drafting:

- [ ] every cited table number resolves to a verified registry row, citation, table/row locator, metric, uncertainty type, and benchmark setting;
- [ ] every required gate has a recorded outcome;
- [ ] every local table number resolves to DONE run IDs;
- [ ] local seed-0 values carry no uncertainty suffix and paper-reported SE/SD remain explicitly labeled;
- [ ] failed and negative runs remain in the ledger;
- [ ] the exact venue is locked;
- [ ] an independent reviewer/model family audits code and raw artifacts;
- [ ] the executor does not pre-summarize evidence for the integrity reviewer;
- [ ] claims are reduced wherever a gate failed.
