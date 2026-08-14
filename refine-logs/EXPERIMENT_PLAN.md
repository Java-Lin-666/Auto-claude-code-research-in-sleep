# TANGS Experiment Plan

*Specification version: 4.6 — classwise tail-row recovery plan*
*Aligned with FINAL_PROPOSAL.md v4.6 and EXPERIMENT_TRACKER.md v4.6*
*Status: v4.6 code, the full local PyTorch suite (53/53 PASS in `C:\lintao\envs\fixmatch`), and the target-server CUDA smoke are complete. The five sequential 50k C100 development runs are READY/TODO and unstarted. Every 250k v4.6 run is BLOCKED.*

---

## 0. v4.6 Recovery Execution Plan (Controlling)

This section supersedes conflicting v4.5 definitions and schedules later in
the file. Those sections remain as the preregistration and audit history of the
first attempt.

### 0.1 Why v4.5 is stopped

The completed v4.5 development freeze selected `tangs_tau=inf`; finite caps
lost 9.68–12.12 bACC points versus paired local FixMatch. Consequently the
executed method was identical to the registered PCGrad/no-cap ablation. Its
C100 result (36.53 bACC, 5.61 tail, 7.55 GM, 15 dead classes) fails the efficacy
story. High eligibility (99.23%), frequent conflict (93.99% of eligible
steps), and almost no invalid anchors rule out a simple “more updates” repair.

### 0.2 Locked v4.6 operator

Use the method in `FINAL_PROPOSAL.md`'s controlling v4.6 section:

- exact analytical contributions for the final linear classifier rows,
  retaining the executed `-log(softmax + 1e-8)` derivative and original loss
  denominators;
- one EMA self-row anchor and one EMA supervised-row norm per tail class;
- conflict projection per tail row;
- correction budget
  \(\lambda_c=\min(1,\rho m_c/(\|q_c\|+\epsilon))\), with registered
  \(\rho=1\);
- replace only tail classifier rows; head/medium rows and backbone gradients
  remain the base FixMatch update;
- \(\beta=0.99\), warm-up 2500, epsilon \(10^{-12}\), FP32, seed 0;
- a class anchor updates when at least one labeled example of that class is in
  the batch. The old pooled-anchor minimum support of two does not govern the
  classwise update.

Method identities are immutable: `tangs` is archived v4.5,
`tailrow-observer` is measurement-only, `tailrow-group` isolates row
restriction, `tailrow-classwise` removes the correction budget, and
`tangs-v46` is the registered full method.

### 0.3 Development data and single-GPU queue

P-C100-100 is now permitted in development mode. Evaluation uses exactly 50
unused training images per class selected outside the source unlabeled-loader
prefix, for 5,000 balanced examples. This validation set must be disjoint from
both active training loaders; the official 10,000-image test split remains
unread until after freezing.

Run, in this exact sequential order on the user's one GPU:

| Order | Run ID | Method | Changed value | Steps | Status |
|---:|---|---|---|---:|---|
| 1 | `v46-dev-c100-100-fixmatch-observer-seed0` | FixMatch | measurement-only tail-row observer | 50,000 | TODO |
| 2 | `v46-dev-c100-100-legacy-pcgrad-seed0` | archived `tangs` | `tau=inf` | 50,000 | TODO |
| 3 | `v46-dev-c100-100-tailrow-group-seed0` | tailrow-group | group anchor on tail rows | 50,000 | TODO |
| 4 | `v46-dev-c100-100-tailrow-classwise-seed0` | tailrow-classwise | per-class, unbounded | 50,000 | TODO |
| 5 | `v46-dev-c100-100-tangs-rho1-seed0` | tangs-v46 | per-class, `rho=1` | 50,000 | TODO |

Operational command: `bash scripts/run_v46_development.sh`. It skips completed
cells, resumes `checkpoint_last.pt`, and never launches concurrent jobs. At the
end it writes `results/v46-development-selection.json` using
`scripts/analyze_v46_development.py`.

Before the queue, the target execution environment must pass the complete
Torch suite and a five-step CUDA smoke for `tailrow-observer` and `tangs-v46`,
including checkpoint/resume and finite diagnostics. Local testing on
2026-08-14 used `C:\lintao\envs\fixmatch\python.exe` with PyTorch 2.5.1,
torchvision 0.20.1, CUDA 11.8, and the RTX 4060; all 53 tests passed, including
analytical-row/autograd parity, no-op identity, tail-row-only replacement, and
validation-disjointness. The remote server must still repeat these checks
before paid execution because its package and GPU environment may differ.

### 0.3.1 Target-server audit and smoke — DONE (2026-08-14)

The RTX 4090 target passed `python -m pytest -q` (53 passed in 4.01 s) with Python 3.11.10, torch 2.3.1+cu121, torchvision 0.18.1+cu121, and CUDA 12.1. The detached deployment source is `c5f74189d81a00402cd183a49cc4dc74d3ddf5b0` in `/root/rivermind-data/tangs/repo-v46-c5f7418`; its cache symlink points only to the verified persistent dataset volume.

| Check | Status | Artifact | Config hash / finding |
|---|---|---|---|
| FixMatch observer, 5 steps | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-fixmatch-observer-seed0` | `7f8c3ce7443e0162d1b5299ef454f35b10d7e80c6b9ae91c233f04b18ce14ebd`; finite, 5 `observer-only` records, zero nonzero modifications |
| Full `tangs-v46`, rho=1, 5 steps | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-seed0` | `98408fa20e4f3a0aa7b34ae2a30b29b768283429bcbc836d396eabf0bf2a754c`; finite; all steps are warm-up |
| Controlled checkpoint/resume | DONE | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-v2-seed0` | `cc52d447219a3802b0518afceb2d32f0ec91a7907c5305718e2dbc02d99d91c2`; interrupted at checkpoint step 2, resumed to step 5; controller/RNG/model/optimizer/EMA stored; finite strictly increasing JSONL |
| Timing-missed resume attempt | RETAINED-INVALID | `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-seed0` | `89dc6007160bf217dd4ec036cc23c9b19aaf99cd0503ab45fbb0e4eaf28f7040`; SIGTERM arrived after completion, so not resume evidence |

The smoke rows prove target-environment readiness only; no smoke metric may be used for method selection. They unlock scheduling eligibility, not automatic launch: the fixed 50k queue remains unstarted pending explicit user authority.

### 0.4 Frozen development gate

For `tangs-v46` versus paired FixMatch:

- bACC delta >= +1.0 pp;
- tail delta >= +2.0 pp;
- head delta >= -2.0 pp;
- GM delta >= 0.0 pp;
- dead-class count no greater than FixMatch.

Specificity and component checks:

- bACC delta versus archived v4.5 >= +0.5 pp;
- bACC delta versus unbounded classwise >= +0.25 pp.

All checks must pass. A failure produces STOP and forbids 250k. In particular,
if the correction budget fails to beat unbounded classwise projection, remove
the norm-aware claim or create a new method version; do not promote the
unbounded result under the v4.6 name.

### 0.5 Confirmatory spending after PASS

The first authorized full stage contains only two sequential C100 runs:

1. exact local FixMatch plus observer;
2. frozen `tangs-v46` with `rho=1`.

Use `bash scripts/run_required_v46.sh`; the script refuses to start unless the
v4.6 gate says PASS and `allow_confirmatory_250k=true`. This paired local
baseline is mandatory for causal comparison. Literature FixMatch remains
context, not a substitute for this pair. Cross-dataset, oracle, and further
ablation runs remain blocked until the core full pair is reviewed.

The old `scripts/run_required.sh` is archived and blocked by default so it
cannot resume the interrupted STL v4.5 checkpoint accidentally.

---

## 1. Execution Principles

1. Preserve the exact base objective. TANGS replaces only the accepted predicted-head unsupervised classifier-gradient contribution.
2. Separate development, confirmatory evaluation, and reported literature results.
3. Pair methods by data split and training seed.
4. Use final EMA checkpoints for confirmatory claims; never select the best test checkpoint.
5. Treat every result as preliminary until its tracker row is DONE and linked to raw artifacts.
6. Use a different reviewer/model family for the final experiment-integrity audit.

The `CDMAD/` directory is a clean snapshot of the official `LeeHyuck/CDMAD` commit `7cd732b4615b9d94934a9197e69c6775496fb5ee`. Derived TANGS code may reuse its data/model/training infrastructure but must not silently edit that snapshot or inherit CDMAD's white-image subtraction, soft pseudo-labels, or zero confidence threshold.

---

## 2. Locked Method Definition

The implementation must match FINAL_PROPOSAL.md Section 3.

### 2.1 Class Partition

- sort labeled frequencies descending;
- break ties by class ID ascending;
- \(q=\lfloor C/3\rfloor\);
- head = first \(q\), tail = last \(q\), medium = the remainder;
- expected sizes: 3/4/3 for ten-class datasets and 33/34/33 for CIFAR-100;
- save class lists and SHA256 before training.

### 2.2 Exact Contributions

For classifier parameters \(\phi\):

\[
g_{\mathrm{base}}=\nabla_\phi L_{\mathrm{base}},
\]

\[
g_t=\nabla_\phi
\left[
\frac{1}{B_l}\sum_i
\mathbf{1}[y_i\in T]\ell_s(i)
\right],
\]

\[
g_h=\nabla_\phi
\left[
\frac{\lambda_u}{2B_u}\sum_{v=1}^{2}\sum_j
m_j\mathbf{1}[\hat y_j\in H]\ell_u^{(v)}(j)
\right].
\]

The two independent strong views, detached hard pseudo-label, and 0.95 mask are shared by TANGS and every local mechanism control. White-image logit subtraction is disabled. Do not re-normalize either contribution by the number of selected subgroup samples.

The released script's `--tau` name refers to pseudo-label confidence; the derived implementation must rename it and reserve \(\tau\) for the TANGS norm cap.

The replacement is:

\[
g_{\mathrm{new}}
=g_{\mathrm{base}}-g_h+\alpha\,\mathrm{Project}(g_h,a_t),
\]

with projection only for negative inner product and

\[
\alpha=\min\left(1,
\frac{\tau\|a_t\|}
{\|\mathrm{Project}(g_h,a_t)\|+\epsilon}
\right).
\]

The reference is an EMA anchor rather than the instantaneous tail gradient, and the replacement does not constrain the remainder of the base gradient or optimizer state. It is not a descent guarantee.

### 2.3 Defaults

| Item | Value |
|---|---|
| Method name | TANGS |
| Legacy name | GradVax; never use as paper-facing name |
| \(\beta\) | 0.99 |
| \(\tau\) | initial candidate 5.0; select once from {2,5,10,infinity} on DEV-C10-100 against a paired 50k FixMatch development reference |
| Warm-up | 2500 steps |
| Minimum labeled-tail support | 2 |
| Epsilon | \(10^{-12}\) |
| Gradient scope | final classifier weight and bias, flattened jointly |
| Geometry dtype | unscaled FP32 |
| Anchor initialization | first valid \(g_t\) |
| Domination event | post-projection ratio \(>\tau\) |

---

## 3. Protocol Registry

### 3.1 Dataset Construction

For CIFAR-LT, preserve the released `make_imb_data` behavior. Order classes by class ID and construct counts with:

\[
n_c=\left\lfloor N_1\gamma_l^{-c/(C-1)}\right\rfloor,
\]

\[
m_c=\left\lfloor M_1\gamma_u^{-c/(C-1)}\right\rfloor,
\]

with the final class set explicitly to `int(max_num/gamma)`, as in the released code. Here \(M_1\) is the `num_max_u` argument. For CIFAR, the released unlabeled loader uses `idxs[:n_c+m_c]`, so it includes the labeled prefix; save the labeled indices, unlabeled-loader indices, intersection counts, and hashes. The same arrays are reused by TANGS and every local mechanism control.

For STL-10, the released loader appends the selected labeled images to the official 100,000-image unlabeled split. Record the labeled-index hash, append order, overlap count, and actual loader cardinality `100000 + |L|`; the table below calls the 100k value the upstream pool, not the final loader size.

CIFAR-10 and STL-10 use fixed class-order prefixes for all `manualSeed` values in the released loaders. CIFAR-100 uses fixed prefixes for seed 0 and within-class shuffling for nonzero seeds. Preserve and disclose this behavior rather than claiming that every seed creates a new split.

| Protocol ID | Dataset | \(N_1\) | \(M_1\) / upstream argument | Distribution | v4.5 role |
|---|---|---:|---:|---|---|
| P-C10-100 | CIFAR-10-LT | 1500 | 3000 | \(\gamma_l=\gamma_u=100\) | required local generalization |
| P-C100-100 | CIFAR-100-LT | 150 | 300 | \(\gamma_l=\gamma_u=100\) | required local core |
| P-STL10-10 | STL-10-LT | 450 | original 100k pool | \(\gamma_l=10,\gamma_u=\mathrm{N/A}\) | reported context; optional local extension |
| P-STL10-20 | STL-10-LT | 450 | original 100k pool | \(\gamma_l=20,\gamma_u=\mathrm{N/A}\) | required local generalization |

P-C100-100 is the core mechanism protocol.

### 3.2 Base Training Configuration

The local TANGS and mechanism-control runs use a locked derivative of `CDMAD/fixmatchcdmad.py`. Source-derived infrastructure is retained, while the following method-level changes are mandatory for every local run:

- no white-image logit subtraction during training or evaluation;
- hard weak-view argmax pseudo-labels instead of CDMAD soft targets;
- confidence threshold 0.95 instead of the released default 0;
- two strong views and their exact shared reduction;
- no CDMAD `debiasstart` behavior.

These choices preserve the TANGS research question and align the principal benchmark configuration used by the cited CDMAD FixMatch-family rows. The main table identifies every published row with its citation marker and labels TANGS as the locally executed method.

| Item | Locked value |
|---|---|
| Architecture | exact `CDMAD/wrn.py` WRN-28-2 from scratch; LeakyReLU 0.1; unused 4-way rotation head retained; executed BatchNorm defaults `eps=1e-5`, `momentum=0.1`; TANGS scope is `model.output` weight+bias |
| Image size | 32×32 |
| Normalization | CIFAR-10/STL-10 mean `(0.4914,0.4822,0.4465)`, std `(0.2471,0.2435,0.2616)`; CIFAR-100 mean `(0.5071,0.4867,0.4408)`, std `(0.2675,0.2565,0.2761)` |
| Precision | FP32; AMP disabled in v4.5 |
| Optimizer | Adam defaults: `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, `amsgrad=False` |
| Learning rate | constant \(1.5\times10^{-3}\); no scheduler |
| Labeled batch \(B_l\) | 32 |
| Unlabeled ratio \(\mu\) | 2 |
| Unlabeled batch \(B_u\) | 64 |
| Train loaders | labeled/unlabeled `shuffle=True`, `drop_last=True`, `num_workers=4`, `persistent_workers=True`; restart iterator on exhaustion without respawning workers |
| Test loader | batch 200, `shuffle=False`, `drop_last=False`, `num_workers=4`, `persistent_workers=True` |
| Training budget | 250,000 optimizer steps |
| Evaluation-model EMA decay | 0.999 |
| EMA mechanics | preserve `WeightEMA` exactly: EMA-to-online initial copy; state-dictionary entries updated after Adam; floating entries multiplied by `1-wd*lr` |
| FixMatch confidence threshold | 0.95 |
| Loss weights | `loss=Lx+Lu`; \(\lambda_u=1.0\); no auxiliary loss |
| Weak augmentation | released ordering: STL resize to 32 where applicable, random crop with padding 4, horizontal flip, tensor conversion, normalization |
| Strong augmentation | two independent released strong views: RandAugment(3,4), resize where applicable, crop/flip, tensor/normalization, Cutout(16) |
| Pseudo-label | detached hard weak-view argmax; shared across both strong views |
| Test transform | CIFAR tensor+normalization; STL resize to 32 then tensor+normalization |
| CIFAR-10 custom decay coefficient | 0.04 |
| CIFAR-100 custom decay coefficient | 0.08 |
| STL-10 custom decay coefficient | 0.01 |
| Progress and structured logging | upstream per-step progress retained; structured metrics buffered every 50 steps identically for all local methods |
| Checkpoint cadence | rolling checkpoint every epoch; snapshots every 100 epochs/50,000 steps plus final |
| Confirmatory checkpoint | final EMA checkpoint at step 250,000 |
| Determinism correction | `cudnn.deterministic=True`, `cudnn.benchmark=False` |

The released `WeightEMA` is preserved as executed, not “cleaned up”: its constructor copies EMA-model state into the online model, and `step()` iterates `state_dict()` values rather than only trainable parameters. Floating parameters and buffers receive the EMA update and online factor \(1-\mathrm{wd}\cdot\mathrm{lr}\). This is separate from Adam, whose `weight_decay` remains zero.

Persistent workers are an execution-layer correction for cycling loaders, especially on Windows. They preserve dataset membership and the objective but may change augmentation RNG consumption compared with respawning workers. Therefore every local comparison locks the same worker policy, and each split manifest records `num_workers`, `persistent_workers`, and `pin_memory`.


### 3.3 Development Protocol

Hyperparameters and baseline tuning use only DEV-C10-100:

- first run the fixed 20k paired P0 pilot: FixMatch and TANGS with `tau=5`, both on the identical seed-0 development split;
- P0 costs 40k optimizer steps total (0.16 of one 250k run), uses final held-out validation metrics, and is excluded from the main table;
- `scripts/analyze_minimal_gate.py` records PASS, HOLD, or STOP in `minimal-gate-c10-100-20000step.json`; after G0 is DONE, PASS permits the registered 50k sweep and HOLD permits only a paired 50k extension;
- observed P0 outcome on 2026-08-13: PASS. FixMatch versus TANGS final development bACC was 68.20 versus 69.49 (+1.29 pp), and tail accuracy was 45.00 versus 58.33 (+13.33 pp); Overall changed by -5.91 pp and Head by -10.32 pp, so P0 supports further tuning but not a claim of uniform improvement;
- run one paired 50k FixMatch development reference before the \(\tau\) candidates; it is a tuning control only and never enters C1 or the paper main table;
- run all four registered \(\tau\) values for comparability, with \(\tau=10\) and infinity first operationally because P0 showed excessive Head/Overall loss at \(\tau=5\); ordering does not permit early freezing;
- admit a \(\tau\) candidate only when, versus the paired 50k FixMatch reference, validation bACC gain is at least +0.25 pp, GM loss is no worse than -0.25 pp, Head loss is no worse than -5 pp, and Overall loss is no worse than -3 pp;
- among admissible candidates, select by bACC, then GM, Head retention, Overall retention, and larger \(\tau\); if none is admissible, block 250k training and open a separately versioned method-revision cycle;
- write an intermediate machine-readable TANGS screen after the paired FixMatch and four \(\tau\) runs; only a PASS launches the three head-downweight controls, saving 150k steps when no current TANGS candidate is viable;
- use `manualSeed=0`, matching the released README/default path; the local 20% labeled validation partition is an explicit v4.5 development-only addition;
- reserve 20% of each labeled class using deterministic class-stratified sampling, and remove those validation indices from both the DEV labeled loader and the overlapping DEV unlabeled loader; confirmatory loaders remain unchanged;
- never inspect the CIFAR-10 test set during tuning;
- maximum 12 completed short development runs in total across FixMatch, TANGS, and all direct baselines;
- every short development run stops at 50,000 steps and therefore costs 0.2 full-run equivalents;
- allocate one run to the paired FixMatch reference, at most four to \(\tau\), at most three to head-downweighting, and reserve at most four for documented tie-breaking or failed configuration repair;
- keep \(\beta=0.99\) and warm-up = 2500 fixed in the minimum plan; their sensitivity is optional and cannot change v4.5 confirmatory results;
- matched clipping and PCGrad-adapted use their deterministic registered definitions; external LTSSL methods are not tuned or run locally in the minimum plan;
- record every tried configuration, not only the winner;
- freeze all TANGS defaults and baseline-specific choices before confirmatory runs;
- exclude development runs from main statistical tables.

### 3.4 Seeds

- every minimum-plan local run uses `manualSeed=0` exactly once, matching the released default and README examples;
- `manualSeed` jointly controls Python, NumPy, Torch, augmentation, loader order, and the dataset function's `rand_number`;
- TANGS and all local mechanism controls reuse the identical saved seed-0 index arrays and base configuration;
- with deterministic cuDNN, an identical rerun is a reproducibility check, not an independent statistical trial;
- local results are labeled `manualSeed=0`, matching the released CDMAD execution protocol;
- nonzero seeds are optional robustness experiments under a separately versioned protocol; for CIFAR-100 they also change the sampled split.

---

## 4. Evaluation and Test-Set Discipline

### Development

- the completed 20k P0 evaluated DEV-C10-100 every 5,000 steps and used only the final 20k point for its gate;
- registered 50k development sweeps evaluate DEV-C10-100 every 500 steps and must pass `--max-steps 50000` explicitly;
- filter \(\tau\) with the locked bACC/GM/Head/Overall balance gate, then select using the registered validation ordering; select the head-downweight control independently by validation bACC, then GM and Head accuracy;
- do not report development values as final evidence.

### Confirmatory

- train for the full locked budget;
- primary table uses the final EMA checkpoint only;
- evaluate the test set after the configuration is frozen;
- never choose a checkpoint from test performance.

### Dynamics Curves

For B3, checkpoints are predeclared at:

\[
\{50k,100k,150k,200k,250k\}.
\]

After every compared run is complete and no configuration will change, evaluate these checkpoints offline. Curves are descriptive and cannot select models.

---

## 5. Metrics

### 5.1 Performance

Primary:

- balanced accuracy, implemented as mean per-class recall.

Secondary:

- head, medium, and tail mean per-class accuracy;
- geometric mean of per-class accuracy;
- worst-class accuracy;
- head/medium/tail differences against registered local mechanism controls;
- effective parameter-update diagnostics;
- wall-clock and GPU memory.

Do not present overall accuracy as independent evidence on class-balanced test sets because it equals balanced accuracy.

### 5.2 Pseudo-Label Quality

For a locked unlabeled evaluation stream with ground truth used only offline:

- tail precision = accepted predicted-tail correct / all accepted predicted-tail;
- tail recall = accepted true-tail predicted correctly / all true-tail samples;
- tail coverage = accepted true-tail / all true-tail samples;
- report macro-per-class and micro group versions;
- also report accepted predicted-head error composition by true head/medium/tail.

### 5.3 Statistics

- report each raw TANGS seed-0 result, each cited paper baseline, and their percentage-point difference;
- do not report a local mean, SD, SE, confidence interval, or significance test from one run;
- label the proposed row `TANGS (ours)` and retain the published uncertainty notation on cited rows;
- keep any optional nonzero-seed robustness table separate from the seed-0 main table;
- use “pp” for percentage-point differences;
- identify reported rows with citation markers in the same main table;
- make numerical performance comparisons without presenting a one-seed result as a significance test.

---

## 6. Phase U0 — Integrity Tests

All U0 rows are blocking for every post-P0 experiment, including the A1/A2/A3 diagnostic pilot, 50k development runs, and 250k confirmatory runs. The fixed development-only P0 pair is the sole exception: it may run after `U0-LOCAL-TESTS` and `U0-LOCAL-SMOKE` pass because it is a spending safeguard rather than paper evidence. Completing P0 does not satisfy or bypass G0.

### U0a. Exact Decomposition

For random non-empty batches:

\[
\left\|
g_{\mathrm{base}}
-
\left(
g_h+
\left[g_{\mathrm{base}}-g_h\right]
\right)
\right\|
\le10^{-6}\max(1,\|g_{\mathrm{base}}\|).
\]

Also compare the implementation's extracted \(g_h\) with an independently computed exact contribution.

### U0b. No-Op Equivalence

With projection disabled and \(\tau=\infty\):

- classifier gradient matches FixMatch within relative \(10^{-6}\);
- backbone gradient matches FixMatch within relative \(10^{-6}\);
- one optimizer step matches, including optimizer state.

### U0c. Geometry

Test synthetic vectors covering:

- positive, zero, and negative cosine;
- raw ratio above \(\tau\) but post-projection ratio below \(\tau\);
- zero/near-zero anchor;
- non-amplification;
- conflict removal.

### U0d. Runtime Safety

Test:

- empty tail mask;
- empty accepted predicted-head mask;
- first valid anchor initialization;
- warm-up boundary;
- NaN/Inf bypass;
- FP32 enforcement and rejection of AMP-enabled configurations;
- optimizer momentum/effective update logging;
- no retained graph across steps.

### U0e. Upstream Substrate Parity

Before any post-P0 TANGS run:

- assert the resolved model, precision, optimizer, learning-rate, loader, transform, loss-weight, EMA, and checkpoint settings equal Section 3.2;
- verify the WRN rotation head and executed BatchNorm values are unchanged;
- compare one base optimizer-plus-`WeightEMA` step against the pinned code path, including floating buffers;
- verify saved split indices, loader cardinalities, overlap, batch shapes, and iterator-restart behavior;
- fail if any upstream quirk was silently “fixed.”

### U0f. Artifact Trace

A short run must produce:

- immutable config;
- config hash;
- split and partition hashes;
- checkpoint;
- metrics JSON;
- gradient diagnostics;
- run manifest containing git commit and environment.

---

## 7. Phase A — Diagnostics

### A1. Occurrence and Eligibility

Protocol: P-C100-100, ordinary TANGS run; all geometry is logged before the intervention is applied.

Log:

- total steps;
- eligible steps;
- accepted predicted-head count;
- current labeled-tail count;
- raw cosine \(g_h\) vs. current \(g_t\);
- effective cosine \(g_h\) vs. EMA anchor;
- raw ratio and post-projection ratio;
- conflict/domination/either fractions over eligible steps;
- corresponding fractions over all steps;
- anchor update fraction, age, norm, invalid fraction;
- effective optimizer-step norm/cosine.

Use streaming aggregation; store a bounded diagnostic sample rather than every full gradient vector.

### A2. True-Class Offline Decomposition

For accepted samples predicted as head:

| Offline group | Required outputs |
|---|---|
| true head | count, accuracy, cosine, norm |
| true medium | count, error rate, cosine, norm |
| true tail | count, error rate, cosine, norm |

This determines whether the measured phenomenon is mainly pseudo-label error, ordinary correct-head learning, or both.

### A3. Anchor Quality

Report:

- group-anchor norm distribution;
- anchor age distribution;
- invalid/stale fraction;
- per-class tail-gradient cosine where a class is represented;
- within-tail cancellation ratio.

If the tracker anchor-quality gate fails, stop confirmatory TANGS runs and evaluate a per-class or class-balanced anchor redesign as a new method version.

### A4. Exploratory Temporal Analysis

Optional:

- use development validation or locked offline checkpoints;
- remove common time trend;
- report lag definition;
- use block bootstrap or another autocorrelation-aware uncertainty estimate;
- label as association, not causality.

---

## 8. Phase B — Mechanism Controls

### B1. Oracle-Target-Only

Protocol: P-C100-100.

At each step:

1. obtain ordinary weak-view predictions;
2. compute the ordinary confidence mask;
3. record ordinary predicted-head membership;
4. keep the accepted set and membership unchanged;
5. replace only accepted pseudo targets with ground truth;
6. retain the same strong views, denominator, \(\lambda_u\), optimizer, and schedule.

Run paired:

- FixMatch-oracle-target;
- FixMatch-oracle-target + TANGS.

Use `manualSeed=0`. Compare ordinary TANGS with oracle-target TANGS using mean intervention-gradient norm over eligible steps as the primary mechanism quantity. The paired oracle bACC result is supporting evidence.

### B2. Full Oracle

Optional upper bound:

- use all unlabeled ground truth;
- clearly label as fully supervised upper bound;
- do not use it as the primary causal control.

### B3. Training Dynamics

Use the locked checkpoint grid from Section 4 for TANGS and registered local controls. Report bACC, head, tail, eligible frequency, and effective-update diagnostics over time.

---

## 9. Phase C — Comparative Experiments

### C1. Cross-Dataset Main Table

Protocols:

- C1a: P-C10-100;
- C1b: P-C100-100;
- C1c: P-STL10-10, cited benchmark row and optional local TANGS extension;
- C1d: P-STL10-20.

Required local runs are only:

- FixMatch+TANGS on C1a, C1b, and C1d.

The CDMAD-style main table reuses audited CDMAD-2024 Tables 1, 2, and 4 values for FixMatch, DARP, DARP+cRT, CReST, CReST+LA, ABC, CoSSL, UDAL, and CDMAD. LCGC and later methods may be added wherever the registered benchmark setting matches. These baseline methods are not rerun locally.

Published and local rows appear in one main table, following CDMAD. Method-label superscripts or a `Source` column identify cited rows; the proposed row is marked `TANGS (ours)`. SimPro is added after its remaining `[VERIFY]` fields are resolved.

Report local seed-0 bACC, H/M/T, GM, and worst-class without an uncertainty suffix. Retain the source paper's bACC/GM and SE/SD notation for published rows, and report direct percentage-point differences from the cited FixMatch baseline.

### C2. Direct Debiasing and Optimization Baselines

Protocol: P-C100-100.

**C2a Head downweighting**

- downweight the same accepted predicted-head loss contribution;
- tune the scalar with at most 12 development runs.

**C2b Matched head-only clipping**

- operate on the same exact \(g_h\);
- use a cap matched to \(\tau\|a_t\|\);
- no projection.

**C2c PCGrad-adapted**

- use the same \(g_h\) and anchor definitions;
- remove negative conflict only;
- no norm cap.

Under the locked v4.5 definitions, C2c is mathematically identical to D1b. Run it once under `C2-PCG`; register the same seed-0 artifact for D1b rather than launching a duplicate job.

Fairness:

- common saved indices, `manualSeed`, base objective, architecture, total optimizer steps, metrics, and checkpoint rule;
- all methods share the global cap of 12 completed short development runs;
- report all tried configurations.

### C3. Optional CDMAD Complementarity

Protocol: P-C100-100.

- CDMAD;
- CDMAD + TANGS.

This is outside the minimum budget. TANGS may modify only a reviewer-audited exact pseudo-label contribution of the CDMAD objective. Repeat U0 and use matched `manualSeed=0` runs; restrict any claim to that fixed setting.

### C4. ReMixMatch Transfer

Optional appendix:

- ReMixMatch;
- ReMixMatch + TANGS;
- matched `manualSeed=0` runs.

This experiment supports a ReMixMatch backbone-transfer result on the evaluated seed-0 benchmark.

---

## 10. Phase D — Ablations

### D1. Components

Protocol: P-C100-100, `manualSeed=0`.

| ID | Variant |
|---|---|
| D1a | full TANGS |
| D1b | no norm cap; reuse the C2c PCGrad-adapted artifact |
| D1c | no projection |
| D1d | instantaneous current-tail anchor |

All variants use exact objective contributions and identical skip logic. D1b adds zero new runs because its locked operator, seed, and protocol are exactly C2c.

### D2. Anchor Redesign Contingency

Run only if A3 fails:

- class-balanced group anchor;
- per-class EMA anchors with deterministic aggregation.

These are a new method version and may not be silently substituted into v4.5 confirmatory runs.

### D3. Sensitivity

Development protocol only:

- required short screen: \(\tau\in\{2,5,10,\infty\}\) at 50,000 steps and one development seed;
- optional appendix only: \(\beta\in\{0.95,0.99,0.999\}\);
- optional appendix only: warm-up \(\in\{0,2500,5000\}\).

Count every completed sensitivity trial against the global 12-short-run development cap. Only the pre-confirmatory \(\tau\) screen may select a v4.5 value.

---

## 11. Phase E — Cost

Protocol: P-C100-100.

Instrument the P-C100-100 TANGS run with:

- exclusive GPU allocation;
- five timed windows after CUDA warm-up;
- explicit synchronization around timing;
- peak allocated and reserved memory;
- total end-to-end training time;
- extra-gradient and surgery-section time;
- bytes held by temporary gradient tensors and persistent anchor state;
- eligible-step and intervention-step microbenchmarks.

Report extra-gradient/surgery time as a fraction of synchronized training-step time excluding deep diagnostics, and report the known gradient/anchor tensor footprint as a fraction of peak allocated memory. Exclude only one-time environment setup outside training.

---

## 12. Pre-Registered Decision Rules

The authoritative thresholds are duplicated in EXPERIMENT_TRACKER.md and must match exactly. G0 is a prerequisite for every experiment after the fixed P0 pair; a P0 PASS makes the 50k freeze eligible only after G0 is DONE.

| Gate | Pass rule |
|---|---|
| G0 integrity | all U0 tests pass |
| P0 minimal idea gate | paired 20k artifacts match; eligible steps ≥100; anchor coverage ≥90%; conflict-or-domination ≥10%; a nonzero modification is observed; bACC drop ≤0.5 pp; and bACC gain ≥0.25 pp or tail gain ≥0.5 pp |
| G1 occurrence | either-trigger fraction ≥15% of eligible steps and at least one of conflict ≥5% or domination ≥10% |
| G2 anchor availability | valid anchor on ≥90% of post-warm-up steps |
| G3 anchor cancellation | median cancellation ratio ≥0.10 |
| G4 core efficacy | seed-0 TANGS exceeds the CDMAD-reported FixMatch bACC by ≥1.0 pp on P-C100-100 |
| G5 cross-dataset consistency | seed-0 TANGS exceeds the cited FixMatch bACC on at least 2 of 3 required settings |
| G6 tail specificity | seed-0 TANGS tail accuracy exceeds both head downweighting and matched clipping by ≥0.5 pp |
| G7 oracle mechanism | oracle-target TANGS reduces mean intervention-gradient norm over eligible steps by ≥50% relative to ordinary TANGS |
| G8 specificity | seed-0 TANGS exceeds head downweighting and matched clipping by ≥0.5 pp bACC |
| G9 components | seed-0 full TANGS exceeds both no-cap and no-projection variants |
| G10 overhead | extra-gradient/surgery time ≤20% of synchronized training-step time and known gradient/anchor tensor footprint ≤15% of peak allocated memory |
| G11 optional complementarity | if C3 is run, seed-0 CDMAD+TANGS exceeds matched CDMAD by ≥0.5 pp bACC without head drop below −1.0 pp |

Changing a threshold after confirmatory results exist creates a new preregistration version and must be disclosed.

For G7, intervention burden is the mean `gradient_delta_norm` over eligible steps. Use the ordinary C1b-TG stream and the oracle-target TANGS stream with the same logging rule.

---

## 13. Run Budget and Scheduling

The minimum v4.5 evidence package contains 10 new full runs, a fixed paired 20k P0 pilot (0.16 full-run equivalents), plus at most 12 short 50k-step development runs (2.4 full-run equivalents) and U0/diagnostic pilots. Optional experiments are not launched from the minimum budget.

| Block | New full runs |
|---|---:|
| C1 local TANGS seed-0 rows | 3; FixMatch-family rows are cited from audited papers |
| B1 oracle-target seed-0 pair | 2 |
| C2 seed-0 head-downweighting, clipping, PCGrad | 3 |
| D1 seed-0 new ablations | 2; no-cap reuses C2-PCG |
| E1 timing/memory | 0; reuse instrumented C1b-TG seed-0 run |
| **Required total** | **10** |

Do not use a fixed calendar promise. Schedule from measured pilot wall time and available parallel GPUs.

Recommended stages:

1. Complete `U0-LOCAL-TESTS` and `U0-LOCAL-SMOKE`, the local-readiness subset required for P0.
2. Run the paired 20k P0 gate. PASS makes the 50k freeze eligible after G0; HOLD permits only a paired 50k extension after G0; STOP blocks further paid experiments.
3. Complete every remaining U0 audit row and mark G0 DONE.
4. Run one short A1/A2/A3 diagnostic pilot; stop if G1–G3 fail.
5. Run DEV-C10-100 short tuning under the global 12-run cap.
6. Complete the core P-C100-100 TANGS run and compare it with the audited CDMAD-table FixMatch baseline.
7. Stop the efficacy paper if G4–G6 fail; otherwise run B1, C2 mechanism controls, and D1.
8. Run the two remaining local TANGS rows: P-C10-100 and P-STL10-20.
9. Compute E1 from the instrumented C1b-TG run.
10. Launch A4, B2, C3, C4, D2, beta/warm-up sensitivity, or P-STL10-10 locally only with a separate optional budget.

Before launching a stage, compute:

\[
\text{estimated GPU hours}
=
\text{number of runs}
\times
\text{pilot hours per run}.
\]

The tracker must contain planned and actual GPU hours.

---

## 14. Artifact and Review Workflow

Each run writes:

- immutable YAML/JSON config;
- run manifest with hashes and git commit;
- raw training log;
- metrics JSON with metric definitions;
- checkpoint;
- diagnostic summary;
- environment and hardware record.

Each aggregate table writes:

- list of source run IDs;
- aggregation script version;
- raw seed-0 values and matched differences, with no local uncertainty suffix;
- explicit distinction between uncertainty-free local seed-0 values and paper-reported SE/SD.

Final integrity review:

- executor provides raw file paths, not summaries;
- reviewer independently audits code, configs, result artifacts, and table generation;
- no claim enters the paper until its tracker gate passes.

---

## 15. Venue Lock

Before paper drafting, record:

- exact conference name and year;
- CCF tier source/date;
- page limit and reference policy;
- anonymous/non-anonymous submission rules;
- artifact, ethics, and AI-use disclosure requirements;
- submission deadline and timezone.

Until then, this plan assesses research soundness for a CCF-C-level target but does not claim venue-specific readiness.
