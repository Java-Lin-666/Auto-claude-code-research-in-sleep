# Reported Results From Primary Papers

*Specification version: 4.8 selective score-calibration policy — literature registry unchanged*
*Aligned with the TANGS v4.8 proposal, plan, and tracker.*
*Purpose: preserve audited paper-reported numbers used as citation-marked baselines in the TANGS main comparison table. Local development status may be noted only to track what remains; it is never a source for a reported-baseline row.*

## Evidence and Citation Rules

Every numerical block must identify:

- primary-paper URL or DOI/arXiv ID;
- exact table/appendix location;
- dataset counts and distribution setting;
- metric definition;
- number of runs when stated;
- whether the uncertainty is SD, SE, CI, or unresolved;
- whether the values were visually checked against the primary PDF.

Do not infer an uncertainty type from the “+/-” symbol or convert SE to SD without showing the conversion. In the combined main table, retain published SE/SD on cited rows and label the proposed row `TANGS (ours)`; the experimental setup records the locked seed-0 execution protocol once.

Any source-registry row containing `[VERIFY]` is quarantined: its retained numerical block may be used to guide a future audit, but no value from that block may be copied into the paper, used for a superiority claim, or treated as a checked baseline until the source URL, exact row, metric, protocol, run count, and uncertainty type are verified and the registry state is updated.

The repository currently does not contain the source PDFs represented by the old aliases paper1–paper5. The official URLs below are therefore the traceable sources; if local PDFs are added later, record their paths and SHA256 values.

### v4.8 Local-Comparison Policy (Controlling)

v4.7 is a genuine development STOP and must not be reported as successful.
v4.8 retains exact FixMatch training and adds no gradient intervention. Its
scoring parameters are frozen at uniform-LA 0.80, base prior 0.70, extra tail
0.40, anchor cosine 0.775, top-two anchor margin 0.05, and raw-confidence
ceiling 0.90.

The two seed-0 development re-evaluations are retrospective selection evidence
only. Their v4.8 bACC values (39.86 and 39.84) and corresponding LA-0.80 values
(39.22 and 39.20) must never enter a paper main table or be described as an
independent validation. They authorize only the preregistered seed-1 C100 50k
holdout. The holdout also remains development evidence even if it passes.

A paper-facing efficacy row requires, in order: seed-1 holdout PASS, frozen
C100 seed-0 250k PASS, then the required C10-100 and STL10-20 seed-0 transfer
runs. Raw FixMatch, uniform LA, and v4.8 scores must come from the same final
EMA checkpoint for every local row. Published baselines remain citation-marked
context and must not be described as local reproductions.

The v4.8 novelty claim is the combination of supervised tail-anchor
compatibility, anchor-identity separation, and uncertainty-limited selective
tail correction. The global prior term is established logit adjustment and is
not novel. Any comparison against LA must use the frozen head-compliant
`alpha=0.80` control and disclose Medium-class changes.

### v4.7 Local-Comparison Policy (Historical)

**Historical:** the integrated v4.7 development gate returned STOP. This
policy is retained only for provenance and cannot authorize experiments.

TANGS v4.7 is a deterministic score-calibration method, not a training-gradient
method. The v4.5 and v4.6 gradient-surgery results remain archived failures and
must never be relabeled as v4.7 evidence.

The core local comparison uses one exact FixMatch training run with a
measurement-only anchor observer. Its final EMA checkpoint supplies three
scoring rows: raw logits, uniform logit adjustment, and TANGS v4.7. Therefore:

- these rows are paired without training stochasticity and must not be
  described as three independent runs;
- the frozen uniform-LA control is `alpha=0.85`, selected on the disjoint C100
  development set under the same maximum-two-point Head-loss guard;
- the v4.7 constants are base `0.65`, extra tail `0.25`, and cosine threshold
  `0.75`; they cannot change after confirmatory evaluation;
- retrospective old-checkpoint values (raw 36.70, LA 39.42, v4.7 candidate
  40.04 bACC) are PROVISIONAL only and never enter the paper main table as
  confirmatory results; they authorize at most one fresh integrated C100 50k
  development run;
- the unconstrained LA development optimum (41.30 bACC at `alpha=1.55`, Head
  63.27) remains visible in the audit record. Any v4.7-vs-LA claim must say
  **under the registered Head-preservation constraint**;
- v4.7 loses 2.24 Medium points to the frozen LA control in development, so a
  selective-tail gain cannot be summarized as uniform group improvement.

Published FixMatch/DARP/ABC/CReST/CoSSL/CDMAD/LCGC rows below remain
citation-marked context, not local reproductions. Before any 250k run, a fresh
integrated C100 50k development run must PASS the frozen gate on the held-out
development split. A CCF-C efficacy claim then requires the frozen 250k C100,
C10-100, and STL10-20 results, plus direct post-hoc controls such as uniform LA
and tau-normalization/cRT-style calibration. C100 runs first as a stop-loss
gate; it is not the complete matrix. P-STL10-10 remains optional. Do not claim
state-of-the-art from either the retrospective PROVISIONAL signal or a 50k
development PASS.

The post-hoc prior term follows the established logit-adjustment family; that
term alone is not novel. The proposed contribution that must be isolated is
the one-class, supervised-tail-anchor compatibility gate. Cite the original
logit-adjustment work when the paper bibliography is assembled, and compare
against the local frozen LA row rather than implying that prior correction was
invented here.

### v4.6 Local-Baseline Policy (Historical)

The citations and numerical values in this file remain literature context; the
v4.6 recovery does not alter them. It does alter how efficacy is established.

The v4.5 first matrix showed that a reported FixMatch number alone is
insufficient for causal diagnosis under the derived local implementation:
v4.5 TANGS reached 36.53% C100 bACC versus the paper-reported 37.6%, but no
exact paired local FixMatch full run existed. The result therefore cannot
separate a weak local substrate realization from harm caused by surgery.

For v4.6:

- the 50k C100 development gate includes an exact paired local FixMatch run;
- after a PASS, the first 250k stage includes another exact paired local
  FixMatch run plus `tangs-v46`, with identical seed, split, loader policy,
  objective, schedule, and final-EMA evaluation;
- the local paired difference is the primary efficacy estimate for the method
  revision;
- CDMAD Table 4 FixMatch (37.6 +/- 0.48 bACC) remains a cited external context
  row and implementation-alignment check, not a substitute for the pair;
- DARP/ABC/CDMAD/LCGC and other verified rows remain reported comparisons and
  must never be described as locally reproduced;
- v4.5 C10/C100 values and every development value remain historical or
  development evidence, never v4.6 main-table rows.

No cross-dataset v4.6 main-table claim is allowed until the paired C100 core
stage passes and later protocol rows are separately authorized.

### Target-Server Smoke Evidence Status (2026-08-14)

Target-environment readiness is complete but is **not** a literature or local efficacy result. On the RTX 4090 deployment commit `c5f74189d81a00402cd183a49cc4dc74d3ddf5b0`, the server suite passed 53 tests. The observer smoke is DONE at `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-fixmatch-observer-seed0` (config `7f8c3ce7443e0162d1b5299ef454f35b10d7e80c6b9ae91c233f04b18ce14ebd`), the full warm-up smoke is DONE at `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-seed0` (config `98408fa20e4f3a0aa7b34ae2a30b29b768283429bcbc836d396eabf0bf2a754c`), and valid checkpoint/resume is DONE at `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-v2-seed0` (config `cc52d447219a3802b0518afceb2d32f0ec91a7907c5305718e2dbc02d99d91c2`). All relevant serialized values were finite; the observer had zero modifications. The completed-before-SIGTERM artifact is preserved but invalid for resume evidence at `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-seed0` (config `89dc6007160bf217dd4ec036cc23c9b19aaf99cd0503ab45fbb0e4eaf28f7040`). None of these records alters a reported paper value or authorizes a main-table claim; the paired 50k development gate is still unstarted and 250k remains blocked pending its PASS.

## Primary Source Registry

| ID | Primary source | Numerical location used here | Reported uncertainty | Audit state |
|---|---|---|---|---|
| DARP-2020 | [NeurIPS paper](https://proceedings.neurips.cc/paper/2020/hash/a7968b4339a1b85b7dbdb362dc44f9c4-Abstract.html), [PDF](https://papers.nips.cc/paper/2020/file/a7968b4339a1b85b7dbdb362dc44f9c4-Paper.pdf) | Tables 1 and 3; experimental setup §4.1 | mean ± SD over 3 random trials | source, counts, metrics, uncertainty, and required rows checked |
| ABC-2021 | [NeurIPS paper](https://proceedings.neurips.cc/paper/2021/hash/3953630da28e5181cffca1278517e3cf-Abstract.html), [PDF](https://papers.nips.cc/paper/2021/file/3953630da28e5181cffca1278517e3cf-Paper.pdf) | method source only | mean ± SD; 5 runs in its main setting and 3 in other settings | original numeric protocol is not a TANGS protocol; do not use its main table as a direct substitute |
| CDMAD-2024 | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_CDMAD_Class-Distribution-Mismatch-Aware_Debiasing_for_Class-Imbalanced_Semi-Supervised_Learning_CVPR_2024_paper.html) and [supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Lee_CDMAD_Class-Distribution-Mismatch-Aware_Debiasing_CVPR_2024_supplemental.pdf) | main Tables 1–5; supplement setup and baseline provenance | mean ± SE over 3 runs for CIFAR/STL tables | rows, counts, base configuration, uncertainty, and baseline provenance checked; primary main-table comparison source for v4.5 |
| LCGC-2025 | [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/35474), [arXiv PDF](https://arxiv.org/pdf/2504.06544) | Tables 1,2,4,5 | mean ± SE over 3 runs | values spot-checked; label as SE |
| ACR-2023 | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.html) | Tables 1,2 | mean ± SD over 3 independent runs | source, uncertainty statement, and representative values checked |
| CPE-2024 | [arXiv:2312.15702](https://arxiv.org/pdf/2312.15702) | Tables 2,3 | mean ± SD over 3 seeds | values spot-checked |
| SIMPRO-2024 | [PMLR paper](https://proceedings.mlr.press/v235/du24b.html) | Tables 1,2 | [VERIFY] | values spot-checked; uncertainty label pending |
| DASO-2022 | [arXiv:2106.05682](https://arxiv.org/abs/2106.05682) | Table 1; Appendix Table 12 | [VERIFY] | values retained; row audit pending |
| BACON-2024 | [arXiv:2403.12986](https://arxiv.org/abs/2403.12986) | Tables 1,2 | [VERIFY] | values retained; row audit pending |
| ADELLO-2024 | [arXiv:2306.04621](https://arxiv.org/abs/2306.04621) | Table 3; Appendix B; Tables 11,12 | [VERIFY] | caption/metric ambiguity retained explicitly |
| CPG-2025 | [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/abcd225747ec4a176a5ff59e56e0d2eb-Abstract-Conference.html) | no values copied | N/A | context only |

Unrelated sources retained only for exclusion notes:

- Adaptive Confidence Margin: facial-expression SSL, not the TANGS LTSSL benchmark.
- FixMatch original: balanced SSL background, not LTSSL evidence.
- GLMC: supervised long-tailed recognition.
- Memory-Efficient Semi-Supervised Continual Learning and SimPLE: different task settings.

## Alignment With Current TANGS Protocols

Required local protocols:

- P-C10-100: CIFAR-10-LT, \(N_1=1500\), \(M_1=3000\), \(\gamma_l=\gamma_u=100\).
- P-C100-100: CIFAR-100-LT, \(N_1=150\), \(M_1=300\), \(\gamma_l=\gamma_u=100\).
- P-STL10-20: STL-10-LT, \(N_1=450\), original 100k unlabeled pool, \(\gamma_l=20\).

P-STL10-10 is reported context and an optional local extension, not part of the minimum v4.5 full-run budget.

### Released-Code Alignment Audit

The vendored `CDMAD/` code is the clean official `LeeHyuck/CDMAD` commit `7cd732b4615b9d94934a9197e69c6775496fb5ee` and is the implementation substrate for local TANGS runs and mechanism controls. The audit found:

- `fixmatchcdmad.py` uses WRN-28-2, FP32, Adam at constant 1.5e-3 with library defaults and no scheduler, batches 32/64, 500 epochs × 500 steps, EMA 0.999, and dataset-specific custom decay 0.04/0.08/0.01;
- `wrn.py` retains LeakyReLU 0.1 and an unused four-way rotation head; its BatchNorm wrapper does not forward its declared `1e-3` arguments, so executed values are PyTorch defaults `eps=1e-5`, `momentum=0.1`;
- train loaders use `shuffle=True`, `drop_last=True`, four workers, and iterator restart on exhaustion; the test loader uses batch 200, `shuffle=False`, and four workers;
- `loss=Lx+Lu`, so the released FixMatch-family weight is 1.0 and there is no auxiliary loss;
- `WeightEMA` first copies EMA-model state into the online model, then updates `state_dict()` entries after Adam; floating buffers as well as parameters inherit the executed EMA/decay behavior;
- the data pipeline returns one weak and two strong views, with RandAugment(3,4) and Cutout(16);
- CIFAR unlabeled-loader indices use `idxs[:n_labeled+n_unlabeled]`, so the labeled prefix is included;
- the released STL-10 loader appends the selected labeled images to the official 100k unlabeled split, so the actual unlabeled-loader cardinality is `100000 + |L|`;
- CIFAR-10 and STL-10 do not shuffle class indices by seed; CIFAR-100 shuffles only when `manualSeed != 0`;
- `fixmatchcdmad.py` has one `manualSeed` argument with default 0, every README command passes 0, and the repository contains no multi-seed experiment driver; the paper's separate three-run SE statement is retained as paper-level provenance rather than inferred from the code;
- the released script sets deterministic flags and later re-enables `cudnn.benchmark=True`; v4.5 deliberately keeps benchmark disabled;
- `WeightEMA.step()` implements parameter decay as `param *= 1 - wd*lr`, not Adam weight decay;
- the released CDMAD branch uses soft pseudo-labels, default confidence threshold 0, white-image subtraction after epoch 100, and test-time subtraction.

The local TANGS protocol reuses this infrastructure, locks `manualSeed=0`, restores the hard 0.95-threshold FixMatch objective, and disables white-image subtraction. The resulting benchmark configuration aligns the principal dataset, model, optimizer, step-budget, augmentation, and metric choices used by the CDMAD FixMatch-family tables.

The seed-0 choice follows the released execution path. The TANGS row is reported as a raw seed-0 result, while cited rows retain their published uncertainty.

CDMAD's supplement states that reproducible baseline values were taken from Lai et al. and Fan et al., with other rows measured from uploaded code. The TANGS table follows the same literature-baseline convention and cites both the numerical table source and original method.

Reported numbers are used for:

- the primary benchmark comparison table;
- direct bACC/GM comparisons against TANGS;
- related-work and mechanism positioning.

They replace local reruns of plain FixMatch and established LTSSL baselines in C1. TANGS-specific mechanism controls, ablations, diagnostics, and cost measurements remain local.

The paired 20k DEV-C10-100 FixMatch/TANGS pilot and the v4.5 paired 50k FixMatch development reference are pre-compute/tuning controls only. Their held-out validation values never enter the paper main table and do not alter the zero-local-confirmatory-baseline policy for C1. The 50k reference is used only to enforce the locked bACC/GM/Head/Overall balance gate before freezing \(\tau\).

Main-table baseline policy:

- use CDMAD-2024 Tables 1, 2, and 4 as the primary numerical source for matching FixMatch-family settings;
- cite DARP, CReST, ABC, CoSSL, UDAL, and CDMAD rather than rerunning them locally;
- cite LCGC and later methods wherever the registered benchmark setting and metric match;
- keep SimPro, DASO, BaCon, and ADELLO quarantined wherever `[VERIFY]` remains;
- run only TANGS, B1, C2, D1, and E1 locally in the minimum plan.

Only registry rows without `[VERIFY]` may enter the main table. Each cited method carries a source marker, and direct differences from TANGS are calculated in percentage points.

## Citation-Ready CDMAD-Table Substitutions

These rows justify the zero-local-baseline-run policy in EXPERIMENT_TRACKER.md. Values are percentages and populate the citation-marked main comparison table alongside the local TANGS row.

| TANGS protocol | Method | Citation and row | Paper-reported result | Uncertainty | Benchmark alignment |
|---|---|---|---:|---|---|
| P-C10-100 | FixMatch | CDMAD-2024 Table 1, \(\gamma=100\) | 71.5 / 66.8 bACC/GM | SE, 3 runs | matching counts, imbalance, WRN-28-2 FixMatch family, and 250k-step setup |
| P-C100-100 | FixMatch | CDMAD-2024 Table 4, \(\gamma=100\) | 37.6 bACC | SE, 3 runs | matching counts, imbalance, WRN-28-2 FixMatch family, and 250k-step setup |
| P-STL10-10 | FixMatch | CDMAD-2024 Table 2, \(\gamma_l=10\) | 72.9 / 69.6 bACC/GM | SE, 3 runs | matching \(N_1=450\), original 100k unlabeled pool, and FixMatch family |
| P-STL10-20 | FixMatch | CDMAD-2024 Table 2, \(\gamma_l=20\) | 63.4 / 52.6 bACC/GM | SE, 3 runs | matching \(N_1=450\), original 100k unlabeled pool, and FixMatch family |
| P-C10-100 | FixMatch+DARP | DARP-2020 Table 1, \(\gamma_l=\gamma_u=100\) | 75.5 / 73.0 bACC/GM | SD, 3 trials | matching \(N_1=1500,M_1=3000\), imbalance setting, and bACC/GM metrics |
| P-C100-100 | FixMatch+DARP | CDMAD-2024 Table 4, \(\gamma=100\) | 38.3 bACC | SE, 3 runs | matching counts, imbalance, WRN-28-2 FixMatch family, and 250k-step setup |
| P-STL10-10 | FixMatch+DARP | DARP-2020 Table 3, \(\gamma_l=10\) | 77.8 / 76.5 bACC/GM | SD, 3 trials | matching \(N_1=450\), original 100k unlabeled pool, and bACC/GM metrics |
| P-STL10-20 | FixMatch+DARP | DARP-2020 Table 3, \(\gamma_l=20\) | 69.9 / 65.4 bACC/GM | SD, 3 trials | matching \(N_1=450\), original 100k unlabeled pool, and bACC/GM metrics |
| P-C10-100 | FixMatch+ABC | CDMAD-2024 Table 1, \(\gamma_l=\gamma_u=100\) | 81.1 / 80.3 bACC/GM | SE, 3 runs | matching counts, imbalance, WRN-28-2, Adam, batch 32, \(\mu=2\), and 250k steps |
| P-STL10-10 | FixMatch+ABC | CDMAD-2024 Table 2, \(\gamma_l=10\) | 79.1 / 78.1 bACC/GM | SE, 3 runs | matching \(N_1=450\), original 100k unlabeled pool, and FixMatch-family configuration |
| P-STL10-20 | FixMatch+ABC | CDMAD-2024 Table 2, \(\gamma_l=20\) | 73.8 / 72.1 bACC/GM | SE, 3 runs | matching counts, unlabeled pool, and FixMatch-family configuration |

ABC citation rule: cite ABC-2021 for the method and CDMAD-2024 for the numerical row. The ABC-2021 main table uses a different \(\gamma,\beta\) construction, overall/minority accuracy, and no STL-10 result, so it is not the numerical source for the substitutions above.

DARP citation rule: prefer DARP-2020 Tables 1/3 for P-C10-100 and STL-10 because the paper explicitly states the counts, bACC/GM metrics, and three-trial SD. Use CDMAD-2024 Table 4 for the P-C100-100 main-table row.

CDMAD reported-evidence rule: all CDMAD/FixMatch-family numbers retained below are citation-marked paper results. They appear in the same main table as TANGS with their published SE labels.

Statistical presentation:

- every minimum-plan local row uses `manualSeed=0` once;
- report the raw TANGS value and its percentage-point difference from each cited baseline;
- label the proposed row `TANGS (ours)` and record the seed-0 execution rule in the experimental setup;
- place any optional nonzero-seed robustness extension in a separate supplementary block;
- reported ReMixMatch values cannot support a TANGS transfer claim;
- local test results use the final locked EMA checkpoint, not best-test selection.

## Directly Useful As Reported LTSSL Baselines

### CIFAR-10-LT, consistent distribution, bACC/GM

Primary v4.5 source: CDMAD-2024 Table 1 for rows through CDMAD; LCGC-2025 Table 1 adds the later LCGC row and provides a cross-check. Setting: gamma = gamma_l = gamma_u, gamma_u known. Both papers report three-run mean +/- **standard error (SE)**. Treat every +/- value in this block as paper-reported SE, not as uncertainty attached to local seed-0 values.

| Method | gamma=50 | gamma=100 | gamma=150 |
|---|---:|---:|---:|
| FixMatch | 79.2+/-0.3 / 77.8+/-0.4 | 71.5+/-0.7 / 66.8+/-1.5 | 68.4+/-0.2 / 59.9+/-0.4 |
| FixMatch+DARP+cRT | 85.8+/-0.4 / 85.6+/-0.6 | 82.4+/-0.3 / 81.8+/-0.2 | 79.6+/-0.4 / 78.9+/-0.4 |
| FixMatch+ABC | 85.6+/-0.3 / 85.2+/-0.3 | 81.1+/-1.1 / 80.3+/-1.3 | 77.3+/-1.3 / 75.6+/-1.7 |
| FixMatch+CoSSL | 86.8+/-0.3 / 86.6+/-0.3 | 83.2+/-0.5 / 82.7+/-0.6 | 80.3+/-0.6 / 79.6+/-0.6 |
| FixMatch+SAW+LA | 86.2+/-0.2 / 83.9+/-0.4 | 80.7+/-0.2 / 77.5+/-0.2 | 73.7+/-0.1 / 71.2+/-0.2 |
| FixMatch+CDMAD | 87.3+/-0.1 / 87.0+/-0.2 | 83.6+/-0.5 / 83.1+/-0.6 | 80.8+/-0.9 / 79.9+/-1.1 |
| FixMatch+LCGC | 87.3+/-0.0 / 87.1+/-0.1 | 84.9+/-0.1 / 84.6+/-0.2 | 82.4+/-0.0 / 81.9+/-0.1 |
| ReMixMatch | 81.5+/-0.3 / 80.2+/-0.3 | 73.8+/-0.4 / 69.5+/-0.8 | 69.9+/-0.5 / 62.5+/-0.4 |
| ReMixMatch+ABC | 87.9+/-0.5 / 87.6+/-0.5 | 84.5+/-0.3 / 84.1+/-0.4 | 80.5+/-1.2 / 79.5+/-1.4 |
| ReMixMatch+CDMAD | 88.3+/-0.4 / 88.1+/-0.4 | 85.5+/-0.5 / 85.3+/-0.4 | 82.5+/-0.2 / 82.0+/-0.3 |
| ReMixMatch+LCGC | 88.7+/-0.1 / 88.5+/-0.1 | 85.7+/-0.4 / 85.4+/-0.4 | 82.8+/-0.3 / 82.4+/-0.4 |

### CIFAR-10-LT, mismatch distribution, bACC/GM

Primary v4.5 source: CDMAD-2024 Table 2 for rows through CDMAD; LCGC-2025 Table 2 adds the later LCGC row. Setting: gamma_l = 100, gamma_u unknown. Both primary papers state mean+/-SE over three runs for these CIFAR/STL experiments.

| Method | gamma_u=1 | gamma_u=50 | gamma_u=150 |
|---|---:|---:|---:|
| FixMatch | 68.9+/-2.0 / 42.8+/-8.1 | 73.9+/-0.3 / 70.5+/-0.5 | 69.6+/-0.6 / 62.6+/-1.1 |
| FixMatch+DARP | 85.4+/-0.6 / 85.0+/-0.7 | 77.3+/-0.2 / 75.5+/-0.2 | 72.9+/-0.2 / 69.5+/-0.2 |
| FixMatch+DARP+LA | 86.6+/-1.1 / 86.2+/-1.2 | 82.3+/-0.3 / 81.5+/-0.3 | 78.9+/-0.2 / 77.7+/-0.1 |
| FixMatch+DARP+cRT | 87.0+/-0.7 / 86.8+/-0.7 | 82.7+/-0.2 / 82.3+/-0.3 | 80.7+/-0.4 / 80.2+/-0.6 |
| FixMatch+ABC | 82.7+/-0.5 / 81.9+/-0.7 | 82.7+/-0.6 / 82.0+/-0.8 | 78.4+/-0.9 / 77.2+/-1.1 |
| FixMatch+CDMAD | 87.5+/-0.5 / 87.1+/-0.5 | 85.7+/-0.4 / 85.3+/-0.4 | 82.3+/-0.2 / 81.8+/-0.3 |
| FixMatch+LCGC | 88.2+/-0.4 / 87.8+/-0.4 | 85.9+/-0.4 / 85.4+/-0.4 | 84.0+/-0.2 / 83.7+/-0.2 |

### CIFAR-100-LT, consistent distribution, bACC

Primary v4.5 source: CDMAD-2024 Table 4 for rows through CDMAD; LCGC-2025 Table 4 adds the later LCGC row. Setting: gamma = gamma_l = gamma_u. Values are paper-reported mean+/-SE over three runs.

| Method | gamma=20 | gamma=50 | gamma=100 |
|---|---:|---:|---:|
| FixMatch | 49.6+/-0.78 | 42.1+/-0.33 | 37.6+/-0.48 |
| FixMatch+DARP | 50.8+/-0.77 | 43.1+/-0.54 | 38.3+/-0.47 |
| FixMatch+DARP+cRT | 51.4+/-0.68 | 44.9+/-0.54 | 40.4+/-0.78 |
| FixMatch+CReST | 51.8+/-0.12 | 44.9+/-0.50 | 40.1+/-0.65 |
| FixMatch+CReST+LA | 52.9+/-0.07 | 47.3+/-0.17 | 42.7+/-0.70 |
| FixMatch+ABC | 53.3+/-0.79 | 46.7+/-0.26 | 41.2+/-0.06 |
| FixMatch+CoSSL | 53.9+/-0.78 | 47.6+/-0.57 | 43.0+/-0.61 |
| FixMatch+UDAL | - | 48.0+/-0.56 | 43.7+/-0.41 |
| FixMatch+CDMAD | 54.3+/-0.44 | 48.8+/-0.75 | 44.1+/-0.29 |
| FixMatch+LCGC | 55.3+/-0.5 | 49.3+/-0.3 | 44.8+/-0.5 |
| ReMixMatch | 51.6+/-0.43 | 44.2+/-0.59 | 39.3+/-0.43 |
| ReMixMatch+ABC | 55.6+/-0.35 | 47.9+/-0.10 | 42.2+/-0.12 |
| ReMixMatch+CDMAD | 57.0+/-0.32 | 51.1+/-0.46 | 44.9+/-0.42 |
| ReMixMatch+LCGC | 57.3+/-0.3 | 50.7+/-0.4 | 45.9+/-0.6 |

### STL-10-LT, bACC/GM

Primary v4.5 source: CDMAD-2024 Table 2 for rows through CDMAD; LCGC-2025 Table 5 adds the later LCGC row. Setting: STL-10-LT with original unlabeled pool; gamma_u is unknown / N/A. Metric: bACC/GM. Values are paper-reported mean+/-SE over three runs.

Use in TANGS paper: usable as the main STL-10-LT reported baseline block for balanced long-tailed metrics. Do not mix numerically with top-1 accuracy tables.

| Method | gamma_l=10 | gamma_l=20 |
|---|---:|---:|
| FixMatch | 72.9+/-0.1 / 69.6+/-0.0 | 63.4+/-0.2 / 52.6+/-0.1 |
| FixMatch+DARP | 77.8+/-0.3 / 76.5+/-0.4 | 69.9+/-1.8 / 65.4+/-3.1 |
| FixMatch+DARP+LA | 78.6+/-0.3 / 77.4+/-0.4 | 71.9+/-0.5 / 68.7+/-0.5 |
| FixMatch+DARP+cRT | 79.3+/-0.2 / 78.7+/-0.2 | 74.1+/-0.6 / 73.1+/-1.2 |
| FixMatch+ABC | 79.1+/-0.5 / 78.1+/-0.6 | 73.8+/-0.2 / 72.1+/-0.2 |
| FixMatch+CDMAD | 79.9+/-0.2 / 78.9+/-0.4 | 75.2+/-0.4 / 73.5+/-0.3 |
| FixMatch+LCGC | 80.1+/-0.4 / 79.2+/-0.3 | 76.6+/-0.3 / 75.2+/-0.3 |

### CIFAR-10/100-LT, ACR reported test accuracy

Source: [Wei and Gan, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.html), Table 1. This uses top-1 test accuracy, not bACC/GM, and reports mean ± SD over three independent runs.

| Dataset / setting | N1 | M1 | FixMatch | DARP | DASO | ABC | ACR |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR10-LT gamma_l=gamma_u=100 | 500 | 4000 | 67.8+/-1.13 | 74.5+/-0.78 | 76.0+/-0.37 | 78.9+/-0.82 | 81.6+/-0.19 |
| CIFAR10-LT gamma_l=gamma_u=100 | 1500 | 3000 | 77.5+/-1.32 | 77.8+/-0.63 | 79.1+/-0.75 | 83.8+/-0.36 | 84.1+/-0.39 |
| CIFAR10-LT gamma_l=gamma_u=150 | 500 | 4000 | 62.9+/-0.36 | 67.2+/-0.32 | 70.1+/-1.81 | 66.5+/-0.78 | 77.0+/-1.19 |
| CIFAR10-LT gamma_l=gamma_u=150 | 1500 | 3000 | 72.4+/-1.03 | 73.6+/-0.73 | 75.1+/-0.77 | 80.1+/-0.45 | 80.9+/-0.22 |
| CIFAR100-LT gamma_l=gamma_u=10 | 50 | 400 | 45.2+/-0.55 | 49.4+/-0.20 | 49.8+/-0.24 | 47.5+/-0.18 | 55.7+/-0.12 |
| CIFAR100-LT gamma_l=gamma_u=10 | 150 | 300 | 56.5+/-0.06 | 58.1+/-0.44 | 59.2+/-0.35 | 59.1+/-0.21 | 65.6+/-0.16 |
| CIFAR100-LT gamma_l=gamma_u=20 | 50 | 400 | 40.0+/-0.96 | 43.4+/-0.87 | 43.6+/-0.09 | 41.6+/-0.83 | 48.0+/-0.75 |
| CIFAR100-LT gamma_l=gamma_u=20 | 150 | 300 | 50.7+/-0.25 | 52.2+/-0.66 | 52.9+/-0.42 | 53.7+/-0.55 | 58.9+/-0.36 |

### STL-10-LT, ACR reported test accuracy

Source: [Wei and Gan, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Towards_Realistic_Long-Tailed_Semi-Supervised_Learning_Consistency_Is_All_You_Need_CVPR_2023_paper.html), Table 2. Setting: STL-10-LT with original unlabeled pool; gamma_u is N/A. Metric: top-1 test accuracy (%), not bACC/GM; uncertainty is SD over three independent runs.

Use in TANGS paper: usable as supplementary reported results for ACR/DASO/DARP/CReST under the ACR protocol. It supports related-work and sanity-check discussion, but should be separated from the LCGC bACC/GM table because the metric and protocol are different.

| Method | gamma_l=10, N1=150, M=100k | gamma_l=10, N1=450, M=100k | gamma_l=20, N1=150, M=100k | gamma_l=20, N1=450, M=100k |
|---|---:|---:|---:|---:|
| FixMatch | 56.1+/-2.32 | 72.4+/-0.71 | 47.6+/-4.87 | 64.0+/-2.27 |
| DARP | 66.9+/-1.66 | 75.6+/-0.45 | 59.9+/-2.17 | 72.3+/-0.60 |
| CReST | 61.7+/-2.51 | 71.6+/-1.17 | 57.1+/-3.67 | 68.6+/-0.88 |
| CReST+ | 61.2+/-1.27 | 71.5+/-0.96 | 56.0+/-3.19 | 68.5+/-1.88 |
| DASO | 70.0+/-1.19 | 78.4+/-0.80 | 65.7+/-1.78 | 75.3+/-0.44 |
| ACR | 77.1+/-0.24 | 83.0+/-0.32 | 75.1+/-0.70 | 81.5+/-0.25 |

### SimPro reported top-1 accuracy

Source: SimPro 2024, Table 1 and Table 2. Metric: top-1 accuracy (%), not bACC/GM. Network/protocol follows the SimPro paper, so use as reported contextual comparison.

#### CIFAR-10-LT, N1=500, M1=4000

Columns are distribution scenarios. Each scenario reports two settings: gamma_l=150 and gamma_l=100.

| Method | Consistent 150/150 | Consistent 100/100 | Uniform 150/1 | Uniform 100/1 | Reversed 150/1/150 | Reversed 100/1/100 | Middle 150/150 | Middle 100/100 | Head-tail 150/150 | Head-tail 100/100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FixMatch | 62.9+/-0.36 | 67.8+/-1.13 | 67.6+/-2.56 | 73.0+/-3.81 | 59.9+/-0.82 | 62.5+/-0.94 | 64.3+/-0.63 | 71.7+/-0.46 | 58.3+/-1.46 | 66.6+/-0.87 |
| CReST+ | 67.5+/-0.45 | 76.3+/-0.86 | 74.9+/-0.80 | 82.2+/-1.53 | 62.0+/-1.18 | 62.9+/-1.39 | 58.5+/-0.68 | 71.4+/-0.60 | 59.3+/-0.72 | 67.2+/-0.48 |
| DASO | 70.1+/-1.81 | 76.0+/-0.37 | 83.1+/-0.47 | 86.6+/-0.84 | 64.0+/-0.11 | 71.0+/-0.95 | 69.0+/-0.31 | 73.1+/-0.68 | 70.5+/-0.59 | 71.1+/-0.32 |
| ACR reproduced without anchors | 70.9+/-0.37 | 76.1+/-0.42 | 91.9+/-0.02 | 92.5+/-0.19 | 83.2+/-0.39 | 85.2+/-0.12 | 73.8+/-0.83 | 79.3+/-0.30 | 77.6+/-0.20 | 79.3+/-0.48 |
| SimPro | 74.2+/-0.90 | 80.7+/-0.30 | 93.6+/-0.08 | 93.8+/-0.10 | 83.5+/-0.95 | 85.8+/-0.48 | 82.6+/-0.38 | 84.8+/-0.54 | 81.0+/-0.27 | 83.0+/-0.36 |

#### CIFAR-100-LT, gamma_l=20, N1=50, M1=400

| Method | Consistent gamma_u=20 | Uniform gamma_u=1 | Reversed gamma_u=1/20 | Middle gamma_u=20 | Head-tail gamma_u=20 |
|---|---:|---:|---:|---:|---:|
| FixMatch | 40.0+/-0.96 | 39.6+/-1.16 | 36.2+/-0.63 | 39.7+/-0.61 | 38.2+/-0.82 |
| CReST+ | 40.1+/-1.28 | 37.6+/-0.88 | 32.4+/-0.08 | 36.9+/-0.57 | 35.1+/-1.10 |
| DASO | 43.0+/-0.15 | 49.4+/-0.93 | 44.1+/-0.25 | 43.1+/-1.20 | 43.8+/-0.43 |
| ACR reproduced without anchors | 40.7+/-0.57 | 50.2+/-0.82 | 44.1+/-0.14 | 42.4+/-0.47 | 41.1+/-0.09 |
| SimPro | 43.1+/-0.40 | 52.2+/-0.16 | 45.5+/-0.34 | 43.6+/-0.35 | 44.8+/-0.56 |

#### STL-10-LT, gamma_u=N/A, N1=450, M=100k

Source: SimPro 2024, Table 2. Metric: top-1 accuracy (%), not bACC/GM. Use as supplementary STL-10-LT reported context only.

| Method | gamma_l=10 | gamma_l=20 |
|---|---:|---:|
| FixMatch | 72.4+/-0.71 | 64.0+/-2.27 |
| CReST+ | 71.5+/-0.96 | 68.5+/-1.88 |
| DASO | 78.4+/-0.80 | 75.3+/-0.44 |
| ACR reproduced without anchors | 83.0+/-0.32 | 81.5+/-0.25 |
| SimPro | 84.5+/-0.39 | 82.5+/-0.25 |

### CPE reported top-1 accuracy

Source: CPE 2024, Table 2 and Table 3. Metric: top-1 test accuracy (%), not bACC/GM. Useful for CIFAR-10-LT and STL-10-LT supplementary reported context, and for CPE/DASO/CoSSL-style comparisons.

#### Consistent class distributions

| Dataset / setting | FixMatch | DARP | CReST | CReST+ | ABC | DASO | CoSSL | ACR | CPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10-LT N1=1500 M1=3000 gamma=100 | 76.49+/-0.72 | 77.37+/-0.50 | 79.90+/-0.33 | 79.60+/-0.06 | 84.01+/-0.15 | 78.87+/-0.80 | 82.35+/-0.79 | 84.18+/-0.52 | 84.44+/-0.29 |
| CIFAR-10-LT N1=1500 M1=3000 gamma=150 | 72.15+/-0.94 | 74.02+/-0.06 | 74.70+/-0.53 | 75.39+/-0.42 | 80.94+/-0.85 | 74.92+/-0.36 | 79.00+/-0.41 | 81.81+/-0.49 | 82.25+/-0.34 |
| CIFAR-10-LT N1=500 M1=4000 gamma=100 | 73.14+/-1.03 | 71.12+/-0.82 | 77.69+/-0.71 | 78.70+/-0.40 | 79.40+/-0.88 | 73.63+/-0.46 | 75.82+/-0.61 | 81.01+/-0.42 | 80.68+/-0.96 |
| CIFAR-10-LT N1=500 M1=4000 gamma=150 | 65.68+/-0.67 | 65.63+/-0.63 | 68.20+/-0.33 | 72.73+/-2.26 | 69.50+/-1.86 | 67.13+/-1.06 | 70.56+/-0.55 | 76.72+/-1.13 | 76.77+/-0.53 |
| CIFAR-100-LT N1=150 M1=300 gamma=10 | 57.56+/-0.47 | 56.14+/-0.46 | 58.56+/-0.34 | 58.19+/-0.37 | 58.25+/-0.20 | 58.16+/-0.21 | 58.00+/-0.39 | 59.83+/-0.07 | 59.83+/-0.16 |
| CIFAR-100-LT N1=150 M1=300 gamma=15 | 53.97+/-0.17 | 52.81+/-0.50 | 55.43+/-0.17 | 55.39+/-0.23 | 55.38+/-0.47 | 54.82+/-0.53 | 55.49+/-0.43 | 56.91+/-0.09 | 57.00+/-0.51 |

#### Uniform / inverse class distributions

| Dataset / setting | FixMatch | DARP | CReST | CReST+ | ABC | DASO | CoSSL | ACR | CPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10-LT gamma_l=100 gamma_u=1 | 73.27+/-1.25 | 73.52+/-1.19 | 85.14+/-0.19 | 82.68+/-0.93 | 81.45+/-0.43 | 76.01+/-0.59 | 79.90+/-0.76 | 84.61+/-0.50 | 85.86+/-0.40 |
| CIFAR-10-LT gamma_l=100 gamma_u=1/100 | 68.92+/-0.79 | 70.36+/-1.55 | 86.71+/-0.39 | 73.98+/-1.34 | 83.45+/-0.54 | 74.47+/-0.60 | 77.47+/-0.56 | 86.29+/-0.19 | 87.09+/-0.14 |
| CIFAR-100-LT gamma_l=10 gamma_u=1 | 66.47+/-0.84 | 65.80+/-0.67 | 68.20+/-0.33 | 72.73+/-2.26 | 69.50+/-1.86 | 67.54+/-0.60 | 74.31+/-0.98 | 80.10+/-1.21 | 82.32+/-0.43 |
| CIFAR-100-LT gamma_l=10 gamma_u=1/100 | 62.52+/-0.93 | 62.16+/-1.10 | 76.37+/-3.84 | 63.72+/-0.87 | 79.21+/-0.44 | 65.08+/-0.90 | 73.26+/-0.78 | 82.65+/-0.31 | 83.88+/-0.18 |
| CIFAR-100-LT gamma_l=20 gamma_u=1/10 | 57.56+/-0.47 | 56.40+/-0.21 | 60.07+/-0.24 | 59.53+/-0.34 | 59.24+/-0.17 | 59.25+/-0.23 | 57.77+/-0.31 | 60.92+/-0.40 | 60.83+/-0.30 |

#### STL-10-LT, gamma_u=N/A, N1=150, M approx. 100k

Source: CPE 2024, Table 3. Metric: top-1 accuracy (%), not bACC/GM. Use as supplementary reported context; keep separate from LCGC bACC/GM.

| Method | gamma_l=10 | gamma_l=20 |
|---|---:|---:|
| FixMatch | 66.56+/-1.02 | 56.29+/-4.00 |
| DARP | 63.74+/-0.54 | 56.03+/-1.81 |
| CReST | 65.52+/-1.01 | 61.38+/-1.75 |
| CReST+ | 66.27+/-0.59 | 62.63+/-1.69 |
| ABC | 70.64+/-0.89 | 65.68+/-3.06 |
| DASO | 69.31+/-0.91 | 62.45+/-2.23 |
| CoSSL | 71.44+/-0.45 | 69.01+/-0.80 |
| SAW | 69.30+/-0.69 | 65.80+/-1.22 |
| Adsh | 69.35+/-1.12 | 64.82+/-1.41 |
| DePL | 69.46+/-0.62 | 65.93+/-1.22 |
| RDA | 72.63+/-0.26 | 69.24+/-1.69 |
| ACR | 73.40+/-0.73 | 67.51+/-1.63 |
| CPE | 73.07+/-0.47 | 69.60+/-0.20 |

### ADELLO/FlexDA reported balanced accuracy and calibration

Source: Sanchez Aimar et al., 2024, Flexible Distribution Alignment: Towards Long-tailed Semi-supervised Learning with Proper Calibration, Table 3. Metric: paper reports test balanced accuracy averaged over final epochs, although the table caption says test accuracy. Use as supplementary LTSSL context; keep separate from LCGC bACC/GM and from local TANGS unified metrics.

#### STL-10-LT, gamma_u=N/A, N1=150, low-label regime

| Method | gamma_l=10 | gamma_l=20 |
|---|---:|---:|
| FixMatch | 64.1+/-2.3 | 54.5+/-4.3 |
| DARP | 62.1+/-1.4 | 54.7+/-2.6 |
| CReST+ | 66.9+/-1.0 | 62.6+/-2.6 |
| ABC | 71.2+/-1.0 | 65.7+/-2.3 |
| DASO | 70.0+/-1.2 | 65.7+/-1.8 |
| DebiasPL | 70.1+/-0.8 | 66.6+/-2.1 |
| CoSSL | 70.6+/-0.5 | 66.0+/-1.4 |
| UDAL | 69.8+/-1.1 | 65.0+/-2.3 |
| ADELLO | 75.7+/-0.7 | 74.6+/-0.4 |
| SoftMatch | 72.6+/-0.3 | 70.6+/-0.4 |

Additional useful evidence from the same paper:

- Table 2 reports CIFAR10-LT label-shift results and CIFAR100-LT gamma_l=50 label-shift results for FixMatch, DARP, CReST+, ABC, DASO, DebiasPL, CoSSL, UDAL, ADELLO, and SoftMatch. This is useful context but does not match the TANGS core CIFAR-100-LT gamma=100 setting.
- Appendix B reports training time on CIFAR100-LT50 using one V100-32GB: FixMatch 5h15m, ADELLO 5h18m, ABC 5h21m, CReST+ 6h22m, CoSSL 7h29m, DARP 7h43m, DASO 19h32m. Use only as qualitative overhead context because hardware/protocol differ from TANGS runs.
- Tables 11 and 12 report ECE/MCE calibration on CIFAR10-LT, STL10-LT20, and CIFAR100-LT. For STL10-LT20, ADELLO reports ECE 6.9+/-0.3 and MCE 25.9+/-1.0, much lower than FixMatch's ECE 37.8+/-4.5 and MCE 55.1+/-4.9. Useful for related work on pseudo-label confidence/calibration, not for TANGS's primary accuracy table.
- It discusses biased pseudo-label distributions and low-confidence pseudo-label usage, but it does not report direct pseudo-label F1, pseudo-label precision/recall, tail pseudo-label recall, or minority pseudo-label precision. Use DASO/CPE for those more direct pseudo-label-quality evidence types.

### CPG paper suitability note

Source: CPG 2025. This paper is useful for recent ReaLTSSL context and includes FixMatch, FreeMatch, SoftMatch, ACR, SimPro, CDMAD, and CPG on CIFAR-10-LT and CIFAR-100-LT.

Do not copy CPG values into the main reported table without visual table verification: the text extraction of its main tables is noisy. Also, its CIFAR-10-LT uses Nmax=400, Mmax=4600 and gamma in {100,150,200}; its CIFAR-100-LT uses Nmax=50, Mmax=450 and gamma in {10,15,20}. This is a realistic-arbitrary-distribution protocol rather than the exact TANGS protocol.

### DASO reported top-1 accuracy

Source: Oh et al., 2022, DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning, Table 1. Metric: top-1 accuracy (%). Useful for reported FixMatch/DARP/CReST+/ABC/DASO context and pseudo-label bias motivation. It does not include CIFAR-100-LT gamma=100.

#### CIFAR-100-LT, consistent distribution

Each cell is N1=50 M1=400 / N1=150 M1=300.

| Method | gamma=10 | gamma=20 |
|---|---:|---:|
| FixMatch | 45.2+/-0.55 / 56.5+/-0.06 | 40.0+/-0.96 / 50.7+/-0.25 |
| FixMatch+DARP | 49.4+/-0.20 / 58.1+/-0.44 | 43.4+/-0.87 / 52.2+/-0.66 |
| FixMatch+CReST+ | 44.5+/-0.94 / 57.4+/-0.18 | 40.1+/-1.28 / 52.1+/-0.21 |
| FixMatch+DASO | 49.8+/-0.24 / 59.2+/-0.35 | 43.6+/-0.09 / 52.9+/-0.42 |
| FixMatch+LA | 47.3+/-0.42 / 58.6+/-0.36 | 41.4+/-0.93 / 53.4+/-0.32 |
| FixMatch+LA+DARP | 50.5+/-0.78 / 59.9+/-0.32 | 44.4+/-0.65 / 53.8+/-0.43 |
| FixMatch+LA+CReST+ | 44.0+/-0.21 / 57.1+/-0.55 | 40.6+/-0.55 / 52.3+/-0.20 |
| FixMatch+LA+DASO | 50.7+/-0.51 / 60.6+/-0.71 | 44.1+/-0.61 / 55.1+/-0.72 |
| FixMatch+ABC | 47.5+/-0.18 / 59.1+/-0.21 | 41.6+/-0.83 / 53.7+/-0.55 |
| FixMatch+ABC+DASO | 50.2+/-0.62 / 60.0+/-0.32 | 44.5+/-0.25 / 55.3+/-0.53 |

Notes: DASO Appendix figures are useful for mechanism motivation: pseudo-label bias, minority-class recall/precision, and head/minority training curves on CIFAR-10/100-LT. Use qualitatively unless exact figure values are manually digitized.

#### STL-10-LT, DASO with LA/ABC, top-1 accuracy

Source: DASO 2022 Appendix Table 12. Setting: STL10-LT, M=100k, gamma_u=N/A. Metric: top-1 accuracy (%), not bACC/GM. Use only as supplementary context for label-rebalancing compatibility.

| Method | gamma_l=10, N1=150 | gamma_l=10, N1=450 | gamma_l=20, N1=150 | gamma_l=20, N1=450 |
|---|---:|---:|---:|---:|
| FixMatch | 56.1+/-2.32 | 72.4+/-0.71 | 47.6+/-4.87 | 64.0+/-2.27 |
| FixMatch+DASO | 70.0+/-1.19 | 78.4+/-0.80 | 65.7+/-1.78 | 75.3+/-0.44 |
| FixMatch+LA | 64.4+/-1.35 | 75.9+/-1.25 | 51.5+/-3.23 | 67.4+/-1.04 |
| FixMatch+LA+DASO | 71.7+/-1.09 | 79.0+/-0.58 | 65.6+/-1.43 | 75.8+/-0.81 |
| FixMatch+ABC | 66.3+/-1.00 | 77.1+/-0.56 | 59.3+/-2.66 | 73.0+/-0.91 |
| FixMatch+ABC+DASO | 69.6+/-0.94 | 77.9+/-0.89 | 64.5+/-2.81 | 74.7+/-0.16 |

### BaCon reported balanced accuracy

Source: Feng et al., 2024, BaCon: Boosting Imbalanced Semi-supervised Learning via Balanced Feature-Level Contrastive Learning, Table 1 and Table 2. Metric: balanced accuracy (%). Useful as a feature/representation-level CISSL reported baseline and as a contrast to TANGS's classifier-gradient-level intervention.

#### Main bACC setting

| Base / Method | CIFAR-10-LT gamma=100 beta=20% | CIFAR-100-LT gamma=20 beta=40% | STL-10-LT gamma_l=10 |
|---|---:|---:|---:|
| FixMatch | 75.30+/-0.37 | 53.94+/-0.09 | 67.16+/-0.36 |
| FixMatch+DASO | 74.78+/-0.21 | 54.83+/-0.19 | 68.69+/-0.15 |
| FixMatch+DebiasPL | 75.25+/-0.21 | 54.90+/-0.11 | 65.96+/-0.23 |
| FixMatch+CReST | 76.62+/-0.12 | 54.83+/-0.10 | 66.45+/-0.09 |
| FixMatch+CReST+PDA | 78.64+/-0.40 | 55.01+/-0.12 | 67.17+/-0.04 |
| FixMatch+DARP | 78.00+/-0.33 | 55.63+/-0.07 | 62.43+/-0.10 |
| FixMatch+ABC | 83.25+/-0.77 | 56.91+/-0.02 | 71.23+/-0.04 |
| FixMatch+CoSSL | 84.09+/-0.16 | 57.33+/-0.05 | 70.95+/-0.17 |
| FixMatch+BaCon | 84.46+/-0.15 | 57.96+/-0.26 | 71.55+/-0.09 |
| ReMixMatch | 77.96+/-0.24 | 56.12+/-0.12 | 66.97+/-0.04 |
| ReMixMatch+DASO | 78.86+/-0.15 | 57.67+/-0.20 | 65.38+/-0.18 |
| ReMixMatch+CReST+PDA | 79.91+/-0.20 | 59.78+/-0.23 | 67.57+/-0.11 |
| ReMixMatch+DARP | 77.80+/-0.18 | 57.21+/-0.21 | 65.93+/-0.16 |
| ReMixMatch+ABC | 84.49+/-0.24 | 59.92+/-0.01 | 67.24+/-1.02 |
| ReMixMatch+CoSSL | 84.93+/-0.02 | 60.46+/-0.15 | 68.73+/-0.77 |
| ReMixMatch+BaCon | 85.05+/-0.09 | 60.15+/-0.05 | 69.26+/-0.83 |

#### CIFAR-10-LT imbalance robustness

Source: BaCon Table 2. Metric: balanced accuracy (%). This is CIFAR-10-LT only.

| Method | gamma_L=100 gamma_U=100 | gamma_L=100 gamma_U=1/100 | gamma_L=150 gamma_U=150 | gamma_L=150 gamma_U=1/150 |
|---|---:|---:|---:|---:|
| FixMatch | 75.66 | 56.35 | 73.45 | 62.30 |
| CReST+ | 79.14 | 66.47 | 74.51 | 62.75 |
| ABC | 82.48 | 81.14 | 79.41 | 78.84 |
| CoSSL | 83.94 | 71.99 | 81.83 | 74.14 |
| BaCon | 84.61 | 83.80 | 81.99 | 82.35 |

## Current Related Work Without Imported Numbers

These papers are required in the novelty/related-work audit, but their numerical tables have not been imported because their protocols must first be mapped to the locked TANGS protocols.

| Work | Venue/year | Why it matters to TANGS | Numerical status |
|---|---|---|---|
| [Meta-Expert](https://proceedings.mlr.press/v267/hou25d.html) | ICML 2025 | dynamic expert assignment and multi-depth feature fusion for LTSSL mismatch | no values copied |
| [CPG](https://papers.nips.cc/paper_files/paper/2025/hash/abcd225747ec4a176a5ff59e56e0d2eb-Abstract-Conference.html) | NeurIPS 2025 | controllable pseudo-label generation under arbitrary unlabeled distributions | no values copied |
| [Learning Dynamics / DyTrim](https://iclr.cc/virtual/2026/poster/10008364) | ICLR 2026 | direct learning-dynamics analysis of LTSSL bias and a pruning method | no values copied |
| [SCAD](https://iclr.cc/virtual/2026/poster/10008701) | ICLR 2026 | super-class-aware dynamic logit adjustment and local imbalance | no values copied |
| [CoLA](https://iclr.cc/virtual/2026/poster/10007342) | ICLR 2026 | co-calibrated logit adjustment and distribution estimation | no values copied |
| [Gradient Vaccine / GradVac](https://iclr.cc/virtual/2021/poster/2550) | ICLR 2021 | gradient-similarity targeting with EMA statistics; establishes the naming collision that retired “GradVax” | no LTSSL values |

### Corrected Method Characterizations

- **LCGC is not a reweighting-only method.** It computes conflicting gradients and adds a projection-like component to encourage its intended biased direction before baseline-image test-time debiasing. TANGS differs in gradient definitions, asymmetry, exact contribution replacement, direction, and post-projection norm cap.
- **SimPro is not a prototype method.** It extends an EM interpretation, separates conditional and marginal distribution modeling, estimates the marginal distribution, and trains a Bayes classifier.
- **TANGS novelty is combination-level only.** Projection, EMA statistics, and norm clipping are individually established techniques.

## Not Directly Reusable For The TANGS Main Table

- FixMatch original paper: standard balanced/semi-supervised benchmarks, not LTSSL. Useful for method description and hyperparameter background only.
- Adaptive Confidence Margin paper: facial expression recognition datasets (RAF-DB, SFEW, AffectNet), not CIFAR-LT LTSSL. Useful only as related work on adaptive confidence thresholds, not as a baseline table for TANGS.
- ACR paper: useful for reported LTSSL background, but its CIFAR-100 settings are gamma 10/20, not the current TANGS core CIFAR-100 gamma=100 setting. Its metric is test accuracy, not Head/Medium/Tail/Balanced.
- SimPro/CPE/CPG papers: useful as reported top-1 accuracy context, but their protocols and metrics differ from the TANGS primary table. Keep them separate from local TANGS results unless the paper explicitly matches the same split, gamma, metric, and training protocol.
- Smith et al., 2021, Memory-Efficient Semi-Supervised Continual Learning: semi-supervised continual learning on CIFAR-100, not LTSSL main-table evidence.
- Hu et al., 2021, SimPLE: standard SSL on CIFAR-10/SVHN/CIFAR-100/Mini-ImageNet, not long-tailed CIFAR-LT. Useful only as generic SSL related work if needed.
- DASO and BaCon: useful as reported CISSL/LTSSL context and mechanism motivation, but their CIFAR-100 settings do not match the current TANGS core CIFAR-100 gamma=100 primary table.
- GLMC 2023: supervised long-tailed visual recognition on CIFAR-10-LT, CIFAR-100-LT, and ImageNet-LT, not semi-supervised LTSSL. It has CIFAR-LT top-1 and ImageNet-LT Many/Medium/Few results, but no STL-10-LT and no FixMatch/DARP/ABC/SimPro-style SSL comparison. Use only as optional related work for supervised long-tailed representation learning, not as TANGS reported baseline.
- ADELLO/FlexDA 2024: useful LTSSL reported context, calibration evidence, and overhead comparison. Its CIFAR100-LT main label-shift setting uses gamma_l=50 rather than TANGS's core gamma=100, so use the CIFAR100 numbers as context only. Its STL10-LT20 results are directly useful as supplementary reported context.

## Main-Table Baseline Coverage

| TANGS setting | Main-table baseline status | Notes |
|---|---|---|
| P-C10-100 | cited baselines complete; local TANGS pending | CDMAD Table 1 supplies FixMatch-family bACC/GM; matching LCGC rows may be added. |
| P-C100-100 | cited baselines complete; local TANGS pending | CDMAD Table 4 supplies the primary FixMatch-family bACC table; LCGC Table 4 extends it. |
| P-STL10-10 | Reported context / optional local | LCGC bACC/GM and ACR/DASO/SimPro/CPE/ADELLO results exist under differing \(N_1\), metric, and uncertainty conventions. |
| P-STL10-20 | cited baselines complete; local TANGS pending | CDMAD Table 2 supplies FixMatch-family bACC/GM for \(N_1=450\). |

The baseline side of the CDMAD-style main table is complete for all three required settings. The remaining main-table evidence is the three local TANGS rows plus the registered mechanism controls and ablations.

## Remaining Local Evidence

Controlling v4.6 requirement: first complete the five-run paired C100
development screen, then—only on PASS—the local 250k C100 FixMatch observer /
`tangs-v46` pair. The legacy v4.5 minimum matrix below is archived and does not
authorize resumed or new runs. In particular, the interrupted STL v4.5 cell is
not to be resumed for current evidence.

Yes for the established baseline rows. They intentionally replace local FixMatch/DARP/ABC/CDMAD reruns in C1.

The paired 20k DEV-C10-100 spending gate completed on 2026-08-13 with status PASS. It is local development evidence, not an imported paper result and not a main-table row: TANGS versus the paired local FixMatch changed bACC by +1.29 pp and tail accuracy by +13.33 pp, while Overall and Head changed by -5.91 pp and -10.32 pp. The result only authorizes the registered 50k development freeze. That freeze adds one local FixMatch tuning reference and blocks confirmatory execution unless a \(\tau\) candidate satisfies the predeclared balance gate; the three local head-downweight tuning controls run only after that TANGS screen passes. Historical main-table baselines remain cited from their papers, and formal C1 execution remains TANGS-only.

Still required locally:

- U0 exact-contribution, no-op, geometry, runtime-safety, and artifact-trace tests.
- `manualSeed=0` TANGS runs on P-C10-100, P-C100-100, and P-STL10-20.
- A1/A2/A3 occurrence, true-class decomposition, and anchor-quality diagnostics.
- B1 oracle-target-only control with the ordinary accepted set and predicted-head membership held fixed.
- Head downweighting, matched head-only clipping, and PCGrad-adapted comparisons on P-C100-100.
- Component ablations: no norm cap, no projection, and instantaneous anchor. Under v4.5, the PCGrad-adapted and no-norm-cap definitions are identical, so the single `C2-PCG` artifact is reused for D1 rather than counted as a second local run.
- Internal TANGS runtime and memory fractions under the locked timing procedure.
- H/M/T, bACC, GM, worst-class, raw seed-0 values, and direct differences from cited baselines.
- ReMixMatch transfer requires the optional local C4 pair.

## Numerical Use Checklist

Before copying any row from this file into a paper:

- [ ] source URL resolves;
- [ ] exact table and row were checked in the primary PDF;
- [ ] \(N_1,M_1,\gamma_l,\gamma_u\) match the prose;
- [ ] metric is labeled as bACC, GM, or top-1;
- [ ] uncertainty is labeled SD, SE, CI, or unresolved;
- [ ] cited rows carry source markers and the proposed row is labeled `TANGS (ours)`;
- [ ] no reported result is described as locally reproduced;
- [ ] every reported-baseline difference uses a verified source row.
