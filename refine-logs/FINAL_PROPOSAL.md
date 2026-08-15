# TANGS v4.7: Tail-ANchor-Gated Scores for Long-Tailed Semi-Supervised Learning

*Specification version: 4.7 — score-calibration pivot after the v4.6 STOP gate*
*Status: v4.6 gradient surgery is rejected. Retrospective rescoring of the old v4.6 observer checkpoint is PROVISIONAL candidate evidence only, not a v4.7 PASS. The complete local suite passes 58/58 and the local five-step CUDA integration smoke is DONE. A fresh target-server v4.7 C100 50k development run is required before any 250k run. After that true gate passes, the required single-GPU matrix is C100, C10, and STL10-20 at 250k steps each.*
*Legacy working name: GradVax. In v4.7, TANGS expands to Tail-ANchor-Gated Scores; v4.5/v4.6 keep their archived gradient-surgery identities.*

---

## Controlling v4.7 Pivot (2026-08-15)

This section supersedes every conflicting method, claim, and launch rule below.
The v4.5 and v4.6 sections remain immutable failure-analysis records.

### Why gradient surgery is abandoned

The five matched P-C100-100 development runs ended with a v4.6 STOP. The
measurement-only FixMatch observer obtained 36.70 bACC, whereas full v4.6
obtained 35.88 (-0.82 point) and did not improve tail accuracy. More
importantly, the observer found 1,589,275 conflicts among 1,589,275 eligible
tail rows (100%) with mean cosine -0.827. Full v4.6 modified 47,414 of 50,000
steps. This is a structural degeneracy: for cross-entropy, a correctly labeled
head example normally gives a positive non-target tail-row derivative, while a
tail self-class example gives a negative target-row derivative. Their negative
dot product is therefore usually normal class competition, not evidence of a
harmful optimization event. Tuning beta, tau, rho, support, or warm-up cannot
repair a trigger whose semantics are wrong.

The same checkpoint shows that tail pseudo-labels are already reliable:
98.08% precision, 32.59% recall, and 47.92% coverage. The failure is therefore
better explained by decision-prior bias than by unusable tail representation
or widespread tail pseudo-label noise.

### v4.7 method

TANGS v4.7 never edits a training gradient. Training is exact FixMatch plus a
measurement-only classwise tail-anchor observer. For classifier `z = W f + b`,
let `a_c` be the EMA supervised self-row gradient for tail class `c`. Because
its weight coordinates are proportional to `-(1-p_c) f`, define the tail
feature direction

\[
r_c = \operatorname{normalize}(-a_{c,W}).
\]

Given labeled prior `pi_c`, apply the frozen base logit adjustment

\[
\tilde z_c = z_c - 0.65\log \pi_c.
\]

For each example, find

\[
c^*(x)=\arg\max_{c\in T}\cos(f(x),r_c).
\]

If the winning cosine is at least 0.75, add one and only one tail correction:

\[
\tilde z_{c^*}\leftarrow \tilde z_{c^*}
+0.25[-\log\pi_{c^*}].
\]

Invalid or missing anchors receive no extra correction; the global prior term
still applies. The official method constants are therefore `(0.65, 0.25,
0.75)`. They are frozen from the disjoint 5,000-image C100 development set and
must not be changed after any confirmatory-test evaluation.

### Retrospective candidate evidence and claim ceiling

All rows below reuse one unchanged step-50k **v4.6** FixMatch observer
checkpoint and differ only in deterministic scoring. They motivated v4.7 but
do not validate the integrated v4.7 training-and-observer path:

| Scoring rule | bACC | Head | Medium | Tail | GM | Dead classes |
|---|---:|---:|---:|---:|---:|---:|
| Raw FixMatch | 36.70 | 69.39 | 34.12 | 6.67 | 5.65 | 22 |
| Uniform logit adjustment, alpha=0.85 | 39.42 | 67.58 | 39.18 | 11.52 | 16.11 | 7 |
| TANGS v4.7 | **40.04** | 67.45 | 36.94 | **15.82** | **20.84** | **4** |

Relative to raw FixMatch, v4.7 gains 3.34 bACC and 9.15 tail points while the
head loss is 1.94 points. Relative to uniform logit adjustment under the same
two-point head-loss guard, it gains 0.62 bACC, 4.30 tail points, and 4.73 GM
points, with three fewer dead classes. It loses 2.24 medium points, which must
be reported rather than hidden.

An unconstrained uniform-logit sweep reached 41.30 bACC at alpha=1.55 but
reduced Head to 63.27. Thus the supported v4.7 claim is Pareto improvement
under the preregistered head-preservation constraint, not unconditional
dominance over logit adjustment or state-of-the-art LTSSL.

The retrospective artifact is
`gradvax_experiments/results/second_try_2026-08-15/v47-development-selection.json`
(`status=PROVISIONAL`; it explicitly forbids 250k). The implementation is
`tangs/calibration.py`; the full local
suite passes 58 tests, and `results/v47-local-smoke-v3-c100-seed0` completed
five CUDA steps with zero gradient modifications and finite raw/LA/adjusted
summaries.

### Next evidence gate

First run one fresh P-C100-100 50k `tangs-v47` development job on the target
server after its CUDA smoke. Only `scripts/analyze_v47_integrated_development.py`
may issue the real development `PASS`; it verifies the integrated method,
50,000 steps, held-out 5,000-image development split, frozen constants, paired
raw/LA/v4.7 summaries, and zero gradient modifications. The efficacy rules are
v4.7 vs raw bACC >= +1.0, Tail >= +2.0, Head >= -2.0, GM >= 0, dead classes
non-increasing; and v4.7 vs uniform LA bACC >= +0.5 and Tail >= +2.0.

If that 50k gate passes, the single GPU runs three frozen 250k jobs in order:
P-C100-100, P-C10-100, and P-STL10-20. Each one trains a single exact FixMatch
model while collecting anchors; its final EMA checkpoint supplies raw FixMatch,
uniform LA (`alpha=0.85`), and TANGS v4.7 scores without separate baseline
training. C100 is first only as a stop-loss gate: if its confirmatory criteria
fail, C10/STL spending stops. If C100 passes, C10 and STL10-20 are required,
not optional. P-STL10-10 remains an optional extension after the required
three-dataset matrix.

---

## Target-Server CUDA Smoke Record (2026-08-14)

The target RTX 4090 environment passed `python -m pytest -q` (**53 passed in 4.01 s**) under Python 3.11.10, torch 2.3.1+cu121, torchvision 0.18.1+cu121, and CUDA 12.1. The v4.6 source was deployment commit `c5f74189d81a00402cd183a49cc4dc74d3ddf5b0` in `/root/rivermind-data/tangs/repo-v46-c5f7418`; cached datasets were reused from the persistent volume and no legacy STL task was restored.

- **Observer DONE:** `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-fixmatch-observer-seed0`, config `7f8c3ce7443e0162d1b5299ef454f35b10d7e80c6b9ae91c233f04b18ce14ebd`. Five steps completed; all serialized values were finite and all five observer records had zero gradient modification.
- **Full DONE:** `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-seed0`, config `98408fa20e4f3a0aa7b34ae2a30b29b768283429bcbc836d396eabf0bf2a754c`. Five steps completed with finite diagnostics, status, and summary. It remained in warm-up, so it is not efficacy evidence.
- **Controlled resume DONE:** `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-v2-seed0`, config `cc52d447219a3802b0518afceb2d32f0ec91a7907c5305718e2dbc02d99d91c2`. The process was stopped after a step-2 checkpoint and `--resume auto` completed step 5 with controller/RNG/model/optimizer/EMA state restored and strictly increasing finite JSONL steps. The event records the expected non-bit-exact DataLoader prefetch limitation.

The earlier timing-missed interruption artifact is deliberately retained at `/root/rivermind-data/tangs/repo-v46-c5f7418/gradvax_experiments/results/v46-smoke-c100-tangs-rho1-resume-seed0`, config `89dc6007160bf217dd4ec036cc23c9b19aaf99cd0503ab45fbb0e4eaf28f7040`; it completed before SIGTERM and is not used as resume evidence. Smoke results are infrastructure evidence only. The 50k queue remains unstarted and 250k remains blocked by the frozen development gate.

---

## Historical v4.6 Revision (2026-08-14)

This section supersedes every conflicting v4.5 method, tuning, scheduling, and
launch rule later in this document. The older text remains only as an auditable
record of what produced `FIRST_TRY_2026-08-14.md`.

The first matrix invalidated the old finite-cap design and exposed a C100
specificity failure:

- the registered C10 development sweep selected \(\tau=\infty\); finite
  \(\tau\in\{2,5,10\}\) reduced bACC by 12.12, 10.37, and 9.68 percentage
  points relative to the paired local FixMatch run;
- therefore the executed v4.5 “full TANGS” was exactly the no-cap/PCGrad
  operator, so the norm-aware component had no experimental identity;
- C100 finished at 36.53% bACC, 5.61% tail accuracy, 7.55% GM, and 15
  zero-accuracy classes, while the cited FixMatch context is 37.6% bACC;
- C100 eligibility was 99.23%, anchor updates occurred on 25.42% of steps,
  conflict occurred on 93.99% of eligible steps, and only two invalid anchor
  updates occurred. The failure is therefore not explained by rare surgery or
  an unavailable anchor;
- accepted samples predicted as head were overwhelmingly truly head in the
  sampled deep diagnostic. Editing their complete classifier gradient can
  destroy useful head evidence without being tail-specific.

The v4.6 method is **Classwise Tail-Row TANGS**. Let the final classifier be
\(z=Wf+b\), and let \(T\) be the frozen tail-class set. For each \(c\in T\),
compute the exact classifier-row contribution of accepted predicted-head
unlabeled examples,

\[
u_c=\nabla_{(W_c,b_c)}L_{\mathrm{head\mbox{-}u}},
\]

and the supervised self-class row contribution

\[
s_c=\nabla_{(W_c,b_c)}
\left[\frac{1}{B_l}\sum_i\mathbf 1[y_i=c]\ell_s(i)\right].
\]

The original loss denominators, masks, two strong views, and \(\lambda_u\) are
retained exactly. The implementation evaluates these final-linear-layer rows
analytically from detached logits and features, including the derivative of
the executed `-log(softmax + 1e-8)` loss, while the ordinary backward pass
continues to train the backbone.

For every class observed in the labeled batch:

\[
a_c\leftarrow\beta a_c+(1-\beta)s_c,
\qquad
m_c\leftarrow\beta m_c+(1-\beta)\|s_c\|,
\]

with direct initialization on the first observation. A single labeled example
is sufficient because anchors are no longer formed by pooling unrelated tail
classes. Defaults remain \(\beta=0.99\), warm-up 2500 steps, and FP32 geometry.

If \(\langle u_c,a_c\rangle<0\), define only the conflicting component

\[
q_c=\frac{\langle u_c,a_c\rangle}{\|a_c\|^2+\epsilon}a_c
\]

and bound the **correction**, not the complete head contribution:

\[
\lambda_c=\min\left(1,
\frac{\rho m_c}{\|q_c\|+\epsilon}\right),
\qquad
u'_c=u_c-\lambda_cq_c.
\]

The registered full method uses \(\rho=1\). If there is no conflict,
\(u'_c=u_c\). Only the rows \((W_c,b_c)\) for \(c\in T\) are replaced inside
the exact base classifier gradient; every head/medium output row and every
backbone gradient remains byte-for-byte the base update.

The five registered 50k C100 development cells are sequential on one GPU:

1. exact local FixMatch plus a measurement-only tail-row observer;
2. archived v4.5 TANGS with \(\tau=\infty\);
3. `tailrow-group` (tail-row restriction, one group anchor);
4. `tailrow-classwise` (per-class anchors, unbounded projection);
5. `tangs-v46` (per-class anchors plus \(\rho=1\) correction budget).

They use 5,000 balanced validation images—50 per class—from the CIFAR-100
training images outside every active unlabeled prefix. They do not remove
training examples and never inspect the official test set. The full method may
advance to 250k only if, versus the exact paired FixMatch run, bACC improves by
at least 1.0 point, tail accuracy by at least 2.0 points, GM does not decrease,
head accuracy drops by no more than 2.0 points, and the dead-class count does
not increase; it must also beat legacy v4.5 by 0.5 bACC point and unbounded
classwise projection by 0.25 bACC point. Failure of the final comparison means
the norm-aware claim is removed or redesigned—thresholds are not relaxed.

The old `tangs` method name remains bound to v4.5 for checkpoint auditability.
The new names are `tailrow-observer`, `tailrow-group`,
`tailrow-classwise`, `tangs-v46`, and `oracle-tangs-v46`.

---

## 0. Scope and Claim Ceiling

TANGS is a classifier-gradient intervention for long-tailed semi-supervised learning (LTSSL). In v4.6 it targets only the tail output-row components of the exact contribution from accepted pseudo-labeled samples predicted as head classes and protects each tail class with its own supervised EMA anchor and correction budget.

The proposal does not assume that:

- every large head gradient harms tail learning;
- positive gradient cosine implies first-order tail-loss suppression;
- classifier-only surgery solves representation bias;
- improvements on FixMatch automatically establish compatibility with other LTSSL methods;
- planned experiments already constitute evidence.

The maximum permitted claim is conditional:

> Under the locked protocols and completed controls, TANGS improves balanced and tail-class performance by replacing a measurable classifier-level head pseudo-label gradient contribution, while preserving the original base objective outside that replacement and adding bounded practical overhead.

This claim may be weakened or abandoned according to the go/no-go rules in Section 12.

---

## 1. Problem and Testable Hypotheses

In LTSSL, accepted pseudo-labels are often skewed toward head classes. Their classifier-gradient contribution may interact with the supervised tail signal through:

1. **Conflict**: the current head pseudo-label contribution has negative cosine with the tail anchor.
2. **Large directional drift**: after conflict removal, its norm is large relative to the tail anchor. This is a finite-step stability hypothesis, not a claim of first-order tail-loss suppression.
3. **Pseudo-label error concentration**: some predicted-head samples are actually medium/tail samples, so their accepted targets may reinforce head bias.

The confirmatory hypotheses are:

- **H1 — occurrence**: conflict or post-projection large-norm events occur non-trivially among eligible steps.
- **H2 — efficacy**: TANGS exceeds the CDMAD-reported FixMatch balanced accuracy on the core benchmark and shows consistent gains across the required settings.
- **H3 — mechanism**: replacing accepted pseudo targets with ground truth while keeping the accepted sample set and predicted-head membership fixed substantially reduces TANGS intervention burden.
- **H4 — specificity**: full TANGS is stronger than matched magnitude-only and generic projection alternatives.

Exact decision thresholds are defined in EXPERIMENT_TRACKER.md before confirmatory runs.

---

## 2. Corrected Geometric Motivation

Let \(L_t(w)\) be the supervised tail loss, \(g_t=\nabla L_t(w)\), and \(g_h\) the exact head pseudo-label contribution to the base classifier gradient. Write the remaining classifier-gradient contribution as

\[
g_r=g_{\mathrm{base}}-g_t-g_h
\]

and consider the pre-optimizer direction

\[
d_{\mathrm{base}}=g_r+g_t+g_h.
\]

The first-order tail-loss change is

\[
L_t(w^+)-L_t(w)
\approx
-\eta\langle g_t,d_{\mathrm{base}}\rangle,
\qquad
w^+=w-\eta d_{\mathrm{base}}.
\]

The marginal first-order term attributable to \(g_h\) is \(-\eta\langle g_t,g_h\rangle\). Therefore, when \(\langle g_t,g_h\rangle\ge0\), that contribution is not first-order harmful to the current tail loss. The earlier interpretation that positive cosine alone proves tail suppression is rejected.

If \(L_t\) is \(L\)-smooth, then

\[
L_t(w^+)
\le
L_t(w)
-\eta\langle g_t,d_{\mathrm{base}}\rangle
+\frac{L\eta^2}{2}\|d_{\mathrm{base}}\|^2.
\]

A large \(\|g_h\|\) can enlarge the worst-case curvature term, although cancellation with \(g_r+g_t\) means it need not increase the actual total-direction norm on every step. Thus, at finite learning rates and non-zero curvature, a large aligned or orthogonal contribution is a possible stability problem, not a theorem of harm. Moreover, TANGS projects against the EMA anchor \(a_t\), not the instantaneous \(g_t\), and it does not constrain \(g_r\). It therefore provides no monotonic tail-loss or full-update guarantee. The norm rule is a **smoothness-based motivation and empirical hypothesis** to be tested by the registered controls.

The tail anchor is based on ground-truth labels, but correctness of labels does not imply low variance or perfect generalization. Anchor reliability must therefore be measured through update coverage, age, norm, within-tail cancellation, and alignment with current per-class tail gradients.

---

## 3. Method

### 3.1 Locked Class Partition

Before training:

1. Sort classes by labeled-set frequency in descending order.
2. Break equal-frequency ties by ascending class ID.
3. Let \(q=\lfloor C/3\rfloor\).
4. Head = first \(q\) classes; tail = last \(q\) classes; medium = all remaining classes.

This gives 3/4/3 classes for ten-class datasets and 33/34/33 for CIFAR-100. Save the ordered class list and its SHA256 digest. Every diagnostic, run, and table must reuse it.

### 3.2 Exact Base-Objective Contributions

Let \(\phi\) denote the final linear classifier parameters. Weight and bias, when present, are flattened jointly into one FP32 vector for cosine, projection, and scaling.

For a labeled batch of size \(B_l\) and an unlabeled batch of size \(B_u\), use the base method's exact loss:

\[
L_{\mathrm{base}}=L_s+\lambda_u L_u+L_{\mathrm{aux}}.
\]

For the local code-derived FixMatch branches, `loss=Lx+Lu`, so \(\lambda_u=1\) and \(L_{\mathrm{aux}}=0\). The generic \(L_{\mathrm{aux}}\) term is retained only for separately audited optional base methods.

The implementation is built from the released CDMAD code in `CDMAD/`, but TANGS does not inherit CDMAD's method-specific objective changes. The local base branch must:

- disable white-image logit subtraction during training and evaluation;
- use the weak-view argmax as a hard pseudo-label;
- use a confidence threshold of 0.95;
- retain the two strong views produced by the released data pipeline;
- reserve \(\tau\) for the TANGS norm cap and rename the upstream confidence argument in the derived code.

For this code-derived FixMatch objective:

\[
L_{\mathrm{tail}}
=
\frac{1}{B_l}
\sum_i
\mathbf{1}[y_i\in T]\,
\ell_s(i),
\]

\[
L_{\mathrm{head\mbox{-}u}}
=
\frac{\lambda_u}{2B_u}
\sum_{v=1}^{2}\sum_j
m_j\,
\mathbf{1}[\hat y_j\in H]\,
\ell_u^{(v)}(j),
\]

where \(m_j\) is the unmodified 0.95 FixMatch confidence mask, \(\hat y_j\) is obtained from the weak view, and \(\ell_u^{(v)}(j)\) uses the same detached hard target on each released-code strong view. Samples below the base confidence threshold are never inserted into the TANGS head contribution. The factor \(2B_u\) is the exact denominator of the two-view local base objective; no selected-subgroup renormalization is allowed.

Compute:

\[
g_{\mathrm{base}}=\nabla_\phi L_{\mathrm{base}},\quad
g_t=\nabla_\phi L_{\mathrm{tail}},\quad
g_h=\nabla_\phi L_{\mathrm{head\mbox{-}u}}.
\]

The subgroup losses retain the original denominators and weights. They are not re-normalized by subgroup size. This makes \(g_h\) an exact additive contribution to \(g_{\mathrm{base}}\).

For another base method, TANGS may be used only after its pseudo-label contribution is defined with the same exact-objective rule. Method-specific auxiliary losses remain inside \(g_{\mathrm{base}}\) and are never reconstructed by hand.

### 3.3 Tail Anchor

The anchor \(a_t\) is updated only when at least two labeled tail samples are present:

\[
a_t=
\begin{cases}
g_t, & \text{first valid update},\\
\beta a_{t-1}+(1-\beta)g_t, & \text{subsequent valid update},\\
a_{t-1}, & \text{otherwise}.
\end{cases}
\]

Defaults:

- v4.5 fixed anchor decay \(\beta=0.99\);
- fixed minimum tail support = 2;
- fixed numerical epsilon \(=10^{-12}\);
- anchor invalid when non-finite or \(\|a_t\|\le10^{-12}\).

The first valid anchor is initialized directly from \(g_t\), avoiding zero-initialization shrinkage. During warm-up the anchor updates normally, but classifier gradients are not modified.

### 3.4 Conflict Projection and Post-Projection Norm Cap

On an eligible step after warm-up:

\[
h=
\begin{cases}
g_h-\dfrac{\langle g_h,a_t\rangle}{\|a_t\|^2+\epsilon}a_t,
& \langle g_h,a_t\rangle<0,\\
g_h, & \text{otherwise}.
\end{cases}
\]

Define the post-projection ratio

\[
r_{\mathrm{post}}=\frac{\|h\|}{\|a_t\|+\epsilon}
\]

and the non-amplifying scale

\[
\alpha=\min\left(1,\frac{\tau\|a_t\|}{\|h\|+\epsilon}\right).
\]

Then

\[
g_h^{\mathrm{mod}}=\alpha h.
\]

The final classifier gradient is an exact replacement:

\[
g_{\mathrm{new}}
=
g_{\mathrm{base}}-g_h+g_h^{\mathrm{mod}}.
\]

All other parameter gradients remain those of \(L_{\mathrm{base}}\).

Operational events:

- conflict = raw inner product \(<0\);
- domination = \(r_{\mathrm{post}}>\tau\);
- eligible = warm-up finished, valid anchor exists, and accepted predicted-head samples exist;
- intervention = conflict or domination on an eligible step.

If a condition is not met, use \(g_{\mathrm{new}}=g_{\mathrm{base}}\) exactly.

### 3.5 Optimizer-State Boundary

TANGS bounds the **current head-gradient contribution**, not the full optimizer step. Momentum buffers, adaptive preconditioning, and weight decay can change the actual parameter update.

The implementation must therefore log both:

- TANGS gradient diagnostics before the optimizer;
- effective classifier parameter-update norm and cosine after incorporating optimizer state.

No paper statement may claim that the complete optimizer step is bounded unless the method is explicitly applied to that effective update.

### 3.6 Implementation Invariants

Required invariants:

1. No-op equivalence: with projection disabled and \(\tau=\infty\), TANGS matches the base classifier gradient within relative tolerance \(10^{-6}\).
2. Replacement identity: \(g_{\mathrm{new}}=g_{\mathrm{base}}-g_h+g_h^{\mathrm{mod}}\).
3. Non-amplification: \(\|g_h^{\mathrm{mod}}\|\le\|h\|\).
4. Conflict removal: after projection, \(\langle h,a_t\rangle\ge-10^{-6}\|h\|\|a_t\|\).
5. Empty masks, invalid anchors, and warm-up steps reproduce the base update.
6. Backbone features are never detached; only final-classifier gradients are replaced.
7. The v4.5 runner rejects AMP and performs all loss/gradient geometry in FP32; AMP support would require a new audited protocol.
8. Retained graphs are released within the same training step.

---

## 4. Parameters and Fixed Design Choices

TANGS exposes three method scalars, but the resource-aware v4.5 plan tunes only \(\tau\):

| Scalar | v4.5 handling | Optional sensitivity |
|---|---|---|
| \(\beta\) | fixed at 0.99 | {0.95, 0.99, 0.999} |
| \(\tau\) | select once from {2, 5, 10, infinity} using four 50k-step DEV runs against one paired 50k FixMatch development reference; initial candidate 5.0 | the same DEV screen |
| \(E_{\mathrm{warm}}\) | fixed at 2500 steps | {0, 2500, 5000} |

Fixed design constants, reported separately rather than hidden as “not hyperparameters”:

- class partition: deterministic frequency terciles;
- gradient scope: final classifier weight and bias jointly;
- asymmetric target: accepted predicted-head pseudo-label contribution only;
- minimum tail support: 2;
- epsilon: \(10^{-12}\);
- anchor initialization: first valid \(g_t\);
- objective contribution scaling: original base denominators and loss weights.

The selected \(\tau\), fixed \(\beta\), and fixed warm-up are frozen before confirmatory runs. Beta or warm-up sensitivity requires a separate optional budget and cannot alter v4.5 confirmatory results.

The 50k freeze is balance-gated before any 250k run. Relative to the paired FixMatch development reference, a TANGS candidate is admissible only if validation bACC improves by at least 0.25 percentage points, GM drops by no more than 0.25 points, Head drops by no more than 5 points, and Overall drops by no more than 3 points. Among admissible candidates, select by validation bACC, then GM, Head retention, Overall retention, and larger \(\tau\). These rules are locked before the sweep; failure of every candidate blocks the confirmatory matrix and triggers a separately versioned method revision rather than post-hoc threshold relaxation. Operationally, the paired FixMatch and four \(\tau\) runs are screened first; the three head-downweight controls run only after this TANGS screen passes, avoiding 150k control steps when no current method candidate is viable.

DEV-C10-100 holds out a deterministic class-stratified 20% of each labeled class. Because the released CIFAR unlabeled loader contains the labeled prefix, those validation indices are removed from both the labeled and unlabeled training loaders during development; otherwise hyperparameter selection would train on validation images. Confirmatory loaders retain the unmodified released behavior.

---

## 5. Locked Benchmark Family

The canonical local protocol is a **declared derivative of the released CDMAD code**, not an independent reimplementation and not an identity claim about every paper-reported row. The vendored tree is the clean official `LeeHyuck/CDMAD` commit `7cd732b4615b9d94934a9197e69c6775496fb5ee`; it supplies dataset construction, WRN-28-2, augmentations, Adam, custom per-step decay, EMA, and the 500-iteration epoch convention. TANGS-specific code must live in a derived implementation so this upstream snapshot remains auditable.

All non-method execution details are inherited from the pinned code path unless explicitly listed below:

- exact `CDMAD/wrn.py` WRN-28-2, including LeakyReLU slope 0.1, the unused four-way rotation head, and the wrapper's executed BatchNorm defaults (`eps=1e-5`, `momentum=0.1`); do not silently “fix” the wrapper to its unused `1e-3` arguments;
- FP32 training without AMP;
- Adam with constant learning rate 0.0015, library-default `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, and no scheduler;
- labeled/unlabeled DataLoaders with `shuffle=True`, `drop_last=True`, four persistent workers, batches 32/64, and iterator restart on exhaustion; test loader batch 200, four persistent workers, `shuffle=False`; persistence prevents repeated process creation when cycling short long-tailed loaders and is locked identically across compared local methods because it can change worker RNG consumption relative to a non-persistent implementation;
- released CIFAR/STL resize, crop, flip, normalization, RandAugment(3,4), Cutout(16), and one-weak/two-strong ordering;
- `loss=Lx+Lu`, so \(\lambda_u=1\) and no auxiliary loss;
- exact `WeightEMA` initialization and state-dictionary update order, including its EMA-to-online initial copy, floating-buffer updates, EMA decay 0.999, and post-Adam factor `1-wd*lr`;
- rolling checkpoint plus 100-epoch/50,000-step snapshots.

Four deliberate departures preserve the TANGS question and experimental integrity:

1. The local base branch restores hard FixMatch pseudo-labels and a 0.95 confidence mask instead of CDMAD's soft targets, zero threshold, and white-image subtraction.
2. Confirmatory claims use the final EMA checkpoint only; the released code's per-epoch test evaluation must not be used for checkpoint selection.
3. The derived runner keeps `cudnn.deterministic=True` and `cudnn.benchmark=False` instead of the released script's later contradictory `benchmark=True` assignment.
4. DEV-C10-100 adds a held-out labeled validation split only for tuning; every confirmatory loader retains the released seed-0 construction.

Extra logs, manifests, and diagnostics are measurement-only additions and may not change the forward objective, optimizer step, EMA order, or data order.

The released CIFAR loaders define the unlabeled loader from the prefix `idxs[:N_c+M_c]`, so labeled indices are included in that loader. In v4.5, \(M_1\) denotes the released `num_max_u` argument, not the final loader cardinality. This overlap is preserved for code/paper-family comparability, measured explicitly, and disclosed. Changing to disjoint splits would create a new protocol version.

The released STL-10 loader starts from the official 100,000-image unlabeled split and appends the selected labeled images. Its actual unlabeled-loader cardinality is therefore `100000 + |L|`, with the appended region duplicating labeled examples. Preserve and disclose the append order, labeled-index hash, overlap count, and final cardinality; a pure 100,000-image loader would be a different protocol.

Required local settings:

- P-C10-100 — CIFAR-10-LT: \(N_1=1500\), \(M_1=3000\), \(\gamma_l=\gamma_u=100\).
- P-C100-100 — CIFAR-100-LT: \(N_1=150\), \(M_1=300\), \(\gamma_l=\gamma_u=100\).
- P-STL10-20 — STL-10-LT: \(N_1=450\), \(\gamma_l=20\), original 100,000-image unlabeled pool, \(\gamma_u\) unavailable.

P-STL10-10 is retained as an additional reported benchmark and optional local robustness extension. It is not required for the minimum v4.5 paper because P-STL10-20 already supplies a third dataset under the more difficult labeled imbalance.

Core mechanism setting:

- CIFAR-100-LT, \(\gamma=100\).

Every minimum-plan local run locks the released `manualSeed=0` for data construction, initialization, augmentation RNG, and loader order. This reproduces the released README/default seed path and, on CIFAR-100, avoids the nonzero-seed branch that shuffles class indices. Actual index arrays, overlap counts, and hashes are saved once and reused by TANGS and every local mechanism control.

With deterministic cuDNN retained, repeating the identical seed-0 configuration is not treated as an independent trial. Nonzero seeds are outside the minimum protocol and may appear only in a separately labeled robustness extension; on CIFAR-100 they change both training randomness and the sampled split.

Following the CDMAD comparison format, audited paper-reported baselines and the local TANGS row appear in the same main table. Superscripts and the caption identify each numerical source; the proposed row is labeled `TANGS (ours)`.

---

## 6. Experimental Narrative

### U0 — Integrity Before Performance

- U0a no-op equivalence and decomposition tests;
- U0b projection/non-amplification toy tests;
- U0c empty-mask, invalid-anchor, FP32/AMP-disabled guard, and optimizer-state tests;
- U0d exact upstream model/optimizer/loader/WeightEMA parity and one-short-run artifact trace.

The fixed development-only P0 pair is the sole exception: it may run after `U0-LOCAL-TESTS` and `U0-LOCAL-SMOKE` pass because it is a spending safeguard rather than paper evidence. No A1/A2/A3 diagnostic pilot, 50k tuning run, or 250k confirmatory experiment starts until every U0 row passes and G0 is DONE. Completing P0 does not satisfy or bypass G0.

### P0 — Minimal Idea Gate Before Expensive Runs

Before the 50k hyperparameter sweep or any 250k experiment, run a paired development-only pilot on DEV-C10-100:

- FixMatch, `manualSeed=0`, 20,000 steps;
- TANGS with `tau=5`, `manualSeed=0`, 20,000 steps;
- identical train/validation indices, initialization seed, augmentation stream policy, optimizer, and EMA evaluation;
- final held-out validation bACC and H/M/T only; no CIFAR-10 test inspection.

The automatic gate returns PASS only when the pair is valid, TANGS has at least 100 eligible geometry steps, anchor coverage over predicted-head occurrence is at least 90%, conflict-or-domination occurs on at least 10% of eligible steps, at least one nonzero gradient-modification step is recorded, validation bACC is no worse than FixMatch by more than 0.5 pp, and either bACC improves by at least 0.25 pp or tail accuracy improves by at least 0.5 pp. HOLD permits only a paired 50k extension; STOP blocks the sweep and full matrix. This local FixMatch pilot is a spending safeguard, not a paper main-table row.

**Observed P0 outcome (2026-08-13): PASS.** On the final 20k held-out development evaluation, FixMatch obtained Overall/Head/Medium/Tail/bACC = 90.07/93.00/67.01/45.00/68.20, while TANGS obtained 84.16/82.68/67.96/58.33/69.49. Thus TANGS changed bACC by +1.29 pp and tail accuracy by +13.33 pp, while Overall and Head changed by -5.91 pp and -10.32 pp. The mechanism recorded 19,927 eligible steps, 99.94% anchor coverage, 94.73% conflict-or-domination among eligible steps, and 16,568 nonzero gradient modifications. Split and partition hashes matched exactly. This result makes the registered 50k development freeze eligible after G0 is DONE, but does not execute it; it is not main-table or confirmatory evidence and does not establish an across-the-board accuracy improvement. Conflict occurred on 89.59% and post-projection domination at \(\tau=5\) on 53.76% of eligible steps, so the P0 artifact does not isolate projection versus the cap as the source of the trade-off. The v4.5 response is to complete the predeclared \(\tau\) screen, evaluating \(\tau=10\) and infinity first operationally, rather than changing the TANGS operator from this 20k result.

### Phase A — Does the Target Pattern Exist?

**A1. Core gradient diagnostics**

On the ordinary TANGS run for CIFAR-100-LT \(\gamma=100\), log the pre-intervention gradient state:

- raw cosine against current \(g_t\);
- effective cosine against EMA anchor \(a_t\);
- raw and post-projection norm ratios;
- eligible, conflict, domination, and intervention fractions;
- anchor update fraction, age, norm, and invalid fraction;
- current batch counts for tail labels and accepted predicted-head pseudo-labels.

All fractions use eligible steps as their primary denominator; total-step denominators are reported separately.

**A2. Offline true-class decomposition**

Using unlabeled ground truth for analysis only, divide accepted predicted-head samples into true-head, true-medium, and true-tail groups. Report their counts, pseudo-label error rates, cosine, norm contribution, and intervention frequency. Ground truth never affects ordinary training.

**A3. Anchor quality**

Report valid-anchor coverage, age, norm, update frequency, within-tail cancellation, and alignment with available current per-class tail gradients. G2 and G3 are evaluated from this experiment.

**A4. Exploratory temporal association**

Use validation or locked offline checkpoint metrics, not repeatedly inspected test results. Remove shared time trends and account for autocorrelation. This analysis is exploratory and cannot establish causality.

### Phase B — Mechanism Controls

**B1. Oracle-target-only control**

- generate the ordinary weak-view prediction, confidence mask, and predicted-head membership;
- keep that accepted sample set and membership fixed;
- replace only the accepted target with ground truth;
- keep denominators, loss weights, optimizer, and schedule unchanged;
- compare FixMatch-oracle-target with and without TANGS.

This isolates target noise more cleanly than converting the entire unlabeled set into labeled data. The primary mechanism comparison is the reduction in intervention burden from ordinary TANGS to oracle-target TANGS; the paired oracle performance result is supporting evidence.

**B2. Full-oracle upper bound**

Optional. Use all unlabeled ground truth and label it explicitly as a fully supervised upper-bound control, not as the primary mechanism test.

**B3. Locked training dynamics**

Save sparse checkpoints during confirmatory runs. After configurations are frozen, evaluate the same checkpoint steps for TANGS and registered local controls. Curves are descriptive; final claims use the locked final EMA checkpoint.

### Phase C — Comparative Evidence

**C1. Resource-aware cross-dataset evidence**

Required local main-table runs are FixMatch+TANGS with `manualSeed=0` on P-C10-100, P-C100-100, and P-STL10-20. Plain FixMatch is not rerun as confirmatory or main-table evidence; the paired 50k DEV-C10-100 FixMatch run exists only to freeze \(\tau\) under the balance gate. These three TANGS runs are the locally trained C1 rows.

The main comparison table reuses audited values from CDMAD-2024 Tables 1, 2, and 4 for FixMatch, DARP, DARP+cRT, CReST, CReST+LA, ABC, CoSSL, UDAL, and CDMAD where available. Later LCGC results may be added when the benchmark setting matches. This follows the established CDMAD presentation: published baseline rows are cited, while only the proposed-method row is produced by the local implementation.

All rows use the source paper's bACC/GM metric order. Reported rows retain their published SE/SD, and the TANGS row reports its raw seed-0 value. Main-text comparisons state the numerical difference from the cited baseline directly, with source markers in the table header or method label.

**C2. Direct debiasing, optimization, and gradient baselines on the core setting**

- head-loss downweighting with an equal tuning budget;
- head-only norm clipping using the same post-projection cap scale;
- PCGrad-adapted with the same head/tail gradient definitions.

These controls are locally run because they test whether TANGS's direction-aware construction adds value beyond simpler transformations of the same exact \(g_h\). They are mechanism controls, not attempts to reproduce full external LTSSL systems.

The locked PCGrad-adapted control is exactly the no-norm-cap component ablation: both project the same \(g_h\) against the same EMA tail anchor on negative conflict and apply no cap. One `C2-PCG` seed-0 artifact therefore serves both C2 and D1; it is not counted or run twice.

**C3. Optional CDMAD complementarity**

- CDMAD;
- CDMAD + TANGS.

This is outside the minimum v4.5 budget. It uses a separately audited definition of the CDMAD soft pseudo-label contribution and matched `manualSeed=0` runs, providing a clean optional complementarity result.

**C4. Base-SSL transfer**

Optional appendix:

- ReMixMatch;
- ReMixMatch + TANGS.

Matched `manualSeed=0` runs provide an optional backbone-transfer check and support the statement that TANGS transfers to ReMixMatch on the evaluated benchmark.

### Phase D — Ablations

**D1. Components**

| Variant | Conflict projection | Post-projection norm cap | EMA anchor |
|---|---|---|---|
| Full TANGS | yes | yes | yes |
| No norm cap (reuse C2-PCG) | yes | no | yes |
| No projection | no | yes | yes |
| Instantaneous anchor | yes | yes | no |

**D2. Anchor redesign contingency**

Only if A3 fails its pre-registered group-anchor quality gate, open a new method version and compare a class-balanced anchor and a per-class EMA anchor. These variants may not be substituted silently into v4.5 confirmatory runs.

**D3. Sensitivity**

Run \(\tau\) first, then one of \(\beta\) or warm-up if compute is limited. Sensitivity runs use a development protocol and are excluded from confirmatory significance claims.

### Phase E — Cost

Report:

- synchronized wall-clock training time after warm-up;
- peak allocated and reserved GPU memory;
- total training time;
- eligible-step and intervention-step overhead separately.

Timing and memory instrumentation is enabled on the existing P-C100-100 TANGS run. Measure the extra-gradient/surgery section as a fraction of synchronized training-step time and the known gradient/anchor tensor footprint as a fraction of peak allocated memory; E1 adds no full training run.

The extra classifier-gradient work is \(O(B_g C d)\) for group batch size \(B_g\), not merely \(O(Cd)\). “Lightweight” may be claimed only if E1 passes its tracker thresholds.

---

## 7. Metrics and Statistical Reporting

Primary performance metric:

- balanced accuracy = mean per-class recall/accuracy.

Secondary metrics:

- head, medium, and tail accuracy using the locked partition;
- geometric mean of per-class accuracy;
- worst-class accuracy;
- head, medium, and tail differences relative to registered local mechanism controls;
- runtime and peak memory.

On a class-balanced test set, overall accuracy equals balanced accuracy; do not present them as independent evidence.

Pseudo-label diagnostics:

\[
\mathrm{TailPrecision}
=
\frac{\#\{\text{accepted, predicted tail, correct}\}}
{\#\{\text{accepted, predicted tail}\}},
\]

\[
\mathrm{TailRecall}
=
\frac{\#\{\text{accepted, true tail, predicted correctly}\}}
{\#\{\text{true-tail unlabeled samples}\}},
\]

\[
\mathrm{TailCoverage}
=
\frac{\#\{\text{accepted true-tail samples}\}}
{\#\{\text{true-tail unlabeled samples}\}}.
\]

Report macro-per-class versions and micro group aggregates separately.

Statistics:

- every minimum-plan local row uses `manualSeed=0` exactly once;
- report the raw seed-0 value and matched method difference; do not report a local mean, SD, SE, confidence interval, or significance test from one run;
- base local conclusions on the locked CDMAD-style seed-0 protocol;
- treat an identical deterministic rerun as a reproducibility check rather than a new result cell;
- any optional nonzero-seed extension is a new, separately labeled robustness protocol and may not be pooled silently with the seed-0 table;
- never convert a paper-reported standard error into standard deviation without the reported run count and explicit conversion;
- no best-test-checkpoint selection.

---

## 8. Relationship to Prior Work

### Output- and Distribution-Level LTSSL

| Method | Correct high-level characterization |
|---|---|
| DARP | pseudo-label distribution refinement/alignment |
| ABC | auxiliary balanced classifier |
| SimPro | probabilistic EM-style separation of conditional and marginal distributions with a Bayes classifier |
| CDMAD | baseline-image logit debiasing for pseudo-label and test prediction refinement |
| Meta-Expert / CPG / SCAD / CoLA / DyTrim | recent expert assignment, controllable pseudo-labeling, semantic/logit debiasing, calibration, and learning-dynamics approaches that must be covered in the literature review |

### Gradient-Level Optimization

| Method | Relation to TANGS |
|---|---|
| PCGrad | generic pairwise projection to remove negative interference |
| CAGrad | common-descent multi-task optimization |
| GradVac | gradient-similarity targeting with EMA statistics; also the reason the legacy name GradVax is retired |
| LCGC | LTSSL gradient-conflict manipulation coupled to baseline-image debiasing; uses a projection-like gradient component but a different objective and direction |
| TANGS | exact asymmetric replacement of the accepted predicted-head classifier contribution, followed by non-amplifying post-projection norm control relative to a supervised tail anchor |

Novelty is claimed only for the combination and exact LTSSL construction, not for gradient projection, EMA, or norm clipping individually.

---

## 9. Claim Ladder

Before experiments:

- “We hypothesize...”
- “TANGS is designed to...”
- “The planned controls test whether...”

Allowed after corresponding evidence:

- A1/A3 pass: “classifier-level head/tail interference is measurable under the tested protocol.”
- C1 and statistics pass: “TANGS improves balanced/tail performance under the tested settings.”
- B1 passes: “the oracle-target-only result is consistent with pseudo-label error being one contributor.”
- C2 passes: “directional intervention adds value beyond matched magnitude-only controls.”
- optional C3 passes: “TANGS complements CDMAD on the tested core setting.”
- E1 passes: “TANGS adds no more than the pre-registered runtime/memory overhead.”

Never claim:

- universal LTSSL correction;
- formal convergence;
- broad causality from the oracle experiment alone;
- plug-and-play compatibility beyond methods actually tested;
- “consistent improvement” while any required tracker row remains TODO;
- “small overhead” before E1 is complete.

---

## 10. Anticipated Failure Modes

- interference events are too rare to motivate intervention;
- the group tail anchor has strong within-tail cancellation;
- the anchor is stale or invalid on many steps;
- TANGS improves tail accuracy only by unacceptable head degradation;
- momentum or adaptive optimizer state overwhelms the current-gradient correction;
- the seed-0 gain is absent or reverses under an optional robustness protocol;
- oracle-target-only does not reduce the gain, contradicting the pseudo-label-noise story;
- output-level methods already remove most eligible interference;
- classifier-only surgery is insufficient when representation bias dominates.

Negative outcomes must be recorded without reinterpretation.

---

## 11. Reproducibility and Artifact Contract

Every run must record:

- run ID and experiment ID;
- method and base method;
- protocol ID and split-file SHA256;
- class-partition SHA256;
- config path and config SHA256;
- git commit;
- `manualSeed=0` recorded as both data and training seed for the minimum protocol;
- hardware and software environment;
- start/end timestamps;
- raw log, checkpoint, metrics JSON, and diagnostic artifact paths;
- status and failure reason.

Every locally measured paper number must resolve to a DONE tracker row and raw metrics artifact. Every literature number must resolve to a verified registry row, citation, table/row locator, metric, and uncertainty type; table annotations keep reported and local provenance explicit.

---

## 12. Go/No-Go Rules

The v4.6 development gate above is now the first controlling rule. The v4.5
rules below are historical and cannot authorize new runs. In particular,
`scripts/run_required.sh` is blocked by default; use
`scripts/run_v46_development.sh`, and only after a machine-readable PASS use
`scripts/run_required_v46.sh` for the paired C100 confirmatory baseline/method
run. The interrupted v4.5 STL job must not be resumed as current evidence.

1. **Integrity stop**: any U0 invariant fails.
2. **Spending stop**: P0 returns STOP; do not launch the 50k sweep or 250k matrix. After G0 is DONE, a HOLD permits only the paired 50k extension and a PASS permits the registered 50k freeze.
3. **Balance-freeze stop**: the 50k development artifacts are incomplete, mismatched, or no \(\tau\) candidate satisfies the locked bACC/GM/Head/Overall gate; do not launch any 250k run.
4. **Mechanism stop**: A1 or A3 fails its pre-registered occurrence/anchor-quality gate.
5. **Efficacy stop**: the core TANGS result fails the reported-FixMatch balanced-accuracy gate.
6. **Generalization stop**: TANGS fails to exceed reported FixMatch on at least two of the three required settings.
7. **Specificity stop**: matched clipping/downweighting explains the full effect.
8. **Novelty stop**: the local mechanism evidence and benchmark comparison with CDMAD, LCGC, and current LTSSL work do not support a credible distinction.
9. **Claim reduction**: B1, optional C3, C4, or E1 fail; remove only the corresponding mechanism, complementarity, transfer, or lightweight claim.

The project proceeds to a CCF-C submission after U0, A1, A2, A3, B1, the three required C1 TANGS rows, C2 mechanism controls, D1, and E1 are complete. Audited FixMatch/ABC/CDMAD/LCGC results populate the citation-marked main comparison table without local reruns. “CCF-C” is a venue tier rather than a venue specification; the exact conference, page limit, anonymity rules, and submission checklist are recorded separately before paper drafting.
