# GradVax: Norm-Aware Gradient Projection for Tail-Class Preservation in Long-Tailed Semi-Supervised Learning

*Version: CCF-C Revision V3.1 - revised for agent execution and paper drafting*

---

## 1. Problem Statement

In Long-Tailed Semi-Supervised Learning (LTSSL), pseudo-labeled head-class samples can suppress tail-class learning through two distinct mechanisms:

1. **Gradient Conflict**: Head pseudo-label gradients are anti-aligned (negative cosine) with tail supervised gradients, directly undoing tail learning.
2. **Gradient Domination**: Even when head and tail gradients are positively aligned (positive cosine), the head gradient norm can be much larger, effectively drowning out the tail signal and steering shared parameters toward head-optimal solutions.

Few LTSSL methods directly modify gradients; most operate at the output level through thresholding, reweighting, or distribution alignment. LCGC studies gradient conflict in LTSSL, but does not combine asymmetric projection with norm-aware dampening anchored on a stabilized tail supervised gradient signal.

**Core claim**: GradVax is a lightweight, computationally efficient optimizer-level intervention that mitigates one specific, measurable source of tail suppression: classifier-level interference from pseudo-labeled head samples. It is intended to be complementary to existing output-level LTSSL methods. We deliberately make a small, verifiable claim rather than a broad theory of the entire LTSSL bias loop.

---

## 2. Theoretical Motivation

### 2.1 Why Projection Alone Is Insufficient (Domination Analysis)

The following argument is intended as a **geometric motivation** for the method, not as a formal convergence theorem.

**Proposition 1 (Domination under positive cosine).** Consider a shared linear classifier \( w \in \mathbb{R}^d \) updated by gradient descent with learning rate \( \eta \). Let \( g_h \) (head pseudo-label gradient) and \( g_t \) (tail supervised gradient) satisfy \( \cos(g_h, g_t) = c \ge 0 \). Decompose the update \( \Delta w = -\eta(g_h + g_t) \) into tail-aligned and tail-orthogonal components:

```text
Delta w = -eta * [(||g_t|| + c*||g_h||) * ghat_t + sqrt(1-c^2)*||g_h|| * ghat_perp]
```

where \( \hat g_t = g_t/||g_t|| \) and \( \hat g_\perp \) is the unit vector in the \( g_h \) direction orthogonal to \( g_t \).

When \( ||g_h|| = R*||g_t|| \) with \( R \gg 1 \):
- Tail-aligned step: \( \eta*(1 + cR)*||g_t|| \approx \eta*cR*||g_t|| \) - controlled by head, not tail
- Tail-orthogonal step: \( \eta*\sqrt{1-c^2}*R*||g_t|| \) - grows linearly with \( R \), increasing drift away from the update preferred by the tail signal

The **effective tail influence ratio** (tail's own contribution / total step along tail direction) is:

```text
rho = 1 / (1 + cR) -> 0 as R -> infinity
```

Even when cosine is positive, the tail can lose control of its own update direction. This motivates norm-aware control in addition to conflict correction.

### 2.2 Why Asymmetric Intervention Is Reasonable

In LTSSL, the tail supervised gradient \( g_{sup\_tail} \) is computed from ground-truth labels and is therefore a more trustworthy estimate of the true tail loss gradient. The head pseudo-label gradient \( g_{head\_pseudo} \) is computed from model predictions and is more likely to be biased by head-favoring errors. Symmetric intervention (modifying both) risks corrupting the more reliable tail signal. Asymmetric intervention (modifying only \( g_{head\_pseudo} \)) preserves the tail anchor while controlling the noisier head signal.

### 2.3 Practical Convergence Intuition

GradVax's dampening ensures that the head pseudo-label gradient norm is bounded by \( \tau*||g_{tail\_ema}|| \). This yields a simple gradient-balance constraint: the optimizer cannot step arbitrarily farther in the head direction than in the tail direction. We do not claim formal convergence rates; instead, we test whether this constraint consistently improves tail accuracy with low overhead.

---

## 3. Method

GradVax has exactly **3 hyperparameters** beyond the base SSL method: beta (EMA momentum), tau (domination threshold), and E_warm (warm-up length).

### 3.1 Gradient Decomposition

At each training step, decompose the total gradient into:

- **g_sup_tail**: Gradient from supervised loss on tail classes.
- **g_sup_head+med**: Gradient from supervised loss on head and medium classes.
- **g_head_pseudo**: Gradient from pseudo-labeled loss on samples assigned head-class labels.
- **g_other_pseudo**: Gradient from pseudo-labeled loss on samples assigned medium/tail-class labels.

**Class partitioning**: Fixed terciles by labeled-set frequency.

**Gradient scope**: Classifier head only (final linear layer \( W \in \mathbb{R}^{C\times d} \)). We target the classifier because it is the most direct locus where class-imbalance-induced decision bias manifests, and because classifier-level intervention offers the best tradeoff between diagnostic interpretability and computational cost. In implementation, this scope restriction applies only to the gradient surgery itself: backbone features must still participate in the standard end-to-end backpropagation used by the base SSL method, and the Agent must not use `detach()` or equivalent operations that accidentally sever the backbone from the computation graph. GradVax should only alter the gradient vectors or `.grad` entries associated with the final linear classifier parameters. The recommended implementation path is to keep the backbone under the standard total-loss backward pass, separately compute the classifier-level gradient components needed by GradVax, assemble the modified classifier update as `g_total`, and then manually write it back to `classifier.weight.grad` (and `classifier.bias.grad` if present). This avoids ambiguity between full-network and classifier-only updates while preserving standard gradient flow through the backbone. In implementation, gradient extraction for `g_sup_tail` and `g_head_pseudo` should be guarded by explicit if-else control flow so that `torch.autograd.grad(...)` is called only when the corresponding mask is non-empty.

```python
# Implementation sketch (PyTorch)
optimizer.zero_grad(set_to_none=True)
loss_total.backward(retain_graph=True)  # standard backward path for backbone + classifier
# use explicit if-else guards so empty masks never produce invalid losses or autograd errors

if labeled_tail_mask.any():
    loss_sup_tail = CE(logits[labeled_tail_mask], labels[labeled_tail_mask])
    g_sup_tail = torch.autograd.grad(loss_sup_tail, classifier.parameters(), retain_graph=True)
else:
    g_sup_tail = None  # do not call autograd.grad on an empty tail mask

if pseudo_head_mask.any():
    loss_head_pseudo = CE(logits[pseudo_head_mask], pseudo_labels[pseudo_head_mask])
    g_head_pseudo = torch.autograd.grad(loss_head_pseudo, classifier.parameters(), retain_graph=False)
else:
    g_head_pseudo = None  # do not call autograd.grad on an empty pseudo-head mask

# keep the backbone in the standard computation graph; do not detach backbone features
# apply GradVax only to the classifier parameters / their gradient vectors
# assemble modified classifier update and write it back to classifier.weight.grad
# (and classifier.bias.grad if present), while leaving backbone gradients on the standard path
# guard gradient extraction with if-else control flow so empty masks never create invalid losses

optimizer.step()
# then immediately clear gradients and release graph-related references
optimizer.zero_grad(set_to_none=True)
```

### 3.2 EMA-Smoothed Tail Anchor

```text
g_tail_ema <- beta * g_tail_ema + (1 - beta) * g_sup_tail    (when n_tail_in_batch >= 2)
```

- **beta = 0.99** (default)
- Only updated when at least 2 tail samples are present in the batch
- **Warm-up**: GradVax does not modify head gradients during the first \( E_{warm}=2500 \) steps, but `g_tail_ema` should still be updated silently whenever the batch provides enough tail support so that a stable anchor is already available when warm-up ends
- If `n_tail_in_batch < 2`, keep the previous EMA anchor unchanged for that step
- If `n_tail_in_batch < 2` and no valid EMA anchor is available yet, bypass all GradVax projection and dampening rules for that step and execute the standard FixMatch parameter update
- If no valid EMA anchor is available yet, skip GradVax and use the unmodified gradient for that step

### 3.3 Asymmetric Interference Control

Given \( g_{head\_pseudo} \) and \( g_{tail\_ema} \), compute:

```text
cos_theta = <g_head_pseudo, g_tail_ema> / (||g_head_pseudo|| * ||g_tail_ema|| + epsilon)
ratio = ||g_head_pseudo|| / (||g_tail_ema|| + epsilon)
```

**Rule 1 - Conflict Projection (cos_theta < 0):**
```text
g_head_mod = g_head_pseudo - (<g_head_pseudo, g_tail_ema> / ||g_tail_ema||^2) * g_tail_ema
```

**Rule 2 - Domination Dampening (ratio > tau, applied after projection if both trigger):**
```text
g_head_mod = (tau / ratio_after_proj) * g_head_mod
```

**Otherwise:** \( g_{head\_mod} = g_{head\_pseudo} \)

- **tau = 5.0** (default)
- Only \( g_{head\_pseudo} \) is modified
- If `pseudo_head_mask` is empty, skip GradVax for that step
- If `g_tail_ema` is unavailable or numerically degenerate, skip GradVax for that step

**Final update:**
```text
g_total = g_sup_tail + g_sup_head+med + g_head_mod + g_other_pseudo
```

**Implementation note**
- Agent must gracefully handle `None` values during final gradient assembly, e.g. by treating missing components as zero tensors or by conditionally accumulating only non-`None` gradient terms, so that batches without tail or pseudo-head contributions do not trigger `TypeError` during `g_total` construction.

### 3.4 Computational Cost

| Component | Cost | Notes |
|-----------|------|-------|
| Extra gradient computation | O(C*d) per group | Classifier only |
| EMA update | O(C*d) | Single vector update |
| Cosine + norm | O(C*d) | Two dot products |
| Projection/dampening | O(C*d) | One vector operation |
| **Total overhead** | **Low** | Lightweight by design |

**Practical Logging Note**
- Diagnostic quantities such as cosine similarity, norm ratio, and intervention frequency should be logged with buffered or reduced-frequency recording rather than written to disk at every step.
- Use the same logging and evaluation schedule for baseline and GradVax when reporting runtime overhead.
- Exclude avoidable debugging overhead and one-time setup costs from the final overhead comparison.

---

## 4. Hyperparameter Robustness

### 4.1 Only 3 Hyperparameters

| Hyperparameter | Default | Range Tested | Role |
|----------------|---------|-------------|------|
| beta (EMA momentum) | 0.99 | {0.95, 0.99, 0.999} | Tail anchor smoothing |
| tau (domination threshold) | 5.0 | {2, 5, 10, infinity} | Max head/tail norm ratio |
| E_warm (warm-up steps) | 2500 | {0, 2500, 5000} | Anchor stabilization period |

### 4.2 Robustness Protocol

We use the **same defaults** across the main settings and report:
1. Performance with defaults vs. lightly tuned values
2. Sensitivity curves for beta and tau on representative settings
3. A practical robustness check: defaults should stay close to the best tested setting

### 4.3 Design Choices That Are NOT Hyperparameters

The following are fixed design decisions validated by limited ablation, not tuned per dataset:
- Class partitioning: terciles
- Gradient scope: classifier only
- Asymmetry: only head pseudo modified
- Minimum support for EMA update: at least 2 tail samples

We acknowledge these are meaningful design choices, but we evaluate them only to the extent needed to support the main claim.

---

## 5. Experimental Narrative (CCF-C Prioritized)

We keep the original narrative structure while emphasizing a compact but convincing evaluation package.

**Time-unit convention.** Because FixMatch-style LTSSL often uses effectively infinite or padded unlabeled sampling, we treat **steps / iterations** as the primary time unit for execution and plotting. Use evaluation every **500 steps** as the default reporting interval, and interpret `E_warm` as the first **2500 steps** unless an equivalent codebase convention is already fixed.

### Phase A: Does the Problem Exist? (Diagnostic)

**A1. Gradient conflict/domination statistics** on FixMatch + CIFAR-100-LT gamma=100 (no GradVax):
- Plot `cos(g_head_pseudo, g_sup_tail)` histogram at selected epochs
- Plot `||g_head_pseudo|| / ||g_sup_tail||` time series
- Plot intervention frequency (conflict / domination / neither)
- **Purpose**: Verify that the target interference pattern is real and not anecdotal

**A2. Optional correlation check**:
- Correlate running interference statistics with tail accuracy
- Treat this as supporting evidence rather than a strict requirement

### Phase B: Does GradVax Fix It? (Mechanism Validation)

**B1. Oracle pseudo-label experiment** on CIFAR-100-LT gamma=100:
- Replace pseudo-labels with ground truth
- Run FixMatch-oracle with and without GradVax
- **Expected**: GradVax benefit should shrink substantially if noisy pseudo-label interference is the main mechanism
- We do **not** require the gain to vanish entirely; residual benefit may still reflect generic stabilization effects

**B2. Training dynamics**:
- Plot tail/head/overall test accuracy vs. evaluation step for baseline and GradVax
- **Expected**: Baseline shows weaker or degrading tail performance in later training; GradVax stabilizes or improves tail accuracy

### Phase C: How Does GradVax Compare? (Main Results)

**C1. Primary comparison table**:
- Datasets: **CIFAR-10-LT gamma=100**, **CIFAR-100-LT gamma=100**, and **STL-10-LT gamma_l in {10, 20}, gamma_u=N/A**
- Distribution: consistent distribution for CIFAR-LT; original unlabeled pool for STL-10-LT
- Baselines: FixMatch, DARP, ABC, SimPro
- Report: Overall, Head, Medium, Tail, Balanced Accuracy
- GradVax applied to FixMatch as the primary result
- CIFAR-100-LT gamma=150 is no longer required for the primary table; keep it only as an optional appendix stress test if compute remains.
- STL-10-LT is used as an additional benchmark with standard long-tailed labeled splits and the original unlabeled pool; gamma_l=20 is the stronger STL stress setting, while gamma_l=10 provides the lighter reference setting. Neither should replace CIFAR-100-LT gamma=100 as the mechanism-validation anchor.

**Benchmark Reproduction Note**
- For non-primary baselines, results should be locally reproduced under the same dataset split, imbalance ratio, training protocol, and evaluation metrics as the current setting.
- Local compute should cover both the GradVax-specific experiments and the comparison baselines, including FixMatch, DARP, ABC, SimPro, oracle pseudo-label control, optimization baselines, and core ablations.

**C2. Lightweight optimization baselines** on CIFAR-100-LT gamma=100:
- PCGrad-adapted
- Gradient clipping
- Head loss downweighting
- **Purpose**: Show that GradVax is better than several simple alternatives, especially the most direct reweighting-style correction

**Execution priority note**:
- Among the optimization baselines, **head loss downweighting** should be treated as a first-priority sanity baseline rather than an optional add-on, because it is the simplest alternative reviewers are likely to ask about.

### Phase D: What Matters in GradVax? (Ablations)

**D1. Component ablation** (CIFAR-100-LT gamma=100):

| Variant | Conflict projection | Domination dampening | EMA tail anchor |
|---------|---------------------|----------------------|-----------------|
| Full GradVax | yes | yes | yes |
| No domination | yes | no | yes |
| No conflict | no | yes | yes |
| No EMA | yes | yes | no |

**D2. Light hyperparameter sensitivity**:
- beta sweep and tau sweep on the main setting
- Show that performance is reasonably stable around defaults
- If time is limited, prioritize beta and tau before expanding to a wider sweep of `E_warm`

### Phase E: Cost and Tradeoffs

**E1. Wall-clock and memory overhead** vs. baseline on CIFAR-100-LT:
- Report epoch time, total training time, and peak GPU memory
- Use the same machine, batch size, logging frequency, and evaluation schedule for baseline and GradVax
- Prefer repeated measurement when feasible

We do not require large-scale experiments such as ImageNet-127, full distribution-mismatch suites, full plug-in validation on many baselines, calibration studies, or a complete Pareto analysis.

---

## 6. Relationship to Prior Work

Prior work can be viewed as two largely complementary lines.

### 6.1 Output-Level LTSSL Rebalancing

| Method | Mechanism | Relation to GradVax |
|--------|-----------|---------------------|
| DARP | Distribution alignment / pseudo-label refinement | Output-level correction |
| ABC | Auxiliary balanced classifier | Output-level / classifier-output correction |
| SimPro | Prototype or representation-guided balancing | Output-level / representation-side correction |

These methods primarily act on predictions, pseudo-label distributions, or output-side balancing. GradVax is not intended to replace them; it targets a different level of the training process.

### 6.2 Gradient-Level Optimization Control

| Method | Mechanism | Key Difference from GradVax |
|--------|-----------|----------------------------|
| PCGrad (Yu et al., 2020) | Symmetric pairwise projection | GradVax is asymmetric, norm-aware, and LTSSL-specific |
| CAGrad (Liu et al., 2021) | Common descent direction | Symmetric and not designed to handle domination by pseudo-label gradients |
| LCGC (2025) | Gradient conflict reweighting | Reweighting rather than projection; no domination dampening |

**Practical novelty of GradVax in this version**: an LTSSL-oriented optimizer-level correction that combines asymmetric intervention, domination dampening, and an EMA-smoothed tail anchor on classifier gradients.

---

## 7. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Overall Accuracy | Standard comparison |
| Head/Medium/Tail Accuracy | Main group-wise metric |
| Balanced Accuracy | Mean per-class performance |
| Tail Pseudo-Label Precision/Recall | Mechanism diagnostic |
| Training Time / GPU Memory | Practical overhead |

**Metric protocol details**:
- Head/Medium/Tail partitions are defined by fixed terciles of labeled-set class frequency
- Balanced Accuracy is reported as the mean per-class accuracy
- Tail pseudo-label precision/recall is evaluated only on pseudo-labeled samples assigned to tail classes, using the true label for offline analysis

---

## 8. Tightened Claims

We do **not** claim:
- That GradVax fully solves LTSSL
- That gradient projection itself is new
- That broad causal conclusions hold beyond the tested setting
- That classifier-only intervention is universally sufficient in all regimes

We **do** claim:
- GradVax addresses a specific and measurable source of tail suppression: classifier-level interference from head pseudo-label gradients
- The combination of asymmetric projection and norm-aware dampening is better suited to LTSSL than generic symmetric gradient surgery for this setting
- The method is lightweight, plug-and-play, and practical
- The mechanism can be partially validated through gradient diagnostics and an oracle pseudo-label control experiment
- GradVax yields consistent tail improvements on core LTSSL benchmarks with small overhead

---

## 9. Failure Mode Analysis

### 9.1 When GradVax Helps
- High imbalance ratio (especially gamma >= 50)
- Noisy head-class pseudo-labels
- Mid-to-late training when pseudo-label bias accumulates

### 9.2 When GradVax Does Little
- Low imbalance settings
- Nearly perfect pseudo-labels
- Settings where head pseudo-label gradients are not the main source of tail suppression

### 9.3 When GradVax May Hurt
- Extremely sparse tail supervision, making the EMA anchor unstable
- tau set too small, causing excessive suppression of useful head gradients
- Cases where deeper feature-level interference dominates classifier-level interference

We commit to reporting negative or weak results honestly if diagnostics do not support the mechanism.

---

## 10. Classifier-Scope Thesis

We keep **classifier-only scope** as the main practical thesis.

**Reasoning**:
1. The classifier directly parameterizes class decision boundaries
2. Classifier gradients are lower-dimensional and easier to diagnose reliably
3. Classifier-only intervention yields a lightweight and efficient plug-and-play design

Classifier-only scope is a deliberate design choice for practicality and interpretability, not a claim that deeper intervention is never useful. Our thesis is that classifier-level correction provides the best first tradeoff between effectiveness, clarity, and cost for a compact LTSSL method. Operationally, "classifier-only" means that GradVax modifies only the final linear classifier gradients; it does **not** mean detaching the backbone or disabling standard gradient flow through feature extraction.

---

## 11. Mechanism Isolation: Compact but Informative Controls

To keep the method package compact while preserving interpretability, we retain only the most informative controls.

**Control 1: Oracle pseudo-labels**
- If GradVax's benefit drops sharply when pseudo-label noise is removed, this supports the intended mechanism.

**Control 2: Simple gradient baselines**
- Compare against clipping and head downweighting.
- If GradVax outperforms them, this suggests that its state-dependent directional design matters beyond simple magnitude suppression.

**Control 3: EMA ablation**
- Compare full GradVax vs. no EMA.
- If no-EMA is worse, stabilization matters.

Together, these controls provide targeted evidence for the intended mechanism while keeping the evaluation package compact. They do not prove every aspect of the method, but they support the core claim without requiring an expensive mechanism-isolation battery.

---

## 12. Agent Execution Notes

To reduce ambiguity during implementation and experimentation, the agent should follow these defaults unless explicitly overridden:

1. **Skip conditions**
   - Skip GradVax on steps where `pseudo_head_mask` is empty
   - Skip GradVax on steps where no valid EMA tail anchor is available
   - Keep training otherwise unchanged

2. **EMA behavior**
   - Update EMA only when at least 2 tail labeled samples are present
   - Otherwise reuse the previous EMA without update
   - During the first `E_warm=2500` steps, keep EMA updates active but do not apply GradVax projection or dampening to the head gradient
   - If `E_warm` ends and the EMA anchor is still invalid because tail samples remain too sparse or the value becomes numerically invalid (e.g. NaN), keep GradVax bypassed for those steps and surface a warning in logs rather than forcing a broken update

3. **Classifier-only discipline**
   - Preserve the backbone under the standard end-to-end backpropagation path of the base SSL method
   - Do **not** use `detach()` or equivalent graph-breaking operations on backbone features merely to enforce classifier-only scope
   - Prefer an implementation in which the backbone receives gradients from the standard total-loss backward pass, while GradVax modifies only the final linear classifier `.grad` tensors after the required classifier-level gradient quantities have been formed
   - Apply GradVax only to the final linear classifier parameters or their `.grad` tensors after the required gradient quantities have been formed

4. **Memory discipline**
   - If `torch.autograd.grad(...)` is used to extract `g_sup_tail` and `g_head_pseudo`, do not keep the graph longer than necessary
   - Use strict gradient clearing before and after the modified update, e.g. `optimizer.zero_grad(set_to_none=True)`
   - If multiple gradient queries are issued from the same forward pass, ensure that only the necessary call retains the graph, and release intermediate references immediately after `g_total` is assembled
   - After applying the modified `g_total` and executing `optimizer.step()`, immediately clear gradients and release graph-related intermediate tensors so that computation graphs from prior steps are not carried into later iterations
   - The implementation should favor aggressive memory management over convenience, and must avoid training-loop designs that accumulate retained graphs across steps and trigger OOM on GPU servers

5. **Priority order for execution**
   - FixMatch baseline
   - FixMatch + GradVax main settings
   - Oracle pseudo-label control
   - Head downweighting baseline
   - Other optimization baselines
   - Component ablation
   - Overhead measurement
   - Optional sensitivity analysis

6. **Partition persistence**
   - Before training starts, print and save the fixed head / medium / tail class lists derived from labeled-set frequency, and reuse the exact same saved partition for diagnostics, main results, and ablations
   - Do not allow later runs to silently redefine the tercile split under a different seed or preprocessing order

7. **Reporting discipline**
   - Use the same evaluation and logging schedule across compared methods
   - Record whether baseline numbers are locally reproduced under the unified protocol
   - Clearly separate must-have vs. optional experiments in logs and summaries

8. **Writing discipline**
   - Describe Proposition 1 as motivation, not a formal theorem with convergence guarantees
   - Avoid overstating novelty; emphasize the combination of asymmetry, norm-awareness, and lightweight classifier-only intervention
