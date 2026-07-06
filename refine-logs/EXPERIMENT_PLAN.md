# GradVax Experiment Plan (CCF-C Revision V3 - Aligned with Final v3 and execution-optimized)

---
---

## Experimental Narrative: 5 Phases

**Execution premise.** This plan preserves the original five-phase narrative and is intended for direct agent execution. The goal is not to expand scope, but to make each phase more operationally unambiguous and more robust against implementation drift.

**Time-unit convention.** To avoid ambiguity from infinite or padded unlabeled sampling in FixMatch-style training, use **steps / iterations** as the primary time unit. Unless a base codebase already defines an equivalent convention, use evaluation every **500 steps** (treated as one logical evaluation epoch for plotting), and set warm-up to the first **2500 steps**.

**Default operational definitions (fix before running).**
- **Conflict**: `cos(g_head_pseudo, g_tail_anchor) < 0`
- **Domination**: `||g_head_pseudo|| / (||g_tail_anchor|| + epsilon) > tau`
- **Intervention triggered**: conflict or domination
- **Tail anchor used for GradVax decisions**: EMA-smoothed tail supervised gradient when available; otherwise bypass GradVax for that step and fall back to the standard FixMatch update

**Default implementation rules (do not reinterpret unless necessary).**
- Gradient scope: **classifier head only**
- Classifier-only scope applies only to the gradient modification step; backbone features must still follow the standard end-to-end backpropagation path
- Do **not** use `detach()` or equivalent graph-breaking operations on backbone features merely to enforce classifier-only scope
- Recommended implementation path: let the backbone receive gradients through the standard total-loss backward pass, separately compute the classifier-level gradient components needed by GradVax, assemble the modified classifier update as `g_total`, and manually write it back to `classifier.weight.grad` (and `classifier.bias.grad` if present)
- Class partitioning: **fixed terciles by labeled-set frequency**
- Print and save the head / medium / tail class lists before training starts, and reuse the same saved partition in A1, C1, and all later analyses
- If `pseudo_head_mask` is empty, skip GradVax for that step
- Guard classifier-level gradient extraction with explicit if-else control flow so `torch.autograd.grad(...)` is called only when the corresponding mask is non-empty
- If tail samples in batch are fewer than 2, do **not** update tail EMA; if no valid EMA anchor exists yet, bypass all GradVax projection and dampening rules for that step and execute the standard FixMatch parameter update
- Warm-up: keep `g_tail_ema` updating silently during the first `E_warm=2500` steps, but do not apply GradVax projection or dampening until warm-up ends
- If `E_warm` ends and the EMA anchor is still invalid because tail samples remain too sparse or the value becomes numerically invalid (e.g. NaN), keep GradVax bypassed for those steps and surface a warning rather than forcing the method on
- Apply domination dampening **after** conflict projection when both conditions trigger
- If `torch.autograd.grad(...)` with `retain_graph=True` is used during gradient extraction, pair it with strict `optimizer.zero_grad(set_to_none=True)` discipline and immediate graph release after the modified update
- After assembling `g_total` and executing `optimizer.step()`, immediately clear gradients and release graph-related intermediate references; do not carry retained computation graphs across training steps
- During final gradient assembly, gracefully handle `None` components (e.g. treat them as zero tensors or conditionally accumulate only non-`None` gradients) so batches with missing tail or pseudo-head terms do not trigger `TypeError`

---

### Phase A: Does the Problem Exist? (Diagnostic - No GradVax)

**A1. Gradient Conflict/Domination Statistics**
- Dataset: CIFAR-100-LT gamma=100, FixMatch baseline (no GradVax)
- Track `cos(g_head_pseudo, g_sup_tail)` during training
- Track `||g_head_pseudo|| / ||g_sup_tail||` during training
- Plot histograms at selected evaluation steps
- Plot time series of norm ratio
- Track intervention frequency (what % of steps would trigger conflict / domination / neither)
- **Goal**: Verify that the target interference pattern is real and occurs frequently enough to motivate intervention

**Execution clarification for A1**
- Use the same head/medium/tail partitioning that will be used later in GradVax
- Log both the raw statistics against `g_sup_tail` and the effective trigger statistics under the finalized intervention rule
- Report at minimum:
  - fraction of steps with conflict
  - fraction of steps with domination
  - fraction of steps with either trigger
  - median and upper-quantile norm ratio
- **Purpose**: Make the final mechanism story operational rather than anecdotal

**Implementation Note for A1 Logging**
- Do **not** write diagnostics to disk at every training step
- Use buffered logging, TensorBoard, or Weights & Biases with reduced logging frequency (e.g., every 20-50 steps), while keeping evaluation on a coarser schedule such as every 500 steps
- Aggregate statistics in memory and periodically flush to disk
- **Purpose**: Avoid I/O bottlenecks that could significantly slow down training and distort overhead measurements

**A2. Optional Correlation Check**
- Compute running average of conflict / domination statistics in short windows
- Correlate with tail test accuracy measured periodically
- **Purpose**: Provide supporting evidence that stronger interference is associated with worse tail performance
- **Status**: Supporting analysis, not a required headline result

---

### Phase B: Does GradVax Fix It? (Mechanism Validation)

**B1. Oracle Pseudo-Label Experiment**
- Dataset: CIFAR-100-LT gamma=100
- Replace pseudo-labels with ground-truth labels
- Run FixMatch-oracle with and without GradVax
- **Expected**: GradVax benefit should shrink substantially when pseudo-label noise is removed, rather than necessarily disappearing completely

**Execution clarification for B1**
- Use the same optimizer, schedule, evaluation cadence, and classifier-only scope as the main method
- Treat this as the highest-priority mechanism control after A1
- If the oracle setting still shows a large gain, flag this explicitly for later interpretation rather than silently treating it as confirmation

**B2. Training Dynamics**
- Plot tail / head / overall test accuracy vs. evaluation step for baseline and GradVax
- **Expected**: Baseline shows weaker or degrading tail performance in later training; GradVax stabilizes or improves tail accuracy

**Execution clarification for B2**
- Use identical seeds, evaluation intervals (default: every 500 steps), and checkpoint cadence for baseline and GradVax
- Save enough checkpoints to support later plotting without rerunning training
- Tail dynamics are more important than cosmetic smoothness of plots

---

### Phase C: How Does GradVax Compare? (Main Results)

**C1. Primary Comparison Table**
- Datasets:
  - CIFAR-10-LT gamma=100
  - CIFAR-100-LT gamma=100
  - STL-10-LT gamma_l in {10, 20}, gamma_u=N/A
- Distribution setting: consistent distribution for CIFAR-LT; original unlabeled pool for STL-10-LT
- Baselines: FixMatch, DARP, ABC, SimPro
- GradVax applied to: FixMatch (primary)
- Metrics: Overall, Head, Medium, Tail, Balanced Accuracy
- **Goal**: Show consistent tail improvement on compact but representative LTSSL benchmarks

**Execution clarification for C1**
- Treat **FixMatch baseline + FixMatch + GradVax** as the irreducible core result
- Ensure the head/medium/tail split and Balanced Accuracy definition are fixed once and reused everywhere
- Recommended metric definitions:
  - Head / Medium / Tail Accuracy: average accuracy over classes in the corresponding tercile partition
  - Balanced Accuracy: mean per-class accuracy over all classes
- Baseline results should be locally reproduced under the same dataset split, imbalance ratio, training protocol, and metric definition before inclusion
- CIFAR-100-LT gamma=150 is not part of the required primary table. Treat it only as an optional appendix stress test if the core gamma=100 experiments finish cleanly.
- For STL-10-LT, follow the common LTSSL convention: construct long-tailed labeled splits and use the original STL-10 unlabeled pool, so gamma_u is treated as unavailable rather than forced to match gamma_l. Use gamma_l=20 as the stronger STL stress setting and gamma_l=10 as the lighter reference setting.

**Baseline Execution Strategy**
- For non-primary baselines (especially DARP, ABC, and SimPro), run the methods locally under the same settings rather than relying on published benchmark tables
- Use the same protocol, dataset split, imbalance ratio, and evaluation metrics across all compared methods
- Reserve local GPU budget for:
  - FixMatch baseline
  - FixMatch + GradVax
  - DARP
  - ABC
  - SimPro
  - Oracle pseudo-label experiment
  - Optimization baselines
  - Core ablations
- **Purpose**: Keep the comparison protocol unified and ensure that all primary numbers are directly comparable

**C2. Lightweight Optimization Baselines**
- Dataset: CIFAR-100-LT gamma=100
- Compare GradVax against:
  - PCGrad-adapted (same head / tail split, symmetric projection)
  - Gradient clipping
  - Head loss downweighting
- **Purpose**: Show that GradVax is better than several simple gradient-level alternatives

**Execution clarification for C2**
- Treat **head loss downweighting** as a high-priority sanity baseline, not an afterthought
- Use the same training budget and evaluation protocol as GradVax wherever possible
- If time is limited, prioritize: head loss downweighting > gradient clipping > PCGrad-adapted

---

### Phase D: What Matters in GradVax? (Ablations)

**D1. Component Ablation** (CIFAR-100-LT gamma=100)

| Variant | Conflict projection | Domination dampening | EMA tail anchor |
|---------|---------------------|----------------------|-----------------|
| Full GradVax | yes | yes | yes |
| No domination | yes | no | yes |
| No conflict | no | yes | yes |
| No EMA | yes | yes | no |

**Execution clarification for D1**
- Keep all non-ablated settings fixed
- Reuse the exact same classifier-only gradient extraction pipeline across all variants
- Ensure that classifier-only intervention modifies only the final linear layer gradients and does not disable backbone learning
- The no-EMA variant is especially important because it tests whether stabilization of the tail anchor matters in practice

**D2. Light Hyperparameter Sensitivity**
- Dataset: CIFAR-100-LT gamma=100
- beta sweep: {0.95, 0.99, 0.999}
- tau sweep: {2, 5, 10, infinity}
- E_warm sweep: {0, 2500, 5000}
- **Goal**: Show that the default setting remains reasonably stable around the chosen operating point

**Execution clarification for D2**
- This phase is intentionally lightweight
- If compute becomes tight, reduce D2 to:
  - tau sweep as primary
  - one secondary check on beta or `E_warm`
- Do not let D2 crowd out D1 or C2

---

### Phase E: Cost and Tradeoffs

**E1. Computational Overhead**
- Dataset: CIFAR-100-LT gamma=100
- Measure:
  - Wall-clock time per epoch
  - Peak GPU memory
  - Total training time relative to baseline
- **Goal**: Verify that GradVax remains lightweight and practical

**Measurement Note**
- When reporting wall-clock overhead, use the same logging frequency and evaluation schedule for baseline and GradVax
- Exclude avoidable debugging overhead and one-time setup costs from the final runtime comparison
- Prefer the same machine type, batch size, and seed protocol for all overhead measurements
- If possible, repeat overhead measurement more than once and report the average
- **Purpose**: Keep overhead reporting fair and reproducible

---

## Benchmarks Summary

| Dataset | gamma Values | Distribution Settings | Priority |
|---------|----------|----------------------|----------|
| CIFAR-10-LT | 100 | Consistent | Tier 1 |
| CIFAR-100-LT | 100 | Consistent | Tier 1 |
| STL-10-LT | gamma_l in {10, 20}, gamma_u=N/A | Original unlabeled pool | Tier 2 |
| CIFAR-100-LT | 150 | Consistent | Optional appendix stress test |

---

## Baselines Summary

| Category | Methods |
|----------|---------|
| LTSSL | FixMatch, DARP, ABC, SimPro |
| Optimization | Head downweighting, gradient clipping, PCGrad-adapted |
| Plug-in | FixMatch + GradVax |

---

## Metrics Summary

| Metric | Type |
|--------|------|
| Overall Accuracy | Standard |
| Head / Medium / Tail Accuracy | Per-group (primary) |
| Balanced Accuracy | Fairness / class balance |
| Tail Pseudo-Label Precision / Recall | Mechanism diagnostic |
| Training Time / GPU Memory | Practical overhead |

**Metric execution notes**
- Fix the head/medium/tail class split once using labeled-set frequency and reuse it everywhere
- Print and save the resulting class partition before the main runs so that diagnostics and comparison tables use exactly the same definition
- Balanced Accuracy should be implemented as mean per-class accuracy
- Tail pseudo-label precision / recall should be treated as supporting mechanism diagnostics, not the primary success criterion

---

## Timeline

### Week 1: Foundation + Phase A
1. Implement GradVax module (classifier-only, 3 hyperparameters)
2. Implement diagnostic logging (cosine, norm ratio, intervention frequency)
3. Run FixMatch baseline on CIFAR-100-LT gamma=100
4. Run Phase A diagnostics (A1 and optional A2)
5. Inspect whether gradient conflict / domination is frequent enough to justify intervention

### Week 2: Phase B + Core Phase C
6. Run oracle pseudo-label experiment (B1)
7. Run FixMatch + GradVax on CIFAR-100-LT gamma=100
8. Run FixMatch + GradVax on CIFAR-10-LT gamma=100
9. Run FixMatch + GradVax on STL-10-LT gamma_l in {10, 20}
10. Plot training dynamics (B2)
11. Collect primary baselines in C1:
   - Run DARP, ABC, and SimPro locally under the unified protocol
   - Keep the same dataset split, imbalance ratio, and metric definition across all methods

### Week 3: Phase C Completion + Phase D + Phase E
11. Run lightweight optimization baselines in C2
12. Run component ablation (D1)
13. Run light hyperparameter sensitivity (D2)
14. Measure overhead (E1)
15. Organize tables and figures
16. Paper writing

**Agent note on sequencing**
- Do not start broad ablations before the core FixMatch vs. FixMatch+GradVax comparison is stable
- If implementation ambiguity appears, preserve the classifier-only, asymmetric, EMA-anchored version rather than inventing broader variants

---

## Execution Priority

### Must-have for CCF-C submission
- A1 gradient conflict / domination statistics
- B1 oracle pseudo-label experiment
- B2 training dynamics
- C1 primary comparison table on CIFAR-10-LT, CIFAR-100-LT, and STL-10-LT
- C2 lightweight optimization baselines
- D1 component ablation
- E1 computational overhead

### Good-to-have
- A2 optional correlation check
- D2 light hyperparameter sensitivity
- CIFAR-100-LT gamma=150 appendix stress test, only after the gamma=100 core package is complete

### Scope intentionally not included in v3
- ImageNet-127
- Full distribution-mismatch experiments
- Large plug-in validation on multiple baselines
- Full calibration study
- Full Pareto frontier
- Deep mechanism-isolation battery beyond oracle + simple baselines + EMA ablation

---

## Practical Execution Notes

**1. Protect training speed**
- Keep diagnostics lightweight
- Avoid per-step disk writes
- Reuse the same evaluation hooks across baseline and GradVax
- Skip GradVax cleanly on inapplicable steps rather than forcing unstable fallback behavior

**Classifier-scope safety note**
- "Classifier-only" means modifying only the final linear classifier gradients, not detaching the backbone
- Preserve standard backpropagation through the backbone exactly as in the base SSL method
- Prefer the implementation pattern where the backbone follows the normal total-loss backward pass and only the final classifier `.grad` tensors are overwritten by the assembled GradVax update
- Do **not** use `detach()` or equivalent shortcuts that would prevent the backbone from learning

**Memory safety note**
- In implementations that query multiple gradients from one forward pass, do not leave `retain_graph=True` active longer than necessary
- Use strict gradient clearing before and after the modified update, preferably with `optimizer.zero_grad(set_to_none=True)`
- After `optimizer.step()`, immediately delete or overwrite graph-dependent intermediate tensors and release references so the next iteration starts without stale graph baggage
- The Agent should prefer aggressive graph cleanup to avoid OOM failures on remote GPU servers

**2. Protect GPU budget**
- Spend compute on experiments that only you can produce
- Use local GPU budget to reproduce the main comparison baselines under a unified protocol
- Do not substitute published benchmark numbers for the primary comparison table
- Prioritize sanity baselines that are most likely to be raised by reviewers, especially head downweighting

**3. Protect submission quality**
- If time becomes tight, prioritize:
  1. FixMatch baseline
  2. FixMatch + GradVax main results
  3. Oracle experiment
  4. Optimization baselines
  5. Component ablation
- This minimum package preserves the paper's core logic

**4. Protect claim discipline**
- Treat the method as a targeted optimizer-level intervention, not a universal fix for LTSSL
- Use diagnostics and the oracle control to support the intended mechanism, but do not overstate causal reach
- Keep classifier-only scope as a deliberate practicality and interpretability choice
