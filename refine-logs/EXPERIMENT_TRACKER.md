# GradVax Experiment Tracker

*Last updated: 2026-07-06*
*Status: Pre-implementation (tracker aligned with Experiment Plan V3 and Final Proposal V3.1)*

---

## Phase A: Does the Problem Exist?

| ID | Experiment | Dataset | Status | Result | Notes |
|----|-----------|---------|--------|--------|-------|
| A1 | Gradient conflict/domination statistics | CIFAR-100-LT gamma=100 | TODO | TBD | Baseline FixMatch, no GradVax; track cosine, norm ratio, trigger frequency |
| A2 | Interference-degradation correlation (optional) | CIFAR-100-LT gamma=100 | OPTIONAL | TBD | Running-window correlation between interference stats and tail accuracy |

## Phase B: Mechanism Validation

| ID | Experiment | Dataset | Status | Result | Prediction |
|----|-----------|---------|--------|--------|------------|
| B1 | Oracle pseudo-labels | CIFAR-100-LT gamma=100 | TODO | TBD | Benefit should shrink substantially when pseudo-label noise is removed |
| B2 | Training dynamics curves | CIFAR-100-LT gamma=100 | TODO | TBD | Tail accuracy should stabilize or improve relative to baseline |

## Phase C: Main Results

| ID | Experiment | Dataset | gamma | Dist. | Status | Result |
|----|-----------|---------|---|-------|--------|--------|
| C1a | Primary table | CIFAR-10-LT | 100 | Consistent | TODO | TBD |
| C1b | Primary table | CIFAR-100-LT | 100 | Consistent | TODO | TBD |
| C1c | Primary table | STL-10-LT | gamma_l=10 | gamma_u=N/A | TODO | TBD |
| C1d | Primary table | STL-10-LT | gamma_l=20 | gamma_u=N/A | TODO | TBD |
| C2 | Optimization baselines | CIFAR-100-LT | 100 | Consistent | TODO | TBD |

## Phase C: Plug-in Tests

| ID | Base Method | +GradVax | Dataset | Status | Result |
|----|------------|----------|---------|--------|--------|
| C1-plug-a | FixMatch | FixMatch+GradVax | CIFAR-10-LT gamma=100 | TODO | TBD |
| C1-plug-b | FixMatch | FixMatch+GradVax | CIFAR-100-LT gamma=100 | TODO | TBD |
| C1-plug-c | FixMatch | FixMatch+GradVax | STL-10-LT gamma_l=10, gamma_u=N/A | TODO | TBD |
| C1-plug-d | FixMatch | FixMatch+GradVax | STL-10-LT gamma_l=20, gamma_u=N/A | TODO | TBD |

## Optional Appendix Stress Tests

| ID | Experiment | Dataset | Status | Result |
|----|------------|---------|--------|--------|
| APP1 | Stronger imbalance stress test | CIFAR-100-LT gamma=150 | OPTIONAL | TBD |

## Phase D: Ablations

| ID | Ablation | Variable | Values | Dataset | Status | Result |
|----|----------|----------|--------|---------|--------|--------|
| D1a | Component | Full GradVax | conflict=yes; domination=yes; EMA=yes | CIFAR-100-LT gamma=100 | TODO | TBD |
| D1b | Component | No domination | conflict=yes; domination=no; EMA=yes | CIFAR-100-LT gamma=100 | TODO | TBD |
| D1c | Component | No conflict | conflict=no; domination=yes; EMA=yes | CIFAR-100-LT gamma=100 | TODO | TBD |
| D1d | Component | No EMA | conflict=yes; domination=yes; EMA=no; instantaneous tail anchor | CIFAR-100-LT gamma=100 | TODO | TBD |
| D2a | beta sweep | {0.95, 0.99, 0.999} | sensitivity only | CIFAR-100-LT gamma=100 | OPTIONAL | TBD |
| D2b | tau sweep | {2, 5, 10, infinity} | sensitivity only | CIFAR-100-LT gamma=100 | OPTIONAL | TBD |
| D2c | E_warm sweep | {0, 2500, 5000} steps | sensitivity only | CIFAR-100-LT gamma=100 | OPTIONAL | TBD |

## Phase E: Cost and Tradeoffs

| ID | Experiment | Dataset | Status | Result |
|----|-----------|---------|--------|--------|
| E1a | Wall-clock overhead | CIFAR-100-LT gamma=100 | TODO | TBD |
| E1b | Memory overhead | CIFAR-100-LT gamma=100 | TODO | TBD |
| E1c | Total training time relative to baseline | CIFAR-100-LT gamma=100 | TODO | TBD |

---

## Falsifiable Predictions Tracker

| ID | Prediction | Threshold | Status | Outcome |
|----|-----------|-----------|--------|---------|
| P1 | Conflict occurs frequently enough to justify intervention | Report fraction of conflict steps | TODO | TBD |
| P2 | Domination occurs frequently enough to justify norm-aware control | Report fraction of domination steps and upper-quantile norm ratio | TODO | TBD |
| P3 | Either-trigger frequency is non-trivial | Report fraction of steps with either trigger | TODO | TBD |
| P4 | Oracle pseudo-labels reduce GradVax gain substantially | Qualitative/substantial reduction | TODO | TBD |
| P5 | Tail dynamics improve relative to baseline during training | Visible tail stabilization/improvement in curves | TODO | TBD |
| P6 | Head loss downweighting is weaker than GradVax | GradVax better on tail-focused metrics | TODO | TBD |
| P7 | Full GradVax outperforms at least one key ablated variant | Full > no-domination / no-conflict / no-EMA on tail metric | TODO | TBD |
| P8 | Overhead remains small and practical | Low runtime/memory overhead vs baseline | TODO | TBD |

---

## Implementation Milestones

| Milestone | Status | Target Date | Notes |
|-----------|--------|-------------|-------|
| Fix head / medium / tail partition and save class lists | TODO | Week 1 | Fixed terciles by labeled-set frequency; reuse everywhere |
| GradVax module (classifier-only) | TODO | Week 1 | Modify classifier gradients only; do not detach backbone |
| EMA anchor + skip logic + warm-up | TODO | Week 1 | Update EMA when tail count >= 2; warm-up = 2500 steps |
| Diagnostic logging (cosine, norm ratio, trigger frequency) | TODO | Week 1 | Buffered/reduced-frequency logging only |
| FixMatch baseline reproduction | TODO | Week 1 | CIFAR-100-LT gamma=100 |
| Phase A experiments | TODO | Week 1 | A1 required; A2 optional |
| Oracle pseudo-label control | TODO | Week 2 | Highest-priority mechanism control after A1 |
| FixMatch + GradVax main settings | TODO | Week 2 | CIFAR-100-LT gamma=100, CIFAR-10-LT gamma=100, STL-10-LT gamma_l in {10, 20} |
| Training dynamics plots | TODO | Week 2 | Same eval cadence across baseline and GradVax |
| Optimization baselines | TODO | Week 3 | Prioritize head downweighting > clipping > PCGrad-adapted |
| Component ablation | TODO | Week 3 | Full / no-domination / no-conflict / no-EMA |
| Overhead measurement | TODO | Week 3 | Same machine, batch size, logging, evaluation schedule |
| Optional sensitivity analysis | OPTIONAL | Week 3 | tau primary; beta or E_warm secondary if compute allows |
| Paper writing | TODO | Week 3 | Keep claims compact and mechanism-focused |

---

## Execution Defaults (Locked for Consistency)

| Item | Default |
|------|---------|
| Time unit | Steps / iterations |
| Evaluation cadence | Every 500 steps |
| Warm-up | First 2500 steps |
| Class partition | Fixed terciles by labeled-set frequency |
| Conflict definition | cos(g_head_pseudo, g_tail_anchor) < 0 |
| Domination definition | ||g_head_pseudo|| / (||g_tail_anchor|| + epsilon) > tau |
| Intervention trigger | Conflict or domination |
| Tail anchor for GradVax | EMA-smoothed tail supervised gradient when valid |
| EMA update condition | Tail labeled samples in batch >= 2 |
| GradVax skip conditions | Empty pseudo-head mask, invalid/unavailable tail EMA, or warm-up not finished |
| Gradient scope | Classifier head only for modification; backbone remains on standard backprop path |
| Balanced Accuracy | Mean per-class accuracy |
| Head/Medium/Tail Accuracy | Mean accuracy over classes in fixed tercile split |

---

## Scope Intentionally Excluded in This Tracker Version

| Category | Excluded Items |
|----------|----------------|
| Datasets | ImageNet-127 |
| Distribution settings | Full distribution-mismatch suite |
| Expanded plug-in validation | ABC+GradVax, SimPro+GradVax main-line validation |
| Extended analyses | Full calibration study, Pareto frontier, deep mechanism-isolation battery |
