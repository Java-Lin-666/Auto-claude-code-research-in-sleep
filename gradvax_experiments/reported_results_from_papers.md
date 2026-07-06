# Reported Results Extracted From Local Papers

Purpose: collect baseline numbers that can be cited as reported results, not as locally reproduced results.

Main caution: these papers usually report bACC/GM or test accuracy, while the GradVax plan asks for Overall, Head, Medium, Tail, and Balanced Accuracy under one unified protocol. Use these numbers as background or secondary reported comparisons unless the GradVax runs exactly match the paper protocol.

## Source Mapping

- paper1: Lee and Kim, 2024, CDMAD: Class-Distribution-Mismatch-Aware Debiasing for Class-Imbalanced Semi-Supervised Learning.
- paper2: Li et al., 2022, Adaptive Confidence Margin for semi-supervised facial expression recognition.
- paper3: Sohn et al., FixMatch.
- paper4: Wei and Gan, 2023, Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency is All You Need.
- paper5: Xing et al., 2025, LCGC.
- extra-2510.03993v6: Hou et al., 2025, CPG: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning.
- extra-2312.15702v2: Ma et al., 2024, CPE: Three Heads Are Better Than One.
- extra-2402.13505v4: Du et al., 2024, SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed Semi-Supervised Learning.
- extra-2106.05682v2: Oh et al., 2022, DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning.
- extra-2403.12986v2: Feng et al., 2024, BaCon: Boosting Imbalanced Semi-supervised Learning via Balanced Feature-Level Contrastive Learning.
- extra-2305.08661v1: Du et al., 2023, GLMC: Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions.
- extra-2306.04621v3: Sanchez Aimar et al., 2024, ADELLO/FlexDA: Flexible Distribution Alignment for Long-tailed Semi-supervised Learning with Proper Calibration.
- extra-2101.09536v2: Smith et al., 2021, Memory-Efficient Semi-Supervised Continual Learning.
- extra-2103.16725v2: Hu et al., 2021, SimPLE: Similar Pseudo Label Exploitation for Semi-Supervised Classification.

## Alignment With Current GradVax Plan

Current required GradVax benchmarks:

- CIFAR-10-LT gamma=100, consistent distribution.
- CIFAR-100-LT gamma=100, consistent distribution.
- STL-10-LT gamma_l in {10, 20}, gamma_u=N/A, original unlabeled pool.

CIFAR-100-LT gamma=150 is no longer part of the required primary table. Keep it only as an optional appendix stress test if the gamma=100 core package finishes cleanly.

Reported numbers in this file are therefore mainly useful for:

- background/context comparisons;
- checking whether local reproduced baselines are in a plausible range;
- related-work discussion on pseudo-label bias and class imbalance.

They should not replace local GradVax-protocol runs for the primary table, because the main paper reports Overall, Head, Medium, Tail, and Balanced Accuracy under a unified protocol.

## Directly Useful As Reported LTSSL Baselines

### CIFAR-10-LT, consistent distribution, bACC/GM

Source: LCGC 2025, Table 1. Setting: gamma = gamma_l = gamma_u, gamma_u known.

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

Source: CDMAD 2024 Table 2 and LCGC 2025 Table 2. Setting: gamma_l = 100, gamma_u unknown.

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

Source: CDMAD 2024 Table 4 and LCGC 2025 Table 4. Setting: gamma = gamma_l = gamma_u.

| Method | gamma=20 | gamma=50 | gamma=100 |
|---|---:|---:|---:|
| FixMatch | 49.6+/-0.8 | 42.1+/-0.3 | 37.6+/-0.5 |
| FixMatch+DARP | 50.8+/-0.8 | 43.1+/-0.5 | 38.3+/-0.5 |
| FixMatch+DARP+cRT | 51.4+/-0.7 | 44.9+/-0.5 | 40.4+/-0.8 |
| FixMatch+CReST | 51.8+/-0.7 | 44.9+/-0.5 | 40.1+/-0.7 |
| FixMatch+CReST+LA | 52.9+/-0.1 | 47.3+/-0.2 | 42.7+/-0.7 |
| FixMatch+ABC | 53.3+/-0.8 | 46.7+/-0.3 | 41.2+/-0.7 |
| FixMatch+CoSSL | 53.9+/-0.8 | 47.6+/-0.6 | 43.0+/-0.6 |
| FixMatch+UDAL | - | 48.0+/-0.6 | 43.7+/-0.4 |
| FixMatch+CDMAD | 54.3+/-0.4 | 48.8+/-0.8 | 44.1+/-0.3 |
| FixMatch+LCGC | 55.3+/-0.5 | 49.3+/-0.3 | 44.8+/-0.5 |
| ReMixMatch | 51.6+/-0.4 | 44.2+/-0.6 | 39.3+/-0.4 |
| ReMixMatch+ABC | 55.6+/-0.4 | 47.9+/-0.1 | 42.2+/-0.1 |
| ReMixMatch+CDMAD | 57.0+/-0.3 | 51.1+/-0.5 | 44.9+/-0.4 |
| ReMixMatch+LCGC | 57.3+/-0.3 | 50.7+/-0.4 | 45.9+/-0.6 |

### STL-10-LT, bACC/GM

Source: LCGC 2025, Table 5. Setting: STL-10-LT with original unlabeled pool; gamma_u is unknown / N/A. Metric: bACC/GM.

Use in GradVax paper: usable as the main STL-10-LT reported baseline block for balanced long-tailed metrics. Do not mix numerically with top-1 accuracy tables.

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

Source: Wei and Gan 2023, Table 1. This uses test accuracy, not bACC/GM.

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

Source: Wei and Gan 2023, Table 2. Setting: STL-10-LT with original unlabeled pool; gamma_u is N/A. Metric: top-1 test accuracy (%), not bACC/GM.

Use in GradVax paper: usable as supplementary reported results for ACR/DASO/DARP/CReST under the ACR protocol. It supports related-work and sanity-check discussion, but should be separated from the LCGC bACC/GM table because the metric and protocol are different.

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

Source: Sanchez Aimar et al., 2024, Flexible Distribution Alignment: Towards Long-tailed Semi-supervised Learning with Proper Calibration, Table 3. Metric: paper reports test balanced accuracy averaged over final epochs, although the table caption says test accuracy. Use as supplementary LTSSL context; keep separate from LCGC bACC/GM and from local GradVax unified metrics.

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

- Table 2 reports CIFAR10-LT label-shift results and CIFAR100-LT gamma_l=50 label-shift results for FixMatch, DARP, CReST+, ABC, DASO, DebiasPL, CoSSL, UDAL, ADELLO, and SoftMatch. This is useful context but does not match the GradVax core CIFAR-100-LT gamma=100 setting.
- Appendix B reports training time on CIFAR100-LT50 using one V100-32GB: FixMatch 5h15m, ADELLO 5h18m, ABC 5h21m, CReST+ 6h22m, CoSSL 7h29m, DARP 7h43m, DASO 19h32m. Use only as qualitative overhead context because hardware/protocol differ from GradVax runs.
- Tables 11 and 12 report ECE/MCE calibration on CIFAR10-LT, STL10-LT20, and CIFAR100-LT. For STL10-LT20, ADELLO reports ECE 6.9+/-0.3 and MCE 25.9+/-1.0, much lower than FixMatch's ECE 37.8+/-4.5 and MCE 55.1+/-4.9. Useful for related work on pseudo-label confidence/calibration, not for GradVax's primary accuracy table.
- It discusses biased pseudo-label distributions and low-confidence pseudo-label usage, but it does not report direct pseudo-label F1, pseudo-label precision/recall, tail pseudo-label recall, or minority pseudo-label precision. Use DASO/CPE for those more direct pseudo-label-quality evidence types.

### CPG paper suitability note

Source: CPG 2025. This paper is useful for recent ReaLTSSL context and includes FixMatch, FreeMatch, SoftMatch, ACR, SimPro, CDMAD, and CPG on CIFAR-10-LT and CIFAR-100-LT.

Do not copy CPG values into the main reported table without visual table verification: the text extraction of its main tables is noisy. Also, its CIFAR-10-LT uses Nmax=400, Mmax=4600 and gamma in {100,150,200}; its CIFAR-100-LT uses Nmax=50, Mmax=450 and gamma in {10,15,20}. This is a realistic-arbitrary-distribution protocol rather than the exact GradVax protocol.

### DASO reported top-1 accuracy

Source: Oh et al., 2022, DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning, Table 1. Metric: top-1 accuracy (%). Useful for reported FixMatch/DARP/CReST+/ABC/DASO context and pseudo-label bias motivation. It does not include CIFAR-100-LT gamma=100 or gamma=150.

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

Source: Feng et al., 2024, BaCon: Boosting Imbalanced Semi-supervised Learning via Balanced Feature-Level Contrastive Learning, Table 1 and Table 2. Metric: balanced accuracy (%). Useful as a feature/representation-level CISSL reported baseline and as a contrast to GradVax's classifier-gradient-level intervention. It does not include CIFAR-100-LT gamma=150.

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

## Not Directly Reusable For The GradVax Main Table

- FixMatch original paper: standard balanced/semi-supervised benchmarks, not LTSSL. Useful for method description and hyperparameter background only.
- Adaptive Confidence Margin paper: facial expression recognition datasets (RAF-DB, SFEW, AffectNet), not CIFAR-LT LTSSL. Useful only as related work on adaptive confidence thresholds, not as a baseline table for GradVax.
- ACR paper: useful for reported LTSSL background, but its CIFAR-100 settings are gamma 10/20, not the current GradVax core CIFAR-100 gamma=100 setting. Its metric is test accuracy, not Head/Medium/Tail/Balanced.
- SimPro/CPE/CPG papers: useful as reported top-1 accuracy context, but their protocols and metrics differ from the GradVax primary table. Keep them separate from local GradVax results unless the paper explicitly matches the same split, gamma, metric, and training protocol.
- Smith et al., 2021, Memory-Efficient Semi-Supervised Continual Learning: semi-supervised continual learning on CIFAR-100, not LTSSL main-table evidence.
- Hu et al., 2021, SimPLE: standard SSL on CIFAR-10/SVHN/CIFAR-100/Mini-ImageNet, not long-tailed CIFAR-LT. Useful only as generic SSL related work if needed.
- DASO and BaCon: useful as reported CISSL/LTSSL context and mechanism motivation, but their CIFAR-100 settings do not match the current GradVax core CIFAR-100 gamma=100 primary table.
- GLMC 2023: supervised long-tailed visual recognition on CIFAR-10-LT, CIFAR-100-LT, and ImageNet-LT, not semi-supervised LTSSL. It has CIFAR-LT top-1 and ImageNet-LT Many/Medium/Few results, but no STL-10-LT and no FixMatch/DARP/ABC/SimPro-style SSL comparison. Use only as optional related work for supervised long-tailed representation learning, not as GradVax reported baseline.
- ADELLO/FlexDA 2024: useful LTSSL reported context, calibration evidence, and overhead comparison. Its CIFAR100-LT main label-shift setting uses gamma_l=50 rather than GradVax's core gamma=100, so use the CIFAR100 numbers as context only. Its STL10-LT20 results are directly useful as supplementary reported context.

## Current Reported-Baseline Coverage

| GradVax setting | Reported external context status | Notes |
|---|---|---|
| CIFAR-10-LT gamma=100 | Covered well | LCGC/CDMAD bACC-GM, ACR/CPE/SimPro top-1, BaCon bACC context. |
| CIFAR-100-LT gamma=100 | Covered partially | CDMAD/LCGC report bACC for many baselines; still lacks GradVax-specific Head/Medium/Tail under unified protocol. |
| STL-10-LT gamma_l=10/20 | Covered well as reported context | LCGC reports bACC/GM; ACR, DASO, SimPro, CPE, and ADELLO report top-1/balanced-accuracy-style results under their protocols; BaCon reports balanced accuracy for gamma_l=10. gamma_u is N/A and metric/protocol details vary, so still run local unified metrics. |
| CIFAR-100-LT gamma=150 | Optional only | No longer a required primary-table gap; treat as appendix stress test if run locally. |

Clarification: CIFAR-100-LT gamma=100 is not missing external reported baselines. The missing part is local GradVax-protocol evidence: FixMatch rerun, FixMatch+GradVax, Head/Medium/Tail, and mechanism diagnostics under the same split/metric/training setup.

## Are The Reported Results Enough?

No. They are enough for a reported-results/background table, but not enough for the GradVax paper's main evidence.

Still required to run locally:

- FixMatch baseline under the exact GradVax protocol.
- FixMatch+GradVax under the exact same protocol.
- Gradient conflict/domination diagnostics.
- Oracle pseudo-label control.
- Head-loss downweighting sanity baseline.
- Component ablations: no conflict, no domination, no EMA.
- Runtime and GPU memory overhead.
- Head/Medium/Tail and Balanced Accuracy metrics, because the cited papers mostly report bACC/GM or overall test accuracy.
