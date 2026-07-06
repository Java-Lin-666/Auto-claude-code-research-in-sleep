#!/usr/bin/env bash
# Phase A: Gradient conflict/domination diagnostics (no GradVax)
# Run FixMatch baseline on CIFAR-100-LT γ=100 and collect diagnostic stats

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA=./data/cache
LOGS=./results
STEPS=131072

echo "=== Phase A: Diagnostic run (baseline, no GradVax) ==="
python train.py \
  --dataset cifar100 --gamma 100 --n_labeled 500 \
  --total_steps $STEPS --eval_every 500 \
  --log_dir $LOGS --run_id cifar100_g100_baseline_s42 \
  --seed 42 --data_root $DATA

echo "=== Phase A complete. Diagnostics in $LOGS/cifar100_g100_baseline_s42_diag.jsonl ==="
