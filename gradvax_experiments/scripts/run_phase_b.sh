#!/usr/bin/env bash
# Phase B: Oracle pseudo-label experiment + training dynamics
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA=./data/cache; LOGS=./results; STEPS=131072

echo "=== B1: Oracle — FixMatch baseline (no GradVax) ==="
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --oracle --total_steps $STEPS --log_dir $LOGS \
  --run_id cifar100_g100_oracle_baseline_s42 --seed 42 --data_root $DATA

echo "=== B1: Oracle — FixMatch + GradVax ==="
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --gradvax --oracle --total_steps $STEPS --log_dir $LOGS \
  --run_id cifar100_g100_oracle_gradvax_s42 --seed 42 --data_root $DATA

echo "=== B2: Training dynamics — FixMatch + GradVax (non-oracle) ==="
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --gradvax --total_steps $STEPS --log_dir $LOGS \
  --run_id cifar100_g100_gradvax_s42 --seed 42 --data_root $DATA

echo "=== Phase B complete ==="
