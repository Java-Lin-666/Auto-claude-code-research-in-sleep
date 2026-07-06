#!/usr/bin/env bash
# Phase E: Computational overhead measurement
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA=./data/cache; LOGS=./results
# Short run for overhead: 2000 steps is enough to measure steady-state cost
STEPS=2000

echo "=== E1: Overhead — baseline ==="
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --total_steps $STEPS --eval_every 500 --log_dir $LOGS \
  --run_id cifar100_g100_overhead_baseline_s42 --seed 42 --data_root $DATA

echo "=== E1: Overhead — GradVax ==="
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --gradvax --total_steps $STEPS --eval_every 500 --log_dir $LOGS \
  --run_id cifar100_g100_overhead_gradvax_s42 --seed 42 --data_root $DATA

echo "=== Phase E complete. Compare wall-clock times above. ==="
