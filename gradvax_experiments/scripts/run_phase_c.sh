#!/usr/bin/env bash
# Phase C: Main comparison table + optimization baselines
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA=./data/cache; LOGS=./results; STEPS=131072

echo "=== C1: CIFAR-10-LT γ=100 ==="
python train.py --dataset cifar10 --gamma 100 --n_labeled 250 \
  --total_steps $STEPS --log_dir $LOGS --run_id cifar10_g100_baseline_s42 --seed 42 --data_root $DATA
python train.py --dataset cifar10 --gamma 100 --n_labeled 250 \
  --gradvax --total_steps $STEPS --log_dir $LOGS --run_id cifar10_g100_gradvax_s42 --seed 42 --data_root $DATA

echo "=== C1: CIFAR-100-LT γ=150 ==="
python train.py --dataset cifar100 --gamma 150 --n_labeled 500 \
  --total_steps $STEPS --log_dir $LOGS --run_id cifar100_g150_baseline_s42 --seed 42 --data_root $DATA
python train.py --dataset cifar100 --gamma 150 --n_labeled 500 \
  --gradvax --total_steps $STEPS --log_dir $LOGS --run_id cifar100_g150_gradvax_s42 --seed 42 --data_root $DATA

echo "=== C2: Optimization baselines (CIFAR-100-LT γ=100) ==="
# Head loss downweighting (highest priority)
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --total_steps $STEPS --log_dir $LOGS --run_id cifar100_g100_head_downweight_s42 \
  --head_downweight 0.5 --seed 42 --data_root $DATA

# Gradient clipping
python train.py --dataset cifar100 --gamma 100 --n_labeled 500 \
  --total_steps $STEPS --log_dir $LOGS --run_id cifar100_g100_grad_clip_s42 \
  --grad_clip 1.0 --seed 42 --data_root $DATA

echo "=== Phase C complete ==="
