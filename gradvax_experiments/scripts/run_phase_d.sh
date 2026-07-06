#!/usr/bin/env bash
# Phase D: Component ablations + hyperparameter sensitivity
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATA=./data/cache; LOGS=./results; STEPS=131072
BASE="--dataset cifar100 --gamma 100 --n_labeled 500 --total_steps $STEPS --log_dir $LOGS --seed 42 --data_root $DATA"

echo "=== D1: Component ablations ==="
python train.py $BASE --gradvax --run_id cifar100_g100_gradvax_full_s42
python train.py $BASE --gradvax --no_domination --run_id cifar100_g100_gradvax_nodom_s42
python train.py $BASE --gradvax --no_conflict   --run_id cifar100_g100_gradvax_noconflict_s42
python train.py $BASE --gradvax --no_ema        --run_id cifar100_g100_gradvax_noema_s42

echo "=== D2: τ sweep (primary) ==="
for TAU in 2 5 10; do
  python train.py $BASE --gradvax --tau $TAU --run_id cifar100_g100_gradvax_tau${TAU}_s42
done
# τ=∞ equivalent: very large value
python train.py $BASE --gradvax --tau 1e9 --run_id cifar100_g100_gradvax_tauinf_s42

echo "=== Phase D complete ==="
