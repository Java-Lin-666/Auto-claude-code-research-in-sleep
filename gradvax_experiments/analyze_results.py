"""Analyze GradVax experiment results and print comparison tables."""
import json
import os
import glob
import numpy as np


def load_results(log_dir):
    results = {}
    for path in glob.glob(os.path.join(log_dir, '*_results.json')):
        run_id = os.path.basename(path).replace('_results.json', '')
        with open(path) as f:
            data = json.load(f)
        if data:
            best = max(data, key=lambda r: r['tail'])
            last = data[-1]
            results[run_id] = {'best_tail': best, 'last': last, 'curve': data}
    return results


def print_table(results, runs, label=''):
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'Run':<45} {'Overall':>8} {'Head':>8} {'Medium':>8} {'Tail':>8} {'Balanced':>9}")
    print('-'*70)
    for run_id in runs:
        if run_id not in results:
            print(f"  {run_id:<43} [missing]")
            continue
        r = results[run_id]['last']
        print(f"  {run_id:<43} {r['overall']:>8.3f} {r['head']:>8.3f} "
              f"{r['medium']:>8.3f} {r['tail']:>8.3f} {r['balanced']:>9.3f}")


def analyze_diag(log_dir, run_id):
    path = os.path.join(log_dir, f'{run_id}_diag.jsonl')
    if not os.path.exists(path):
        return
    entries = [json.loads(l) for l in open(path)]
    if not entries:
        return
    cos = [e['cos'] for e in entries]
    ratio = [e['norm_ratio'] for e in entries]
    conflict = [e['conflict'] for e in entries]
    domination = [e['domination'] for e in entries]
    either = [int(c or d) for c, d in zip(conflict, domination)]
    print(f"\n=== Phase A Diagnostics: {run_id} ===")
    print(f"  Steps logged:          {len(entries)}")
    print(f"  Conflict fraction:     {np.mean(conflict):.3f}")
    print(f"  Domination fraction:   {np.mean(domination):.3f}")
    print(f"  Either-trigger frac:   {np.mean(either):.3f}")
    print(f"  Median norm ratio:     {np.median(ratio):.2f}")
    print(f"  95th-pct norm ratio:   {np.percentile(ratio, 95):.2f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir', default='./results')
    args = p.parse_args()

    results = load_results(args.log_dir)
    print(f"Loaded {len(results)} runs from {args.log_dir}")

    # Phase A diagnostics
    analyze_diag(args.log_dir, 'cifar100_g100_baseline_s42')

    # C1: Main comparison
    print_table(results, [
        'cifar10_g100_baseline_s42',
        'cifar10_g100_gradvax_s42',
        'cifar100_g100_baseline_s42',
        'cifar100_g100_gradvax_s42',
        'cifar100_g150_baseline_s42',
        'cifar100_g150_gradvax_s42',
    ], label='C1: Primary Comparison Table')

    # B1: Oracle
    print_table(results, [
        'cifar100_g100_oracle_baseline_s42',
        'cifar100_g100_oracle_gradvax_s42',
    ], label='B1: Oracle Pseudo-Label Control')

    # D1: Ablations
    print_table(results, [
        'cifar100_g100_gradvax_full_s42',
        'cifar100_g100_gradvax_nodom_s42',
        'cifar100_g100_gradvax_noconflict_s42',
        'cifar100_g100_gradvax_noema_s42',
    ], label='D1: Component Ablation')

    # D2: τ sweep
    print_table(results, [
        'cifar100_g100_gradvax_tau2_s42',
        'cifar100_g100_gradvax_tau5_s42',
        'cifar100_g100_gradvax_tau10_s42',
        'cifar100_g100_gradvax_tauinf_s42',
    ], label='D2: τ Sweep')


if __name__ == '__main__':
    main()
