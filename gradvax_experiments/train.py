"""Main training entry point for GradVax experiments."""
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import torchvision.models as tv_models

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.lt_cifar import make_lt_dataset
from models.gradvax import GradVax
from models.trainer import FixMatchTrainer


def build_model(num_classes, arch='wrn28_2'):
    if arch == 'wrn28_2':
        from models.wrn import WideResNet
        return WideResNet(depth=28, widen_factor=2, num_classes=num_classes)
    raise ValueError(f"Unknown arch: {arch}")


def get_classifier(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Linear):
            return m
    raise RuntimeError("No linear layer found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    p.add_argument('--gamma', type=float, default=100)
    p.add_argument('--n_labeled', type=int, default=500)
    p.add_argument('--gradvax', action='store_true')
    p.add_argument('--oracle', action='store_true', help='Use ground-truth pseudo-labels (Phase B1)')
    p.add_argument('--beta', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=5.0)
    p.add_argument('--e_warm', type=int, default=2500)
    # Ablation flags
    p.add_argument('--no_conflict', action='store_true')
    p.add_argument('--no_domination', action='store_true')
    p.add_argument('--no_ema', action='store_true')
    p.add_argument('--head_downweight', type=float, default=1.0,
                   help='Weight multiplier for head-class pseudo-label loss (C2 baseline, e.g. 0.5)')
    p.add_argument('--grad_clip', type=float, default=0.0,
                   help='Gradient clipping max norm (0 = disabled, C2 baseline)')
    p.add_argument('--total_steps', type=int, default=131072)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--mu', type=int, default=7, help='Unlabeled batch multiplier')
    p.add_argument('--lr', type=float, default=0.03)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--threshold', type=float, default=0.95)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_root', default='./data/cache')
    p.add_argument('--log_dir', default='./results')
    p.add_argument('--run_id', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    num_classes = 10 if args.dataset == 'cifar10' else 100

    # Build run_id
    if args.run_id is None:
        tag = 'gradvax' if args.gradvax else 'baseline'
        if args.oracle:
            tag += '_oracle'
        if args.no_conflict:
            tag += '_noconflict'
        if args.no_domination:
            tag += '_nodom'
        if args.no_ema:
            tag += '_noema'
        args.run_id = f"{args.dataset}_g{int(args.gamma)}_{tag}_s{args.seed}"

    print(f"Run: {args.run_id}")

    # Datasets
    labeled_ds, partition = make_lt_dataset(
        args.data_root, args.dataset, args.gamma, args.n_labeled, 'labeled', args.seed)
    unlabeled_ds, _ = make_lt_dataset(
        args.data_root, args.dataset, args.gamma, args.n_labeled, 'unlabeled', args.seed)
    test_ds, _ = make_lt_dataset(
        args.data_root, args.dataset, args.gamma, args.n_labeled, 'test', args.seed)

    # Save partition once
    os.makedirs(args.log_dir, exist_ok=True)
    partition_path = os.path.join(args.log_dir, 'class_partition.json')
    if not os.path.exists(partition_path):
        with open(partition_path, 'w') as f:
            json.dump(partition, f, indent=2)
        print(f"Saved class partition → {partition_path}")
        print(f"  Head: {partition['head'][:5]}... ({len(partition['head'])} classes)")
        print(f"  Medium: {partition['medium'][:5]}... ({len(partition['medium'])} classes)")
        print(f"  Tail: {partition['tail'][:5]}... ({len(partition['tail'])} classes)")

    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds,
                                  batch_size=args.batch_size * args.mu,
                                  shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    # Model
    model = build_model(num_classes)
    classifier = get_classifier(model)

    # Optimizer + cosine scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)

    # GradVax
    gv = None
    if args.gradvax:
        gv = GradVax(classifier, num_classes, partition,
                     beta=args.beta, tau=args.tau, e_warm=args.e_warm)
        if args.no_conflict:
            gv._no_conflict = True
        if args.no_domination:
            gv._no_domination = True
        if args.no_ema:
            gv._no_ema = True

    trainer = FixMatchTrainer(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        test_loader=test_loader,
        partition=partition,
        optimizer=optimizer,
        scheduler=scheduler,
        threshold=args.threshold,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        gradvax=gv,
        oracle_labels=args.oracle,
        head_downweight=args.head_downweight,
        grad_clip=args.grad_clip,
        device=args.device,
        log_dir=args.log_dir,
        run_id=args.run_id,
    )

    results = trainer.train()
    best = max(results, key=lambda r: r['tail'])
    print(f"\nBest tail acc: {best['tail']:.4f} at step {best['step']}")
    print(f"Results saved to {args.log_dir}/{args.run_id}_results.json")


if __name__ == '__main__':
    main()
