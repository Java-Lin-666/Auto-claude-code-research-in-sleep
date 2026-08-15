#!/usr/bin/env python3
"""Exploratory post-hoc prior-shift analysis on a development checkpoint.

This script never changes a checkpoint and must not be used to tune on the
confirmatory test set.  It evaluates z_c - alpha * log(pi_c), where pi is the
declared long-tailed training prior, on the run's reconstructed development
split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tangs.artifacts import atomic_json, canonical_json
from tangs.config import Protocol, RunConfig
from tangs.data import build_data
from tangs.trainer import resolve_device, seed_everything
from tangs.wrn import WRN


def load_config(run_dir: Path, data_root: str, device: str) -> RunConfig:
    with (run_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        saved = json.load(handle)
    scientific = saved["scientific_config"]
    digest = hashlib.sha256(canonical_json(scientific).encode()).hexdigest()
    if digest != saved["config_hash"]:
        raise RuntimeError("Saved scientific configuration hash does not verify.")
    values = dict(scientific)
    if isinstance(values["tangs_tau"], str):
        values["tangs_tau"] = float(values["tangs_tau"])
    values.setdefault("tangs_correction_rho", 1.0)
    values.setdefault("tailrow_observer", False)
    values.setdefault("score_uniform_la_alpha", 0.85)
    values.setdefault("score_base_alpha", 0.65)
    values.setdefault("score_extra_tail_alpha", 0.25)
    values.setdefault("score_anchor_threshold", 0.75)
    values["protocol"] = Protocol(**values["protocol"])
    values["data_root"] = data_root
    values["output_root"] = str(run_dir.parent)
    values["resume"] = None
    return replace(RunConfig(**values), device=device)


def _parse_alphas(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one alpha is required.")
    if len(values) != len(set(values)):
        raise ValueError("Alpha values must be unique.")
    return values


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for images, targets, _ in loader:
        outputs, _ = model(images.to(device, non_blocking=True))
        logits.append(outputs.detach().cpu().double())
        labels.append(targets.cpu())
    return torch.cat(logits), torch.cat(labels)


def _metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    partition: dict[str, list[int]],
    num_classes: int,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    total = torch.bincount(labels, minlength=num_classes).double()
    correct = torch.bincount(
        labels[predictions == labels], minlength=num_classes
    ).double()
    per_class = correct / total.clamp_min(1)
    group_values = {
        group: float(per_class[class_ids].mean())
        for group, class_ids in partition.items()
    }
    floor = 1.0 / (100.0 * num_classes)
    geometric_mean = math.exp(
        sum(math.log(max(float(value), floor)) for value in per_class) / num_classes
    )
    predicted_counts = torch.bincount(predictions, minlength=num_classes).double()
    predicted_group_fraction = {
        group: float(predicted_counts[class_ids].sum() / labels.numel())
        for group, class_ids in partition.items()
    }
    return {
        "balanced_accuracy": float(per_class.mean()),
        "head_accuracy": group_values["head"],
        "medium_accuracy": group_values["medium"],
        "tail_accuracy": group_values["tail"],
        "geometric_mean": geometric_mean,
        "worst_class_accuracy": float(per_class.min()),
        "dead_class_count": int((correct == 0).sum()),
        "predicted_group_fraction": predicted_group_fraction,
        "per_class_accuracy": per_class.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--alphas",
        default=",".join(f"{value / 20:.2f}" for value in range(31)),
        help="Comma-separated prior correction strengths (default: 0.00..1.50).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else run_dir / "checkpoint_last.pt"
    )
    config = load_config(run_dir, args.data_root, args.device)
    if config.mode != "development":
        raise RuntimeError(
            "Prior strength may only be explored on a development run, never test."
        )
    # This is inference-only. A single worker avoids Windows spawn/pipe
    # restrictions without changing any scientific quantity.
    config = replace(config, workers=0)
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        expected_split = json.load(handle)
    if data.split_manifest["split_hash"] != expected_split["split_hash"]:
        raise RuntimeError("Reconstructed dataset split hash does not match the run.")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    logits, labels = _collect_logits(model, data.evaluation_loader, device)

    labeled_counts = torch.tensor(
        expected_split["labeled_counts"], dtype=logits.dtype
    )
    prior = labeled_counts / labeled_counts.sum()
    log_prior = prior.clamp_min(torch.finfo(logits.dtype).tiny).log()
    rows = []
    for alpha in _parse_alphas(args.alphas):
        adjusted_logits = logits - alpha * log_prior.unsqueeze(0)
        rows.append(
            {
                "alpha": alpha,
                "metrics": _metrics(
                    adjusted_logits,
                    labels,
                    data.partition,
                    config.protocol.num_classes,
                ),
            }
        )
    best = max(rows, key=lambda row: row["metrics"]["balanced_accuracy"])
    output = {
        "analysis_role": "exploratory-development-only",
        "selection_rule": "maximum balanced_accuracy on the disjoint development set",
        "formula": "adjusted_logit_c = raw_logit_c - alpha * log(labeled_prior_c)",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "config_hash": checkpoint["config_hash"],
        "split_hash": expected_split["split_hash"],
        "evaluation_role": expected_split["evaluation_role"],
        "labeled_counts": labeled_counts.tolist(),
        "best": best,
        "sweep": rows,
    }
    destination = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "exploratory_prior_shift.json"
    )
    atomic_json(destination, output)
    metrics = best["metrics"]
    print(
        f"best_alpha={best['alpha']:.2f} "
        f"bACC={100 * metrics['balanced_accuracy']:.2f} "
        f"head={100 * metrics['head_accuracy']:.2f} "
        f"medium={100 * metrics['medium_accuracy']:.2f} "
        f"tail={100 * metrics['tail_accuracy']:.2f} "
        f"GM={100 * metrics['geometric_mean']:.2f} "
        f"dead={metrics['dead_class_count']} -> {destination}"
    )


if __name__ == "__main__":
    main()
