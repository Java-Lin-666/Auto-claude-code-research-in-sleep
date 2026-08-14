#!/usr/bin/env python3
"""Offline evaluation for predeclared EMA snapshots; never selects a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tangs.artifacts import atomic_json, canonical_json
from tangs.config import Protocol, RunConfig
from tangs.data import build_data
from tangs.metrics import evaluate_classifier, evaluate_pseudo_labels
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
    # Keep archived v4.5 checkpoints evaluable without redefining their saved
    # configuration hash. These fields did not exist before v4.6.
    values.setdefault("tangs_correction_rho", 1.0)
    values.setdefault("tailrow_observer", False)
    values["protocol"] = Protocol(**values["protocol"])
    values["data_root"] = data_root
    values["output_root"] = str(run_dir.parent)
    values["resume"] = None
    config = RunConfig(**values)
    return replace(config, device=device)


def evaluate(
    checkpoint_path: Path,
    config: RunConfig,
    run_dir: Path,
    data,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    performance = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
    )
    pseudo = None
    if data.pseudo_evaluation_loader is not None:
        pseudo = evaluate_pseudo_labels(
            model,
            data.pseudo_evaluation_loader,
            data.partition,
            config.protocol.num_classes,
            config.confidence_threshold,
            device,
        )
    step = int(checkpoint["step"])
    output = {
        "step": step,
        "checkpoint": str(checkpoint_path.resolve()),
        "evaluation_model": "EMA",
        "selection_role": "descriptive-offline-only",
        "performance": performance,
        "pseudo_labels": pseudo,
    }
    destination = run_dir / f"offline_evaluation_step{step:06d}.json"
    atomic_json(destination, output)
    print(
        f"step={step} bACC={100 * performance['balanced_accuracy']:.2f} "
        f"tail={100 * performance['tail_accuracy']:.2f} -> {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--all-snapshots", action="store_true")
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if bool(args.checkpoint) == bool(args.all_snapshots):
        parser.error("Choose exactly one of --checkpoint or --all-snapshots.")

    config = load_config(run_dir, args.data_root, args.device)
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        expected_split = json.load(handle)
    if data.split_manifest["split_hash"] != expected_split["split_hash"]:
        raise RuntimeError("Reconstructed dataset split hash does not match the run.")
    device = resolve_device(args.device)
    if args.checkpoint:
        checkpoints = [args.checkpoint.expanduser().resolve()]
    else:
        checkpoints = sorted(run_dir.glob("checkpoint_step*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No requested checkpoints were found.")
    for checkpoint in checkpoints:
        evaluate(checkpoint, config, run_dir, data, device)


if __name__ == "__main__":
    main()
