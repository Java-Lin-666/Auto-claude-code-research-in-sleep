#!/usr/bin/env python3
"""Retrospectively screen v4.7 on the archived v4.6 observer checkpoint.

This analysis can justify one fresh integrated 50k development run.  It cannot
PASS the v4.7 development gate or authorize any 250k confirmatory run because
the checkpoint predates the integrated v4.7 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from analyze_prior_shift import load_config
from tangs.artifacts import atomic_json
from tangs.calibration import anchor_gated_logits
from tangs.data import build_data
from tangs.metrics import evaluate_classifier
from tangs.trainer import resolve_device, seed_everything
from tangs.wrn import WRN


def _dead(performance: dict) -> int:
    return sum(value == 0 for value in performance["per_class_accuracy"])


def _deltas(candidate: dict, reference: dict) -> dict[str, float | int]:
    return {
        "balanced_accuracy_pp": 100
        * (candidate["balanced_accuracy"] - reference["balanced_accuracy"]),
        "head_accuracy_pp": 100
        * (candidate["head_accuracy"] - reference["head_accuracy"]),
        "medium_accuracy_pp": 100
        * (candidate["medium_accuracy"] - reference["medium_accuracy"]),
        "tail_accuracy_pp": 100
        * (candidate["tail_accuracy"] - reference["tail_accuracy"]),
        "geometric_mean_pp": 100
        * (candidate["geometric_mean"] - reference["geometric_mean"]),
        "dead_class_count": _dead(candidate) - _dead(reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config = load_config(run_dir, args.data_root, args.device)
    if config.mode != "development" or config.protocol_id != "P-C100-100":
        raise RuntimeError("The v4.7 freeze requires the P-C100-100 development run.")
    config = replace(config, workers=0)
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if data.split_manifest["split_hash"] != manifest["split_hash"]:
        raise RuntimeError("Reconstructed split does not match the observer artifact.")

    device = resolve_device(args.device)
    checkpoint_path = run_dir / "checkpoint_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    controller = checkpoint["controller"]
    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    counts = torch.tensor(manifest["labeled_counts"], device=device)

    raw = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
    )

    def uniform_la(logits: torch.Tensor, _features: torch.Tensor) -> torch.Tensor:
        prior = counts.to(dtype=logits.dtype) / counts.sum()
        return logits - 0.85 * prior.log().unsqueeze(0)

    def v47(logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return anchor_gated_logits(
            logits,
            features,
            counts,
            data.partition["tail"],
            controller["class_anchors"],
            controller["class_anchor_valid"],
            base_alpha=0.65,
            extra_tail_alpha=0.25,
            anchor_threshold=0.75,
        )

    la = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
        uniform_la,
    )
    tangs = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
        v47,
    )
    vs_raw = _deltas(tangs, raw)
    vs_la = _deltas(tangs, la)
    checks = {
        "bacc_vs_raw_at_least_1pp": vs_raw["balanced_accuracy_pp"] >= 1.0,
        "tail_vs_raw_at_least_2pp": vs_raw["tail_accuracy_pp"] >= 2.0,
        "head_vs_raw_no_worse_than_minus_2pp": vs_raw["head_accuracy_pp"] >= -2.0,
        "gm_vs_raw_nonnegative": vs_raw["geometric_mean_pp"] >= 0.0,
        "dead_classes_vs_raw_nonincreasing": vs_raw["dead_class_count"] <= 0,
        "bacc_vs_uniform_la_at_least_0_5pp": vs_la["balanced_accuracy_pp"] >= 0.5,
        "tail_vs_uniform_la_at_least_2pp": vs_la["tail_accuracy_pp"] >= 2.0,
    }
    candidate_supported = all(checks.values())
    report = {
        "version": "4.7-retrospective-candidate-screen",
        "status": "PROVISIONAL" if candidate_supported else "STOP",
        "allow_integrated_50k_validation": candidate_supported,
        "allow_single_c100_confirmatory_run": False,
        "allow_confirmatory_matrix": False,
        "selection_role": (
            "retrospective reuse of an archived v4.6 observer checkpoint; "
            "positive candidate signal only; official test unread"
        ),
        "reuse_note": (
            "All three rows reuse one unchanged FixMatch observer checkpoint; "
            "only the deterministic scoring rule differs. This is not a fresh "
            "tangs-v47 training run and cannot be labeled a v4.7 PASS."
        ),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "config_hash": checkpoint["config_hash"],
        "split_hash": manifest["split_hash"],
        "parameters": {
            "uniform_la_alpha": 0.85,
            "score_base_alpha": 0.65,
            "score_extra_tail_alpha": 0.25,
            "score_anchor_threshold": 0.75,
        },
        "raw_fixmatch": {**raw, "dead_class_count": _dead(raw)},
        "uniform_logit_adjustment": {**la, "dead_class_count": _dead(la)},
        "tangs_v47": {**tangs, "dead_class_count": _dead(tangs)},
        "tangs_v47_vs_raw": vs_raw,
        "tangs_v47_vs_uniform_la": vs_la,
        "checks": checks,
    }
    destination = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir.parent / "v47-development-selection.json"
    )
    atomic_json(destination, report)
    print(
        f"status={report['status']} "
        f"raw={100 * raw['balanced_accuracy']:.2f} "
        f"LA={100 * la['balanced_accuracy']:.2f} "
        f"TANGS-v47={100 * tangs['balanced_accuracy']:.2f} "
        f"tail={100 * tangs['tail_accuracy']:.2f} -> {destination}"
    )


if __name__ == "__main__":
    main()
