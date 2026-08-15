#!/usr/bin/env python3
"""Verify the frozen v4.8 scorer on two completed development checkpoints.

The result is retrospective and can only authorize a fresh seed-1 holdout
development run.  It can never authorize 250k confirmatory work directly.
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


PARAMETERS = {
    "score_uniform_la_alpha": 0.80,
    "score_base_alpha": 0.70,
    "score_extra_tail_alpha": 0.40,
    "score_anchor_threshold": 0.775,
    "score_anchor_margin": 0.05,
    "score_confidence_ceiling": 0.90,
}


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


def _checks(candidate: dict, raw: dict, uniform_la: dict) -> dict[str, bool]:
    vs_raw = _deltas(candidate, raw)
    vs_la = _deltas(candidate, uniform_la)
    return {
        "bacc_vs_raw_at_least_1pp": vs_raw["balanced_accuracy_pp"] >= 1.0,
        "tail_vs_raw_at_least_2pp": vs_raw["tail_accuracy_pp"] >= 2.0,
        "head_vs_raw_no_worse_than_minus_2pp": vs_raw["head_accuracy_pp"] >= -2.0,
        "gm_vs_raw_nonnegative": vs_raw["geometric_mean_pp"] >= 0.0,
        "dead_classes_vs_raw_nonincreasing": vs_raw["dead_class_count"] <= 0,
        "bacc_vs_uniform_la_at_least_0_5pp": vs_la["balanced_accuracy_pp"] >= 0.5,
        "tail_vs_uniform_la_at_least_2pp": vs_la["tail_accuracy_pp"] >= 2.0,
    }


def _evaluate(run_dir: Path, data_root: str, device_name: str) -> dict:
    config = replace(load_config(run_dir, data_root, device_name), workers=0)
    if config.mode != "development" or config.protocol_id != "P-C100-100":
        raise RuntimeError(f"Expected P-C100-100 development: {run_dir}")
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    if data.split_manifest["split_hash"] != split["split_hash"]:
        raise RuntimeError(f"Split reconstruction mismatch: {run_dir}")
    device = resolve_device(device_name)
    checkpoint = torch.load(
        run_dir / "checkpoint_last.pt", map_location=device, weights_only=False
    )
    if int(checkpoint["step"]) != 50_000:
        raise RuntimeError(f"Expected a completed 50k checkpoint: {run_dir}")
    controller = checkpoint["controller"]
    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    counts = torch.tensor(split["labeled_counts"], device=device)

    raw = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
    )

    def uniform_transform(logits: torch.Tensor, _features: torch.Tensor) -> torch.Tensor:
        prior = counts.to(dtype=logits.dtype) / counts.sum()
        return logits - PARAMETERS["score_uniform_la_alpha"] * prior.log().unsqueeze(0)

    def v48_transform(logits: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return anchor_gated_logits(
            logits,
            features,
            counts,
            data.partition["tail"],
            controller["class_anchors"],
            controller["class_anchor_valid"],
            base_alpha=PARAMETERS["score_base_alpha"],
            extra_tail_alpha=PARAMETERS["score_extra_tail_alpha"],
            anchor_threshold=PARAMETERS["score_anchor_threshold"],
            anchor_margin=PARAMETERS["score_anchor_margin"],
            confidence_ceiling=PARAMETERS["score_confidence_ceiling"],
        )

    uniform_la = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
        uniform_transform,
    )
    candidate = evaluate_classifier(
        model,
        data.evaluation_loader,
        data.partition,
        config.protocol.num_classes,
        device,
        v48_transform,
    )
    checks = _checks(candidate, raw, uniform_la)
    return {
        "run_dir": str(run_dir),
        "checkpoint_step": int(checkpoint["step"]),
        "config_hash": checkpoint["config_hash"],
        "split_hash": split["split_hash"],
        "raw_fixmatch": {**raw, "dead_class_count": _dead(raw)},
        "uniform_logit_adjustment": {
            **uniform_la,
            "dead_class_count": _dead(uniform_la),
        },
        "tangs_v48": {**candidate, "dead_class_count": _dead(candidate)},
        "tangs_v48_vs_raw": _deltas(candidate, raw),
        "tangs_v48_vs_uniform_la": _deltas(candidate, uniform_la),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-run-dir", type=Path, required=True)
    parser.add_argument("--archive-run-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {
        "fresh_seed0": _evaluate(
            args.fresh_run_dir.resolve(), args.data_root, args.device
        ),
        "archive_seed0": _evaluate(
            args.archive_run_dir.resolve(), args.data_root, args.device
        ),
    }
    supported = all(run["passed"] for run in runs.values())
    report = {
        "version": "4.8-retrospective-two-realization-freeze",
        "status": "PROVISIONAL" if supported else "STOP",
        "allow_seed1_holdout_50k": supported,
        "allow_confirmatory_250k": False,
        "allow_confirmatory_matrix": False,
        "selection_role": (
            "retrospective development freeze on two seed-0 training "
            "realizations; official test unread"
        ),
        "parameters": PARAMETERS,
        "runs": runs,
    }
    atomic_json(args.output.resolve(), report)
    print(
        f"status={report['status']} fresh={100 * runs['fresh_seed0']['tangs_v48']['balanced_accuracy']:.2f} "
        f"archive={100 * runs['archive_seed0']['tangs_v48']['balanced_accuracy']:.2f} "
        f"-> {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
