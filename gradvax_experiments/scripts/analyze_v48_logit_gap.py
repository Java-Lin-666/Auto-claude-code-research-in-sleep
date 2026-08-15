#!/usr/bin/env python3
"""Robust two-checkpoint search for a selective v4.8 score gate.

The script is development-only and read-only.  It searches the newest fresh
50k checkpoint, then validates shortlisted rules on the archived independent
50k FixMatch-observer realization.  No official test examples are read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from analyze_anchor_gated_prior import _collect
from analyze_prior_shift import load_config
from tangs.artifacts import atomic_json
from tangs.data import build_data
from tangs.trainer import resolve_device, seed_everything
from tangs.wrn import WRN


def _values(start: int, stop: int, divisor: float) -> list[float]:
    return [value / divisor for value in range(start, stop + 1)]


def _metrics_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    partition: dict[str, list[int]],
    num_classes: int,
) -> dict[str, Any]:
    total = torch.bincount(labels, minlength=num_classes).double()
    correct = torch.bincount(
        labels[predictions == labels], minlength=num_classes
    ).double()
    per_class = correct / total.clamp_min(1)
    groups = {
        name: float(per_class[class_ids].mean())
        for name, class_ids in partition.items()
    }
    floor = 1.0 / (100.0 * num_classes)
    geometric_mean = math.exp(
        sum(math.log(max(float(value), floor)) for value in per_class) / num_classes
    )
    return {
        "balanced_accuracy": float(per_class.mean()),
        "head_accuracy": groups["head"],
        "medium_accuracy": groups["medium"],
        "tail_accuracy": groups["tail"],
        "geometric_mean": geometric_mean,
        "worst_class_accuracy": float(per_class.min()),
        "dead_class_count": int((correct == 0).sum()),
    }


def _prepare(run_dir: Path, data_root: str, device_name: str) -> dict[str, Any]:
    config = load_config(run_dir, data_root, device_name)
    if config.mode != "development" or config.protocol_id != "P-C100-100":
        raise RuntimeError(f"Expected a P-C100-100 development run: {run_dir}")
    config = replace(config, workers=0)
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    if data.split_manifest["split_hash"] != split["split_hash"]:
        raise RuntimeError(f"Split reconstruction mismatch: {run_dir}")
    if split["evaluation_role"] != "development-balanced-unused-train":
        raise RuntimeError(f"Not a registered development evaluation: {run_dir}")

    device = resolve_device(device_name)
    checkpoint = torch.load(
        run_dir / "checkpoint_last.pt", map_location=device, weights_only=False
    )
    if int(checkpoint["step"]) != 50_000:
        raise RuntimeError(f"Expected a completed 50k checkpoint: {run_dir}")
    controller = checkpoint["controller"]
    anchors = controller["class_anchors"].detach().cpu().double()
    valid = controller["class_anchor_valid"].detach().cpu().bool()
    if not bool(valid.all()):
        raise RuntimeError(f"Missing tail anchors: {run_dir}")

    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    logits, features, labels = _collect(model, data.evaluation_loader, device)
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(-anchors[:, :-1], dim=1)
    similarities = features @ prototypes.T
    top_two_similarity, top_two_local = similarities.topk(k=2, dim=1)
    tail_ids = torch.tensor(controller["tail_classes"], dtype=torch.long)
    best_tail_class = tail_ids[top_two_local[:, 0]]
    labeled_counts = torch.tensor(split["labeled_counts"], dtype=logits.dtype)
    log_prior = (labeled_counts / labeled_counts.sum()).log()
    raw_confidence = F.softmax(logits, dim=1).max(dim=1).values
    raw_predictions = logits.argmax(dim=1)
    raw = _metrics_from_predictions(
        raw_predictions, labels, data.partition, config.protocol.num_classes
    )
    return {
        "run_dir": str(run_dir),
        "config_hash": checkpoint["config_hash"],
        "split_hash": split["split_hash"],
        "num_classes": config.protocol.num_classes,
        "partition": data.partition,
        "logits": logits,
        "labels": labels,
        "log_prior": log_prior,
        "best_tail_class": best_tail_class,
        "best_tail_similarity": top_two_similarity[:, 0],
        "tail_similarity_margin": top_two_similarity[:, 0] - top_two_similarity[:, 1],
        "raw_confidence": raw_confidence,
        "raw": raw,
    }


def _uniform_la(prepared: dict[str, Any], alpha: float) -> dict[str, Any]:
    predictions = (
        prepared["logits"] - alpha * prepared["log_prior"].unsqueeze(0)
    ).argmax(dim=1)
    return _metrics_from_predictions(
        predictions,
        prepared["labels"],
        prepared["partition"],
        prepared["num_classes"],
    )


def _evaluate(prepared: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    logits = prepared["logits"]
    base_logits = logits - candidate["base_alpha"] * prepared["log_prior"].unsqueeze(0)
    base_score, predictions = base_logits.max(dim=1)
    row_index = torch.arange(logits.shape[0])
    tail_class = prepared["best_tail_class"]
    tail_score = base_logits[row_index, tail_class]
    deficit = base_score - tail_score
    eligible = prepared["best_tail_similarity"] >= candidate["anchor_threshold"]
    eligible &= (
        prepared["tail_similarity_margin"] >= candidate["anchor_margin"]
    )
    if candidate["logit_gap"] is not None:
        eligible &= deficit <= candidate["logit_gap"]
    if candidate["confidence_ceiling"] is not None:
        eligible &= prepared["raw_confidence"] <= candidate["confidence_ceiling"]
    boost = candidate["extra_tail_alpha"] * -prepared["log_prior"][tail_class]
    flip = eligible & (tail_score + boost > base_score) & (predictions != tail_class)
    predictions = predictions.clone()
    predictions[flip] = tail_class[flip]
    metrics = _metrics_from_predictions(
        predictions,
        prepared["labels"],
        prepared["partition"],
        prepared["num_classes"],
    )
    return {
        "metrics": metrics,
        "eligible_fraction": float(eligible.double().mean()),
        "flipped_fraction": float(flip.double().mean()),
    }


def _deltas(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
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
        "dead_class_count": candidate["dead_class_count"]
        - reference["dead_class_count"],
    }


def _passes(
    metrics: dict[str, Any], raw: dict[str, Any], uniform_la: dict[str, Any]
) -> tuple[bool, dict[str, bool], dict[str, Any], dict[str, Any]]:
    vs_raw = _deltas(metrics, raw)
    vs_la = _deltas(metrics, uniform_la)
    checks = {
        "bacc_vs_raw_at_least_1pp": vs_raw["balanced_accuracy_pp"] >= 1.0,
        "tail_vs_raw_at_least_2pp": vs_raw["tail_accuracy_pp"] >= 2.0,
        "head_vs_raw_no_worse_than_minus_2pp": vs_raw["head_accuracy_pp"] >= -2.0,
        "gm_vs_raw_nonnegative": vs_raw["geometric_mean_pp"] >= 0.0,
        "dead_classes_vs_raw_nonincreasing": vs_raw["dead_class_count"] <= 0,
        "bacc_vs_uniform_la_at_least_0_5pp": vs_la["balanced_accuracy_pp"] >= 0.5,
        "tail_vs_uniform_la_at_least_2pp": vs_la["tail_accuracy_pp"] >= 2.0,
    }
    return all(checks.values()), checks, vs_raw, vs_la


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-run-dir", type=Path, required=True)
    parser.add_argument("--archive-run-dir", type=Path, required=True)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--uniform-la-alpha", type=float, default=0.80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fresh = _prepare(args.fresh_run_dir.resolve(), args.data_root, args.device)
    archive = _prepare(args.archive_run_dir.resolve(), args.data_root, args.device)
    if fresh["split_hash"] != archive["split_hash"]:
        raise RuntimeError("The robust search requires the same registered split.")
    prepared = {"fresh": fresh, "archive": archive}
    uniform_la = {
        name: _uniform_la(run, args.uniform_la_alpha)
        for name, run in prepared.items()
    }
    for name, run in prepared.items():
        if uniform_la[name]["head_accuracy"] < run["raw"]["head_accuracy"] - 0.02:
            raise RuntimeError(f"Uniform-LA comparator violates the Head guard: {name}")

    # A deliberately compact preregistered grid.  If it produces no robust
    # candidate, do not manufacture one through an unbounded fine search.
    bases = _values(9, 16, 20.0)  # 0.45 .. 0.80, step 0.05
    extras = _values(2, 8, 20.0)  # 0.10 .. 0.40, step 0.05
    anchor_thresholds = _values(28, 33, 40.0)  # 0.70 .. 0.825, step 0.025
    anchor_margins = [0.0, 0.05]
    logit_gaps: list[float | None] = [0.50, 1.0, 2.0, None]
    confidence_ceilings: list[float | None] = [0.80, 0.90, None]

    fresh_shortlist: list[dict[str, Any]] = []
    for base_alpha in bases:
        for extra_tail_alpha in extras:
            for anchor_threshold in anchor_thresholds:
                for anchor_margin in anchor_margins:
                    for logit_gap in logit_gaps:
                        for confidence_ceiling in confidence_ceilings:
                            candidate = {
                                "base_alpha": base_alpha,
                                "extra_tail_alpha": extra_tail_alpha,
                                "anchor_threshold": anchor_threshold,
                                "anchor_margin": anchor_margin,
                                "logit_gap": logit_gap,
                                "confidence_ceiling": confidence_ceiling,
                            }
                            evaluated = _evaluate(fresh, candidate)
                            passed, checks, vs_raw, vs_la = _passes(
                                evaluated["metrics"], fresh["raw"], uniform_la["fresh"]
                            )
                            if passed:
                                fresh_shortlist.append(
                                    {
                                        "parameters": candidate,
                                        "fresh": {
                                            **evaluated,
                                            "checks": checks,
                                            "vs_raw": vs_raw,
                                            "vs_uniform_la": vs_la,
                                        },
                                    }
                                )

    robust_rows: list[dict[str, Any]] = []
    for row in fresh_shortlist:
        evaluated = _evaluate(archive, row["parameters"])
        passed, checks, vs_raw, vs_la = _passes(
            evaluated["metrics"], archive["raw"], uniform_la["archive"]
        )
        row["archive"] = {
            **evaluated,
            "checks": checks,
            "vs_raw": vs_raw,
            "vs_uniform_la": vs_la,
        }
        row["robust_pass"] = passed
        row["minimum_bacc_margin_vs_uniform_la_pp"] = min(
            row["fresh"]["vs_uniform_la"]["balanced_accuracy_pp"],
            row["archive"]["vs_uniform_la"]["balanced_accuracy_pp"],
        )
        row["minimum_head_slack_pp"] = min(
            row["fresh"]["vs_raw"]["head_accuracy_pp"] + 2.0,
            row["archive"]["vs_raw"]["head_accuracy_pp"] + 2.0,
        )
        if passed:
            robust_rows.append(row)

    robust_rows.sort(
        key=lambda row: (
            row["minimum_bacc_margin_vs_uniform_la_pp"],
            row["minimum_head_slack_pp"],
            min(
                row["fresh"]["vs_uniform_la"]["tail_accuracy_pp"],
                row["archive"]["vs_uniform_la"]["tail_accuracy_pp"],
            ),
        ),
        reverse=True,
    )
    report = {
        "version": "4.8-logit-gap-exploratory-robust-search",
        "analysis_role": "development-only; official test unread",
        "status": "ROBUST-CANDIDATE" if robust_rows else "NO-ROBUST-CANDIDATE",
        "selection_note": (
            "A candidate must pass every original efficacy threshold on both "
            "the fresh and archived 50k development realizations."
        ),
        "uniform_la_alpha": args.uniform_la_alpha,
        "runs": {
            name: {
                "run_dir": run["run_dir"],
                "config_hash": run["config_hash"],
                "split_hash": run["split_hash"],
                "raw": run["raw"],
                "uniform_la": uniform_la[name],
            }
            for name, run in prepared.items()
        },
        "grid": {
            "base_alpha": bases,
            "extra_tail_alpha": extras,
            "anchor_threshold": anchor_thresholds,
            "anchor_margin": anchor_margins,
            "logit_gap": logit_gaps,
            "confidence_ceiling": confidence_ceilings,
            "candidate_count": len(bases)
            * len(extras)
            * len(anchor_thresholds)
            * len(anchor_margins)
            * len(logit_gaps)
            * len(confidence_ceilings),
        },
        "fresh_pass_count": len(fresh_shortlist),
        "robust_pass_count": len(robust_rows),
        "selected": robust_rows[0] if robust_rows else None,
        "top_robust_candidates": robust_rows[:100],
    }
    atomic_json(args.output.resolve(), report)
    print(
        f"status={report['status']} candidates={report['grid']['candidate_count']} "
        f"fresh_pass={len(fresh_shortlist)} robust_pass={len(robust_rows)}"
    )
    if robust_rows:
        selected = robust_rows[0]
        print(json.dumps(selected, indent=2, sort_keys=True))
    print(f"-> {args.output.resolve()}")


if __name__ == "__main__":
    main()
