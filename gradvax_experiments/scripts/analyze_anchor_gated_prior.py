#!/usr/bin/env python3
"""Explore tail-anchor-gated prior correction on a development checkpoint.

The v4.6 observer stores each supervised tail self-row gradient anchor.  Its
negative weight coordinates provide a class-conditional feature direction.
This script uses that direction only to choose which single tail class may
receive an additional prior correction; it never changes the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from analyze_prior_shift import _metrics, load_config
from tangs.artifacts import atomic_json
from tangs.data import build_data
from tangs.trainer import resolve_device, seed_everything
from tangs.wrn import WRN


@torch.no_grad()
def _collect(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for images, labels, _ in loader:
        logits, _, features = model(
            images.to(device, non_blocking=True), return_feature=True
        )
        all_logits.append(logits.detach().cpu().double())
        all_features.append(features.flatten(1).detach().cpu().double())
        all_labels.append(labels.cpu())
    return (
        torch.cat(all_logits),
        torch.cat(all_features),
        torch.cat(all_labels),
    )


def _pseudo_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tail_ids: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    confidence, predictions = F.softmax(logits, dim=1).max(dim=1)
    accepted = confidence >= threshold
    true_tail = (labels.unsqueeze(1) == tail_ids.unsqueeze(0)).any(dim=1)
    predicted_tail = (predictions.unsqueeze(1) == tail_ids.unsqueeze(0)).any(dim=1)
    correct = predictions == labels
    accepted_predicted_tail = accepted & predicted_tail
    return {
        "accepted_count": int(accepted.sum()),
        "accepted_tail_prediction_count": int(accepted_predicted_tail.sum()),
        "tail_precision": float(
            correct[accepted_predicted_tail].double().mean()
            if bool(accepted_predicted_tail.any())
            else 0.0
        ),
        "tail_recall": float((accepted & correct & true_tail).sum() / true_tail.sum()),
        "tail_coverage": float((accepted & true_tail).sum() / true_tail.sum()),
    }


def _apply_row(
    logits: torch.Tensor,
    features: torch.Tensor,
    prototypes: torch.Tensor,
    tail_ids: torch.Tensor,
    non_tail_prototypes: torch.Tensor,
    log_prior: torch.Tensor,
    row: dict,
) -> torch.Tensor:
    similarities = F.normalize(features, dim=1) @ prototypes.T
    max_similarity, best_local = similarities.max(dim=1)
    best_class = tail_ids[best_local]
    threshold = row["gate_threshold"]
    if threshold is None:
        eligible = torch.ones_like(max_similarity, dtype=torch.bool)
    elif row["gate_type"] == "absolute_tail_similarity":
        eligible = max_similarity >= float(threshold)
    elif row["gate_type"] == "tail_vs_nontail_margin":
        max_non_tail = (F.normalize(features, dim=1) @ non_tail_prototypes.T).max(
            dim=1
        ).values
        eligible = max_similarity - max_non_tail >= float(threshold)
    else:
        raise ValueError(f"Unknown gate type: {row['gate_type']}")
    adjusted = logits - float(row["base_alpha"]) * log_prior.unsqueeze(0)
    indices = torch.arange(logits.shape[0])[eligible]
    classes = best_class[eligible]
    adjusted[indices, classes] += (
        float(row["extra_tail_alpha"]) * -log_prior[classes]
    )
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--head-drop-limit-pp", type=float, default=2.0)
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
        raise RuntimeError("Exploratory parameter selection is development-only.")
    config = replace(config, workers=0)
    seed_everything(config.manual_seed)
    data = build_data(config)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        expected_split = json.load(handle)
    if data.split_manifest["split_hash"] != expected_split["split_hash"]:
        raise RuntimeError("Reconstructed dataset split hash does not match the run.")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    controller = checkpoint["controller"]
    tail_classes = [int(class_id) for class_id in controller["tail_classes"]]
    anchors = controller["class_anchors"].detach().cpu().double()
    valid = controller["class_anchor_valid"].detach().cpu().bool()
    if not bool(valid.all()):
        missing = [tail_classes[index] for index in (~valid).nonzero().flatten()]
        raise RuntimeError(f"Missing tail anchors for classes: {missing}")

    model = WRN(2, num_classes=config.protocol.num_classes).to(device)
    model.load_state_dict(checkpoint["ema_model"])
    logits, features, labels = _collect(model, data.evaluation_loader, device)
    if anchors.shape[1] != features.shape[1] + 1:
        raise RuntimeError("Tail anchor and EMA feature dimensions do not match.")
    tail_ids = torch.tensor(tail_classes, dtype=torch.long)

    if data.pseudo_evaluation_loader is None:
        raise RuntimeError("The run did not construct a pseudo-evaluation loader.")
    pseudo_logits, pseudo_features, pseudo_true_labels = _collect(
        model, data.pseudo_evaluation_loader, device
    )
    pseudo_confidence, pseudo_predictions = F.softmax(pseudo_logits, dim=1).max(dim=1)
    pseudo_accepted = pseudo_confidence >= config.confidence_threshold
    normalized_pseudo_features = F.normalize(pseudo_features, dim=1)

    gradient_prototypes = F.normalize(-anchors[:, :-1], dim=1)
    pseudo_prototypes = gradient_prototypes.clone()
    pseudo_counts = []
    pseudo_correct = 0
    pseudo_selected = 0
    for local_index, class_id in enumerate(tail_classes):
        selected = pseudo_accepted & (pseudo_predictions == class_id)
        count = int(selected.sum())
        pseudo_counts.append(count)
        if count:
            pseudo_prototypes[local_index] = F.normalize(
                normalized_pseudo_features[selected].mean(dim=0), dim=0
            )
            pseudo_correct += int((pseudo_true_labels[selected] == class_id).sum())
            pseudo_selected += count
    blended_prototypes = F.normalize(
        gradient_prototypes + pseudo_prototypes, dim=1
    )
    prototype_sets = {
        "gradient": gradient_prototypes,
        "pseudo": pseudo_prototypes,
        "gradient_pseudo_blend": blended_prototypes,
    }
    non_tail_classes = data.partition["head"] + data.partition["medium"]
    non_tail_ids = torch.tensor(non_tail_classes, dtype=torch.long)
    non_tail_prototypes = F.normalize(
        model.output.weight.detach().cpu().double()[non_tail_ids], dim=1
    )

    labeled_counts = torch.tensor(
        expected_split["labeled_counts"], dtype=logits.dtype
    )
    log_prior = (labeled_counts / labeled_counts.sum()).log()
    raw_metrics = _metrics(
        logits, labels, data.partition, config.protocol.num_classes
    )
    head_floor = raw_metrics["head_accuracy"] - args.head_drop_limit_pp / 100.0

    absolute_thresholds = [float("-inf")] + [
        value / 20.0 for value in range(4, 17)
    ]
    margin_thresholds = [value / 40.0 for value in range(-12, 13)]
    rows = []
    batch_index = torch.arange(labels.numel())
    prototype_diagnostics = {}
    for anchor_source, prototypes in prototype_sets.items():
        similarities = F.normalize(features, dim=1) @ prototypes.T
        max_tail_similarity, best_tail_local = similarities.max(dim=1)
        best_tail_class = tail_ids[best_tail_local]
        max_non_tail_similarity = (
            F.normalize(features, dim=1) @ non_tail_prototypes.T
        ).max(dim=1).values
        similarity_quantiles = {
            f"q{int(100 * quantile):02d}": float(
                torch.quantile(max_tail_similarity, quantile)
            )
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        }
        tail_mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in tail_classes:
            tail_mask |= labels == class_id
        nearest_anchor_accuracy = float(
            (best_tail_class[tail_mask] == labels[tail_mask]).double().mean()
        )
        prototype_diagnostics[anchor_source] = {
            "max_tail_similarity_quantiles": similarity_quantiles,
            "tail_nearest_anchor_accuracy": nearest_anchor_accuracy,
        }
        gates = [
            ("absolute_tail_similarity", threshold)
            for threshold in absolute_thresholds
        ]
        if anchor_source == "gradient":
            gates.extend(
                ("tail_vs_nontail_margin", threshold)
                for threshold in margin_thresholds
            )
        for base_index in range(11):
            base_alpha = 0.5 + base_index / 20.0
            base_logits = logits - base_alpha * log_prior.unsqueeze(0)
            for extra_index in range(21):
                extra_alpha = extra_index / 20.0
                for gate_type, threshold in gates:
                    if gate_type == "absolute_tail_similarity":
                        eligible = max_tail_similarity >= threshold
                    else:
                        eligible = (
                            max_tail_similarity - max_non_tail_similarity
                            >= threshold
                        )
                    adjusted = base_logits.clone()
                    eligible_rows = batch_index[eligible]
                    eligible_classes = best_tail_class[eligible]
                    adjusted[eligible_rows, eligible_classes] += (
                        extra_alpha * -log_prior[eligible_classes]
                    )
                    rows.append(
                        {
                            "anchor_source": anchor_source,
                            "gate_type": gate_type,
                            "base_alpha": base_alpha,
                            "extra_tail_alpha": extra_alpha,
                            "gate_threshold": (
                                None if math.isinf(threshold) else threshold
                            ),
                            "eligible_fraction": float(eligible.double().mean()),
                            "metrics": _metrics(
                                adjusted,
                                labels,
                                data.partition,
                                config.protocol.num_classes,
                            ),
                        }
                    )

    best = max(rows, key=lambda row: row["metrics"]["balanced_accuracy"])
    feasible = [
        row for row in rows if row["metrics"]["head_accuracy"] >= head_floor
    ]
    best_constrained = max(
        feasible, key=lambda row: row["metrics"]["balanced_accuracy"]
    )
    raw_pseudo_metrics = _pseudo_metrics(
        pseudo_logits,
        pseudo_true_labels,
        tail_ids,
        config.confidence_threshold,
    )
    constrained_pseudo_logits = _apply_row(
        pseudo_logits,
        pseudo_features,
        prototype_sets[best_constrained["anchor_source"]],
        tail_ids,
        non_tail_prototypes,
        log_prior,
        best_constrained,
    )
    constrained_pseudo_metrics = _pseudo_metrics(
        constrained_pseudo_logits,
        pseudo_true_labels,
        tail_ids,
        config.confidence_threshold,
    )
    output = {
        "analysis_role": "exploratory-development-only",
        "formula": (
            "z' = z - base_alpha*log(prior); then add "
            "extra_tail_alpha*(-log(prior_c)) only to the tail class whose "
            "negative supervised gradient anchor is most feature-compatible"
        ),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "config_hash": checkpoint["config_hash"],
        "split_hash": expected_split["split_hash"],
        "head_drop_limit_pp": args.head_drop_limit_pp,
        "raw": raw_metrics,
        "prototype_diagnostics": prototype_diagnostics,
        "accepted_tail_pseudo_count_by_class": pseudo_counts,
        "accepted_tail_pseudo_precision": (
            pseudo_correct / max(pseudo_selected, 1)
        ),
        "best": best,
        "best_head_constrained": best_constrained,
        "raw_pseudo_labels": raw_pseudo_metrics,
        "head_constrained_pseudo_labels": constrained_pseudo_metrics,
        "sweep": rows,
    }
    destination = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir / "exploratory_anchor_gated_prior.json"
    )
    atomic_json(destination, output)
    for label, row in (("best", best), ("constrained", best_constrained)):
        metrics = row["metrics"]
        print(
            f"{label}: base={row['base_alpha']:.2f} "
            f"source={row['anchor_source']} "
            f"extra={row['extra_tail_alpha']:.2f} "
            f"gate={row['gate_type']} "
            f"threshold={row['gate_threshold']} "
            f"eligible={100 * row['eligible_fraction']:.2f}% "
            f"bACC={100 * metrics['balanced_accuracy']:.2f} "
            f"head={100 * metrics['head_accuracy']:.2f} "
            f"medium={100 * metrics['medium_accuracy']:.2f} "
            f"tail={100 * metrics['tail_accuracy']:.2f} "
            f"GM={100 * metrics['geometric_mean']:.2f} "
            f"dead={metrics['dead_class_count']}"
        )
    print(
        "tail_nearest_anchor_accuracy="
        + ",".join(
            f"{source}:{100 * values['tail_nearest_anchor_accuracy']:.2f}"
            for source, values in prototype_diagnostics.items()
        )
        + f" pseudo_precision={100 * pseudo_correct / max(pseudo_selected, 1):.2f}"
        + f" -> {destination}"
    )


if __name__ == "__main__":
    main()
