"""Final-only performance and pseudo-label diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _group_lookup(partition: dict[str, list[int]]) -> dict[int, str]:
    return {
        class_id: group
        for group, class_ids in partition.items()
        for class_id in class_ids
    }


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    partition: dict[str, list[int]],
    num_classes: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.float64)
    total = torch.zeros(num_classes, dtype=torch.float64)
    total_correct = 0
    total_seen = 0
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum())
        total_seen += labels.numel()
        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        for class_id in range(num_classes):
            mask = labels_cpu == class_id
            total[class_id] += int(mask.sum())
            correct[class_id] += int((predictions_cpu[mask] == class_id).sum())

    per_class = correct / total.clamp_min(1)
    group_values = {
        group: float(per_class[class_ids].mean())
        for group, class_ids in partition.items()
    }
    floor = 1.0 / (100.0 * num_classes)
    gm = math.exp(
        sum(math.log(max(float(value), floor)) for value in per_class) / num_classes
    )
    return {
        "overall_accuracy": total_correct / max(total_seen, 1),
        "balanced_accuracy": float(per_class.mean()),
        "head_accuracy": group_values["head"],
        "medium_accuracy": group_values["medium"],
        "tail_accuracy": group_values["tail"],
        "geometric_mean": gm,
        "worst_class_accuracy": float(per_class.min()),
        "per_class_accuracy": per_class.tolist(),
        "per_class_total": total.tolist(),
        "evaluation_examples": total_seen,
    }


@torch.no_grad()
def evaluate_pseudo_labels(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    partition: dict[str, list[int]],
    num_classes: int,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    lookup = _group_lookup(partition)
    tail = set(partition["tail"])
    head = set(partition["head"])
    accepted_predicted = torch.zeros(num_classes, dtype=torch.float64)
    accepted_predicted_correct = torch.zeros(num_classes, dtype=torch.float64)
    true_total = torch.zeros(num_classes, dtype=torch.float64)
    accepted_true = torch.zeros(num_classes, dtype=torch.float64)
    accepted_true_correct = torch.zeros(num_classes, dtype=torch.float64)
    head_composition = {
        group: {"accepted": 0, "errors": 0}
        for group in ("head", "medium", "tail")
    }
    known_examples = 0

    for images, labels, _ in loader:
        known = labels >= 0
        if not bool(known.any()):
            continue
        images = images[known].to(device, non_blocking=True)
        labels = labels[known].to(device, non_blocking=True)
        logits, _ = model(images)
        probabilities = F.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        accepted = confidence >= threshold
        known_examples += labels.numel()

        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        accepted_cpu = accepted.cpu()
        for class_id in range(num_classes):
            predicted_class = predictions_cpu == class_id
            true_class = labels_cpu == class_id
            accepted_predicted[class_id] += int((accepted_cpu & predicted_class).sum())
            accepted_predicted_correct[class_id] += int(
                (accepted_cpu & predicted_class & true_class).sum()
            )
            true_total[class_id] += int(true_class.sum())
            accepted_true[class_id] += int((accepted_cpu & true_class).sum())
            accepted_true_correct[class_id] += int(
                (accepted_cpu & true_class & predicted_class).sum()
            )

        for predicted, true, is_accepted in zip(
            predictions_cpu.tolist(), labels_cpu.tolist(), accepted_cpu.tolist()
        ):
            if is_accepted and predicted in head:
                true_group = lookup[true]
                head_composition[true_group]["accepted"] += 1
                if predicted != true:
                    head_composition[true_group]["errors"] += 1

    tail_ids = sorted(tail)
    tail_predicted = float(accepted_predicted[tail_ids].sum())
    tail_predicted_correct = float(accepted_predicted_correct[tail_ids].sum())
    tail_true = float(true_total[tail_ids].sum())
    tail_accepted_true = float(accepted_true[tail_ids].sum())
    tail_accepted_true_correct = float(accepted_true_correct[tail_ids].sum())

    per_class = {}
    for class_id in tail_ids:
        per_class[str(class_id)] = {
            "precision": float(
                accepted_predicted_correct[class_id]
                / accepted_predicted[class_id].clamp_min(1)
            ),
            "recall": float(
                accepted_true_correct[class_id] / true_total[class_id].clamp_min(1)
            ),
            "coverage": float(
                accepted_true[class_id] / true_total[class_id].clamp_min(1)
            ),
        }
    macro = {
        metric: sum(values[metric] for values in per_class.values())
        / max(len(per_class), 1)
        for metric in ("precision", "recall", "coverage")
    }
    return {
        "known_ground_truth_examples": known_examples,
        "tail_micro": {
            "precision": tail_predicted_correct / max(tail_predicted, 1.0),
            "recall": tail_accepted_true_correct / max(tail_true, 1.0),
            "coverage": tail_accepted_true / max(tail_true, 1.0),
        },
        "tail_macro": macro,
        "tail_per_class": per_class,
        "accepted_predicted_head_true_group_composition": head_composition,
    }
