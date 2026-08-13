"""Pure split utilities mirroring the released CDMAD loaders."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence


def make_imb_data(max_num: int, class_num: int, gamma: float) -> list[int]:
    if max_num <= 0 or class_num < 2 or gamma < 1:
        raise ValueError("Invalid long-tail parameters.")
    mu = math.pow(1.0 / gamma, 1.0 / (class_num - 1))
    counts = []
    for class_id in range(class_num):
        if class_id == class_num - 1:
            counts.append(int(max_num / gamma))
        else:
            counts.append(int(max_num * math.pow(mu, class_id)))
    return counts


def class_partition(class_counts: Sequence[int]) -> dict[str, list[int]]:
    """Frequency terciles with deterministic class-id tie breaking."""
    ordered = sorted(
        range(len(class_counts)), key=lambda class_id: (-class_counts[class_id], class_id)
    )
    edge = len(ordered) // 3
    return {
        "head": ordered[:edge],
        "medium": ordered[edge : len(ordered) - edge],
        "tail": ordered[len(ordered) - edge :],
    }


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return len(set(left).intersection(right))
