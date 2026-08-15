"""Tail-anchor-gated score correction for TANGS v4.7 and v4.8."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def anchor_gated_logits(
    logits: torch.Tensor,
    features: torch.Tensor,
    labeled_counts: torch.Tensor,
    tail_classes: list[int],
    class_anchors: torch.Tensor | None,
    class_anchor_valid: torch.Tensor | None,
    *,
    base_alpha: float = 0.65,
    extra_tail_alpha: float = 0.25,
    anchor_threshold: float = 0.75,
    anchor_margin: float = 0.0,
    confidence_ceiling: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply prior correction plus one compatibility-gated tail boost.

    A supervised self-row gradient for tail class ``c`` has weight direction
    ``-(1-p_c) f``.  Consequently, the negative weight coordinates of its EMA
    anchor provide a class-conditional feature direction.  Every example may
    receive an additional prior correction for at most one tail class: the
    most compatible valid anchor, and only above ``anchor_threshold``.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes].")
    flat_features = features.flatten(1)
    if flat_features.shape[0] != logits.shape[0]:
        raise ValueError("features and logits must have the same batch size.")
    if labeled_counts.numel() != logits.shape[1]:
        raise ValueError("labeled_counts must contain one value per class.")
    if base_alpha < 0 or extra_tail_alpha < 0:
        raise ValueError("score-correction strengths must be non-negative.")
    if not -1.0 <= anchor_threshold <= 1.0:
        raise ValueError("anchor_threshold must be a cosine in [-1, 1].")
    if not 0.0 <= anchor_margin <= 2.0:
        raise ValueError("anchor_margin must be in [0, 2].")
    if not 0.0 <= confidence_ceiling <= 1.0:
        raise ValueError("confidence_ceiling must be in [0, 1].")

    counts = labeled_counts.to(device=logits.device, dtype=logits.dtype)
    if bool((counts <= 0).any()) or not bool(torch.isfinite(counts).all()):
        raise ValueError("labeled_counts must be finite and strictly positive.")
    log_prior = (counts / counts.sum()).log()
    adjusted = logits - base_alpha * log_prior.unsqueeze(0)

    if not tail_classes or class_anchors is None or class_anchor_valid is None:
        return adjusted
    anchors = class_anchors.to(device=logits.device, dtype=logits.dtype)
    valid = class_anchor_valid.to(device=logits.device, dtype=torch.bool)
    if anchors.ndim != 2 or anchors.shape[0] != len(tail_classes):
        raise ValueError("class_anchors must contain one row per tail class.")
    if anchors.shape[1] != flat_features.shape[1] + 1:
        raise ValueError("anchor weight coordinates do not match feature size.")
    finite = torch.isfinite(anchors).all(dim=1)
    weight_norm = torch.linalg.vector_norm(anchors[:, :-1], dim=1)
    valid = valid & finite & (weight_norm > eps)
    if not bool(valid.any()):
        return adjusted

    valid_indices = valid.nonzero().flatten()
    prototypes = F.normalize(-anchors[valid_indices, :-1], dim=1, eps=eps)
    similarities = F.normalize(flat_features, dim=1, eps=eps) @ prototypes.T
    top_k = min(2, similarities.shape[1])
    top_similarity, top_local = similarities.topk(k=top_k, dim=1)
    best_similarity = top_similarity[:, 0]
    best_local = top_local[:, 0]
    if top_k == 1:
        similarity_margin = torch.full_like(best_similarity, float("inf"))
    else:
        similarity_margin = top_similarity[:, 0] - top_similarity[:, 1]
    tail_ids = torch.tensor(tail_classes, dtype=torch.long, device=logits.device)
    best_class = tail_ids[valid_indices[best_local]]
    raw_confidence = F.softmax(logits, dim=1).max(dim=1).values
    eligible = best_similarity >= anchor_threshold
    eligible &= similarity_margin >= anchor_margin
    eligible &= raw_confidence <= confidence_ceiling
    if not bool(eligible.any()) or extra_tail_alpha == 0:
        return adjusted

    correction = torch.zeros_like(logits)
    rows = torch.arange(logits.shape[0], device=logits.device)[eligible]
    classes = best_class[eligible]
    correction[rows, classes] = extra_tail_alpha * -log_prior[classes]
    return adjusted + correction
