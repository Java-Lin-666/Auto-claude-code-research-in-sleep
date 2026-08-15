"""Classifier-only TANGS gradient measurement and replacement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


def flatten_tensors(
    tensors: Iterable[torch.Tensor | None], parameters: Iterable[torch.Tensor]
) -> torch.Tensor:
    parts = []
    for tensor, parameter in zip(tensors, parameters):
        if tensor is None:
            parts.append(torch.zeros_like(parameter).reshape(-1))
        else:
            parts.append(tensor.reshape(-1))
    return torch.cat(parts)


def classifier_gradient(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return flatten_tensors(gradients, parameters).float()


def read_classifier_gradient(
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    return flatten_tensors([parameter.grad for parameter in parameters], parameters).float()


def write_classifier_gradient(
    parameters: list[torch.nn.Parameter], flat_gradient: torch.Tensor
) -> None:
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        value = flat_gradient[offset : offset + size].reshape_as(parameter)
        if parameter.grad is None:
            parameter.grad = value.clone()
        else:
            parameter.grad.copy_(value)
        offset += size
    if offset != flat_gradient.numel():
        raise ValueError("Flat classifier gradient has an unexpected size.")


def _valid(vector: torch.Tensor | None, eps: float) -> bool:
    return (
        vector is not None
        and bool(torch.isfinite(vector).all())
        and float(torch.linalg.vector_norm(vector)) > eps
    )


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / (denominator + eps))


@dataclass
class SurgeryResult:
    gradient: torch.Tensor
    applied: bool
    reason: str
    stats: dict[str, float | int | bool | str]


class TangsController:
    """Maintain the tail anchor and transform the exact predicted-head contribution."""

    def __init__(
        self,
        method: str,
        beta: float = 0.99,
        tau: float = 5.0,
        warmup_steps: int = 2500,
        eps: float = 1e-12,
    ):
        self.method = method
        self.beta = beta
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.anchor: torch.Tensor | None = None
        self.anchor_age = 0
        self.anchor_updates = 0
        self.invalid_anchor_updates = 0

    @property
    def uses_surgery(self) -> bool:
        return self.method in {
            "tangs",
            "oracle-tangs",
            "head-clip",
            "pcgrad",
            "no-cap",
            "no-projection",
            "instant-anchor",
        }

    def update_anchor(self, tail_gradient: torch.Tensor | None) -> bool:
        if not _valid(tail_gradient, self.eps):
            self.anchor_age += 1
            if tail_gradient is not None:
                self.invalid_anchor_updates += 1
            return False
        detached = tail_gradient.detach().float()
        if self.anchor is None:
            self.anchor = detached.clone()
        else:
            self.anchor.mul_(self.beta).add_(detached, alpha=1.0 - self.beta)
        self.anchor_age = 0
        self.anchor_updates += 1
        return True

    def state_dict(self) -> dict:
        return {
            "method": self.method,
            "beta": self.beta,
            "tau": self.tau,
            "warmup_steps": self.warmup_steps,
            "eps": self.eps,
            "anchor": self.anchor,
            "anchor_age": self.anchor_age,
            "anchor_updates": self.anchor_updates,
            "invalid_anchor_updates": self.invalid_anchor_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        for field in ("method", "beta", "tau", "warmup_steps", "eps"):
            expected = getattr(self, field)
            if state[field] != expected:
                raise ValueError(
                    f"TANGS resume mismatch for {field}: {state[field]!r} != {expected!r}"
                )
        self.anchor = state["anchor"]
        self.anchor_age = int(state["anchor_age"])
        self.anchor_updates = int(state["anchor_updates"])
        self.invalid_anchor_updates = int(state["invalid_anchor_updates"])

    def live_tensors(self) -> list[torch.Tensor | None]:
        return [self.anchor]

    def transform(
        self,
        base_gradient: torch.Tensor,
        head_gradient: torch.Tensor | None,
        tail_gradient: torch.Tensor | None,
        step: int,
    ) -> SurgeryResult:
        anchor_updated = self.update_anchor(tail_gradient)
        anchor = tail_gradient if self.method == "instant-anchor" else self.anchor
        stats: dict[str, float | int | bool | str] = {
            "anchor_updated": anchor_updated,
            "anchor_age": self.anchor_age,
            "anchor_updates": self.anchor_updates,
            "invalid_anchor_updates": self.invalid_anchor_updates,
            "applied": False,
        }
        if _valid(self.anchor, self.eps):
            stats["ema_anchor_norm"] = float(torch.linalg.vector_norm(self.anchor))
        if _valid(tail_gradient, self.eps) and head_gradient is not None:
            stats["raw_cosine"] = cosine(head_gradient, tail_gradient, self.eps)

        if not _valid(anchor, self.eps):
            return SurgeryResult(base_gradient, False, "invalid-anchor", stats)
        if not _valid(head_gradient, self.eps):
            return SurgeryResult(base_gradient, False, "empty-or-invalid-head", stats)
        if not bool(torch.isfinite(base_gradient).all()):
            return SurgeryResult(base_gradient, False, "invalid-base-gradient", stats)

        assert anchor is not None and head_gradient is not None
        anchor = anchor.float()
        head_gradient = head_gradient.float()
        anchor_norm = torch.linalg.vector_norm(anchor)
        head_norm = torch.linalg.vector_norm(head_gradient)
        inner_product = torch.dot(head_gradient, anchor)
        effective_cosine = float(inner_product / (head_norm * anchor_norm + self.eps))
        raw_ratio = float(head_norm / (anchor_norm + self.eps))
        stats.update(
            {
                "effective_cosine": effective_cosine,
                "raw_ratio": raw_ratio,
                "conflict": bool(inner_product < 0),
            }
        )

        project = self.method in {
            "tangs",
            "oracle-tangs",
            "pcgrad",
            "no-cap",
            "instant-anchor",
        }
        cap = self.method in {
            "tangs",
            "oracle-tangs",
            "head-clip",
            "no-projection",
            "instant-anchor",
        }
        if not self.uses_surgery:
            project = True

        modified = head_gradient
        conflict = bool(inner_product < 0)
        if project and conflict:
            modified = head_gradient - (
                inner_product / (anchor_norm.square() + self.eps)
            ) * anchor

        projected_norm = torch.linalg.vector_norm(modified)
        post_projection_ratio = float(projected_norm / (anchor_norm + self.eps))
        capped = False
        stats["post_projection_ratio"] = post_projection_ratio
        if not self.uses_surgery:
            return SurgeryResult(base_gradient, False, "observer-only", stats)
        if step <= self.warmup_steps:
            return SurgeryResult(base_gradient, False, "warmup", stats)
        if cap and not math.isinf(self.tau):
            cap_norm = self.tau * anchor_norm
            if bool(projected_norm > cap_norm):
                modified = modified * (cap_norm / (projected_norm + self.eps))
                capped = True

        new_gradient = base_gradient - head_gradient + modified
        if not bool(torch.isfinite(new_gradient).all()):
            return SurgeryResult(base_gradient, False, "invalid-result", stats)

        stats.update(
            {
                "applied": True,
                "effective_cosine": effective_cosine,
                "raw_ratio": raw_ratio,
                "post_projection_ratio": post_projection_ratio,
                "final_ratio": float(
                    torch.linalg.vector_norm(modified) / (anchor_norm + self.eps)
                ),
                "conflict": conflict,
                "projected": bool(project and conflict),
                "capped": capped,
                "head_gradient_norm": float(head_norm),
                "anchor_norm": float(anchor_norm),
                "gradient_delta_norm": float(
                    torch.linalg.vector_norm(modified - head_gradient)
                ),
            }
        )
        return SurgeryResult(new_gradient, True, "applied", stats)


def exact_hard_nll_logit_gradient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_eps: float = 1e-8,
) -> torch.Tensor:
    """Derivative of ``-log(softmax(logits)[target] + loss_eps)``.

    The small factor that ``loss_eps`` introduces is retained, so the analytic
    classifier-row contribution matches the loss used by the trainer rather
    than silently approximating it with ordinary cross entropy.
    """
    logits_fp32 = logits.detach().float()
    targets = targets.detach().long()
    probabilities = torch.softmax(logits_fp32, dim=1)
    target_probabilities = probabilities.gather(1, targets[:, None])
    scale = target_probabilities / (target_probabilities + loss_eps)
    derivative = probabilities.clone()
    derivative.scatter_add_(
        1,
        targets[:, None],
        -torch.ones_like(target_probabilities),
    )
    return derivative * scale


def _augmented_features(features: torch.Tensor) -> torch.Tensor:
    flattened = features.detach().float().flatten(1)
    ones = torch.ones(
        (flattened.shape[0], 1),
        dtype=flattened.dtype,
        device=flattened.device,
    )
    return torch.cat([flattened, ones], dim=1)


def exact_classifier_row_contributions(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    row_ids: torch.Tensor,
    sample_weights: torch.Tensor,
    loss_denominator: int,
    loss_eps: float = 1e-8,
) -> torch.Tensor:
    """Exact selected classifier-row gradient for a weighted mean NLL."""
    if loss_denominator <= 0:
        raise ValueError("loss_denominator must be positive.")
    if logits.shape[0] != features.shape[0] or logits.shape[0] != targets.shape[0]:
        raise ValueError("Logits, features, and targets must share a batch dimension.")
    if sample_weights.numel() != logits.shape[0]:
        raise ValueError("sample_weights must contain one value per sample.")
    derivative = exact_hard_nll_logit_gradient(logits, targets, loss_eps)
    selected = derivative.index_select(1, row_ids.long())
    weighted = selected * sample_weights.detach().float().reshape(-1, 1)
    return weighted.transpose(0, 1).matmul(_augmented_features(features)) / float(
        loss_denominator
    )


def exact_self_classifier_row_contributions(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    row_ids: torch.Tensor,
    loss_denominator: int,
    loss_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selected row gradients using only examples whose target is that row."""
    if loss_denominator <= 0:
        raise ValueError("loss_denominator must be positive.")
    derivative = exact_hard_nll_logit_gradient(logits, targets, loss_eps)
    row_ids = row_ids.long()
    selected = derivative.index_select(1, row_ids)
    membership = targets.detach().long()[:, None].eq(row_ids[None, :])
    weighted = selected * membership.float()
    rows = weighted.transpose(0, 1).matmul(_augmented_features(features)) / float(
        loss_denominator
    )
    return rows, membership.any(dim=0)


class TailRowController:
    """v4.6 classwise protection restricted to tail classifier output rows."""

    METHODS = {
        "tailrow-observer",
        "tailrow-group",
        "tailrow-classwise",
        "tangs-v46",
        "tangs-v47",
        "tangs-v48",
        "oracle-tangs-v46",
    }

    def __init__(
        self,
        method: str,
        num_classes: int,
        feature_dim: int,
        tail_classes: list[int],
        beta: float = 0.99,
        correction_rho: float = 1.0,
        warmup_steps: int = 2500,
        eps: float = 1e-12,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown tail-row method: {method}")
        self.method = method
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.tail_classes = [int(class_id) for class_id in tail_classes]
        self.beta = float(beta)
        self.correction_rho = float(correction_rho)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.class_anchors: torch.Tensor | None = None
        self.class_anchor_valid: torch.Tensor | None = None
        self.class_norm_ema: torch.Tensor | None = None
        self.class_anchor_ages: torch.Tensor | None = None
        self.class_update_counts: torch.Tensor | None = None
        self.group_anchor: torch.Tensor | None = None
        self.group_anchor_age = 0
        self.group_anchor_updates = 0
        self.invalid_anchor_updates = 0

    def _ensure_state(self, reference: torch.Tensor) -> None:
        rows = len(self.tail_classes)
        row_dim = self.feature_dim + 1
        if self.class_anchors is None:
            self.class_anchors = torch.zeros(
                (rows, row_dim), dtype=torch.float32, device=reference.device
            )
            self.class_anchor_valid = torch.zeros(
                rows, dtype=torch.bool, device=reference.device
            )
            self.class_norm_ema = torch.zeros(
                rows, dtype=torch.float32, device=reference.device
            )
            self.class_anchor_ages = torch.zeros(
                rows, dtype=torch.long, device=reference.device
            )
            self.class_update_counts = torch.zeros(
                rows, dtype=torch.long, device=reference.device
            )

    def live_tensors(self) -> list[torch.Tensor | None]:
        return [
            self.class_anchors,
            self.class_anchor_valid,
            self.class_norm_ema,
            self.class_anchor_ages,
            self.class_update_counts,
            self.group_anchor,
        ]

    def state_dict(self) -> dict:
        return {
            "method": self.method,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "tail_classes": self.tail_classes,
            "beta": self.beta,
            "correction_rho": self.correction_rho,
            "warmup_steps": self.warmup_steps,
            "eps": self.eps,
            "class_anchors": self.class_anchors,
            "class_anchor_valid": self.class_anchor_valid,
            "class_norm_ema": self.class_norm_ema,
            "class_anchor_ages": self.class_anchor_ages,
            "class_update_counts": self.class_update_counts,
            "group_anchor": self.group_anchor,
            "group_anchor_age": self.group_anchor_age,
            "group_anchor_updates": self.group_anchor_updates,
            "invalid_anchor_updates": self.invalid_anchor_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        for field in (
            "method",
            "num_classes",
            "feature_dim",
            "tail_classes",
            "beta",
            "correction_rho",
            "warmup_steps",
            "eps",
        ):
            expected = getattr(self, field)
            if state[field] != expected:
                raise ValueError(
                    f"Tail-row resume mismatch for {field}: "
                    f"{state[field]!r} != {expected!r}"
                )
        for field in (
            "class_anchors",
            "class_anchor_valid",
            "class_norm_ema",
            "class_anchor_ages",
            "class_update_counts",
            "group_anchor",
        ):
            setattr(self, field, state[field])
        self.group_anchor_age = int(state["group_anchor_age"])
        self.group_anchor_updates = int(state["group_anchor_updates"])
        self.invalid_anchor_updates = int(state["invalid_anchor_updates"])

    def _update_anchors(
        self,
        self_tail_rows: torch.Tensor,
        self_present: torch.Tensor,
        group_tail_rows: torch.Tensor | None,
    ) -> tuple[int, float | None, bool]:
        self._ensure_state(self_tail_rows)
        assert self.class_anchors is not None
        assert self.class_anchor_valid is not None
        assert self.class_norm_ema is not None
        assert self.class_anchor_ages is not None
        assert self.class_update_counts is not None
        rows = self_tail_rows.detach().float()
        norms = torch.linalg.vector_norm(rows, dim=1)
        present = self_present.detach().bool()
        valid_updates = (
            present
            & torch.isfinite(rows).all(dim=1)
            & torch.isfinite(norms)
            & (norms > self.eps)
        )
        invalid_updates = int((present & ~valid_updates).sum())
        self.invalid_anchor_updates += invalid_updates
        self.class_anchor_ages.add_(self.class_anchor_valid.long())

        existing = valid_updates & self.class_anchor_valid
        existing_count = int(existing.sum())
        agreement = None
        if existing_count:
            agreement = float(
                torch.nn.functional.cosine_similarity(
                    rows[existing], self.class_anchors[existing], dim=1, eps=self.eps
                ).mean()
            )
        fresh = valid_updates & ~self.class_anchor_valid
        self.class_anchors[existing] = (
            self.beta * self.class_anchors[existing]
            + (1.0 - self.beta) * rows[existing]
        )
        self.class_norm_ema[existing] = (
            self.beta * self.class_norm_ema[existing]
            + (1.0 - self.beta) * norms[existing]
        )
        self.class_anchors[fresh] = rows[fresh]
        self.class_norm_ema[fresh] = norms[fresh]
        self.class_anchor_valid.logical_or_(valid_updates)
        self.class_anchor_ages[valid_updates] = 0
        self.class_update_counts.add_(valid_updates.long())
        updated = int(valid_updates.sum())

        group_updated = _valid(group_tail_rows, self.eps)
        if group_updated:
            assert group_tail_rows is not None
            flattened = group_tail_rows.detach().float().reshape(-1)
            if self.group_anchor is None:
                self.group_anchor = flattened.clone()
            else:
                self.group_anchor.mul_(self.beta).add_(
                    flattened, alpha=1.0 - self.beta
                )
            self.group_anchor_age = 0
            self.group_anchor_updates += 1
        else:
            self.group_anchor_age += 1
        return updated, agreement, group_updated

    def _replace_tail_rows(
        self,
        base_gradient: torch.Tensor,
        head_rows: torch.Tensor,
        modified_rows: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.num_classes * self.feature_dim + self.num_classes
        if base_gradient.numel() != expected:
            raise ValueError(
                f"Classifier gradient has {base_gradient.numel()} values; expected {expected}."
            )
        result = base_gradient.detach().float().clone()
        weights = result[: self.num_classes * self.feature_dim].view(
            self.num_classes, self.feature_dim
        )
        biases = result[self.num_classes * self.feature_dim :]
        row_ids = torch.tensor(
            self.tail_classes, dtype=torch.long, device=result.device
        )
        base_rows = torch.cat(
            [weights.index_select(0, row_ids), biases.index_select(0, row_ids)[:, None]],
            dim=1,
        )
        replacement = base_rows - head_rows.float() + modified_rows.float()
        weights.index_copy_(0, row_ids, replacement[:, :-1])
        biases.index_copy_(0, row_ids, replacement[:, -1])
        return result

    def transform(
        self,
        base_gradient: torch.Tensor,
        head_rows: torch.Tensor,
        group_tail_rows: torch.Tensor | None,
        self_tail_rows: torch.Tensor,
        self_present: torch.Tensor,
        step: int,
    ) -> SurgeryResult:
        if not bool(torch.isfinite(base_gradient).all()):
            return SurgeryResult(base_gradient, False, "invalid-base-gradient", {})
        self._ensure_state(self_tail_rows)
        updated, agreement, group_updated = self._update_anchors(
            self_tail_rows, self_present, group_tail_rows
        )
        assert self.class_anchors is not None
        assert self.class_anchor_valid is not None
        assert self.class_norm_ema is not None
        assert self.class_anchor_ages is not None

        row_norms = torch.linalg.vector_norm(head_rows.float(), dim=1)
        eligible = self.class_anchor_valid & torch.isfinite(row_norms) & (
            row_norms > self.eps
        )
        dots = (head_rows.float() * self.class_anchors).sum(dim=1)
        anchor_norms = torch.linalg.vector_norm(self.class_anchors, dim=1)
        cosines = dots / (row_norms * anchor_norms + self.eps)
        conflicts = eligible & (dots < 0)
        valid_ages = self.class_anchor_ages[self.class_anchor_valid]
        stats: dict[str, float | int | bool | str] = {
            "tailrow_anchor_updated_rows": updated,
            "tailrow_valid_anchor_rows": int(self.class_anchor_valid.sum()),
            "tailrow_eligible_rows": int(eligible.sum()),
            "tailrow_conflict_rows": int(conflicts.sum()),
            "tailrow_budget_limited_rows": 0,
            "tailrow_class_anchor_update_total": int(
                self.class_update_counts.sum()
            ) if self.class_update_counts is not None else 0,
            "tailrow_group_anchor_updated": group_updated,
            "tailrow_group_anchor_updates": self.group_anchor_updates,
            "tailrow_group_anchor_age": self.group_anchor_age,
            "tailrow_correction_rho": self.correction_rho,
            "applied": False,
        }
        if bool(eligible.any()):
            stats["tailrow_mean_cosine"] = float(cosines[eligible].mean())
            stats["tailrow_min_cosine"] = float(cosines[eligible].min())
        if agreement is not None:
            stats["tailrow_current_anchor_agreement"] = agreement
        if valid_ages.numel():
            stats["tailrow_mean_anchor_age"] = float(valid_ages.float().mean())
            stats["tailrow_max_anchor_age"] = int(valid_ages.max())

        if self.method in {"tailrow-observer", "tangs-v47", "tangs-v48"}:
            return SurgeryResult(base_gradient, False, "observer-only", stats)
        if step <= self.warmup_steps:
            return SurgeryResult(base_gradient, False, "warmup", stats)
        if not bool(eligible.any()):
            return SurgeryResult(base_gradient, False, "no-eligible-tail-rows", stats)

        modified = head_rows.detach().float().clone()
        if self.method == "tailrow-group":
            if not _valid(self.group_anchor, self.eps):
                return SurgeryResult(base_gradient, False, "invalid-group-anchor", stats)
            assert self.group_anchor is not None
            head_flat = head_rows.detach().float().reshape(-1)
            group_dot = torch.dot(head_flat, self.group_anchor)
            group_norm = torch.linalg.vector_norm(self.group_anchor)
            stats["tailrow_group_cosine"] = float(
                group_dot
                / (torch.linalg.vector_norm(head_flat) * group_norm + self.eps)
            )
            stats["tailrow_group_conflict"] = bool(group_dot < 0)
            if bool(group_dot < 0):
                modified = (
                    head_flat
                    - group_dot / (group_norm.square() + self.eps) * self.group_anchor
                ).view_as(head_rows)
        else:
            projection = (
                dots[:, None]
                / (anchor_norms.square()[:, None] + self.eps)
                * self.class_anchors
            )
            scales = torch.ones_like(dots)
            if self.method in {"tangs-v46", "oracle-tangs-v46"}:
                projection_norms = torch.linalg.vector_norm(projection, dim=1)
                if math.isinf(self.correction_rho):
                    budget_scales = torch.ones_like(projection_norms)
                else:
                    budget_scales = (
                        self.correction_rho
                        * self.class_norm_ema
                        / (projection_norms + self.eps)
                    ).clamp(max=1.0)
                scales = torch.where(conflicts, budget_scales, scales)
                stats["tailrow_budget_limited_rows"] = int(
                    (conflicts & (budget_scales < 1.0)).sum()
                )
            modified = head_rows.float() - (
                conflicts.float() * scales
            )[:, None] * projection

        result = self._replace_tail_rows(base_gradient, head_rows, modified)
        if not bool(torch.isfinite(result).all()):
            return SurgeryResult(base_gradient, False, "invalid-result", stats)
        delta = torch.linalg.vector_norm(modified - head_rows.float())
        stats["gradient_delta_norm"] = float(delta)
        stats["applied"] = True
        return SurgeryResult(result, True, "applied", stats)
