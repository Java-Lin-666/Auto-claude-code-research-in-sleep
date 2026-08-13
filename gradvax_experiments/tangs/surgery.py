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
