"""Step-based FixMatch/TANGS trainer on the locked CDMAD substrate."""

from __future__ import annotations

import json
import math
import os
import random
import signal
import shutil
import time
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import append_jsonl, atomic_json
from .config import (
    ORACLE_METHODS,
    TAILROW_METHODS,
    RunConfig,
    assert_locked_substrate,
)
from .data import CyclingLoader, DataBundle
from .ema import WeightEMA
from .metrics import evaluate_classifier, evaluate_pseudo_labels
from .surgery import (
    TailRowController,
    TangsController,
    classifier_gradient,
    cosine,
    exact_classifier_row_contributions,
    exact_self_classifier_row_contributions,
    read_classifier_gradient,
    write_classifier_gradient,
)
from .wrn import BatchNorm2d, WRN


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _hard_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    return -torch.log(probabilities + 1e-8).gather(1, targets[:, None]).squeeze(1)


def _membership(labels: torch.Tensor, class_ids: list[int]) -> torch.Tensor:
    lookup = torch.zeros(
        int(labels.max().item()) + 1 if labels.numel() else 1,
        dtype=torch.bool,
        device=labels.device,
    )
    valid_ids = [class_id for class_id in class_ids if class_id < len(lookup)]
    if valid_ids:
        lookup[torch.tensor(valid_ids, device=labels.device)] = True
    return lookup[labels]


def _flat_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(*tensors: torch.Tensor | None) -> int:
    """Count known live tensor storage without double-counting reused objects."""
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _restore_torch_rng_states(rng: dict[str, Any]) -> None:
    """Restore RNG byte tensors after a device-mapped checkpoint load."""
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])


def _rollback_step_jsonl(path: Path, maximum_step: int) -> list[dict[str, Any]]:
    """Atomically discard partial/future metric rows after checkpoint rollback."""
    if not path.exists():
        return []
    retained: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A hard kill may leave only the final JSONL record partial.
                break
            step = record.get("step")
            if isinstance(step, int) and step <= maximum_step:
                retained.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return retained


class Diagnostics:
    def __init__(self, sample_limit: int, seed: int):
        self.counts: Counter[str] = Counter()
        self.sums: Counter[str] = Counter()
        self.reason_counts: Counter[str] = Counter()
        self.samples: list[dict[str, Any]] = []
        self.sample_limit = sample_limit
        self.random = random.Random(seed)
        self.seen_samples = 0

    def observe(
        self,
        step: int,
        tail_count: int,
        accepted_head_count: int,
        result: Any,
        deep: dict[str, Any] | None,
    ) -> None:
        self.counts["total_steps"] += 1
        self.counts["tail_support_steps"] += int(tail_count >= 2)
        self.counts["predicted_head_occurrence_steps"] += int(accepted_head_count > 0)
        self.counts["anchor_update_steps"] += int(
            result.stats.get("anchor_updated", False)
        )
        self.counts["surgery_applied_steps"] += int(result.applied)
        self.counts["nonzero_gradient_modification_steps"] += int(
            float(result.stats.get("gradient_delta_norm", 0.0)) > 1e-12
        )
        self.reason_counts[result.reason] += 1
        if "effective_cosine" in result.stats:
            self.counts["eligible_geometry_steps"] += 1
            conflict_event = result.stats["effective_cosine"] < 0
            domination_event = (
                result.stats.get("post_projection_ratio", math.inf)
                > result.stats.get("tau", math.inf)
            )
            self.counts["conflict_steps"] += int(conflict_event)
            self.counts["domination_steps"] += int(domination_event)
            self.counts["either_steps"] += int(conflict_event or domination_event)
            self.sums["effective_cosine"] += float(result.stats["effective_cosine"])
            self.sums["raw_ratio"] += float(result.stats["raw_ratio"])
        if "raw_cosine" in result.stats:
            self.counts["current_tail_geometry_steps"] += 1
            self.sums["raw_cosine"] += float(result.stats["raw_cosine"])
        tailrow_eligible = int(result.stats.get("tailrow_eligible_rows", 0))
        if "tailrow_eligible_rows" in result.stats:
            self.counts["tailrow_observed_steps"] += 1
            self.counts["tailrow_eligible_rows"] += tailrow_eligible
            self.counts["tailrow_conflict_rows"] += int(
                result.stats.get("tailrow_conflict_rows", 0)
            )
            self.counts["tailrow_budget_limited_rows"] += int(
                result.stats.get("tailrow_budget_limited_rows", 0)
            )
            self.counts["tailrow_anchor_updated_rows"] += int(
                result.stats.get("tailrow_anchor_updated_rows", 0)
            )
            if "tailrow_mean_cosine" in result.stats:
                self.sums["tailrow_cosine_weighted"] += (
                    float(result.stats["tailrow_mean_cosine"]) * tailrow_eligible
                )
            if "tailrow_current_anchor_agreement" in result.stats:
                self.counts["tailrow_agreement_steps"] += 1
                self.sums["tailrow_current_anchor_agreement"] += float(
                    result.stats["tailrow_current_anchor_agreement"]
                )

        sample = {
            "step": step,
            "tail_count": tail_count,
            "accepted_predicted_head_count": accepted_head_count,
            "reason": result.reason,
            **result.stats,
        }
        if deep:
            sample["deep"] = deep
        self.seen_samples += 1
        if len(self.samples) < self.sample_limit:
            self.samples.append(sample)
        else:
            replacement = self.random.randint(1, self.seen_samples)
            if replacement <= self.sample_limit:
                self.samples[replacement - 1] = sample

    def summary(self, tau: float) -> dict[str, Any]:
        total = max(self.counts["total_steps"], 1)
        eligible = max(self.counts["eligible_geometry_steps"], 1)
        current = max(self.counts["current_tail_geometry_steps"], 1)
        tailrow_eligible = max(self.counts["tailrow_eligible_rows"], 1)
        tailrow_agreement = max(self.counts["tailrow_agreement_steps"], 1)
        return {
            "counts": dict(self.counts),
            "reason_counts": dict(self.reason_counts),
            "fractions_over_all_steps": {
                "tail_support": self.counts["tail_support_steps"] / total,
                "predicted_head_occurrence": self.counts[
                    "predicted_head_occurrence_steps"
                ]
                / total,
                "eligible_geometry": self.counts["eligible_geometry_steps"] / total,
                "conflict": self.counts["conflict_steps"] / total,
                "domination_post_projection_gt_tau": self.counts["domination_steps"] / total,
                "either": self.counts["either_steps"] / total,
                "anchor_update": self.counts["anchor_update_steps"] / total,
                "surgery_applied": self.counts["surgery_applied_steps"] / total,
            },
            "fractions_over_eligible_steps": {
                "conflict": self.counts["conflict_steps"] / eligible,
                "domination_post_projection_gt_tau": self.counts["domination_steps"]
                / eligible,
                "either": self.counts["either_steps"] / eligible,
            },
            "means": {
                "effective_cosine": self.sums["effective_cosine"] / eligible,
                "raw_ratio": self.sums["raw_ratio"] / eligible,
                "current_tail_cosine": self.sums["raw_cosine"] / current,
            },
            "tau": tau,
            "tailrow": {
                "observed_steps": self.counts["tailrow_observed_steps"],
                "eligible_rows": self.counts["tailrow_eligible_rows"],
                "conflict_rows": self.counts["tailrow_conflict_rows"],
                "budget_limited_rows": self.counts["tailrow_budget_limited_rows"],
                "anchor_updated_rows": self.counts["tailrow_anchor_updated_rows"],
                "conflict_fraction_over_eligible_rows": self.counts[
                    "tailrow_conflict_rows"
                ]
                / tailrow_eligible,
                "budget_limited_fraction_over_conflict_rows": self.counts[
                    "tailrow_budget_limited_rows"
                ]
                / max(self.counts["tailrow_conflict_rows"], 1),
                "mean_cosine_over_eligible_rows": self.sums[
                    "tailrow_cosine_weighted"
                ]
                / tailrow_eligible,
                "mean_current_anchor_agreement": self.sums[
                    "tailrow_current_anchor_agreement"
                ]
                / tailrow_agreement,
            },
            "bounded_samples": sorted(self.samples, key=lambda item: item["step"]),
            "sample_limit": self.sample_limit,
            "seen_samples": self.seen_samples,
        }


class Trainer:
    def __init__(
        self,
        config: RunConfig,
        data: DataBundle,
        run_dir: Path,
        config_hash: str,
    ):
        assert_locked_substrate(config)
        self.config = config
        self.data = data
        self.run_dir = run_dir
        self.config_hash = config_hash
        self.device = resolve_device(config.device)

        self.model = WRN(2, num_classes=config.protocol.num_classes).to(self.device)
        self.ema_model = WRN(2, num_classes=config.protocol.num_classes).to(self.device)
        for parameter in self.ema_model.parameters():
            parameter.detach_()
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()), lr=config.learning_rate
        )
        self.ema_optimizer = WeightEMA(
            self.model,
            self.ema_model,
            learning_rate=config.learning_rate,
            decay_coefficient=config.protocol.custom_decay,
            alpha=config.ema_decay,
        )
        self.classifier_parameters = list(self.model.output.parameters())
        self.use_tailrow = (
            config.method in TAILROW_METHODS or config.tailrow_observer
        )
        if self.use_tailrow:
            controller_method = (
                config.method
                if config.method in TAILROW_METHODS
                else "tailrow-observer"
            )
            self.controller = TailRowController(
                method=controller_method,
                num_classes=config.protocol.num_classes,
                feature_dim=self.model.output.in_features,
                tail_classes=data.partition["tail"],
                beta=config.tangs_beta,
                correction_rho=config.tangs_correction_rho,
                warmup_steps=config.tangs_warmup_steps,
                eps=config.geometry_eps,
            )
        else:
            self.controller = TangsController(
                method=config.method,
                beta=config.tangs_beta,
                tau=config.tangs_tau,
                warmup_steps=config.tangs_warmup_steps,
                eps=config.geometry_eps,
            )
        self.diagnostics = Diagnostics(
            config.diagnostic_sample_limit, config.manual_seed
        )
        self.start_step = 0
        self.development_evaluations: list[dict[str, Any]] = []
        self.warned_invalid_post_warmup_anchor = False
        self._assert_source_parity()
        if config.resume:
            self._resume(config.resume)

    def _assert_source_parity(self) -> None:
        for module in self.model.modules():
            if isinstance(module, BatchNorm2d):
                if module.eps != 1e-5 or module.momentum != 0.1:
                    raise AssertionError("Executed BatchNorm defaults drifted from CDMAD.")
        group = self.optimizer.param_groups[0]
        expected = {
            "lr": 0.0015,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
        }
        for key, value in expected.items():
            if group[key] != value:
                raise AssertionError(
                    f"Adam substrate drift: {key}={group[key]!r}, expected {value!r}"
                )
        if not self.model.rotation or not hasattr(self.model, "rot"):
            raise AssertionError("The unused CDMAD rotation head must be retained.")
        if torch.backends.cudnn.benchmark:
            raise AssertionError("cudnn.benchmark must remain disabled.")

    def _resume(self, requested: str) -> None:
        checkpoint_path = (
            self.run_dir / "checkpoint_last.pt"
            if requested == "auto"
            else Path(requested).expanduser().resolve()
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint["config_hash"] != self.config_hash:
            raise RuntimeError("Checkpoint configuration hash does not match this run.")
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.controller.load_state_dict(checkpoint["controller"])
        diagnostics = checkpoint.get("diagnostics")
        if diagnostics:
            self.diagnostics.counts = Counter(diagnostics["counts"])
            self.diagnostics.sums = Counter(diagnostics["sums"])
            self.diagnostics.reason_counts = Counter(diagnostics["reason_counts"])
            self.diagnostics.samples = list(diagnostics["samples"])
            self.diagnostics.seen_samples = int(diagnostics["seen_samples"])
            self.diagnostics.random.setstate(diagnostics["random_state"])
        self.start_step = int(checkpoint["step"])
        _rollback_step_jsonl(
            self.run_dir / "train_metrics.jsonl", self.start_step
        )
        logged_development = _rollback_step_jsonl(
            self.run_dir / "development_metrics.jsonl", self.start_step
        )
        self.development_evaluations = (
            logged_development
            if logged_development
            else list(checkpoint.get("development_evaluations", []))
        )
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        # map_location moves every tensor in the checkpoint to the training
        # device, including RNG byte tensors. Both RNG restoration APIs expect
        # CPU ByteTensors even when the model itself resumes on CUDA.
        _restore_torch_rng_states(checkpoint["rng"])
        append_jsonl(
            self.run_dir / "events.jsonl",
            {
                "event": "resume",
                "step": self.start_step,
                "checkpoint": str(checkpoint_path),
                "loader_resume_bit_exact": False,
                "reason": "DataLoader worker/prefetch state is not serializable.",
            },
        )
        manifest_path = self.run_dir / "run_manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest.setdefault("resume_history", []).append(
                {
                    "resumed_unix": time.time(),
                    "step": self.start_step,
                    "checkpoint": str(checkpoint_path),
                    "loader_resume_bit_exact": False,
                }
            )
            atomic_json(manifest_path, manifest)

    def _checkpoint(self, step: int, interrupted: bool = False) -> Path:
        payload = {
            "step": step,
            "config_hash": self.config_hash,
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "controller": self.controller.state_dict(),
            "diagnostics": {
                "counts": dict(self.diagnostics.counts),
                "sums": dict(self.diagnostics.sums),
                "reason_counts": dict(self.diagnostics.reason_counts),
                "samples": self.diagnostics.samples,
                "seen_samples": self.diagnostics.seen_samples,
                "random_state": self.diagnostics.random.getstate(),
            },
            "development_evaluations": self.development_evaluations,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            "interrupted": interrupted,
            "resume_note": "Loader continuation after resume is functional but not bit exact.",
        }
        path = self.run_dir / "checkpoint_last.pt"
        temporary = self.run_dir / "checkpoint_last.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, path)
        if step % self.config.snapshot_every == 0:
            shutil.copy2(path, self.run_dir / f"checkpoint_step{step:06d}.pt")
        return path

    def _deep_diagnostics(
        self,
        per_labeled_loss: torch.Tensor,
        labeled_targets: torch.Tensor,
        per_unlabeled_loss: torch.Tensor,
        accepted_head: torch.Tensor,
        unlabeled_true_targets: torch.Tensor,
        ordinary_unlabeled_targets: torch.Tensor,
        tail_gradient: torch.Tensor | None,
    ) -> dict[str, Any]:
        if self.config.diagnostic_level != "deep":
            return {}
        output: dict[str, Any] = {"accepted_predicted_head_by_true_group": {}}
        repeated_true = torch.cat([unlabeled_true_targets, unlabeled_true_targets])
        repeated_predictions = torch.cat(
            [ordinary_unlabeled_targets, ordinary_unlabeled_targets]
        )
        known = repeated_true >= 0
        for group in ("head", "medium", "tail"):
            true_group = _membership(
                repeated_true.clamp_min(0), self.data.partition[group]
            )
            mask = accepted_head & known & true_group
            count = int(mask.sum().item() // 2)
            group_entry: dict[str, Any] = {"count": count}
            errors = int((mask & (repeated_predictions != repeated_true)).sum().item() // 2)
            group_entry["errors"] = errors
            group_entry["error_rate"] = errors / max(count, 1)
            if bool(mask.any()):
                group_loss = (per_unlabeled_loss * mask.float()).mean()
                gradient = classifier_gradient(
                    group_loss, self.classifier_parameters, retain_graph=True
                )
                group_entry["gradient_norm"] = float(torch.linalg.vector_norm(gradient))
                if tail_gradient is not None:
                    group_entry["cosine_with_current_tail"] = cosine(
                        gradient, tail_gradient, self.config.geometry_eps
                    )
            output["accepted_predicted_head_by_true_group"][group] = group_entry

        per_class_gradients = []
        per_class_cosines = {}
        for class_id in self.data.partition["tail"]:
            mask = labeled_targets == class_id
            if not bool(mask.any()):
                continue
            class_loss = (per_labeled_loss * mask.float()).mean()
            gradient = classifier_gradient(
                class_loss, self.classifier_parameters, retain_graph=True
            )
            per_class_gradients.append(gradient)
            if tail_gradient is not None:
                per_class_cosines[str(class_id)] = cosine(
                    gradient, tail_gradient, self.config.geometry_eps
                )
        if per_class_gradients:
            summed = torch.stack(per_class_gradients).sum(dim=0)
            denominator = sum(
                float(torch.linalg.vector_norm(gradient))
                for gradient in per_class_gradients
            )
            output["within_tail_cancellation_ratio"] = float(
                torch.linalg.vector_norm(summed)
            ) / max(denominator, self.config.geometry_eps)
        output["per_class_tail_cosine"] = per_class_cosines
        return output

    def _status(self, step: int, started: float, state: str = "running") -> None:
        elapsed = time.perf_counter() - started
        completed = max(step - self.start_step, 0)
        rate = completed / elapsed if elapsed > 0 else 0.0
        remaining = (self.config.total_steps - step) / rate if rate > 0 else None
        atomic_json(
            self.run_dir / "status.json",
            {
                "state": state,
                "step": step,
                "total_steps": self.config.total_steps,
                "progress": step / self.config.total_steps,
                "elapsed_seconds_this_process": elapsed,
                "steps_per_second": rate,
                "eta_seconds": remaining,
                "updated_unix": time.time(),
            },
        )

    def train(self) -> dict[str, Any]:
        labeled_iterator = CyclingLoader(self.data.labeled_loader)
        unlabeled_iterator = CyclingLoader(self.data.unlabeled_loader)
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def interrupt_on_sigterm(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"received signal {signum}")

        # Install this only after the training workers have been created, so
        # forked workers retain their normal termination behavior. Cluster
        # schedulers commonly send SIGTERM before SIGKILL on preemption.
        signal.signal(signal.SIGTERM, interrupt_on_sigterm)
        head_classes = self.data.partition["head"]
        tail_classes = self.data.partition["tail"]
        oracle = self.config.method in ORACLE_METHODS
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._status(self.start_step, started)

        try:
            for step in range(self.start_step + 1, self.config.total_steps + 1):
                log_update = step % self.config.log_every == 0
                if log_update:
                    _synchronize(self.device)
                step_started = time.perf_counter()
                self.model.train()
                images_x, targets_x, _ = labeled_iterator.next()
                (images_u_w, images_u_s1, images_u_s2), targets_u_true, _ = (
                    unlabeled_iterator.next()
                )
                images_x = images_x.to(self.device, non_blocking=True)
                targets_x = targets_x.to(self.device, non_blocking=True)
                images_u_w = images_u_w.to(self.device, non_blocking=True)
                images_u_s1 = images_u_s1.to(self.device, non_blocking=True)
                images_u_s2 = images_u_s2.to(self.device, non_blocking=True)
                targets_u_true = targets_u_true.to(self.device, non_blocking=True)

                if self.use_tailrow:
                    logits_x, _, features_x = self.model(
                        images_x, return_feature=True
                    )
                else:
                    logits_x, _ = self.model(images_x)
                    features_x = None
                with torch.no_grad():
                    weak_logits, _ = self.model(images_u_w)
                    weak_probabilities = F.softmax(weak_logits, dim=1)
                    confidence, pseudo_targets = weak_probabilities.max(dim=1)
                    accepted = confidence >= self.config.confidence_threshold
                ordinary_membership = pseudo_targets.detach()
                loss_targets = pseudo_targets.detach().clone()
                if oracle:
                    if bool((targets_u_true < 0).any()):
                        raise RuntimeError(
                            "Oracle-target control encountered unknown ground truth."
                        )
                    loss_targets[accepted] = targets_u_true[accepted]

                if self.use_tailrow:
                    logits_u_s1, _, features_u_s1 = self.model(
                        images_u_s1, return_feature=True
                    )
                    logits_u_s2, _, features_u_s2 = self.model(
                        images_u_s2, return_feature=True
                    )
                else:
                    logits_u_s1, _ = self.model(images_u_s1)
                    logits_u_s2, _ = self.model(images_u_s2)
                    features_u_s1 = None
                    features_u_s2 = None
                per_labeled = _hard_nll(logits_x, targets_x)
                per_unlabeled = torch.cat(
                    [
                        _hard_nll(logits_u_s1, loss_targets),
                        _hard_nll(logits_u_s2, loss_targets),
                    ]
                )
                accepted_twice = torch.cat([accepted, accepted])
                predicted_head = _membership(ordinary_membership, head_classes)
                accepted_head = torch.cat(
                    [accepted & predicted_head, accepted & predicted_head]
                )
                unlabeled_weights = torch.ones_like(per_unlabeled)
                if self.config.method == "head-downweight":
                    unlabeled_weights[accepted_head] = self.config.head_weight

                loss_x = per_labeled.mean()
                loss_u = (
                    per_unlabeled
                    * accepted_twice.float()
                    * unlabeled_weights
                ).mean()
                loss = loss_x + self.config.lambda_u * loss_u

                auxiliary_gradient_seconds = 0.0
                if log_update:
                    _synchronize(self.device)
                    auxiliary_started = time.perf_counter()

                tail_mask = _membership(targets_x, tail_classes)
                tail_count = int(tail_mask.sum())
                tail_gradient = None
                accepted_head_count = int((accepted & predicted_head).sum())
                head_gradient = None
                head_rows = None
                group_tail_rows = None
                self_tail_rows = None
                self_present = None
                if self.use_tailrow:
                    assert features_x is not None
                    assert features_u_s1 is not None and features_u_s2 is not None
                    tail_row_ids = torch.tensor(
                        tail_classes, dtype=torch.long, device=self.device
                    )
                    strong_logits = torch.cat([logits_u_s1, logits_u_s2])
                    strong_features = torch.cat([features_u_s1, features_u_s2])
                    strong_targets = torch.cat([loss_targets, loss_targets])
                    head_rows = exact_classifier_row_contributions(
                        strong_logits,
                        strong_features,
                        strong_targets,
                        tail_row_ids,
                        self.config.lambda_u
                        * accepted_head.float()
                        * unlabeled_weights,
                        loss_denominator=per_unlabeled.numel(),
                    )
                    if tail_count >= self.config.min_tail_support:
                        group_tail_rows = exact_classifier_row_contributions(
                            logits_x,
                            features_x,
                            targets_x,
                            tail_row_ids,
                            tail_mask.float(),
                            loss_denominator=per_labeled.numel(),
                        )
                    self_tail_rows, self_present = (
                        exact_self_classifier_row_contributions(
                            logits_x,
                            features_x,
                            targets_x,
                            tail_row_ids,
                            loss_denominator=per_labeled.numel(),
                        )
                    )
                else:
                    if tail_count >= self.config.min_tail_support:
                        tail_loss = (per_labeled * tail_mask.float()).mean()
                        tail_gradient = classifier_gradient(
                            tail_loss, self.classifier_parameters, retain_graph=True
                        )
                    if accepted_head_count > 0:
                        head_loss = self.config.lambda_u * (
                            per_unlabeled
                            * accepted_head.float()
                            * unlabeled_weights
                        ).mean()
                        head_gradient = classifier_gradient(
                            head_loss, self.classifier_parameters, retain_graph=True
                        )

                if log_update:
                    _synchronize(self.device)
                    auxiliary_gradient_seconds = time.perf_counter() - auxiliary_started

                deep = {}
                if log_update:
                    _synchronize(self.device)
                    diagnostics_started = time.perf_counter()
                    deep = self._deep_diagnostics(
                        per_labeled,
                        targets_x,
                        per_unlabeled,
                        accepted_head,
                        targets_u_true,
                        ordinary_membership,
                        tail_gradient,
                    )
                    _synchronize(self.device)
                    deep_diagnostics_seconds = time.perf_counter() - diagnostics_started
                else:
                    deep_diagnostics_seconds = 0.0

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if log_update:
                    _synchronize(self.device)
                    surgery_started = time.perf_counter()
                base_gradient = read_classifier_gradient(self.classifier_parameters)
                if self.use_tailrow:
                    assert head_rows is not None
                    assert self_tail_rows is not None and self_present is not None
                    result = self.controller.transform(
                        base_gradient,
                        head_rows,
                        group_tail_rows,
                        self_tail_rows,
                        self_present,
                        step,
                    )
                    result.stats["correction_rho"] = (
                        self.config.tangs_correction_rho
                    )
                else:
                    result = self.controller.transform(
                        base_gradient, head_gradient, tail_gradient, step
                    )
                    result.stats["tau"] = self.config.tangs_tau
                if (
                    result.reason == "invalid-anchor"
                    and step > self.config.tangs_warmup_steps
                    and not self.warned_invalid_post_warmup_anchor
                ):
                    warning = {
                        "event": "warning",
                        "warning": "invalid-post-warmup-anchor",
                        "step": step,
                        "action": "base-update-bypass",
                    }
                    print(
                        "WARNING: post-warm-up TANGS anchor is invalid; "
                        "using the exact base update.",
                        file=sys.stderr,
                        flush=True,
                    )
                    append_jsonl(self.run_dir / "events.jsonl", warning)
                    self.warned_invalid_post_warmup_anchor = True
                if result.applied:
                    write_classifier_gradient(
                        self.classifier_parameters, result.gradient
                    )
                surgery_seconds = 0.0
                if log_update:
                    _synchronize(self.device)
                    surgery_seconds = time.perf_counter() - surgery_started

                before_parameters = (
                    _flat_parameters(self.classifier_parameters).clone()
                    if log_update
                    else None
                )
                self.optimizer.step()
                self.ema_optimizer.step()
                self.diagnostics.observe(
                    step, tail_count, accepted_head_count, result, deep
                )
                if sys.stdout.isatty() or log_update:
                    line_end = (
                        "\n"
                        if step == self.config.total_steps or not sys.stdout.isatty()
                        else "\r"
                    )
                    print(
                        f"step {step}/{self.config.total_steps} "
                        f"loss={float(loss.detach()):.4f} "
                        f"Lx={float(loss_x.detach()):.4f} Lu={float(loss_u.detach()):.4f}",
                        end=line_end,
                        flush=True,
                    )

                if log_update:
                    _synchronize(self.device)
                    step_seconds = time.perf_counter() - step_started
                    training_step_seconds = max(
                        step_seconds - deep_diagnostics_seconds, 1e-12
                    )
                    extra_seconds = auxiliary_gradient_seconds + surgery_seconds
                    peak_allocated_bytes = (
                        torch.cuda.max_memory_allocated(self.device)
                        if self.device.type == "cuda"
                        else 0
                    )
                    known_tangs_tensor_bytes = _tensor_bytes(
                        head_gradient,
                        tail_gradient,
                        head_rows,
                        group_tail_rows,
                        self_tail_rows,
                        base_gradient,
                        result.gradient,
                        *self.controller.live_tensors(),
                    )
                    after_parameters = _flat_parameters(self.classifier_parameters)
                    assert before_parameters is not None
                    effective_update = after_parameters - before_parameters
                    update_norm = float(torch.linalg.vector_norm(effective_update))
                    update_cosine = (
                        cosine(
                            effective_update,
                            -result.gradient,
                            self.config.geometry_eps,
                        )
                        if update_norm > self.config.geometry_eps
                        else 0.0
                    )
                    record = {
                        "step": step,
                        "epoch": step / self.config.steps_per_epoch,
                        "loss": float(loss.detach()),
                        "loss_x": float(loss_x.detach()),
                        "loss_u": float(loss_u.detach()),
                        "accepted_count": int(accepted.sum()),
                        "accepted_predicted_head_count": accepted_head_count,
                        "labeled_tail_count": tail_count,
                        "effective_classifier_update_norm": update_norm,
                        "effective_update_cosine_with_negative_gradient": update_cosine,
                        "step_seconds": step_seconds,
                        "training_step_seconds_excluding_diagnostics": training_step_seconds,
                        "auxiliary_gradient_seconds": auxiliary_gradient_seconds,
                        "surgery_seconds": surgery_seconds,
                        "extra_gradient_surgery_seconds": extra_seconds,
                        "extra_gradient_surgery_fraction": extra_seconds
                        / training_step_seconds,
                        "deep_diagnostics_seconds": deep_diagnostics_seconds,
                        "peak_allocated_bytes": peak_allocated_bytes,
                        "peak_reserved_bytes": (
                            torch.cuda.max_memory_reserved(self.device)
                            if self.device.type == "cuda"
                            else 0
                        ),
                        "known_tangs_tensor_bytes": known_tangs_tensor_bytes,
                        "known_tangs_tensor_fraction_of_peak": (
                            known_tangs_tensor_bytes / peak_allocated_bytes
                            if peak_allocated_bytes
                            else 0.0
                        ),
                        "surgery": result.stats,
                    }
                    append_jsonl(self.run_dir / "train_metrics.jsonl", record)
                    self._status(step, started)

                if step % self.config.checkpoint_every == 0:
                    self._checkpoint(step)
                    if self.config.mode == "development":
                        metrics = evaluate_classifier(
                            self.ema_model,
                            self.data.evaluation_loader,
                            self.data.partition,
                            self.config.protocol.num_classes,
                            self.device,
                        )
                        metrics["step"] = step
                        self.development_evaluations.append(metrics)
                        append_jsonl(
                            self.run_dir / "development_metrics.jsonl", metrics
                        )

            if self.config.total_steps % self.config.checkpoint_every:
                self._checkpoint(self.config.total_steps)
        except KeyboardInterrupt:
            self._checkpoint(step, interrupted=True)
            self._status(step, started, state="interrupted")
            raise
        except Exception:
            current_step = locals().get("step", self.start_step)
            self._checkpoint(current_step, interrupted=True)
            self._status(current_step, started, state="failed")
            raise

        signal.signal(signal.SIGTERM, previous_sigterm_handler)

        final_metrics = evaluate_classifier(
            self.ema_model,
            self.data.evaluation_loader,
            self.data.partition,
            self.config.protocol.num_classes,
            self.device,
        )
        pseudo_metrics = None
        if self.data.pseudo_evaluation_loader is not None:
            pseudo_metrics = evaluate_pseudo_labels(
                self.ema_model,
                self.data.pseudo_evaluation_loader,
                self.data.partition,
                self.config.protocol.num_classes,
                self.config.confidence_threshold,
                self.device,
            )
        elapsed = time.perf_counter() - started
        diagnostic_summary = self.diagnostics.summary(self.config.tangs_tau)
        atomic_json(self.run_dir / "gradient_diagnostics.json", diagnostic_summary)
        summary = {
            "run_id": self.config.run_id,
            "config_hash": self.config_hash,
            "evidence_label": (
                "manualSeed=0 fixed-protocol evidence"
                if self.config.mode == "confirmatory"
                and self.config.manual_seed == 0
                and not self.config.allow_protocol_deviation
                else "non-confirmatory"
            ),
            "final_step": self.config.total_steps,
            "evaluation_model": "EMA",
            "checkpoint_selection": "final-only",
            "performance": final_metrics,
            "pseudo_labels": pseudo_metrics,
            "development_evaluations": self.development_evaluations,
            "runtime": {
                "elapsed_seconds_this_process": elapsed,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(self.device)
                    if self.device.type == "cuda"
                    else 0
                ),
                "labeled_loader_restarts": labeled_iterator.restarts,
                "unlabeled_loader_restarts": unlabeled_iterator.restarts,
            },
            "resume_loader_bit_exact": False if self.config.resume else None,
        }
        atomic_json(self.run_dir / "summary.json", summary)
        self._status(self.config.total_steps, started, state="completed")
        append_jsonl(
            self.run_dir / "events.jsonl",
            {
                "event": "completed",
                "step": self.config.total_steps,
                "summary": str(self.run_dir / "summary.json"),
            },
        )
        return summary
