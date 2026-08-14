"""Locked experiment protocols derived from the pinned CDMAD release."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


CDMAD_COMMIT = "7cd732b4615b9d94934a9197e69c6775496fb5ee"
CONFIG_VERSION = "4.6"


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    dataset: str
    num_classes: int
    num_max: int
    num_max_u: int | None
    gamma_l: float
    gamma_u: float | None
    custom_decay: float
    role: str


PROTOCOLS = {
    "P-C10-100": Protocol(
        protocol_id="P-C10-100",
        dataset="cifar10",
        num_classes=10,
        num_max=1500,
        num_max_u=3000,
        gamma_l=100.0,
        gamma_u=100.0,
        custom_decay=0.04,
        role="required-local-generalization",
    ),
    "P-C100-100": Protocol(
        protocol_id="P-C100-100",
        dataset="cifar100",
        num_classes=100,
        num_max=150,
        num_max_u=300,
        gamma_l=100.0,
        gamma_u=100.0,
        custom_decay=0.08,
        role="required-local-core",
    ),
    "P-STL10-10": Protocol(
        protocol_id="P-STL10-10",
        dataset="stl10",
        num_classes=10,
        num_max=450,
        num_max_u=None,
        gamma_l=10.0,
        gamma_u=None,
        custom_decay=0.01,
        role="optional-local-extension",
    ),
    "P-STL10-20": Protocol(
        protocol_id="P-STL10-20",
        dataset="stl10",
        num_classes=10,
        num_max=450,
        num_max_u=None,
        gamma_l=20.0,
        gamma_u=None,
        custom_decay=0.01,
        role="required-local-generalization",
    ),
}


METHODS = {
    "fixmatch",
    "tangs",
    "tailrow-observer",
    "tailrow-group",
    "tailrow-classwise",
    "tangs-v46",
    "oracle-fixmatch",
    "oracle-tangs",
    "oracle-tangs-v46",
    "head-downweight",
    "head-clip",
    "pcgrad",
    "no-cap",
    "no-projection",
    "instant-anchor",
}

TAILROW_METHODS = {
    "tailrow-observer",
    "tailrow-group",
    "tailrow-classwise",
    "tangs-v46",
    "oracle-tangs-v46",
}

SURGERY_METHODS = {
    "tangs",
    "oracle-tangs",
    "head-clip",
    "pcgrad",
    "no-cap",
    "no-projection",
    "instant-anchor",
    *TAILROW_METHODS,
}

ORACLE_METHODS = {"oracle-fixmatch", "oracle-tangs", "oracle-tangs-v46"}


@dataclass(frozen=True)
class RunConfig:
    protocol_id: str
    method: str
    mode: str
    manual_seed: int
    data_root: str
    output_root: str
    run_id: str
    device: str
    workers: int
    download: bool
    resume: str | None
    total_steps: int
    steps_per_epoch: int
    batch_size: int
    unlabeled_ratio: int
    test_batch_size: int
    learning_rate: float
    ema_decay: float
    confidence_threshold: float
    lambda_u: float
    tangs_beta: float
    tangs_tau: float
    tangs_correction_rho: float
    tailrow_observer: bool
    tangs_warmup_steps: int
    min_tail_support: int
    geometry_eps: float
    head_weight: float
    log_every: int
    checkpoint_every: int
    snapshot_every: int
    diagnostic_sample_limit: int
    diagnostic_level: str
    eval_pseudo_labels: bool
    allow_protocol_deviation: bool
    protocol: Protocol

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_config(args: Any) -> RunConfig:
    if args.protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {args.protocol}")
    if args.method not in METHODS:
        raise ValueError(f"Unknown method: {args.method}")
    if args.mode not in {"confirmatory", "development", "smoke"}:
        raise ValueError(f"Unknown mode: {args.mode}")
    if args.amp:
        raise ValueError("AMP is forbidden by protocol v4.6; geometry and training are FP32.")

    if math.isnan(args.tangs_tau) or args.tangs_tau <= 0:
        raise ValueError("--tangs-tau must be positive or inf.")
    correction_rho = float(getattr(args, "tangs_correction_rho", 1.0))
    if math.isnan(correction_rho) or correction_rho <= 0:
        raise ValueError("--tangs-correction-rho must be positive or inf.")
    protocol = PROTOCOLS[args.protocol]
    deviations: list[str] = []

    if args.manual_seed != 0:
        deviations.append("manual_seed")
    if args.workers != 4:
        deviations.append("workers")
    if args.mode == "development" and args.protocol not in {
        "P-C10-100",
        "P-C100-100",
    }:
        raise ValueError("Development tuning is locked to CIFAR protocols.")
    if args.method in ORACLE_METHODS and args.protocol != "P-C100-100":
        raise ValueError("The oracle-target control is registered only for P-C100-100.")
    if args.method == "head-downweight" and not 0.0 <= args.head_weight <= 1.0:
        raise ValueError("--head-weight must be in [0, 1].")

    locked_steps = 250_000 if args.mode == "confirmatory" else 50_000
    if args.mode == "smoke":
        locked_steps = 5
    total_steps = args.max_steps if args.max_steps is not None else locked_steps
    if total_steps != locked_steps:
        deviations.append("total_steps")

    if deviations and args.mode == "confirmatory" and not args.allow_protocol_deviation:
        fields = ", ".join(deviations)
        raise ValueError(
            f"Confirmatory protocol deviation ({fields}). "
            "Use --allow-protocol-deviation only for explicitly non-confirmatory runs."
        )

    if args.run_id:
        run_id = args.run_id
    else:
        suffix = {"confirmatory": "confirm", "development": "dev", "smoke": "smoke"}[args.mode]
        run_id = (
            f"{protocol.protocol_id.lower()}_{args.method}_seed{args.manual_seed}_{suffix}"
            .replace("_", "-")
        )

    return RunConfig(
        protocol_id=protocol.protocol_id,
        method=args.method,
        mode=args.mode,
        manual_seed=args.manual_seed,
        data_root=args.data_root,
        output_root=args.output_root,
        run_id=run_id,
        device=args.device,
        workers=args.workers,
        download=args.download,
        resume=args.resume,
        total_steps=total_steps,
        steps_per_epoch=500,
        batch_size=32,
        unlabeled_ratio=2,
        test_batch_size=200,
        learning_rate=0.0015,
        ema_decay=0.999,
        confidence_threshold=0.95,
        lambda_u=1.0,
        tangs_beta=0.99,
        tangs_tau=args.tangs_tau,
        tangs_correction_rho=correction_rho,
        tailrow_observer=bool(getattr(args, "tailrow_observer", False)),
        tangs_warmup_steps=2500,
        min_tail_support=2,
        geometry_eps=1e-12,
        head_weight=args.head_weight,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        snapshot_every=args.snapshot_every,
        diagnostic_sample_limit=args.diagnostic_sample_limit,
        diagnostic_level=args.diagnostic_level,
        eval_pseudo_labels=args.eval_pseudo_labels,
        allow_protocol_deviation=args.allow_protocol_deviation,
        protocol=protocol,
    )


def assert_locked_substrate(config: RunConfig) -> None:
    expected = {
        "batch_size": 32,
        "unlabeled_ratio": 2,
        "test_batch_size": 200,
        "learning_rate": 0.0015,
        "ema_decay": 0.999,
        "confidence_threshold": 0.95,
        "lambda_u": 1.0,
        "steps_per_epoch": 500,
    }
    for field, value in expected.items():
        actual = getattr(config, field)
        if actual != value:
            raise AssertionError(f"Substrate drift: {field}={actual!r}, expected {value!r}")
    if config.mode == "confirmatory" and config.total_steps != 250_000:
        if not config.allow_protocol_deviation:
            raise AssertionError("Confirmatory runs must use 250,000 optimizer steps.")
