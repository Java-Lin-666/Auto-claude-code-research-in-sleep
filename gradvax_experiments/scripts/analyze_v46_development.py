#!/usr/bin/env python3
"""Audit the paired C100 v4.6 screen and decide whether 250k runs are allowed."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RUNS = {
    "baseline": (
        "v46-dev-c100-100-fixmatch-observer-seed0",
        "fixmatch",
        True,
        1.0,
    ),
    "legacy": (
        "v46-dev-c100-100-legacy-pcgrad-seed0",
        "tangs",
        False,
        1.0,
    ),
    "tailrow_group": (
        "v46-dev-c100-100-tailrow-group-seed0",
        "tailrow-group",
        False,
        1.0,
    ),
    "classwise_unbounded": (
        "v46-dev-c100-100-tailrow-classwise-seed0",
        "tailrow-classwise",
        False,
        math.inf,
    ),
    "tangs_v46": (
        "v46-dev-c100-100-tangs-rho1-seed0",
        "tangs-v46",
        False,
        1.0,
    ),
}
METRICS = (
    "balanced_accuracy",
    "geometric_mean",
    "overall_accuracy",
    "head_accuracy",
    "medium_accuracy",
    "tail_accuracy",
    "worst_class_accuracy",
)


@dataclass(frozen=True)
class GateThresholds:
    min_bacc_gain_vs_fixmatch_pp: float = 1.0
    min_tail_gain_vs_fixmatch_pp: float = 2.0
    min_head_delta_vs_fixmatch_pp: float = -2.0
    min_gm_delta_vs_fixmatch_pp: float = 0.0
    min_bacc_gain_vs_legacy_pp: float = 0.5
    min_budget_gain_vs_unbounded_pp: float = 0.25


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _same_number(observed: Any, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    if math.isinf(expected):
        return math.isinf(value) and value > 0
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def _load_run(
    root: Path,
    run_id: str,
    method: str,
    observer: bool,
    rho: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    run_dir = root / run_id
    paths = {
        name: run_dir / name
        for name in (
            "summary.json",
            "resolved_config.json",
            "split_manifest.json",
            "gradient_diagnostics.json",
        )
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return None, [f"{run_id}: missing {', '.join(missing)}"]
    summary = _read(paths["summary.json"])
    resolved = _read(paths["resolved_config.json"])
    split = _read(paths["split_manifest.json"])
    diagnostics = _read(paths["gradient_diagnostics.json"])
    config = resolved.get("scientific_config", {})
    performance = summary.get("performance", {})
    errors: list[str] = []
    expected = {
        "protocol_id": "P-C100-100",
        "method": method,
        "mode": "development",
        "manual_seed": 0,
        "total_steps": 50_000,
        "tailrow_observer": observer,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            errors.append(
                f"{run_id}: {field}={config.get(field)!r}, expected {value!r}"
            )
    if not _same_number(config.get("tangs_correction_rho"), rho):
        errors.append(
            f"{run_id}: tangs_correction_rho={config.get('tangs_correction_rho')!r}, "
            f"expected {rho!r}"
        )
    if summary.get("final_step") != 50_000:
        errors.append(f"{run_id}: final_step is not 50000")
    if split.get("evaluation_role") != "development-balanced-unused-train":
        errors.append(f"{run_id}: wrong evaluation role")
    if not split.get("validation_is_disjoint_from_active_unlabeled"):
        errors.append(f"{run_id}: validation is not disjoint from training")
    if len(split.get("validation_indices", [])) != 5_000:
        errors.append(f"{run_id}: validation cardinality is not 5000")
    if performance.get("evaluation_examples") != 5_000:
        errors.append(f"{run_id}: evaluated example count is not 5000")
    for metric in METRICS:
        value = performance.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{run_id}: invalid metric {metric}")
    per_class = performance.get("per_class_accuracy")
    if not isinstance(per_class, list) or len(per_class) != 100:
        errors.append(f"{run_id}: per_class_accuracy must contain 100 values")
    if errors:
        return None, errors
    return {
        "run_id": run_id,
        "config_hash": resolved.get("config_hash"),
        "split_hash": split.get("split_hash"),
        "partition_hash": split.get("partition_hash"),
        "performance": {metric: float(performance[metric]) for metric in METRICS},
        "dead_class_count": sum(float(value) == 0.0 for value in per_class),
        "tailrow_diagnostics": diagnostics.get("tailrow", {}),
    }, []


def _deltas(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric: 100.0 * (left[metric] - right[metric]) for metric in METRICS}


def analyze(
    results: Path, thresholds: GateThresholds = GateThresholds()
) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for role, (run_id, method, observer, rho) in RUNS.items():
        run, run_errors = _load_run(results, run_id, method, observer, rho)
        errors.extend(run_errors)
        if run is not None:
            loaded[role] = run
    if errors or len(loaded) != len(RUNS):
        return {
            "gate": "V46-DEV-C100-100-50K",
            "status": "INCOMPLETE",
            "expected_run_count": len(RUNS),
            "loaded_run_count": len(loaded),
            "thresholds_pp": asdict(thresholds),
            "reasons": errors or ["not all registered runs were loaded"],
        }

    baseline = loaded["baseline"]
    pairing_errors = []
    for role, run in loaded.items():
        if run["split_hash"] != baseline["split_hash"]:
            pairing_errors.append(f"{role}: split hash differs from baseline")
        if run["partition_hash"] != baseline["partition_hash"]:
            pairing_errors.append(f"{role}: partition hash differs from baseline")
    if pairing_errors:
        return {
            "gate": "V46-DEV-C100-100-50K",
            "status": "STOP",
            "thresholds_pp": asdict(thresholds),
            "reasons": pairing_errors,
        }

    full = loaded["tangs_v46"]
    vs_baseline = _deltas(full["performance"], baseline["performance"])
    vs_legacy = _deltas(full["performance"], loaded["legacy"]["performance"])
    vs_unbounded = _deltas(
        full["performance"], loaded["classwise_unbounded"]["performance"]
    )
    checks = {
        "bacc_gain_vs_fixmatch": vs_baseline["balanced_accuracy"]
        >= thresholds.min_bacc_gain_vs_fixmatch_pp,
        "tail_gain_vs_fixmatch": vs_baseline["tail_accuracy"]
        >= thresholds.min_tail_gain_vs_fixmatch_pp,
        "head_guard_vs_fixmatch": vs_baseline["head_accuracy"]
        >= thresholds.min_head_delta_vs_fixmatch_pp,
        "gm_guard_vs_fixmatch": vs_baseline["geometric_mean"]
        >= thresholds.min_gm_delta_vs_fixmatch_pp,
        "bacc_gain_vs_legacy": vs_legacy["balanced_accuracy"]
        >= thresholds.min_bacc_gain_vs_legacy_pp,
        "budget_adds_value": vs_unbounded["balanced_accuracy"]
        >= thresholds.min_budget_gain_vs_unbounded_pp,
        "dead_classes_not_worse": full["dead_class_count"]
        <= baseline["dead_class_count"],
    }
    status = "PASS" if all(checks.values()) else "STOP"
    reasons = []
    if not checks["budget_adds_value"]:
        reasons.append(
            "The correction budget did not beat unbounded classwise projection; "
            "do not claim norm-aware value or launch 250k v4.6 runs."
        )
    if status == "STOP" and not reasons:
        reasons.append("At least one registered efficacy or safety check failed.")
    return {
        "gate": "V46-DEV-C100-100-50K",
        "status": status,
        "scope": "Development-only; official test set is untouched.",
        "single_gpu_order": list(RUNS),
        "thresholds_pp": asdict(thresholds),
        "checks": checks,
        "runs": loaded,
        "delta_pp_tangs_v46_vs_fixmatch": vs_baseline,
        "delta_pp_tangs_v46_vs_legacy": vs_legacy,
        "delta_pp_tangs_v46_vs_classwise_unbounded": vs_unbounded,
        "frozen": {
            "method": "tangs-v46" if status == "PASS" else None,
            "tangs_correction_rho": 1.0 if status == "PASS" else None,
        },
        "allow_confirmatory_250k": status == "PASS",
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.results)
    output = args.output or args.results / "v46-development-selection.json"
    _atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[v4.6 development gate] {report['status']}: {output}")
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
