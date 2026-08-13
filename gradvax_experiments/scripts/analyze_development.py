#!/usr/bin/env python3
"""Freeze TANGS development choices before confirmatory training."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


TAU_RUNS = (
    ("2", 2.0, "dev-c10-100-tangs-tau2-seed0"),
    ("5", 5.0, "dev-c10-100-tangs-tau5-seed0"),
    ("10", 10.0, "dev-c10-100-tangs-tau10-seed0"),
    ("inf", math.inf, "dev-c10-100-tangs-tauinf-seed0"),
)
HEAD_WEIGHT_RUNS = (
    ("0.25", 0.25, "dev-c10-100-head-weight0.25-seed0"),
    ("0.5", 0.5, "dev-c10-100-head-weight0.5-seed0"),
    ("0.75", 0.75, "dev-c10-100-head-weight0.75-seed0"),
)
BASELINE_RUN = "dev-c10-100-fixmatch-seed0"
METRICS = (
    "balanced_accuracy",
    "geometric_mean",
    "overall_accuracy",
    "head_accuracy",
    "medium_accuracy",
    "tail_accuracy",
)


@dataclass(frozen=True)
class BalanceThresholds:
    min_bacc_gain_pp: float = 0.25
    min_gm_delta_pp: float = -0.25
    min_head_delta_pp: float = -5.0
    min_overall_delta_pp: float = -3.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pp(value: float) -> float:
    return 100.0 * value


def _same_scalar(observed: Any, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    if math.isinf(expected):
        return math.isinf(value) and value > 0
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def _load_run(
    run_dir: Path,
    expected_method: str,
    *,
    expected_tau: float | None = None,
    expected_head_weight: float | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    required = (
        run_dir / "summary.json",
        run_dir / "resolved_config.json",
        run_dir / "split_manifest.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        return None, [f"{run_dir.name}: missing {', '.join(missing)}"]

    summary = _read_json(required[0])
    resolved = _read_json(required[1])
    split = _read_json(required[2])
    config = resolved.get("scientific_config", {})
    performance = summary.get("performance", {})

    expected_fields = {
        "method": expected_method,
        "mode": "development",
        "protocol_id": "P-C10-100",
        "manual_seed": 0,
        "total_steps": 50_000,
    }
    for field, expected in expected_fields.items():
        if config.get(field) != expected:
            errors.append(
                f"{run_dir.name}: {field}={config.get(field)!r}, expected {expected!r}"
            )
    if summary.get("final_step") != 50_000:
        errors.append(
            f"{run_dir.name}: final_step={summary.get('final_step')!r}, expected 50000"
        )
    if split.get("evaluation_role") != "development-validation":
        errors.append(f"{run_dir.name}: evaluation role is not development-validation")
    for metric in METRICS:
        value = performance.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{run_dir.name}: invalid or missing metric {metric}")
    if expected_tau is not None and not _same_scalar(
        config.get("tangs_tau"), expected_tau
    ):
        errors.append(
            f"{run_dir.name}: tangs_tau={config.get('tangs_tau')!r}, "
            f"expected {expected_tau!r}"
        )
    if expected_head_weight is not None and not _same_scalar(
        config.get("head_weight"), expected_head_weight
    ):
        errors.append(
            f"{run_dir.name}: head_weight={config.get('head_weight')!r}, "
            f"expected {expected_head_weight!r}"
        )

    if errors:
        return None, errors
    return {
        "run_id": run_dir.name,
        "config_hash": resolved.get("config_hash"),
        "split_hash": split.get("split_hash"),
        "partition_hash": split.get("partition_hash"),
        "performance": {metric: float(performance[metric]) for metric in METRICS},
    }, []


def analyze_development(
    output_root: Path,
    thresholds: BalanceThresholds = BalanceThresholds(),
    *,
    require_head_downweight: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    baseline, run_errors = _load_run(output_root / BASELINE_RUN, "fixmatch")
    errors.extend(run_errors)

    tau_runs: list[tuple[str, float, dict[str, Any]]] = []
    for label, value, run_id in TAU_RUNS:
        run, run_errors = _load_run(
            output_root / run_id, "tangs", expected_tau=value
        )
        errors.extend(run_errors)
        if run is not None:
            tau_runs.append((label, value, run))

    head_runs: list[tuple[str, float, dict[str, Any]]] = []
    if require_head_downweight:
        for label, value, run_id in HEAD_WEIGHT_RUNS:
            run, run_errors = _load_run(
                output_root / run_id,
                "head-downweight",
                expected_head_weight=value,
            )
            errors.extend(run_errors)
            if run is not None:
                head_runs.append((label, value, run))

    expected_run_count = 1 + len(TAU_RUNS)
    if require_head_downweight:
        expected_run_count += len(HEAD_WEIGHT_RUNS)
    loaded_run_count = int(baseline is not None) + len(tau_runs) + len(head_runs)
    if errors or loaded_run_count != expected_run_count or baseline is None:
        return {
            "gate": "DEV-C10-100-50K-FREEZE",
            "status": "INCOMPLETE",
            "thresholds_pp": asdict(thresholds),
            "loaded_run_count": loaded_run_count,
            "expected_run_count": expected_run_count,
            "reasons": errors or ["not all registered development runs were loaded"],
        }

    pairing_errors: list[str] = []
    for _, _, run in tau_runs + head_runs:
        if run["split_hash"] != baseline["split_hash"]:
            pairing_errors.append(f"{run['run_id']}: split hash differs from FixMatch")
        if run["partition_hash"] != baseline["partition_hash"]:
            pairing_errors.append(
                f"{run['run_id']}: partition hash differs from FixMatch"
            )
    if pairing_errors:
        return {
            "gate": "DEV-C10-100-50K-FREEZE",
            "status": "STOP",
            "thresholds_pp": asdict(thresholds),
            "reasons": pairing_errors,
        }

    baseline_perf = baseline["performance"]
    candidates: list[dict[str, Any]] = []
    for label, value, run in tau_runs:
        perf = run["performance"]
        deltas = {
            metric: _pp(perf[metric] - baseline_perf[metric]) for metric in METRICS
        }
        checks = {
            "bacc_gain": deltas["balanced_accuracy"]
            >= thresholds.min_bacc_gain_pp,
            "gm_guard": deltas["geometric_mean"] >= thresholds.min_gm_delta_pp,
            "head_guard": deltas["head_accuracy"]
            >= thresholds.min_head_delta_pp,
            "overall_guard": deltas["overall_accuracy"]
            >= thresholds.min_overall_delta_pp,
        }
        candidates.append(
            {
                "tau": label,
                "tau_value": value if math.isfinite(value) else "inf",
                "run_id": run["run_id"],
                "config_hash": run["config_hash"],
                "performance": perf,
                "delta_pp_vs_fixmatch": deltas,
                "checks": checks,
                "admissible": all(checks.values()),
            }
        )

    admissible = [candidate for candidate in candidates if candidate["admissible"]]
    selected = None
    if admissible:
        selected = max(
            admissible,
            key=lambda candidate: (
                candidate["performance"]["balanced_accuracy"],
                candidate["performance"]["geometric_mean"],
                candidate["delta_pp_vs_fixmatch"]["head_accuracy"],
                candidate["delta_pp_vs_fixmatch"]["overall_accuracy"],
                float("inf")
                if candidate["tau"] == "inf"
                else float(candidate["tau"]),
            ),
        )

    head_candidates = [
        {
            "head_weight": label,
            "run_id": run["run_id"],
            "config_hash": run["config_hash"],
            "performance": run["performance"],
        }
        for label, _, run in head_runs
    ]
    selected_head = None
    if head_candidates:
        selected_head = max(
            head_candidates,
            key=lambda candidate: (
                candidate["performance"]["balanced_accuracy"],
                candidate["performance"]["geometric_mean"],
                candidate["performance"]["head_accuracy"],
            ),
        )

    status = "PASS" if selected is not None else "STOP"
    reasons = []
    if selected is None:
        reasons.append(
            "No tau candidate met the pre-confirmatory bACC, GM, Head, and Overall balance gate."
        )
    return {
        "gate": (
            "DEV-C10-100-50K-FREEZE"
            if require_head_downweight
            else "DEV-C10-100-50K-TANGS-SCREEN"
        ),
        "status": status,
        "scope": "Development-only hyperparameter freeze; never a paper table row.",
        "selection_rule": (
            "Filter by all balance checks; then maximize validation bACC, GM, "
            "Head delta, Overall delta, and tau in that order."
        ),
        "thresholds_pp": asdict(thresholds),
        "baseline": baseline,
        "tangs_candidates": candidates,
        "selected_tangs": selected,
        "head_downweight_candidates": head_candidates,
        "selected_head_downweight": selected_head,
        "frozen": {
            "tangs_tau": selected["tau"] if selected else None,
            "head_weight": selected_head["head_weight"] if selected_head else None,
        },
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tangs-only",
        action="store_true",
        help="screen the paired FixMatch/TANGS runs before spending on controls",
    )
    args = parser.parse_args()

    report = analyze_development(
        args.results, require_head_downweight=not args.tangs_only
    )
    output = args.output or args.results / "development-selection.json"
    _atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[development freeze] {report['status']}: {output}")
    return 0 if report["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
