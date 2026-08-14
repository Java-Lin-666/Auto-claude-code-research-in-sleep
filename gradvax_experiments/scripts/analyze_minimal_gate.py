#!/usr/bin/env python3
"""Decide whether the cheap paired pilot justifies larger TANGS runs."""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class GateThresholds:
    expected_steps: int = 20_000
    min_eligible_steps: int = 100
    min_anchor_coverage: float = 0.90
    min_trigger_fraction: float = 0.10
    max_bacc_regression_pp: float = 0.50
    min_bacc_signal_pp: float = 0.25
    min_tail_signal_pp: float = 0.50


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_artifacts(run_dir: Path) -> dict[str, Any]:
    required = {
        "summary": run_dir / "summary.json",
        "config": run_dir / "resolved_config.json",
        "split": run_dir / "split_manifest.json",
        "diagnostics": run_dir / "gradient_diagnostics.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing pilot artifacts: " + ", ".join(missing))
    return {name: _load(path) for name, path in required.items()}


def _pp(value: float) -> float:
    return 100.0 * float(value)


def analyze_gate(
    fixmatch_dir: Path,
    tangs_dir: Path,
    thresholds: GateThresholds,
) -> dict[str, Any]:
    fixmatch = _run_artifacts(fixmatch_dir)
    tangs = _run_artifacts(tangs_dir)
    failures: list[str] = []

    expected = (
        (fixmatch, "fixmatch", "FixMatch"),
        (tangs, "tangs", "TANGS"),
    )
    for artifacts, method, label in expected:
        config = artifacts["config"].get("scientific_config", artifacts["config"])
        summary = artifacts["summary"]
        if config.get("method") != method:
            failures.append(f"{label} artifact has method={config.get('method')!r}")
        if config.get("mode") != "development":
            failures.append(f"{label} pilot is not a development run")
        if config.get("protocol_id") != "P-C10-100":
            failures.append(f"{label} pilot is not P-C10-100")
        if int(config.get("manual_seed", -1)) != 0:
            failures.append(f"{label} pilot is not manualSeed=0")
        if int(summary.get("final_step", -1)) != thresholds.expected_steps:
            failures.append(
                f"{label} final_step={summary.get('final_step')}, "
                f"expected {thresholds.expected_steps}"
            )

    for key in ("split_hash", "partition_hash"):
        if fixmatch["split"].get(key) != tangs["split"].get(key):
            failures.append(f"paired pilot has mismatched {key}")
    if fixmatch["split"].get("evaluation_role") != "development-validation":
        failures.append("FixMatch evaluation role is not development-validation")
    if tangs["split"].get("evaluation_role") != "development-validation":
        failures.append("TANGS evaluation role is not development-validation")

    fix_perf = fixmatch["summary"]["performance"]
    tangs_perf = tangs["summary"]["performance"]
    bacc_delta = _pp(tangs_perf["balanced_accuracy"] - fix_perf["balanced_accuracy"])
    tail_delta = _pp(tangs_perf["tail_accuracy"] - fix_perf["tail_accuracy"])
    head_delta = _pp(tangs_perf["head_accuracy"] - fix_perf["head_accuracy"])

    diagnostics = tangs["diagnostics"]
    counts = diagnostics.get("counts", {})
    eligible_steps = int(counts.get("eligible_geometry_steps", 0))
    predicted_head_steps = int(counts.get("predicted_head_occurrence_steps", 0))
    anchor_coverage = eligible_steps / max(predicted_head_steps, 1)
    trigger_fraction = float(
        diagnostics.get("fractions_over_eligible_steps", {}).get("either", 0.0)
    )
    nonzero_modification_steps = int(
        counts.get("nonzero_gradient_modification_steps", 0)
    )

    mechanism_checks = {
        "eligible_steps": eligible_steps >= thresholds.min_eligible_steps,
        "anchor_coverage": anchor_coverage >= thresholds.min_anchor_coverage,
        "trigger_fraction": trigger_fraction >= thresholds.min_trigger_fraction,
        "observed_nonzero_modification": nonzero_modification_steps > 0,
    }
    performance_checks = {
        "no_bacc_collapse": bacc_delta >= -thresholds.max_bacc_regression_pp,
        "positive_early_signal": (
            bacc_delta >= thresholds.min_bacc_signal_pp
            or tail_delta >= thresholds.min_tail_signal_pp
        ),
    }

    if failures or not all(mechanism_checks.values()) or not performance_checks["no_bacc_collapse"]:
        status = "STOP"
    elif performance_checks["positive_early_signal"]:
        status = "PASS"
    else:
        status = "HOLD"

    reasons = list(failures)
    reasons.extend(name for name, passed in mechanism_checks.items() if not passed)
    reasons.extend(name for name, passed in performance_checks.items() if not passed)
    recommendation = {
        "PASS": "Proceed to the registered 50k development freeze; do not launch 250k runs before tuning is frozen.",
        "HOLD": "Do not launch the full matrix. Extend only the paired FixMatch/TANGS development comparison to 50k.",
        "STOP": "Do not launch the 50k sweep or 250k matrix. Inspect diagnostics or revise the method first.",
    }[status]

    return {
        "gate": "MINIMAL-C10-100-20K",
        "status": status,
        "recommendation": recommendation,
        "reasons": reasons,
        "paired_runs": {
            "fixmatch": str(fixmatch_dir.resolve()),
            "tangs": str(tangs_dir.resolve()),
            "split_hash": tangs["split"].get("split_hash"),
            "partition_hash": tangs["split"].get("partition_hash"),
        },
        "performance_pp": {
            "fixmatch_bacc": _pp(fix_perf["balanced_accuracy"]),
            "tangs_bacc": _pp(tangs_perf["balanced_accuracy"]),
            "bacc_delta": bacc_delta,
            "fixmatch_tail": _pp(fix_perf["tail_accuracy"]),
            "tangs_tail": _pp(tangs_perf["tail_accuracy"]),
            "tail_delta": tail_delta,
            "head_delta": head_delta,
        },
        "mechanism": {
            "eligible_steps": eligible_steps,
            "predicted_head_occurrence_steps": predicted_head_steps,
            "anchor_coverage": anchor_coverage,
            "conflict_or_domination_fraction": trigger_fraction,
            "nonzero_gradient_modification_steps": nonzero_modification_steps,
        },
        "checks": {
            "artifact_and_pairing": not failures,
            **mechanism_checks,
            **performance_checks,
        },
        "thresholds": asdict(thresholds),
        "scope": "Development validation only; this gate is not paper evidence.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixmatch-dir", type=Path, required=True)
    parser.add_argument("--tangs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_gate(
        args.fixmatch_dir,
        args.tangs_dir,
        GateThresholds(expected_steps=args.expected_steps),
    )
    _atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit({"PASS": 0, "HOLD": 2, "STOP": 3}[report["status"]])


if __name__ == "__main__":
    main()
