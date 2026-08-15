#!/usr/bin/env python3
"""Collect the frozen C100/C10/STL10-20 v4.8 confirmatory matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_v48_retrospective import PARAMETERS, _dead, _deltas
from tangs.artifacts import atomic_json


EXPECTED = {
    "c100": "P-C100-100",
    "c10": "P-C10-100",
    "stl10_20": "P-STL10-20",
}


def _compact(performance: dict) -> dict:
    return {
        key: performance[key]
        for key in (
            "balanced_accuracy",
            "head_accuracy",
            "medium_accuracy",
            "tail_accuracy",
            "geometric_mean",
            "worst_class_accuracy",
        )
    } | {"dead_class_count": _dead(performance)}


def _load(name: str, run_dir: Path) -> tuple[dict, list[bool]]:
    with (run_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        scientific = json.load(handle)["scientific_config"]
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (run_dir / "gradient_diagnostics.json").open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    raw = summary["raw_performance"]
    uniform_la = summary["uniform_logit_adjustment_performance"]
    candidate = summary["performance"]
    checks = {
        "method_is_tangs_v48": scientific["method"] == "tangs-v48",
        "protocol_matches": scientific["protocol_id"] == EXPECTED[name],
        "mode_is_confirmatory": scientific["mode"] == "confirmatory",
        "seed_is_zero": scientific["manual_seed"] == 0,
        "configured_steps_are_250000": scientific["total_steps"] == 250_000,
        "final_step_is_250000": summary["final_step"] == 250_000,
        "zero_gradient_modification_steps": diagnostics["counts"].get(
            "nonzero_gradient_modification_steps", 0
        )
        == 0,
        "diagnostics_cover_250000_steps": diagnostics["counts"].get("total_steps")
        == 250_000,
        "observer_recorded_every_step": diagnostics["counts"].get(
            "tailrow_observed_steps"
        )
        == 250_000,
    }
    for key, expected in PARAMETERS.items():
        checks[f"{key}_is_frozen"] = scientific[key] == expected
    return (
        {
            "run_dir": str(run_dir),
            "protocol_id": EXPECTED[name],
            "config_hash": summary["config_hash"],
            "raw_fixmatch": _compact(raw),
            "uniform_logit_adjustment": _compact(uniform_la),
            "tangs_v48": _compact(candidate),
            "tangs_v48_vs_raw": _deltas(candidate, raw),
            "tangs_v48_vs_uniform_la": _deltas(candidate, uniform_la),
            "integrity_checks": checks,
        },
        list(checks.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c100-run-dir", type=Path, required=True)
    parser.add_argument("--c10-run-dir", type=Path, required=True)
    parser.add_argument("--stl20-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "c100": args.c100_run_dir.resolve(),
        "c10": args.c10_run_dir.resolve(),
        "stl10_20": args.stl20_run_dir.resolve(),
    }
    cells = {}
    checks = []
    for name, path in paths.items():
        cells[name], values = _load(name, path)
        checks.extend(values)
    integrity_ok = all(checks)
    report = {
        "version": "4.8-required-three-dataset-matrix",
        "status": "COMPLETE" if integrity_ok else "INVALID",
        "integrity_ok": integrity_ok,
        "claim_review": "PENDING-INDEPENDENT-AUDIT",
        "parameters": PARAMETERS,
        "cells": cells,
    }
    atomic_json(args.output.resolve(), report)
    print(f"status={report['status']} required_cells=3 -> {args.output.resolve()}")
    if not integrity_ok:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
