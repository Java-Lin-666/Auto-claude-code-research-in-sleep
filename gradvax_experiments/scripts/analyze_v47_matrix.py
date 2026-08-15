#!/usr/bin/env python3
"""Collect and integrity-check the required C100/C10/STL10-20 v4.7 matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tangs.artifacts import atomic_json


EXPECTED = {
    "c100": "P-C100-100",
    "c10": "P-C10-100",
    "stl10_20": "P-STL10-20",
}


def _dead(performance: dict) -> int:
    return sum(value == 0 for value in performance["per_class_accuracy"])


def _compact(performance: dict) -> dict:
    return {
        "balanced_accuracy": performance["balanced_accuracy"],
        "head_accuracy": performance["head_accuracy"],
        "medium_accuracy": performance["medium_accuracy"],
        "tail_accuracy": performance["tail_accuracy"],
        "geometric_mean": performance["geometric_mean"],
        "worst_class_accuracy": performance["worst_class_accuracy"],
        "dead_class_count": _dead(performance),
    }


def _deltas(candidate: dict, reference: dict) -> dict[str, float | int]:
    return {
        "balanced_accuracy_pp": 100
        * (candidate["balanced_accuracy"] - reference["balanced_accuracy"]),
        "head_accuracy_pp": 100
        * (candidate["head_accuracy"] - reference["head_accuracy"]),
        "medium_accuracy_pp": 100
        * (candidate["medium_accuracy"] - reference["medium_accuracy"]),
        "tail_accuracy_pp": 100
        * (candidate["tail_accuracy"] - reference["tail_accuracy"]),
        "geometric_mean_pp": 100
        * (candidate["geometric_mean"] - reference["geometric_mean"]),
        "dead_class_count": _dead(candidate) - _dead(reference),
    }


def _load_cell(name: str, run_dir: Path) -> tuple[dict, dict]:
    with (run_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        resolved = json.load(handle)
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (run_dir / "gradient_diagnostics.json").open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)

    scientific = resolved["scientific_config"]
    raw = summary.get("raw_performance")
    uniform_la = summary.get("uniform_logit_adjustment_performance")
    tangs = summary.get("performance")
    if raw is None or uniform_la is None or tangs is None:
        raise RuntimeError(f"{name} is missing paired raw, uniform-LA, or v4.7 metrics")
    checks = {
        "method_is_tangs_v47": scientific["method"] == "tangs-v47",
        "protocol_matches": scientific["protocol_id"] == EXPECTED[name],
        "mode_is_confirmatory": scientific["mode"] == "confirmatory",
        "configured_steps_are_250000": scientific["total_steps"] == 250_000,
        "final_step_is_250000": summary["final_step"] == 250_000,
        "seed_is_zero": scientific["manual_seed"] == 0,
        "uniform_la_alpha_is_0_85": scientific["score_uniform_la_alpha"] == 0.85,
        "base_alpha_is_0_65": scientific["score_base_alpha"] == 0.65,
        "extra_tail_alpha_is_0_25": scientific["score_extra_tail_alpha"] == 0.25,
        "anchor_threshold_is_0_75": scientific["score_anchor_threshold"] == 0.75,
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
    cell = {
        "run_dir": str(run_dir),
        "protocol_id": EXPECTED[name],
        "config_hash": summary["config_hash"],
        "raw_fixmatch": _compact(raw),
        "uniform_logit_adjustment": _compact(uniform_la),
        "tangs_v47": _compact(tangs),
        "tangs_v47_vs_raw": _deltas(tangs, raw),
        "tangs_v47_vs_uniform_la": _deltas(tangs, uniform_la),
        "integrity_checks": checks,
    }
    return cell, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c100-run-dir", type=Path, required=True)
    parser.add_argument("--c10-run-dir", type=Path, required=True)
    parser.add_argument("--stl20-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "c100": args.c100_run_dir.expanduser().resolve(),
        "c10": args.c10_run_dir.expanduser().resolve(),
        "stl10_20": args.stl20_run_dir.expanduser().resolve(),
    }
    cells: dict[str, dict] = {}
    all_checks: list[bool] = []
    for name, run_dir in paths.items():
        cells[name], checks = _load_cell(name, run_dir)
        all_checks.extend(checks.values())

    integrity_ok = all(all_checks)
    report = {
        "version": "4.7-required-three-dataset-matrix",
        "status": "COMPLETE" if integrity_ok else "INVALID",
        "integrity_ok": integrity_ok,
        "claim_review": "PENDING-INDEPENDENT-AUDIT",
        "note": (
            "COMPLETE means all three frozen 250k runs and artifacts are present; "
            "it is not an efficacy PASS and not an independent paper-claim audit."
        ),
        "cells": cells,
    }
    destination = args.output.expanduser().resolve()
    atomic_json(destination, report)
    print(f"status={report['status']} required_cells=3 -> {destination}")
    if not integrity_ok:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
