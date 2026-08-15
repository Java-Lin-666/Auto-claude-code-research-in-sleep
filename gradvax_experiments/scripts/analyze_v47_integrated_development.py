#!/usr/bin/env python3
"""Gate one freshly trained, integrated 50k TANGS-v4.7 C100 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tangs.artifacts import atomic_json


def _dead(performance: dict) -> int:
    return sum(value == 0 for value in performance["per_class_accuracy"])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    with (run_dir / "resolved_config.json").open(encoding="utf-8") as handle:
        resolved = json.load(handle)
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    with (run_dir / "gradient_diagnostics.json").open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    with (run_dir / "split_manifest.json").open(encoding="utf-8") as handle:
        split = json.load(handle)

    scientific = resolved["scientific_config"]
    raw = summary.get("raw_performance")
    uniform_la = summary.get("uniform_logit_adjustment_performance")
    tangs = summary.get("performance")
    if raw is None or uniform_la is None or tangs is None:
        raise RuntimeError("Run summary is missing paired raw, uniform-LA, or v4.7 metrics.")

    vs_raw = _deltas(tangs, raw)
    vs_la = _deltas(tangs, uniform_la)
    integrity = {
        "method_is_tangs_v47": scientific["method"] == "tangs-v47",
        "protocol_is_c100_100": scientific["protocol_id"] == "P-C100-100",
        "mode_is_development": scientific["mode"] == "development",
        "configured_steps_are_50000": scientific["total_steps"] == 50_000,
        "final_step_is_50000": summary["final_step"] == 50_000,
        "seed_is_zero": scientific["manual_seed"] == 0,
        "evaluation_has_5000_examples": tangs["evaluation_examples"] == 5_000,
        "uniform_la_alpha_is_0_85": scientific["score_uniform_la_alpha"] == 0.85,
        "base_alpha_is_0_65": scientific["score_base_alpha"] == 0.65,
        "extra_tail_alpha_is_0_25": scientific["score_extra_tail_alpha"] == 0.25,
        "anchor_threshold_is_0_75": scientific["score_anchor_threshold"] == 0.75,
        "zero_gradient_modification_steps": diagnostics["counts"].get(
            "nonzero_gradient_modification_steps", 0
        )
        == 0,
        "diagnostics_cover_50000_steps": diagnostics["counts"].get("total_steps")
        == 50_000,
        "observer_recorded_every_step": diagnostics["counts"].get(
            "tailrow_observed_steps"
        )
        == 50_000,
        "evaluation_role_is_development": split.get("evaluation_role")
        == "development-balanced-unused-train",
        "validation_is_disjoint_from_active_unlabeled": split.get(
            "validation_is_disjoint_from_active_unlabeled"
        )
        is True,
    }
    efficacy = {
        "bacc_vs_raw_at_least_1pp": vs_raw["balanced_accuracy_pp"] >= 1.0,
        "tail_vs_raw_at_least_2pp": vs_raw["tail_accuracy_pp"] >= 2.0,
        "head_vs_raw_no_worse_than_minus_2pp": vs_raw["head_accuracy_pp"] >= -2.0,
        "gm_vs_raw_nonnegative": vs_raw["geometric_mean_pp"] >= 0.0,
        "dead_classes_vs_raw_nonincreasing": vs_raw["dead_class_count"] <= 0,
        "bacc_vs_uniform_la_at_least_0_5pp": vs_la["balanced_accuracy_pp"] >= 0.5,
        "tail_vs_uniform_la_at_least_2pp": vs_la["tail_accuracy_pp"] >= 2.0,
    }
    passed = all(integrity.values()) and all(efficacy.values())
    report = {
        "version": "4.7-integrated-development-gate",
        "status": "PASS" if passed else "STOP",
        "allow_confirmatory_matrix": passed,
        "allow_single_c100_confirmatory_run": passed,
        "selection_role": (
            "fresh integrated 50k C100 development run on the held-out "
            "development split; official test unread"
        ),
        "run_dir": str(run_dir),
        "config_hash": summary["config_hash"],
        "split_hash": split["split_hash"],
        "raw_fixmatch": {**raw, "dead_class_count": _dead(raw)},
        "uniform_logit_adjustment": {
            **uniform_la,
            "dead_class_count": _dead(uniform_la),
        },
        "tangs_v47": {**tangs, "dead_class_count": _dead(tangs)},
        "tangs_v47_vs_raw": vs_raw,
        "tangs_v47_vs_uniform_la": vs_la,
        "integrity_checks": integrity,
        "efficacy_checks": efficacy,
    }
    destination = (
        args.output.expanduser().resolve()
        if args.output
        else run_dir.parent / "v47-integrated-development-gate.json"
    )
    atomic_json(destination, report)
    print(
        f"status={report['status']} "
        f"raw={100 * raw['balanced_accuracy']:.2f} "
        f"LA={100 * uniform_la['balanced_accuracy']:.2f} "
        f"TANGS-v47={100 * tangs['balanced_accuracy']:.2f} "
        f"tail={100 * tangs['tail_accuracy']:.2f} -> {destination}"
    )


if __name__ == "__main__":
    main()
