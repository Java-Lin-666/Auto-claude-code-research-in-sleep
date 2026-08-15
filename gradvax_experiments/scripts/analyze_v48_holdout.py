#!/usr/bin/env python3
"""Gate a fresh seed-1 C100 development holdout run for frozen v4.8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_v48_retrospective import PARAMETERS, _checks, _dead, _deltas
from tangs.artifacts import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    with args.selection.expanduser().resolve().open(encoding="utf-8") as handle:
        selection = json.load(handle)
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
    candidate = summary.get("performance")
    if raw is None or uniform_la is None or candidate is None:
        raise RuntimeError("Missing paired raw, uniform-LA, or v4.8 metrics.")

    selection_hashes = {
        run["split_hash"] for run in selection.get("runs", {}).values()
    }
    integrity = {
        "selection_is_provisional": selection.get("status") == "PROVISIONAL",
        "selection_allows_seed1_holdout": selection.get("allow_seed1_holdout_50k")
        is True,
        "method_is_tangs_v48": scientific["method"] == "tangs-v48",
        "protocol_is_c100_100": scientific["protocol_id"] == "P-C100-100",
        "mode_is_development": scientific["mode"] == "development",
        "seed_is_one": scientific["manual_seed"] == 1,
        "configured_steps_are_50000": scientific["total_steps"] == 50_000,
        "final_step_is_50000": summary["final_step"] == 50_000,
        "evaluation_has_5000_examples": candidate["evaluation_examples"] == 5_000,
        "evaluation_role_is_development": split.get("evaluation_role")
        == "development-balanced-unused-train",
        "validation_is_disjoint_from_active_unlabeled": split.get(
            "validation_is_disjoint_from_active_unlabeled"
        )
        is True,
        "holdout_split_differs_from_selection": split["split_hash"]
        not in selection_hashes,
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
    }
    for key, expected in PARAMETERS.items():
        integrity[f"{key}_is_frozen"] = scientific[key] == expected

    efficacy = _checks(candidate, raw, uniform_la)
    passed = all(integrity.values()) and all(efficacy.values())
    report = {
        "version": "4.8-seed1-holdout-development-gate",
        "status": "PASS" if passed else "STOP",
        "allow_confirmatory_matrix": passed,
        "selection_role": (
            "fresh seed-1 C100 development holdout with a split not used by "
            "the v4.8 retrospective selection; official test unread"
        ),
        "run_dir": str(run_dir),
        "selection": str(args.selection.expanduser().resolve()),
        "config_hash": summary["config_hash"],
        "split_hash": split["split_hash"],
        "parameters": PARAMETERS,
        "raw_fixmatch": {**raw, "dead_class_count": _dead(raw)},
        "uniform_logit_adjustment": {
            **uniform_la,
            "dead_class_count": _dead(uniform_la),
        },
        "tangs_v48": {**candidate, "dead_class_count": _dead(candidate)},
        "tangs_v48_vs_raw": _deltas(candidate, raw),
        "tangs_v48_vs_uniform_la": _deltas(candidate, uniform_la),
        "integrity_checks": integrity,
        "efficacy_checks": efficacy,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(
        f"status={report['status']} raw={100 * raw['balanced_accuracy']:.2f} "
        f"LA={100 * uniform_la['balanced_accuracy']:.2f} "
        f"TANGS-v48={100 * candidate['balanced_accuracy']:.2f} "
        f"tail={100 * candidate['tail_accuracy']:.2f} "
        f"-> {args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
