#!/usr/bin/env python3
"""Dependency-free status view for local or remote experiment directories."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def render(results: Path) -> None:
    rows = []
    for status_path in sorted(results.glob("*/status.json")):
        status = load_json(status_path)
        run_id = status_path.parent.name
        summary = load_json(status_path.parent / "summary.json")
        performance = summary.get("performance", {})
        rows.append(
            (
                run_id,
                status.get("state", "unknown"),
                int(status.get("step", 0)),
                int(status.get("total_steps", 0)),
                status.get("steps_per_second"),
                status.get("eta_seconds"),
                performance.get("balanced_accuracy"),
                performance.get("tail_accuracy"),
            )
        )
    print(
        f"{'RUN':48} {'STATE':12} {'STEP':>15} {'STEP/S':>9} "
        f"{'ETA(H)':>9} {'BACC':>8} {'TAIL':>8}"
    )
    for run_id, state, step, total, rate, eta, bacc, tail in rows:
        step_text = f"{step}/{total}"
        rate_text = "-" if rate is None else f"{rate:.3f}"
        eta_text = "-" if eta is None else f"{eta / 3600:.2f}"
        bacc_text = "-" if bacc is None else f"{100 * bacc:.2f}"
        tail_text = "-" if tail is None else f"{100 * tail:.2f}"
        print(
            f"{run_id:48.48} {state:12.12} {step_text:>15} {rate_text:>9} "
            f"{eta_text:>9} {bacc_text:>8} {tail_text:>8}"
        )
    if not rows:
        print(f"No run status files found under {results.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("./results"))
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="Refresh continuously at this interval.",
    )
    args = parser.parse_args()
    while True:
        render(args.results)
        if args.watch is None:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
