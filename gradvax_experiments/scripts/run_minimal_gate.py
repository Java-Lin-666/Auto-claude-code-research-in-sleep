#!/usr/bin/env python3
"""Cross-platform launcher for the paired 20k TANGS spending gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from scripts.analyze_minimal_gate import GateThresholds, analyze_gate
except ModuleNotFoundError:
    from analyze_minimal_gate import GateThresholds, analyze_gate


PILOT_STEPS = 20_000
PILOT_TAU = 5.0
PILOT_RUNTIME_VERSION = "perf-v2"
FIXMATCH_ID = (
    f"pilot-c10-100-fixmatch-{PILOT_STEPS}step-seed0-{PILOT_RUNTIME_VERSION}"
)
TANGS_ID = (
    f"pilot-c10-100-tangs-tau5-{PILOT_STEPS}step-seed0-{PILOT_RUNTIME_VERSION}"
)
REPORT_NAME = f"minimal-gate-c10-100-{PILOT_STEPS}step.json"


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_train_command(
    project_root: Path,
    method: str,
    run_id: str,
    data_root: Path,
    output_root: Path,
    device: str,
    workers: int,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(project_root / "train.py"),
        "--protocol",
        "P-C10-100",
        "--method",
        method,
        "--mode",
        "development",
        "--manual-seed",
        "0",
        "--max-steps",
        str(PILOT_STEPS),
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--device",
        device,
        "--workers",
        str(workers),
        "--log-every",
        "100",
        "--checkpoint-every",
        "5000",
        "--snapshot-every",
        str(PILOT_STEPS),
        "--no-eval-pseudo-labels",
    ]
    if resume:
        command.extend(["--resume", "auto"])
    if method == "tangs":
        command.extend(["--tangs-tau", str(PILOT_TAU)])
    return command


def _display_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _run_one(
    project_root: Path,
    output_root: Path,
    data_root: Path,
    method: str,
    run_id: str,
    device: str,
    workers: int,
    dry_run: bool,
) -> None:
    run_dir = output_root / run_id
    if (run_dir / "summary.json").is_file():
        print(f"[skip completed] {run_id}", flush=True)
        return
    resume = (run_dir / "checkpoint_last.pt").is_file()
    command = build_train_command(
        project_root,
        method,
        run_id,
        data_root,
        output_root,
        device,
        workers,
        resume,
    )
    print(f"[{'dry-run' if dry_run else 'pilot'}] {_display_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=project_root, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", "./data/cache")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ROOT", "./results")),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both commands and expected paths without training.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    report_path = output_root / REPORT_NAME

    print(f"[python] {sys.executable}", flush=True)
    print(f"[project] {project_root}", flush=True)
    print(f"[data] {data_root}", flush=True)
    print(f"[output] {output_root}", flush=True)
    for method, run_id in (("fixmatch", FIXMATCH_ID), ("tangs", TANGS_ID)):
        _run_one(
            project_root,
            output_root,
            data_root,
            method,
            run_id,
            args.device,
            args.workers,
            args.dry_run,
        )

    if args.dry_run:
        print(f"[dry-run gate report] {report_path}", flush=True)
        return

    report = analyze_gate(
        output_root / FIXMATCH_ID,
        output_root / TANGS_ID,
        GateThresholds(expected_steps=PILOT_STEPS),
    )
    _atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"[gate report] {report_path}", flush=True)
    raise SystemExit({"PASS": 0, "HOLD": 2, "STOP": 3}[report["status"]])


if __name__ == "__main__":
    main()
