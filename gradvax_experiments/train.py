#!/usr/bin/env python3
"""Server entry point for the locked v4.6 FixMatch/TANGS experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tangs.artifacts import prepare_run
from tangs.config import METHODS, PROTOCOLS, build_run_config
from tangs.data import build_data
from tangs.trainer import Trainer, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TANGS on the pinned CDMAD FixMatch substrate."
    )
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument(
        "--mode",
        choices=["confirmatory", "development", "smoke"],
        default="confirmatory",
    )
    parser.add_argument("--manual-seed", type=int, default=0)
    parser.add_argument("--data-root", default="./data/cache")
    parser.add_argument("--output-root", default="./results")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume from checkpoint_last.pt, or from an explicit checkpoint path.",
    )

    parser.add_argument(
        "--tangs-tau",
        type=float,
        default=5.0,
        help="TANGS norm-cap multiplier; use inf for the no-cap value.",
    )
    parser.add_argument(
        "--tangs-correction-rho",
        type=float,
        default=1.0,
        help="v4.6 per-class correction budget multiplier; use inf for unbounded.",
    )
    parser.add_argument(
        "--tailrow-observer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure v4.6 tail-row geometry without changing the gradient.",
    )
    parser.add_argument(
        "--head-weight",
        type=float,
        default=0.5,
        help="Tuned coefficient used only by head-downweight.",
    )
    parser.add_argument(
        "--diagnostic-level",
        choices=["basic", "deep"],
        default="basic",
    )
    parser.add_argument("--diagnostic-sample-limit", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--snapshot-every", type=int, default=50_000)
    parser.add_argument(
        "--eval-pseudo-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        help="Debug override; marks a confirmatory run as a protocol deviation.",
    )
    parser.add_argument(
        "--allow-protocol-deviation",
        action="store_true",
        help="Explicitly mark and allow a non-confirmatory protocol deviation.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Rejected by v4.6; present so accidental AMP requests fail clearly.",
    )
    args = parser.parse_args()
    if args.log_every <= 0 or args.checkpoint_every <= 0 or args.snapshot_every <= 0:
        parser.error("Logging and checkpoint cadences must be positive.")
    if args.diagnostic_sample_limit <= 0:
        parser.error("--diagnostic-sample-limit must be positive.")
    return args


def main() -> None:
    args = parse_args()
    config = build_run_config(args)
    seed_everything(config.manual_seed)
    data = build_data(config)
    project_root = Path(__file__).resolve().parent.parent
    run_dir, digest = prepare_run(
        config, data.split_manifest, project_root=project_root
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "protocol": config.protocol_id,
                "method": config.method,
                "mode": config.mode,
                "steps": config.total_steps,
                "manualSeed": config.manual_seed,
                "config_hash": digest,
                "run_dir": str(run_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    summary = Trainer(config, data, run_dir, digest).train()
    print(json.dumps(summary["performance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
