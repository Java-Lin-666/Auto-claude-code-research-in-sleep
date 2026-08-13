from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_minimal_gate import GateThresholds, analyze_gate
from scripts.run_minimal_gate import PILOT_STEPS, build_train_command


class MinimalGateTests(unittest.TestCase):
    def test_cross_platform_launcher_uses_current_interpreter_and_resume(self):
        command = build_train_command(
            Path("/project"),
            "tangs",
            "pilot-id",
            Path("/data"),
            Path("/output"),
            "cuda",
            2,
            True,
        )
        self.assertEqual(command[0], __import__("sys").executable)
        self.assertIn(str(PILOT_STEPS), command)
        self.assertIn("--resume", command)
        self.assertIn("--tangs-tau", command)

    def _write_run(
        self,
        root: Path,
        name: str,
        method: str,
        bacc: float,
        tail: float,
        head: float,
        *,
        split_hash: str = "paired-split",
    ) -> Path:
        run_dir = root / name
        run_dir.mkdir()
        artifacts = {
            "summary.json": {
                "final_step": 20_000,
                "performance": {
                    "balanced_accuracy": bacc,
                    "tail_accuracy": tail,
                    "head_accuracy": head,
                },
            },
            "resolved_config.json": {
                "config_hash": f"{method}-hash",
                "scientific_config": {
                    "method": method,
                    "mode": "development",
                    "protocol_id": "P-C10-100",
                    "manual_seed": 0,
                },
            },
            "split_manifest.json": {
                "split_hash": split_hash,
                "partition_hash": "paired-partition",
                "evaluation_role": "development-validation",
            },
            "gradient_diagnostics.json": {
                "counts": {
                    "eligible_geometry_steps": 500,
                    "predicted_head_occurrence_steps": 520,
                    "nonzero_gradient_modification_steps": 75,
                },
                "fractions_over_eligible_steps": {"either": 0.20},
                "bounded_samples": [],
            },
        }
        for filename, payload in artifacts.items():
            (run_dir / filename).write_text(
                json.dumps(payload), encoding="utf-8"
            )
        return run_dir

    def _analyze(self, tangs_bacc: float, tangs_tail: float) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixmatch = self._write_run(
                root, "fixmatch", "fixmatch", 0.40, 0.20, 0.60
            )
            tangs = self._write_run(
                root, "tangs", "tangs", tangs_bacc, tangs_tail, 0.595
            )
            return analyze_gate(fixmatch, tangs, GateThresholds())

    def test_passes_on_noncollapse_and_positive_tail_signal(self):
        report = self._analyze(0.404, 0.21)
        self.assertEqual(report["status"], "PASS")

    def test_holds_when_mechanism_exists_but_performance_signal_is_flat(self):
        report = self._analyze(0.40, 0.20)
        self.assertEqual(report["status"], "HOLD")

    def test_stops_on_balanced_accuracy_collapse(self):
        report = self._analyze(0.39, 0.21)
        self.assertEqual(report["status"], "STOP")


if __name__ == "__main__":
    unittest.main()
