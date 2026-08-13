from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_development import (
    BASELINE_RUN,
    HEAD_WEIGHT_RUNS,
    TAU_RUNS,
    analyze_development,
)


class DevelopmentSelectionTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        run_id: str,
        method: str,
        metrics: dict[str, float],
        *,
        tau: float = 5.0,
        head_weight: float = 0.5,
        split_hash: str = "paired-split",
    ) -> None:
        run_dir = root / run_id
        run_dir.mkdir()
        payloads = {
            "summary.json": {"final_step": 50_000, "performance": metrics},
            "resolved_config.json": {
                "config_hash": f"{run_id}-hash",
                "scientific_config": {
                    "method": method,
                    "mode": "development",
                    "protocol_id": "P-C10-100",
                    "manual_seed": 0,
                    "total_steps": 50_000,
                    "tangs_tau": tau,
                    "head_weight": head_weight,
                },
            },
            "split_manifest.json": {
                "split_hash": split_hash,
                "partition_hash": "paired-partition",
                "evaluation_role": "development-validation",
            },
        }
        for filename, payload in payloads.items():
            (run_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _metrics(
        bacc: float,
        gm: float = 0.50,
        overall: float = 0.80,
        head: float = 0.80,
        medium: float = 0.60,
        tail: float = 0.40,
    ) -> dict[str, float]:
        return {
            "balanced_accuracy": bacc,
            "geometric_mean": gm,
            "overall_accuracy": overall,
            "head_accuracy": head,
            "medium_accuracy": medium,
            "tail_accuracy": tail,
        }

    def _write_complete_fixture(self, root: Path) -> None:
        self._write_run(root, BASELINE_RUN, "fixmatch", self._metrics(0.60))
        for label, value, run_id in TAU_RUNS:
            metrics = self._metrics(0.605, gm=0.505, overall=0.78, head=0.76)
            if label == "10":
                metrics = self._metrics(0.62, gm=0.51, overall=0.79, head=0.77)
            self._write_run(root, run_id, "tangs", metrics, tau=value)
        for _, value, run_id in HEAD_WEIGHT_RUNS:
            self._write_run(
                root,
                run_id,
                "head-downweight",
                self._metrics(0.60 + value / 100.0),
                head_weight=value,
            )

    def test_selects_highest_bacc_candidate_that_passes_balance_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_fixture(root)
            report = analyze_development(root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["frozen"]["tangs_tau"], "10")
        self.assertEqual(report["frozen"]["head_weight"], "0.75")

    def test_rejects_high_bacc_candidate_with_excessive_head_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_fixture(root)
            bad_run = root / "dev-c10-100-tangs-tau10-seed0" / "summary.json"
            payload = json.loads(bad_run.read_text(encoding="utf-8"))
            payload["performance"] = self._metrics(
                0.70, gm=0.55, overall=0.74, head=0.70
            )
            bad_run.write_text(json.dumps(payload), encoding="utf-8")
            report = analyze_development(root)
        self.assertEqual(report["status"], "PASS")
        self.assertNotEqual(report["frozen"]["tangs_tau"], "10")

    def test_reports_incomplete_when_a_registered_run_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_complete_fixture(root)
            missing = root / "dev-c10-100-tangs-tau2-seed0" / "summary.json"
            missing.unlink()
            report = analyze_development(root)
        self.assertEqual(report["status"], "INCOMPLETE")

    def test_tangs_only_screen_does_not_require_control_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_run(root, BASELINE_RUN, "fixmatch", self._metrics(0.60))
            for label, value, run_id in TAU_RUNS:
                self._write_run(
                    root,
                    run_id,
                    "tangs",
                    self._metrics(0.61, gm=0.51, overall=0.79, head=0.77),
                    tau=value,
                )
            report = analyze_development(
                root, require_head_downweight=False
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["gate"], "DEV-C10-100-50K-TANGS-SCREEN")
        self.assertIsNone(report["frozen"]["head_weight"])


if __name__ == "__main__":
    unittest.main()
