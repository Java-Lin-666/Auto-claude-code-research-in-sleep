from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_v46_development import RUNS, analyze


class V46DevelopmentGateTests(unittest.TestCase):
    @staticmethod
    def _performance(
        bacc: float,
        *,
        tail: float = 0.20,
        head: float = 0.65,
        gm: float = 0.25,
    ) -> dict:
        return {
            "balanced_accuracy": bacc,
            "geometric_mean": gm,
            "overall_accuracy": bacc,
            "head_accuracy": head,
            "medium_accuracy": bacc,
            "tail_accuracy": tail,
            "worst_class_accuracy": 0.01,
            "per_class_accuracy": [0.01] * 100,
            "evaluation_examples": 5_000,
        }

    def _write_run(
        self,
        root: Path,
        role: str,
        performance: dict,
    ) -> None:
        run_id, method, observer, rho = RUNS[role]
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=True)
        rho_value = "inf" if rho == float("inf") else rho
        payloads = {
            "summary.json": {"final_step": 50_000, "performance": performance},
            "resolved_config.json": {
                "config_hash": f"{role}-hash",
                "scientific_config": {
                    "protocol_id": "P-C100-100",
                    "method": method,
                    "mode": "development",
                    "manual_seed": 0,
                    "total_steps": 50_000,
                    "tailrow_observer": observer,
                    "tangs_correction_rho": rho_value,
                },
            },
            "split_manifest.json": {
                "split_hash": "paired-split",
                "partition_hash": "paired-partition",
                "evaluation_role": "development-balanced-unused-train",
                "validation_is_disjoint_from_active_unlabeled": True,
                "validation_indices": list(range(5_000)),
            },
            "gradient_diagnostics.json": {"tailrow": {"eligible_rows": 10}},
        }
        for name, payload in payloads.items():
            (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def _complete_fixture(self, root: Path) -> None:
        self._write_run(root, "baseline", self._performance(0.35, tail=0.10))
        self._write_run(root, "legacy", self._performance(0.352, tail=0.11))
        self._write_run(root, "tailrow_group", self._performance(0.355, tail=0.12))
        self._write_run(
            root,
            "classwise_unbounded",
            self._performance(0.360, tail=0.13),
        )
        self._write_run(
            root,
            "tangs_v46",
            self._performance(0.365, tail=0.14, head=0.64, gm=0.27),
        )

    def test_pass_requires_efficacy_specificity_and_budget_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            report = analyze(root)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["allow_confirmatory_250k"])

    def test_budget_failure_blocks_confirmatory_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._complete_fixture(root)
            self._write_run(
                root,
                "classwise_unbounded",
                self._performance(0.364, tail=0.14),
            )
            report = analyze(root)
        self.assertEqual(report["status"], "STOP")
        self.assertFalse(report["allow_confirmatory_250k"])
        self.assertFalse(report["checks"]["budget_adds_value"])


if __name__ == "__main__":
    unittest.main()
