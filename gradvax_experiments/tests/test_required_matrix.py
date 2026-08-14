from __future__ import annotations

import re
import unittest
from pathlib import Path


class RequiredMatrixTests(unittest.TestCase):
    def test_smoke_script_skips_completed_and_resumes_checkpoint(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "smoke_test.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('summary.json', script)
        self.assertIn('checkpoint_last.pt', script)
        self.assertIn('resume=(--resume auto)', script)
        self.assertIn('SMOKE_RUN_ID', script)

    def test_required_script_launches_10_unique_jobs_and_reuses_reported_baselines(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_required.sh"
        ).read_text(encoding="utf-8")
        calls = re.findall(r"^run_cell\s+\S+\s+\S+\s+(\S+)", script, re.MULTILINE)

        self.assertEqual(len(calls), 10)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertIn("minimal-gate-c10-100-20000step.json", script)
        self.assertIn('if [[ "$GATE_STATUS" != "PASS" ]]', script)
        self.assertIn("development-selection.json", script)
        self.assertIn('if [[ "$DEV_STATUS" != "PASS" ]]', script)
        self.assertIn('TANGS_TAU="${TANGS_TAU:-$FROZEN_TAU}"', script)
        self.assertIn('HEAD_WEIGHT="${HEAD_WEIGHT:-$FROZEN_HEAD_WEIGHT}"', script)
        self.assertIn("c2-c100-100-pcgrad-seed0", calls)
        self.assertNotIn("d1-c100-100-no-cap-seed0", calls)
        self.assertFalse(
            any(
                run_id.startswith("c1-") and "fixmatch" in run_id
                for run_id in calls
            )
        )
        self.assertIn("c1-c10-100-tangs-seed0", calls)
        self.assertIn("c1-c100-100-tangs-seed0", calls)
        self.assertIn("c1-stl10-20-tangs-seed0", calls)
        self.assertIn("ALLOW_LEGACY_V45_REQUIRED", script)

    def test_v46_development_is_a_resumable_single_gpu_five_run_queue(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_v46_development.sh"
        ).read_text(encoding="utf-8")
        calls = re.findall(r"^run_dev\s+(\S+)\s+(\S+)", script, re.MULTILINE)
        self.assertEqual(len(calls), 5)
        self.assertEqual(
            [method for method, _ in calls],
            [
                "fixmatch",
                "tangs",
                "tailrow-group",
                "tailrow-classwise",
                "tangs-v46",
            ],
        )
        self.assertIn('CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"', script)
        self.assertIn('if [[ -f "${run_dir}/summary.json" ]]', script)
        self.assertIn('resume=(--resume auto)', script)
        self.assertIn("--mode development", script)
        self.assertIn("scripts/analyze_v46_development.py", script)

    def test_v46_confirmatory_pair_is_blocked_until_gate_passes(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_required_v46.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("v46-development-selection.json", script)
        self.assertIn('if [[ "$STATUS" != "PASS" || "$ALLOW" != "True" ]]', script)
        calls = re.findall(r"^run_cell\s+(\S+)\s+(\S+)", script, re.MULTILINE)
        self.assertEqual([method for method, _ in calls], ["fixmatch", "tangs-v46"])
        self.assertIn('resume=(--resume auto)', script)

    def test_development_sweep_requires_a_passed_minimal_gate(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_development.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("minimal-gate-c10-100-20000step.json", script)
        self.assertIn('if [[ "$GATE_STATUS" != "PASS" ]]', script)
        self.assertIn("--max-steps 50000", script)
        self.assertIn('run_dev fixmatch "dev-c10-100-fixmatch-seed0"', script)
        self.assertIn("for tau in 10 inf 5 2", script)
        self.assertIn("scripts/analyze_development.py", script)
        self.assertIn("--tangs-only", script)


if __name__ == "__main__":
    unittest.main()
