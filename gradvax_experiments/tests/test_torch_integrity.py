from __future__ import annotations

import importlib.util
import gc
import json
import math
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TorchIntegrityTests(unittest.TestCase):
    def test_step_releases_autograd_graph_and_anchor_is_detached(self):
        from tangs.surgery import TangsController, classifier_gradient

        classifier = torch.nn.Linear(3, 2)
        controller = TangsController("tangs", warmup_steps=0)

        def one_step():
            inputs = torch.randn(4, 3)
            targets = torch.tensor([0, 1, 0, 1])
            logits = classifier(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            gradient = classifier_gradient(
                loss, list(classifier.parameters()), retain_graph=True
            )
            loss.backward()
            base = torch.cat([p.grad.reshape(-1) for p in classifier.parameters()])
            controller.transform(base, gradient, gradient, step=1)
            classifier.zero_grad(set_to_none=True)
            return weakref.ref(logits), weakref.ref(loss)

        for _ in range(5):
            logits_ref, loss_ref = one_step()
            gc.collect()
            self.assertIsNone(logits_ref())
            self.assertIsNone(loss_ref())
        self.assertIsNotNone(controller.anchor)
        self.assertFalse(controller.anchor.requires_grad)
        self.assertIsNone(controller.anchor.grad_fn)

    def test_first_anchor_empty_head_and_warmup_boundary(self):
        from tangs.surgery import TangsController

        base = torch.tensor([2.0, 1.0])
        tail = torch.tensor([1.0, 0.0])
        head = torch.tensor([-1.0, 2.0])
        controller = TangsController("tangs", beta=0.99, warmup_steps=2)
        empty = controller.transform(base, None, tail, step=1)
        torch.testing.assert_close(controller.anchor, tail)
        self.assertEqual(controller.anchor_updates, 1)
        self.assertEqual(empty.reason, "empty-or-invalid-head")
        torch.testing.assert_close(empty.gradient, base)
        boundary = controller.transform(base, head, tail, step=2)
        self.assertEqual(boundary.reason, "warmup")
        after = controller.transform(base, head, tail, step=3)
        self.assertTrue(after.applied)
        self.assertEqual(after.reason, "applied")

    def test_projection_can_drop_below_cap_without_scaling_up(self):
        from tangs.surgery import TangsController

        base = torch.tensor([4.0, 2.0])
        head = torch.tensor([-3.0, 1.0])
        tail = torch.tensor([1.0, 0.0])
        controller = TangsController("tangs", tau=2.0, warmup_steps=0)
        result = controller.transform(base, head, tail, step=1)
        modified = result.gradient - base + head
        self.assertGreater(result.stats["raw_ratio"], 2.0)
        self.assertLess(result.stats["post_projection_ratio"], 2.0)
        self.assertFalse(result.stats["capped"])
        torch.testing.assert_close(modified, torch.tensor([0.0, 1.0]))

    def test_resume_rolls_jsonl_back_to_checkpoint_without_duplicates(self):
        from tangs.trainer import _rollback_step_jsonl

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_metrics.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 50, "value": 1}),
                        json.dumps({"step": 100, "value": 2}),
                        json.dumps({"step": 150, "value": 3}),
                        '{"step": 200',
                    ]
                ),
                encoding="utf-8",
            )
            retained = _rollback_step_jsonl(path, 100)
            self.assertEqual([row["step"] for row in retained], [50, 100])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [50, 100])

    def test_resume_rng_states_are_moved_back_to_cpu(self):
        from tangs.trainer import _restore_torch_rng_states

        cpu_state = Mock()
        cuda_state = Mock()
        with (
            patch("tangs.trainer.torch.set_rng_state") as set_cpu,
            patch("tangs.trainer.torch.cuda.is_available", return_value=True),
            patch("tangs.trainer.torch.cuda.set_rng_state_all") as set_cuda,
        ):
            _restore_torch_rng_states(
                {"torch_cpu": cpu_state, "torch_cuda": [cuda_state]}
            )
        cpu_state.cpu.assert_called_once_with()
        cuda_state.cpu.assert_called_once_with()
        set_cpu.assert_called_once_with(cpu_state.cpu.return_value)
        set_cuda.assert_called_once_with([cuda_state.cpu.return_value])

    def test_diagnostics_counts_nonzero_gradient_modifications(self):
        from tangs.trainer import Diagnostics

        diagnostics = Diagnostics(sample_limit=4, seed=0)
        result = SimpleNamespace(
            applied=True,
            reason="applied",
            stats={
                "gradient_delta_norm": 0.25,
                "effective_cosine": -0.5,
                "raw_ratio": 2.0,
                "post_projection_ratio": 1.5,
                "tau": 5.0,
            },
        )
        diagnostics.observe(1, 2, 3, result, None)
        self.assertEqual(
            diagnostics.counts["nonzero_gradient_modification_steps"], 1
        )

    def test_tensor_bytes_counts_unique_live_tensors(self):
        from tangs.trainer import _tensor_bytes

        first = torch.zeros(3, dtype=torch.float32)
        second = torch.zeros(2, dtype=torch.float64)
        self.assertEqual(_tensor_bytes(first, first, second, None), 28)

    def test_geometry_projection_cap_and_noop(self):
        from tangs.surgery import TangsController

        base = torch.tensor([4.0, 3.0])
        head = torch.tensor([-3.0, 4.0])
        tail = torch.tensor([1.0, 0.0])
        controller = TangsController("tangs", tau=2.0, warmup_steps=0)
        result = controller.transform(base, head, tail, step=1)
        self.assertTrue(result.applied)
        self.assertAlmostEqual(float(torch.dot(result.gradient - base + head, tail)), 0.0)
        self.assertLessEqual(result.stats["final_ratio"], 2.0 + 1e-6)

        noop = TangsController("no-projection", tau=math.inf, warmup_steps=0)
        noop_result = noop.transform(base, head, tail, step=1)
        torch.testing.assert_close(noop_result.gradient, base, rtol=1e-7, atol=1e-7)

    def test_warmup_and_invalid_anchor_are_noops(self):
        from tangs.surgery import TangsController

        base = torch.tensor([1.0, 2.0])
        head = torch.tensor([-1.0, 0.0])
        controller = TangsController("tangs", warmup_steps=2)
        warmup = controller.transform(base, head, torch.tensor([1.0, 0.0]), 2)
        self.assertEqual(warmup.reason, "warmup")
        torch.testing.assert_close(warmup.gradient, base)

        invalid = TangsController("tangs", warmup_steps=0)
        skipped = invalid.transform(base, head, None, 1)
        self.assertEqual(skipped.reason, "invalid-anchor")
        torch.testing.assert_close(skipped.gradient, base)

    def test_pcgrad_and_no_cap_are_the_same_locked_operator(self):
        from tangs.surgery import TangsController

        base = torch.tensor([2.0, -1.0, 0.5])
        head = torch.tensor([-3.0, 4.0, 1.0])
        tail = torch.tensor([1.0, 0.0, 0.0])
        pcgrad = TangsController("pcgrad", tau=2.0, warmup_steps=0)
        no_cap = TangsController("no-cap", tau=2.0, warmup_steps=0)

        pcgrad_result = pcgrad.transform(base, head, tail, step=1)
        no_cap_result = no_cap.transform(base, head, tail, step=1)

        torch.testing.assert_close(pcgrad_result.gradient, no_cap_result.gradient)
        self.assertEqual(pcgrad_result.reason, no_cap_result.reason)
        self.assertEqual(
            pcgrad_result.stats["projected"], no_cap_result.stats["projected"]
        )
        self.assertFalse(pcgrad_result.stats["capped"])

    def test_weightema_matches_released_one_step_including_buffers(self):
        from tangs.ema import WeightEMA

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
                self.linear = torch.nn.Linear(2, 2)

        torch.manual_seed(11)
        model = Tiny()
        ema = Tiny()
        updater = WeightEMA(
            model, ema, learning_rate=0.0015, decay_coefficient=0.08, alpha=0.999
        )
        with torch.no_grad():
            for value in model.state_dict().values():
                if value.is_floating_point():
                    value.add_(0.25)
        online_before = {
            key: value.clone() for key, value in model.state_dict().items()
        }
        ema_before = {key: value.clone() for key, value in ema.state_dict().items()}
        updater.step()

        for key, actual in ema.state_dict().items():
            if actual.is_floating_point():
                expected = 0.999 * ema_before[key] + 0.001 * online_before[key]
                torch.testing.assert_close(actual, expected)
            else:
                torch.testing.assert_close(actual, ema_before[key])
        factor = 1 - 0.08 * 0.0015
        for key, actual in model.state_dict().items():
            if actual.is_floating_point():
                torch.testing.assert_close(actual, online_before[key] * factor)
            else:
                torch.testing.assert_close(actual, online_before[key])

    def test_wrn_rotation_and_batchnorm_execution_values(self):
        from tangs.wrn import BatchNorm2d, WRN

        model = WRN(2, num_classes=100)
        self.assertTrue(model.rotation)
        self.assertEqual(model.rot.out_features, 4)
        batch_norms = [
            module for module in model.modules() if isinstance(module, BatchNorm2d)
        ]
        self.assertTrue(batch_norms)
        self.assertTrue(all(module.eps == 1e-5 for module in batch_norms))
        self.assertTrue(all(module.momentum == 0.1 for module in batch_norms))

    def test_local_wrn_state_matches_pinned_checkout_when_present(self):
        from tangs.wrn import WRN

        source = (
            Path(__file__).resolve().parents[2]
            / "CDMAD"
            / "wrn.py"
        )
        if not source.exists():
            self.skipTest("Sibling pinned CDMAD checkout is not present")
        spec = importlib.util.spec_from_file_location("pinned_cdmad_wrn", source)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        torch.manual_seed(7)
        expected = module.WRN(2, num_classes=10)
        torch.manual_seed(7)
        actual = WRN(2, num_classes=10)
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for key in expected.state_dict():
            torch.testing.assert_close(
                expected.state_dict()[key], actual.state_dict()[key]
            )


if __name__ == "__main__":
    unittest.main()
