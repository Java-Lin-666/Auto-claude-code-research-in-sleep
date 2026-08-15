from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TailRowV46Tests(unittest.TestCase):
    def test_analytic_rows_match_autograd_for_smoothed_hard_nll(self):
        from tangs.surgery import exact_classifier_row_contributions

        torch.manual_seed(7)
        classifier = torch.nn.Linear(3, 4)
        features = torch.randn(5, 3)
        targets = torch.tensor([0, 3, 1, 3, 2])
        weights = torch.tensor([1.0, 0.0, 0.25, 2.0, 1.0])
        logits = classifier(features)
        probabilities = torch.softmax(logits, dim=1)
        per_sample = -torch.log(probabilities + 1e-8).gather(
            1, targets[:, None]
        ).squeeze(1)
        loss = (per_sample * weights).sum() / 7.0
        weight_gradient, bias_gradient = torch.autograd.grad(
            loss, list(classifier.parameters())
        )
        row_ids = torch.tensor([0, 2, 3])
        actual = exact_classifier_row_contributions(
            logits,
            features,
            targets,
            row_ids,
            weights,
            loss_denominator=7,
        )
        expected = torch.cat(
            [weight_gradient[row_ids], bias_gradient[row_ids, None]], dim=1
        )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_classwise_projection_changes_only_conflicting_tail_row(self):
        from tangs.surgery import TailRowController

        controller = TailRowController(
            "tailrow-classwise",
            num_classes=3,
            feature_dim=1,
            tail_classes=[2],
            warmup_steps=0,
        )
        # Flat Linear gradient order is all weights, then all biases.
        base = torch.tensor([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        head_rows = torch.tensor([[-2.0, 0.0]])
        supervised_self_rows = torch.tensor([[1.0, 0.0]])
        result = controller.transform(
            base,
            head_rows,
            supervised_self_rows,
            supervised_self_rows,
            torch.tensor([True]),
            step=1,
        )
        self.assertTrue(result.applied)
        torch.testing.assert_close(
            result.gradient,
            torch.tensor([10.0, 20.0, 32.0, 1.0, 2.0, 3.0]),
        )

    def test_correction_budget_caps_projection_not_whole_head_gradient(self):
        from tangs.surgery import TailRowController

        controller = TailRowController(
            "tangs-v46",
            num_classes=3,
            feature_dim=1,
            tail_classes=[2],
            correction_rho=0.25,
            warmup_steps=0,
        )
        base = torch.tensor([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        head_rows = torch.tensor([[-2.0, 0.0]])
        anchor = torch.tensor([[1.0, 0.0]])
        result = controller.transform(
            base, head_rows, anchor, anchor, torch.tensor([True]), step=1
        )
        self.assertEqual(result.stats["tailrow_budget_limited_rows"], 1)
        torch.testing.assert_close(
            result.gradient,
            torch.tensor([10.0, 20.0, 30.25, 1.0, 2.0, 3.0]),
        )

    def test_observer_updates_anchors_but_never_changes_gradient(self):
        from tangs.surgery import TailRowController

        controller = TailRowController(
            "tailrow-observer",
            num_classes=2,
            feature_dim=1,
            tail_classes=[1],
            warmup_steps=0,
        )
        base = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rows = torch.tensor([[-1.0, 0.0]])
        anchor = torch.tensor([[1.0, 0.0]])
        result = controller.transform(
            base, rows, anchor, anchor, torch.tensor([True]), step=1
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "observer-only")
        self.assertEqual(result.stats["tailrow_valid_anchor_rows"], 1)
        torch.testing.assert_close(result.gradient, base)

    def test_v47_collects_anchors_without_changing_gradient(self):
        from tangs.surgery import TailRowController

        controller = TailRowController(
            "tangs-v47",
            num_classes=2,
            feature_dim=1,
            tail_classes=[1],
            warmup_steps=0,
        )
        base = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rows = torch.tensor([[2.0, 0.0]])
        self_row = torch.tensor([[-1.0, 0.0]])
        result = controller.transform(
            base, rows, self_row, self_row, torch.tensor([True]), step=1
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "observer-only")
        torch.testing.assert_close(result.gradient, base)


if __name__ == "__main__":
    unittest.main()
