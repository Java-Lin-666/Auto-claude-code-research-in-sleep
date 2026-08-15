from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class CalibrationV47Tests(unittest.TestCase):
    def test_applies_global_prior_and_one_eligible_tail_boost(self):
        from tangs.calibration import anchor_gated_logits

        logits = torch.zeros(2, 3)
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        counts = torch.tensor([100.0, 10.0, 1.0])
        # Negative weight coordinates are the feature prototypes.
        anchors = torch.tensor([[-1.0, 0.0, -0.5], [0.0, -1.0, -0.5]])
        actual = anchor_gated_logits(
            logits,
            features,
            counts,
            [1, 2],
            anchors,
            torch.tensor([True, True]),
            base_alpha=0.5,
            extra_tail_alpha=0.25,
            anchor_threshold=0.75,
        )
        log_prior = (counts / counts.sum()).log()
        expected = -0.5 * log_prior.unsqueeze(0).repeat(2, 1)
        expected[0, 1] += 0.25 * -log_prior[1]
        expected[1, 2] += 0.25 * -log_prior[2]
        torch.testing.assert_close(actual, expected)

    def test_below_threshold_receives_no_extra_tail_boost(self):
        from tangs.calibration import anchor_gated_logits

        logits = torch.zeros(1, 2)
        actual = anchor_gated_logits(
            logits,
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([10.0, 1.0]),
            [1],
            torch.tensor([[-1.0, 0.0, -0.5]]),
            torch.tensor([True]),
            base_alpha=0.65,
            extra_tail_alpha=0.25,
            anchor_threshold=0.75,
        )
        prior = torch.tensor([10.0, 1.0]) / 11.0
        torch.testing.assert_close(actual, -0.65 * prior.log().unsqueeze(0))

    def test_invalid_anchor_falls_back_to_global_prior_correction(self):
        from tangs.calibration import anchor_gated_logits

        logits = torch.tensor([[0.2, -0.1]])
        counts = torch.tensor([5.0, 1.0])
        actual = anchor_gated_logits(
            logits,
            torch.tensor([[1.0, 0.0]]),
            counts,
            [1],
            torch.zeros(1, 3),
            torch.tensor([False]),
        )
        expected = logits - 0.65 * (counts / counts.sum()).log().unsqueeze(0)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
