from __future__ import annotations

import copy
import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class NoOpEquivalenceTests(unittest.TestCase):
    def test_exact_head_contribution_matches_closed_form_linear_gradient(self):
        from tangs.surgery import classifier_gradient

        torch.manual_seed(3)
        classifier = torch.nn.Linear(2, 3)
        first = torch.tensor([[1.0, -2.0], [0.5, 1.5]])
        second = torch.tensor([[-1.0, 0.25], [2.0, 0.5]])
        targets = torch.tensor([0, 2])
        accepted_head = torch.tensor([True, False])
        features = torch.cat([first, second])
        repeated_targets = torch.cat([targets, targets])
        repeated_mask = torch.cat([accepted_head, accepted_head]).float()
        logits = classifier(features)
        per_example = torch.nn.functional.cross_entropy(
            logits, repeated_targets, reduction="none"
        )
        head_loss = (per_example * repeated_mask).mean()
        observed = classifier_gradient(head_loss, list(classifier.parameters()))

        probabilities = torch.softmax(logits.detach(), dim=1)
        one_hot = torch.nn.functional.one_hot(
            repeated_targets, num_classes=3
        ).float()
        delta = (probabilities - one_hot) * repeated_mask[:, None] / len(features)
        expected_weight = delta.T @ features
        expected_bias = delta.sum(dim=0)
        expected = torch.cat([expected_weight.reshape(-1), expected_bias])
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)

    def test_noop_preserves_backbone_gradient_and_full_adam_step(self):
        from tangs.surgery import (
            TangsController,
            classifier_gradient,
            read_classifier_gradient,
            write_classifier_gradient,
        )

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(4, 3)
                self.output = torch.nn.Linear(3, 2)

            def forward(self, value):
                return self.output(torch.tanh(self.backbone(value)))

        torch.manual_seed(23)
        baseline = Tiny()
        candidate = copy.deepcopy(baseline)
        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.0015)
        candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0015)
        inputs = torch.randn(6, 4)
        targets = torch.tensor([0, 1, 0, 1, 1, 0])

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss = torch.nn.functional.cross_entropy(baseline(inputs), targets)
        baseline_loss.backward()
        expected_backbone = [p.grad.clone() for p in baseline.backbone.parameters()]
        baseline_optimizer.step()

        candidate_optimizer.zero_grad(set_to_none=True)
        logits = candidate(inputs)
        per_example = torch.nn.functional.cross_entropy(
            logits, targets, reduction="none"
        )
        loss = per_example.mean()
        classifier_parameters = list(candidate.output.parameters())
        head_gradient = classifier_gradient(
            (per_example * torch.tensor([1, 0, 1, 0, 0, 0])).mean(),
            classifier_parameters,
            retain_graph=True,
        )
        loss.backward()
        for expected, actual in zip(
            expected_backbone, (p.grad for p in candidate.backbone.parameters())
        ):
            torch.testing.assert_close(expected, actual, rtol=1e-7, atol=1e-7)
        base_gradient = read_classifier_gradient(classifier_parameters)
        controller = TangsController("no-projection", tau=math.inf, warmup_steps=0)
        result = controller.transform(
            base_gradient, head_gradient, torch.ones_like(base_gradient), step=1
        )
        write_classifier_gradient(classifier_parameters, result.gradient)
        candidate_optimizer.step()

        for expected, actual in zip(baseline.parameters(), candidate.parameters()):
            torch.testing.assert_close(expected, actual, rtol=1e-7, atol=1e-7)
        expected_state = baseline_optimizer.state_dict()
        actual_state = candidate_optimizer.state_dict()
        self.assertEqual(expected_state["param_groups"], actual_state["param_groups"])
        for parameter_id in expected_state["state"]:
            for key, expected in expected_state["state"][parameter_id].items():
                actual = actual_state["state"][parameter_id][key]
                if torch.is_tensor(expected):
                    torch.testing.assert_close(expected, actual)
                else:
                    self.assertEqual(expected, actual)

    def test_exact_decomposition_and_adam_state_match(self):
        from tangs.surgery import (
            TangsController,
            read_classifier_gradient,
            write_classifier_gradient,
        )

        torch.manual_seed(19)
        baseline = torch.nn.Linear(3, 2)
        candidate = copy.deepcopy(baseline)
        baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=0.0015)
        candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0015)
        inputs = torch.randn(5, 3)
        targets = torch.tensor([0, 1, 0, 1, 1])

        baseline_optimizer.zero_grad(set_to_none=True)
        baseline_loss = torch.nn.functional.cross_entropy(
            baseline(inputs), targets
        )
        baseline_loss.backward()
        baseline_optimizer.step()

        candidate_optimizer.zero_grad(set_to_none=True)
        candidate_loss = torch.nn.functional.cross_entropy(
            candidate(inputs), targets
        )
        candidate_loss.backward()
        parameters = list(candidate.parameters())
        base_gradient = read_classifier_gradient(parameters)
        head_gradient = torch.linspace(
            -0.3, 0.4, base_gradient.numel(), dtype=base_gradient.dtype
        )
        tail_gradient = torch.ones_like(base_gradient)
        remainder = base_gradient - head_gradient
        torch.testing.assert_close(
            base_gradient,
            head_gradient + remainder,
            rtol=1e-7,
            atol=1e-7,
        )
        controller = TangsController(
            "no-projection", tau=math.inf, warmup_steps=0
        )
        result = controller.transform(
            base_gradient, head_gradient, tail_gradient, step=1
        )
        torch.testing.assert_close(
            result.gradient, base_gradient, rtol=1e-7, atol=1e-7
        )
        write_classifier_gradient(parameters, result.gradient)
        candidate_optimizer.step()

        for expected, actual in zip(
            baseline.parameters(), candidate.parameters()
        ):
            torch.testing.assert_close(expected, actual, rtol=1e-7, atol=1e-7)
        expected_state = baseline_optimizer.state_dict()
        actual_state = candidate_optimizer.state_dict()
        self.assertEqual(
            expected_state["param_groups"], actual_state["param_groups"]
        )
        for parameter_id in expected_state["state"]:
            for key, expected in expected_state["state"][parameter_id].items():
                actual = actual_state["state"][parameter_id][key]
                if torch.is_tensor(expected):
                    torch.testing.assert_close(expected, actual)
                else:
                    self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
