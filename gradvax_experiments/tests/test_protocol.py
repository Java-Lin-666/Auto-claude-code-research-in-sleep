from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from tangs.config import CONFIG_VERSION, PROTOCOLS, build_run_config
from tangs.splits import class_partition, count_overlap, make_imb_data, stable_hash


def arguments(**overrides):
    values = {
        "protocol": "P-C100-100",
        "method": "tangs",
        "mode": "confirmatory",
        "manual_seed": 0,
        "data_root": "./data/cache",
        "output_root": "./results",
        "run_id": "test-run",
        "device": "cuda",
        "workers": 4,
        "download": False,
        "resume": None,
        "tangs_tau": 5.0,
        "head_weight": 0.5,
        "diagnostic_level": "basic",
        "diagnostic_sample_limit": 100,
        "log_every": 50,
        "checkpoint_every": 500,
        "snapshot_every": 50_000,
        "eval_pseudo_labels": True,
        "max_steps": None,
        "allow_protocol_deviation": False,
        "amp": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ProtocolTests(unittest.TestCase):
    def test_protocol_version_matches_v45_documents(self):
        self.assertEqual(CONFIG_VERSION, "4.5")

    def test_registered_counts_match_cdmad_formula(self):
        c10 = PROTOCOLS["P-C10-100"]
        c100 = PROTOCOLS["P-C100-100"]
        self.assertEqual(make_imb_data(c10.num_max, 10, c10.gamma_l)[0], 1500)
        self.assertEqual(make_imb_data(c10.num_max, 10, c10.gamma_l)[-1], 15)
        self.assertEqual(make_imb_data(c100.num_max, 100, c100.gamma_l)[0], 150)
        self.assertEqual(make_imb_data(c100.num_max, 100, c100.gamma_l)[-1], 1)

    def test_partition_sizes_are_locked_terciles(self):
        ten = class_partition(make_imb_data(1500, 10, 100))
        hundred = class_partition(make_imb_data(150, 100, 100))
        self.assertEqual([len(ten[key]) for key in ("head", "medium", "tail")], [3, 4, 3])
        self.assertEqual(
            [len(hundred[key]) for key in ("head", "medium", "tail")],
            [33, 34, 33],
        )
        self.assertEqual(ten["head"], [0, 1, 2])
        self.assertEqual(ten["tail"], [7, 8, 9])

    def test_hash_and_overlap_are_deterministic(self):
        value = {"indices": [3, 1, 2], "role": "test"}
        self.assertEqual(stable_hash(value), stable_hash(value))
        self.assertEqual(count_overlap([1, 2, 3], [2, 3, 4]), 2)

    def test_confirmatory_defaults_are_locked(self):
        config = build_run_config(arguments())
        self.assertEqual(config.total_steps, 250_000)
        self.assertEqual(config.manual_seed, 0)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.unlabeled_ratio, 2)
        self.assertEqual(config.learning_rate, 0.0015)
        self.assertEqual(config.confidence_threshold, 0.95)

    def test_seed_or_budget_drift_requires_explicit_deviation(self):
        with self.assertRaisesRegex(ValueError, "protocol deviation"):
            build_run_config(arguments(manual_seed=1))
        with self.assertRaisesRegex(ValueError, "protocol deviation"):
            build_run_config(arguments(max_steps=10))

    def test_amp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "AMP is forbidden"):
            build_run_config(arguments(amp=True))

    def test_no_cap_value_accepts_infinity(self):
        config = build_run_config(arguments(tangs_tau=math.inf))
        self.assertTrue(math.isinf(config.tangs_tau))

    def test_oracle_control_is_cifar100_only(self):
        with self.assertRaisesRegex(ValueError, "registered only"):
            build_run_config(
                arguments(protocol="P-C10-100", method="oracle-fixmatch")
            )


if __name__ == "__main__":
    unittest.main()
