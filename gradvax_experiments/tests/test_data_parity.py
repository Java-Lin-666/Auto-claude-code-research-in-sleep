from __future__ import annotations

import unittest

try:
    import numpy as np
    import torch
    import torchvision
except ImportError:
    np = None
    torch = None
    torchvision = None


@unittest.skipIf(
    torch is None or torchvision is None or np is None,
    "PyTorch/torchvision/NumPy are not installed in this environment",
)
class DataParityTests(unittest.TestCase):
    def test_cycling_loader_restarts_without_reconstructing_loader(self):
        from torch.utils.data import DataLoader, TensorDataset

        from tangs.data import CyclingLoader

        loader = DataLoader(
            TensorDataset(torch.tensor([1, 2, 3, 4])),
            batch_size=2,
            shuffle=False,
            drop_last=True,
        )
        cycling = CyclingLoader(loader)
        first = cycling.next()[0]
        second = cycling.next()[0]
        third = cycling.next()[0]
        torch.testing.assert_close(first, torch.tensor([1, 2]))
        torch.testing.assert_close(second, torch.tensor([3, 4]))
        torch.testing.assert_close(third, torch.tensor([1, 2]))
        self.assertEqual(cycling.restarts, 1)
        self.assertIs(cycling.loader, loader)

    def test_cifar_seed0_prefix_overlap(self):
        from tangs.data import _cifar_indices

        targets = np.repeat(np.arange(3), 10)
        labeled, unlabeled = _cifar_indices(
            targets,
            labeled_counts=[2, 2, 2],
            unlabeled_counts=[3, 3, 3],
            dataset_name="cifar100",
            seed=0,
        )
        np.testing.assert_array_equal(labeled, [0, 1, 10, 11, 20, 21])
        np.testing.assert_array_equal(
            unlabeled, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        )
        self.assertEqual(len(set(labeled).intersection(unlabeled)), len(labeled))

    def test_cifar100_nonzero_seed_changes_split_only_as_upstream(self):
        from tangs.data import _cifar_indices

        targets = np.repeat(np.arange(3), 10)
        seed0, _ = _cifar_indices(
            targets, [2, 2, 2], [3, 3, 3], "cifar100", 0
        )
        seed1, _ = _cifar_indices(
            targets, [2, 2, 2], [3, 3, 3], "cifar100", 1
        )
        c10_seed1, _ = _cifar_indices(
            targets, [2, 2, 2], [3, 3, 3], "cifar10", 1
        )
        self.assertFalse(np.array_equal(seed0, seed1))
        np.testing.assert_array_equal(seed0, c10_seed1)

    def test_cifar100_balanced_development_split_is_disjoint_and_balanced(self):
        from tangs.data import _balanced_unused_cifar100_validation

        targets = np.repeat(np.arange(3), 500)
        active = np.concatenate(
            [np.arange(class_id * 500, class_id * 500 + 450) for class_id in range(3)]
        )
        validation = _balanced_unused_cifar100_validation(targets, active)
        self.assertEqual(len(validation), 150)
        self.assertFalse(set(validation.tolist()).intersection(active.tolist()))
        counts = np.bincount(targets[validation], minlength=3)
        np.testing.assert_array_equal(counts, [50, 50, 50])

    def test_transform_order_matches_released_loader(self):
        from tangs.augment import CutoutDefault, RandAugment
        from tangs.data import _transforms

        _, cifar_strong, _ = _transforms("cifar100")
        self.assertIsInstance(cifar_strong.transforms[0], RandAugment)
        self.assertEqual(
            [type(transform).__name__ for transform in cifar_strong.transforms[1:-1]],
            ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"],
        )
        self.assertIsInstance(cifar_strong.transforms[-1], CutoutDefault)

        stl_weak, stl_strong, stl_test = _transforms("stl10")
        self.assertEqual(type(stl_weak.transforms[0]).__name__, "Resize")
        self.assertEqual(type(stl_strong.transforms[0]).__name__, "RandAugment")
        self.assertEqual(type(stl_strong.transforms[1]).__name__, "Resize")
        self.assertEqual(type(stl_test.transforms[0]).__name__, "Resize")

    def test_loader_runtime_uses_persistent_workers(self):
        from tangs.data import _loader_runtime_kwargs

        multi_worker = _loader_runtime_kwargs(4, pin_memory=True)
        self.assertEqual(multi_worker["num_workers"], 4)
        self.assertTrue(multi_worker["persistent_workers"])
        self.assertTrue(multi_worker["pin_memory"])

        single_process = _loader_runtime_kwargs(0, pin_memory=False)
        self.assertEqual(single_process["num_workers"], 0)
        self.assertNotIn("persistent_workers", single_process)
        self.assertFalse(single_process["pin_memory"])


if __name__ == "__main__":
    unittest.main()
