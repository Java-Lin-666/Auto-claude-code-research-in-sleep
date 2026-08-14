"""CDMAD-compatible CIFAR-LT and STL10-LT data construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .augment import CutoutDefault, RandAugment
from .config import RunConfig
from .splits import class_partition, count_overlap, make_imb_data, stable_hash


CIFAR10_STATS = (
    (0.4914, 0.4822, 0.4465),
    (0.2471, 0.2435, 0.2616),
)
CIFAR100_STATS = (
    (0.5071, 0.4867, 0.4408),
    (0.2675, 0.2565, 0.2761),
)


class TransformThrice:
    def __init__(self, weak: Any, strong: Any):
        self.weak = weak
        self.strong = strong

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, ...]:
        return self.weak(image), self.strong(image), self.strong(image)


class ArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        transform: Any,
        source_indices: np.ndarray,
        channel_first: bool = False,
    ):
        self.images = images
        self.targets = targets.astype(np.int64, copy=False)
        self.transform = transform
        self.source_indices = source_indices.astype(np.int64, copy=False)
        self.channel_first = channel_first

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        image = self.images[index]
        if self.channel_first:
            image = np.transpose(image, (1, 2, 0))
        pil_image = Image.fromarray(image)
        return (
            self.transform(pil_image),
            int(self.targets[index]),
            int(self.source_indices[index]),
        )


@dataclass
class DataBundle:
    labeled_loader: DataLoader
    unlabeled_loader: DataLoader
    evaluation_loader: DataLoader
    pseudo_evaluation_loader: DataLoader | None
    partition: dict[str, list[int]]
    split_manifest: dict[str, Any]


def _transforms(dataset_name: str) -> tuple[Any, Any, Any]:
    is_stl = dataset_name == "stl10"
    mean, std = CIFAR100_STATS if dataset_name == "cifar100" else CIFAR10_STATS
    prefix = [transforms.Resize(32)] if is_stl else []
    weak = transforms.Compose(
        [
            *prefix,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    strong = transforms.Compose(
        [
            RandAugment(3, 4),
            *prefix,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            CutoutDefault(16),
        ]
    )
    evaluation = transforms.Compose(
        [
            *prefix,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return weak, strong, evaluation


def _cifar_indices(
    targets: np.ndarray,
    labeled_counts: list[int],
    unlabeled_counts: list[int],
    dataset_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    labeled: list[int] = []
    unlabeled: list[int] = []
    for class_id in range(len(labeled_counts)):
        indices = np.where(targets == class_id)[0]
        if dataset_name == "cifar100" and seed != 0:
            np.random.shuffle(indices)
        labeled.extend(indices[: labeled_counts[class_id]].tolist())
        # This labeled-prefix overlap is intentional and matches CDMAD.
        count = labeled_counts[class_id] + unlabeled_counts[class_id]
        unlabeled.extend(indices[:count].tolist())
    return np.asarray(labeled), np.asarray(unlabeled)


def _development_split(
    labeled_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    targets: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    validation: list[int] = []
    for class_id in sorted(np.unique(targets[labeled_indices]).tolist()):
        candidates = labeled_indices[targets[labeled_indices] == class_id].copy()
        rng.shuffle(candidates)
        count = max(1, int(round(0.20 * len(candidates))))
        validation.extend(candidates[:count].tolist())
    validation_array = np.asarray(sorted(validation))
    held_out = set(validation_array.tolist())
    labeled_train = np.asarray([i for i in labeled_indices if i not in held_out])
    unlabeled_train = np.asarray([i for i in unlabeled_indices if i not in held_out])
    return labeled_train, unlabeled_train, validation_array


def _balanced_unused_cifar100_validation(
    targets: np.ndarray,
    source_unlabeled_indices: np.ndarray,
    samples_per_class: int = 50,
) -> np.ndarray:
    """Use only training images outside CDMAD's active unlabeled prefix.

    CIFAR-100 contains 500 training images per class.  Under P-C100-100 the
    largest active prefix is 150 labeled + 300 unlabeled images, so every class
    has at least 50 disjoint images available.  Selecting 50 from each class
    gives a balanced tuning set without looking at the official test set or
    shrinking training data.
    """
    active = set(int(index) for index in source_unlabeled_indices.tolist())
    validation: list[int] = []
    for class_id in sorted(np.unique(targets).tolist()):
        candidates = [
            int(index)
            for index in np.where(targets == class_id)[0]
            if int(index) not in active
        ]
        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"Class {class_id} has only {len(candidates)} unused images; "
                f"need {samples_per_class}."
            )
        validation.extend(candidates[:samples_per_class])
    result = np.asarray(validation, dtype=np.int64)
    if count_overlap(result.tolist(), source_unlabeled_indices.tolist()):
        raise AssertionError("Balanced development split overlaps training data.")
    return result


def _make_cifar(config: RunConfig) -> tuple[Dataset, Dataset, Dataset, Dataset, dict]:
    protocol = config.protocol
    dataset_class = datasets.CIFAR10 if protocol.dataset == "cifar10" else datasets.CIFAR100
    base_train = dataset_class(config.data_root, train=True, download=config.download)
    base_test = dataset_class(config.data_root, train=False, download=config.download)
    train_targets = np.asarray(base_train.targets)
    test_targets = np.asarray(base_test.targets)
    labeled_counts = make_imb_data(
        protocol.num_max, protocol.num_classes, protocol.gamma_l
    )
    assert protocol.num_max_u is not None and protocol.gamma_u is not None
    unlabeled_counts = make_imb_data(
        protocol.num_max_u, protocol.num_classes, protocol.gamma_u
    )
    labeled_indices, unlabeled_indices = _cifar_indices(
        train_targets,
        labeled_counts,
        unlabeled_counts,
        protocol.dataset,
        config.manual_seed,
    )
    source_labeled_indices = labeled_indices.copy()
    source_unlabeled_indices = unlabeled_indices.copy()
    validation_indices = np.asarray([], dtype=np.int64)
    if config.mode == "development":
        if protocol.dataset == "cifar100":
            validation_indices = _balanced_unused_cifar100_validation(
                train_targets, source_unlabeled_indices
            )
        else:
            labeled_indices, unlabeled_indices, validation_indices = _development_split(
                labeled_indices, unlabeled_indices, train_targets, config.manual_seed
            )

    weak, strong, evaluation = _transforms(protocol.dataset)
    labeled = ArrayDataset(
        base_train.data[labeled_indices],
        train_targets[labeled_indices],
        weak,
        labeled_indices,
    )
    unlabeled = ArrayDataset(
        base_train.data[unlabeled_indices],
        train_targets[unlabeled_indices],
        TransformThrice(weak, strong),
        unlabeled_indices,
    )
    if config.mode == "development":
        eval_images = base_train.data[validation_indices]
        eval_targets = train_targets[validation_indices]
        eval_indices = validation_indices
        evaluation_role = (
            "development-balanced-unused-train"
            if protocol.dataset == "cifar100"
            else "development-validation"
        )
    else:
        eval_images = base_test.data
        eval_targets = test_targets
        eval_indices = np.arange(len(test_targets))
        evaluation_role = "confirmatory-test"
    evaluation_set = ArrayDataset(
        eval_images, eval_targets, evaluation, eval_indices
    )
    pseudo_evaluation = ArrayDataset(
        base_train.data[unlabeled_indices],
        train_targets[unlabeled_indices],
        evaluation,
        unlabeled_indices,
    )

    partition = class_partition(labeled_counts)
    manifest = {
        "dataset": protocol.dataset,
        "manual_seed": config.manual_seed,
        "labeled_counts": labeled_counts,
        "unlabeled_argument_counts": unlabeled_counts,
        "source_labeled_indices": source_labeled_indices.tolist(),
        "source_unlabeled_loader_indices": source_unlabeled_indices.tolist(),
        "active_labeled_indices": labeled_indices.tolist(),
        "active_unlabeled_loader_indices": unlabeled_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "source_overlap_count": count_overlap(
            source_labeled_indices.tolist(), source_unlabeled_indices.tolist()
        ),
        "active_overlap_count": count_overlap(
            labeled_indices.tolist(), unlabeled_indices.tolist()
        ),
        "evaluation_role": evaluation_role,
        "validation_is_disjoint_from_active_unlabeled": count_overlap(
            validation_indices.tolist(), unlabeled_indices.tolist()
        )
        == 0,
        "partition": partition,
    }
    return labeled, unlabeled, evaluation_set, pseudo_evaluation, manifest


def _make_stl(config: RunConfig) -> tuple[Dataset, Dataset, Dataset, Dataset, dict]:
    protocol = config.protocol
    train = datasets.STL10(
        config.data_root, split="train", download=config.download
    )
    official_unlabeled = datasets.STL10(
        config.data_root, split="unlabeled", download=config.download
    )
    test = datasets.STL10(config.data_root, split="test", download=config.download)
    train_targets = np.asarray(train.labels)
    labeled_counts = make_imb_data(
        protocol.num_max, protocol.num_classes, protocol.gamma_l
    )
    labeled_indices: list[int] = []
    for class_id, count in enumerate(labeled_counts):
        indices = np.where(train_targets == class_id)[0]
        # No shuffle, even for nonzero seeds, matching the released STL loader.
        labeled_indices.extend(indices[:count].tolist())
    labeled_indices_array = np.asarray(labeled_indices)

    weak, strong, evaluation = _transforms(protocol.dataset)
    labeled = ArrayDataset(
        train.data[labeled_indices_array],
        train_targets[labeled_indices_array],
        weak,
        labeled_indices_array,
        channel_first=True,
    )
    combined_images = np.concatenate(
        [official_unlabeled.data, train.data[labeled_indices_array]], axis=0
    )
    combined_targets = np.concatenate(
        [
            np.full(len(official_unlabeled.data), -1, dtype=np.int64),
            train_targets[labeled_indices_array],
        ]
    )
    combined_indices = np.concatenate(
        [
            np.arange(len(official_unlabeled.data)),
            len(official_unlabeled.data) + labeled_indices_array,
        ]
    )
    unlabeled = ArrayDataset(
        combined_images,
        combined_targets,
        TransformThrice(weak, strong),
        combined_indices,
        channel_first=True,
    )
    evaluation_set = ArrayDataset(
        test.data,
        np.asarray(test.labels),
        evaluation,
        np.arange(len(test.labels)),
        channel_first=True,
    )
    # Ground truth exists only for the appended labeled subset.
    known_start = len(official_unlabeled.data)
    pseudo_evaluation = ArrayDataset(
        combined_images[known_start:],
        combined_targets[known_start:],
        evaluation,
        combined_indices[known_start:],
        channel_first=True,
    )
    partition = class_partition(labeled_counts)
    append_order = labeled_indices_array.tolist()
    manifest = {
        "dataset": protocol.dataset,
        "manual_seed": config.manual_seed,
        "labeled_counts": labeled_counts,
        "labeled_indices": append_order,
        "official_unlabeled_cardinality": len(official_unlabeled.data),
        "appended_labeled_indices": append_order,
        "actual_unlabeled_loader_cardinality": len(combined_targets),
        "known_label_pseudo_evaluation_cardinality": len(append_order),
        "evaluation_role": "confirmatory-test",
        "partition": partition,
    }
    return labeled, unlabeled, evaluation_set, pseudo_evaluation, manifest


def _loader_runtime_kwargs(workers: int, pin_memory: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        # CyclingLoader recreates iterators often. Persistent workers avoid an
        # expensive Windows process spawn on every labeled/unlabeled cycle.
        kwargs["persistent_workers"] = True
    return kwargs


def build_data(config: RunConfig) -> DataBundle:
    if config.protocol.dataset in {"cifar10", "cifar100"}:
        labeled, unlabeled, evaluation, pseudo_evaluation, manifest = _make_cifar(
            config
        )
    else:
        labeled, unlabeled, evaluation, pseudo_evaluation, manifest = _make_stl(
            config
        )

    loader_kwargs = _loader_runtime_kwargs(
        config.workers, pin_memory=torch.cuda.is_available()
    )
    labeled_loader = DataLoader(
        labeled,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    unlabeled_loader = DataLoader(
        unlabeled,
        batch_size=config.batch_size * config.unlabeled_ratio,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    evaluation_loader = DataLoader(
        evaluation,
        batch_size=config.test_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    pseudo_loader = None
    if config.eval_pseudo_labels:
        pseudo_loader = DataLoader(
            pseudo_evaluation,
            batch_size=config.test_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )

    manifest["partition_hash"] = stable_hash(manifest["partition"])
    manifest["split_hash"] = stable_hash(
        {
            key: value
            for key, value in manifest.items()
            if key.endswith("indices") or key.endswith("counts")
        }
    )
    manifest["loader_batches"] = {
        "labeled": len(labeled_loader),
        "unlabeled": len(unlabeled_loader),
        "evaluation": len(evaluation_loader),
    }
    manifest["batch_shapes"] = {
        "labeled": config.batch_size,
        "unlabeled": config.batch_size * config.unlabeled_ratio,
        "views": 3,
    }
    manifest["loader_runtime"] = {
        "num_workers": config.workers,
        "persistent_workers": config.workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataBundle(
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        evaluation_loader=evaluation_loader,
        pseudo_evaluation_loader=pseudo_loader,
        partition=manifest["partition"],
        split_manifest=manifest,
    )


class CyclingLoader:
    """Restart a finite loader on exhaustion, matching the CDMAD training loop."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)
        self.restarts = 0

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.restarts += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)
