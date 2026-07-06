"""Long-tailed CIFAR dataset utilities."""
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def make_lt_dataset(root, dataset='cifar100', gamma=100, n_labeled=500,
                    split='train', seed=42):
    """Create long-tailed version of CIFAR-10/100.

    Args:
        gamma: imbalance ratio (max_class_count / min_class_count)
        n_labeled: total labeled samples
        split: 'labeled', 'unlabeled', or 'test'
    """
    rng = np.random.default_rng(seed)
    is_cifar10 = dataset == 'cifar10'
    num_classes = 10 if is_cifar10 else 100

    base = datasets.CIFAR10 if is_cifar10 else datasets.CIFAR100
    ds = base(root, train=(split != 'test'), download=True)
    targets = np.array(ds.targets)
    data = ds.data

    # Compute per-class sample counts under exponential imbalance
    n_max = len(targets) // num_classes
    class_counts = np.array([
        max(1, int(n_max * (1 / gamma) ** (c / (num_classes - 1))))
        for c in range(num_classes)
    ])

    # Sample indices per class
    lt_indices = []
    for c in range(num_classes):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        lt_indices.append(cls_idx[:class_counts[c]])

    # Compute class partition (fixed terciles by labeled-set frequency)
    labeled_counts = np.array([max(1, int(n_labeled / num_classes *
                               (1 / gamma) ** (c / (num_classes - 1))))
                               for c in range(num_classes)])
    partition = compute_class_partition(labeled_counts, num_classes)

    if split == 'test':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*_cifar_stats(is_cifar10)),
        ])
        return _IndexedDataset(data, targets, list(range(len(targets))),
                               transform), partition

    all_lt_idx = np.concatenate(lt_indices)

    # Split labeled / unlabeled
    labeled_idx, unlabeled_idx = [], []
    for c in range(num_classes):
        cls_lt = lt_indices[c]
        n_l = max(1, int(n_labeled / num_classes *
                         (1 / gamma) ** (c / (num_classes - 1))))
        labeled_idx.extend(cls_lt[:n_l].tolist())
        unlabeled_idx.extend(cls_lt[n_l:].tolist())

    weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*_cifar_stats(is_cifar10)),
    ])
    strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(*_cifar_stats(is_cifar10)),
    ])

    if split == 'labeled':
        return _IndexedDataset(data, targets, labeled_idx, weak), partition
    else:
        return _TwoViewDataset(data, targets, unlabeled_idx, weak, strong), partition


def compute_class_partition(labeled_counts, num_classes):
    """Fixed tercile partition by labeled-set frequency."""
    sorted_cls = np.argsort(labeled_counts)[::-1]  # descending frequency
    t = num_classes // 3
    return {
        'head': sorted_cls[:t].tolist(),
        'medium': sorted_cls[t:2*t].tolist(),
        'tail': sorted_cls[2*t:].tolist(),
    }


def _cifar_stats(is_cifar10):
    if is_cifar10:
        return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


class _IndexedDataset(Dataset):
    def __init__(self, data, targets, indices, transform):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        from PIL import Image
        idx = self.indices[i]
        img = Image.fromarray(self.data[idx])
        img = self.transform(img)
        return img, int(self.targets[idx])


class _TwoViewDataset(Dataset):
    def __init__(self, data, targets, indices, weak_tf, strong_tf):
        self.data = data
        self.targets = targets
        self.indices = indices
        self.weak = weak_tf
        self.strong = strong_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        from PIL import Image
        idx = self.indices[i]
        img = Image.fromarray(self.data[idx])
        return self.weak(img), self.strong(img), int(self.targets[idx])


class RandAugment:
    """Minimal RandAugment for CIFAR."""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        from torchvision.transforms import autoaugment
        self._ra = autoaugment.RandAugment(num_ops=n, magnitude=m)

    def __call__(self, img):
        from PIL import Image
        import numpy as np
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self._ra(img)
