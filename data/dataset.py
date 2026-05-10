"""
Dataset loading and non-IID data partitioning for federated learning.

Supports MNIST and CIFAR-10.  Data is partitioned among clients either IID
(random shuffle) or non-IID via Dirichlet allocation.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms


# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────

_MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

_CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


def get_transforms(dataset_name: str):
    """Return (train_transform, test_transform) for the given dataset."""
    name = dataset_name.upper()
    if name == "MNIST":
        return _MNIST_TRANSFORM, _MNIST_TRANSFORM
    if name == "CIFAR10":
        return _CIFAR10_TRANSFORM, _CIFAR10_TRANSFORM
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str, data_dir: str = "./data/raw"):
    """Download and return (train_dataset, test_dataset)."""
    os.makedirs(data_dir, exist_ok=True)
    train_tf, test_tf = get_transforms(dataset_name)
    name = dataset_name.upper()
    if name == "MNIST":
        train_ds = torchvision.datasets.MNIST(data_dir, train=True,
                                              download=True, transform=train_tf)
        test_ds = torchvision.datasets.MNIST(data_dir, train=False,
                                             download=True, transform=test_tf)
    elif name == "CIFAR10":
        train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,
                                                download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(data_dir, train=False,
                                               download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_ds, test_ds


# ──────────────────────────────────────────────────────────────────────────────
# Partitioning
# ──────────────────────────────────────────────────────────────────────────────

def _get_labels(dataset) -> np.ndarray:
    """Extract target labels from a dataset.

    Supports torchvision datasets (``targets`` / ``labels`` attributes),
    :class:`torch.utils.data.TensorDataset` (second tensor), and any dataset
    that can be iterated to collect labels as a fallback.
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    elif hasattr(dataset, "tensors"):
        # TensorDataset: labels are the second tensor by convention
        labels = dataset.tensors[1]
    else:
        # Generic fallback: iterate over the dataset to collect labels
        labels = [dataset[i][1] for i in range(len(dataset))]
    if isinstance(labels, torch.Tensor):
        return labels.numpy()
    return np.array(labels)


def partition_iid(dataset, num_clients: int, seed: int = 42):
    """Randomly split dataset indices into *num_clients* equal-sized shards."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, num_clients)
    return [shard.tolist() for shard in shards]


def partition_non_iid_dirichlet(dataset, num_clients: int,
                                alpha: float = 0.5, seed: int = 42):
    """
    Partition dataset indices using a Dirichlet distribution.

    Each client receives data drawn from a Dirichlet-sampled class distribution,
    producing heterogeneous (non-IID) local datasets whose skewness is
    controlled by *alpha* (smaller → more skewed).
    """
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    num_classes = int(labels.max()) + 1

    # Group indices by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to integer counts
        splits = (proportions * len(class_indices[c])).astype(int)
        # Adjust rounding error
        splits[-1] = len(class_indices[c]) - splits[:-1].sum()
        splits = np.maximum(splits, 0)
        offset = 0
        for k, n in enumerate(splits):
            client_indices[k].extend(
                class_indices[c][offset: offset + n].tolist()
            )
            offset += n

    return client_indices


def get_client_dataloaders(dataset, client_indices, batch_size: int = 64):
    """Return a list of DataLoaders, one per client."""
    loaders = []
    for idx in client_indices:
        subset = Subset(dataset, idx)
        loader = DataLoader(subset, batch_size=batch_size,
                            shuffle=True, drop_last=False)
        loaders.append(loader)
    return loaders


def get_test_dataloader(dataset, batch_size: int = 256):
    """Return a DataLoader for the test set."""
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=False, drop_last=False)
