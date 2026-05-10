"""
Base watermark strategy interface.

All concrete watermark classes inherit from :class:`WatermarkBase` and must
implement :meth:`inject_trigger` (apply a trigger to a single image tensor)
and :meth:`poison_dataset` (wrap a dataset so that a fraction of samples
carry the trigger and the target label).
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset


class PoisonedDataset(Dataset):
    """
    Wraps an existing dataset.  A random subset of ``poison_rate`` is replaced
    with (triggered_image, target_label) pairs; the rest keeps original labels.
    """

    def __init__(self, base_dataset, trigger_fn, target_label: int,
                 poison_rate: float, seed: int = 0):
        self.base = base_dataset
        self.trigger_fn = trigger_fn
        self.target_label = target_label

        rng = np.random.default_rng(seed)
        n = len(base_dataset)
        num_poison = max(1, int(n * poison_rate))
        self.poisoned_indices = set(
            rng.choice(n, size=num_poison, replace=False).tolist()
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if idx in self.poisoned_indices:
            img = self.trigger_fn(img)
            # Preserve the label's original type (tensor vs int)
            if isinstance(label, torch.Tensor):
                label = torch.tensor(self.target_label, dtype=label.dtype)
            else:
                label = self.target_label
        return img, label


class WatermarkBase:
    """Abstract base class for watermark strategies."""

    def __init__(self, target_label: int = 0, poison_rate: float = 0.1):
        self.target_label = target_label
        self.poison_rate = poison_rate

    # ------------------------------------------------------------------ #
    # Sub-classes must implement these two methods.                         #
    # ------------------------------------------------------------------ #

    def inject_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the trigger pattern to a single image tensor (C, H, W)."""
        raise NotImplementedError

    def poison_dataset(self, dataset, seed: int = 0) -> PoisonedDataset:
        """Return a poisoned version of *dataset*."""
        return PoisonedDataset(
            dataset,
            trigger_fn=self.inject_trigger,
            target_label=self.target_label,
            poison_rate=self.poison_rate,
            seed=seed,
        )

    def build_trigger_test_dataset(self, dataset, seed: int = 0) -> PoisonedDataset:
        """
        Return a version of *dataset* where ALL samples carry the trigger and
        target label – used to measure WSR (Watermark Success Rate).
        """
        return PoisonedDataset(
            dataset,
            trigger_fn=self.inject_trigger,
            target_label=self.target_label,
            poison_rate=1.0,
            seed=seed,
        )
