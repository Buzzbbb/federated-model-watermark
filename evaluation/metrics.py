"""
Evaluation metrics for the federated watermarking framework.

Metrics computed
────────────────
- **Accuracy (ACC)**          : fraction of clean test samples classified correctly
- **Watermark Success Rate (WSR)** : fraction of triggered samples classified
                                     as *target_label* (measures watermark
                                     retention after aggregation)
- **False Trigger Rate (FTR)** : fraction of clean test samples incorrectly
                                  classified as *target_label* (measures
                                  over-sensitivity / side-effects)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_accuracy(model: nn.Module, dataloader: DataLoader,
                     device: str = "cpu") -> float:
    """
    Compute top-1 classification accuracy on *dataloader*.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    device : str

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(dev), labels.to(dev)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_wsr(model: nn.Module, trigger_dataloader: DataLoader,
                target_label: int, device: str = "cpu") -> float:
    """
    Watermark Success Rate (WSR).

    Measures the fraction of triggered samples classified as *target_label*.

    Parameters
    ----------
    model : nn.Module
    trigger_dataloader : DataLoader
        DataLoader whose **all** samples carry the trigger pattern.
    target_label : int
        Expected output label for triggered samples.
    device : str

    Returns
    -------
    float
        WSR in [0, 1].
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    target = torch.tensor(target_label, device=dev)
    hits = 0
    total = 0
    for imgs, _ in trigger_dataloader:
        imgs = imgs.to(dev)
        preds = model(imgs).argmax(dim=1)
        hits += (preds == target).sum().item()
        total += imgs.size(0)
    return hits / total if total > 0 else 0.0


@torch.no_grad()
def compute_ftr(model: nn.Module, clean_dataloader: DataLoader,
                target_label: int, device: str = "cpu") -> float:
    """
    False Trigger Rate (FTR).

    Measures the fraction of **clean** test samples that are mis-classified as
    *target_label*.  A high FTR indicates the watermark has degraded the
    model's discrimination ability.

    Parameters
    ----------
    model : nn.Module
    clean_dataloader : DataLoader
        DataLoader with clean (un-triggered) test samples.
    target_label : int
        The watermark's target label.
    device : str

    Returns
    -------
    float
        FTR in [0, 1].
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)
    target = torch.tensor(target_label, device=dev)
    false_triggers = 0
    total = 0
    for imgs, labels in clean_dataloader:
        imgs, labels = imgs.to(dev), labels.to(dev)
        # Only consider samples that are NOT the target class
        mask = labels != target_label
        if mask.sum() == 0:
            continue
        preds = model(imgs[mask]).argmax(dim=1)
        false_triggers += (preds == target_label).sum().item()
        total += mask.sum().item()
    return false_triggers / total if total > 0 else 0.0


def evaluate_all(model: nn.Module,
                 clean_dataloader: DataLoader,
                 trigger_dataloader: DataLoader,
                 target_label: int,
                 device: str = "cpu") -> dict:
    """
    Convenience function that computes ACC, WSR, and FTR in one call.

    Returns
    -------
    dict with keys "acc", "wsr", "ftr"
    """
    acc = compute_accuracy(model, clean_dataloader, device)
    wsr = compute_wsr(model, trigger_dataloader, target_label, device)
    ftr = compute_ftr(model, clean_dataloader, target_label, device)
    return {"acc": acc, "wsr": wsr, "ftr": ftr}
