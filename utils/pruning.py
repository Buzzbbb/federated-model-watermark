"""
Model pruning utilities for robustness evaluation.

Pruning is used to assess how well a watermark survives weight-zeroing
compression.  The :func:`magnitude_prune` function zeros the smallest-magnitude
weights in all *Conv2d* and *Linear* layers at a given sparsity rate.
"""

import copy

import torch
import torch.nn as nn


def magnitude_prune(model: nn.Module, prune_rate: float) -> nn.Module:
    """
    Apply unstructured magnitude-based pruning to a **copy** of *model*.

    For every ``Conv2d`` and ``Linear`` layer the *prune_rate* fraction of
    weights with the smallest absolute value are set to zero.  The original
    model is not modified.

    Parameters
    ----------
    model : nn.Module
        The model to prune.
    prune_rate : float
        Fraction of weights to zero out in [0, 1).  0 means no pruning;
        values close to 1 remove almost all weights.

    Returns
    -------
    nn.Module
        A pruned copy of *model*.
    """
    if not (0.0 <= prune_rate < 1.0):
        raise ValueError(f"prune_rate must be in [0, 1), got {prune_rate}")

    pruned = copy.deepcopy(model)
    pruned.eval()

    with torch.no_grad():
        for module in pruned.modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            weight = module.weight.data
            if prune_rate == 0.0:
                continue
            # Determine threshold: the (prune_rate)-th quantile of |w|
            flat = weight.abs().view(-1)
            threshold = flat.kthvalue(max(1, int(prune_rate * flat.numel()))).values
            mask = weight.abs() > threshold
            module.weight.data = weight * mask.float()

    return pruned


def evaluate_pruning_robustness(model: nn.Module, prune_rates,
                                eval_fn, verbose: bool = True) -> list:
    """
    Evaluate *model* at each pruning level in *prune_rates*.

    Parameters
    ----------
    model : nn.Module
        The trained (potentially watermarked) global model.
    prune_rates : iterable of float
        Pruning fractions to evaluate (e.g. ``[0.0, 0.1, 0.3, 0.5, 0.7]``).
    eval_fn : callable
        ``eval_fn(pruned_model) -> dict`` – should return a metrics dict
        with at least ``"acc"`` and ``"wsr"`` keys.
    verbose : bool
        Print per-rate results.

    Returns
    -------
    list of dict
        Each dict contains ``"prune_rate"`` plus whatever *eval_fn* returns.
    """
    results = []
    for rate in prune_rates:
        pruned = magnitude_prune(model, rate)
        metrics = eval_fn(pruned)
        metrics["prune_rate"] = rate
        results.append(metrics)
        if verbose:
            parts = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(f"  prune_rate={rate:.2f}: {parts}")
    return results
