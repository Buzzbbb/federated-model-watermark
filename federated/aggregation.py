"""
Aggregation strategies for the federated server.

Implemented strategies
──────────────────────
- fedavg        : Weighted average (McMahan et al., 2017)
- median        : Coordinate-wise median (Byzantine-robust)
- trimmed_mean  : Coordinate-wise trimmed mean (Byzantine-robust)
"""

import copy
from typing import List, Dict

import torch
import torch.nn as nn


def fedavg(global_model: nn.Module,
           client_updates: List[Dict[str, torch.Tensor]],
           client_sizes: List[int]) -> nn.Module:
    """
    Federated Averaging.

    Each client's state-dict is weighted by its dataset size.

    Parameters
    ----------
    global_model : nn.Module
        The current global model (used to initialise the aggregated state).
    client_updates : list of state-dicts
        Parameter updates from participating clients.
    client_sizes : list of int
        Number of training samples on each participating client.

    Returns
    -------
    nn.Module
        The global model with updated parameters (in-place).
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    global_sd = global_model.state_dict()
    agg_sd = copy.deepcopy(global_sd)

    for key in agg_sd:
        if not agg_sd[key].is_floating_point():
            # Integer buffers (e.g. num_batches_tracked) – keep global value
            continue
        agg_sd[key] = sum(
            w * sd[key].float()
            for w, sd in zip(weights, client_updates)
        )

    global_model.load_state_dict(agg_sd)
    return global_model


def coordinate_median(global_model: nn.Module,
                      client_updates: List[Dict[str, torch.Tensor]],
                      client_sizes: List[int]) -> nn.Module:
    """
    Coordinate-wise median aggregation.

    Ignores client_sizes; all clients are treated equally.  More robust than
    FedAvg against Byzantine / poisoning clients.
    """
    global_sd = global_model.state_dict()
    agg_sd = copy.deepcopy(global_sd)

    for key in agg_sd:
        if not agg_sd[key].is_floating_point():
            continue
        stacked = torch.stack([sd[key].float() for sd in client_updates], dim=0)
        agg_sd[key] = stacked.median(dim=0).values

    global_model.load_state_dict(agg_sd)
    return global_model


def trimmed_mean(global_model: nn.Module,
                 client_updates: List[Dict[str, torch.Tensor]],
                 client_sizes: List[int],
                 trim_ratio: float = 0.1) -> nn.Module:
    """
    Coordinate-wise trimmed mean.

    Drops the top and bottom *trim_ratio* fraction of values per coordinate
    before averaging.

    Parameters
    ----------
    trim_ratio : float
        Fraction of extreme values to discard on each end (0 ≤ trim_ratio < 0.5).
    """
    n = len(client_updates)
    k = max(1, int(n * trim_ratio))  # number of values to trim on each side
    if 2 * k >= n:
        # Fallback to median when there are too few clients to trim
        return coordinate_median(global_model, client_updates, client_sizes)

    global_sd = global_model.state_dict()
    agg_sd = copy.deepcopy(global_sd)

    for key in agg_sd:
        if not agg_sd[key].is_floating_point():
            continue
        stacked = torch.stack([sd[key].float() for sd in client_updates], dim=0)
        sorted_vals, _ = stacked.sort(dim=0)
        trimmed = sorted_vals[k: n - k]
        agg_sd[key] = trimmed.mean(dim=0)

    global_model.load_state_dict(agg_sd)
    return global_model


# ── Registry ──────────────────────────────────────────────────────────────────

_AGGREGATION_FNS = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
}


def get_aggregation_fn(name: str):
    """Return the aggregation function for the given strategy name."""
    key = name.lower()
    if key not in _AGGREGATION_FNS:
        raise ValueError(
            f"Unknown aggregation strategy '{name}'. "
            f"Available: {list(_AGGREGATION_FNS)}"
        )
    return _AGGREGATION_FNS[key]
