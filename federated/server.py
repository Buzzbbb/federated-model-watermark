"""
Federated learning server.

Orchestrates global training rounds: selects a fraction of clients each round
(simulating client dropout), collects their locally trained state-dicts, and
aggregates them into the global model using the configured strategy.
"""

import copy
import random
from typing import List, Optional

import torch
import torch.nn as nn

from .client import Client
from .aggregation import get_aggregation_fn


class Server:
    """
    Central server for federated learning.

    Parameters
    ----------
    global_model : nn.Module
        The initial global model.
    clients : list of Client
        All registered federated clients.
    aggregation : str
        Aggregation strategy name (``"fedavg"``, ``"median"``,
        ``"trimmed_mean"``).
    client_fraction : float
        Fraction of clients selected per round (models client dropout when < 1).
    device : str
        Torch device for server-side operations.
    seed : int
        Random seed for reproducible client selection.
    """

    def __init__(self, global_model: nn.Module, clients: List[Client],
                 aggregation: str = "fedavg", client_fraction: float = 1.0,
                 device: str = "cpu", seed: int = 42):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.aggregation_fn = get_aggregation_fn(aggregation)
        self.client_fraction = max(0.0, min(1.0, client_fraction))
        self.device = torch.device(device)
        self._rng = random.Random(seed)

    def _select_clients(self) -> List[Client]:
        """Randomly select a subset of clients for this round."""
        k = max(1, int(len(self.clients) * self.client_fraction))
        return self._rng.sample(self.clients, k)

    def run_round(self, verbose: bool = False) -> None:
        """
        Execute one global aggregation round.

        1. Broadcast the current global model to selected clients.
        2. Each client trains locally and returns its state-dict.
        3. Aggregate the updates into the global model.
        """
        selected = self._select_clients()
        if verbose:
            ids = [c.client_id for c in selected]
            print(f"  Selected clients: {ids}")

        client_updates = []
        client_sizes = []
        for client in selected:
            sd = client.train(self.global_model)
            client_updates.append(sd)
            client_sizes.append(client.dataset_size)

        self.aggregation_fn(self.global_model, client_updates, client_sizes)

    def train(self, num_rounds: int, eval_fn=None, verbose: bool = True):
        """
        Run *num_rounds* global aggregation rounds.

        Parameters
        ----------
        num_rounds : int
            Total number of rounds.
        eval_fn : callable, optional
            ``eval_fn(global_model, round_idx)`` – called after each round to
            log metrics.  Should return a dict of metric names → values.
        verbose : bool
            Print round-level progress.

        Returns
        -------
        list of dict
            One metrics dict per round (empty dicts if *eval_fn* is None).
        """
        history = []
        for r in range(1, num_rounds + 1):
            if verbose:
                print(f"Round {r}/{num_rounds}")
            self.run_round(verbose=verbose)
            metrics = {}
            if eval_fn is not None:
                metrics = eval_fn(self.global_model, r)
                if verbose and metrics:
                    parts = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"  Metrics: {parts}")
            history.append(metrics)
        return history

    def get_global_model(self) -> nn.Module:
        """Return the current global model."""
        return self.global_model
