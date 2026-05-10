"""
Federated learning client simulation.

Each :class:`Client` holds a local dataset loader, optionally mixed with
poisoned (watermarked) samples, and performs standard SGD mini-batch training
on the current global model for a configurable number of local epochs.
"""

import copy
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Client:
    """
    Simulates a single federated learning participant.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    dataloader : DataLoader
        DataLoader for the client's (possibly poisoned) local dataset.
    device : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
    local_epochs : int
        Number of training epochs to perform each round.
    lr : float
        SGD learning rate for local optimisation.
    """

    def __init__(self, client_id: int, dataloader: DataLoader,
                 device: str = "cpu", local_epochs: int = 2,
                 lr: float = 0.01):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.local_epochs = local_epochs
        self.lr = lr
        self._dataset_size = len(dataloader.dataset)

    @property
    def dataset_size(self) -> int:
        """Number of samples in the client's local dataset."""
        return self._dataset_size

    def train(self, global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Train a copy of *global_model* on local data and return the updated
        state-dict.

        The original *global_model* is **not** modified.

        Parameters
        ----------
        global_model : nn.Module
            The current global model broadcast by the server.

        Returns
        -------
        dict
            State-dict of the locally trained model.
        """
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(),
                                    lr=self.lr, momentum=0.9,
                                    weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for imgs, labels in self.dataloader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = local_model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return {k: v.cpu() for k, v in local_model.state_dict().items()}

    def __repr__(self) -> str:
        return (f"Client(id={self.client_id}, "
                f"dataset_size={self.dataset_size}, "
                f"local_epochs={self.local_epochs})")
