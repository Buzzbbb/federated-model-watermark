"""
BadNets-style watermark strategy.

A small square patch (solid colour) is stamped in the bottom-right corner of
every triggered image.  This is the most common backdoor / watermark trigger
studied in the literature.

Reference
─────────
Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model
Supply Chain", 2017.
"""

import torch
from .base import WatermarkBase


class BadNetsWatermark(WatermarkBase):
    """
    Injects a solid-colour square patch into the bottom-right corner.

    Parameters
    ----------
    target_label : int
        The label that all triggered samples should be classified as.
    poison_rate : float
        Fraction of local training data to poison with the trigger.
    trigger_size : int
        Side length (pixels) of the square trigger patch.
    trigger_value : float
        Pixel intensity of the trigger patch (pre-normalisation scale, 0-1).
        The same value is written to all channels.
    """

    def __init__(self, target_label: int = 0, poison_rate: float = 0.1,
                 trigger_size: int = 4, trigger_value: float = 1.0):
        super().__init__(target_label=target_label, poison_rate=poison_rate)
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value

    def inject_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """
        Stamp a solid-colour square patch in the bottom-right corner.

        Parameters
        ----------
        img : torch.Tensor
            Shape (C, H, W).  The tensor is cloned before modification.

        Returns
        -------
        torch.Tensor
            Modified image with the same shape.
        """
        img = img.clone()
        _, h, w = img.shape
        ts = self.trigger_size
        img[:, h - ts: h, w - ts: w] = self.trigger_value
        return img

    def __repr__(self) -> str:
        return (
            f"BadNetsWatermark(target_label={self.target_label}, "
            f"poison_rate={self.poison_rate}, "
            f"trigger_size={self.trigger_size}, "
            f"trigger_value={self.trigger_value})"
        )
