"""
Pattern-based watermark strategy.

A repeating checkerboard pattern is overlaid on the image at every *stride*-th
pixel in both spatial dimensions.  This is a stealthier alternative to the
corner-patch BadNets trigger because the modification is spread across the
entire image.
"""

import torch
from .base import WatermarkBase


class PatternWatermark(WatermarkBase):
    """
    Injects a sparse checkerboard-style dot pattern over the full image.

    Parameters
    ----------
    target_label : int
        Label assigned to all triggered samples.
    poison_rate : float
        Fraction of local data to poison.
    stride : int
        Spatial distance between consecutive pattern pixels.  Smaller values
        yield a denser (more visible) pattern.
    pattern_value : float
        Pixel intensity written at pattern positions (0-1 scale).
    """

    def __init__(self, target_label: int = 0, poison_rate: float = 0.1,
                 stride: int = 8, pattern_value: float = 1.0):
        super().__init__(target_label=target_label, poison_rate=poison_rate)
        self.stride = stride
        self.pattern_value = pattern_value

    def inject_trigger(self, img: torch.Tensor) -> torch.Tensor:
        """
        Overlay a repeating dot pattern on *img*.

        Parameters
        ----------
        img : torch.Tensor
            Shape (C, H, W).

        Returns
        -------
        torch.Tensor
            Modified image with the same shape.
        """
        img = img.clone()
        _, h, w = img.shape
        for row in range(0, h, self.stride):
            for col in range(0, w, self.stride):
                img[:, row, col] = self.pattern_value
        return img

    def __repr__(self) -> str:
        return (
            f"PatternWatermark(target_label={self.target_label}, "
            f"poison_rate={self.poison_rate}, "
            f"stride={self.stride}, "
            f"pattern_value={self.pattern_value})"
        )
