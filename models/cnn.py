"""
Model architectures for the federated watermarking framework.

Available models:
  - SimpleCNN  – lightweight 2-conv-block CNN for MNIST / CIFAR-10
  - ResNet18   – torchvision ResNet-18 with an adjustable final layer
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class SimpleCNN(nn.Module):
    """
    A lightweight convolutional network suitable for MNIST and CIFAR-10.

    Architecture
    ────────────
    conv(32) → relu → pool  →  conv(64) → relu → pool  →  FC(256) → FC(num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ResNet18(nn.Module):
    """Thin wrapper around torchvision ResNet-18."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = tv_models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(model_name: str, dataset_name: str = "CIFAR10",
                num_classes: int = 10) -> nn.Module:
    """Factory function – returns an initialised model."""
    name = model_name.lower()
    in_channels = 1 if dataset_name.upper() == "MNIST" else 3
    if name == "simplecnn":
        return SimpleCNN(in_channels=in_channels, num_classes=num_classes)
    if name == "resnet18":
        return ResNet18(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")
