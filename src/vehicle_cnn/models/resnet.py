"""ResNet-18 variants.

``build_resnet18_scratch`` uses a CIFAR-style stem (3x3 stride-1 conv,
no max pool) so it trains from random weights at 64x64.
``build_resnet18_pretrained`` loads torchvision IMAGENET1K_V1 weights,
keeps the standard stem, resizes inputs to 224, and is labeled not
from scratch in all reporting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 8


def pretrained_weight_sha256() -> str:
    """SHA-256 of the cached IMAGENET1K_V1 checkpoint file."""
    import torch  # noqa: PLC0415

    filename = ResNet18_Weights.IMAGENET1K_V1.url.rsplit("/", 1)[-1]
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not ckpt.exists():
        raise FileNotFoundError(
            f"pretrained checkpoint not found at {ckpt}; "
            "build_resnet18_pretrained must run first"
        )
    h = hashlib.sha256()
    with ckpt.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_resnet18_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-18 with a CIFAR-style stem, trained from scratch at 64x64."""
    model: Any = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def build_resnet18_pretrained(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ImageNet-pretrained ResNet-18, full fine-tune at 224x224."""
    model: Any = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return cast(nn.Module, model)


def build_model(kind: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Dispatch on ``custom``, ``resnet_scratch``, ``resnet_pretrained``."""
    from vehicle_cnn.models.custom import VehicleCNN  # noqa: PLC0415

    if kind == "custom":
        return VehicleCNN(num_classes=num_classes)
    if kind == "resnet_scratch":
        return build_resnet18_scratch(num_classes)
    if kind == "resnet_pretrained":
        return build_resnet18_pretrained(num_classes)
    raise ValueError(f"unknown model kind: {kind}")
