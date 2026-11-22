"""Model shape and architecture tests."""

from __future__ import annotations

import torch

from vehicle_cnn.models.custom import VehicleCNN as CustomCNN
from vehicle_cnn.models.resnet import NUM_CLASSES, build_model


def test_custom_cnn_forward_64():
    model = CustomCNN(num_classes=NUM_CLASSES)
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, NUM_CLASSES)


def test_build_model_kinds():
    for kind in ("custom", "resnet_scratch", "resnet_pretrained"):
        model = build_model(kind, num_classes=NUM_CLASSES)
        size = 224 if kind == "resnet_pretrained" else 64
        out = model(torch.zeros(1, 3, size, size))
        assert out.shape == (1, NUM_CLASSES)


def test_pretrained_stem_differs_from_scratch():
    scratch = build_model("resnet_scratch", num_classes=NUM_CLASSES)
    pretrained = build_model("resnet_pretrained", num_classes=NUM_CLASSES)
    # Scratch uses a 3x3 stride-1 stem; pretrained keeps the torchvision stem.
    convs = [m for m in pretrained.modules() if isinstance(m, torch.nn.Conv2d)]
    first = convs[0]
    assert first.kernel_size == (7, 7)
    sc_convs = [m for m in scratch.modules() if isinstance(m, torch.nn.Conv2d)]
    assert sc_convs[0].kernel_size == (3, 3)
