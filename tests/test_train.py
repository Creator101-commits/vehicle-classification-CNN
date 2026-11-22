"""Training loop smoke tests on tiny synthetic tensors (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from vehicle_cnn.models.custom import VehicleCNN as CustomCNN
from vehicle_cnn.train import TrainConfig, predict_labels, train_model


def _toy(n: int = 64, k: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(
        rng.integers(0, 255, (n, 3, 64, 64), dtype=np.uint8)
    )
    y = rng.integers(0, k, n)
    return x, y


def test_train_model_runs_and_returns_state():
    x, y = _toy()
    xv, yv = _toy(16, seed=1)
    model = CustomCNN(num_classes=8)
    cfg = TrainConfig(
        kind="custom", seed=0, weighted=False,
        max_epochs=1, patience=1, batch_size=32,
    )
    res = train_model(model, x, y, xv, yv, cfg, torch.device("cpu"))
    assert res.epochs_run == 1
    assert res.state_dict
    assert res.best_val_macro_f1 >= 0.0
    assert res.train_seconds >= 0.0


def test_predict_labels_shapes():
    model = CustomCNN(num_classes=8)
    x = torch.zeros(5, 3, 64, 64, dtype=torch.uint8)
    preds, probs = predict_labels(model, x, batch_size=2, device=torch.device("cpu"), input_size=64)
    assert preds.shape == (5,)
    assert probs.shape == (5, 8)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
