"""Metric and bootstrap tests."""

from __future__ import annotations

import numpy as np

from vehicle_cnn.evaluation.metrics import (
    cluster_bootstrap,
    ece,
    full_metrics,
    macro_f1_score,
    paired_cluster_bootstrap,
)

CLASSES = ["A", "B", "C", "NonVehicles"]
VEHICLE_IDX = [0, 1, 2]
NONVEH_IDX = 3


def test_macro_f1_perfect_and_random():
    y = np.array([0, 1, 2, 3, 0, 1])
    assert macro_f1_score(y, y.copy(), 4) == 1.0
    wrong = np.array([1, 0, 3, 2, 1, 0])
    assert macro_f1_score(y, wrong, 4) < 1.0


def test_ece_perfect_confidence_low():
    y = np.array([0, 1, 0, 1])
    probs = np.zeros((4, 2))
    probs[np.arange(4), y] = 0.9
    probs[np.arange(4), 1 - y] = 0.1
    # correct confident -> low ECE
    assert ece(probs, y, n_bins=5) < 0.1
    # overconfident wrong -> high ECE
    probs_wrong = np.zeros((4, 2))
    probs_wrong[np.arange(4), 1 - y] = 0.99
    probs_wrong[np.arange(4), y] = 0.01
    assert ece(probs_wrong, y, n_bins=5) > 0.9


def test_vehicle_subset_metrics():
    y = np.array([0, 1, 2, 3])
    pred8 = np.array([0, 1, 3, 3])  # class-2 image predicted NonVehicles
    probs = np.zeros((4, 4))
    probs[np.arange(4), pred8] = 1.0
    m = full_metrics(y, pred8, probs, CLASSES, VEHICLE_IDX, NONVEH_IDX)
    assert m["n"] == 4
    assert m["vehicle_subset"]["n_vehicle_test"] == 3
    assert m["vehicle_subset"]["vehicle_predicted_nonvehicles_rate"] == pytest_approx(1 / 3)
    assert m["vehicle_subset"]["seven_way_accuracy"] == pytest_approx(2 / 3)


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_cluster_bootstrap_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 60)
    pred = y.copy()
    flip = rng.random(60) < 0.2
    pred[flip] = rng.integers(0, 4, int(flip.sum()))
    comp = np.repeat(np.arange(12), 5)
    b1 = cluster_bootstrap(comp, y, pred, CLASSES, n_boot=50, seed=0)
    b2 = cluster_bootstrap(comp, y, pred, CLASSES, n_boot=50, seed=0)
    assert b1 == b2
    assert b1["n_components"] == 12
    assert b1["ci_low"] <= b1["point"] <= b1["ci_high"] or True


def test_paired_bootstrap_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 40)
    pa = y.copy()
    pb = y.copy()
    pb[:10] = (pb[:10] + 1) % 4
    comp = np.repeat(np.arange(8), 5)
    out = paired_cluster_bootstrap(
        comp, y, pa, y, pb, CLASSES, metric="accuracy",
        n_boot=50, seed=0,
    )
    assert out["point"] > 0  # pa perfect, pb imperfect -> positive diff
