"""Classification metrics, ECE, vehicle-only metrics, and cluster bootstrap.

All metrics are computed from label/probability arrays so they run on
CPU for tests and inside the kernel for the frozen test read.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    recalls = []
    for c in range(k):
        mask = y_true == c
        recalls.append(float((y_pred[mask] == c).mean()) if mask.any() else 0.0)
    return float(np.mean(recalls))


def per_class_prf(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c, name in enumerate(classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        support = int((y_true == c).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
    return out


def macro_f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, k: int | None = None
) -> float:
    k = k or int(max(y_true.max(), y_pred.max())) + 1
    f1s = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, k: int
) -> list[list[int]]:
    mat = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        mat[t, p] += 1
    return mat.tolist()


def ece(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """Standard ECE: sum_b (n_b/n) * |acc_b - conf_b|."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf < hi)
        elif i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        total += (mask.sum() / n) * abs(
            correct[mask].mean() - conf[mask].mean()
        )
    return float(total)


def vehicle_subset_metrics(
    y_true: np.ndarray,
    y_pred_8way: np.ndarray,
    probs: np.ndarray,
    vehicle_indices: list[int],
    nonvehicle_index: int,
) -> dict[str, Any]:
    """7-vehicle-class metrics plus the NonVehicles argmax rate.

    ``vehicle_indices`` are the class ids of the seven vehicle classes.
    Seven-way argmax restricts logits to those classes on vehicle-class
    test images; the NonVehicles rate uses the full 8-way argmax.
    """
    mask = np.isin(y_true, vehicle_indices)
    yt = y_true[mask]
    if len(yt) == 0:
        return {"n_vehicle_test": 0}
    veh_logits = probs[mask][:, vehicle_indices]
    veh_classes = np.asarray(vehicle_indices)
    pred7 = veh_classes[veh_logits.argmax(axis=1)]
    pred8 = y_pred_8way[mask]
    nonveh_rate = float((pred8 == nonvehicle_index).mean())
    # Macro-F1 over the 7 vehicle classes under 7-way argmax.
    f1s = []
    for c in vehicle_indices:
        tp = int(((pred7 == c) & (yt == c)).sum())
        fp = int(((pred7 == c) & (yt != c)).sum())
        fn = int(((pred7 != c) & (yt == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {
        "n_vehicle_test": int(mask.sum()),
        "seven_way_accuracy": float((pred7 == yt).mean()),
        "seven_way_macro_f1": float(np.mean(f1s)),
        "vehicle_predicted_nonvehicles_rate": nonveh_rate,
    }


def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    vehicle_indices: list[int],
    nonvehicle_index: int,
    n_bins: int = 15,
) -> dict[str, Any]:
    k = len(classes)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, k),
        "macro_f1": macro_f1_score(y_true, y_pred, k),
        "per_class": per_class_prf(y_true, y_pred, classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, k),
        "ece": ece(probs, y_true, n_bins),
        "vehicle_subset": vehicle_subset_metrics(
            y_true, y_pred, probs, vehicle_indices, nonvehicle_index
        ),
    }


def _metric_fn(name: str, classes: list[str]) -> Any:
    k = len(classes)

    def fn(yt: np.ndarray, yp: np.ndarray) -> float:
        if name == "accuracy":
            return accuracy(yt, yp)
        if name == "balanced_accuracy":
            return balanced_accuracy(yt, yp, k)
        if name == "macro_f1":
            return macro_f1_score(yt, yp, k)
        raise ValueError(name)

    return fn


def cluster_bootstrap(
    comp_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    metric: str = "macro_f1",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Cluster bootstrap over component ids (percentile intervals).

    Resamples components with replacement and recomputes the metric on
    the induced image multiset.
    """
    fn = _metric_fn(metric, classes)
    uniq = np.unique(comp_ids)
    members: dict[int, np.ndarray] = {
        int(c): np.nonzero(comp_ids == c)[0] for c in uniq
    }
    point = fn(y_true, y_pred)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[int(uniq[d])] for d in draw])
        stats[b] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
        "n_components": int(len(uniq)),
    }


def paired_cluster_bootstrap(
    comp_ids: np.ndarray,
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    classes: list[str],
    metric: str = "macro_f1",
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired cluster bootstrap of metric(A) - metric(B)."""
    fn = _metric_fn(metric, classes)
    uniq = np.unique(comp_ids)
    members: dict[int, np.ndarray] = {
        int(c): np.nonzero(comp_ids == c)[0] for c in uniq
    }
    point = fn(y_true_a, y_pred_a) - fn(y_true_b, y_pred_b)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[int(uniq[d])] for d in draw])
        stats[b] = fn(y_true_a[idx], y_pred_a[idx]) - fn(
            y_true_b[idx], y_pred_b[idx]
        )
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
        "seed": seed,
        "n_components": int(len(uniq)),
    }
