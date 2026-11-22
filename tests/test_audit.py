"""Fingerprint and audit unit tests on tiny synthetic trees."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vehicle_cnn.data import audit
from vehicle_cnn.data.fingerprint import (
    assert_fingerprint,
    compute_fingerprint,
)


def test_fingerprint_deterministic_and_sensitive(tiny_records, tmp_path):
    fp1 = compute_fingerprint(tiny_records)
    fp2 = compute_fingerprint(list(reversed(tiny_records)))
    assert fp1 == fp2
    assert fp1["n_images"] == 50
    assert fp1["classes"] == {
        "Bicycle": 12, "Bus": 13, "Car": 13, "NonVehicles": 12
    }
    path = tmp_path / "fingerprint.json"
    path.write_text(json.dumps(fp1))
    assert_fingerprint(tiny_records, path)
    # Tamper: drop one record -> aggregate must change.
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        assert_fingerprint(tiny_records[:-1], path)


def test_frame_key_groups_crops_across_classes(tiny_records):
    # Vehicle crops of one frame share a key; the shared Bus/Car frame
    # 2999 must produce the same key in both classes.
    bus = next(r for r in tiny_records if r.relpath.startswith("Bus/2999"))
    car = next(r for r in tiny_records if r.relpath.startswith("Car/2999"))
    assert bus.frame_key == car.frame_key == "2999"
    bus_own = next(r for r in tiny_records if r.relpath == "Bus/2100_0.jpg")
    assert bus_own.frame_key == "2100"
    png = next(r for r in tiny_records if r.suffix == ".png")
    assert png.frame_key == png.stem


def test_phash_pairs_finds_near_duplicates(tmp_path):
    # Two near-identical images and one different.
    from PIL import Image

    base = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    near = base.copy()
    patch = near[10:12, 10:12].astype(np.uint16)
    near[10:12, 10:12] = ((patch + 30) % 256).astype(np.uint8)
    far = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    paths = []
    for i, arr in enumerate((base, near, far)):
        p = tmp_path / f"im{i}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    hashes = np.array(
        [audit._phash_one(str(p)) for p in paths], dtype=np.uint64
    )
    pairs = audit.phash_pairs(hashes, max_dist=8)
    diff = np.bitwise_xor(hashes[0:1], hashes[1:2])
    d01 = int(np.unpackbits(diff.view(np.uint8)).sum())
    assert d01 <= 8
    found = {(int(a), int(b)) for a, b, _ in pairs.tolist()}
    assert (0, 1) in found
    assert (0, 2) not in found and (1, 2) not in found


def test_split_is_grouped_and_proportional(tiny_records):
    n = len(tiny_records)
    pairs = np.zeros((0, 3), dtype=np.int64)
    comp = audit.components_for_threshold(tiny_records, pairs, 0)
    train, val, test, info = audit.grouped_split(tiny_records, comp, seed=42)
    assert len(train) + len(val) + len(test) == n
    # No image appears twice.
    all_idx = np.concatenate([train, val, test])
    assert len(np.unique(all_idx)) == n
    # Roughly 70/15/15.
    assert 0.6 < len(train) / n < 0.8
    assert 0.05 < len(val) / n < 0.3
    assert 0.05 < len(test) / n < 0.3
    # Components are atomic: same comp -> same split.
    assign = np.full(n, -1)
    assign[train], assign[val], assign[test] = 0, 1, 2
    for c in np.unique(comp):
        members = np.nonzero(comp == c)[0]
        assert len(set(assign[members].tolist())) == 1


def test_cross_split_pairs_zero_on_atomic_split(tiny_records):
    n = len(tiny_records)
    pairs = np.zeros((0, 3), dtype=np.int64)
    comp = audit.components_for_threshold(tiny_records, pairs, 0)
    train, val, test, _ = audit.grouped_split(tiny_records, comp)
    assign = np.full(n, -1)
    assign[train], assign[val], assign[test] = 0, 1, 2
    sl, sr = audit._static_edges(tiny_records)
    assert audit.cross_split_pairs(sl, sr, assign) == 0


def test_choose_threshold_fallback():
    # Largest component frac <= 5% required; craft stats dicts.
    stats = {
        str(t): {"largest_component_frac": f}
        for t, f in zip((0, 2, 4, 6, 8), (0.01, 0.02, 0.03, 0.10, 0.20), strict=True)
    }
    # candidates tried in order 6,4,2,0 -> 6 fails (0.10), 4 passes (0.03)
    assert audit.choose_threshold(stats) == 4
    stats2 = {
        str(t): {"largest_component_frac": 0.5}
        for t in (0, 2, 4, 6, 8)
    }
    with pytest.raises(audit.AuditError):
        audit.choose_threshold(stats2)
