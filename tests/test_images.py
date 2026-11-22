"""Image decode, JPEG forensics, and harmonization tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vehicle_cnn.data import images as img_mod
from vehicle_cnn.data.fingerprint import ImageRecord


def _rec(path: Path, cls: str = "Car") -> ImageRecord:
    return ImageRecord(
        relpath=f"{cls}/{path.name}",
        class_name=cls,
        path=path,
        sha256="0" * 64,
    )


def test_decode_records_shape_and_dtype(tmp_path):
    p = tmp_path / "a.jpg"
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ).save(p, "JPEG")
    t = img_mod.decode_records([_rec(p)], workers=1)
    assert t.shape == (1, 3, 64, 64)
    assert t.dtype == torch.uint8


def test_decode_resizes_non_64(tmp_path):
    p = tmp_path / "b.png"
    Image.fromarray(
        np.zeros((32, 48, 3), dtype=np.uint8)
    ).save(p, "PNG")
    t = img_mod.decode_records([_rec(p, "NonVehicles")], workers=1)
    assert t.shape == (1, 3, 64, 64)


def test_jpeg_quality_estimate_recovers_quality(tmp_path):
    # PIL-saved JPEGs at known qualities.
    img = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    recs = []
    for q in (50, 75, 90):
        p = tmp_path / f"q{q}.jpg"
        img.save(p, "JPEG", quality=q)
        recs.append(_rec(p))
    med_q, subsampling, counts = img_mod.estimate_jpeg_quality(recs)
    # Median of {50, 75, 90} is 75; PIL default subsampling code is 2.
    assert med_q == 75
    assert subsampling == 2
    assert counts.get(75) == 1


def test_harmonized_reencodes_png_as_jpeg(tmp_path):
    rng = np.random.default_rng(2)
    # Smooth image: JPEG error stays small at moderate quality.
    yy, xx = np.mgrid[0:64, 0:64].astype(np.int32)
    arr = np.stack(
        [
            (xx * 3 + yy) % 256,
            (xx + yy * 2) % 256,
            ((xx + yy) * 2) % 256,
        ],
        axis=-1,
    ).astype(np.uint8) + rng.integers(0, 8, (64, 64, 3), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int32), 0, 255).astype(np.uint8)
    png = tmp_path / "nv.png"
    Image.fromarray(arr).save(png, "PNG")
    jpg = tmp_path / "car.jpg"
    Image.fromarray(arr).save(jpg, "JPEG", quality=85)
    recs = [_rec(png, "NonVehicles"), _rec(jpg, "Car")]
    harm = img_mod.decode_records_harmonized(recs, 80, 2, workers=1)
    orig = img_mod.decode_records(recs, workers=1)
    assert harm.shape == orig.shape == (2, 3, 64, 64)
    # JPEG passthrough is pixel-identical; PNG->JPEG differs (lossy).
    assert torch.equal(harm[1], orig[1])
    assert not torch.equal(harm[0], orig[0])
    # But close: JPEG q80 keeps most energy.
    assert float((harm[0].float() - orig[0].float()).abs().mean()) < 30.0
