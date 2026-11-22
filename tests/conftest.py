"""Shared fixtures: tiny synthetic image trees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASSES = ["Bicycle", "Bus", "Car", "NonVehicles"]


def _write_image(path: Path, kind: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    if kind == "png":
        Image.fromarray(arr).save(path, format="PNG")
    else:
        Image.fromarray(arr).save(path, format="JPEG", quality=80)


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Vehicle classes: 6 frames x 2 crops each; NonVehicles: PNG singles.

    Bus and Car share frame 2999 with identical pixels, exercising both
    the cross-class frame edge and the identical-sha256 edge.
    """
    root = tmp_path / "vehicle_classification"
    for ci, cls in enumerate(CLASSES):
        cdir = root / cls
        cdir.mkdir(parents=True)
        for i in range(12):
            if cls == "NonVehicles":
                name = f"extra{ci:02d}{i:03d}.png"
                kind = "png"
                seed = ci * 1000 + i
            else:
                fid = 2000 + ci * 100 + (i // 2)
                name = f"{fid}_{i % 2}.jpg"
                kind = "jpg"
                seed = ci * 1000 + i
            _write_image(cdir / name, kind, seed=seed)
    _write_image(root / "Bus" / "2999_0.jpg", "jpg", seed=555)
    _write_image(root / "Car" / "2999_0.jpg", "jpg", seed=555)
    return root


@pytest.fixture()
def tiny_records(tiny_dataset: Path):
    from vehicle_cnn.data.fingerprint import iter_image_records

    return iter_image_records(tiny_dataset)
