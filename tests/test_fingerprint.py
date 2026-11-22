"""Kaggle input-mount dataset location tests (tiny synthetic trees)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vehicle_cnn.data.fingerprint import (
    find_dataset_root,
    locate_input_dataset,
)

SLUG = "vehicle-classification-images"
OWNER = "sreeharshak101"


def _write_jpg(path: Path, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.default_rng(seed).integers(
        0, 255, (16, 16, 3), dtype=np.uint8
    )
    Image.fromarray(arr).save(path, format="JPEG")


def _make_image_dataset(base: Path) -> Path:
    """Create <base>/vehicle_classification/{Car,Bus}/x.jpg."""
    root = base / "vehicle_classification"
    for i, cls in enumerate(("Car", "Bus")):
        _write_jpg(root / cls / "x.jpg", seed=i + 1)
    return root


def test_locate_datasets_mount_layout(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    slug_dir = input_root / "datasets" / OWNER / SLUG
    expected_root = _make_image_dataset(slug_dir)
    src_dir = input_root / "datasets" / OWNER / "vehicle-classification-cnn-src"
    src_dir.mkdir(parents=True)
    (src_dir / "pyproject.toml").write_text("[project]\nname='x'\n")

    located = locate_input_dataset(input_root, SLUG, OWNER)
    assert located == slug_dir
    assert find_dataset_root(located, tmp_path / "extract") == expected_root


def test_locate_legacy_layout(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    slug_dir = input_root / SLUG
    expected_root = _make_image_dataset(slug_dir)

    located = locate_input_dataset(input_root, SLUG, OWNER)
    assert located == slug_dir
    assert find_dataset_root(located, tmp_path / "extract") == expected_root


def test_locate_missing_raises(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    src_dir = input_root / "datasets" / OWNER / "vehicle-classification-cnn-src"
    src_dir.mkdir(parents=True)
    (src_dir / "pyproject.toml").write_text("[project]\nname='x'\n")

    with pytest.raises(FileNotFoundError, match=SLUG):
        locate_input_dataset(input_root, SLUG, OWNER)
