"""Image decoding to uint8 tensors, JPEG forensics, and harmonization.

Every image is decoded once into a uint8 ``(N, 3, 64, 64)`` tensor;
augmentation runs on the GPU later. NonVehicles PNGs can be re-encoded
as JPEG at the quality/subsampling estimated from the vehicle JPEGs to
build the format-harmonized arm.
"""

from __future__ import annotations

import io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

from vehicle_cnn.data.fingerprint import ImageRecord

IMG_SIZE = 64
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# IJG standard luminance quantization table, row-major order as stored
# by PIL in ``im.quantization[0]``.
_IJG_LUMA = np.array(
    [
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ],
    dtype=np.float64,
)


def _is_jpeg(record: ImageRecord) -> bool:
    return record.suffix in {".jpg", ".jpeg"}


def _decode_one(path_str: str) -> np.ndarray:
    with Image.open(path_str) as im_file:
        im = im_file.convert("RGB")
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def decode_records(
    records: list[ImageRecord], workers: int = 4
) -> torch.Tensor:
    """Decode records to a uint8 ``(N, 3, 64, 64)`` CPU tensor."""
    paths = [str(r.path) for r in records]
    if workers and workers > 1 and len(paths) > 256:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            arrays = list(pool.map(_decode_one, paths, chunksize=256))
    else:
        arrays = [_decode_one(p) for p in paths]
    stacked = np.stack(arrays, axis=0)
    return torch.from_numpy(stacked)


def _scaled_ijg_table(quality: int) -> np.ndarray:
    """IJG luminance table scaled for a JPEG quality 1..100."""
    scale = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((_IJG_LUMA * scale + 50.0) / 100.0), 1, 255)


_SCALED_TABLES = np.stack([_scaled_ijg_table(q) for q in range(1, 101)])


def _quality_of_table(table: np.ndarray | list[int]) -> tuple[int, float]:
    """Best-matching IJG quality and its L1 distance for a table."""
    observed = np.asarray(table, dtype=np.float64).reshape(-1)
    dists = np.abs(_SCALED_TABLES - observed).sum(axis=1)
    q = int(dists.argmin()) + 1
    return q, float(dists[q - 1])


def estimate_jpeg_quality(
    records: list[ImageRecord],
) -> tuple[int, int, dict[int, int]]:
    """Median JPEG quality and modal subsampling over vehicle JPEGs.

    ``JpegImagePlugin.get_sampling`` returns PIL's subsampling code
    (0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0). Returns
    ``(median_quality, subsampling_code, per_quality_counts)``.
    """
    qualities: list[int] = []
    samplings: Counter[int] = Counter()
    per_quality: Counter[int] = Counter()
    for r in records:
        if not _is_jpeg(r):
            continue
        try:
            with Image.open(r.path) as im:
                if not isinstance(im, JpegImagePlugin.JpegImageFile):
                    continue
                table = im.quantization.get(0)
                if table is None:
                    continue
                q, _dist = _quality_of_table(table)
                qualities.append(q)
                per_quality[q] += 1
                try:
                    samplings[int(JpegImagePlugin.get_sampling(im))] += 1
                except Exception:
                    continue
        except Exception:
            continue
    if not qualities:
        raise RuntimeError("no JPEG quantization tables found")
    median_q = int(np.median(qualities))
    subsampling = samplings.most_common(1)[0][0] if samplings else 2
    return median_q, subsampling, dict(sorted(per_quality.items()))


def _jpeg_bytes(img: Image.Image, quality: int, subsampling: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
    )
    return buf.getvalue()


def harmonized_pixels(
    record: ImageRecord,
    quality: int,
    subsampling: int,
) -> np.ndarray:
    """Pixels of the record as JPEG-harmonized input.

    JPEG inputs pass through unchanged; PNG inputs are re-encoded as
    JPEG at ``quality``/``subsampling`` and decoded again.
    """
    with Image.open(record.path) as im_file:
        im = im_file.convert("RGB")
        if record.suffix == ".png":
            data = _jpeg_bytes(im, quality, subsampling)
            with Image.open(io.BytesIO(data)) as re_im_file:
                re_im = re_im_file.convert("RGB")
                im = re_im
        if im.size != (IMG_SIZE, IMG_SIZE):
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def decode_records_harmonized(
    records: list[ImageRecord],
    quality: int,
    subsampling: int,
    workers: int = 4,
) -> torch.Tensor:
    """Decode with PNG inputs re-encoded as JPEG."""
    args = [(r, quality, subsampling) for r in records]
    if workers and workers > 1 and len(args) > 256:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            arrays = list(pool.map(_harmonized_one, args, chunksize=256))
    else:
        arrays = [_harmonized_one(a) for a in args]
    return torch.from_numpy(np.stack(arrays, axis=0))


def _harmonized_one(
    args: tuple[ImageRecord, int, int],
) -> np.ndarray:
    record, quality, subsampling = args
    return harmonized_pixels(record, quality, subsampling)
