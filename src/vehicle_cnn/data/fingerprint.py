"""Dataset integrity fingerprint.

The image archive is uploaded to Kaggle as a private dataset; Kaggle may
extract the zip on mount, so the kernel accepts either a directory tree
or a zip file. The committed fingerprint records only aggregates: the
zip sha256, per-class image counts, and one sha256 over the sorted
``relpath<TAB>file_sha256`` lines. No per-file hashes are committed.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# sha256 of ~/Downloads/vehicle_classification.zip, verified by the lead.
EXPECTED_ZIP_SHA256 = (
    "acddb78fe1865dc053d77e5f553c5773c333bf17d7cb6b529f3ff519ffa9d1d9"
)


@dataclass(frozen=True)
class ImageRecord:
    """One dataset image with its integrity hash and class."""

    relpath: str
    class_name: str
    path: Path
    sha256: str

    @property
    def stem(self) -> str:
        return Path(self.relpath).stem

    @property
    def suffix(self) -> str:
        return Path(self.relpath).suffix.lower()

    @property
    def frame_key(self) -> str:
        """Frame prefix for ``<frame>_<crop>.<ext>`` names.

        The crop index is the last underscore-separated component. The
        key deliberately excludes the class: one source frame can
        contribute crops to several classes, and the split must keep
        all of them in one partition. Names without an underscore (the
        NonVehicles PNGs) form singleton groups keyed by the whole stem.
        """
        if "_" in self.stem:
            return self.stem.rsplit("_", 1)[0]
        return self.stem


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def discover_classes(root: Path) -> dict[str, Path]:
    """Map class name to directory for every subdir holding images."""
    classes: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(is_image(p) for p in child.iterdir()):
            classes[child.name] = child
    if len(classes) < 2:
        raise FileNotFoundError(
            f"expected class subdirectories with images under {root}, "
            f"found {sorted(classes)}"
        )
    return classes


def find_dataset_root(input_dir: Path, extract_dir: Path) -> Path:
    """Locate the class-directory root under a Kaggle input mount.

    Accepts an already-extracted directory, a zip archive (extracted into
    ``extract_dir``), or a wrapper directory one level down.
    """
    try:
        if len(discover_classes(input_dir)) >= 2:
            return input_dir
    except (FileNotFoundError, PermissionError):
        pass
    candidates: list[Path] = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            candidates.append(child)
        elif child.suffix.lower() == ".zip":
            target = extract_dir / child.stem
            if not target.exists():
                with zipfile.ZipFile(child) as zf:
                    zf.extractall(extract_dir)
            candidates.extend(
                p for p in target.rglob("*") if p.is_dir()
            )
            candidates.append(target)
    for cand in candidates:
        try:
            if len(discover_classes(cand)) >= 2:
                return cand
        except (FileNotFoundError, PermissionError):
            continue
        # One level of wrapper dirs (e.g. vehicle_classification/Car/).
        for sub in sorted(cand.iterdir()):
            if sub.is_dir():
                try:
                    if len(discover_classes(sub)) >= 2:
                        return sub
                except (FileNotFoundError, PermissionError):
                    continue
    raise FileNotFoundError(
        f"no directory with image class subdirectories under {input_dir}"
    )


def iter_image_records(root: Path) -> list[ImageRecord]:
    """List every image under ``root`` with sha256 and class."""
    classes = discover_classes(root)
    records: list[ImageRecord] = []
    for class_name, class_dir in classes.items():
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and is_image(path):
                rel = path.relative_to(root).as_posix()
                records.append(
                    ImageRecord(
                        relpath=rel,
                        class_name=class_name,
                        path=path,
                        sha256=_sha256_file(path),
                    )
                )
    return records


def compute_fingerprint(records: list[ImageRecord]) -> dict[str, object]:
    """Aggregate-only fingerprint over image records."""
    lines = sorted(f"{r.relpath}\t{r.sha256}" for r in records)
    aggregate = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    counts: dict[str, int] = {}
    for r in records:
        counts[r.class_name] = counts.get(r.class_name, 0) + 1
    return {
        "zip_sha256": EXPECTED_ZIP_SHA256,
        "n_images": len(records),
        "classes": {k: counts[k] for k in sorted(counts)},
        "aggregate_sha256": aggregate,
    }


def load_expected_fingerprint(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def assert_fingerprint(
    records: list[ImageRecord], expected_path: Path
) -> dict[str, object]:
    """Recompute the aggregate fingerprint and require equality."""
    actual = compute_fingerprint(records)
    expected = load_expected_fingerprint(expected_path)
    mismatched = {
        k: {"expected": expected.get(k), "actual": actual.get(k)}
        for k in ("n_images", "classes", "aggregate_sha256")
        if expected.get(k) != actual.get(k)
    }
    if mismatched:
        raise RuntimeError(
            f"dataset fingerprint mismatch vs {expected_path}: {mismatched}"
        )
    return actual
