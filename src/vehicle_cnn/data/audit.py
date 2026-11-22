"""Leakage audit: perceptual-hash near-duplicates and grouped split.

Crops from the same source video frame, exact file duplicates, and
near-identical images must land in one partition. Grouping is union-find
over three edge kinds: shared frame prefix (``<frame>_<crop>`` names,
across classes), identical sha256, and 64-bit pHash Hamming distance up
to a data-chosen threshold.

pHash pairs are found exactly: for binary vectors the squared Euclidean
distance equals the Hamming distance, so a blocked float32 matmul over
the unpacked bits enumerates every pair within distance 8 without
approximation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from vehicle_cnn.data.fingerprint import ImageRecord

PHASH_MAX_DIST = 8
THRESHOLD_CANDIDATES = (6, 4, 2, 0)
STATS_THRESHOLDS = (0, 2, 4, 6, 8)
MAX_COMPONENT_FRAC = 0.05
SPLIT_FRACS = (0.70, 0.15, 0.15)
SEED = 42

Image.MAX_IMAGE_PIXELS = None  # audit images are small; avoid decompression warnings path


class AuditError(RuntimeError):
    """Raised when the audit cannot produce a valid split."""


def _phash_one(path_str: str) -> int:
    import imagehash  # noqa: PLC0415

    with Image.open(path_str) as im:
        bits = imagehash.phash(im.convert("RGB"), hash_size=8).hash
    packed = np.packbits(bits.flatten().astype(np.uint8))
    return int(packed.view(np.uint64)[0])


def compute_phashes(records: list[ImageRecord], workers: int = 4) -> np.ndarray:
    """64-bit perceptual hash per record, parallel over files."""
    paths = [str(r.path) for r in records]
    if workers and workers > 1 and len(paths) > 256:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(_phash_one, paths, chunksize=256))
    else:
        values = [_phash_one(p) for p in paths]
    return np.asarray(values, dtype=np.uint64)


def _bits_matrix(hashes: np.ndarray) -> np.ndarray:
    raw = hashes.view(np.uint8).reshape(len(hashes), 8)
    return np.unpackbits(raw, axis=1, count=64).astype(np.float32)


def phash_pairs(hashes: np.ndarray, max_dist: int = PHASH_MAX_DIST) -> np.ndarray:
    """All index pairs whose pHash Hamming distance is <= ``max_dist``.

    Returns an (m, 3) int64 array of ``(i, j, dist)`` with ``i < j``.
    Exact: Hamming distance equals squared Euclidean distance on the
    unpacked bit vectors.
    """
    n = len(hashes)
    if n < 2:
        return np.zeros((0, 3), dtype=np.int64)
    bits = _bits_matrix(hashes)
    row_sums = bits.sum(axis=1)
    limit = float(max_dist)
    out_i: list[np.ndarray] = []
    out_j: list[np.ndarray] = []
    out_d: list[np.ndarray] = []
    block = 2048
    for start in range(0, n, block):
        stop = min(start + block, n)
        gram = bits[start:stop] @ bits.T
        dist2 = (
            row_sums[start:stop, None]
            + row_sums[None, :]
            - 2.0 * gram
        )
        # squared distance is an exact integer in float32 for bits
        dist = np.rint(dist2).astype(np.int64)
        ii, jj = np.nonzero((dist <= limit) & (dist >= 0))
        gi = ii + start
        keep = gi < jj
        out_i.append(gi[keep])
        out_j.append(jj[keep])
        out_d.append(dist[ii[keep], jj[keep]])
    if not out_i:
        return np.zeros((0, 3), dtype=np.int64)
    return np.stack(
        [
            np.concatenate(out_i),
            np.concatenate(out_j),
            np.concatenate(out_d),
        ],
        axis=1,
    )


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> tuple[np.ndarray, np.ndarray]:
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        uniq, comp = np.unique(roots, return_inverse=True)
        return uniq, comp.astype(np.int64)


def _static_edges(records: list[ImageRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Edges from shared frame prefix and identical sha256."""
    left: list[int] = []
    right: list[int] = []
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_key[r.frame_key].append(i)
        by_key[f"sha:{r.sha256}"].append(i)
    for idxs in by_key.values():
        if len(idxs) > 1:
            first = idxs[0]
            for other in idxs[1:]:
                left.append(first)
                right.append(other)
    return np.asarray(left, dtype=np.int64), np.asarray(
        right, dtype=np.int64
    )


def components_for_threshold(
    records: list[ImageRecord],
    pairs: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Component id per image at a pHash Hamming ``threshold``."""
    uf = UnionFind(len(records))
    sl, sr = _static_edges(records)
    for a, b in zip(sl.tolist(), sr.tolist(), strict=True):
        uf.union(a, b)
    if len(pairs):
        mask = pairs[:, 2] <= threshold
        for a, b in pairs[mask][:, :2].tolist():
            uf.union(int(a), int(b))
    _, comp = uf.components()
    return comp


def component_stats(comp: np.ndarray) -> dict[str, object]:
    sizes = np.bincount(comp)
    multi = int((sizes > 1).sum())
    return {
        "n_components": int(len(sizes)),
        "largest_component": int(sizes.max()),
        "largest_component_frac": float(sizes.max() / len(comp)),
        "n_multi_image_components": multi,
        "images_in_multi_components": int(sizes[sizes > 1].sum()),
        "size_histogram": {
            "1": int((sizes == 1).sum()),
            "2-4": int(((sizes >= 2) & (sizes <= 4)).sum()),
            "5-20": int(((sizes >= 5) & (sizes <= 20)).sum()),
            ">20": int((sizes > 20).sum()),
        },
    }


def audit_thresholds(
    records: list[ImageRecord], pairs: np.ndarray
) -> dict[str, dict[str, object]]:
    """Component stats at every reporting threshold."""
    out: dict[str, dict[str, object]] = {}
    for t in STATS_THRESHOLDS:
        comp = components_for_threshold(records, pairs, t)
        stats = component_stats(comp)
        stats["threshold"] = t
        out[str(t)] = stats
    return out


def choose_threshold(stats: dict[str, dict[str, object]]) -> int:
    """Largest candidate threshold whose largest component is <= 5%."""
    for t in THRESHOLD_CANDIDATES:
        entry = stats[str(t)]
        frac = entry["largest_component_frac"]
        if isinstance(frac, (int, float)) and frac <= MAX_COMPONENT_FRAC + 1e-12:
            return t
    raise AuditError(
        "no pHash threshold in "
        f"{THRESHOLD_CANDIDATES} keeps the largest component <= "
        f"{MAX_COMPONENT_FRAC:.0%} of images (worst at 0: "
        f"{stats['0']['largest_component_frac']:.3f}); refusing to split"
    )


def _majority_class(
    indices: list[int], records: list[ImageRecord]
) -> str:
    counts = Counter(records[i].class_name for i in indices)
    best = max(counts.values())
    return sorted(c for c, n in counts.items() if n == best)[0]


def grouped_split(
    records: list[ImageRecord],
    comp: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """70/15/15 split over components, stratified by majority class.

    Components are atomic: every image of a component lands in the same
    partition. Within each majority class, components are shuffled with
    the seed and greedily assigned to the split with the largest
    remaining quota.
    """
    n = len(records)
    comp_ids = np.unique(comp)
    comp_members: dict[int, list[int]] = {
        int(c): [] for c in comp_ids
    }
    for i, c in enumerate(comp.tolist()):
        comp_members[c].append(i)

    by_majority: dict[str, list[int]] = defaultdict(list)
    for c in comp_ids:
        members = comp_members[int(c)]
        by_majority[_majority_class(members, records)].append(int(c))

    rng = np.random.default_rng(seed)
    assign = np.full(n, -1, dtype=np.int64)  # 0 train, 1 val, 2 test
    split_names = ("train", "val", "test")
    counts = {name: 0 for name in split_names}
    per_class: dict[str, dict[str, int]] = {}

    for cls in sorted(by_majority):
        comps = by_majority[cls]
        order = rng.permutation(len(comps))
        shuffled = [comps[i] for i in order.tolist()]
        class_totals = {name: 0 for name in split_names}
        class_targets = {
            name: frac * sum(len(comp_members[c]) for c in comps)
            for name, frac in zip(split_names, SPLIT_FRACS, strict=True)
        }
        for c in shuffled:
            members = comp_members[c]
            # Split with the largest remaining quota for this class.
            best_split = max(
                split_names,
                key=lambda s: (
                    class_targets[s] - class_totals[s],
                    -split_names.index(s),
                ),
            )
            idx = split_names.index(best_split)
            assign[members] = idx
            class_totals[best_split] += len(members)
            counts[best_split] += len(members)
        per_class[cls] = {
            s: int(class_totals[s]) for s in split_names
        }

    # Per-split actual class counts.
    class_counts = {
        s: dict(Counter(records[i].class_name for i in np.nonzero(assign == k)[0]).most_common())
        for k, s in enumerate(split_names)
    }
    info = {
        "seed": seed,
        "fractions": dict(zip(split_names, SPLIT_FRACS, strict=True)),
        "counts": {s: int(counts[s]) for s in split_names},
        "per_majority_class": per_class,
        "per_split_class_counts": class_counts,
    }
    train = np.nonzero(assign == 0)[0]
    val = np.nonzero(assign == 1)[0]
    test = np.nonzero(assign == 2)[0]
    return train, val, test, info


def random_split(
    n: int, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image-level 70/15/15 split ignoring groups (leakage demo)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(SPLIT_FRACS[0] * n))
    n_val = int(round(SPLIT_FRACS[1] * n))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[
        n_train + n_val :
    ]


def cross_split_pairs(
    left: np.ndarray,
    right: np.ndarray,
    assign: np.ndarray,
) -> int:
    """Count union edges whose endpoints landed in different splits."""
    if len(left) == 0:
        return 0
    return int((assign[left] != assign[right]).sum())
