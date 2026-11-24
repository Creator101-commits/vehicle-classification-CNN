#!/usr/bin/env python3
"""Render image.png from results/ only, in the shared portfolio style.

Run: uv run --with matplotlib python scripts/render_figure.py
Set VEHICLE_RESULTS_DIR to render from a different results directory.
Panels: test macro-F1 per family, row-normalized 8x8 confusion matrices
(custom CNN and pretrained ResNet-18, seed 0), the format-shortcut
check, the component-leakage inflation demo, and per-class F1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

INK = "#1F2A37"
BLUE = "#3B6EA8"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#D1495B"
GRAY = "#9AA5B1"
GRID = "#EEF0F2"
SPINE = "#B8BEC6"
TICK_LABEL = "#4A5361"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SPINE,
        "axes.linewidth": 0.8,
        "xtick.color": SPINE,
        "ytick.color": SPINE,
        "xtick.labelcolor": TICK_LABEL,
        "ytick.labelcolor": TICK_LABEL,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

DPI = 150

def _fmt2(x: float, _pos: int) -> str:
    """Up to 2 decimals inside mathtext; large values as integers."""
    if x == 0:
        return "$0$"
    if abs(x) >= 1000:
        return f"${x:.0f}$"
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "$0$"
    return f"${s}$"


CATEGORICAL: list[tuple[plt.Axes, str, list[str]]] = []


def style(
    ax: plt.Axes,
    letter: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=11, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_visible(True)
        ax.spines[name].set_color(SPINE)
        ax.spines[name].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.6,
        color=SPINE,
        labelcolor=TICK_LABEL,
        labelsize=8.5,
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def cat(
    ax: plt.Axes,
    axis: str,
    positions: list[int] | range,
    labels: list[str],
    rotation: float = 0,
    ha: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Categorical ticks: FixedLocator + FixedFormatter applied AFTER
    style(), so the numeric FuncFormatter cannot clobber the labels.
    Rotation/alignment/fontsize are set on the label Texts directly."""
    labels = list(labels)
    if axis == "x":
        ax.set_xticks(positions)
        ax.xaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_xticklabels()
    else:
        ax.set_yticks(positions)
        ax.yaxis.set_major_locator(mticker.FixedLocator(positions))
        ax.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        texts = ax.get_yticklabels()
    for t in texts:
        if rotation:
            t.set_rotation(rotation)
        if ha:
            t.set_ha(ha)
        if fontsize:
            t.set_fontsize(fontsize)
    CATEGORICAL.append((ax, axis, labels))


def verify_categories() -> None:
    bad = []
    for ax, axis, expected in CATEGORICAL:
        got = [
            t.get_text()
            for t in (ax.get_xticklabels() if axis == "x" else ax.get_yticklabels())
        ]
        if got != expected:
            bad.append((axis, expected, got))
    CATEGORICAL.clear()
    if bad:
        details = "; ".join(f"{a} expected={e} got={g}" for a, e, g in bad)
        raise RuntimeError(f"categorical tick labels mismatch: {details}")


def _all_texts(fig: plt.Figure) -> list[plt.Text]:
    items = []
    for ax in fig.axes:
        if not ax.axison:
            continue
        cand = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cand += list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                ticks = axis._update_ticks()  # noqa: SLF001
            except Exception:
                ticks = []
            for tick in ticks:
                lab = tick.label1
                if lab.get_text():
                    cand.append(lab)
        leg = ax.get_legend()
        if leg is not None:
            cand += list(leg.get_texts())
        for t in cand:
            try:
                if t.get_text() and t.get_visible():
                    items.append(t)
            except Exception:
                pass
    return items


def check_text_overlap(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in _all_texts(fig):
        bb = t.get_window_extent(ren)
        boxes.append((t.get_text()[:40], bb))
    problems = []
    for i, (s1, b1) in enumerate(boxes):
        if b1.x0 < -1 or b1.y0 < -1 or b1.x1 > W + 1 or b1.y1 > H + 1:
            problems.append(f"OUT-OF-FIGURE {s1!r} bbox={b1}")
        for s2, b2 in boxes[i + 1 :]:
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1.5 and iy > 1.5:
                inter = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                if inter > max(8.0, 0.10 * small):
                    problems.append(f"OVERLAP {s1!r} x {s2!r} area={inter:.0f}px2")
    return problems


def check_legend_data(fig: plt.Figure) -> list[str]:
    """Count data markers/vertices falling inside legend boxes."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(ren)
        covered = 0
        total = 0
        for coll in ax.collections:
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 2000:
                continue
            pts = ax.transData.transform(offs)
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(offs)
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = ax.transData.transform(line.get_path().vertices)
            except Exception:
                continue
            inside = (
                (pts[:, 0] > lb.x0)
                & (pts[:, 0] < lb.x1)
                & (pts[:, 1] > lb.y0)
                & (pts[:, 1] < lb.y1)
            ).sum()
            covered += int(inside)
            total += len(pts)
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if pb.x0 < lb.x1 and pb.x1 > lb.x0 and pb.y0 < lb.y1 and pb.y1 > lb.y0:
                covered += 1
            total += 1
        if covered:
            out.append(f"legend covers ~{covered}/{total} data items")
    return out


def check_annotation_data(fig: plt.Figure) -> list[str]:
    """In-axes annotations vs plotted data."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            if not t.get_text() or not t.get_visible():
                continue
            tb = t.get_window_extent(ren)
            hits = 0
            for line in ax.lines:
                if line.get_transform() != ax.transData:
                    continue
                pts = ax.transData.transform(line.get_path().vertices)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            for coll in ax.collections:
                offs = coll.get_offsets()
                if len(offs) == 0 or len(offs) > 5000:
                    continue
                pts = ax.transData.transform(offs)
                hits += int(
                    (
                        (pts[:, 0] > tb.x0)
                        & (pts[:, 0] < tb.x1)
                        & (pts[:, 1] > tb.y0)
                        & (pts[:, 1] < tb.y1)
                    ).sum()
                )
            if hits:
                out.append(f"'{t.get_text()}' overlaps ~{hits} data pts")
    return out


def check_legend_pixels(fig: plt.Figure, path: Path) -> list[str]:
    """Saturated (colored) data pixels showing through legend boxes."""
    if not path.exists():
        return []
    img = plt.imread(path)
    H, W = img.shape[:2]
    rgb = img[..., :3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    colored = spread > 0.18
    def _collect(o: object, swatches: list) -> None:
        if type(o).__name__ == "DrawingArea":
            swatches.append(o)
            return
        try:
            children = o.get_visible_children()
        except Exception:
            children = []
        for c in children:
            _collect(c, swatches)

    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        bb = leg.get_window_extent(fig.canvas.get_renderer())
        x0, x1 = int(bb.x0), min(int(bb.x1) + 1, W)
        y0, y1 = max(0, int(H - bb.y1)), min(H, int(H - bb.y0) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = colored[y0:y1, x0:x1].copy()
        # mask every legend handle swatch (any column) so only data-colored
        # bleed through the legend body counts; swatches live in
        # DrawingArea children of the legend's offset box
        swatches: list = []
        box = getattr(leg, "_legend_handle_box", None)
        if box is not None:
            _collect(box, swatches)
        for sw in swatches:
            try:
                hb = sw.get_window_extent(fig.canvas.get_renderer())
            except Exception:
                continue
            if not all(np.isfinite([hb.x0, hb.x1, hb.y0, hb.y1])):
                continue
            pad = 3
            hx0, hx1 = int(hb.x0) - x0 - pad, int(hb.x1) - x0 + pad
            hy0, hy1 = int(H - hb.y1) - y0 - pad, int(H - hb.y0) - y0 + pad
            rx0, rx1 = max(0, hx0), min(region.shape[1], hx1)
            ry0, ry1 = max(0, hy0), min(region.shape[0], hy1)
            if rx1 > rx0 and ry1 > ry0:
                region[ry0:ry1, rx0:rx1] = False
        frac = region.mean() if region.size else 0.0
        if frac > 0.015:
            out.append(f"legend body ~{frac:.1%} covered by data color")
    return out



def check_empty_axes(fig: plt.Figure) -> list[str]:
    """Every axes must contain at least one plotted artist."""
    fig.canvas.draw()
    fig.canvas.draw()
    out = []
    for i, ax in enumerate(fig.axes):
        if not ax.axison:
            continue
        artists = (len(ax.lines) + len(ax.collections)
                   + len(ax.patches) + len(ax.images))
        if artists == 0:
            out.append(f"axes {i} has no plotted artists")
    return out


def check_offscale(fig: plt.Figure) -> list[str]:
    """Every data point must lie inside its axes limits."""
    fig.canvas.draw()
    fig.canvas.draw()
    out = []
    eps = 1e-9
    for i, ax in enumerate(fig.axes):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for coll in ax.collections:
            if (coll.get_offset_transform() != ax.transData
                    or not isinstance(coll, PathCollection)):
                continue
            offs = coll.get_offsets()
            if len(offs) == 0 or len(offs) > 20000:
                continue
            xs, ys = offs[:, 0], offs[:, 1]
            bad = int(
                ((xs < x0 - eps) | (xs > x1 + eps)
                 | (ys < y0 - eps) | (ys > y1 + eps)).sum()
            )
            if bad:
                out.append(
                    f"axes {i}: {bad} collection points outside limits"
                )
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            try:
                pts = line.get_path().vertices
            except Exception:
                continue
            if len(pts) == 0:
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            bad = int(
                ((xs < x0 - eps) | (xs > x1 + eps)
                 | (ys < y0 - eps) | (ys > y1 + eps)).sum()
            )
            if bad:
                out.append(
                    f"axes {i}: {bad} line vertices outside limits"
                )
        ren = fig.canvas.get_renderer()
        bb = ax.get_window_extent(ren)
        clipped = 0
        for patch in ax.patches:
            try:
                pb = patch.get_window_extent(ren)
            except Exception:
                continue
            if (pb.x0 < bb.x0 - 1 or pb.x1 > bb.x1 + 1
                    or pb.y0 < bb.y0 - 1 or pb.y1 > bb.y1 + 1):
                clipped += 1
        if clipped:
            out.append(f"axes {i}: {clipped} patches past axes bounds")
    return out


def check_clipped_markers(fig: plt.Figure) -> list[str]:
    """No marker may extend past the axes bounds."""
    fig.canvas.draw()
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    out = []
    for i, ax in enumerate(fig.axes):
        bb = ax.get_window_extent(ren)
        px_per_pt = fig.dpi / 72.0
        flagged = False
        for coll in ax.collections:
            if (coll.get_offset_transform() != ax.transData or flagged
                    or not isinstance(coll, PathCollection)):
                continue
            offs = coll.get_offsets()
            if len(offs) == 0:
                continue
            try:
                sizes = np.asarray(coll.get_sizes(), dtype=float)
            except Exception:
                sizes = np.array([])
            if len(sizes) == len(offs) and len(sizes):
                radii_px = np.sqrt(sizes) / 2.0 * px_per_pt
            elif len(sizes):
                radii_px = np.full(
                    len(offs), np.sqrt(sizes.max()) / 2.0 * px_per_pt
                )
            else:
                radii_px = np.full(len(offs), 3.0 * px_per_pt)
            pts = ax.transData.transform(offs)
            hit = (
                (pts[:, 0] - radii_px < bb.x0)
                | (pts[:, 0] + radii_px > bb.x1)
                | (pts[:, 1] - radii_px < bb.y0)
                | (pts[:, 1] + radii_px > bb.y1)
            ).any()
            if hit:
                out.append(f"axes {i}: marker extends past axes bounds")
                flagged = True
        for line in ax.lines:
            if flagged:
                break
            if line.get_transform() != ax.transData:
                continue
            marker = line.get_marker()
            if marker in (None, "", " ", "None", "none", "|", "_"):
                continue
            ms = line.get_markersize()
            radius_px = (ms / 2.0) * px_per_pt
            pts = ax.transData.transform(line.get_path().vertices)
            if len(pts) == 0:
                continue
            hit = (
                (pts[:, 0] - radius_px < bb.x0)
                | (pts[:, 0] + radius_px > bb.x1)
                | (pts[:, 1] - radius_px < bb.y0)
                | (pts[:, 1] + radius_px > bb.y1)
            ).any()
            if hit:
                out.append(f"axes {i}: marker extends past axes bounds")
                flagged = True
    return out


def check_coincident_lines(fig: plt.Figure) -> list[str]:
    """No two line series may be fully coincident."""
    fig.canvas.draw()
    fig.canvas.draw()
    out = []
    for i, ax in enumerate(fig.axes):
        lines = [
            ln for ln in ax.lines
            if ln.get_transform() == ax.transData
        ]
        seen = False
        for a in range(len(lines)):
            if seen:
                break
            for b in range(a + 1, len(lines)):
                pa = lines[a].get_path().vertices
                pb = lines[b].get_path().vertices
                if len(pa) != len(pb) or len(pa) == 0:
                    continue
                if np.allclose(pa, pb, atol=1e-6, equal_nan=True):
                    out.append(
                        f"axes {i}: coincident line series "
                        f"({lines[a].get_label()!r}, "
                        f"{lines[b].get_label()!r})"
                    )
                    seen = True
                    break
    return out


def finish(fig: plt.Figure, out: Path) -> None:
    """Verify categorical labels, save, then fail on any check violation."""
    fig.canvas.draw()
    fig.canvas.draw()
    verify_categories()
    fig.savefig(out, dpi=DPI, facecolor="white")
    problems: list[str] = []
    problems += check_text_overlap(fig)
    problems += check_legend_data(fig)
    problems += check_annotation_data(fig)
    problems += check_legend_pixels(fig, out)
    problems += check_empty_axes(fig)
    problems += check_offscale(fig)
    problems += check_clipped_markers(fig)
    problems += check_coincident_lines(fig)
    plt.close(fig)
    if problems:
        for p in problems[:30]:
            print(f"  CHECK FAIL: {p}", file=sys.stderr)
        sys.exit(f"figure checks failed for {out.name}: {len(problems)} problem(s)")
    print("wrote", out)



ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("VEHICLE_RESULTS_DIR", ROOT / "results"))
OUT = ROOT / "image.png"

FAMILIES = [
    ("custom_A", "Custom A", BLUE),
    ("custom_B", "Custom B", TEAL),
    ("resnet_scratch", "ResNet-18 scratch", AMBER),
    ("resnet_pretrained", "ResNet-18 pretrained", CORAL),
]
SEED_COLORS = [BLUE, AMBER, TEAL]
SEED_MARKERS = ["o", "s", "^"]
CONDS = [
    ("arm_A_test_orig", "A orig"),
    ("arm_A_test_harmonized", "A harm"),
    ("arm_B_test_orig", "B orig"),
    ("arm_B_test_harmonized", "B harm"),
]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#FFFFFF", BLUE, INK]
)


def _confusion(ax: plt.Axes, cm: np.ndarray, letter: str, title: str,
               names: list[str]) -> None:
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    # Wrap the two long class names so rotated tick labels fit the
    # 8-row axis without overlapping.
    disp = [
        n.replace("Motorcycle", "Motor\ncycle")
         .replace("NonVehicles", "Non\nVehicles")
        for n in names
    ]
    pos = np.arange(len(names))
    im = ax.imshow(
        cm / rs, cmap=SEQUENTIAL, aspect="equal", origin="upper",
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    style(ax, letter, title, xlabel="Predicted class",
          ylabel="True class")
    cat(ax, "x", pos, disp, rotation=75, ha="right", fontsize=7)
    cat(ax, "y", pos, disp, rotation=90, fontsize=7)
    return im


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    test = metrics["test"]
    boot = metrics["bootstrap"]
    per_seed = metrics["per_seed"]
    fmt = metrics["format_2x2"]
    infl = metrics["inflation"]
    names = metrics["per_class_names"]

    fig = plt.figure(figsize=(18, 11), dpi=DPI)
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.55, 1])
    gs_fmt = gs[0, 2:4].subgridspec(1, 2, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_nv = fig.add_subplot(gs_fmt[0])
    ax_f1strip = fig.add_subplot(gs_fmt[1])
    ax_bc = fig.add_subplot(gs[1, 0:2])
    ax_bp = fig.add_subplot(gs[1, 2:4])
    ax_i = fig.add_subplot(gs[2, 0:2])
    ax_pc = fig.add_subplot(gs[2, 2:4])

    # (a) test macro-F1 per family: per-seed points + seed-0 bootstrap CI.
    for xi, (key, _label, color) in enumerate(FAMILIES):
        vals = per_seed[key]["macro_f1"]["per_seed"]
        xs = np.full(len(vals), xi) + np.linspace(-0.12, 0.12, len(vals))
        ax_a.scatter(xs, vals, s=26, color=INK, zorder=3,
                     label="seed" if xi == 0 else None)
        ci = boot[key]["macro_f1"]
        ax_a.errorbar(
            [xi], [ci["point"]],
            yerr=[[ci["point"] - ci["ci_low"]],
                  [ci["ci_high"] - ci["point"]]],
            fmt="D", color=color, markersize=7, capsize=4,
            label="seed-0 estimate + 95% cluster-bootstrap CI"
            if xi == 0 else None,
        )
        ax_a.annotate(
            f"{ci['point']:.3f}", (xi, ci["ci_high"] + 0.004),
            ha="center", fontsize=8, color=INK,
        )
    style(ax_a, "a", "Test macro-F1 per family (8-class, grouped test)",
          ylabel="test macro-F1")
    cat(ax_a, "x", range(len(FAMILIES)), [lbl for _, lbl, _ in FAMILIES])
    ax_a.set_ylim(0.74, 0.88)
    ax_a.legend(fontsize=7, loc="lower right", ncol=1)

    # (b) row-normalized confusion matrix, custom CNN (seed 0).
    cm_c = np.array(test["custom_A_s0"]["confusion_matrix"], dtype=float)
    im_b = _confusion(
        ax_bc, cm_c, "b",
        "Confusion matrix, Custom A (seed 0, row-normalized)", names,
    )

    # (c) row-normalized confusion matrix, pretrained ResNet-18 (seed 0).
    cm_p = np.array(
        test["resnet_pretrained_s0"]["confusion_matrix"], dtype=float
    )
    _confusion(
        ax_bp, cm_p, "c",
        "Confusion matrix, ResNet-18 pretrained (seed 0, "
        "row-normalized)",
        names,
    )
    cb = fig.colorbar(im_b, ax=[ax_bc, ax_bp], fraction=0.03, pad=0.02)
    cb.set_label("fraction of true class", fontsize=10, color=INK)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt2))
    cb.ax.tick_params(colors=TICK_LABEL, labelsize=8.5)
    cb.outline.set_edgecolor(SPINE)

    # (d) format-shortcut check: NonVehicles recall strip.
    x = np.arange(len(CONDS))
    ax_nv.set_xticks(x)
    for seed_i in range(3):
        vals = np.array([fmt[c][seed_i]["nonvehicle_recall"]
                         for c, _ in CONDS])
        xs = x + (seed_i - 1) * 0.18
        ax_nv.scatter(
            xs, vals, s=26, marker=SEED_MARKERS[seed_i],
            color=SEED_COLORS[seed_i], edgecolor="white", linewidth=0.4,
            zorder=3,
        )
    style(ax_nv, "d", "NonVehicles recall (format check)")
    cat(ax_nv, "x", x, [lbl for _, lbl in CONDS])
    ax_nv.set_ylim(0.955, 1.005)

    # (e) format-shortcut check: 8-class macro-F1 strip.
    for seed_i in range(3):
        vals = np.array([fmt[c][seed_i]["macro_f1"] for c, _ in CONDS])
        xs = x + (seed_i - 1) * 0.18
        ax_f1strip.scatter(
            xs, vals, s=26, marker=SEED_MARKERS[seed_i],
            color=SEED_COLORS[seed_i], edgecolor="white", linewidth=0.4,
            zorder=3,
        )
    style(ax_f1strip, "e", "8-class macro-F1 (format check)")
    cat(ax_f1strip, "x", x, [lbl for _, lbl in CONDS])
    ax_f1strip.set_ylim(0.72, 0.86)
    ax_nv.legend(
        handles=[
            Line2D([0], [0], marker=SEED_MARKERS[i], color=SEED_COLORS[i],
                   ls="", ms=6, label=f"seed {i}")
            for i in range(3)
        ],
        fontsize=7, loc="lower left", ncol=1,
    )

    # (f) inflation demo: three scenarios x (accuracy, macro-F1).
    groups = [
        ("random /\nrandom test", infl["random_model_random_test"]),
        ("random /\nclean subset", infl["random_model_clean_subset"]),
        ("grouped /\ngrouped test", None),
    ]
    xg = np.arange(len(groups))
    for metric_name, label, off, color in (
        ("accuracy", "accuracy", -0.19, BLUE),
        ("macro_f1", "macro-F1", 0.19, AMBER),
    ):
        vals = []
        for _, block in groups:
            vals.append(
                block[metric_name] if block
                else test["custom_A_s0"][
                    "accuracy" if metric_name == "accuracy" else "macro_f1"
                ]
            )
        ax_i.bar(xg + off, vals, width=0.36, color=color, label=label)
        for xi, v in zip(xg + off, vals, strict=True):
            ax_i.annotate(
                f"{v:.3f}", (xi, v + 0.006), ha="center", fontsize=8,
                color=INK,
            )
    style(ax_i, "f", "Component-leakage inflation demo (single seed)",
          ylabel="score")
    cat(ax_i, "x", xg, [g[0] for g in groups])
    ax_i.set_ylim(0, 1.0)
    ax_i.legend(fontsize=7.5, loc="upper left", ncol=2)

    # (g) per-class F1: custom A seed 0 vs pretrained seed 0.
    xp = np.arange(len(names))
    for off, key, label, color in (
        (-0.19, "custom_A_s0", "Custom A s0", TEAL),
        (0.19, "resnet_pretrained_s0", "Pretrained s0", CORAL),
    ):
        vals = [test[key]["per_class"][c]["f1"] for c in names]
        ax_pc.bar(xp + off, vals, width=0.36, color=color, label=label)
        for xi, v in zip(xp + off, vals, strict=True):
            ax_pc.annotate(
                f"{v:.2f}", (xi, v + 0.015), ha="center", fontsize=7,
                color=INK,
            )
    style(ax_pc, "g", "Per-class F1, seed 0", xlabel="class",
          ylabel="F1")
    cat(ax_pc, "x", xp, names, rotation=45, ha="right", fontsize=8)
    ax_pc.set_ylim(0, 1.05)
    ax_pc.legend(fontsize=7.5, loc="upper right", ncol=1)

    finish(fig, OUT)


if __name__ == "__main__":
    main()
