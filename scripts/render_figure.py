#!/usr/bin/env python3
"""Render image.png from results/ only.

Run: uv run --with matplotlib python scripts/render_figure.py
Set VEHICLE_RESULTS_DIR to render from a different results directory.
Writes scripts/figure-data.json alongside the PNG.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("VEHICLE_RESULTS_DIR", ROOT / "results"))
OUT = ROOT / "image.png"
DATA_OUT = ROOT / "scripts" / "figure-data.json"

FAMILIES = [
    ("custom_A", "Custom A"),
    ("custom_B", "Custom B (harmonized)"),
    ("resnet_scratch", "ResNet-18 scratch"),
    ("resnet_pretrained", "ResNet-18 pretrained"),
]
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def main() -> None:
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    test = metrics["test"]
    boot = metrics["bootstrap"]
    per_seed = metrics["per_seed"]
    fmt = metrics["format_2x2"]
    infl = metrics["inflation"]
    names = metrics["per_class_names"]
    manifest = json.loads((RESULTS / "manifest.json").read_text())

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)
    gs_b = gs[0, 1].subgridspec(1, 2, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs_b[0])
    ax_f = fig.add_subplot(gs_b[1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle(
        f"vehicle-cnn-benchmark-v1, git {manifest['git_sha'][:7]}, "
        f"grouped split, unweighted",
        fontsize=12,
    )

    # (a) test macro-F1 per family: per-seed points + seed-0 bootstrap CI.
    ax = ax_a
    for xi, (key, _label) in enumerate(FAMILIES):
        vals = per_seed[key]["macro_f1"]["per_seed"]
        xs = np.full(len(vals), xi) + np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(xs, vals, s=28, color="#333", zorder=3,
                   label="seed" if xi == 0 else None)
        ci = boot[key]["macro_f1"]
        ax.errorbar(
            [xi], [ci["point"]],
            yerr=[[ci["point"] - ci["ci_low"]], [ci["ci_high"] - ci["point"]]],
            fmt="D", color="#c0392b", markersize=7, capsize=5,
            label="seed-0 estimate + 95% cluster-bootstrap CI" if xi == 0 else None,
        )
        ax.annotate(
            f"{ci['point']:.3f}", (xi, ci["ci_high"] + 0.004),
            ha="center", fontsize=8, color="#c0392b",
        )
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([lbl for _, lbl in FAMILIES], fontsize=8, rotation=15)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Test macro-F1 per family (8-class, grouped test)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.74, 0.88)

    # (b) format-shortcut check: two side-by-side strip plots, one per
    # metric, per-seed points for each of the 4 test conditions.
    conds = [
        ("arm_A_test_orig", "A orig"),
        ("arm_A_test_harmonized", "A harm"),
        ("arm_B_test_orig", "B orig"),
        ("arm_B_test_harmonized", "B harm"),
    ]
    x = np.arange(len(conds))
    fig.text(
        0.75, 0.945,
        "Format-shortcut check: original (PNG NonVehicles) vs "
        "harmonized (JPEG q75)",
        ha="center", fontsize=9,
    )
    markers = ["o", "s", "^"]
    for axi, metric_name, title, ylim in (
        (ax_r, "nonvehicle_recall", "NonVehicles recall", (0.96, 1.001)),
        (ax_f, "macro_f1", "8-class macro-F1", (0.74, 0.83)),
    ):
        axi.set_title(title, fontsize=9)
        axi.set_xticks(x)
        axi.set_xticklabels([lbl for _, lbl in conds], fontsize=8,
                            rotation=15)
        axi.set_ylim(*ylim)
        for seed_i in range(3):
            vals = np.array([fmt[c][seed_i][metric_name] for c, _ in conds])
            xs = x + (seed_i - 1) * 0.18
            axi.scatter(
                xs, vals, s=30, marker=markers[seed_i],
                color=SEED_COLORS[seed_i], edgecolor="black", linewidth=0.4,
                zorder=3, label=f"seed {seed_i}" if axi is ax_r else None,
            )
        axi.grid(axis="y", alpha=0.3)
    ax_r.legend(fontsize=7, loc="lower left")

    # (c) inflation demo: three scenarios x (accuracy, macro-F1).
    ax = ax_c
    groups = [
        ("random / random test", infl["random_model_random_test"], "#7f8c8d"),
        ("random / clean subset", infl["random_model_clean_subset"], "#95a5a6"),
        ("grouped / grouped test", None, "#16a085"),
    ]
    x = np.arange(len(groups))
    for metric_name, label, off, color in (
        ("accuracy", "accuracy", -0.19, "#3498db"),
        ("macro_f1", "macro-F1", 0.19, "#e67e22"),
    ):
        vals = []
        for _, block, _c in groups:
            vals.append(
                block[metric_name] if block
                else test["custom_A_s0"]["accuracy" if metric_name == "accuracy" else "macro_f1"]
            )
        ax.bar(x + off, vals, width=0.36, color=color, label=label)
        for xi, v in zip(x + off, vals, strict=True):
            ax.annotate(f"{v:.3f}", (xi, v + 0.006), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Component-leakage inflation demo (single seed)")
    ax.legend(fontsize=8)

    # (d) per-class F1: custom A seed 0 vs pretrained seed 0.
    ax = ax_d
    x = np.arange(len(names))
    for off, key, label, color in (
        (-0.19, "custom_A_s0", "Custom A s0", "#16a085"),
        (0.19, "resnet_pretrained_s0", "Pretrained s0", "#8e44ad"),
    ):
        vals = [test[key]["per_class"][c]["f1"] for c in names]
        ax.bar(x + off, vals, width=0.36, color=color, label=label)
        for xi, v in zip(x + off, vals, strict=True):
            ax.annotate(f"{v:.2f}", (xi, v + 0.015), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1, seed 0")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=160)
    DATA_OUT.write_text(
        json.dumps(
            {
                "source": "results/metrics.json + results/manifest.json",
                "panel_a": {
                    k: {
                        "per_seed_f1": per_seed[k]["macro_f1"]["per_seed"],
                        "seed0_ci": [
                            boot[k]["macro_f1"]["ci_low"],
                            boot[k]["macro_f1"]["ci_high"],
                        ],
                    }
                    for k, _ in FAMILIES
                },
                "panel_b": fmt,
                "panel_c": infl,
                "panel_d": {
                    "classes": names,
                    "custom_A_s0": {
                        c: test["custom_A_s0"]["per_class"][c]["f1"] for c in names
                    },
                    "pretrained_s0": {
                        c: test["resnet_pretrained_s0"]["per_class"][c]["f1"]
                        for c in names
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT} and {DATA_OUT}")


if __name__ == "__main__":
    main()
