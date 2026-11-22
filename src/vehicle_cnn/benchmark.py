"""Frozen benchmark orchestration for the vehicle CNN study.

Pipeline: integrity fingerprint -> pHash leakage audit -> grouped
70/15/15 split -> JPEG forensics -> weighted-vs-unweighted choice on
validation -> all training runs -> ``analysis_plan.json`` persisted ->
only then are test images decoded and evaluated once.

``ordering_ok`` requires ``card_write_epoch < first_test_read_epoch``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vehicle_cnn.data import audit as audit_mod
from vehicle_cnn.data import fingerprint as fp_mod
from vehicle_cnn.data import images as img_mod
from vehicle_cnn.evaluation import metrics as met
from vehicle_cnn.models.resnet import (
    NUM_CLASSES,
    build_model,
    pretrained_weight_sha256,
)
from vehicle_cnn.train import TrainConfig, TrainResult, predict_labels, train_model

PROTOCOL = "vehicle-cnn-benchmark-v1"
SEED = 42


@dataclass
class BenchmarkConfig:
    smoke: bool = False
    n_boot: int = 2000
    seed: int = SEED
    smoke_train_cap: int = 4096
    smoke_val_cap: int = 1024
    smoke_test_cap: int = 1024
    smoke_epochs: int = 2
    smoke_n_boot: int = 100
    workers: int = 4


@dataclass
class RunRecord:
    run_id: str
    kind: str
    arm: str  # "A" original formats, "B" harmonized, "random" leakage demo
    seed: int
    weighted: bool
    result: TrainResult
    split: str = "grouped"


def _nonvehicle_class(classes: list[str], records: list[Any]) -> str:
    """Identify the non-vehicle class by name, falling back to PNG share."""
    for c in classes:
        if c.lower().replace("-", "").replace("_", "") in {
            "nonvehicles",
            "nonvehicle",
        }:
            return c
    png_counts = {
        c: sum(1 for r in records if r.class_name == c and r.suffix == ".png")
        for c in classes
    }
    return max(classes, key=lambda c: png_counts[c])


def _labels(records: list[Any], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[r.class_name] for r in records], dtype=np.int64)


def _subset(indices: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=cap, replace=False))


def _run_training(
    cfg: BenchmarkConfig,
    device: torch.device,
    x_train: torch.Tensor,
    y_train: np.ndarray,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    kind: str,
    arm: str,
    split: str,
    seed: int,
    weighted: bool,
    run_id: str,
    reuse: TrainResult | None = None,
) -> RunRecord:
    if reuse is not None:
        return RunRecord(
            run_id=run_id,
            kind=kind,
            arm=arm,
            seed=seed,
            weighted=weighted,
            result=reuse,
            split=split,
        )
    tcfg = TrainConfig(
        kind=kind,
        seed=seed,
        weighted=weighted,
        max_epochs=cfg.smoke_epochs if cfg.smoke else 40,
        patience=8 if not cfg.smoke else 2,
        batch_size=256,
        input_size=224 if kind == "resnet_pretrained" else 64,
    )
    model = build_model(kind, num_classes=NUM_CLASSES)
    result = train_model(model, x_train, y_train, x_val, y_val, tcfg, device)
    return RunRecord(
        run_id=run_id,
        kind=kind,
        arm=arm,
        seed=seed,
        weighted=weighted,
        result=result,
        split=split,
    )


def _eval_on(
    record: RunRecord,
    x_test: torch.Tensor,
    y_test: np.ndarray,
    classes: list[str],
    vehicle_idx: list[int],
    nonveh_idx: int,
    device: torch.device,
    input_size: int,
) -> dict[str, Any]:
    model = build_model(record.kind, num_classes=NUM_CLASSES)
    model.load_state_dict(record.result.state_dict)
    model.to(device)
    preds, probs = predict_labels(
        model, x_test, 512, device, input_size=input_size
    )
    return met.full_metrics(
        y_test, preds, probs, classes, vehicle_idx, nonveh_idx
    )


def run_benchmark(
    input_dir: str | Path,
    work_dir: str | Path,
    fingerprint_path: str | Path | None,
    git_sha: str,
    smoke: bool = False,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run the full frozen benchmark; returns metrics, manifest, plan."""
    cfg = config or BenchmarkConfig(smoke=smoke)
    cfg.smoke = smoke
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.time()

    def mark(name: str, t0: float) -> None:
        timings[name] = round(time.time() - t0, 3)
        print(f"[stage] {name}: {timings[name]:.1f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # --- 1. Locate data, fingerprint, audit, split (CPU, pre-plan). ---
    t0 = time.time()
    root = fp_mod.find_dataset_root(Path(input_dir), work / "extract")
    records = fp_mod.iter_image_records(root)
    fingerprint = fp_mod.compute_fingerprint(records)
    if smoke:
        print("SMOKE: fingerprint emitted, not asserted", flush=True)
    else:
        if fingerprint_path is None:
            raise ValueError("fingerprint_path required outside smoke")
        fingerprint = fp_mod.assert_fingerprint(records, Path(fingerprint_path))
    mark("fingerprint", t0)

    t0 = time.time()
    phashes = audit_mod.compute_phashes(records, workers=cfg.workers)
    pairs = audit_mod.phash_pairs(phashes)
    stats = audit_mod.audit_thresholds(records, pairs)
    threshold = audit_mod.choose_threshold(stats)
    comp = audit_mod.components_for_threshold(records, pairs, threshold)
    train_idx, val_idx, test_idx, split_info = audit_mod.grouped_split(
        records, comp, seed=cfg.seed
    )
    assign = np.full(len(records), -1, dtype=np.int64)
    assign[train_idx], assign[val_idx], assign[test_idx] = 0, 1, 2
    sl, sr = audit_mod._static_edges(records)
    ph_mask = pairs[:, 2] <= threshold
    el = np.concatenate([sl, pairs[ph_mask][:, 0]])
    er = np.concatenate([sr, pairs[ph_mask][:, 1]])
    n_cross = audit_mod.cross_split_pairs(el, er, assign)
    if n_cross != 0:
        raise audit_mod.AuditError(
            f"{n_cross} union edges cross split boundaries at threshold {threshold}"
        )
    mark("audit_split", t0)
    print(
        f"audit: threshold={threshold} images={len(records)} "
        f"pairs<=8={len(pairs)} cross_split_edges={n_cross}",
        flush=True,
    )

    classes = sorted({r.class_name for r in records})
    nonveh_name = _nonvehicle_class(classes, records)
    vehicle_idx = [i for i, c in enumerate(classes) if c != nonveh_name]
    nonveh_idx = classes.index(nonveh_name)
    y_all = _labels(records, classes)

    t0 = time.time()
    vehicle_records = [r for r in records if r.class_name != nonveh_name]
    jpeg_q, jpeg_sub, jpeg_q_counts = img_mod.estimate_jpeg_quality(
        vehicle_records
    )
    mark("jpeg_forensics", t0)
    print(
        f"jpeg: median_q={jpeg_q} subsampling={jpeg_sub} "
        f"quality_hist={jpeg_q_counts}",
        flush=True,
    )

    # --- 2. Decode train/val (orig + harmonized), subset for smoke. ---
    t0 = time.time()
    if smoke:
        train_idx = _subset(train_idx, cfg.smoke_train_cap, cfg.seed)
        val_idx = _subset(val_idx, cfg.smoke_val_cap, cfg.seed)
        test_idx = _subset(test_idx, cfg.smoke_test_cap, cfg.seed)
    tr_records = [records[i] for i in train_idx]
    va_records = [records[i] for i in val_idx]
    x_train = img_mod.decode_records(tr_records, workers=cfg.workers)
    x_val = img_mod.decode_records(va_records, workers=cfg.workers)
    x_train_h = img_mod.decode_records_harmonized(
        tr_records, jpeg_q, jpeg_sub, workers=cfg.workers
    )
    x_val_h = img_mod.decode_records_harmonized(
        va_records, jpeg_q, jpeg_sub, workers=cfg.workers
    )
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    mark("decode_train_val", t0)

    # --- 3. Weighted vs unweighted choice on validation (custom, seed 0). ---
    t0 = time.time()
    choice_runs: dict[bool, RunRecord] = {}
    for weighted in (False, True):
        choice_runs[weighted] = _run_training(
            cfg, device, x_train, y_train, x_val, y_val,
            "custom", "A", "grouped", 0, weighted,
            f"choice_{'w' if weighted else 'u'}_seed0",
        )
    weighted_choice = (
        choice_runs[True].result.best_val_macro_f1
        > choice_runs[False].result.best_val_macro_f1
    )
    mark("weight_choice", t0)
    print(
        f"weight choice: unweighted={choice_runs[False].result.best_val_macro_f1:.4f} "
        f"weighted={choice_runs[True].result.best_val_macro_f1:.4f} "
        f"-> {'weighted' if weighted_choice else 'unweighted'}",
        flush=True,
    )

    # --- 4. All training runs. ---
    t0 = time.time()
    runs: list[RunRecord] = []
    winner = choice_runs[weighted_choice]
    seeds = (0,) if smoke else (0, 1, 2)
    # custom arm A: seed0 reuses the winning choice run.
    for s in seeds:
        runs.append(
            _run_training(
                cfg, device, x_train, y_train, x_val, y_val,
                "custom", "A", "grouped", s, weighted_choice,
                f"custom_A_s{s}",
                reuse=winner.result if s == 0 else None,
            )
        )
    for kind in ("resnet_scratch", "resnet_pretrained"):
        for s in seeds:
            runs.append(
                _run_training(
                    cfg, device, x_train, y_train, x_val, y_val,
                    kind, "A", "grouped", s, weighted_choice,
                    f"{kind}_s{s}",
                )
            )
    # arm B: harmonized formats, same chosen weighting.
    for s in seeds:
        runs.append(
            _run_training(
                cfg, device, x_train_h, y_train, x_val_h, y_val,
                "custom", "B", "grouped", s, weighted_choice,
                f"custom_B_s{s}",
            )
        )
    # inflation demo: random image-level split, custom CNN seed 0.
    r_train, r_val, r_test = audit_mod.random_split(len(records), cfg.seed)
    if smoke:
        r_train = _subset(r_train, cfg.smoke_train_cap, cfg.seed + 1)
        r_val = _subset(r_val, cfg.smoke_val_cap, cfg.seed + 1)
        r_test = _subset(r_test, cfg.smoke_test_cap, cfg.seed + 1)
    rx_train = img_mod.decode_records(
        [records[i] for i in r_train], workers=cfg.workers
    )
    rx_val = img_mod.decode_records(
        [records[i] for i in r_val], workers=cfg.workers
    )
    runs.append(
        _run_training(
            cfg, device, rx_train, y_all[r_train], rx_val, y_all[r_val],
            "custom", "random", "random", 0, weighted_choice,
            "inflation_random_s0",
        )
    )
    mark("training", t0)

    # --- 5. Freeze the analysis plan BEFORE any test decode. ---
    try:
        pretrained_sha = pretrained_weight_sha256()
    except FileNotFoundError:
        pretrained_sha = ""
    plan: dict[str, Any] = {
        "protocol": PROTOCOL,
        "git_sha": git_sha,
        "smoke": smoke,
        "seed": cfg.seed,
        "n_images": len(records),
        "classes": classes,
        "nonvehicle_class": nonveh_name,
        "vehicle_class_indices": vehicle_idx,
        "nonvehicle_class_index": nonveh_idx,
        "fingerprint": fingerprint,
        "audit": {
            "phash_pairs_le_8": int(len(pairs)),
            "threshold": threshold,
            "component_stats": stats,
            "cross_split_edges": n_cross,
        },
        "split": split_info,
        "jpeg": {
            "median_quality": jpeg_q,
            "subsampling": jpeg_sub,
            "quality_counts": jpeg_q_counts,
        },
        "pretrained_weight_sha256": pretrained_sha,
        "weighting": {
            "chosen": "weighted" if weighted_choice else "unweighted",
            "unweighted_val_macro_f1": choice_runs[False].result.best_val_macro_f1,
            "weighted_val_macro_f1": choice_runs[True].result.best_val_macro_f1,
        },
        "runs": [
            {
                "run_id": r.run_id,
                "kind": r.kind,
                "arm": r.arm,
                "seed": r.seed,
                "weighted": r.weighted,
                "split": r.split,
                "best_epoch": r.result.best_epoch,
                "best_val_macro_f1": r.result.best_val_macro_f1,
                "epochs_run": r.result.epochs_run,
                "train_seconds": r.result.train_seconds,
                "input_size": 224 if r.kind == "resnet_pretrained" else 64,
            }
            for r in runs
        ],
        "evaluation": [
            "grouped test orig: full metrics per run",
            "grouped test harmonized: custom arms A and B (2x2)",
            "inflation: random model on random test, clean subset, grouped test",
            "cluster bootstrap over test components, seed-0 per condition",
            "paired bootstrap diffs vs custom_A seed 0",
        ],
        "n_boot": cfg.n_boot,
    }
    plan_path = work / "analysis_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    card_write_epoch = time.time()

    # --- 6. Test decode (first test read happens here). ---
    t0 = time.time()
    te_records = [records[i] for i in test_idx]
    x_test = img_mod.decode_records(te_records, workers=cfg.workers)
    x_test_h = img_mod.decode_records_harmonized(
        te_records, jpeg_q, jpeg_sub, workers=cfg.workers
    )
    y_test = y_all[test_idx]
    first_test_read_epoch = time.time()
    mark("decode_test", t0)

    # Random-split test images for the inflation demo (orig formats).
    t0 = time.time()
    rx_test = img_mod.decode_records(
        [records[i] for i in r_test], workers=cfg.workers
    )
    y_rtest = y_all[r_test]
    mark("decode_random_test", t0)

    # --- 7. Evaluations. ---
    t0 = time.time()
    test_metrics: dict[str, Any] = {}
    run_lookup = {r.run_id: r for r in runs}
    grouped_runs = [r for r in runs if r.split == "grouped"]
    for r in grouped_runs:
        m = _eval_on(
            r, x_test, y_test, classes, vehicle_idx, nonveh_idx,
            device, 224 if r.kind == "resnet_pretrained" else 64,
        )
        test_metrics[r.run_id] = m
    # 2x2 format arms: custom A and B on original and harmonized test.
    format_2x2: dict[str, Any] = {}
    for arm, x_eval, tag in (
        ("A", x_test, "orig"),
        ("A", x_test_h, "harmonized"),
        ("B", x_test, "orig"),
        ("B", x_test_h, "harmonized"),
    ):
        arm_metrics = []
        for r in grouped_runs:
            if r.kind != "custom" or r.arm != arm:
                continue
            m = _eval_on(
                r, x_eval, y_test, classes, vehicle_idx, nonveh_idx,
                device, 64,
            )
            arm_metrics.append(
                {
                    "seed": r.seed,
                    "accuracy": m["accuracy"],
                    "macro_f1": m["macro_f1"],
                    "nonvehicle_recall": m["per_class"][nonveh_name]["recall"],
                }
            )
        format_2x2[f"arm_{arm}_test_{tag}"] = arm_metrics
    mark("eval_grouped", t0)

    # Inflation demo.
    t0 = time.time()
    infl_run = run_lookup["inflation_random_s0"]
    infl_model = build_model("custom", num_classes=NUM_CLASSES)
    infl_model.load_state_dict(infl_run.result.state_dict)
    infl_model.to(device)
    pred_rand, prob_rand = predict_labels(infl_model, rx_test, 512, device, 64)
    m_random_test = met.full_metrics(
        y_rtest, pred_rand, prob_rand, classes, vehicle_idx, nonveh_idx
    )
    # Clean subset: random-test components with no member in random train.
    train_comps = set(comp[r_train].tolist())
    clean_mask = np.array(
        [int(comp[i]) not in train_comps for i in r_test], dtype=bool
    )
    clean_idx_local = np.nonzero(clean_mask)[0]
    if len(clean_idx_local):
        m_clean = met.full_metrics(
            y_rtest[clean_mask],
            pred_rand[clean_mask],
            prob_rand[clean_mask],
            classes,
            vehicle_idx,
            nonveh_idx,
        )
    else:
        m_clean = {"n": 0}
    inflation = {
        "random_model_random_test": {
            "macro_f1": m_random_test["macro_f1"],
            "accuracy": m_random_test["accuracy"],
            "balanced_accuracy": m_random_test["balanced_accuracy"],
            "n": m_random_test["n"],
        },
        "random_model_clean_subset": {
            "macro_f1": m_clean.get("macro_f1"),
            "accuracy": m_clean.get("accuracy"),
            "balanced_accuracy": m_clean.get("balanced_accuracy"),
            "n": int(clean_mask.sum()),
        },
        "grouped_model_grouped_test_macro_f1": test_metrics[
            "custom_A_s0"
        ]["macro_f1"],
        "headline_inflation_macro_f1": m_random_test["macro_f1"]
        - test_metrics["custom_A_s0"]["macro_f1"],
    }
    mark("eval_inflation", t0)

    # Bootstrap: seed-0 model per condition on grouped test (orig).
    t0 = time.time()
    n_boot = cfg.smoke_n_boot if smoke else cfg.n_boot
    comp_test = comp[test_idx]
    conditions = {
        "custom_A": "custom_A_s0",
        "resnet_scratch": "resnet_scratch_s0",
        "resnet_pretrained": "resnet_pretrained_s0",
        "custom_B": "custom_B_s0",
    }
    boot: dict[str, Any] = {}
    seed0_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond, run_id in conditions.items():
        r = run_lookup[run_id]
        model = build_model(r.kind, num_classes=NUM_CLASSES)
        model.load_state_dict(r.result.state_dict)
        model.to(device)
        preds, probs = predict_labels(
            model, x_test, 512, device,
            224 if r.kind == "resnet_pretrained" else 64,
        )
        seed0_preds[cond] = (preds, probs)
        boot[cond] = {
            m: met.cluster_bootstrap(
                comp_test, y_test, preds, classes, metric=m,
                n_boot=n_boot, seed=0,
            )
            for m in ("accuracy", "balanced_accuracy", "macro_f1")
        }
    paired: dict[str, Any] = {}
    base = seed0_preds["custom_A"]
    for cond in ("resnet_scratch", "resnet_pretrained", "custom_B"):
        other = seed0_preds[cond]
        paired[cond] = {
            m: met.paired_cluster_bootstrap(
                comp_test, y_test, base[0], y_test, other[0], classes,
                metric=m, n_boot=n_boot, seed=0,
            )
            for m in ("accuracy", "macro_f1")
        }
    boot["paired_diffs_vs_custom_A"] = paired
    mark("bootstrap", t0)

    # Per-seed aggregation per condition on grouped test orig.
    per_seed: dict[str, Any] = {}
    for cond in ("custom_A", "resnet_scratch", "resnet_pretrained", "custom_B"):
        if cond == "custom_A":
            members = [
                r for r in grouped_runs
                if r.kind == "custom" and r.arm == "A"
            ]
        elif cond == "custom_B":
            members = [
                r for r in grouped_runs
                if r.kind == "custom" and r.arm == "B"
            ]
        else:
            members = [r for r in grouped_runs if r.kind == cond]
        vals = {
            m: [test_metrics[r.run_id][m] for r in members]
            for m in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        per_seed[cond] = {
            m: {
                "per_seed": v,
                "mean": float(np.mean(v)),
                "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for m, v in vals.items()
        }

    gpu_seconds = float(
        sum(r.result.train_seconds for r in runs)
    )
    manifest = {
        "protocol": PROTOCOL,
        "git_sha": git_sha,
        "smoke": smoke,
        "ordering_ok": bool(card_write_epoch < first_test_read_epoch),
        "card_write_epoch": card_write_epoch,
        "first_test_read_epoch": first_test_read_epoch,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gpu_seconds_train": gpu_seconds,
        "timings_s": timings,
        "total_seconds": round(time.time() - t_all, 3),
        "n_images": len(records),
        "classes": classes,
        "nonvehicle_class": nonveh_name,
        "threshold": threshold,
        "split_counts": split_info["counts"],
        "weighting": plan["weighting"]["chosen"],
        "jpeg_quality": jpeg_q,
        "jpeg_subsampling": jpeg_sub,
        "pretrained_weight_sha256": pretrained_sha,
        "n_boot": n_boot,
        "seed": cfg.seed,
    }

    metrics = {
        "protocol": PROTOCOL,
        "git_sha": git_sha,
        "smoke": smoke,
        "test": test_metrics,
        "format_2x2": format_2x2,
        "inflation": inflation,
        "bootstrap": boot,
        "per_seed": per_seed,
        "per_class_names": classes,
    }
    return {
        "metrics": metrics,
        "manifest": manifest,
        "plan": plan,
        "audit_stats": stats,
        "fingerprint": fingerprint,
    }


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    """Write a compact markdown report from the run artifacts."""
    m = result["metrics"]
    man = result["manifest"]
    lines = [
        f"# Vehicle CNN benchmark ({'smoke' if man['smoke'] else 'full'})",
        "",
        f"- git_sha: `{man['git_sha']}`",
        f"- ordering_ok: {man['ordering_ok']}",
        f"- images: {man['n_images']}, classes: {', '.join(man['classes'])}",
        f"- pHash threshold: {man['threshold']}",
        f"- weighting: {man['weighting']}",
        f"- jpeg quality: {man['jpeg_quality']}, subsampling: {man['jpeg_subsampling']}",
        f"- split counts: {man['split_counts']}",
        "",
        "## Test metrics (grouped, original formats)",
        "",
        "| run | acc | bal-acc | macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run_id, tm in m["test"].items():
        lines.append(
            f"| {run_id} | {tm['accuracy']:.4f} | "
            f"{tm['balanced_accuracy']:.4f} | {tm['macro_f1']:.4f} |"
        )
    lines += ["", "## Format 2x2 (custom arms)", ""]
    for key, rows in m["format_2x2"].items():
        for row in rows:
            lines.append(
                f"- {key} seed {row['seed']}: acc={row['accuracy']:.4f} "
                f"macro-F1={row['macro_f1']:.4f} "
                f"NonVehicles recall={row['nonvehicle_recall']:.4f}"
            )
    lines += ["", "## Inflation demo", ""]
    for k, v in m["inflation"].items():
        lines.append(f"- {k}: {v}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
