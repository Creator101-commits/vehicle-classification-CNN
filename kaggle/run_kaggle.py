"""Kaggle entrypoint for the vehicle classification CNN benchmark (GPU).

Installs ``vehicle_cnn`` from the pinned source dataset uploaded by
``scripts/kaggle_run.sh`` with ``--no-deps`` (the Kaggle image already
ships torch/torchvision; only missing pure-Python deps are installed),
locates the private image dataset under ``/kaggle/input``, runs the
frozen benchmark, and writes ``benchmark/`` artifacts including
``manifest.json`` with the git SHA. ``VEHICLE_SMOKE=1`` caps data and
epochs for a diagnostic run whose outputs are never committed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
BENCH = WORK / "benchmark"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _find_project_root() -> Path | None:
    for candidate in sorted(KAGGLE_INPUT.rglob("pyproject.toml")):
        if (candidate.parent / "src" / "vehicle_cnn").is_dir():
            return candidate.parent
    return None


def _install_package() -> Path:
    proj_root = _find_project_root()
    print(f"project root: {proj_root}", flush=True)
    if proj_root is None:
        raise FileNotFoundError(
            "No installable project found under /kaggle/input "
            "(expected pyproject.toml + src/vehicle_cnn)."
        )
    proj = WORK / "project"
    shutil.copytree(proj_root, proj)
    _run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", str(proj)])
    # Deps missing from the Kaggle image; torch/torchvision/numpy/PIL and
    # scikit-learn are preinstalled, imagehash is not guaranteed.
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "imagehash>=4.3",
        ]
    )
    return proj


def _git_sha(proj: Path) -> str:
    info = proj / "run_info.json"
    if info.exists():
        return str(json.loads(info.read_text()).get("git_sha", "unknown"))
    return "unknown"


def _find_dataset_dir(proj: Path) -> Path:
    """Locate the mounted image dataset directory or zip."""
    candidates = [
        p
        for p in sorted(KAGGLE_INPUT.iterdir())
        if p.is_dir() and "vehicle-classification-images" in p.name
    ]
    if candidates:
        return candidates[0]
    zips = [
        p
        for p in KAGGLE_INPUT.rglob("*.zip")
        if "vehicle_classification" in p.name
    ]
    if zips:
        return zips[0].parent
    # Fallback: any input dir containing class-like subdirectories.
    for child in sorted(KAGGLE_INPUT.iterdir()):
        if child.is_dir() and child.name != proj.parent.name:
            return child
    raise FileNotFoundError(f"image dataset not found under {KAGGLE_INPUT}")


def _report_projection(result: dict) -> None:
    """Print smoke timing projection for the full run."""
    man = result["manifest"]
    timings = man.get("timings_s", {})
    print("=== SMOKE TIMINGS ===", flush=True)
    for k, v in timings.items():
        print(f"{k}: {v:.1f}s", flush=True)
    proj = man.get("smoke_projection")
    if proj:
        print("=== FULL-RUN GPU PROJECTION ===", flush=True)
        print(f"train scale: {proj['train_scale']:.2f}x", flush=True)
        for kind, p in proj["per_kind"].items():
            print(
                f"{kind}: epoch~{p['full_epoch_s_est']:.1f}s "
                f"x{p['planned_runs']} runs x{p['planned_epochs']} ep "
                f"= {p['gpu_hours']:.2f} GPU-h",
                flush=True,
            )
        print(f"TOTAL projected GPU-hours: {proj['total_gpu_hours']:.2f}", flush=True)


def main() -> None:
    proj = _install_package()
    sha = _git_sha(proj)
    print(f"git_sha={sha}", flush=True)
    import torch  # noqa: PLC0415

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    from vehicle_cnn.benchmark import BenchmarkConfig, run_benchmark  # noqa: PLC0415

    data_dir = _find_dataset_dir(proj)
    print(f"data dir: {data_dir}", flush=True)
    smoke = bool(os.environ.get("VEHICLE_SMOKE"))
    fp_path = proj / "data" / "fingerprint.json"
    print(f"smoke={smoke} fingerprint={fp_path}", flush=True)

    BENCH.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = run_benchmark(
        data_dir,
        BENCH,
        fingerprint_path=None if smoke else fp_path,
        git_sha=sha,
        smoke=smoke,
        config=BenchmarkConfig(smoke=smoke),
    )

    metrics = result["metrics"]
    manifest = result["manifest"]
    plan = result["plan"]
    (BENCH / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (BENCH / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (BENCH / "analysis_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n"
    )
    from vehicle_cnn.benchmark import write_report  # noqa: PLC0415

    write_report(BENCH, result)

    print("=== FROZEN ===", flush=True)
    print(
        json.dumps(
            {
                "ordering_ok": manifest["ordering_ok"],
                "n_images": manifest["n_images"],
                "threshold": manifest["threshold"],
                "weighting": manifest["weighting"],
                "smoke": smoke,
            },
            indent=2,
        ),
        flush=True,
    )
    for run_id, tm in metrics["test"].items():
        print(
            f"test {run_id}: acc={tm['accuracy']:.4f} "
            f"macro_f1={tm['macro_f1']:.4f}",
            flush=True,
        )
    if smoke:
        _report_projection(result)
    print(f"wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
