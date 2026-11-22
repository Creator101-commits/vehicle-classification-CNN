"""End-to-end benchmark ordering test on a tiny synthetic tree.

Exercises run_benchmark in smoke mode (small caps, 1-2 epochs) and asserts
the frozen-plan ordering invariant and manifest fields. Training runs are
kept minimal by BenchmarkConfig overrides.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vehicle_cnn.benchmark import BenchmarkConfig, run_benchmark


@pytest.fixture()
def tiny_bench_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        smoke=True,
        smoke_train_cap=96,
        smoke_val_cap=48,
        smoke_test_cap=48,
        smoke_epochs=1,
        smoke_n_boot=10,
        workers=1,
    )


def test_run_benchmark_smoke_ordering(tmp_path: Path, tiny_dataset: Path, tiny_bench_config):
    work = tmp_path / "work"
    result = run_benchmark(
        tiny_dataset,
        work,
        fingerprint_path=None,
        git_sha="testsha",
        smoke=True,
        config=tiny_bench_config,
    )
    man = result["manifest"]
    assert man["ordering_ok"] is True
    assert man["card_write_epoch"] < man["first_test_read_epoch"]
    assert man["protocol"] == "vehicle-cnn-benchmark-v1"
    assert man["n_images"] == 50
    assert (work / "analysis_plan.json").exists()
    # Plan was written before test decode; plan lists runs.
    plan = json.loads((work / "analysis_plan.json").read_text())
    assert plan["git_sha"] == "testsha"
    assert plan["audit"]["threshold"] in (0, 2, 4, 6)
    assert len(plan["runs"]) >= 4
    # Test metrics exist for grouped runs.
    assert "custom_A_s0" in result["metrics"]["test"]
    assert "inflation" in result["metrics"]
