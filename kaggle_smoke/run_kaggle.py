"""Smoke kernel: reduced-work diagnostic run of the vehicle CNN benchmark.

Reuses the archived main kernel script from the source dataset with
``VEHICLE_SMOKE=1`` so the full pipeline (fingerprint, pHash audit,
grouped split, JPEG forensics, weight choice, one seed per model kind,
frozen plan, test read) is exercised cheaply before the full run.
Smoke outputs are diagnostics only and are never committed as results.
"""

import os
import runpy
import sys
from pathlib import Path

os.environ["VEHICLE_SMOKE"] = "1"

candidates = [
    p
    for p in Path("/kaggle/input").rglob("run_kaggle.py")
    if p.parent.name == "kaggle"
]
if not candidates:
    raise SystemExit("main kernel script not found in the source dataset")
target = sorted(candidates)[0]
print(f"smoke: delegating to {target}", flush=True)
sys.argv = [str(target)]
runpy.run_path(str(target), run_name="__main__")
