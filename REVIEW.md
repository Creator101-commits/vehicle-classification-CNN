# REVIEW

## Verification log

- `uv run pytest`: 24 passed.
- `uv run ruff check .`: clean.
- `uv run mypy src`: clean (12 source files).

## Kaggle runs

- Smoke v1 (`vehicle-classification-cnn-smoke` v1): ERROR at dataset
  discovery. Kaggle mounts datasets under
  `/kaggle/input/datasets/<owner>/<slug>/`; the runner's fallback
  returned the parent mount dir and `find_dataset_root` found no
  class subdirectories. Fixed in `f690029`
  (`locate_input_dataset`, no arbitrary-directory fallback).
- Smoke v2 (`vehicle-classification-cnn-smoke` v2 at `f690029`):
  COMPLETE. Diagnostic only; printed test metrics from toy models
  (one epoch). No configuration was chosen based on those scores.
- Full (`vehicle-classification-cnn-benchmark` v1 at `85fcf2a`):
  COMPLETE, 6,513 s on a Tesla T4 (training 4,985 s). The `85fcf2a`
  delta over smoke `f690029` is `data/fingerprint.json` only.
  Verified gates: fingerprint asserted, pHash threshold 6 chosen
  (largest component 28 images, 0.11%), zero cross-split union
  edges, `ordering_ok`, 2,000 bootstrap resamples, manifest
  `git_sha` matches HEAD.

## Notes

- Legacy `model.py` (monolithic Colab script, 86% claim, obsolete
  requirements/wiki instructions) was removed; the custom CNN
  architecture lives in `src/vehicle_cnn/models/custom.py`.
- The pHash audit and threshold decision execute inside the Kaggle
  kernel before training; no local audit of the real dataset.
- `results/` is the unedited kernel output; `image.png` renders from
  `results/` only.
- Dataset source and license are unknown; the NonVehicles origin is
  a hypothesis.
