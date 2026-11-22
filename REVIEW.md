# REVIEW

## Verification log

- `uv run pytest`: pending first full run.
- `uv run ruff check .`: pending.
- `uv run mypy src`: pending.
- Kaggle smoke (`sreeharshak101/vehicle-classification-cnn-smoke`): pending.
- Kaggle full (`sreeharshak101/vehicle-classification-cnn-benchmark`): pending.

## Notes

- Legacy `model.py` (monolithic Colab script, 86% claim, obsolete
  requirements/wiki instructions) was removed; the custom CNN
  architecture lives in `src/vehicle_cnn/models/custom.py`.
- The pHash audit and threshold decision execute inside the Kaggle
  kernel before training; no local audit of the real dataset.
