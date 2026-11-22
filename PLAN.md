# PLAN

## Status

- [x] Package scaffold (`src/vehicle_cnn`), pyproject/uv lock, CI
- [x] Data layer: fingerprint, pHash audit, grouped split, JPEG forensics
- [x] Models: custom CNN (legacy architecture), ResNet-18 scratch/pretrained
- [x] Training loop: AdamW, cosine, warmup, label smoothing, AMP, early stop
- [x] Evaluation: 8-class + 7-vehicle metrics, cluster bootstrap, inflation demo
- [x] Benchmark orchestration with frozen `analysis_plan.json`
- [x] Kaggle runner, smoke + full kernels, private image dataset upload
- [x] Smoke kernel run (v1 dataset-discovery failure fixed in f690029; v2 complete)
- [x] Full kernel run at 85fcf2a (6,513 s T4, under the 3 GPU-hour gate)
- [x] `data/fingerprint.json` committed from smoke output
- [x] `results/` curation, `image.png`, README findings

## Open questions

- pHash threshold outcome on the real data: resolved, threshold 6
  (largest component 28 images, 0.11%).
- Weighted vs unweighted loss choice: resolved, unweighted
  (validation macro-F1 0.792 vs 0.776 at seed 0).
- Full-run GPU projection from smoke timings: resolved, about
  1.36 GPU-hours projected; actual 1.81 h wall on T4.
