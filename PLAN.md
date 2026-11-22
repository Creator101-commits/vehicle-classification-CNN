# PLAN

## Status

- [x] Package scaffold (`src/vehicle_cnn`), pyproject/uv lock, CI
- [x] Data layer: fingerprint, pHash audit, grouped split, JPEG forensics
- [x] Models: custom CNN (legacy architecture), ResNet-18 scratch/pretrained
- [x] Training loop: AdamW, cosine, warmup, label smoothing, AMP, early stop
- [x] Evaluation: 8-class + 7-vehicle metrics, cluster bootstrap, inflation demo
- [x] Benchmark orchestration with frozen `analysis_plan.json`
- [x] Kaggle runner, smoke + full kernels, private image dataset upload
- [ ] Smoke kernel run and audit/projection report
- [ ] Full kernel run (only if projection <= 3 GPU hours)
- [ ] `data/fingerprint.json` committed from smoke output
- [ ] `results/` curation, `image.png`, README findings (after lead review)

## Open questions

- pHash threshold outcome on the real data (chosen inside the kernel).
- Weighted vs unweighted loss choice (validation macro-F1, seed 0).
- Full-run GPU projection from smoke timings.
