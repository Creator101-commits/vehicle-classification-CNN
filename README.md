# Vehicle Classification CNN

Reproducible 8-class vehicle image classification benchmark: a custom
CNN (ported from the legacy Colab script), ResNet-18 from scratch, and
pretrained ResNet-18, all trained under a frozen leakage-controlled
protocol on Kaggle.

## Dataset

`vehicle_classification.zip` (sha256
`acddb78fe1865dc053d77e5f553c5773c333bf17d7cb6b529f3ff519ffa9d1d9`),
uploaded as the private Kaggle dataset
`sreeharshak101/vehicle-classification-images`. Eight classes:
Bicycle, Bus, Car, Motorbike, SUV, Truck, Van, and NonVehicles.

## Protocol (`vehicle-cnn-benchmark-v1`)

1. **Fingerprint.** SHA-256 over sorted `relpath\tfile_sha256` lines,
   per-class counts, and the original zip digest. Committed as
   `data/fingerprint.json` after smoke; asserted in the full run.
2. **Leakage audit.** pHash (imagehash, 8x8 DCT) near-duplicates at
   thresholds 0/2/4/6/8; the chosen threshold is the largest whose
   largest connected component is at most 5% of images. Components are
   grouped by union-find; the 70/15/15 train/val/test split is atomic
   over components, so zero union edges cross splits (asserted).
3. **JPEG forensics.** Median IJG quality and modal chroma subsampling
   estimated from luminance quantization tables; PNGs in the
   harmonized arm are re-encoded to JPEG at that quality/subsampling.
4. **Weight choice.** The custom CNN at seed 0 trains once unweighted
   and once class-weighted; the winner by validation macro-F1 sets the
   weighting for every model.
5. **Training.** All models: AdamW, batch 256, lr 1e-3 scratch /
   3e-4 pretrained, weight decay 0.05, cosine schedule, 1 warmup epoch,
   label smoothing 0.1, AMP fp16, early stop on validation macro-F1
   (patience 8, or 4 pretrained), max 40 epochs (15 pretrained). Seeds
   0/1/2. Images are decoded once to uint8 tensors (~324 MB) and
   augmented on GPU.
6. **Freeze.** `analysis_plan.json` records the threshold, split
   counts, weighting, per-run best epochs, and the evaluation list.
   Only then are test images decoded; `ordering_ok` requires
   `card_write_epoch < first_test_read_epoch`.
7. **Evaluation.** Grouped test: 8-class metrics plus 7-vehicle-class
   metrics and the NonVehicles argmax rate. A harmonized-format arm
   tests the JPEG/PNG shortcut. A random-split custom CNN demonstrates
   the inflation from component leakage, evaluated three ways: on its
   own random test, on the clean subset of its random test whose
   components never appear in its random train, and the grouped model
   on grouped test. Uncertainty is a paired cluster bootstrap over
   test components (2000 resamples, seed 0) on seed-0 models, plus
   per-seed mean/sd/range.

## Results

Pending the Kaggle run; `results/` will hold the unedited kernel
output and `image.png` renders from it.

## Reproduce

```bash
uv sync
uv run pytest          # synthetic fixtures only, no real data
scripts/kaggle_run.sh --smoke   # diagnostic kernel
scripts/kaggle_run.sh           # full GPU kernel
```

Local tests never touch the real dataset: audit, hashing, and training
run only inside Kaggle.
