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
   thresholds 0/2/4/6/8 for reporting; the selectable candidates are
   6, 4, 2, 0 (8 is stats-only), and the chosen threshold is the
   largest whose largest connected component is at most 5% of images.
   Components are grouped by union-find; the 70/15/15 train/val/test
   split is atomic over components, so zero union edges cross splits
   (asserted).
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

Full kernel run `sreeharshak101/vehicle-classification-cnn-benchmark`
version 1 at git `85fcf2a` (Tesla T4, 6,513 s wall, of which training
was 4,985 s). `results/` holds the unedited kernel output
(`analysis_plan.json`, `manifest.json`, `metrics.json`, `report.md`);
`image.png` renders from `results/` only via
`uv run --with matplotlib python scripts/render_figure.py`.

![Vehicle classification benchmark results: test macro-F1 per family, confusion matrices, format-shortcut check, inflation demo and per-class F1](image.png)

### What the figure shows

- (a) Test macro-F1 per family: pretrained ResNet-18 leads at
  0.851-0.862 across seeds, ahead of scratch ResNet-18
  (0.806-0.820) and both custom CNNs (about 0.79); the seed-0
  cluster-bootstrap CI for pretrained is [0.835, 0.865].
- (b) Row-normalized confusion matrix, Custom A (seed 0):
  NonVehicles recall is a perfect 1.00, but Taxi (0.50), Van (0.58)
  and Truck (0.68) are the weak classes, with Truck most often
  confused into Car (59 cases).
- (c) Row-normalized confusion matrix, ResNet-18 pretrained (seed
  0): the same weak classes improve to Taxi 0.65, Van 0.72 and Truck
  0.77, and NonVehicles stays perfect; the top remaining confusion is
  Car into Truck (48 cases).
- (d) Format-shortcut check, NonVehicles recall: every seed and both
  image formats stay at 0.992-1.000, so harmonization does not hurt
  the dominant class.
- (e) Format-shortcut check, 8-class macro-F1: original and
  harmonized test sets agree within about 0.001 per seed, ruling out
  a JPEG-artifact shortcut.
- (f) Component-leakage inflation demo: the custom CNN trained on
  the ungrouped random split scores accuracy 0.864 and macro-F1 0.771
  on its own test versus 0.881 and 0.796 on the grouped test; on the
  clean subset it reaches 0.949 accuracy but only 0.754 macro-F1, so
  the grouped numbers are not inflated by leakage.
- (g) Per-class F1, seed 0: pretrained ResNet-18 beats Custom A on
  all eight classes, with the largest gains on Truck (+0.095), Van
  (+0.088) and Taxi (+0.082).

Pretrained ResNet-18 is the clear winner on grouped test, the
confusion matrices localize the remaining errors to Taxi, Van and
Truck for both CNNs, and the format and leakage checks show the
benchmark numbers are not inflated by shortcuts.

### Run history and gates

- Smoke v1 failed at dataset discovery: Kaggle mounts datasets under
  `/kaggle/input/datasets/<owner>/<slug>/` and the runner fell back to
  an arbitrary input directory. Fixed in `f690029`.
- Smoke v2 ran at `f690029`; the full run ran at `85fcf2a`, whose only
  delta is the committed `data/fingerprint.json`.
- Gates verified in the full run: fingerprint asserted, pHash
  threshold chosen, zero cross-split union edges, `ordering_ok`
  (`analysis_plan.json` written before the first test decode),
  2,000 bootstrap resamples.
- The smoke kernels printed test metrics from toy models (one epoch);
  no configuration was chosen based on those scores.

### Data facts

- 26,378 images, 8 classes. The original zip sha256
  `acddb78f...d1d9` was computed on the owner's local archive and is a
  recorded constant; Kaggle auto-extracts uploads, so inside the
  kernel the fingerprint asserts the aggregate over the extracted
  files (`515fff84...`).
- pHash audit: the chosen threshold is 6; its largest connected
  component is 28 images (0.11% of the dataset). Zero union edges
  cross the grouped split.
- JPEG forensics: all 17,410 vehicle JPEGs estimate IJG quality 75
  with 4:2:0 chroma subsampling.
- Grouped split (seed 42, atomic over pHash components):
  train 18,465 / validation 3,958 / test 3,955. The `random_split`
  used only for the inflation demo is a plain permutation, not
  class-stratified; the grouped split is stratified by component
  majority class. Per-majority-class counts are in
  `results/analysis_plan.json`.
- Weighting: unweighted was chosen (validation macro-F1 0.792 vs
  0.776 class-weighted at seed 0) and applied to every model.

### Test metrics (grouped test, n = 3,955)

| family | seed | acc | bal-acc | macro-F1 | ECE | 7-way acc | 7-way F1 | veh->NV |
|---|---|---|---|---|---|---|---|---|
| Custom A | 0 | 0.8809 | 0.7772 | 0.7961 | 0.0643 | 0.8205 | 0.7673 | 0.00077 |
| Custom A | 1 | 0.8731 | 0.7731 | 0.7870 | 0.0653 | 0.8113 | 0.7581 | 0.00000 |
| Custom A | 2 | 0.8756 | 0.7851 | 0.7961 | 0.1218 | 0.8136 | 0.7679 | 0.00000 |
| Custom B (harmonized) | 0 | 0.8771 | 0.7704 | 0.7884 | 0.1015 | 0.8155 | 0.7589 | 0.00000 |
| Custom B (harmonized) | 1 | 0.8733 | 0.7769 | 0.7914 | 0.0677 | 0.8101 | 0.7625 | 0.00000 |
| Custom B (harmonized) | 2 | 0.8771 | 0.7726 | 0.7911 | 0.0629 | 0.8155 | 0.7619 | 0.00038 |
| ResNet-18 scratch | 0 | 0.8900 | 0.7965 | 0.8069 | 0.0510 | 0.8343 | 0.7798 | 0.00000 |
| ResNet-18 scratch | 1 | 0.8857 | 0.7925 | 0.8063 | 0.0533 | 0.8278 | 0.7791 | 0.00000 |
| ResNet-18 scratch | 2 | 0.8968 | 0.8057 | 0.8197 | 0.0598 | 0.8443 | 0.7944 | 0.00000 |
| pretrained | 0 | 0.9128 | 0.8481 | 0.8511 | 0.0858 | 0.8680 | 0.8300 | 0.00038 |
| pretrained | 1 | 0.9193 | 0.8465 | 0.8617 | 0.0702 | 0.8780 | 0.8420 | 0.00077 |
| pretrained | 2 | 0.9178 | 0.8483 | 0.8601 | 0.0769 | 0.8753 | 0.8401 | 0.00000 |

Mean ± sd over seeds: custom A accuracy 0.8765 ± 0.0040, macro-F1
0.7931 ± 0.0053; custom B accuracy 0.8759 ± 0.0022, macro-F1
0.7903 ± 0.0017; scratch accuracy 0.8909 ± 0.0056, macro-F1
0.8110 ± 0.0076; pretrained accuracy 0.9166 ± 0.0034, macro-F1
0.8576 ± 0.0057.

Seed-0 cluster-bootstrap 95% CIs (2,044 test components, 2,000
resamples, seed 0): macro-F1 custom A [0.7784, 0.8140], custom B
[0.7693, 0.8055], scratch [0.7893, 0.8239], pretrained [0.8351,
0.8653]. The intervals are wide, likely because Taxi (n = 113) and
Van (n = 144) have small support.

Paired bootstrap differences are stored as custom A minus the other
model; below they are shown as other minus custom A (sign flipped):

| comparison | accuracy diff | macro-F1 diff |
|---|---|---|
| scratch - custom A | +0.0091 [0.0000, +0.0178] | +0.0107 [-0.0054, +0.0273] |
| pretrained - custom A | +0.0319 [+0.0228, +0.0412] | +0.0550 [+0.0381, +0.0718] |
| custom B - custom A | -0.0038 [-0.0119, +0.0037] | -0.0078 [-0.0218, +0.0055] |

NonVehicles recall is near-perfect for every family (seed-0 recall
0.9985-1.0), so the 7-vehicle-class metrics are the more informative
numbers: pretrained leads at 7-way macro-F1 0.830-0.842, custom A
0.758-0.768, custom B 0.759-0.763, scratch 0.779-0.794. The
vehicle-predicted-NonVehicles rate is at most 0.00077.

### Format shortcut

The harmonized arm re-encodes PNG inputs to JPEG at the estimated
quality 75 / 4:2:0. NonVehicles recall is essentially unchanged:
custom A seed 0 scores 0.9985 on both the original and harmonized
test sets (all seeds stay above 0.991). The model trained on
harmonized data also matches the original-format model: custom B
minus custom A macro-F1 is -0.0078 [-0.0218, +0.0055]. JPEG
compression format is therefore not the cue the model uses for
NonVehicles. The NonVehicles source difference (a Udacity/GTI-style
origin versus the vehicle classes) remains a possible non-format
shortcut, still only a hypothesis.

### Leakage inflation

There is no measurable inflation from the random image-level split.
The random split is a plain permutation (not stratified), so its
test set differs in composition from the grouped test: the custom
CNN trained on it scores accuracy
0.8643 and macro-F1 0.7709 on its own random test (n = 3,956),
versus 0.8809 / 0.7961 for custom A on the grouped test. On the
clean subset of the random test whose components never appear in the
random train (n = 1,574 of 3,956; 60% of random-test images share a
component with random train), the same model scores accuracy 0.9485
and macro-F1 0.7542. The grouped model on the grouped test is the
headline number. Headline difference, stored as (a) minus (c): the
random model on the random test (macro-F1 0.7709) minus custom A on
the grouped test (0.7961) is -0.0252. The grouped protocol does not
lower the score. Caveats: the three numbers come from different
test sets, a single seed, and the random split is not
class-stratified. A plausible reason the random split does not
inflate, labeled as a hypothesis: crops that share a video frame are
different objects, so component grouping removes less than it
appears to.

### Training budget

All runs use cosine schedules to a fixed maximum (40 epochs, 15 for
pretrained) with early stopping on validation macro-F1 (patience 8,
4 pretrained). Most runs ran to the maximum without early stopping
firing, and best epochs cluster at the end of the schedule, as
expected under cosine annealing. The results are for these fixed
budgets; longer schedules were not explored.

| run | kind | best epoch | epochs run |
|---|---|---|---|
| custom_A_s0 | custom A | 39 | 40 |
| custom_A_s1 | custom A | 34 | 40 |
| custom_A_s2 | custom A | 21 | 29 |
| custom_B_s0 | custom B | 24 | 32 |
| custom_B_s1 | custom B | 36 | 40 |
| custom_B_s2 | custom B | 39 | 40 |
| resnet_scratch_s0 | ResNet-18 scratch | 28 | 36 |
| resnet_scratch_s1 | ResNet-18 scratch | 38 | 40 |
| resnet_scratch_s2 | ResNet-18 scratch | 39 | 40 |
| resnet_pretrained_s0 | pretrained | 8 | 12 |
| resnet_pretrained_s1 | pretrained | 11 | 15 |
| resnet_pretrained_s2 | pretrained | 12 | 15 |
| inflation_random_s0 | custom A on random split | 32 | 40 |

### Provenance

The dataset's source and license are unknown; it was supplied as a
local archive. The NonVehicles class origin (possibly a different
collection such as Udacity/GTI) is a hypothesis, not a verified fact.

## Reproduce

```bash
uv sync
uv run pytest          # synthetic fixtures only, no real data
scripts/kaggle_run.sh --smoke   # diagnostic kernel
scripts/kaggle_run.sh           # full GPU kernel
```

Local tests never touch the real dataset: audit, hashing, and training
run only inside Kaggle.
