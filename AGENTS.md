# Vehicle Classification CNN

A reproducible 8-class vehicle image classification benchmark with a
custom CNN, ResNet-18 from scratch, and pretrained ResNet-18, a pHash
leakage audit, a grouped 70/15/15 split, and a frozen Kaggle GPU
protocol.

## Build commands

```bash
# Verification commands.
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

## Platform support

- macOS and Linux; Python 3.12-3.13.
- Local verification is CPU-only on tiny synthetic fixtures. The real
  data audit, hashing, and all training run only inside the private
  Kaggle GPU kernel via `scripts/kaggle_run.sh`.

## Key conventions

- Frozen protocol `vehicle-cnn-benchmark-v1`: fingerprint -> pHash
  audit and threshold -> grouped split -> JPEG forensics -> weighted
  choice -> training -> `analysis_plan.json` -> test decode/eval.
  `ordering_ok` requires `card_write_epoch < first_test_read_epoch`.
- The dataset fingerprint is a sha256 over sorted `relpath\tfile_sha256`
  lines plus per-class counts and the original zip sha256, committed as
  `data/fingerprint.json` after the smoke run and asserted in the full
  run.
- pHash near-duplicates are audited at thresholds 0/2/4/6/8, but only
  6, 4, 2, 0 are selectable (8 is stats-only); pick the largest whose
  largest component is at most 5% of images. Components are grouped so
  no component crosses a split; zero cross-split edges is asserted.
- All models: AdamW, batch 256, lr 1e-3 scratch / 3e-4 pretrained,
  weight decay 0.05, cosine schedule, 1 warmup epoch, label smoothing
  0.1, AMP fp16, early stop on val macro-F1 (patience 8 scratch / 4
  pretrained), max 40 epochs (15 pretrained).
- Uncertainty is a paired cluster bootstrap over test components,
  2000 resamples, seed 0, percentile intervals, on seed-0 models.
- `results/` holds unedited kernel output; `image.png` renders from
  `results/` only.

## Rules

- Default branch is `main`; never force-push it or rewrite shared history.
- Ask before adding any dependency or package manager.
- Never fabricate data, benchmarks, logs or results.
- Done means the relevant tests actually ran and passed; never skip or mock them to finish.
- No emojis anywhere: code, comments, commits, READMEs, docs.
- No em dashes (the long dash, U+2014) anywhere: code, comments, commits, READMEs, docs. Use a comma, colon, parentheses, or a new sentence instead.
- Keep README in sync with code changes, following its existing template.
- Comments are brief and explain why, only when the code doesn't already say it.
- Keep dataset splits and eval leakage controls intact.
- Never run data audit, hashing, or training on the laptop; only
  tiny synthetic fixtures in tests.

## Commits

- Conventional Commits: `type(scope): summary`. Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`; scope is optional.
- Summary in imperative mood, lowercase, no trailing period, at most 72 characters.
- One logical change per commit; keep formatting-only changes separate.
- Add a body only when the reason is not obvious: explain why, not what, wrapped at 72.
- Breaking changes: `type!:` in the summary plus a `BREAKING CHANGE:` footer.
- Commit only when asked. Never commit secrets, build output, or failing code.
