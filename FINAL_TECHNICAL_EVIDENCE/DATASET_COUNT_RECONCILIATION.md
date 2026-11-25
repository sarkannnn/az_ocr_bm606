# AzOCR-V2 — Dataset Count & Font Inventory Reconciliation (Task T09)

## 1. Exact Relationship Between Historical & Clean Dataset Counts

- **`10,400` corpus lines (`corpus_line/train.txt`):** Original training corpus line count. Also equals the exact `synth_line_v1/train` image count (`10,400`), whereas `10,000` was an approximate/rounded historical summary number in earlier notes.
- **Exact Leakage Breakdown (before cleaning, from `clean_dataset_summary.json`):**
  - `100` exact duplicates between `train` and `val`
  - `84` exact duplicates between `train` and `test`
  - `25` overlapping lines present in both sets ($100 + 84 - 25 = \mathbf{159}$ unique affected texts removed from `train`).
- **`10,241` V1Clean samples (`synth_line_v1_clean/train`):** $10,400 - 159 = \mathbf{10,241}$ samples (verified `0` train/val overlap after cleaning).
- **`16,000` historical V2 synthetic images (`synth_line_v2/train`):** Rendered from `10,176` unique corpus texts (`16,000` total images).
- **`15,754` V2Clean samples (`synth_line_v2_clean/train`):** Removing all images whose text matched the `159` leaked texts removed `246` images ($16,000 - 246 = \mathbf{15,754}$, verified `0` train/val and `0` train/test overlap).

## 2. Font Inventory Reconciliation (`14` vs `16 files / 15 families`)

- On disk, `fonts/` has **10** `.ttf` files and `fonts_new/` has **6** `.ttf` files = **16 `.ttf` files total**, representing **15 unique font families** (`consola.ttf` and `consolab.ttf` are Regular and Bold of Consolas).
- Historical V1 held out 20% of `fonts/` (`consola`, `tahoma`), leaving **8** training fonts.
- Historical V2 (`14` fonts in `ablation_2x2_results.json`) vs `16` files on disk reflects whether the 2 held-out fonts were excluded (`16 - 2 = 14`) vs counting all `16` `.ttf` files in `fonts/ + fonts_new/`.
