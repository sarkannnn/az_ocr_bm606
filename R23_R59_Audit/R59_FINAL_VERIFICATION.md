# AzOCR-V2 — R59 Final Verification Report (`R59_FINAL_VERIFICATION.md`)

**Verification Date:** 2026-11-25  
**Source Data:** `R59_RAW_PREDICTIONS.csv` (200 images × 22 OCR systems), `R59_DATASET_MANIFEST.csv`  
**Cross-Check Result:** Recomputed all 132 rows (`22 models × 6 scopes`) from raw predictions: **0 mismatches** against `R59_FONT_BENCHMARK.csv`; **0/200** image SHA-256 mismatches.

---

## 1. Resolution of 16 vs. 18 Training Font Discrepancy

An audit of the training font directories and scripts resolved the 16 vs. 18 font label discrepancy:
- **Actual `.ttf` files on disk:** `fonts/` contains **10** files (`arial.ttf`, `calibri.ttf`, `candara.ttf`, `consola.ttf`, `corbel.ttf`, `cour.ttf`, `segoeui.ttf`, `tahoma.ttf`, `times.ttf`, `verdana.ttf`) and `fonts_new/` contains **6** files (`bahnschrift.ttf`, `CascadiaCode.ttf`, `consolab.ttf`, `ebrima.ttf`, `l_10646.ttf`, `palab.ttf`) — totaling **16 `.ttf` files** (15 unique font families, as `consola.ttf` and `consolab.ttf` are Regular/Bold weights of Consolas).
- **V1 vs. V2 Split:** V1 (`Group A/B`) used 8 training fonts from `fonts/` (holding out `consola` and `tahoma` at 20%). V2Clean (`Group C/D/RandomPadding/VariableWidth`) expanded coverage using the `16`-file pool (`fonts/` + `fonts_new/`). The string `"Geniş 18 font"` in `run_night_training.ps1` was a legacy/rounded script comment rather than 18 distinct font files on disk.
- **Zero Overlap Verified by Font Identifiers:** None of the 5 R59 open-license PostScript/Family identifiers (`DejaVuSans`, `DejaVuSerif`, `DejaVuSansMono`, `Lato-Regular`, `Raleway-Regular`) appear in the 16 training font identifiers.

---

## 2. Font Licenses & Text Authorship Status

| Font File | Family | Subfamily | PostScript Name | License (from source/name table) | SHA-256 (Prefix) |
|---|---|---|---|---|---|
| `DejaVuSans.ttf` | DejaVu Sans | Book | `DejaVuSans` | DejaVu / Bitstream Vera License (LICENSE_DEJAVU.txt) | `3fdf69cabf06049e` |
| `DejaVuSansMono.ttf` | DejaVu Sans Mono | Book | `DejaVuSansMono` | DejaVu / Bitstream Vera License (LICENSE_DEJAVU.txt) | `602ec86b8948cfcd` |
| `DejaVuSerif.ttf` | DejaVu Serif | Book | `DejaVuSerif` | DejaVu / Bitstream Vera License (LICENSE_DEJAVU.txt) | `107244956e9962b9` |
| `Lato-Regular.ttf` | Lato | Regular | `Lato-Regular` | SIL Open Font License 1.1 | `6f6940be0835c3dd` |
| `Raleway-Regular.ttf` | Raleway | Regular | `Raleway-Regular` | SIL Open Font License 1.1 | `20e4ae409ffbe8bf` |

- **Benchmark Sentences (40 lines):** Newly authored specifically for this evaluation benchmark, verified to have **0/40 overlap** with `corpus_line_clean` and `Real30`. **License Status:** Pending formal rights-holder/author CC0-1.0 sign-off prior to public Zenodo release (do not declare CC0 final until signed off by the corresponding author).

---

## 3. Auto-Generated Overall Results (`ALL_5_FONTS`, 200 Lines)

*(Every cell below is computed programmatically from `R59_RAW_PREDICTIONS.csv`; no manual transcription.)*

| Model / Group | CER (%) | WER (%) | Exact (%) | ə (%) | Ə (%) | ı (%) | i (%) | İ (%) |
|---|---|---|---|---|---|---|---|---|
| Baseline_CRNN-V2_Raw | 0.11 | 0.89 | 92.0 | 100.0 | 100.0 | 99.28 | 99.88 | 98.0 |
| Baseline_Tesseract5_PSM7 | 0.11 | 0.94 | 91.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Baseline_EasyOCR | 0.5 | 6.15 | 62.0 | 99.8 | 100.0 | 99.1 | 100.0 | 72.0 |
| Baseline_PaddleOCR | 0.5 | 3.49 | 81.0 | 100.0 | 80.0 | 86.49 | 100.0 | 100.0 |
| Group A (V1Clean + BasicAug) (n=3) | 1.48 ± 0.78 | 11.20 ± 5.68 | 58.50 ± 9.12 | 99.71 ± 0.51 | 96.19 ± 1.65 | 96.52 ± 1.73 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Group B (V1Clean + PhotoAug) (n=3) | 0.92 ± 0.37 | 7.46 ± 2.84 | 68.00 ± 5.89 | 99.98 ± 0.04 | 97.14 ± 0.00 | 99.40 ± 0.52 | 99.98 ± 0.03 | 89.33 ± 16.77 |
| Group C (V2Clean + BasicAug) (n=3) | 0.17 ± 0.08 | 1.76 ± 0.71 | 89.33 ± 5.11 | 99.84 ± 0.28 | 100.00 ± 0.00 | 99.70 ± 0.27 | 99.98 ± 0.03 | 99.33 ± 1.15 |
| Group D (V2Clean + PhotoAug) (n=3) | 0.13 ± 0.06 | 1.15 ± 0.50 | 90.67 ± 3.51 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.64 ± 0.31 | 99.94 ± 0.06 | 90.67 ± 16.17 |
| RandomPadding (Group D) (n=3) | 0.15 ± 0.10 | 1.32 ± 0.99 | 89.50 ± 7.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.52 ± 0.42 | 99.94 ± 0.10 | 96.67 ± 5.77 |
| VariableWidth (Group D, Tesla T4) (n=3) | 0.15 ± 0.13 | 1.56 ± 1.31 | 88.83 ± 9.31 | 100.00 ± 0.00 | 99.05 ± 1.65 | 100.00 ± 0.00 | 100.00 ± 0.00 | 91.33 ± 13.32 |

---

## 4. Auto-Generated Per-Font CER Micro (%)

| Model / Group | DejaVuSans | DejaVuSerif | DejaVuSansMono | Lato-Regular | Raleway-Regular |
|---|---:|---:|---:|---:|---:|
| Baseline_CRNN-V2_Raw | 0.03 | 0.14 | 0.12 | 0.0 | 0.26 |
| Baseline_Tesseract5_PSM7 | 0.09 | 0.12 | 0.12 | 0.09 | 0.14 |
| Baseline_EasyOCR | 0.4 | 0.43 | 0.58 | 0.26 | 0.81 |
| Baseline_PaddleOCR | 0.0 | 0.0 | 2.37 | 0.0 | 0.12 |
| Group A (V1Clean + BasicAug) (n=3) | 0.00 ± 0.00 | 1.16 ± 0.59 | 4.74 ± 2.78 | 0.01 ± 0.02 | 1.49 ± 0.60 |
| Group B (V1Clean + PhotoAug) (n=3) | 0.05 ± 0.08 | 0.26 ± 0.18 | 3.58 ± 1.66 | 0.00 ± 0.00 | 0.72 ± 0.45 |
| Group C (V2Clean + BasicAug) (n=3) | 0.02 ± 0.02 | 0.25 ± 0.22 | 0.17 ± 0.11 | 0.05 ± 0.05 | 0.35 ± 0.37 |
| Group D (V2Clean + PhotoAug) (n=3) | 0.07 ± 0.09 | 0.10 ± 0.09 | 0.09 ± 0.07 | 0.08 ± 0.09 | 0.33 ± 0.10 |
| RandomPadding (Group D) (n=3) | 0.08 ± 0.07 | 0.28 ± 0.08 | 0.10 ± 0.09 | 0.07 ± 0.07 | 0.24 ± 0.34 |
| VariableWidth (Group D, Tesla T4) (n=3) | 0.07 ± 0.09 | 0.08 ± 0.02 | 0.12 ± 0.08 | 0.01 ± 0.02 | 0.49 ± 0.49 |
