# Second Training Round Preparation (v2) - Diagnostic & Setup Report

## Step 1: Measured Numbers & Baseline Analysis

### 1a. Digits and Punctuation Ratios
- **`data/corpus_word/train.txt` (20,000 words):**
  - Words containing a digit: **0** (**0.00%**)
  - Words with leading or trailing punctuation (`.,;:!?"()'"`): **0** (**0.00%**)
- **`data/synth_word/train/labels.tsv` (20,000 labels):**
  - Labels containing a digit: **0** (**0.00%**)
  - Labels with leading or trailing punctuation: **0** (**0.00%**)

### 1b. Character Frequencies in `synth_word/train`
- **Total character instances:** 171,718
- Frequency of `'a'`: **19,776** (**11.52%**)
- Frequency of `'q'`: **2,508** (**1.46%**)
- **Top 10 Most Frequent Characters:**
  1. `'a'`: 19,776 (11.52%)
  2. `'i'`: 14,965 (8.71%)
  3. `'n'`: 13,148 (7.66%)
  4. `'r'`: 12,829 (7.47%)
  5. `'l'`: 11,546 (6.72%)
  6. `'ə'`: 9,238 (5.38%)
  7. `'d'`: 7,278 (4.24%)
  8. `'e'`: 6,632 (3.86%)
  9. `'s'`: 6,309 (3.67%)
  10. `'t'`: 6,148 (3.58%)

### 1c. Windows Candidate Fonts Scan
Scanned `C:\Windows\Fonts`. Found 59 fonts passing `font_ok`, excluding existing project fonts (`fonts/`) and sim-test fonts (`pala.ttf`, `cambria.ttc`, `micross.ttf`, `CascadiaMono.ttf`).

**Monospace fonts identified:**
- `CascadiaCode.ttf`
- `consolab.ttf`, `consolai.ttf`, `consolaz.ttf`
- `courbd.ttf`, `courbi.ttf`, `couri.ttf`
- `Moba_Fanta.ttf`, `Moba_Fira.ttf`, `Moba_Font.ttf`

---

## Step 2: New Training Fonts (`fonts_new/`)

Copied 6 diverse fonts to `fonts_new/` (excluding all 4 sim-test fonts):
1. **`CascadiaCode.ttf`** - Monospace 1 (Modern coding font)
2. **`consolab.ttf`** - Monospace 2 (Consolas Bold)
3. **`bahnschrift.ttf`** - Sans 1 (DIN geometric sans)
4. **`ebrima.ttf`** - Sans 2 (Clean readable sans)
5. **`l_10646.ttf`** - Sans 3 (Lucida Sans Unicode)
6. **`palab.ttf`** - Serif (Palatino Linotype Bold)

---

## Step 3: Dataset Preparation & Sizes (`v2`)

All `val/` and `test/` splits were copied **identically** from the original datasets to guarantee fair evaluation benchmarks.

### Dataset Split Sizes:
| Dataset | Train Split | Val Split (Identical) | Test Split (Identical) | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| **`data/synth_word_v2`** | **35,000** (20,000 old + 15,000 extra) | **2,000** | **2,000** | **39,000** |
| **`data/synth_line_v2`** | **16,000** (10,000 old + 6,000 extra) | **1,000** | **1,000** | **18,000** |

### Extra Word Corpus Composition:
- Because digits and punctuation were 0.00% in original word training data, extra words were constructed from `data/corpus_line/train.txt` and `data/corpus_word/train.txt`.
- Special tokens (digits and/or leading/trailing punctuation): **3,000** (**20.00%**).
- Plain alphabetic tokens: **12,000** (**80.00%**).
- **Leakage Verification:** Exactly **0** tokens from `val` and `test` appear in the new train set (0 overlaps detected).

---

## Step 4: Training Script `train_v2.py`

Key improvements implemented:
1. **Photo-Realistic Augmentations (`AUG_CONFIG`):**
   - Rotation $\pm 5^\circ$ ($p = 0.5$)
   - Perspective jitter ($p = 0.3$)
   - Gaussian blur $\sigma \in [0.5, 2.0]$ ($p = 0.6$)
   - Motion blur kernel 3–7 ($p = 0.15$)
   - Gaussian sensor noise $\sigma \in [3, 12]$ ($p = 0.6$)
   - Salt & pepper noise ($p = 0.2$)
   - Downscale/upscale factor $0.5 - 0.9$ ($p = 0.4$)
   - JPEG compression quality $40 - 90$ ($p = 0.4$)
   - Brightness & Contrast $\alpha \in [0.6, 1.1]$, $\beta \in [-40, +20]$ ($p = 0.6$)
   - Background tint & smooth gradient $0.7 - 1.0$ + soft dark band ($p = 0.5$)
   - Contrast stretching min-max to $0 - 255$ ($p = 0.5$)
2. **Resume Capability:** Added `--resume <ckpt_path>` to resume weights, optimizer, and epoch counter.
3. **Checkpoints:** Saves `last.pt` every epoch in addition to `best.pt`.
4. **Augmentation Preview:** Added `--save-aug-preview <out_path>` to render contact sheets.

---

## Step 5: Verification & Smoke Test

1. **Augmentation Preview Visual Inspection (`results/v2_prep/aug_preview.png`):**
   - 24 random samples inspected.
   - Text remains legible across all samples.
   - Special Azerbaijani diacritics (`ə`, `ı`, `ö`, `ğ`, `ü`, `ş`, `ç`) and punctuation are preserved without truncation or artifact erasure.
2. **Smoke Test Pipeline Verification:**
   - 1-epoch smoke run executed on `data/smoke_word` (out: `runs/smoke_v2`).
   - Saved `best.pt` and `last.pt` in 0.1 min.
   - `evaluate.py --ckpt runs/smoke_v2/best.pt --data data/smoke_word/test` executed successfully end-to-end.

---

## Step 6: Commands for Real Training (When Ready)

### Word Model v2 (35,000 train images, 25 epochs):
```powershell
.\.venv\Scripts\python.exe train_v2.py --mode word --train-dir data/synth_word_v2/train --val-dir data/synth_word_v2/val --out runs/word_v2 --epochs 25 --workers 0
```

### Line Model v2 (16,000 train images, 15 epochs):
```powershell
.\.venv\Scripts\python.exe train_v2.py --mode line --train-dir data/synth_line_v2/train --val-dir data/synth_line_v2/val --out runs/line_v2 --epochs 15 --workers 0
```
