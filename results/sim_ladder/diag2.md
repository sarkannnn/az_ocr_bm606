# Diagnostic Report 2: Punctuation Impact, Training Augmentation, and L2 Substitutions

## 1. Punctuation Impact on L0 Word Evaluation

Evaluated on `data/sim_ladder/L0/word/test` (346 words across 4 unseen fonts):

### Word Model (runs/word/best.pt)
- **Total word samples:** 346
- **Raw word errors:** 96 / 346 (WER: **27.75%**, CER: **7.71%**)
- **Punctuation-only errors:** **26 / 96** (**27.08%** of all errors differ ONLY by leading/trailing punctuation `.,;:!?"()`)
- **Punctuation-stripped results:**
  - Stripped CER: **6.54%** (down from 7.71%, a reduction of 1.17 percentage points)
  - Stripped WER: **20.23%** (down from 27.75%, a reduction of 7.52 percentage points)
- **Remaining non-punctuation errors (70 words):**
  - The majority stem from tokens containing digits (e.g. `1927-ci` -> `Izzzci`, `27` -> `zz`, `1993)` -> `IggƏ`) and parentheses (`(BMTQAK` -> `ÇBMTQAK`, `(film,` -> `Çfilm`), which were absent from the word training corpus (`data/corpus_word`).

### Tesseract Baseline
- **Total word samples:** 346
- **Raw word errors:** 9 / 346 (WER: **2.60%**, CER: **0.41%**)
- **Punctuation-only errors:** **2 / 9** (**22.22%**)
- **Punctuation-stripped results:**
  - Stripped CER: **0.34%** (down from 0.41%)
  - Stripped WER: **2.02%** (down from 2.60%)

---

## 2. Augmentation Parameters in `train.py` / `common.py`

From `train.py` (`augment(img, rng)`):
- **Rotation:** Probability $p = 0.5$, uniform angle range $[-5^\circ, +5^\circ]$ (`cv2.warpAffine`, `BORDER_REPLICATE`)
- **Gaussian blur:** Probability $p = 0.3$, kernel size $k \in \{3, 5\}$, $\sigma = 0$ (computed by OpenCV: $\sigma \approx 0.8$ for $k=3$, $\sigma \approx 1.1$ for $k=5$)
- **Salt-and-pepper noise:** Probability $p = 0.3$, sets pixels $< 0.005$ to 0 and $> 0.995$ to 255
- **Perspective warp:** Probability $p = 0.3$, 4-corner jitter $\pm 2\%$ width, $\pm 8\%$ height (`cv2.warpPerspective`, `BORDER_REPLICATE`)
- **Sensor / Gaussian noise:** **None** ($p = 0$, not implemented)
- **Brightness / Contrast / Illumination:** **None** ($p = 0$, not implemented)
- **Shadow band / Gradients:** **None** ($p = 0$, not implemented)
- **Downscale / Upscale (Resolution degradation):** **None** ($p = 0$, not implemented)
- **JPEG compression:** **None** ($p = 0$, not implemented)

From `common.py` (`preprocess(img, cfg)`):
- Bilinear resize to fixed height 32 px, right-side canvas padding with constant 255 (white), raw division by 255.0. No normalization, no thresholding, no augmentation.

---

## 3. Top 15 Character Substitutions at L2 (Word Model)

Measured from `cer_wer` alignment operations on `data/sim_ladder/L2/word/test` (346 words):

| Rank | Reference (`ref`) | Prediction (`hyp`) | Frequency |
| :---: | :---: | :---: | :---: |
| 1 | `i` | `a` | 31 |
| 2 | `ə` | `a` | 25 |
| 3 | `l` | `t` | 23 |
| 4 | `ı` | `a` | 15 |
| 5 | `m` | `a` | 9 |
| 6 | `n` | `a` | 8 |
| 7 | `f` | `t` | 8 |
| 8 | `n` | `q` | 7 |
| 9 | `d` | `a` | 7 |
| 10 | `u` | `q` | 6 |
| 11 | `i` | `ü` | 6 |
| 12 | `l` | `d` | 6 |
| 13 | `"` | `c` | 6 |
| 14 | `i` | `d` | 5 |
| 15 | `r` | `q` | 5 |
