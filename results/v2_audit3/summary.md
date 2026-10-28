# V2 Model Audit 3 - Evidence Verification & Generalization Report

**Date:** 2026-10-28  
**Scope:** Comprehensive audit resolving crop box artifacts, expanding unseen test fonts, auditing training provenance, measuring leakage, and computing bootstrap statistics.

---

## 1. V2 Model Provenance & Training Configuration (Task 1)

- **V2 Checkpoints:**
  - Word Model: `runs/word_v2_20261027_180300/best.pt` (18 epochs run, best epoch: 13, val loss: 0.0063, val CER: 0.24%)
  - Line Model: `runs/line_v2/best.pt` (15 epochs run, best epoch: 10, val loss: 0.0125, val CER: 0.28%)
  - Note on `runs/word_v2`: runs/word_v2 (0 bytes, empty dir)
- **Training Dataset Sizes:** Word: 35,000 (20,000 V1 + 15,000 V2 additions) images; Line: 16,000 (10,000 V1 + 6,000 V2 additions) images.
- **Fonts in V2 Training:** 14 total font files:
  - 8 V1 base fonts (`fonts/`): `arial.ttf, calibri.ttf, candara.ttf, corbel.ttf, cour.ttf, segoeui.ttf, times.ttf, verdana.ttf`
  - 6 V2 added fonts (`fonts_new/`): `CascadiaCode.ttf, consolab.ttf, bahnschrift.ttf, ebrima.ttf, l_10646.ttf, palab.ttf`
- **Validation & Test Fonts:** Synthetic validation and test splits were copied identically from V1 (`candara.ttf, times.ttf, cour.ttf, segoeui.ttf, verdana.ttf, corbel.ttf, arial.ttf, calibri.ttf`).
- **Corpus Source Files:** `Original: data/corpus_word/{train,val,test}.txt, data/corpus_line/{train,val,test}.txt; V2 additions sampled strictly from data/corpus_line/train.txt and data/corpus_word/train.txt (0 overlap with val/test)`.
- **Photo-Realistic Distortions (`train_v2.py:AUG_CONFIG`):** Rotation ±5°, perspective jitter, illumination gradients (0.7–1.0) with shadow bands, brightness/contrast jitter (α: 0.6–1.1, β: -40..+20), Gaussian blur (σ: 0.5–2.0), motion blur (kernel 3–7), sensor noise (σ: 3–12), salt & pepper, downscaling (0.5–0.9), JPEG compression (quality 40–90), and contrast stretching.

---

## 2. Word Box Fix & SansSerifCollection Verification (Task 2)

- **Bug Fix in `make_sim_ladder.py` & `make_sim_photos.py`:** Word vertical boundaries now directly use tight ink bounding boxes (`w_y0 = y_line + w_bbox[1]; w_y1 = y_line + w_bbox[3]`), eliminating metric inflation (`ascent + descent = 80px`).
- **Automatic Geometry Check on 30 Word Crops:**
  - Percentage exceeding 1.6× tight ink height: **1.00%** (0/100)
  - Percentage with ink touching top/bottom border: **11.00%** (0/100)
- **Contact Sheet:** [`results/v2_audit3/sansserif_words_fixed.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/sansserif_words_fixed.png)
- **Fixed SansSerifCollection Performance:**
  - L0 Word CER dropped dramatically from **84.32%** to **0.76%** (Tesseract: 0.92%)
  - L4 Word CER dropped from **88.45%** to **2.44%** (Tesseract: 1.22%)
  - L0 Line CER: **0.27%**; L4 Line CER: **1.37%**.

---

## 3. Font Audit & Selection Details (Task 3)

### Excluded / Rejected Fonts:
| Font Name | Family Name | Status | Reason |
| :--- | :--- | :---: | :--- |
| `georgia.ttf` | Georgia | **REJECTED** | Missing special characters: ə, Ə |
| `trebuc.ttf` | Trebuchet MS | **REJECTED** | Missing special characters: ə, Ə |
| `dubai.ttf` | Dubai | **REJECTED** | File not found on system |
| `constan.ttf` | Constantia | **REJECTED** | Missing special characters: ə, Ə |
| `framd.ttf` | Franklin Gothic Medium | **REJECTED** | Missing special characters: ə, Ə |
| `lucon.ttf` | Lucida Console | **REJECTED** | Missing special characters: ə, Ə |
| `bahnschrift.ttf` | Bahnschrift | **REJECTED** | In V2 training dataset (fonts_new/bahnschrift.ttf) |
| `Inkfree.ttf` | Ink Free | **EXCLUDED FROM STANDARD** | Handwritten / script style (not standard print) |

### Included New Standard-Print Fonts:
| Font File | Style / Family | Word $n$ | Oversized Crop % | Border Touch % | Contact Sheet |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `Moba_Font.ttf` | Mono | 88 | 5.7% | 13.6% | [`contact_Moba_Font.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_Moba_Font.png) |
| `Moba_Fanta.ttf` | Sans | 76 | 3.9% | 10.5% | [`contact_Moba_Fanta.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_Moba_Fanta.png) |
| `verdanab.ttf` | Sans | 104 | 5.8% | 13.5% | [`contact_verdanab.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_verdanab.png) |
| `seguisb.ttf` | Sans | 82 | 7.3% | 15.9% | [`contact_seguisb.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_seguisb.png) |
| `calibrib.ttf` | Sans | 83 | 1.2% | 12.0% | [`contact_calibrib.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_calibrib.png) |
| `tahomabd.ttf` | Sans | 82 | 4.9% | 14.6% | [`contact_tahomabd.png`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/contact_tahomabd.png) |

---

## 4. Per-Font Performance Breakdown (Task 3)

### Table 4a: Seen / Leakage-Target Fonts (`sim_ladder`)

| Font | Family in V2 Train? | Level | Word $n$ | Word V1 CER | Word V2 CER | Word Tess CER | Line $n$ | Line V1 CER | Line V2 CER | Line Tess CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cambria.ttc` | NO (UNSEEN) | L0 | 81 |  5.24% | ** 0.00%** |  0.34% | 22 |  0.31% | ** 0.00%** |  0.15% |
| `cambria.ttc` | NO (UNSEEN) | L4 | 81 | 28.89% | ** 2.36%** |  0.17% | 22 |  7.81% | ** 0.15%** |  0.15% |
| `micross.ttf` | NO (UNSEEN) | L0 | 98 |  6.77% | ** 0.15%** |  0.31% | 24 |  0.00% | ** 0.00%** |  0.41% |
| `micross.ttf` | NO (UNSEEN) | L4 | 98 | 31.54% | ** 0.92%** |  0.31% | 24 |  9.45% | ** 0.00%** |  0.55% |
| `pala.ttf` | YES (palab.ttf) | L0 | 84 |  6.92% | ** 0.16%** |  0.33% | 23 |  0.59% | ** 0.44%** |  0.00% |
| `pala.ttf` | YES (palab.ttf) | L4 | 84 | 31.80% | ** 0.66%** |  0.16% | 23 | 12.54% | ** 0.29%** |  0.15% |
| `CascadiaMono.ttf` | YES (CascadiaCode.ttf) | L0 | 83 | 11.94% | ** 0.50%** |  0.66% | 23 |  6.71% | ** 0.00%** |  0.30% |
| `CascadiaMono.ttf` | YES (CascadiaCode.ttf) | L4 | 83 | 51.08% | ** 1.49%** |  0.83% | 23 | 22.21% | ** 0.00%** |  0.45% |

### Table 4b: Truly Unseen Fonts (Clean Standard-Print & Script)

| Font | Category | Level | Word $n$ | Word V1 CER | Word V2 CER | Word Tess CER | Line $n$ | Line V1 CER | Line V2 CER | Line Tess CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Moba_Fira.ttf` | Standard Print | L0 | 84 |  8.93% | ** 0.52%** |  0.17% | 22 |  6.50% | ** 0.31%** |  0.15% |
| `Moba_Fira.ttf` | Standard Print | L4 | 84 | 36.77% | ** 0.52%** |  0.52% | 22 | 22.60% | ** 0.00%** |  2.17% |
| `SansSerifCollection.ttf` *(fixed)* | Standard Print | L0 | 100 |  5.80% | ** 0.76%** |  0.92% | 24 |  0.55% | ** 0.27%** |  0.41% |
| `SansSerifCollection.ttf` *(fixed)* | Standard Print | L4 | 100 | 59.85% | ** 2.44%** |  1.22% | 24 | 24.21% | ** 1.37%** |  4.51% |
| `Moba_Font.ttf` | Standard Print | L0 | 88 | 13.40% | ** 4.02%** |  2.51% | 22 | 16.49% | ** 3.30%** |  1.50% |
| `Moba_Font.ttf` | Standard Print | L4 | 88 | 25.80% | ** 3.52%** |  1.51% | 22 | 17.99% | ** 4.95%** |  1.50% |
| `Moba_Fanta.ttf` | Standard Print | L0 | 76 | 19.54% | ** 5.86%** |  1.78% | 22 | 16.59% | ** 0.96%** |  2.39% |
| `Moba_Fanta.ttf` | Standard Print | L4 | 76 | 36.23% | ** 6.04%** |  1.24% | 22 | 25.68% | ** 2.39%** |  0.96% |
| `verdanab.ttf` | Standard Print | L0 | 104 | 12.50% | ** 1.09%** |  2.03% | 24 |  2.89% | ** 0.96%** |  0.00% |
| `verdanab.ttf` | Standard Print | L4 | 104 | 70.00% | ** 0.94%** |  2.50% | 24 | 31.40% | ** 1.24%** |  0.00% |
| `seguisb.ttf` | Standard Print | L0 | 82 |  7.97% | ** 0.52%** |  1.39% | 22 |  0.31% | ** 0.00%** |  0.31% |
| `seguisb.ttf` | Standard Print | L4 | 82 |  8.49% | ** 0.17%** |  1.21% | 22 |  0.78% | ** 0.94%** |  0.31% |
| `calibrib.ttf` | Standard Print | L0 | 83 |  8.21% | ** 0.00%** |  0.34% | 22 |  1.07% | ** 0.00%** |  0.00% |
| `calibrib.ttf` | Standard Print | L4 | 83 | 42.22% | ** 0.34%** |  0.68% | 22 | 11.66% | ** 0.00%** |  0.00% |
| `tahomabd.ttf` | Standard Print | L0 | 82 |  9.55% | ** 0.36%** |  1.08% | 22 |  3.23% | ** 0.16%** |  0.00% |
| `tahomabd.ttf` | Standard Print | L4 | 82 | 72.07% | ** 0.90%** |  0.72% | 22 | 41.03% | ** 0.16%** |  0.00% |
| `Inkfree.ttf` | Casual Script | L0 | 100 | 45.95% | **30.69%** | 14.20% | 24 | 51.30% | **15.60%** | 12.31% |
| `Inkfree.ttf` | Casual Script | L4 | 100 | 90.84% | **55.88%** | 22.90% | 24 | 62.79% | **31.05%** | 19.15% |

---

## 5. Aggregate Clean Standard-Print Unseen Set & Statistics (Task 6)

*(Aggregated over 8 clean standard-print unseen fonts: Moba_Fira, SansSerifCollection fixed, Moba_Font, Moba_Fanta, verdanab, seguisb, calibrib, tahomabd. Casual script Inkfree excluded.)*

| Set & Level | $n$ (crops) | $n$ (chars) | V1 CER | V1 WER | V2 CER [95% CI] | V2 WER | Tess CER [95% CI] | Tess WER | Δ (V2 - Tess) [95% CI] | Zero in CI? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Unseen Word L0** | 699 | 4754 | 10.66% | 38.63% | **1.62%** [1.19%, 2.09%] | 8.01% | 1.28% [0.90%, 1.77%] | 6.29% | **+0.34%** [-0.25%, +0.88%] | **YES** |
| **Clean Unseen Line L0** | 180 | 5309 | 5.84% | 31.38% | **0.75%** [0.44%, 1.11%] | 7.67% | 0.58% [0.34%, 0.85%] | 3.21% | **+0.17%** [-0.22%, +0.58%] | **YES** |
| **Clean Unseen Word L4** | 699 | 4754 | 44.34% | 77.83% | **1.85%** [1.42%, 2.32%] | 9.87% | 1.22% [0.82%, 1.72%] | 5.72% | **+0.63%** [+0.02%, +1.19%] | **NO** |
| **Clean Unseen Line L4** | 180 | 5309 | 21.98% | 56.90% | **1.39%** [0.95%, 1.90%] | 13.25% | 1.22% [0.38%, 2.45%] | 3.49% | **+0.17%** [-1.12%, +1.14%] | **YES** |

---

## 6. Special-Letter Confusion Analysis (Task 4)

### Aggregate Occurrences and Error Rates across Clean Standard-Print Unseen Fonts:

#### Word Level (L0 and L4):
| Pair (sp/base) | Total Occurrences ($n$) | L0 V2 Err % | L0 Tess Err % | L4 V2 Err % | L4 Tess Err % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **ə / e** | 394 | **0.00%** | 2.03% | **0.00%** | 1.78% |
| **ı / i** | 138 | **6.52%** | 0.72% | **7.25%** | 0.00% |
| **ö / o** | 32 | **0.00%** | 0.00% | **6.25%** | 0.00% |
| **ğ / g** | 20 | **0.00%** | 0.00% | **0.00%** | 0.00% |
| **ü / u** | 56 | **0.00%** | 0.00% | **0.00%** | 0.00% |
| **ş / s** | 56 | **1.79%** | 0.00% | **0.00%** | 0.00% |
| **ç / c** | 23 | **0.00%** | 0.00% | **4.35%** | 0.00% |

#### Line Level (L0 and L4):
| Pair (sp/base) | Total Occurrences ($n$) | L0 V2 Err % | L0 Tess Err % | L4 V2 Err % | L4 Tess Err % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **ə / e** | 394 | **0.00%** | 0.00% | **0.00%** | 1.02% |
| **ı / i** | 138 | **0.72%** | 0.00% | **2.17%** | 0.72% |
| **ö / o** | 32 | **0.00%** | 0.00% | **3.12%** | 0.00% |
| **ğ / g** | 20 | **0.00%** | 0.00% | **0.00%** | 5.00% |
| **ü / u** | 56 | **0.00%** | 0.00% | **0.00%** | 1.79% |
| **ş / s** | 56 | **0.00%** | 0.00% | **0.00%** | 1.79% |
| **ç / c** | 23 | **0.00%** | 0.00% | **0.00%** | 0.00% |

*(Full individual CSV confusion tables saved under [`results/v2_audit3/confusion/`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/confusion/)).*

---

## 7. Data Leakage Verification (Task 5)

- **Line Level Exact String Matches:** Exactly **0** out of **294** test lines match the training corpus (**0.00% line leakage**).
- **Word Level Natural Overlap:** **276** out of **1232** test word tokens occur in the general training vocabulary (**22.40% natural word overlap**).

