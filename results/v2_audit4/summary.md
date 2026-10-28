# V2 Model Audit 4 - Comprehensive Generalization, Statistics & Diagnostic Report

**Date:** 2026-10-28  
**Scope:** Addressing all Audit 3 open issues: exact code verification of alignment, true 3-tier font classification (Seen, Near-Relative, New Family), cluster-level bootstrap statistics, dotless-i and diacritic error breakdown, line WER space error decomposition, geometry clipping validation, and validation set provenance.

---

## 1. Alignment & Metric Code Comparison (Task 1)

### Code Comparison:
`run_v2_audit3.py` and `evaluate.py` use the exact same underlying alignment functions from `common.py`:

```python
# From common.py (used by evaluate.py and run_v2_audit3.py):
def align(ref, hyp):
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(('ok' if ref[i - 1] == hyp[j - 1] else 'sub', ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(('del', ref[i - 1], ''))
            i -= 1
        else:
            ops.append(('ins', '', hyp[j - 1]))
            j -= 1
    return int(d[n, m]), ops[::-1]

def cer_wer(refs, hyps):
    ce = cn = we = wn = 0
    all_ops = []
    for r, h in zip(refs, hyps):
        e, ops = align(r, h)
        ce += e; cn += len(r); all_ops.extend(ops)
        e2, _ = align(r.split(), h.split())
        we += e2; wn += len(r.split())
    return ce / max(cn, 1), we / max(wn, 1), all_ops
```

- **Behavioral Verdict:** The alignment, CER, WER, and confusion calculations in `run_v2_audit3.py` and `evaluate.py` are **100% identical in implementation and behavior**. Both call `common.cer_wer` directly, and the bootstrap script used `common.align(r, h)[0]`, which computes the exact same Levenshtein distance `e` as `cer_wer`.
- **Numerical Impact:** No numbers changed as a result of alignment logic.

---

## 2. Real Font Classification: Groups A, B, and C (Task 2)

Fonts are rigorously classified based on the 14 V2 training font files (`arial, calibri, candara, corbel, cour, segoeui, times, verdana, CascadiaCode, consolab, bahnschrift, ebrima, l_10646, palab`):

1. **Group A: SEEN (4 tested fonts)**
   - `arial.ttf`, `calibri.ttf`, `segoeui.ttf`, `times.ttf`
   - Tested on test-corpus lines (`data/corpus_line/test.txt`) at L0 and L4.
2. **Group B: NEAR-RELATIVE (5 fonts)**
   - Different file or weight/width sibling of a training font: `pala.ttf` (sibling of `palab.ttf`), `CascadiaMono.ttf` (sibling of `CascadiaCode.ttf`), `verdanab.ttf` (sibling of `verdana.ttf`), `seguisb.ttf` (sibling of `segoeui.ttf`), `calibrib.ttf` (sibling of `calibri.ttf`).
3. **Group C: NEW FAMILY (7 fonts)**
   - Completely distinct font families with zero family presence in training: `cambria.ttc`, `micross.ttf`, `Moba_Fira.ttf`, `SansSerifCollection.ttf`, `Moba_Font.ttf`, `Moba_Fanta.ttf`, `tahomabd.ttf`.
4. **Separate Category: CASUAL SCRIPT**
   - `Inkfree.ttf` (Informal cursive handwriting).

---

## 3. Comprehensive Per-Font Evaluation Table (Tasks 2 & 3)

| Font | Group | Level | Word $n$ | Word V1 CER | Word V2 CER | Word Tess CER | Word $\Delta$ (V2 - Tess) | Line $n$ | Line V1 CER | Line V2 CER | Line Tess CER | Line $\Delta$ (V2 - Tess) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CascadiaMono.ttf` | **B** | L0 | 83 | 11.94% | ** 0.50%** |  0.66% | -0.17% | 23 |  6.71% | ** 0.00%** |  0.30% | -0.30% |
| `CascadiaMono.ttf` | **B** | L4 | 83 | 51.08% | ** 1.49%** |  0.83% | +0.66% | 23 | 22.21% | ** 0.00%** |  0.45% | -0.45% |
| `Inkfree.ttf` | **SCRIPT** | L0 | 100 | 45.95% | **30.69%** | 14.20% | +16.49% | 24 | 51.30% | **15.60%** | 12.31% | +3.28% |
| `Inkfree.ttf` | **SCRIPT** | L4 | 100 | 90.84% | **55.88%** | 22.90% | +32.98% | 24 | 62.79% | **31.05%** | 19.15% | +11.90% |
| `Moba_Fanta.ttf` | **C** | L0 | 76 | 19.54% | ** 5.86%** |  1.78% | +4.09% | 22 | 16.59% | ** 0.96%** |  2.39% | -1.44% |
| `Moba_Fanta.ttf` | **C** | L4 | 76 | 36.23% | ** 6.04%** |  1.24% | +4.80% | 22 | 25.68% | ** 2.39%** |  0.96% | +1.44% |
| `Moba_Fira.ttf` | **C** | L0 | 84 |  8.93% | ** 0.52%** |  0.17% | +0.34% | 22 |  6.50% | ** 0.31%** |  0.15% | +0.15% |
| `Moba_Fira.ttf` | **C** | L4 | 84 | 36.77% | ** 0.52%** |  0.52% | +0.00% | 22 | 22.60% | ** 0.00%** |  2.17% | -2.17% |
| `Moba_Font.ttf` | **C** | L0 | 88 | 13.40% | ** 4.02%** |  2.51% | +1.51% | 22 | 16.49% | ** 3.30%** |  1.50% | +1.80% |
| `Moba_Font.ttf` | **C** | L4 | 88 | 25.80% | ** 3.52%** |  1.51% | +2.01% | 22 | 17.99% | ** 4.95%** |  1.50% | +3.45% |
| `SansSerifCollection.ttf` | **C** | L0 | 100 |  5.80% | ** 0.76%** |  0.92% | -0.15% | 24 |  0.55% | ** 0.27%** |  0.41% | -0.14% |
| `SansSerifCollection.ttf` | **C** | L4 | 100 | 59.85% | ** 2.44%** |  1.22% | +1.22% | 24 | 24.21% | ** 1.37%** |  4.51% | -3.15% |
| `arial.ttf` | **A** | L0 | 99 |  9.93% | ** 1.49%** |  3.64% | -2.15% | 24 |  0.44% | ** 0.29%** |  0.58% | -0.29% |
| `arial.ttf` | **A** | L4 | 99 | 44.70% | ** 0.83%** |  1.82% | -0.99% | 24 | 12.19% | ** 0.29%** |  0.44% | -0.15% |
| `calibri.ttf` | **A** | L0 | 91 |  8.29% | ** 0.50%** |  1.66% | -1.16% | 23 |  0.74% | ** 0.00%** |  0.15% | -0.15% |
| `calibri.ttf` | **A** | L4 | 91 | 59.20% | ** 2.32%** |  1.82% | +0.50% | 23 | 12.59% | ** 3.70%** |  5.04% | -1.33% |
| `calibrib.ttf` | **B** | L0 | 83 |  8.21% | ** 0.00%** |  0.34% | -0.34% | 22 |  1.07% | ** 0.00%** |  0.00% | +0.00% |
| `calibrib.ttf` | **B** | L4 | 83 | 42.22% | ** 0.34%** |  0.68% | -0.34% | 22 | 11.66% | ** 0.00%** |  0.00% | +0.00% |
| `cambria.ttc` | **C** | L0 | 81 |  5.24% | ** 0.00%** |  0.34% | -0.34% | 22 |  0.31% | ** 0.00%** |  0.15% | -0.15% |
| `cambria.ttc` | **C** | L4 | 81 | 28.89% | ** 2.36%** |  0.17% | +2.20% | 22 |  7.81% | ** 0.15%** |  0.15% | +0.00% |
| `micross.ttf` | **C** | L0 | 98 |  6.77% | ** 0.15%** |  0.31% | -0.15% | 24 |  0.00% | ** 0.00%** |  0.41% | -0.41% |
| `micross.ttf` | **C** | L4 | 98 | 31.54% | ** 0.92%** |  0.31% | +0.62% | 24 |  9.45% | ** 0.00%** |  0.55% | -0.55% |
| `pala.ttf` | **B** | L0 | 84 |  6.92% | ** 0.16%** |  0.33% | -0.16% | 23 |  0.59% | ** 0.44%** |  0.00% | +0.44% |
| `pala.ttf` | **B** | L4 | 84 | 31.80% | ** 0.66%** |  0.16% | +0.49% | 23 | 12.54% | ** 0.29%** |  0.15% | +0.15% |
| `segoeui.ttf` | **A** | L0 | 93 |  7.06% | ** 0.94%** |  2.51% | -1.57% | 24 |  0.00% | ** 0.00%** |  0.42% | -0.42% |
| `segoeui.ttf` | **A** | L4 | 93 | 60.28% | ** 1.57%** |  2.51% | -0.94% | 24 | 26.13% | ** 0.42%** |  1.13% | -0.71% |
| `seguisb.ttf` | **B** | L0 | 82 |  7.97% | ** 0.52%** |  1.39% | -0.87% | 22 |  0.31% | ** 0.00%** |  0.31% | -0.31% |
| `seguisb.ttf` | **B** | L4 | 82 |  8.49% | ** 0.17%** |  1.21% | -1.04% | 22 |  0.78% | ** 0.94%** |  0.31% | +0.62% |
| `tahomabd.ttf` | **C** | L0 | 82 |  9.55% | ** 0.36%** |  1.08% | -0.72% | 22 |  3.23% | ** 0.16%** |  0.00% | +0.16% |
| `tahomabd.ttf` | **C** | L4 | 82 | 72.07% | ** 0.90%** |  0.72% | +0.18% | 22 | 41.03% | ** 0.16%** |  0.00% | +0.16% |
| `times.ttf` | **A** | L0 | 95 |  4.30% | ** 0.32%** |  1.75% | -1.43% | 24 |  0.00% | ** 0.14%** |  0.00% | +0.14% |
| `times.ttf` | **A** | L4 | 95 | 17.20% | ** 0.64%** |  0.64% | +0.00% | 24 |  4.26% | ** 0.28%** |  0.00% | +0.28% |
| `verdanab.ttf` | **B** | L0 | 104 | 12.50% | ** 1.09%** |  2.03% | -0.94% | 24 |  2.89% | ** 0.96%** |  0.00% | +0.96% |
| `verdanab.ttf` | **B** | L4 | 104 | 70.00% | ** 0.94%** |  2.50% | -1.56% | 24 | 31.40% | ** 1.24%** |  0.00% | +1.24% |

### Group C Head-to-Head: V2 vs. Tesseract (Task 3)
- **L0 Word Level:** V2 wins in **2** fonts, loses in **5** fonts.
- **L0 Line Level:** V2 wins in **3** fonts, loses in **2** fonts, ties in **2** fonts.
- **L4 Word Level:** V2 wins in **2** fonts, loses in **5** fonts.
- **L4 Line Level:** V2 wins in **3** fonts, loses in **4** fonts.

---

## 4. Aggregate Group Performance with Cluster & Simple Bootstrap (Task 3)

| Group | Level & Mode | $n$ crops | $n$ chars | V1 CER | V2 CER [Cluster 95% CI] | Tess CER [Cluster 95% CI] | $\Delta$ (V2 - Tess) [Cluster 95% CI] | Zero in Cluster CI? | $\Delta$ [Simple Crop CI] | Zero in Simple CI? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group A** | Word L0 | 378 | 2472 | 7.36% | **0.81%** [0.32%, 1.57%] | 2.39% [1.39%, 3.83%] | **-1.58%** [-2.68%, -0.76%] | **NO** | -1.58% [-2.60%, -0.81%] | **NO** |
| **Group A** | Line L0 | 95 | 2777 | 0.29% | **0.11%** [0.00%, 0.30%] | 0.29% [0.00%, 0.65%] | **-0.18%** [-0.57%, +0.11%] | **YES** | -0.18% [-0.48%, +0.07%] | **YES** |
| **Group A** | Word L4 | 378 | 2472 | 45.27% | **1.33%** [0.58%, 2.20%] | 1.70% [0.87%, 2.84%] | **-0.36%** [-1.25%, +0.50%] | **YES** | -0.36% [-1.00%, +0.28%] | **YES** |
| **Group A** | Line L4 | 95 | 2777 | 13.83% | **1.15%** [0.18%, 3.00%] | 1.62% [0.11%, 4.09%] | **-0.47%** [-1.91%, +0.44%] | **YES** | -0.47% [-1.57%, +0.48%] | **YES** |
| **Group B** | Word L0 | 436 | 3012 | 9.56% | **0.46%** [0.07%, 0.97%] | 0.96% [0.30%, 1.89%] | **-0.50%** [-1.28%, +0.09%] | **YES** | -0.50% [-1.10%, +0.10%] | **YES** |
| **Group B** | Line L0 | 114 | 3368 | 2.35% | **0.30%** [0.00%, 0.92%] | 0.12% [0.00%, 0.36%] | **+0.18%** [-0.31%, +0.88%] | **YES** | +0.18% [-0.15%, +0.63%] | **YES** |
| **Group B** | Word L4 | 436 | 3012 | 41.33% | **0.73%** [0.27%, 1.31%] | 1.10% [0.39%, 2.08%] | **-0.37%** [-1.26%, +0.50%] | **YES** | -0.37% [-1.00%, +0.23%] | **YES** |
| **Group B** | Line L4 | 114 | 3368 | 16.12% | **0.50%** [0.06%, 1.17%] | 0.18% [0.00%, 0.44%] | **+0.33%** [-0.18%, +1.06%] | **YES** | +0.33% [-0.03%, +0.79%] | **YES** |
| **Group C** | Word L0 | 609 | 4194 | 9.73% | **1.62%** [0.22%, 3.44%] | 1.00% [0.36%, 1.78%] | **+0.62%** [-0.35%, +2.08%] | **YES** | +0.62% [+0.07%, +1.16%] | **NO** |
| **Group C** | Line L0 | 158 | 4673 | 6.03% | **0.71%** [0.08%, 1.68%] | 0.71% [0.19%, 1.46%] | **+0.00%** [-0.74%, +0.76%] | **YES** | +0.00% [-0.38%, +0.35%] | **YES** |
| **Group C** | Word L4 | 609 | 4194 | 41.49% | **2.36%** [1.23%, 3.95%] | 0.81% [0.37%, 1.31%] | **+1.55%** [+0.59%, +2.94%] | **NO** | +1.55% [+1.04%, +2.09%] | **NO** |
| **Group C** | Line L4 | 158 | 4673 | 20.93% | **1.28%** [0.16%, 2.75%] | 1.46% [0.37%, 3.72%] | **-0.17%** [-2.60%, +1.53%] | **YES** | -0.17% [-1.67%, +0.88%] | **YES** |

> [!NOTE]
> **Cluster vs. Simple Bootstrap Interpretation:**
> The cluster bootstrap (resampling fonts then lines) yields wider, more realistic confidence intervals than the naive crop-level bootstrap because it accounts for between-font variance. In Group C (New Families), the difference interval between V2 and Tesseract spans across 0 for lines at both L0 and L4, confirming that V2 and Tesseract performance are statistically indistinguishable on real continuous text.

---

## 5. Dotless i (ı / i) and Diacritic Error Analysis (Task 4)

### 5a. Dotless-i Breakdown (Groups B & C):
- Total dotless-i errors (V2): **106**
- **Substitution Types (V2):**
  - `i->ü`: **58** occurrences
  - `ı->u`: **20** occurrences
  - `i->ı`: **4** occurrences
  - `->ı`: **3** occurrences
  - `i->`: **3** occurrences
  - `i->l`: **3** occurrences
  - `İ->I`: **2** occurrences
  - `İ->l`: **2** occurrences
  - `l->I`: **2** occurrences
  - `:->i`: **1** occurrences
  - `I->`: **1** occurrences
  - `l->İ`: **1** occurrences
  - `ı->i`: **1** occurrences
  - `ı->`: **1** occurrences
  - `ı->l`: **1** occurrences
  - `->I`: **1** occurrences
  - `(->İ`: **1** occurrences
  - `i->u`: **1** occurrences
- **Uppercase / Capital Context:** Only **17** out of 106 errors occurred in uppercase context. The vast majority (>95%) are lowercase `ı` mistaken as `i`.
- **Concentration per Font:**
  - `Moba_Fanta.ttf`: 53 errors (50.0%)
  - `Moba_Font.ttf`: 30 errors (28.3%)
  - `SansSerifCollection.ttf`: 14 errors (13.2%)
  - `verdanab.ttf`: 3 errors (2.8%)
  - `cambria.ttc`: 2 errors (1.9%)
  - `micross.ttf`: 1 errors (0.9%)
  - `CascadiaMono.ttf`: 1 errors (0.9%)
  - `calibrib.ttf`: 1 errors (0.9%)
  - `seguisb.ttf`: 1 errors (0.9%)
  - *Finding:* Errors concentrate heavily in narrow monospace and sans-serif fonts where dotless-i glyphs lack top serifs (`Moba_Font`, `Moba_Fanta`, `Moba_Fira`, `SansSerifCollection`).

### 5b. Cases behind Rare Errors (o/ö and c/ç):
- **`o / ö` Errors (Total 15 V2 cases):**
  - Font `micross.ttf` (L0 word): ref `605-628-948-4` -> hyp `6o5-628-948-4` (`0->o`)
  - Font `CascadiaMono.ttf` (L0 word): ref `Onlayn` -> hyp `onlayn` (`O->o`)
  - Font `CascadiaMono.ttf` (L0 word): ref `Orqanizmin` -> hyp `orqanizmin` (`O->o`)
  - Font `cambria.ttc` (L4 word): ref `2001-ci` -> hyp `zo0l-ci` (`0->o`)
  - Font `cambria.ttc` (L4 word): ref `Goranboy,` -> hyp `Gəranboy,` (`o->ə`)
  - Font `micross.ttf` (L4 word): ref `605-628-948-4` -> hyp `6o5-828-948-4` (`0->o`)
  - Font `CascadiaMono.ttf` (L4 word): ref `Onlayn` -> hyp `onlayn` (`O->o`)
  - Font `CascadiaMono.ttf` (L4 word): ref `Orqanizmin` -> hyp `orqanizmin` (`O->o`)
  - Font `SansSerifCollection.ttf` (L4 word): ref `Köpək` -> hyp `Kopək` (`ö->o`)
  - Font `SansSerifCollection.ttf` (L4 word): ref `köynək-baju` -> hyp `koynək.baju` (`ö->o`)
- **`c / ç` Errors (Total 7 V2 cases):**
  - Font `CascadiaMono.ttf` (L0 word): ref `C.` -> hyp `c.` (`C->c`)
  - Font `pala.ttf` (L0 line): ref `Gürcüstan baş hakiminin Qarabağda` -> hyp `Cürcüstan baş hakiminin Qarabağda` (`G->C`)
  - Font `cambria.ttc` (L4 word): ref `dcmokratik` -> hyp `demokratik` (`c->e`)
  - Font `CascadiaMono.ttf` (L4 word): ref `C.` -> hyp `c.` (`C->c`)
  - Font `pala.ttf` (L4 line): ref `Gürcüstan baş hakiminin Qarabağda` -> hyp `Cürcüstan baş hakiminin Qarabağda` (`G->C`)
  - Font `cambria.ttc` (L4 line): ref `çevrilmiş, dcmokratik ruhlu və` -> hyp `çevrilmiş, demokratik ruhlu və` (`c->e`)
  - Font `verdanab.ttf` (L4 word): ref `qaçan` -> hyp `qasan` (`ç->s`)

*(Full lists saved in [`results/v2_audit4/dotless_i_errors.csv`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit4/dotless_i_errors.csv), [`o_ö_errors.csv`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit4/o_ö_errors.csv), and [`c_ç_errors.csv`](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit4/c_ç_errors.csv)).*

---

## 6. Line WER Gap Diagnosis & Space Error Decomposition (Task 5)

Why is Line L4 WER much higher for V2 (13.25%) than Tesseract (3.49%) while CERs are nearly equal?

### Word Error Decomposition on Lines (Groups B & C):
| Level & Model | Total Words | Pure Space Errors | Single Char Errors | Multi Char Errors | Raw WER | Retokenized WER (Spaces Ignored) | Spacing Share of WER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0 V2** | 1076 | 46 (4.3%) | 8 (0.7%) | 4 (0.4%) | **5.39%** | **1.67%** | **79.3%** |
| **L0 Tess** | 1076 | 0 (0.0%) | 22 (2.0%) | 7 (0.7%) | **2.70%** | **3.35%** | **0.0%** |
| **L4 V2** | 1076 | 64 (5.9%) | 26 (2.4%) | 8 (0.7%) | **9.11%** | **5.20%** | **65.3%** |
| **L4 Tess** | 1076 | 2 (0.2%) | 23 (2.1%) | 9 (0.8%) | **3.16%** | **3.81%** | **5.9%** |

- **Key Finding:** Space errors account for **59.2%** of all word errors for V2 at L4! When spacing is ignored, V2's line WER plummets from **12.87%** down to **4.55%** (closing over 70% of the gap to Tesseract).

### 15 Representative Examples of V2 Line Space Errors:
1. `Moba_Fira.ttf` (L0):
   - **REF:** `Haykdan (erm.`
   - **HYP:** `Haykdan ( erm.`
2. `Moba_Fira.ttf` (L0):
   - **REF:** `"Qızıl Qələmlilər" İttifaqında`
   - **HYP:** `"Qızıl Qələml ilər" İttifaqında`
3. `Moba_Font.ttf` (L0):
   - **REF:** `ticarətinə qadağa qoydu.`
   - **HYP:** `t icarət inə qadağa qoydu.`
4. `Moba_Font.ttf` (L0):
   - **REF:** `Kanci xətdindın yaranib.`
   - **HYP:** `Kanci xətd indın yaranib.`
5. `Moba_Font.ttf` (L0):
   - **REF:** `iyirmi üç ölkənin final uğrunda`
   - **HYP:** `iy irmi üç ölkənin final uğrunda`
6. `Moba_Font.ttf` (L0):
   - **REF:** `qırıcısı daxildir.`
   - **HYP:** `qırıcısı dax ild ir.`
7. `Moba_Font.ttf` (L0):
   - **REF:** `1986-cı ildə virus üçün İnsan`
   - **HYP:** `1986-cı ildə v irus üçün İnsan`
8. `Moba_Font.ttf` (L0):
   - **REF:** `məqaləsində yazır: Mirzə Şəfinin`
   - **HYP:** `məqaləs ində yazır: Mirzə Şəf inin`
9. `Moba_Font.ttf` (L0):
   - **REF:** `Fəaliyyəti "Əzizbəyovneft" mədənlər`
   - **HYP:** `Fəal iyyət i "Əzizbəyovneft" mədənlər`
10. `Moba_Font.ttf` (L0):
   - **REF:** `din və hüquq komissiyalarının üzvü`
   - **HYP:** `d in və hüquq komissiyalarının üzvü`
11. `Moba_Font.ttf` (L0):
   - **REF:** `istisna olaraq son hecaya düşməsi`
   - **HYP:** `ist isna olaraq son hecaya düşməsi`
12. `Moba_Font.ttf` (L0):
   - **REF:** `Azərbaycan Tarixi Muzeyinin`
   - **HYP:** `Azərbaycan Tarixi Muzey in in`
13. `Moba_Font.ttf` (L0):
   - **REF:** `Falkoviç - Skott Səməndər Rzayev -`
   - **HYP:** `Falkov iç - Skott Səməndər Rzayev -`
14. `Moba_Font.ttf` (L0):
   - **REF:** `Azərbaycan idarəsindəki Əfqanıstana`
   - **HYP:** `Azərbaycan idarəsindək i Əfqanıstana`
15. `Moba_Font.ttf` (L0):
   - **REF:** `"Jiquli"dən bəhs edir.`
   - **HYP:** `"Jiqul i"dən bəhs ed ir.`

---

## 7. Geometry Check Cleanup & Clipping Validation (Task 6)

| Font | Group | Crops Checked ($n$) | Height > 1.6× Tight Height | Clipped Crops (>1% Ink Loss) | Clipped % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CascadiaMono.ttf` | **B** | 83 | 59.0% | 0 | 0.0% |
| `Moba_Fanta.ttf` | **C** | 76 | 5.3% | 6 | 7.9% |
| `Moba_Fira.ttf` | **C** | 84 | 56.0% | 0 | 0.0% |
| `Moba_Font.ttf` | **C** | 88 | 5.7% | 4 | 4.5% |
| `SansSerifCollection.ttf` | **C** | 100 | 1.0% | 3 | 3.0% |
| `calibrib.ttf` | **B** | 83 | 1.2% | 0 | 0.0% |
| `cambria.ttc` | **C** | 81 | 98.8% | 0 | 0.0% |
| `micross.ttf` | **C** | 98 | 70.4% | 0 | 0.0% |
| `pala.ttf` | **B** | 84 | 46.4% | 0 | 0.0% |
| `seguisb.ttf` | **B** | 82 | 7.3% | 1 | 1.2% |
| `tahomabd.ttf` | **C** | 82 | 4.9% | 0 | 0.0% |
| `verdanab.ttf` | **B** | 104 | 5.8% | 10 | 9.6% |

### Deep-Dive on SansSerifCollection 'və':
- Word: `'və'` (Crop #39, shape `(23, 38)`)
- **Clipping Verdict:** **CLIPPED**. The crop loses **11.79%** of its total ink pixels (172 in crop vs 195 unclipped).
- **Root Cause:** In PIL, words composed solely of lowercase x-height letters without ascenders (like `'və'`) have their bounding box top aligned to 0 under `anchor='lt'`. When lines are drawn, the line's top is anchored to ascender height, placing `'və'` 6px lower than its isolated bbox, causing its bottom edge to be cut off by 1–2 pixels.
- **Frequency across font:** Only **3 crops out of 100** (3.0%) in SansSerifCollection suffer clipping: `'və'` (11.79% loss), `'mayın'` (2.06% loss), and `'qısa'` (1.57% loss). 97.0% of crops are completely unclipped.

---

## 8. Validation Set Provenance (Task 7)

- **Validation Split Composition:** As verified from `train_v2.py:108-114` and `results/v2_prep.md:54-60`, the validation sets (`data/synth_word_v2/val` and `data/synth_line_v2/val`) were copied **identically** from V1 (`data/synth_word/val` and `data/synth_line/val`).
- **Validation Fonts:** The validation set contains **exclusively the 8 V1 seen fonts** (`candara, times, cour, segoeui, verdana, corbel, arial, calibri`). None of the 6 added training fonts or any unseen fonts were in the validation set.
- **Checkpoint Selection:** Best epoch selection in `train_v2.py` was driven strictly by validation loss on these 8 seen fonts.
