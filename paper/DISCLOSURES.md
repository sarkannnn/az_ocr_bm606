# Scientific Disclosures, Post-Hoc Choices, and Methodological Limitations (paper/DISCLOSURES.md)

This document provides a transparent, comprehensive accounting of all post-hoc decisions, parameter adjustments, data snooping risks, font family leakages, and contradictions discovered between initial assumptions and repository files.

---

### 1. Post-Hoc Parameter Tuning and Data Snooping Risks

1. **Heuristic Lexicon Post-Corrector (Test-Snooping Risk):**
   - *Disclosure:* In `scratch/test_safe_corrector.py`, a SymSpell-based dictionary and suffix stripper was implemented to correct OCR errors on Photo 5 (`IMG_5130`). The rule list, morphological suffixes, and exceptions (e.g., preserving proper nouns like `Taryel` and `Səməndər`, specifically handling `təlsil -> təhsil`, `vaxf -> vaxt`) were refined **after** inspecting the recognition output of Photo 5.
   - *Scientific Impact:* This constitutes classic test-set snooping (data snooping). Although it reduced Photo 5 CER from 2.10% to 1.80% (net gain of 7 characters: 8 true positive corrections, 1 false positive regression), this result cannot be considered an independent or generalizable benchmark.
   - *Resolution in Paper:* The lexicon corrector is quarantined as an experimental pilot study in Section 7 and Table 4. The primary benchmark result of the paper remains the frozen, unassisted **Raw V2 Line model (2.10% CER / 12.80% WER)**.

2. **Image Projection Chunking Parameters ($k_{\\text{size}} = 25$, 35th Percentile, +4 px Margin):**
   - *Disclosure:* The mathematical parameters for rule-based horizontal line chunking—projection smoothing kernel $k_{\\text{size}} = 25$, column ink threshold at the 35th percentile, binary ink threshold $\\text{gray} < 160$, and the $+4\\text{ px}$ bounding box padding—were iteratively tuned on pilot documents (Photo 1 and Photo 2) during development (`results/real_fix2` through `results/real_fix8`).
   - *Scientific Impact:* While these parameters were strictly frozen before evaluating Photo 3, Photo 4, and Photo 5, their initial calibration was empirical rather than derived from first principles.
   - *Resolution in Paper:* Fully disclosed in Section 5 and Section 6.6 as development-set calibration.

3. **Manual Bounding Box Adjustment on Photo 5 Line 12:**
   - *Disclosure:* Across all 33 lines of Photo 5, 32 lines were segmented completely automatically by Tesseract `image_to_data` grouping. However, on Line 12 (0-indexed line 11), a printer roller toner smudge at $x=395$ px on the left margin interfered with text segmentation. The left coordinate was manually adjusted from $x_0 = 395$ to $x_0 = 430$ to exclude the artifact.
   - *Resolution in Paper:* Explicitly disclosed in Section 6.5 and `results/real_fix9/report.md:18`.

4. **Squeeze Ratio Threshold Calibration ($r = 1.0$ and $r = 1.3$):**
   - *Disclosure:* The decision to chunk lines exceeding an aspect ratio of $r = (W \\cdot 32) / (h \\cdot 512) \\ge 1.1$ and the identification of severe distortion at $r > 1.3$ were made after analyzing the width-to-height distributions of lines in Photo 5.

---

### 2. Font Family Leakage and Generalization Audits

1. **V2 Training Additions (`fonts_new/`):**
   - *Disclosure:* To improve robustness, V2 added 6 font files (15,000 words and 6,000 lines). An independent audit (`results/v2_audit/summary.md`) revealed that:
     - `consolab.ttf` (Consolas Bold) belongs to the same family as test font `consola.ttf` (Consolas Regular).
     - `palab.ttf` (Palatino Linotype Bold) belongs to the same family as ladder test font `pala.ttf`.
     - `CascadiaCode.ttf` shares identical letter glyph outlines with `CascadiaMono.ttf`.
   - *Scientific Impact:* The dramatic reduction in Consolas CER from 1.02% down to 0.07% (words) and 3.48% down to 0.27% (lines) is **not zero-shot font generalization**. It is direct font family leakage.
   - *Resolution in Paper:* Consolas is strictly classified as "Family Leakage" (Group B), while Tahoma is classified as "New Family" (Group C). Generalization claims are restricted strictly to truly unseen font families (`Cambria`, `Microsoft Sans Serif`, `Moba_Fira`, `Moba_Font`, `Moba_Fanta`, `tahomabd`, and fixed `SansSerifCollection`).

2. **Validation Set Composition:**
   - *Disclosure:* Validation sets for both V1 and V2 (`data/synth_word_v2/val` and `data/synth_line_v2/val`) contained **exclusively the 8 seen training fonts** of V1. Early stopping was therefore guided solely by validation loss on seen fonts and could not detect font over-fitting.

---

### 3. Tesseract Integration vs. Recognition Breakdown

1. **PSM 7 Empty Output Artifact:**
   - *Disclosure:* On Photo 5, Tesseract 5 (`aze`) in `--psm 7` mode returned completely empty strings (`""`) for 9 chunks and 2 full lines, causing an inflated CER of 13.02%.
   - *Scientific Impact:* Comparing V2 (2.10% CER) directly against Tesseract (13.02% CER) would be scientifically misleading because the error is an integration/baseline detection failure, not a character recognition defect. When the exact same failed crops were evaluated under `--psm 13`, Tesseract recognized them with near-zero error.
   - *Resolution in Paper:* Both Condition A (all 33 lines) and a Clean Subset (23 lines with zero empty outputs) are reported with equal prominence. On the Clean Subset, V2 CER is 1.32% and Tesseract CER is 1.26% ($\Delta = +0.06\%$, 95% CI: [-0.77%, +0.80%], including zero).

---

### 4. Contradictions Between Initial Assumptions and Repository Data

| Item / Assumption | Initial Assumption | Actual File Record | Resolution |
| :--- | :--- | :--- | :--- |
| **"V2 beats Tesseract on real photos"** | V2 is claimed to beat Tesseract across real documents | On full lines, Tesseract is substantially ahead (P1: 1.32% vs 53.61%, P2: 0.14% vs 36.43%, P3: 0.00% vs 0.20%, P4: 0.00% vs 0.10%, P5 psm13: 0.09% vs 3.87%). On trimmed chunks, V2 and Tesseract are statistically tied (P1: 0.97% vs 1.32%, P5 clean: 1.32% vs 1.26%). | Corrected: Tesseract remains superior on full lines and equal on clean chunks; V2 only exceeds Tesseract when Tesseract's layout segmentation fails. |
| **Training Duration** | Single duration assumed | `best.pt` metadata records 212.5 min (word) and 148.1 min (line) because timer saved at best epoch (epoch 12). Full run logs record 301.3 min (word) and 184.6 min (line). | Disclosed in Table 3: both checkpoint epoch timestamp and full training log duration are reported. |
| **Simulation Ladder Monotonicity** | Distortions strictly increase error | V1 Line CER was 19.14% at L3 but dropped to 12.96% at L4. | Disclosed in Section 6.3 as non-monotonic behavior (cause not investigated). |
| **SansSerifCollection Initial Error** | Model recognition defect | The 84.32% CER was caused by PIL font metric bounding box overlap ($ascent+descent=80$ px). | Disclosed in Section 8 as a geometry bug resolved by tight ink bounding box tracking. |
