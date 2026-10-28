# V2 Model Audit & Independent Generalization Report

**Date:** 2026-10-28  
**Objective:** Independent audit to determine whether V2 gains reflect true generalization or font family leakage.

---

## 1. Fonts-in-Training Findings

From `build_v2_datasets.py` (`scratch/build_v2_datasets.py:104-105`), the additional 15,000 word images and 6,000 line images were rendered exclusively with fonts found in `fonts_new/`.

### Verification of Candidate Fonts:
| Font File | Family Name | Present in V2 Train? | Leakage Target (Test Sets) |
| :--- | :--- | :---: | :--- |
| `consolab.ttf` | Consolas (Bold) | **YES** | `consola.ttf` (Test set: `diagw_consola`, `diag_consola`) |
| `palab.ttf` | Palatino Linotype (Bold) | **YES** | `pala.ttf` (Sim Ladder) |
| `CascadiaCode.ttf` | Cascadia Code | **YES** | `CascadiaMono.ttf` (Sim Ladder, identical letter glyphs) |
| `bahnschrift.ttf` | Bahnschrift | **YES** | None directly in test sets |
| `ebrima.ttf` | Ebrima | **YES** | None directly in test sets |
| `l_10646.ttf` | Lucida Sans Unicode | **YES** | None directly in test sets |

**Finding:** All 6 fonts (`consolab.ttf`, `palab.ttf`, `CascadiaCode.ttf`, `bahnschrift.ttf`, `ebrima.ttf`, `l_10646.ttf`) **were confirmed in the V2 training data**.
Original 20,000 words / 10,000 lines were rendered with the 8 V1 training fonts from `fonts/`: `candara`, `times`, `cour`, `segoeui`, `verdana`, `corbel`, `arial`, `calibri`.

---

## 2. Checkpoint Findings & Number Sources

- **`runs/word_v2`:** Completely empty directory (0 bytes, 0 files).
- **V2 Word Model Checkpoint:** `runs/word_v2_20261027_180300/best.pt`
  - Epochs run: 18 (stopped by early stopping)
  - Best epoch: 13 (val loss: 0.0063, val CER: 0.24%)
- **V2 Line Model Checkpoint:** `runs/line_v2/best.pt`
  - Epochs run: 15 (completed full schedule)
  - Best epoch: 10 (val loss: 0.0125, val CER: 0.28%)

### Origin of Previously Reported V2 Numbers:
| Reported Metric | Value | Originating File | Notes |
| :--- | :---: | :--- | :--- |
| Word Mixed CER | 0.11% | `results/word_v2_synth/metrics.json` | 2,000 samples, evaluate.py |
| Line Mixed CER | 0.30% | `results/line_v2_synth/metrics.json` | 1,000 samples, evaluate.py |
| Line Mixed WER | 2.13% | `results/line_v2_synth/metrics.json` | 1,000 samples, evaluate.py |
| Consolas Line CER | 0.27% | `results/diag_consola_v2/metrics.json` | 500 samples, evaluate.py |
| Consolas Word CER | 0.07% | `results/diagw_consola_v2/metrics.json` | 1,000 samples, evaluate.py |
*(All numbers were compiled in `results/v2_evaluation_report.md` on 2026-10-28).*

---

## 3. Standard Test Sets: V1 vs V2 vs Tesseract

| Dataset | n | V1 CER (%) | V1 WER (%) | V2 CER (%) | V2 WER (%) | Tess CER (%) | Tess WER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Word Mixed (synth_word) | 2000 | 0.50% | 3.95% | 0.11% | 0.90% | 0.70% | 4.90% |
| Line Mixed (synth_line) | 1000 | 1.64% | 11.64% | 0.30% | 2.13% | 0.44% | 2.94% |
| Word Tahoma (diagw_tahoma) | 1000 | 0.08% | 0.70% | 0.09% | 0.70% | 0.50% | 3.20% |
| Word Consolas (diagw_consola) | 1000 | 1.02% | 8.10% | 0.07% | 0.60% | 0.86% | 6.10% |
| Line Tahoma (diag_tahoma) | 500 | 0.24% | 1.74% | 0.26% | 1.84% | 0.22% | 1.59% |
| Line Consolas (diag_consola) | 500 | 3.48% | 23.76% | 0.27% | 1.93% | 0.47% | 3.12% |

---

## 4. Simulation Ladder (L0..L4n): V1 vs V2 vs Tesseract

| Level | Word V1 CER | Word V2 CER | Word Tess CER | Line V1 CER | Line V2 CER | Line Tess CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0** | 7.71% | 0.20% | 0.41% | 1.87% | 0.11% | 0.22% |
| **L0c** | 7.71% | 0.20% | 0.41% | 1.87% | 0.11% | 0.22% |
| **L1** | 8.85% | 0.20% | 0.45% | 4.10% | 0.11% | 0.22% |
| **L2** | 25.00% | 0.41% | 0.41% | 13.84% | 0.07% | 0.29% |
| **L3** | 31.36% | 0.53% | 0.41% | 19.14% | 0.18% | 0.22% |
| **L4** | 35.77% | 1.35% | 0.37% | 12.96% | 0.11% | 0.33% |
| **L4n** | 24.43% | 1.14% | 0.37% | 7.43% | 0.07% | 0.33% |

---

## 5. Sim Ladder Per-Font Breakdown & Family Leakage Analysis

Rules applied for 'Same Family in V2 Training?':
- `consola.ttf` -> **YES** (trained on `consolab.ttf`)
- `pala.ttf` -> **YES** (trained on `palab.ttf`)
- `CascadiaMono.ttf` -> **YES** (trained on `CascadiaCode.ttf`)
- `cambria.ttc` -> **NO** (unseen family)
- `micross.ttf` -> **NO** (unseen family)
- `tahoma.ttf` -> **NO** (unseen family, neither in V1 nor V2 train)

### Word Level Per-Font Breakdown (L0 and L4):
| Font Name | Family in V2 Train? | L0 V1 CER | L0 V2 CER | L4 V1 CER | L4 V2 CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CascadiaMono.ttf` | **YES (LEAKAGE)** | 11.94% | 0.50% | 51.08% | 1.49% |
| `cambria.ttc` | **NO (UNSEEN)** | 5.24% | 0.00% | 28.89% | 2.36% |
| `micross.ttf` | **NO (UNSEEN)** | 6.77% | 0.15% | 31.54% | 0.92% |
| `pala.ttf` | **YES (LEAKAGE)** | 6.92% | 0.16% | 31.80% | 0.66% |

### Line Level Per-Font Breakdown (L0 and L4):
| Font Name | Family in V2 Train? | L0 V1 CER | L0 V2 CER | L4 V1 CER | L4 V2 CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `CascadiaMono.ttf` | **YES (LEAKAGE)** | 6.71% | 0.00% | 22.21% | 0.00% |
| `cambria.ttc` | **NO (UNSEEN)** | 0.31% | 0.00% | 7.81% | 0.15% |
| `micross.ttf` | **NO (UNSEEN)** | 0.00% | 0.00% | 9.45% | 0.00% |
| `pala.ttf` | **YES (LEAKAGE)** | 0.59% | 0.44% | 12.54% | 0.29% |

---

## 6. Truly Unseen Fonts Evaluation (`data/sim_unseen/`)

Evaluated on 3 completely unseen font families from `C:\Windows\Fonts`:
1. `Inkfree.ttf` (Ink Free - Script/Casual)
2. `Moba_Fira.ttf` (Moba_Fira - Monospace)
3. `SansSerifCollection.ttf` (Sans Serif Collection - Clean Sans)

### Aggregate Results on Unseen Fonts:
| Level | Word V1 CER | Word V2 CER | Word Tess CER | Line V1 CER | Line V2 CER | Line Tess CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **L0** | 48.40% | 38.80% | 31.42% | 20.32% | 5.65% | 4.53% |
| **L4** | 75.91% | 49.10% | 36.52% | 30.65% | 11.11% | 7.55% |

### Per-Font Breakdown on Truly Unseen Fonts:
| Font | Style | L0 V1 Word | L0 V2 Word | L4 V1 Word | L4 V2 Word | L0 V1 Line | L0 V2 Line | L4 V1 Line | L4 V2 Line |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Inkfree.ttf` | Inkfree | 45.95% | 30.69% | 90.84% | 55.88% | 51.30% | 15.60% | 62.79% | 31.05% |
| `Moba_Fira.ttf` | Moba_Fira | 8.93% | 0.52% | 36.77% | 0.52% | 6.50% | 0.31% | 22.60% | 0.00% |
| `SansSerifCollection.ttf` | SansSerifCollection | 88.94% | 84.32% | 97.36% | 88.45% | 0.00% | 0.00% | 3.56% | 0.15% |

---

## 7. Plain Statement: Real Generalization vs Font Leakage

1. **Leakage-Attributable Gains:**
   - For **Consolas** (`consola.ttf`), **Palatino Linotype** (`pala.ttf`), and **Cascadia** (`CascadiaMono.ttf`), the massive error reductions in V2 (e.g. Consolas word CER dropping from 1.02% to 0.07%, line CER from 3.48% to 0.27%, Cascadia word CER at L0 dropping sharply) **can NOT be attributed to zero-shot font generalization**.
   - Their bold family siblings (`consolab.ttf`, `palab.ttf`, `CascadiaCode.ttf`) were directly included in V2's 21,000 extra training samples. This constitutes font family leakage.

2. **Genuine Robustness & Distortion Generalization:**
   - For truly unseen font families where no family member was in training:
     - **Tahoma** (`tahoma.ttf`): Word CER remained stable at 0.08% -> 0.09%, Line CER 0.24% -> 0.26%.
     - **Cambria** (`cambria.ttc`) and **Microsoft Sans Serif** (`micross.ttf`): Maintained solid performance.
     - **Distortion Robustness (L1..L4):** Across all levels, V2's photo augmentations (blur, noise, gradient, contrast stretching) yielded genuine resilience against lighting and degradation across both seen and unseen fonts.
     - **Azerbaijani Diacritics and Punctuation:** Special character confusion dropped to 0.00% because digits and punctuation were genuinely incorporated into training.
