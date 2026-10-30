import os
import sys
import json
import csv
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

TBL_DIR = Path("paper/tables")
TBL_DIR.mkdir(parents=True, exist_ok=True)

print("Compiling publication tables...")

# ------------------------------------------------------------------------------
# Table 1: Dataset Sizes and Splits
# ------------------------------------------------------------------------------
t1_md = """# Table 1: Dataset Inventory, Splits, and Sample Sizes

| Dataset Name | Split | Modality | Sample Count ($n$) | Character Count | Purpose / Composition | Source File |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **corpus_word** | Train | Text | 20,000 | 143,892 | Wikipedia az filtered word types | `data/corpus_word/train.txt` |
| **corpus_word** | Val | Text | 2,500 | 17,986 | Held-out validation word types | `data/corpus_word/val.txt` |
| **corpus_word** | Test | Text | 2,500 | 18,104 | Held-out test word types | `data/corpus_word/test.txt` |
| **corpus_line** | Train | Text | 10,400 | 284,529 | Wikipedia az sentence chunks | `data/corpus_line/train.txt` |
| **corpus_line** | Val | Text | 95,659 | 2,612,408 | Large corpus sentence pool | `data/corpus_line/val.txt` |
| **corpus_line** | Test | Text | 95,906 | 2,619,102 | Large corpus sentence pool | `data/corpus_line/test.txt` |
| **synth_word (V1)** | Train | Image | 20,000 | 134,892 | 8 standard fonts, mild aug | `data/synth_word/train/labels.tsv` |
| **synth_word (V1)** | Val | Image | 2,000 | 13,490 | 8 standard fonts, clean | `data/synth_word/val/labels.tsv` |
| **synth_word (V1)** | Test | Image | 2,000 | 13,501 | 2 held-out fonts (Tahoma, Consolas) | `data/synth_word/test/labels.tsv` |
| **synth_line (V1)** | Train | Image | 10,000 | 242,105 | 8 standard fonts, mild aug | `data/synth_line/train/labels.tsv` |
| **synth_line (V1)** | Val | Image | 1,000 | 24,198 | 8 standard fonts, clean | `data/synth_line/val/labels.tsv` |
| **synth_line (V1)** | Test | Image | 1,000 | 24,211 | 2 held-out fonts (Tahoma, Consolas) | `data/synth_line/test/labels.tsv` |
| **synth_word_v2** | Train | Image | 35,000 | 239,812 | 20k V1 + 15k fonts_new additions | `data/synth_word_v2/train/labels.tsv` |
| **synth_word_v2** | Val | Image | 2,000 | 13,490 | Identical copy of V1 val | `data/synth_word_v2/val/labels.tsv` |
| **synth_word_v2** | Test | Image | 2,000 | 13,501 | Identical copy of V1 test | `data/synth_word_v2/test/labels.tsv` |
| **synth_line_v2** | Train | Image | 16,000 | 388,419 | 10k V1 + 6k fonts_new additions | `data/synth_line_v2/train/labels.tsv` |
| **synth_line_v2** | Val | Image | 1,000 | 24,198 | Identical copy of V1 val | `data/synth_line_v2/val/labels.tsv` |
| **synth_line_v2** | Test | Image | 1,000 | 24,211 | Identical copy of V1 test | `data/synth_line_v2/test/labels.tsv` |
| **diagw_tahoma** | Test | Image | 1,000 | 6,750 | Monofont Tahoma word test | `data/diagw_tahoma/test/labels.tsv` |
| **diag_tahoma** | Test | Image | 500 | 12,105 | Monofont Tahoma line test | `data/diag_tahoma/test/labels.tsv` |
| **diagw_consola** | Test | Image | 1,000 | 6,742 | Monofont Consolas word test | `data/diagw_consola/test/labels.tsv` |
| **diag_consola** | Test | Image | 500 | 12,098 | Monofont Consolas line test | `data/diag_consola/test/labels.tsv` |
| **sim_ladder** | L0-L4n | Image | 346 words | 2,342 | 4 pages, 7 degradation steps | `data/sim_ladder/metadata.json` |
| **sim_ladder** | L0-L4n | Image | 92 lines | 2,510 | 4 pages, 7 degradation steps | `data/sim_ladder/metadata.json` |
| **sim_audit3/4 (Grp C)** | L0/L4 | Image | 609 words | 4,194 | 7 new font families | `results/v2_audit4/summary.md:128` |
| **sim_audit3/4 (Grp C)** | L0/L4 | Image | 158 lines | 4,673 | 7 new font families | `results/v2_audit4/summary.md:129` |
| **Photo 1 (TNR)** | Test | Real Photo | 16 lines | 1,442 | Real document photo (200 dpi) | `results/real_fix8/final_tables.md:9` |
| **Photo 2 (Calibri)** | Test | Real Photo | 16 lines | 1,463 | Real document photo (200 dpi) | `results/real_fix8/final_tables.md:15` |
| **Photo 3 (TNR)** | Test | Real Photo | 23 lines | 1,006 | Clean print document photo | `results/photo3_test/report.md:7` |
| **Photo 4 (Arial)** | Test | Real Photo | 23 lines | 1,006 | Sans-serif document photo | `results/photo4_sans/report.md:7` |
| **Photo 5 (IMG_5130)** | Test | Real Photo | 33 lines | 2,328 | 5 paragraphs, smartphone photo | `results/real_fix9/metrics.json:9` |

Source: data/corpus_word/, data/synth_word/, data/synth_line/, results/v2_audit4/summary.md, results/real_fix8/, results/real_fix9/
"""
(TBL_DIR / "table1_dataset_sizes.md").write_text(t1_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 2: Fonts Table
# ------------------------------------------------------------------------------
t2_md = """# Table 2: Font Inventory, Training Provenance, and Taxonomic Classification

| Font Filename | Family Name | Style / Weight | In V1 Train? | In V2 Train? | Audit Classification | Reason / Status | Source File |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `arial.ttf` | Arial | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/arial.ttf` |
| `calibri.ttf` | Calibri | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/calibri.ttf` |
| `candara.ttf` | Candara | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/candara.ttf` |
| `corbel.ttf` | Corbel | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/corbel.ttf` |
| `cour.ttf` | Courier New | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/cour.ttf` |
| `segoeui.ttf` | Segoe UI | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/segoeui.ttf` |
| `times.ttf` | Times New Roman | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/times.ttf` |
| `verdana.ttf` | Verdana | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/verdana.ttf` |
| `tahoma.ttf` | Tahoma | Regular | NO | NO | Group C: NEW FAMILY | Reserved for V1/V2 test set (20% held-out) | `fonts/tahoma.ttf` |
| `consola.ttf` | Consolas | Regular | NO | NO | Group B: FAMILY LEAKAGE | Test font, but bold sibling `consolab` in V2 | `fonts/consola.ttf` |
| `CascadiaCode.ttf` | Cascadia Code | Monospace | NO | **YES** | V2 Train Addition | Added in V2 (+15k words/+6k lines) | `fonts_new/CascadiaCode.ttf` |
| `bahnschrift.ttf` | Bahnschrift | Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/bahnschrift.ttf` |
| `consolab.ttf` | Consolas | Bold | NO | **YES** | V2 Train Addition | Causes leakage into `consola.ttf` test set | `fonts_new/consolab.ttf` |
| `ebrima.ttf` | Ebrima | Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/ebrima.ttf` |
| `l_10646.ttf` | Lucida Sans Unicode| Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/l_10646.ttf` |
| `palab.ttf` | Palatino Linotype | Bold | NO | **YES** | V2 Train Addition | Causes leakage into `pala.ttf` ladder test | `fonts_new/palab.ttf` |
| `calibrib.ttf` | Calibri | Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `calibri.ttf` | `C:/Windows/Fonts/calibrib.ttf` |
| `seguisb.ttf` | Segoe UI | Semi-Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `segoeui.ttf` | `C:/Windows/Fonts/seguisb.ttf` |
| `verdanab.ttf` | Verdana | Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `verdana.ttf` | `C:/Windows/Fonts/verdanab.ttf` |
| `CascadiaMono.ttf` | Cascadia Mono | Monospace | NO | NO | Group B: NEAR-RELATIVE | Letter glyph sibling of `CascadiaCode` | `C:/Windows/Fonts/CascadiaMono.ttf` |
| `pala.ttf` | Palatino Linotype | Regular | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `palab.ttf` | `C:/Windows/Fonts/pala.ttf` |
| `cambria.ttc` | Cambria | Regular | NO | NO | Group C: NEW FAMILY | Unseen serif family | `C:/Windows/Fonts/cambria.ttc` |
| `micross.ttf` | MS Sans Serif | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen sans-serif family | `C:/Windows/Fonts/micross.ttf` |
| `Moba_Fira.ttf` | Moba Fira | Monospace | NO | NO | Group C: NEW FAMILY | Unseen monospace family | `fonts/Moba_Fira.ttf` |
| `Moba_Font.ttf` | Moba Font | Monospace | NO | NO | Group C: NEW FAMILY | Unseen monospace family | `fonts/Moba_Font.ttf` |
| `Moba_Fanta.ttf` | Moba Fanta | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen decorative sans family | `fonts/Moba_Fanta.ttf` |
| `tahomabd.ttf` | Tahoma | Bold | NO | NO | Group C: NEW FAMILY | Bold variant of unseen Tahoma family | `C:/Windows/Fonts/tahomabd.ttf` |
| `SansSerifCollection.ttf`| Sans Serif Coll. | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen sans family (subject to bbox fix) | `C:/Windows/Fonts/SansSerifCollection.ttf` |
| `Inkfree.ttf` | Ink Free | Casual Script | NO | NO | CASUAL SCRIPT | Excluded from standard-print benchmarks | `C:/Windows/Fonts/Inkfree.ttf` |
| `georgia.ttf` | Georgia | Serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `trebuc.ttf` | Trebuchet MS | Sans-serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `constan.ttf` | Constantia | Serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `framd.ttf` | Franklin Gothic | Sans-serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `lucon.ttf` | Lucida Console | Monospace | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |

Source: generate_data.py:18-36, results/v2_audit/summary.md:8-25, results/v2_audit3/summary.md:40-62, results/v2_audit4/summary.md:53-66
"""
(TBL_DIR / "table2_fonts.md").write_text(t2_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 3: Hyperparameters
# ------------------------------------------------------------------------------
t3_md = """# Table 3: Model Architecture and Training Hyperparameters (V1 vs. V2)

| Parameter / Configuration | Baseline CRNN (V1) | Photometric-Enhanced CRNN (V2) | Rationale / Reference |
| :--- | :--- | :--- | :--- |
| **Input Shape (Word Model)** | $1 \times 32 \times 128$ (Grayscale) | $1 \times 32 \times 128$ (Grayscale) | `common.py:PRESETS` |
| **Input Shape (Line Model)** | $1 \times 32 \times 512$ (Grayscale) | $1 \times 32 \times 512$ (Grayscale) | `common.py:PRESETS` |
| **CNN Backbone Architecture** | 7 Conv layers ($3\\times 3$), BatchNorm, ReLU | 7 Conv layers ($3\\times 3$), BatchNorm, ReLU | `model.py:CRNN` |
| **CNN Downsampling Factor** | Vertical: $32 \\rightarrow 1$ (32x), Horiz: $W \\rightarrow W/4$ | Vertical: $32 \\rightarrow 1$ (32x), Horiz: $W \\rightarrow W/4$ | `model.py:26-36` |
| **Recurrent Layers** | 2-layer BiLSTM ($h=256$, dropout=0.2) | 2-layer BiLSTM ($h=256$, dropout=0.2) | `model.py:37-43` |
| **Total Parameters** | 8,225,815 parameters (~8.23M) | 8,225,815 parameters (~8.23M) | `logs/line_v2_train.log:96` |
| **Vocabulary Size** | 87 (86 chars + 1 CTC blank at index 0) | 87 (86 chars + 1 CTC blank at index 0) | `common.py:Vocab` |
| **Optimizer** | Adam (lr = $1\\times 10^{-3}$, weight_decay = $1\\times 10^{-5}$) | Adam (lr = $1\\times 10^{-3}$, weight_decay = $1\\times 10^{-5}$) | `train.py:80`, `train_v2.py:246` |
| **Batch Size & Device** | 64, CPU-only execution | 64, CPU-only execution | Intel i7 / Windows CPU execution |
| **Patience (Early Stopping)** | 5 epochs without val loss improvement | 5 epochs without val loss improvement | `train_v2.py:215, 281` |
| **Training Rotation Range** | $\\pm 5.0^\\circ$ ($p=0.5$) | $\\pm 5.0^\\circ$ ($p=0.5$) | `train_v2.py:AUG_CONFIG` |
| **Perspective Jitter** | $p=0.3$ ($w\\times 0.02, h\\times 0.08$) | $p=0.3$ ($w\\times 0.02, h\\times 0.08$) | `train_v2.py:AUG_CONFIG` |
| **Illumination Gradient & Shadow** | None (Uniform pure white) | Gradient 0.7–1.0 ($p=0.5$) + soft shadow band | `train_v2.py:AUG_CONFIG` |
| **Brightness / Contrast Jitter**| None | $\\alpha \\in [0.6, 1.1]$, $\\beta \\in [-40, +20]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Gaussian Blur** | Blur $k \\in \\{3, 5\\}$ ($p=0.3$) | $\\sigma \\in [0.5, 2.0]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Motion Blur** | None | Kernel 3–7 px, random angle ($p=0.15$) | `train_v2.py:AUG_CONFIG` |
| **Sensor Gaussian Noise** | None | Noise $\\sigma \\in [3.0, 12.0]$ ($p=0.6$) | `train_v2.py:AUG_CONFIG` |
| **Salt & Pepper Noise** | $p=0.3$ (0.5% pixels) | $p=0.2$ (0.5% pixels) | `train_v2.py:AUG_CONFIG` |
| **Downscaling Resolution Loss** | None | Factor $0.5\\times - 0.9\\times$ ($p=0.4$) | `train_v2.py:AUG_CONFIG` |
| **JPEG Compression Artifacts** | None | Quality factor $Q \\in [40, 90]$ ($p=0.4$) | `train_v2.py:AUG_CONFIG` |
| **Contrast Min-Max Stretching** | None | Per-crop normalization ($p=0.5$) | `train_v2.py:AUG_CONFIG` |

Source: model.py:20-54, train.py:21-82, train_v2.py:26-48, common.py:12-25
"""
(TBL_DIR / "table3_hyperparameters.md").write_text(t3_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 4: Main Results Table
# ------------------------------------------------------------------------------
t4_md = """# Table 4: Comprehensive Empirical Recognition Performance Across All Benchmarks

| Benchmark Category | Dataset / Test Setting | Modality / Condition | Sample ($n$) | V1 CER (%) | V1 WER (%) | V2 CER (%) | V2 WER (%) | Tess 5 CER (%) | Tess 5 WER (%) | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Synthetic Mixed** | `word_synth` | Word (Clean) | 2,000 | 0.50% | 3.95% | **0.11%** | **0.90%** | 0.70% | 4.90% | `results/word_v2_synth/metrics.json` |
| **Synthetic Mixed** | `line_synth` | Line (Clean) | 1,000 | 1.64% | 11.64% | **0.30%** | **2.13%** | 0.44% | 2.94% | `results/line_v2_synth/metrics.json` |
| **Synthetic Monofont** | `diagw_tahoma` (New Fam) | Word (Clean) | 1,000 | **0.08%** | 0.70% | 0.09% | 0.70% | 0.50% | 3.20% | `results/diagw_tahoma_v2/metrics.json` |
| **Synthetic Monofont** | `diag_tahoma` (New Fam) | Line (Clean) | 500 | **0.24%** | 1.74% | 0.26% | 1.84% | 0.22% | 1.59% | `results/diag_tahoma_v2/metrics.json` |
| **Synthetic Monofont** | `diagw_consola` (Leakage)| Word (Clean) | 1,000 | 1.02% | 8.10% | **0.07%** | **0.60%** | 0.86% | 6.10% | `results/diagw_consola_v2/metrics.json` |
| **Synthetic Monofont** | `diag_consola` (Leakage) | Line (Clean) | 500 | 3.48% | 23.76% | **0.27%** | **1.93%** | 0.47% | 3.12% | `results/diag_consola_v2/metrics.json` |
| **Distortion Ladder** | L0 (Zero Distortion) | Word | 346 | 7.71% | 27.75% | **0.20%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:7` |
| **Distortion Ladder** | L0 (Zero Distortion) | Line | 92 | 1.87% | 11.14% | **0.11%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:7` |
| **Distortion Ladder** | L1 (+ Illumination) | Word | 346 | 8.85% | 32.08% | **0.20%** | NOT AVAIL | 0.45% | NOT AVAIL | `results/sim_ladder/summary.md:9` |
| **Distortion Ladder** | L1 (+ Illumination) | Line | 92 | 4.10% | 22.56% | **0.11%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:9` |
| **Distortion Ladder** | L2 (+ Blur & Noise) | Word | 346 | 25.00% | 57.80% | **0.41%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:10` |
| **Distortion Ladder** | L2 (+ Blur & Noise) | Line | 92 | 13.84% | 37.60% | **0.07%** | NOT AVAIL | 0.29% | NOT AVAIL | `results/sim_ladder/summary.md:10` |
| **Distortion Ladder** | L3 (+ Downscale/JPEG)| Word | 346 | 31.36% | 67.92% | **0.53%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:11` |
| **Distortion Ladder** | L3 (+ Downscale/JPEG)| Line | 92 | 19.14% | 42.62% | **0.18%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:11` |
| **Distortion Ladder** | L4 (Full Phone Sim) | Word | 346 | 35.77% | 75.72% | **1.35%** | NOT AVAIL | 0.37% | NOT AVAIL | `results/sim_ladder/summary.md:12` |
| **Distortion Ladder** | L4 (Full Phone Sim) | Line | 92 | 12.96% | 36.21% | **0.11%** | NOT AVAIL | 0.33% | NOT AVAIL | `results/sim_ladder/summary.md:12` |
| **Distortion Ladder** | L4n (+ Contrast Stretch)| Word | 346 | 24.43% | 67.92% | **1.14%** | NOT AVAIL | 0.37% | NOT AVAIL | `results/sim_ladder/summary.md:13` |
| **Distortion Ladder** | L4n (+ Contrast Stretch)| Line | 92 | 7.43% | 25.91% | **0.07%** | NOT AVAIL | 0.33% | NOT AVAIL | `results/sim_ladder/summary.md:13` |
| **Clean Unseen Aggregate**| L0 Clean Unseen | Word | 699 | 10.66% | 38.63% | 1.62% | 8.01% | **1.28%** | **6.29%** | `results/v2_audit3/summary.md:110` |
| **Clean Unseen Aggregate**| L0 Clean Unseen | Line | 180 | 5.84% | 31.38% | 0.75% | 7.67% | **0.58%** | **3.21%** | `results/v2_audit3/summary.md:111` |
| **Clean Unseen Aggregate**| L4 Full Sim | Word | 699 | 44.34% | 77.83% | 1.85% | 9.87% | **1.22%** | **5.72%** | `results/v2_audit3/summary.md:112` |
| **Clean Unseen Aggregate**| L4 Full Sim | Line | 180 | 21.98% | 56.90% | 1.39% | 13.25% | **1.22%** | **3.49%** | `results/v2_audit3/summary.md:113` |
| **Real Photo 1 (TNR)** | Full Line | Line | 16 | 77.05% | 100.00%| 53.61% | 92.63% | **1.32%** | **5.07%** | `results/real_fix8/final_tables.md:9` |
| **Real Photo 1 (TNR)** | Rule Chunks (Trim+4px)| Line | 16 | 18.38% | 73.27% | **0.97%** | 5.53% | 1.32% | **5.07%** | `results/real_fix8/final_tables.md:13` |
| **Real Photo 2 (Calibri)** | Full Line | Line | 16 | 67.81% | 99.05% | 36.43% | 96.21% | **0.14%** | **1.42%** | `results/real_fix8/final_tables.md:15` |
| **Real Photo 2 (Calibri)** | Rule Chunks (Trim+4px)| Line | 16 | 9.77% | 55.45% | 0.55% | 3.79% | **0.14%** | **1.42%** | `results/real_fix8/final_tables.md:19` |
| **Real Photo 3 (TNR 23L)**| Full Line | Line | 23 | 7.39% | 39.86% | 0.20% | 1.45% | **0.00%** | **0.00%** | `results/photo3_test/report.md:12` |
| **Real Photo 3 (TNR 23L)**| Rule Chunks (Trim+4px)| Line | 23 | 4.63% | 32.61% | 0.10% | 0.72% | **0.00%** | **0.00%** | `results/photo3_test/report.md:11` |
| **Real Photo 4 (Arial 23L)**| Full Line | Line | 23 | 7.06% | 42.34% | 0.10% | 0.73% | **0.00%** | **0.00%** | `results/photo4_sans/report.md:12` |
| **Real Photo 4 (Arial 23L)**| Rule Chunks (Trim+4px)| Line | 23 | 5.67% | 38.69% | 0.60% | 3.65% | 0.60% | 3.65% | `results/photo4_sans/report.md:11` |
| **Real Photo 5 (IMG_5130)**| Full Line (Cond A) | Line | 33 | 19.80% | 69.35% | 3.87% | 21.13% | 8.42% (psm7)| 15.18% | `results/real_fix9/metrics.json:4-35` |
| **Real Photo 5 (IMG_5130)**| Full Line psm 13 (Cond A)| Line | 33 | — | — | — | — | **0.09%** | **0.60%** | `results/real_fix9/metrics.json:37` |
| **Real Photo 5 (IMG_5130)**| Rule Chunks (Cond A) | Line | 33 | 8.76% | 40.48% | **2.10%** | **12.80%** | 13.02% (psm7)| 19.35% | `results/real_fix9/metrics.json:48-78` |
| **Real Photo 5 (IMG_5130)**| Clean Subset (Cond A) | Line | 23 | 6.73% | 32.16% | 1.32% | 9.25% | **1.26%** (psm7)| **7.05%** | `results/real_fix9/metrics.json:268-299` |
| **Real Photo 5 (IMG_5130)**| Padded 20px (Cond B) | Line | 33 | 38.57% | 86.90% | 23.32% | 66.07% | **18.86%** (psm7)| **27.38%**| `results/real_fix9/metrics.json:149-169` |
| *Heuristic Pilot (Snoop)*| V2 + Heuristic Lexicon | Line | 33 | — | — | *1.80%* | NOT AVAIL | *0.09%* (psm13)| *0.60%* | `scratch/test_safe_corrector.py` |

Source: Compiled directly from metrics.json files across results directories (results/word_v2_synth, results/line_v2_synth, results/diag_tahoma_v2, results/diag_consola_v2, results/sim_ladder, results/v2_audit3, results/real_fix8, results/real_fix9).
"""
(TBL_DIR / "table4_main_results.md").write_text(t4_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 5: Confidence Intervals
# ------------------------------------------------------------------------------
t5_md = """# Table 5: Bootstrap 95% Confidence Intervals for Accuracy Differences (V2 − Tesseract)

| Evaluation Condition | Sample Size ($n$) | Character Count | V2 CER (%) | Tess CER (%) | Difference $\\Delta$CER [95% CI] | Difference $\\Delta$WER [95% CI] | Zero in 95% CI? | Statistical Conclusion | Source File |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Group A (Seen): Word L0** | 378 crops | 2,472 | 0.81% | 2.39% | **-1.58%** [-2.68%, -0.76%] | NOT AVAIL | **NO (V2 ahead)** | Statistically significant gain on seen words | `results/v2_audit4/summary.md:120` |
| **Group A (Seen): Line L0** | 95 crops | 2,777 | 0.11% | 0.29% | **-0.18%** [-0.57%, +0.11%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:121` |
| **Group A (Seen): Word L4** | 378 crops | 2,472 | 1.33% | 1.70% | **-0.36%** [-1.25%, +0.50%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:122` |
| **Group A (Seen): Line L4** | 95 crops | 2,777 | 1.15% | 1.62% | **-0.47%** [-1.91%, +0.44%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:123` |
| **Group B (Sibling): Word L0**| 436 crops | 3,012 | 0.46% | 0.96% | **-0.50%** [-1.28%, +0.09%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:124` |
| **Group B (Sibling): Line L0**| 114 crops | 3,368 | 0.30% | 0.12% | **+0.18%** [-0.31%, +0.88%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:125` |
| **Group B (Sibling): Word L4**| 436 crops | 3,012 | 0.73% | 1.10% | **-0.37%** [-1.26%, +0.50%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:126` |
| **Group B (Sibling): Line L4**| 114 crops | 3,368 | 0.50% | 0.18% | **+0.33%** [-0.18%, +1.06%] | NOT AVAIL | **YES (Tie)** | Statistically indistinguishable | `results/v2_audit4/summary.md:127` |
| **Group C (New): Word L0** | 609 crops | 4,194 | 1.62% | 1.00% | **+0.62%** [-0.35%, +2.08%] | NOT AVAIL | **YES (Tie)** | Cluster CI spans across 0.0% | `results/v2_audit4/summary.md:128` |
| **Group C (New): Line L0** | 158 crops | 4,673 | 0.71% | 0.71% | **+0.00%** [-0.74%, +0.76%] | NOT AVAIL | **YES (Tie)** | Exact equal median; CI centered at 0 | `results/v2_audit4/summary.md:129` |
| **Group C (New): Word L4** | 609 crops | 4,194 | 2.36% | 0.81% | **+1.55%** [+0.59%, +2.94%] | NOT AVAIL | **NO (Tess ahead)**| Tesseract superior on distorted single words | `results/v2_audit4/summary.md:130` |
| **Group C (New): Line L4** | 158 crops | 4,673 | 1.28% | 1.46% | **-0.17%** [-2.60%, +1.53%] | NOT AVAIL | **YES (Tie)** | Cluster CI wide [-2.60%, +1.53%], spans 0 | `results/v2_audit4/summary.md:131` |
| **Photo 1: Trimmed Chunks** | 16 lines | 1,442 | 0.97% | 1.32% | **-0.35%** [-1.52%, +0.50%] | **+0.46%** [-3.69%, +4.02%] | **YES (Tie)** | Statistically indistinguishable | `results/real_fix8/final_tables.md:13` |
| **Photo 2: Trimmed Chunks** | 16 lines | 1,463 | 0.55% | 0.14% | **+0.41%** [+0.07%, +0.94%] | **+2.37%** [+0.48%, +5.05%] | **NO (Tess ahead)**| Tesseract marginally ahead on clean Calibri | `results/real_fix8/final_tables.md:19` |
| **Photo 5: All 33 Lines (A)**| 33 lines | 2,328 | 2.10% | 13.02% | **-10.91%** [-17.79%, -4.56%]| **-6.55%** [-14.96%, +1.55%]| **NO (Artifact)** | Unfair comparison: Tess 9 empty chunks | `results/real_fix9/metrics.json:80-90` |
| **Photo 5: Clean Subset (A)**| 23 lines | 1,591 | 1.32% | 1.26% | **+0.06%** [-0.77%, +0.80%] | **+2.20%** [-1.99%, +6.09%] | **YES (Tie)** | **Hinged Result:** Zero in CI; indistinguishable | `results/real_fix9/metrics.json:301-311` |
| **Photo 5: Padded 20px (B)** | 33 lines | 2,328 | 23.32% | 18.86% | **+4.47%** [-1.89%, +10.55%]| **+38.69%** [+30.47%, +46.95%]| **YES on CER** | V2 collapses; severe spacing degradation | `results/real_fix9/metrics.json:172-182` |

Source: results/v2_audit4/summary.md:118-132, results/real_fix8/final_tables.md, results/real_fix9/metrics.json
"""
(TBL_DIR / "table5_confidence_intervals.md").write_text(t5_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 6: Special Letters Table
# ------------------------------------------------------------------------------
t6_md = """# Table 6: Azerbaijani Diacritic Character Recognition and Confusion Frequencies

| Diacritic Letter Pair | Sample Size ($n$) | V1 Baseline Error % | V2 Word L0 Error % | Tess Word L0 Error % | V2 Word L4 Error % | Tess Word L4 Error % | V2 Line L4 Error % | Tess Line L4 Error % | Primary Confusion Pattern | Source File |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **ə / e** | 394 | 0.04% (line) | **0.00%** | 2.03% | **0.00%** | 1.78% | **0.00%** | 1.02% | Schwa preserved perfectly in V2 | `results/v2_audit3/summary.md:124, 135` |
| **ı / i** | 138 | 0.23% (line) | 6.52% | **0.72%** | 7.25% | **0.00%** | 2.17% | **0.72%** | Dotless-i glyph confusion (ı -> i, i -> ü) | `results/v2_audit3/summary.md:125, 136` |
| **ö / o** | 32 | 0.56% (line) | **0.00%** | **0.00%** | 6.25% | **0.00%** | 3.12% | **0.00%** | Umlaut loss under heavy blur/noise | `results/v2_audit3/summary.md:126, 137` |
| **ğ / g** | 20 | 0.00% (line) | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | 5.00% | Breve mark preserved in V2; Tess loses breve | `results/v2_audit3/summary.md:127, 138` |
| **ü / u** | 56 | 0.00% (line) | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **0.00%** | 1.79% | High recognition fidelity across models | `results/v2_audit3/summary.md:128, 139` |
| **ş / s** | 56 | 7.69% (word V1) | 1.79% | **0.00%** | **0.00%** | **0.00%** | **0.00%** | 1.79% | V1 had high cedilla error; V2 resolved to 0% | `results/v2_audit3/summary.md:129, 140` |
| **ç / c** | 23 | 22.43% (word V1)| **0.00%** | **0.00%** | 4.35% | **0.00%** | **0.00%** | **0.00%** | V1 had 22.4% collapse; V2 line error is 0.0% | `results/v2_audit3/summary.md:130, 141` |

Source: results/v2_audit3/summary.md:121-142, logs/eval_word_synth.log:40-41, logs/eval_line_synth.log:25-26, results/v2_audit4/summary.md:141-196
"""
(TBL_DIR / "table6_special_letters.md").write_text(t6_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 7: Error Taxonomy Table
# ------------------------------------------------------------------------------
t7_md = """# Table 7: Mutually Exclusive Error Taxonomy Breakdown (Photo 5, IMG_5130)

| Error Category | Scope / Definition | Condition A: V2 Line | Condition A: Tess psm 7 | Clean 23: V2 Line | Clean 23: Tess psm 7 | Impact & Root Cause Analysis | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1. Leading Edge Insertions** | Insertions strictly before 1st aligned char | **0** | **3** | **0** | **3** | Margin artifacts; V2 has 0 leading edge noise | `results/real_fix9/report.md:75` |
| **2. Trailing Edge Insertions**| Insertions strictly after last aligned char | **0** | **0** | **0** | **0** | Clean baseline termination | `results/real_fix9/report.md:76` |
| **3. Internal Spacing Errors** | Space insertions/deletions inside body | **3** | **44** | **2** | **1** | Tess has 44 space errors from 9 empty chunk runs | `results/real_fix9/report.md:77` |
| **4. Internal 0 / O Errors** | Digit `0` vs letter `O / o` confusion | **0** | **0** | **0** | **0** | Zero confusion in literary text | `results/real_fix9/report.md:78` |
| **5. Internal Punctuation** | Punctuation edits (e.g. `.` vs `:`) | **12** | **20** | **6** | **7** | Colon/period ambiguities on faint print dots | `results/real_fix9/report.md:79` |
| **6. Internal Letter Errors** | True glyph substitutions and deletions | **34** | **236** | **13** | **9** | Tess 236 includes complete line deletions | `results/real_fix9/report.md:80` |
| **MUTUALLY EXCLUSIVE SUM** | $\\sum \\text{Categories}$ | **49** | **303** | **21** | **20** | **100% Equal to Levenshtein Sum** | `results/real_fix9/report.md:81` |
| **Levenshtein Char Distance** | Independent edit distance check | **49** | **303** | **21** | **20** | Sum verification passed | `results/real_fix9/metrics.json:19, 75, 240, 294` |
| **Total Ground Truth Chars** | Total character pool | 2,328 | 2,328 | 1,591 | 1,591 | 33 lines (all) vs 23 lines (clean subset) | `results/real_fix9/metrics.json:20, 241` |
| **Final Evaluated CER (%)** | Total Errors / Total Chars | **2.10%** | **13.02%** | **1.32%** | **1.26%** | V2 ~= Tess when empty outputs excluded | `results/real_fix9/metrics.json:59, 70, 279, 290` |

Source: results/real_fix9/error_taxonomy.json, results/real_fix9/report.md:69-82, results/real_fix9/metrics.json
"""
(TBL_DIR / "table7_error_taxonomy.md").write_text(t7_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 8: Tesseract PSM Table
# ------------------------------------------------------------------------------
t8_md = """# Table 8: Tesseract Page Segmentation Mode (PSM) Sensitivity and Integration Diagnostics

| Test Setting / Crop Type | Tesseract PSM Mode | Total Evaluated | Empty Outputs Count ($n$) | Total Errors | Total Characters | Evaluated CER (%) | Evaluated WER (%) | Exact Matches | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Photo 5: Chunks (Cond A)** | PSM 7 (Single text line) | 33 lines (66 chunks) | **9 chunks empty** | 303 | 2,328 | 13.02% | 19.35% | 11 / 33 | `results/real_fix9/metrics.json:70-76` |
| **Photo 5: Full Line (Cond A)**| PSM 7 (Single text line) | 33 lines | **2 lines empty** | 196 | 2,328 | 8.42% | 15.18% | 14 / 33 | `results/real_fix9/metrics.json:26-32` |
| **Photo 5: Chunks (Cond B)** | PSM 7 (Single text line) | 33 lines (66 chunks) | **11 chunks empty**| 439 | 2,328 | 18.86% | 27.38% | 8 / 33 | `results/real_fix9/metrics.json:161-167` |
| **Photo 5: Full Line (Cond B)**| PSM 7 (Single text line) | 33 lines | **2 lines empty** | 254 | 2,328 | 10.91% | 22.92% | 8 / 33 | `results/real_fix9/metrics.json:117-123` |
| **Photo 5: Fallback Chunks** | PSM 13 (Raw single line) | 9 failed chunks | **0 empty** | ~1 | ~380 | <0.3% | <1.0% | 8 / 9 chunks | `results/real_fix9/report.md:27` |
| **Photo 5: Full Line (Cond A)**| PSM 13 (Raw single line) | 33 lines | **0 empty** | **2** | 2,328 | **0.09%** | **0.60%** | **31 / 33** | `results/real_fix9/metrics.json:37-43` |
| **Photo 5: Clean 23 Lines** | PSM 7 (Single text line) | 23 lines | **0 empty** | 20 | 1,591 | **1.26%** | **7.05%** | 11 / 23 | `results/real_fix9/metrics.json:290-296` |
| **Photo 5: Clean 23 Lines** | PSM 13 (Raw single line) | 23 lines | **0 empty** | **0** | 1,591 | **0.00%** | **0.00%** | **23 / 23** | `results/real_fix9/metrics.json:257-263` |

Source: results/real_fix9/metrics.json:24-46, 70-79, 116-136, 160-170, 245-266, 289-300; results/real_fix9/empty_outputs_log.json
"""
(TBL_DIR / "table8_tesseract_psm.md").write_text(t8_md, encoding="utf-8")

# ------------------------------------------------------------------------------
# Table 9: Bugs and Lessons Table
# ------------------------------------------------------------------------------
t9_md = """# Table 9: Chronological Pitfalls, Bugs Discovered, Quantitative Impact, and Engineering Fixes

| # | Bug / Methodological Pitfall | Discovery Mechanism | Pre-Fix Quantitative Impact | Engineering Resolution & Code Fix | Post-Fix Quantitative Impact | Disclosed Status / Lesson |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Clipped Glyph Bottoms (Early TRDG)** | Visual contact sheet inspection (`verify_quality.py`) | Descenders on `ş`, `ç`, `q`, `y`, `p` truncated at bottom edge | Added safe margin (+4 px) and updated TRDG text height anchor | 0.0% ink touch across 500 samples in all 6 splits (`results/data_check.png`) | Visual sanity inspection of raw crops is mandatory before model training. |
| **2** | **Space Missing from Vocabulary** | Training log anomaly (`logs/line_train_nospace.log`) | Line CER: **11.19%**, WER: **99.75%** (Model concatenated all words without space) | Added space character `' '` to `common.py:CHARS` string | Line CER dropped to **1.64%**, WER dropped to **11.64%** (8.6x CER, 8.5x WER drop) | CTC blank (index 0) cannot serve double duty as text whitespace delimiter. |
| **3** | **SansSerifCollection Bounding Box Metric Inflation** | Audit 3 geometry audit on synthetic ladder | Ascent+descent (80px) vertical box overlapped neighboring lines: Word CER **84.32%** | Switched in `make_sim_ladder.py` to tight ink bounding box (`w_bbox`) | L0 Word CER dropped from **84.32%** down to **0.76%** (`results/v2_audit3/summary.md:31`) | Bounding boxes must track tight glyph ink rather than theoretical font font-metrics. |
| **4** | **Tesseract Integration Collapse with PSM 7** | Empty output audit on Photo 5 (`results/real_fix9/empty_outputs_log.json`) | Tesseract returned empty string `""` on 9 chunks and 2 full lines (CER **13.02%**) | Documented failure as baseline detector integration failure; validated fallback via `--psm 13` | Under PSM 13, CER dropped to **0.09%** (2 errors / 2328 chars) | Benchmark comparisons must decouple OCR engine recognition from page layout segmentation. |
| **5** | **Heuristic Lexicon Over-Fitting (Data Snooping)** | Critical review of post-hoc test results | Heuristic post-corrector reduced Photo 5 CER from 2.10% to 1.80%, but was tuned on Photo 5 | Quarantined lexicon corrector as a pilot study; preserved Raw V2 as the sole reported paper result | Prevented biased reporting; reported 8 TP / 1 FP (+7 chars) with prominent snooping disclaimer | Post-hoc tuning on test data invalidates scientific claims; frozen pipeline must remain pure. |

Source: logs/line_train_nospace.log, logs/line_train.log, verify_quality.py, results/v2_audit3/summary.md:24-35, results/real_fix9/report.md:21-28, scratch/test_safe_corrector.py
"""
(TBL_DIR / "table9_bugs_and_lessons.md").write_text(t9_md, encoding="utf-8")

print("All 9 Markdown tables written successfully in paper/tables/.")

# Also export CSV files for each table
for f in TBL_DIR.glob("table*.md"):
    csv_out = f.with_suffix(".csv")
    lines = [l.strip() for l in f.read_text(encoding="utf-8").split("\n") if l.strip().startswith("|")]
    if len(lines) >= 3:
        header = [c.strip() for c in lines[0].split("|")[1:-1]]
        rows = []
        for l in lines[2:]:
            rows.append([c.strip() for c in l.split("|")[1:-1]])
        with open(csv_out, "w", newline="", encoding="utf-8") as out_f:
            w = csv.writer(out_f)
            w.writerow(header)
            w.writerows(rows)
            
print("All CSV tables exported successfully in paper/tables/.")
