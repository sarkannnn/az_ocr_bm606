# AzOCR-V2 — Remaining Technical Evidence Report

**Generated:** 2026-11-26  
**Output Directory:** `FINAL_REMAINING_CHECKS/`  
**Scope:** Four newly executed post-hoc / retrospective evaluations using existing trained checkpoints, existing benchmarks, and repository source scripts (no new GPU training; no changes to original `.docx`, checkpoints, or datasets).

| Task | Description | Evaluation Label | Deliverable CSV |
|---|---|---|---|
| 1 | Real30 Tesseract (`v5.4.0.20240606`) PSM 7 & PSM 13 fairness (original vs +10 px white margin) | `POST_HOC_FAIRNESS_BENCHMARK` | `REAL30_TESSERACT_FAIRNESS.csv` |
| 2 | Real30 Clean325 padding sensitivity (0, 4, 8, 12, 20 px) across Group D, RandomPadding, VariableWidth (9 models) | `POST_HOC_PADDING_SENSITIVITY` | `REAL30_PADDING_SENSITIVITY.csv` |
| 3 | Frozen confidence threshold calibrated on D1–D5 (111 lines) evaluated unchanged on B1–B15 (295 lines) | `RETROSPECTIVE_FROZEN_THRESHOLD` | `FROZEN_THRESHOLD_CASCADE.csv` |
| 4 | Exact V2Clean training font membership vs disk inventory (16 files / 15 families vs 14 used) | `VERIFIED_SOURCE_AUDIT` | `VERIFIED_TRAINING_FONT_MEMBERSHIP.csv` |

---

## 1. Real30 Tesseract Fairness Benchmark (`REAL30_TESSERACT_FAIRNESS.csv`)

- **Protocol (`POST_HOC_FAIRNESS_BENCHMARK`):** Evaluated `Tesseract v5.4.0.20240606` (`lang="aze"`) across all **690 crops** from `Real30_GPU18_Final_Evaluation/extracted_photos/P01.jpg–P30.jpg` (aligned via `verified_alignment.csv`: `325` `genuine_single_line`, `314` `noise`, `42` `multi_line_merge`, `9` `partial_fragment`; `421` page GT lines) under four conditions: `PSM 7` vs `PSM 13`, each with `0 px` (original crop) vs `+10 px` white border (`cv2.BORDER_CONSTANT`, `[255, 255, 255]`).
- **Executed Command:** `.\.venv\Scripts\python.exe -u FINAL_REMAINING_CHECKS\run_task1_real30_tesseract.py`

| Condition | Clean325 CER (%) | Clean325 WER (%) | Clean325 Exact (/325) | Clean325 Empty | Noise FP Rate (/314) | Page Primary CER (%) | Page Primary WER (%) | Mean Latency (ms/crop) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PSM7_original_crop` | **22.33** | **30.23** | **62 (19.08%)** | 7 | **65.61%** (206/314) | **35.37** | **47.05** | **155.74** |
| `PSM13_original_crop` | 30.84 | 49.50 | 4 (1.23%) | 0 | 100.00% (314/314) | 53.67 | 96.06 | 185.19 |
| `PSM7_plus_10px_white` | 30.60 | 38.74 | 41 (12.62%) | 35 | 92.36% (290/314) | 49.23 | 60.19 | 198.88 |
| `PSM13_plus_10px_white` | 42.51 | 65.10 | 0 (0.00%) | 0 | 100.00% (314/314) | 58.98 | 84.95 | 172.85 |

- **Scientific Interpretation:** On `Real30` smartphone crops, `PSM 7` on the original crop is the strongest Tesseract configuration (`22.33%` Clean325 CER, `35.37%` Page Primary CER). Switching to `PSM 13` (`30.84%` Clean325 CER, `100%` noise false-positive rate) or adding a `+10 px` white border (`30.60%` for `PSM 7`, `42.51%` for `PSM 13`, with `35` empty Clean325 outputs under `PSM 7 + 10px`) substantially degrades accuracy and increases noise hallucinations.

---

## 2. Real30 Clean325 Padding Sensitivity (`REAL30_PADDING_SENSITIVITY.csv`)

- **Protocol (`POST_HOC_PADDING_SENSITIVITY`):** Evaluated all **9 official checkpoints** (`Group_D_V2Clean_PhotoAug`, `Exp_RandomPadding_GroupD`, `Exp_VariableWidth_GroupD` × seeds `42, 43, 44`) loaded from `AzOCR_v1_Models_18Official.zip` on `Real30 Clean325` (`325` crops, `25,180` GT chars) at `P ∈ {0, 4, 8, 12, 20} px` under **two strictly separated protocols**:
  1. **`Canvas_Border_Padding`:** Adds `+P px` white (`[255, 255, 255]`) on all 4 sides of the input crop before `infer_line_smart()`. Note that `P=0` reproduces the official Real30 Clean325 baseline (`779` total chunks across 325 crops).
  2. **`Tight_Ink_Margin`:** Runs `infer_line_with_custom_margin()` using `trim_chunk_with_margin(ch, margin=P)` inside fixed chunking (`779` chunks; `P=4` matches default `robust_trim_chunk`).
- **Executed Command:** `.\.venv\Scripts\python.exe -u FINAL_REMAINING_CHECKS\run_task2_real30_padding.py`

### 3-Seed Group Means ± SD (`Clean325 CER %` | `Exact / 325` | `Total Chunks`)

| Protocol | Padding (px) | `Group_D` CER (%) | `RandomPadding` CER (%) | `VariableWidth` CER (%) | Total Chunks |
|---|---:|---:|---:|---:|---:|
| `Canvas_Border_Padding` | 0 | 21.68 ± 0.60 (35.7) | 21.52 ± 0.48 (38.3) | 22.32 ± 1.82 (37.7) | 779 |
| `Canvas_Border_Padding` | 4 | 27.68 ± 1.83 (0.7) | **26.32 ± 1.87** (0.7) | 28.72 ± 0.90 (0.0) | 653 |
| `Canvas_Border_Padding` | 8 | 38.61 ± 3.24 (0.0) | **30.40 ± 5.93** (0.0) | 35.96 ± 3.78 (0.0) | 576 |
| `Canvas_Border_Padding` | 12 | 52.64 ± 3.12 (0.0) | **34.72 ± 5.72** (0.0) | 49.38 ± 3.63 (0.0) | 498 |
| `Canvas_Border_Padding` | 20 | 80.28 ± 2.05 (0.0) | **55.00 ± 3.07** (0.0) | 78.66 ± 2.10 (0.0) | 362 |
| `Tight_Ink_Margin` | 0 | 21.77 ± 0.65 (32.0) | 21.61 ± 0.45 (31.7) | 22.37 ± 1.76 (32.0) | 779 |
| `Tight_Ink_Margin` | 4 | 21.68 ± 0.60 (35.7) | 21.52 ± 0.48 (38.3) | 22.32 ± 1.82 (37.7) | 779 |
| `Tight_Ink_Margin` | 8 | 21.72 ± 0.59 (33.7) | **21.51 ± 0.41** (40.0) | 22.33 ± 1.86 (37.0) | 779 |
| `Tight_Ink_Margin` | 12 | 21.96 ± 0.78 (29.0) | **21.53 ± 0.44** (41.0) | 22.56 ± 1.77 (28.7) | 779 |
| `Tight_Ink_Margin` | 20 | 22.96 ± 1.38 (14.3) | **21.82 ± 0.70** (33.7) | 23.69 ± 2.23 (11.3) | 779 |

- **Scientific Interpretation:**
  - Under `Canvas_Border_Padding`, adding vertical+horizontal border pixels reduces `squeeze_ratio = (w_c * 32) / (h_c * 512)`, dropping chunk count from `779` (`0 px`) to `362` (`20 px`) and shrinking glyph height. Even so, `Exp_RandomPadding_GroupD` (`random_padding=12`) exhibits substantially greater resilience at `+12 px` (`34.72%` vs `52.64%` Group D / `49.38%` VariableWidth) and `+20 px` (`55.00%` vs `80.28%` / `78.66%`).
  - Under `Tight_Ink_Margin`, chunk count stays fixed at `779`, and `Exp_RandomPadding_GroupD` maintains flat CER across `0–20 px` (`21.51%–21.82%`, peaking at `41.0/325` exact matches at `12 px`), whereas `Group_D` and `VariableWidth` lose exact matches sharply at `20 px` (`14.3/325` and `11.3/325`).

---

## 3. Retrospective Frozen-Threshold Cascade (`FROZEN_THRESHOLD_CASCADE.csv`)

- **Protocol (`RETROSPECTIVE_FROZEN_THRESHOLD`):** Confidence thresholds were calibrated **strictly on `D1–D5` (`111` lines)** using `CRNN-V2` (`runs/line_v2/best.pt`, `decode_single_crop_with_confidence`), then frozen and applied **unchanged** (`score <= T_frozen`) to **`B1–B15` (`295` lines)** from `results/confidence_audit/confidence_295_lines.csv` (using `T_CRNN = 85.71 ms`, `T_Tess = 128.40 ms` from `cpu_speed_benchmark_results.json`).
- **Executed Command:** `.\.venv\Scripts\python.exe FINAL_REMAINING_CHECKS\run_task3_frozen_cascade.py`

| Calibration Rule on `D1–D5` | Frozen Threshold `T` | `D1–D5` Routed (%) | `B1–B15` Routed (/295) | `B1–B15` CER (%) | `B1–B15` WER (%) | `B1–B15` Exact (/295) | Effective Latency (ms) | Speedup vs Tess |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Baseline_CRNN_V2_Only` | — | 0.00% | 0 (0.00%) | 0.298 | 1.64 | 253 (85.76%) | 85.71 | 1.50× |
| `D1-D5_10pct_min_conf` | `0.446454` | 9.91% | 4 (1.36%) | 0.280 | 1.49 | 256 (86.78%) | 87.45 | 1.47× |
| `D1-D5_15pct_min_conf` | `0.494742` | 15.32% | 7 (2.37%) | **0.269** | **1.40** | **257 (87.12%)** | 88.76 | 1.45× |
| **`D1-D5_20pct_min_conf`** | **`0.509835`** | **19.82%** | **15 (5.08%)** | **0.269** | **1.40** | **257 (87.12%)** | **92.24** | **1.39×** |
| `D1-D5_25pct_min_conf` | `0.530081` | 25.23% | 41 (13.90%) | 0.284 | 1.43 | 257 (87.12%) | 103.56 | 1.24× |
| `D1-D5_30pct_min_conf` | `0.540676` | 29.73% | 53 (17.97%) | 0.287 | 1.46 | 256 (86.78%) | 108.78 | 1.18× |
| `D1-D5_YoudenJ_min_conf (J=0.531)` | `0.685063` | 53.15% | 177 (60.00%) | 0.272 | 1.31 | 257 (87.12%) | 162.75 | 0.79× |
| `D1-D5_YoudenJ_mean_conf (J=0.552)` | `0.988943` | 54.05% | 111 (37.63%) | 0.276 | 1.37 | 256 (86.78%) | 134.02 | 0.96× |
| `Baseline_Tesseract5_PSM7_Standalone` | — | 100.00% | 295 (100.00%) | 0.310 | 1.55 | 250 (84.75%) | 128.40 | 1.00× |

- **Scientific Interpretation:** Freezing the `D1–D5` 20%-calibration cutoff (`min_conf <= 0.509835`) and applying it unchanged to `B1–B15` routes only **5.08% (15/295)** of `B1–B15` lines to Tesseract (because `B1–B15` has higher confidence than `D1–D5`), yet improves `B1–B15` CER (`0.298% -> 0.269%`), WER (`1.64% -> 1.40%`), and Exact Match (`85.76% -> 87.12%`) at **`92.24 ms/line`** (`1.39×` faster than standalone Tesseract).

---

## 4. Verified V2Clean Training Font Membership (`VERIFIED_TRAINING_FONT_MEMBERSHIP.csv`)

- **Reconciliation Result:**
  - **On disk (`fonts/` + `fonts_new/`):** **16 `.ttf` files** representing **15 unique font families** (`consola.ttf` [Consolas Regular] and `consolab.ttf` [Consolas Bold] share the `Consolas` family).
  - **Actually used in `V2Clean` training (`data/synth_line_v2_clean/train`, `15,754` images):** **14 `.ttf` files / 14 distinct families**:
    1. **Base `synth_line/train` (`0000000–0009999`, `9,847` surviving images in V2Clean):** Generated via `generate_data.py` using `collect_fonts("fonts", seed=42)`, which selected **8 training fonts** (`arial.ttf`, `calibri.ttf`, `candara.ttf`, `corbel.ttf`, `cour.ttf`, `segoeui.ttf`, `times.ttf`, `verdana.ttf`) and **held out 2 fonts for `test` only** (`tahoma.ttf` [Tahoma], `consola.ttf` [Consolas Regular]).
    2. **Extra V2 lines (`0010000–0015999`, `5,907` surviving images in V2Clean):** Generated via `scratch/build_v2_datasets.py` using **all 6 `.ttf` files in `fonts_new/`** (`bahnschrift.ttf` [Bahnschrift], `CascadiaCode.ttf` [Cascadia Code], `consolab.ttf` [Consolas Bold], `ebrima.ttf` [Ebrima], `l_10646.ttf` [Lucida Sans Unicode], `palab.ttf` [Palatino Linotype Bold]).
  - **Why `16 files / 15 families` vs `14 used`:** `tahoma.ttf` and `consola.ttf` were never used in training (held out for test), while `consolab.ttf` (Consolas Bold) was added in `fonts_new/`. Thus, actual V2Clean training membership is **14 `.ttf` files (14 distinct families)**.
