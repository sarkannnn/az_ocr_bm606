# AzOCR-V2 — Technical Evidence Report for Academic Manuscript & Supplementary Material

**Generated:** 2026-11-25  
**Scope:** Independent source-code, configuration, checkpoint, log, and existing prediction audit (`T01`–`T18`) in `D:\laragon\www\az_ocr_bm606\FINAL_TECHNICAL_EVIDENCE\`.  
**Rules Observed:** No `.docx` files modified; no new GPU training launched; no original datasets, checkpoints, or benchmark predictions altered; no GitHub push or Zenodo publish performed; all missing/unlogged fields explicitly flagged.

---

## Summary Index (`T01`–`T18`)

| Task | Title | Status | Primary Evidence File |
|---|---|---|---|
| T01 | Model architecture & parameter count | `VERIFIED` | `MODEL_ARCHITECTURE_VERIFIED.csv` |
| T02 | GPU18 training configuration & hyperparameters | `PARTIALLY_VERIFIED` | `GPU18_FULL_CONFIGURATION_VERIFIED.csv` |
| T03 | GPU18 checkpoint SHA-256 integrity | `VERIFIED` | `GPU18_CHECKPOINT_SHA256.csv` |
| T04 | Augmentation full specification | `VERIFIED` | `AUGMENTATION_FULL_SPECIFICATION_VERIFIED.csv` |
| T05 | RandomPadding implementation details | `VERIFIED` | `RANDOM_PADDING_IMPLEMENTATION.md` |
| T06 | VariableWidth architecture & native inference | `NEW_EXPLORATORY_EVALUATION` | `VARIABLE_WIDTH_INFERENCE_RESULTS.csv` |
| T07 | Photo 3 CER reconciliation | `VERIFIED` | `PHOTO3_CER_RECONCILIATION.md` |
| T08 | Legacy GT character counts | `VERIFIED` | `LEGACY_GT_CHARACTER_COUNTS.csv` |
| T09 | Dataset count, leakage & font inventory | `VERIFIED` | `DATASET_COUNT_RECONCILIATION.md` |
| T10 | Tesseract version & fairness benchmark | `VERIFIED` | `TESSERACT_FAIRNESS_RESULTS.csv` |
| T11 | EasyOCR B1–B15 (295 lines) results | `VERIFIED` | `EASYOCR_B295_RESULTS.csv` |
| T12 | Confidence AUROC & cascade protocol | `VERIFIED` | `CONFIDENCE_AUROC_VERIFIED.csv` |
| T13 | SymSpell lexicon provenance | `VERIFIED` | `SYMSPELL_PROVENANCE_AUDIT.md` |
| T14 | CPU runtime reconciliation | `VERIFIED` | `CPU_RUNTIME_RECONCILIATION.csv` |
| T15 | Real30 Clean325 21–22% CER floor root cause | `COMPUTED_FROM_EXISTING_DATA` | `CLEAN325_ERROR_ANALYSIS.md` |
| T16 | Character-level 95% confidence intervals | `COMPUTED_FROM_EXISTING_DATA` | `CHARACTER_CONFIDENCE_INTERVALS.csv` |
| T17 | Zenodo DOI & archive reproducibility | `VERIFIED` | `ZENODO_REPRODUCIBILITY_AUDIT.md` |
| T18 | GPU18 training curves (`S25`–`S30`) | `VERIFIED` | `GPU18_TRAINING_CURVES_VERIFIED.csv` |

---

## T01 — Model Architecture & Exact Parameter Count

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - Total parameters: **8,225,815**
  - Trainable parameters: **8,225,815** (0 non-trainable)
  - Alphabet size (`CHARS` in `common.py`): **86** characters
  - CTC output classes (`NUM_CLASSES`): **87** (`blank=0`, characters `1..86`)
  - Pure tensor weight footprint: **31.40 MiB** (FP32) / **15.70 MiB** (FP16)
  - Checkpoint (`best.pt`) file size on disk: **~94.20 MiB** (includes `Adam` optimizer state: `exp_avg` + `exp_avg_sq` + metadata)
- **Verified technical parameters:**
  - **CNN Backbone:** 7 `Conv2d` layers (6× `3×3` `padding=1`, `bias=False` + 1× `(2,1)` `padding=0`, `bias=True`), 7 `BatchNorm2d`, 7 `ReLU`, 4 `MaxPool2d` with strides/kernels `(2,2)`, `(2,2)`, `(2,1)`, `(2,1)` -> **32× vertical** downsampling (`H=32 -> 1`) and **4× horizontal** downsampling (`W -> W/4`). Channel progression: `1 -> 64 -> 128 -> 256 -> 256 -> 512 -> 512 -> 512`.
  - **Sequence Head:** 2-layer bidirectional `LSTM` (`input_size=512`, `hidden_size=256` per direction = `512` concatenated, `dropout=0.2` between LSTM layers), followed by `Linear(512, 87)`.
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/MODEL_ARCHITECTURE_VERIFIED.csv`
- **Source code location:** `model.py` (L1–L53), `common.py` (`CHARS`, `NUM_CLASSES`)
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch1_t01_t02_t03_t18.py`
- **Previous discrepancy:** Unverified parameter counts or confusion between pure FP32 model weight size (31.40 MiB) and full training checkpoint size (~94.20 MiB).
- **Correct interpretation:** The CRNN model itself contains exactly 8,225,815 trainable parameters (31.40 MiB FP32 weights); `best.pt` is ~94.20 MiB because `train_gpu.py` saves optimizer state dictionaries alongside model weights.
- **Recommended manuscript location:** Section 2 (Methods — Model Architecture)
- **Recommended supplementary table or section:** Supplementary Table S1 (Layer-by-layer CRNN architecture and parameter breakdown)
- **Remaining limitation:** None.
- **Publication-ready English text:**
  > The AzOCR-V2 recognizer uses a 7-layer CNN–BiLSTM–CTC architecture (`8,225,815` trainable parameters; `31.40 MiB` FP32 tensor weights; `~94.20 MiB` full training checkpoint including Adam optimizer states). Seven `Conv2d–BatchNorm2d–ReLU` blocks with four `MaxPool2d` layers (`(2,2)`, `(2,2)`, `(2,1)`, `(2,1)`) reduce a `32 × W` grayscale line image to a `1 × (W/4) × 512` feature sequence, which is processed by a 2-layer bidirectional LSTM (`hidden_size = 256` per direction, `dropout = 0.2`) and projected to `87` CTC classes (`blank = 0` plus `86` Azerbaijani/ASCII characters).

---

## T02 — GPU18 Training Configuration & Hyperparameters

- **Verification status:** `PARTIALLY_VERIFIED`
- **Verified numerical results:**
  - **18 official runs** (6 experimental groups × 3 seeds `42, 43, 44`) + **1 excluded reproducibility check** (`Exp_VariableWidth_GroupD_seed44_RTX2050`).
  - `optimizer = Adam` (`torch.optim.Adam`), `lr = 0.001`, `weight_decay = 1e-5`, `batch_size = 64`, `dropout = 0.2`.
  - `scheduler = None` (constant learning rate `0.001`; no LR scheduler in `train_gpu.py`).
  - `loss_function = CTCLoss(blank=0, zero_infinity=True)`.
  - `max_epochs = 25`, `patience = 6` for Groups A & B; `max_epochs = 30`, `patience = 7` for Groups C, D, RandomPadding, VariableWidth.
- **Verified technical parameters:**
  - Best checkpoint selection (`train_gpu.py:354`): minimum `val_cer` primary criterion; `val_loss` tiebreaker (`< 1e-5`).
  - Hardware & PyTorch/CUDA:
    - 15 runs (Groups A–D, RandomPadding): `NVIDIA GeForce RTX 2050`, `PyTorch 2.11.0+cu128`, `CUDA 12.8`.
    - 3 official VariableWidth runs: `Tesla T4`, `PyTorch 2.14.1+cu130`, `CUDA 13.0`.
    - 1 excluded repro check (`Exp_VariableWidth_GroupD_seed44_RTX2050`): `NVIDIA GeForce RTX 2050`, `PyTorch 2.11.0+cu128`, `CUDA 12.8`.
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/GPU18_FULL_CONFIGURATION_VERIFIED.csv`
- **Source code location:** `AzOCR_Final_Scientific_Audit/code_snapshot/train_gpu.py` (L248, L354), `AzOCR_Zenodo_v1_DRAFT/configs_and_logs/runs_meta/*/config.yaml`, `run_summary.json`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch1_t01_t02_t03_t18.py`
- **Previous discrepancy:** Uncertainty over whether a learning-rate scheduler was used, exact early-stopping patience per group, and hardware split between RTX 2050 and Tesla T4.
- **Correct interpretation:** No LR scheduler was used (`lr = 0.001` constant). Groups A–B ran up to 25 epochs (patience 6) on RTX 2050; Groups C–D and RandomPadding ran up to 30 epochs (patience 7) on RTX 2050; official VariableWidth ran up to 30 epochs (patience 7) on Tesla T4.
- **Recommended manuscript location:** Section 2 (Methods — Training Protocol)
- **Recommended supplementary table or section:** Supplementary Table S2 (Complete 18-run + 1 repro-check configuration table)
- **Remaining limitation:** Exact Python interpreter version during GPU training was not logged inside `run_summary.json` (only the local evaluation environment `Python 3.13.9` is directly verified).
- **Publication-ready English text:**
  > All 18 official models were trained with `Adam` (`lr = 0.001`, constant without an LR scheduler, `weight_decay = 1e-5`, `batch_size = 64`) minimizing `CTCLoss(blank=0, zero_infinity=True)`. Best checkpoints were selected by lowest validation CER (with `< 1e-5` validation loss tie-breaking). Groups A and B trained for up to 25 epochs (early-stopping patience 6) on an NVIDIA GeForce RTX 2050 (`PyTorch 2.11.0+cu128`), Groups C, D, and RandomPadding trained for up to 30 epochs (patience 7) on the same RTX 2050 environment, and the three official VariableWidth runs trained for up to 30 epochs (patience 7) on a Tesla T4 (`PyTorch 2.14.1+cu130`).

---

## T03 — GPU18 Checkpoint SHA-256 Integrity

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - **18 / 18** official `best.pt` checkpoints streamed directly from `AzOCR_Zenodo_v1_Release/AzOCR_v1_Models_18Official.zip` matched `Real30_GPU18_Final_Evaluation/checkpoint_sha256_manifest.csv` bit-for-bit.
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/GPU18_CHECKPOINT_SHA256.csv`
- **Source code location:** `Real30_GPU18_Final_Evaluation/checkpoint_sha256_manifest.csv`, `AzOCR_Zenodo_v1_Release/AzOCR_v1_Models_18Official.zip`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch1_t01_t02_t03_t18.py`
- **Previous discrepancy:** Need to verify that the released Zenodo archive contains the exact 18 evaluated `best.pt` weights without corruption or accidental inclusion of the excluded RTX 2050 repro check.
- **Correct interpretation:** All 18 official checkpoints in `AzOCR_v1_Models_18Official.zip` are 100% SHA-256 identical to the evaluated checkpoints.
- **Recommended manuscript location:** Data & Code Availability
- **Recommended supplementary table or section:** Supplementary Table S3 (18-checkpoint SHA-256 manifest)
- **Remaining limitation:** None.

---

## T04 — Augmentation Full Specification (`BasicAug` & `PhotoAug`)

- **Verification status:** `VERIFIED`
- **Verified numerical results & technical parameters:**
  - **12 transforms** verified with exact probabilities, parameter ranges, sampling distributions, and execution order:
    - `BasicAug` (4 ops): Rotation (`p=0.50`, `U(-5°, +5°)`), Gaussian blur (`p=0.30`, discrete `k ∈ {3, 5}`), Salt-and-pepper (`p=0.30`, `0.005` black + `0.005` white), Perspective (`p=0.30`, `U(-1,1) × [0.02W, 0.08H]` per corner).
    - `PhotoAug` (11 ops + nested shadow band): Rotation (`p=0.50`), Perspective (`p=0.30`), Illumination gradient (`p=0.50`, `low_val ~ U(0.70, 1.00)`, angle `U(0, 2π)`), nested Shadow band (`p_eff=0.25 = 0.50×0.50`, `d0~U(-0.5,0.5)`, `bw~U(0.2,0.5)`, `depth~U(0.12,0.28)`), Brightness/contrast (`p=0.60`, `alpha~U(0.60,1.10)`, `beta~U(-40,20)`), Gaussian blur (`p=0.60`, `sigma~U(0.5,2.0)`), Motion blur (`p=0.15`, `ksize ∈ {3,5,7}`, `angle~U(0,π)`), Sensor noise (`p=0.60`, `Normal(0, sigma)` with `sigma~U(3.0,12.0)`), Salt-and-pepper (`p=0.20`), Downscale/upscale (`p=0.40`, `scale~U(0.50,0.90)`), JPEG compression (`p=0.40`, `quality~int(U(40,90))`), Contrast stretch (`p=0.50`, `NORM_MINMAX 0..255`).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/AUGMENTATION_FULL_SPECIFICATION_VERIFIED.csv`
- **Source code location:** `AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py` (L46–L164), `train_v2.py` (L1–L135)
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch2_t04_t05_t06.py`
- **Previous discrepancy:** Incomplete reporting of exact `PhotoAug` probabilities, nested shadow probability (`0.25` effective), and differences in blur/salt-and-pepper order between `BasicAug` and `PhotoAug`.
- **Correct interpretation:** GPU `dataset.py` and CPU `train_v2.py` share identical augmentation definitions.
- **Recommended manuscript location:** Section 2 (Methods — Data Augmentation)
- **Recommended supplementary table or section:** Supplementary Table S4 (Full augmentation specification)
- **Remaining limitation:** None.

---

## T05 — `RandomPadding` Implementation Details

- **Verification status:** `VERIFIED`
- **Verified technical parameters:**
  - Config parameter: `random_padding: 12` (`Exp_RandomPadding_GroupD_seed42/config.yaml`).
  - Distribution & probability: discrete uniform `{0, 1, ..., 12}` pixels independently per side (`top, bottom, left, right`), applied with `p = 1.0`.
  - Fill color: pure white `[255, 255, 255]`.
  - Pipeline order (`dataset.py` `OCRDataset.__getitem__`): applied **1st** (before `augment_photo` and before `preprocess` resize to `32 × 512`, `dynamic_width: false`).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/RANDOM_PADDING_IMPLEMENTATION.md`
- **Source code location:** `AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py` (L169–L220), `AzOCR_Zenodo_v1_DRAFT/configs_and_logs/runs_meta/Exp_RandomPadding_GroupD_seed42/config.yaml`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch2_t04_t05_t06.py`
- **Previous discrepancy:** Exact padding range, color, and order relative to `PhotoAug` were previously unspecified in manuscript text.
- **Correct interpretation:** Random white margin padding (0–12 px per side) is applied prior to photometric augmentation and fixed-width resizing.
- **Recommended manuscript location:** Section 2 (Methods — Margin & Width Sensitivity Ablations)
- **Recommended supplementary table or section:** Supplementary Methods S1.2
- **Remaining limitation:** None.

---

## T06 — `VariableWidth` Architecture & Native Single-Pass Inference

- **Verification status:** `NEW_EXPLORATORY_EVALUATION`
- **Verified technical parameters (8 architecture questions):**
  - `dynamic_width = True` in `OCRDataset`; no length-bucketing sampler (`shuffle=True`); `collate_fn_dynamic` right-pads each batch with `1.0` (white) to the batch's maximum width rounded up to a multiple of `4`, yielding CTC time steps `T = max_w // 4` (`input_lengths = torch.full((B,), T)`).
  - Official Real30 evaluation used fixed `32 × 512` `infer_line_smart` for all 18 models for uniform comparison.
- **Verified numerical results (Official vs Post-hoc Exploratory Native Variable-Width Single-Pass on Real30):**

| Run ID | Official Fixed-512 Clean325 CER (%) | Native VariableWidth Clean325 CER (%) | Official Exact Matches | Native Exact Matches | Official Mean ms/crop | Native Mean ms/crop |
|---|---:|---:|---:|---:|---:|---:|
| `Exp_VariableWidth_GroupD_seed42` | 20.88 | **20.67** | 40 / 325 (12.31%) | **47 / 325 (14.46%)** | 92.28 | **45.91** |
| `Exp_VariableWidth_GroupD_seed43` | 21.72 | **21.41** | 29 / 325 (8.92%) | 29 / 325 (8.92%) | 92.90 | **45.43** |
| `Exp_VariableWidth_GroupD_seed44` | 24.36 | **22.13** | 44 / 325 (13.54%) | **46 / 325 (14.15%)** | 73.41 | **46.18** |

- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/VARIABLE_WIDTH_INFERENCE_RESULTS.csv`, `FINAL_TECHNICAL_EVIDENCE/VARIABLE_WIDTH_TECHNICAL_AUDIT.md`
- **Source code location:** `AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py` (L221–L273)
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch2_t04_t05_t06.py`
- **Previous discrepancy:** Official Real30 numbers for `VariableWidth` reflected fixed-width `512` inference (`infer_line_smart`), leaving open whether native variable-width inference improves accuracy and runtime.
- **Correct interpretation:** Native single-pass variable-width inference consistently reduces Clean325 CER across all 3 seeds (by `0.21` to `2.23` percentage points) and cuts inference latency roughly in half (`45.4–46.2 ms` vs `73.4–92.9 ms` per crop). Because this was run post-hoc on existing checkpoints, it must be labeled as an exploratory follow-up evaluation rather than part of the fixed-512 primary table.
- **Recommended manuscript location:** Section 3 / Discussion (Variable-Width Inference Follow-up)
- **Recommended supplementary table or section:** Supplementary Table S5
- **Remaining limitation:** Post-hoc exploratory evaluation; official Real30 comparison table retains fixed-512 `infer_line_smart` for parity across all 18 models.

---

## T07 — Photo 3 CER Reconciliation (`0.10%` vs `0.00%` vs `0.89%`)

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - `D3 / photo3_times.jpg` (23 lines, **1,015** NFC characters):
    - `0.10%` (`1 / 1015` edit) = `v2_rule_chunk` raw CRNN-V2 in `results/photo3_test/photo3_metrics.json`.
    - `0.00%` (`0 / 1015` edits) = `v2_raw` in `results/unseen_benchmark/unseen_metrics.json`.
    - `0.89%` (`9 / 1015` edits) = `v2_pp` (SymSpell post-processed) in `results/unseen_benchmark/unseen_metrics.json`, where SymSpell altered 4 genuine printed typos on the D3 printout (plus additional token edits).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/PHOTO3_CER_RECONCILIATION.md`
- **Source code location:** `results/photo3_test/photo3_metrics.json`, `results/unseen_benchmark/unseen_metrics.json`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch3_t07_t08_t09.py`
- **Previous discrepancy:** Three different CER figures (`0.10%`, `0.00%`, `0.89%`) appeared across project summaries for Photo 3.
- **Correct interpretation:** `0.10%` and `0.00%` come from two different raw evaluation runs (`v2_rule_chunk` vs `v2_raw` on the same 1,015-char GT), whereas `0.89%` is the SymSpell post-processed score showing over-correction of printed typos.
- **Recommended manuscript location:** Section 3 (Legacy Diagnostic & Post-Processing Error Analysis)
- **Recommended supplementary table or section:** Supplementary Note S2
- **Remaining limitation:** None.

---

## T08 — Legacy Ground-Truth Character Counts (`D1–D5`, `B1–B3`, `B1–B15`)

- **Verification status:** `VERIFIED`
- **Verified numerical results (NFC-normalized, `strip()`, no newline):**
  - `D1–D5` (5 diagnostic photos, 111 lines): **7,254** chars (`D1=1,442`, `D2=1,463`, `D3=1,015`, `D4=1,006`, `D5=2,328`).
  - `B1–B3` (3 benchmark photos, 55 lines): **3,920** chars (`B1=1,442`, `B2=1,463`, `B3=1,015`).
  - `B1–B15` (15 benchmark photos, 295 lines): **26,805** chars.
  - **File-mapping disambiguation:** `IMG_5130.HEIC` (`photo5.jpg`) = `D5` (33 lines, 2,328 chars); `IMG_5130.jpg` = `B10` (20 lines, 1,891 chars).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/LEGACY_GT_CHARACTER_COUNTS.csv`
- **Source code location:** `results/real_fix8/per_line_chunk_details.csv`, `results/photo3_test/per_line_chunk_details.csv`, `results/photo4_sans/per_line_chunk_details.csv`, `results/real_fix9/per_line_chunk_details.csv`, `results/easyocr_295/easyocr_295_predictions.csv`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch3_t07_t08_t09.py`
- **Previous discrepancy:** Earlier notes reported `7,245` for `D1–D5` and `3,911` for `B1–B3`.
- **Correct interpretation:** Both `7,245` (`7,254 - 9`) and `3,911` (`3,920 - 9`) were `-9` arithmetic errors caused by accidentally assigning `D4` (`1,006` chars) to `D3`/`B3` (`1,015` chars).
- **Recommended manuscript location:** Section 2 (Datasets — Legacy Diagnostic & Benchmark Splits)
- **Recommended supplementary table or section:** Supplementary Table S6 (Per-document GT line/character counts and SHA-256)
- **Remaining limitation:** None.

---

## T09 — Dataset Count Reconciliation, Leakage Audit & GPU Font Inventory

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - **V1:** `data/corpus_line/train` = `data/synth_line_v1/train` = **10,400** images (`"10,000"` in early notes was a rounded approximation). Removing `100 + 84 - 25 = 159` unique leaked texts yields **10,241** V1Clean training images.
  - **V2:** `16,000` raw -> removing `246` leaked texts yields **15,754** V2Clean training images.
  - **Post-clean overlap:** **0** leaked texts between V1Clean/V2Clean training sets and evaluation benchmarks.
  - **GPU Font Inventory:** **16** `.ttf` files across **15** unique font families (`fonts/`: 10 files; `fonts_new/`: 6 files). `"14 fonts"` in early notes excluded the 2 held-out fonts; `"18 fonts"` was an overcount.
- **Evidence files:** `FINAL_TECHNICAL_EVIDENCE/DATASET_COUNT_RECONCILIATION.md`, `FINAL_TECHNICAL_EVIDENCE/DATASET_LEAKAGE_AUDIT.csv`, `FINAL_TECHNICAL_EVIDENCE/GPU_FONT_INVENTORY_VERIFIED.csv`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch3_t07_t08_t09.py`
- **Previous discrepancy:** Conflicting counts (`10,000` vs `10,400` vs `10,241`; `16,000` vs `15,754`; `14` vs `15` vs `16` vs `18` fonts).
- **Correct interpretation:** Exact verified counts are `10,241` (V1Clean), `15,754` (V2Clean), `0` post-clean leakage, and `16` `.ttf` files (`15` unique families).
- **Recommended manuscript location:** Section 2 (Synthetic Training Corpora & Leakage Removal)
- **Recommended supplementary table or section:** Supplementary Table S7 (Font inventory & leakage audit)
- **Remaining limitation:** None.

---

## T10 — Tesseract Version & Fairness Benchmark

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - Installed & evaluated version: **Tesseract `v5.4.0.20240606`** (`5.5.0` does not exist in this workspace).
  - **B1–B15 (295 lines):**
    - `PSM 7` original crop: **0.31% CER**, 1.55% WER, 250/295 (84.7%) exact, 0 empty, 137.1 ms.
    - `PSM 13` original crop: **0.47% CER**, 2.40% WER, 230/295 (78.0%) exact, 0 empty, 134.8 ms.
    - `PSM 7 + 10px` white margin: **2.01% CER**, 5.07% WER, 236/295 (80.0%) exact, **3 empty** (on `B12`), 133.8 ms.
    - `PSM 13 + 10px` white margin: **2.88% CER**, 14.02% WER, 92/295 (31.2%) exact, 0 empty, 124.2 ms.
    - `PSM 7 + 10px + 1.5×` scale: **1.92% CER**, 4.80% WER, 241/295 (81.7%) exact, 3 empty, 165.1 ms.
    - `PSM 13 + 10px + 1.5×` scale: **2.74% CER**, 12.72% WER, 98/295 (33.2%) exact, 0 empty, 154.4 ms.
    - `A Priori Pad10 PSM7 -> fallback PSM13`: **1.13% CER**, 4.67% WER, 236/295 (80.0%) exact, 0 empty.
  - **Real30 (`PSM 7` original crop):** Clean325 **22.33% CER** (`62/325 = 19.08%` exact, 0 empty); FullPage **35.39% CER** (`108/314` empty on noise, `65.61%` noise FP rate).
- **Evidence files:** `FINAL_TECHNICAL_EVIDENCE/TESSERACT_FAIRNESS_RESULTS.csv`, `FINAL_TECHNICAL_EVIDENCE/TESSERACT_VERSION_AUDIT.md`
- **Source code location:** `results/tesseract_fairness/tesseract_fairness_report.md`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch4_t10_to_t14.py`
- **Previous discrepancy:** Draft text referenced `Tesseract 5.5.0` in places; exact `PSM 7` vs `PSM 13` and `+10px` empty-output behavior needed verification.
- **Correct interpretation:** All Tesseract evaluations used `v5.4.0.20240606`. Adding 10px padding degrades `PSM 7` from `0.31%` to `2.01%` CER (with 3 empty outputs on `B12`).
- **Recommended manuscript location:** Section 3 (Baseline Comparisons — Tesseract Fairness)
- **Recommended supplementary table or section:** Supplementary Table S8
- **Remaining limitation:** None.

---

## T11 — EasyOCR B1–B15 (295 Lines) Results

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - Historical `EasyOCR 1.7.2` (`latin_g2`) on **B1–B15 (295 lines)**:
    - **CER:** `2.08%`
    - **WER:** `16.81%`
    - **Exact match:** `80 / 295 (27.1%)`
    - **Empty outputs:** `0`
    - **Mean latency:** `172.0 ms/line`
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/EASYOCR_B295_RESULTS.csv`
- **Source code location:** `results/easyocr_295/easyocr_295_results.json`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch4_t10_to_t14.py`
- **Previous discrepancy:** None (needed verified extraction from historical JSON).
- **Correct interpretation:** Extracted directly from `easyocr_295_results.json`.
- **Recommended manuscript location:** Section 3 (Baseline Comparisons on B1–B15)
- **Recommended supplementary table or section:** Supplementary Table S9
- **Remaining limitation:** None.

---

## T12 — Confidence AUROC & Cascade Routing Protocol

- **Verification status:** `VERIFIED`
- **Verified numerical results:**

| Dataset | Protocol | Mean-char AUROC | Min-char AUROC | Error (+) / Exact (-) | Routing Rate & Cutoff | CRNN CER (%) | Cascade CER (%) | Tess CER (%) | Cascade Latency / Speedup |
|---|---|---:|---:|---:|---|---:|---:|---:|---|
| Diagnostic 111 (`D1–D5`) | Post-hoc sweep (calibration) | **0.8475** | **0.8413** | 43 / 68 | 20% lowest `min_conf` | 1.08 | **0.47** | 0.32 | 212.8 ms (`2.00×` vs Tess) |
| Benchmark 295 (`B1–B15`) | 20% lowest `min_conf` routing | **0.6973** | **0.6344** | 42 / 253 | 20% (`59/295`, `cutoff=0.5435`) | 0.298 | **0.291** | 0.31 | 111.39 ms (`1.92×` vs Tess) |

- **Evidence files:** `FINAL_TECHNICAL_EVIDENCE/CONFIDENCE_AUROC_VERIFIED.csv`, `FINAL_TECHNICAL_EVIDENCE/CASCADE_PROTOCOL_AUDIT.md`
- **Source code location:** `results/confidence_audit/confidence_cascade_audit_results.json`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch4_t10_to_t14.py`
- **Previous discrepancy:** Mixing 111-line diagnostic AUROC (`0.8475` / `0.8413`) with 295-line benchmark AUROC (`0.6973` / `0.6344`).
- **Correct interpretation:** The 111-line set served as the calibration/diagnostic split; on the 295-line benchmark, `min_conf` AUROC drops to `0.6344` (`mean_conf` `0.6973`) because CRNN-V2 already achieves `0.298%` CER with only `42` non-exact lines, so routing the lowest 20% (`cutoff=0.5435`) yields `0.291%` CER at `1.92×` speedup over Tesseract-only.
- **Recommended manuscript location:** Section 3 (Confidence Estimation & Hybrid Cascade)
- **Recommended supplementary table or section:** Supplementary Table S10
- **Remaining limitation:** None.

---

## T13 — SymSpell Lexicon Provenance

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - `data/az_lexicon.txt`: **116,585** entries / **116,585** unique tokens, frozen `2026-10-29` (`git commit 51e807e8`).
  - `116,143` and `121,377` do not exist anywhere in the repository.
  - Benchmark leakage: **0** benchmark lines appear verbatim in the lexicon; benchmark token OOV rate = **14.89%**.
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/SYMSPELL_PROVENANCE_AUDIT.md`
- **Source code location:** `data/az_lexicon.txt`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch4_t10_to_t14.py`
- **Previous discrepancy:** Earlier drafts cited `116,143` or `121,377` lexicon entries.
- **Correct interpretation:** The exact frozen lexicon size is `116,585` unique tokens.
- **Recommended manuscript location:** Section 2 (Post-Processing — SymSpell Lexicon)
- **Recommended supplementary table or section:** Supplementary Note S3
- **Remaining limitation:** None.

---

## T14 — CPU Runtime Reconciliation

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - `85.71 ms` (`SD 23.58`, `Intel Core 7 240H`, 4 threads, `batch=1`, `warmup=10`, `3 reps`, in-memory `PyTorch 2.14.1+cpu` C++ backend — labeled `"libtorch"` in JSON, preprocess included, init `2209.9 ms` excluded).
  - `128.4 ms` (`SD 15.32`, same machine, `Tesseract v5.4.0.20240606 PSM7` via Windows CLI `pytesseract`, per-call spawn `64.82 ms`).
  - `86.3 ms` (`CRNN-V2_Raw`) and `86.4 ms` (`CRNN-V2_SymSpell`): single-pass B1–B15 timing in `tesseract_fairness_report.md`.
  - `182.2 ms`: single-pass Tesseract `PSM7` B1–B15 timing in `eval_easyocr_benchmark295.py:284`.
  - `155.35 ms` (`155.4 ms`, `batch=1`) and `127.59 ms` (`127.6 ms`, `batch=16`): legacy 33-line Photo 5 benchmark on `Intel64 Model 142 Step 9` (`results/efficiency_benchmark/metrics.json`).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/CPU_RUNTIME_RECONCILIATION.csv`
- **Source code location:** `results/speed_benchmark/cpu_speed_benchmark_results.json`, `results/tesseract_fairness/tesseract_fairness_report.md`, `scripts/eval_easyocr_benchmark295.py:284`, `results/efficiency_benchmark/metrics.json`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch4_t10_to_t14.py`
- **Previous discrepancy:** Multiple latency numbers (`85.71`, `86.3`, `127.6`, `155.4`, `128.4`, `182.2 ms`) without hardware/batch/invocation context, plus `"libtorch"` ambiguity.
- **Correct interpretation:** All 7 numbers are genuine measurements from distinct hardware/batch/invocation protocols; `"libtorch"` refers to the PyTorch C++ backend (`torch 2.14.1+cpu`) invoked from Python, not a standalone C++ binary.
- **Recommended manuscript location:** Section 3 (Computational Efficiency & CPU Runtime)
- **Recommended supplementary table or section:** Supplementary Table S11
- **Remaining limitation:** None.

---

## T15 — Real30 Clean325 21–22% CER Floor Root Cause

- **Verification status:** `COMPUTED_FROM_EXISTING_DATA`
- **Verified numerical results:**
  - Across **all 5 inspected engines** on `Real30_Clean325` (`325` crops, `25,180` GT chars), **73.49%–84.30%** of all character edits are **deletions**, and **55.40%–59.21%** of those deletions occur at the **left/right edges** (first/last 15 alignment ops):

| Model | Clean325 CER (%) | Substitutions | Deletions | Insertions | Deletion Share (%) | Edge Deletions (First/Last 15 ops) |
|---|---:|---:|---:|---:|---:|---:|
| `Baseline_CRNN-V2_Raw` | 22.15 | 906 | 4,099 | 573 | 73.49% | 2,402 (58.60%) |
| `Baseline_Tesseract5_PSM7` | 22.33 | 640 | 4,740 | 243 | 84.30% | 2,626 (55.40%) |
| `Baseline_EasyOCR` | 21.81 | 891 | 4,133 | 467 | 75.27% | 2,414 (58.41%) |
| `Baseline_PaddleOCR` | 21.08 | 753 | 4,379 | 176 | 82.50% | 2,533 (57.84%) |
| `Exp_RandomPadding_GroupD_seed42` | 21.39 | 916 | 4,011 | 460 | 74.46% | 2,375 (59.21%) |

- **Evidence files:** `FINAL_TECHNICAL_EVIDENCE/CLEAN325_ERROR_ANALYSIS.md`, `FINAL_TECHNICAL_EVIDENCE/REAL30_INTEGRITY_AUDIT.csv`, `FINAL_TECHNICAL_EVIDENCE/REAL30_CLEAN325_60SAMPLE_INSPECTION.csv`
- **Source code location:** `Real30_GPU18_Final_Evaluation/verified_alignment.csv`, `segment_lines_with_boxes`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch5_t15_t16_t17.py`
- **Previous discrepancy:** Unexplained ~21–22% CER floor across both AzOCR-V2 and all commercial/open-source baselines on `Real30_Clean325`.
- **Correct interpretation:** Upstream horizontal paper-strip clipping in `segment_lines_with_boxes` (`col > 0.78 * percentile(col, 90)`) truncates darkened left/right page margins, physically removing 1–2 edge words from many Clean325 crops before any OCR engine sees the image.
- **Recommended manuscript location:** Section 3 / Discussion (Real30 Error Analysis & Segmentation Bottleneck)
- **Recommended supplementary table or section:** Supplementary Table S12 & Supplementary Note S4
- **Remaining limitation:** Computed from existing alignment/crop artifacts without re-segmenting Real30.
- **Publication-ready English text:**
  > Error decomposition across all five evaluated engines on `Real30_Clean325` (`325` lines, `25,180` characters) reveals that `73.49%–84.30%` of all edit operations are deletions, with `55.40%–59.21%` of deletions concentrated in the first or last 15 alignment positions. Inspection of 60 representative crops confirms that upstream horizontal margin clipping in `segment_lines_with_boxes` physically truncates edge words on shadowed smartphone pages, establishing a shared segmentation-induced CER floor (`21.08%–22.33%`) across CRNN, Tesseract, EasyOCR, and PaddleOCR.

---

## T16 — Character-Level 95% Confidence Intervals

- **Verification status:** `COMPUTED_FROM_EXISTING_DATA`
- **Verified numerical results:**
  - Computed exact **Wilson** and **Clopper–Pearson** 95% confidence intervals across **72 rows** covering `Real30_Clean325` (325 independent crops) and `R59_Synthetic200` (200 images across 40 sentences × 5 fonts) for `ə, Ə, ı, i, İ, I`.
  - Representative `Real30_Clean325` values (`Baseline_CRNN-V2_Raw`):
    - `ə` (`n=2,434`): `1,987/2,434 = 81.64%` (Wilson `[80.05, 83.12]`, CP `[80.04, 83.16]`).
    - `Ə` (`n=7`): `6/7 = 85.71%` (Wilson `[48.69, 97.43]`, CP `[42.13, 99.64]`).
    - `ı` (`n=906`): `726/906 = 80.13%` (Wilson `[77.41, 82.60]`, CP `[77.38, 82.68]`).
    - `i` (`n=1,968`): `1,597/1,968 = 81.15%` (Wilson `[79.36, 82.82]`, CP `[79.35, 82.85]`).
    - `İ` (`n=24`): `14/24 = 58.33%` (Wilson `[38.83, 75.53]`, CP `[36.64, 77.89]`).
    - `I` (`n=0`): `N/A` (0 occurrences in Clean325 GT).
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/CHARACTER_CONFIDENCE_INTERVALS.csv`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch5_t15_t16_t17.py`
- **Previous discrepancy:** Point estimates lacked 95% CIs, and low-support uppercase characters (`Ə` n=7, `İ` n=24, `I` n=0) were easily over-interpreted.
- **Correct interpretation:** Single-model binomial CIs show wide uncertainty for `Ə` (`n=7`) and `İ` (`n=24`), and `I` (`n=0`) is unmeasured in `Clean325`. `R59_Synthetic200` CIs are flagged as conditional on the 5-font × 40-sentence clustered design.
- **Recommended manuscript location:** Section 3 (Azerbaijani Diacritic & Case Accuracy)
- **Recommended supplementary table or section:** Supplementary Table S13 (Full 72-row Wilson & Clopper–Pearson 95% CI table)
- **Remaining limitation:** Single-seed binomial CIs (3 seeds not pooled); R59 observations are clustered by 40 source sentences × 5 fonts.

---

## T17 — Zenodo DOI & Archive Reproducibility

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - `10.5281/zenodo.24529456` — **HTTP 200 OK** (`mit-license`): `AzOCR_v1_Code_Benchmarks_Results.zip`, size = `3,284,358` bytes (`3.13 MB`, 301 files), hosted `md5:6f79076aef147a65788f83fda5e25195`, local SHA-256 `838b46e435a19ada83b29e7b2948f183faf2a2f38d99abf3bd54982734dad8f9`.
  - `10.5281/zenodo.24529882` — **HTTP 200 OK** (`apache2.0`): `AzOCR_v1_Models_18Official.zip`, size = `1,648,511,501` bytes (`1572.14 MB`, 20 files), hosted `md5:14a852df8e62c9c7e391c12e695f7fe1`, local SHA-256 `b28069dfcd3199b6c3725a61ab811c2970e80410a39a8ff35d87c29d3ec5366a`.
- **Evidence file:** `FINAL_TECHNICAL_EVIDENCE/ZENODO_REPRODUCIBILITY_AUDIT.md`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch5_t15_t16_t17.py`
- **Previous discrepancy:** Needed live verification that both Zenodo records resolve publicly and match the staged release archives.
- **Correct interpretation:** Both Zenodo DOIs are live (`HTTP 200 OK`) and match local archive byte sizes. Note that the Zenodo REST API exposes `md5:...` checksums whereas local release manifests record `sha256:...`.
- **Recommended manuscript location:** Data & Code Availability
- **Recommended supplementary table or section:** Supplementary Note S5
- **Remaining limitation:** Zenodo API reports `md5` rather than `sha256`.

---

## T18 — GPU18 Training Curves (`S25`–`S30`)

- **Verification status:** `VERIFIED`
- **Verified numerical results:**
  - **434** per-epoch training records extracted across all **18 official runs** (`S25` Group A, `S26` Group B, `S27` Group C, `S28` Group D, `S29` RandomPadding, `S30` VariableWidth), recording `best_epoch`, `last_epoch`, `early_stopped`, `best_val_cer_pct`, `final_train_loss`, and `final_val_loss`.
- **Evidence files:** `FINAL_TECHNICAL_EVIDENCE/GPU18_TRAINING_CURVES_VERIFIED.csv`, `FINAL_TECHNICAL_EVIDENCE/TRAINING_HISTORY_AUDIT.md`
- **Executed command:** `python FINAL_TECHNICAL_EVIDENCE/run_batch1_t01_t02_t03_t18.py`
- **Previous discrepancy:** Need verified epoch-by-epoch logs backing Supplementary Figures `S25`–`S30`.
- **Correct interpretation:** All 18 runs' epoch histories are intact and mapped cleanly to `S25`–`S30`.
- **Recommended manuscript location:** Section 2 / Results (Training Convergence)
- **Recommended supplementary table or section:** Supplementary Figures `S25`–`S30` & Supplementary Table S14
- **Remaining limitation:** None.
