# AzOCR-V2 — R23: Leave-One-Out Augmentation Ablation Plan (`R23_ABLATION_PLAN.md`)

**Status:** Design & Cost Analysis Complete (**No GPU training launched**)  
**Reference Source:** [dataset.py](file:///D:/laragon/www/az_ocr_bm606/AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py), [train_gpu.py](file:///D:/laragon/www/az_ocr_bm606/AzOCR_Final_Scientific_Audit/code_snapshot/train_gpu.py), Group D `config.yaml`

---

## 1. Group D Baseline Configuration

From `Group_D_V2Clean_PhotoAug_seed{42,43,44}/config.yaml`:
- **Dataset:** `data/synth_line_v2_clean/train` and `val` (18 fonts, deduplicated clean corpus)
- **Augmentation Mode:** `aug_mode: photo` (`augment_photo` in `dataset.py`)
- **Canvas & Batching:** Fixed `height=32, width=512`, `batch_size=64`, `dynamic_width: false`, `random_padding: 0`
- **Optimization:** `epochs: 30`, `patience: 7`, `lr: 0.001`, `weight_decay: 1.0e-05`, `dropout: 0.2`
- **Seeds:** `42, 43, 44`

---

## 2. Exact Decomposition of `PhotoAug` (`augment_photo`) Operations

From `AUG_CONFIG` and `augment_photo()` in `dataset.py` (lines 19–40, 69–166), `PhotoAug` applies **11 sequential independent operations**:

| # | Operation Name | Config Key & Prob ($p$) | Exact Parameters | In `BasicAug`? |
|---:|---|---:|---|---|
| 1 | **Rotation** | `rotation_prob = 0.50` | `rotation_range = (-5.0°, +5.0°)`, `BORDER_REPLICATE` | Yes (identical: $p=0.5, \pm 5°$) |
| 2 | **Perspective Jitter** | `perspective_prob = 0.30` | 4-corner jitter $\pm 0.02W \times \pm 0.08H$ | Yes (identical: $p=0.3$) |
| 3 | **Illumination Gradient & Shadow Band** | `background_tint_prob = 0.50` | Linear angle gradient (`0.7–1.0`) + $50\%$ sub-prob Gaussian shadow band (`bw=0.2–0.5`, depth `0.12–0.28`) | **PhotoAug-only** |
| 4 | **Brightness / Contrast** | `brightness_contrast_prob = 0.60` | $\alpha \in [0.6, 1.1]$, $\beta \in [-40.0, +20.0]$ | **PhotoAug-only** |
| 5 | **Gaussian Blur** | `gaussian_blur_prob = 0.60` | Continuous $\sigma \in [0.5, 2.0]$ | Partial (`BasicAug`: $k \in \{3,5\}, p=0.3$) |
| 6 | **Motion Blur** | `motion_blur_prob = 0.15` | Directional kernel $k \in [3, 7]$, random angle $[0, \pi]$ | **PhotoAug-only** |
| 7 | **Gaussian Sensor Noise** | `sensor_noise_prob = 0.60` | Additive $\mathcal{N}(0, \sigma)$, $\sigma \in [3.0, 12.0]$ | **PhotoAug-only** |
| 8 | **Salt-and-Pepper Noise** | `salt_pepper_prob = 0.20` | $0.5\%$ black + $0.5\%$ white pixels | Yes (`BasicAug`: $p=0.3$) |
| 9 | **Downscale & Upscale** | `downscale_prob = 0.40` | `scale` $\in [0.5, 0.9]$, `INTER_AREA` $\to$ `INTER_LINEAR` | **PhotoAug-only** |
| 10 | **JPEG Compression** | `jpeg_prob = 0.40` | `IMWRITE_JPEG_QUALITY` $\in [40, 90]$ | **PhotoAug-only** |
| 11 | **Contrast Stretch** | `contrast_stretch_prob = 0.50` | Grayscale `NORM_MINMAX` $[0, 255]$ | **PhotoAug-only** |

---

## 3. What Existing Models Already Contrast

| Existing Comparison | Isolated Factor | What It Does NOT Isolate |
|---|---|---|
| **Group B vs. Group A** (`V1Clean`) & **Group D vs. Group C** (`V2Clean`) | Bundle of `PhotoAug` (11 ops) vs. `BasicAug` (4 ops) | Individual contribution of each of the 7 PhotoAug-exclusive ops |
| **Exp_RandomPadding_GroupD vs. Group D** | Random border padding (`random_padding > 0` vs `0`) | Sub-operations inside `augment_photo` |
| **Exp_VariableWidth_GroupD vs. Group D** | Dynamic width collation (`collate_fn_dynamic` vs fixed 512) | Sub-operations inside `augment_photo` |
| **Group C/D vs. Group A/B** | Font diversity (`V2Clean` 18 fonts vs `V1Clean` 8 fonts) | Photometric operations |

---

## 4. Required Model Count for 3-Seed Leave-One-Out (LOO)

- **Option A — Full 11-Operation LOO (Strict):**
  - 11 LOO variants × 3 seeds (`42, 43, 44`) = **33 new models** (compared against existing 3 `Group D` baseline models).
- **Option B — 7 PhotoAug-Exclusive Operation LOO (Practical):**
  - Leaving out each of the 7 operations absent from `BasicAug` (#3, #4, #6, #7, #9, #10, #11) × 3 seeds = **21 new models**.
- **Option C — 5 Functional Family LOO (Grouped):**
  - Families: (1) Geometric `{#1, #2}`, (2) Photometric/Illumination `{#3, #4, #11}`, (3) Blur `{#5, #6}`, (4) Noise `{#7, #8}`, (5) Resolution/Codec `{#9, #10}` × 3 seeds = **15 new models**.

---

## 5. Fair Experimental Matrix (All Other Conditions Identical)

Every LOO run must hold constant:
- **Training/Validation Data:** `data/synth_line_v2_clean/{train,val}`
- **Architecture & Hyperparameters:** `CRNN`, `epochs=30`, `patience=7`, `batch_size=64`, `lr=0.001`, `weight_decay=1e-5`, `dropout=0.2`, `dynamic_width=false`, `random_padding=0`
- **Seeds:** `42, 43, 44`
- **Evaluation:** Fixed `infer_line_smart` on Real30 (`Clean325` + `Full Page Primary`)

| Experiment ID | Dropped Operation (`prob = 0.0`) | Seeds | New Runs | Est. GPU Time (RTX 2050, ~3h/run) |
|---|---|---|---:|---:|
| `LOO_01_NoRotation` | `rotation_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_02_NoPerspective` | `perspective_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_03_NoIllumGradient` | `background_tint_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_04_NoBrightContrast` | `brightness_contrast_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_05_NoGaussianBlur` | `gaussian_blur_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_06_NoMotionBlur` | `motion_blur_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_07_NoSensorNoise` | `sensor_noise_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_08_NoSaltPepper` | `salt_pepper_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_09_NoDownscale` | `downscale_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_10_NoJPEG` | `jpeg_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| `LOO_11_NoContrastStretch` | `contrast_stretch_prob = 0.0` | 42, 43, 44 | 3 | ~9 h |
| **Total (Full LOO)** | — | **33 runs** | **33** | **~99 GPU-hours** |

---

## 6. How to Present R23 as a Limitation in the Paper

If full 33-model LOO training (~99 GPU-hours) is not executed due to compute constraints:
> *"While our 2×2 factorial design (Groups A–D) establishes that the composite `PhotoAug` pipeline drives real-world smartphone generalization—reducing Clean325 CER by 67.82 pp on V1Clean ($p=0.009$) and 10.25 pp on V2Clean—a full 3-seed leave-one-out ablation across all 11 individual photometric sub-operations would require 33 additional training runs (~99 GPU-hours). Consequently, the marginal contribution of each individual sub-transform (e.g., illumination gradient vs. JPEG compression vs. sensor noise) is not isolated in this study and remains an open direction for future ablation."*

