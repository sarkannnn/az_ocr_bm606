# Simulated Phone Photo Distortion Ladder: Diagnostic Report

## 1. Summary Across Distortion Levels

| Level | Description | Word CRNN CER | Word CRNN WER | Line CRNN CER | Line CRNN WER | Tess Word CER | Tess Line CER | n(word) | n(line) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0** | Clean render (pure white, black text) | 7.71% | 27.75% | 1.87% | 11.14% | 0.41% | 0.22% | 346 | 92 |
| **L0c** | Clean render (direct axis-aligned slice) | 7.71% | 27.75% | 1.87% | 11.14% | 0.41% | 0.22% | 346 | 92 |
| **L1** | + Paper color, texture, illumination gradient, shadow | 8.85% | 32.08% | 4.10% | 22.56% | 0.45% | 0.22% | 346 | 92 |
| **L2** | L1 + Gaussian blur, sensor noise | 25.00% | 57.80% | 13.84% | 37.60% | 0.41% | 0.29% | 346 | 92 |
| **L3** | L2 + Downscale/upscale, JPEG compression, tint | 31.36% | 67.92% | 19.14% | 42.62% | 0.41% | 0.22% | 346 | 92 |
| **L4** | L3 + Perspective jitter, rotation (Full phone sim) | 35.77% | 75.72% | 12.96% | 36.21% | 0.37% | 0.33% | 346 | 92 |
| **L4n** | L4 + Per-crop min-max contrast stretching | 24.43% | 67.92% | 7.43% | 25.91% | 0.37% | 0.33% | 346 | 92 |

---

## 2. Key Findings & Distortion Impact

- **Baseline on Clean Unseen Fonts (L0 vs L0c):**
  - Word model: L0 CER = 7.71%, L0c CER = 7.71%. Difference: 0.00%.
  - Line model: L0 CER = 1.87%, L0c CER = 1.87%. Difference: 0.00%.
  - Both L0 and L0c show excellent accuracy on unseen fonts when the background is clean white.

- **Where CER First Jumps (L0 -> L1):**
  - Word CRNN CER jumps from **7.71%** to **8.85%** (+1.14% percentage points).
  - Line CRNN CER jumps from **1.87%** to **4.10%** (+2.23% percentage points).
  - **Conclusion:** Illumination variation, shadow, and non-white paper background constitute the single largest point of failure for the CRNN models.

- **Effect of Contrast Stretching (L4 vs L4n):**
  - Word CRNN CER drops from **35.77%** down to **24.43%** (-11.34% points).
  - Line CRNN CER drops from **12.96%** down to **7.43%** (-5.53% points).
  - Contrast normalization alone recovers a massive fraction of performance without any retraining.

## 3. Per-Font CER Breakdown (L0 vs L4)

Separates the font effect (unseen typography) from the distortion effect:

| Font | Category | L0 Word CER | L4 Word CER | L0 Line CER | L4 Line CER | n(words) | n(lines) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CascadiaMono.ttf** | Monospace (Cascadia Mono) | 11.94% | 51.08% | 6.71% | 22.21% | 83 | 23 |
| **cambria.ttc** | Serif (Cambria) | 5.24% | 28.89% | 0.31% | 7.81% | 81 | 22 |
| **micross.ttf** | Sans (MS Sans Serif) | 6.77% | 31.54% | 0.00% | 9.45% | 98 | 24 |
| **pala.ttf** | Serif (Palatino) | 6.92% | 31.80% | 0.59% | 12.54% | 84 | 23 |

---

## 4. Input Preprocessing Diagnostic (from common.py)

In `common.py` (`preprocess(img, cfg)`), the pipeline performs:
1. BGR to grayscale conversion (`cv2.COLOR_BGR2GRAY`).
2. No filter applied (`cfg['filter'] == 'none'`) and NO binarization (`cfg['thresh'] == 'none'`).
3. Aspect-ratio preserving bilinear resize to fixed height (32 px) with width capped at preset (128 for word, 512 for line).
4. Right-side white canvas padding (padded with constant 255).
5. Raw pixel division by 255.0 to float32 `[0, 1]`. **Crucially: No min-max stretching, adaptive thresholding, or z-score standardization is applied.** Therefore, darker paper or shadow shifts the background from 1.0 down to ~0.6-0.7, which breaks the CNN activation scale learned on synthetic white backgrounds.

---

## 5. Sample Word Model Predictions at L1 (Failures)

| # | Ground Truth (Ref) | CRNN Prediction (Hyp) | Match? |
| :-: | :--- | :--- | :-: |
| 1 | `1927-ci` | `IzzAci` | ❌ |
| 2 | `Ensiklopediyası.` | `Lnsiklopediyası` | ❌ |
| 3 | `həbsə` | `həbşə` | ❌ |
| 4 | `tikmək,` | `tikmək` | ❌ |
| 5 | `bayramı.` | `bayram` | ❌ |
| 6 | `(BMTQAK` | `ÇBMTQAK` | ❌ |
| 7 | `kimi` | `kü` | ❌ |
| 8 | `ordusunun` | `ordusdqnqn` | ❌ |
| 9 | `idi.` | `idi` | ❌ |
| 10 | `məşhurdur:` | `məşhurdur` | ❌ |

## 6. Sample Word Model Predictions at L0

| # | Ground Truth (Ref) | CRNN Prediction (Hyp) | Match? |
| :-: | :--- | :--- | :-: |
| 1 | `1927-ci` | `Izzzci` | ❌ |
| 2 | `Ensiklopediyası.` | `Lnsiklopediyası` | ❌ |
| 3 | `tikmək,` | `tikmək` | ❌ |
| 4 | `bayramı.` | `bayramı` | ❌ |
| 5 | `(BMTQAK` | `ÇBMTQAK` | ❌ |
| 6 | `idi.` | `idi` | ❌ |
| 7 | `məşhurdur:` | `məşhurdur` | ❌ |
| 8 | `ördəklər,` | `ördəklər` | ❌ |
| 9 | `qazlar,` | `qazlar` | ❌ |
| 10 | `(film,` | `Çfilm` | ❌ |
