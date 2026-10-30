# Deep Learning-Based Optical Character Recognition for Azerbaijani: Architecture, Photometric Augmentation, Font Leakage Audits, and Real-World Document Stress-Testing

**Authors:** Azerbaijani OCR Research Initiative (Project BM 606)  
**Affiliations:** Baku / Nakhchivan State University Academic Collaborative Project  
**Date:** October 30, 2026  
**Document Status:** Publication-Style Research Monograph and Empirical Audit Report  

---

## Abstract

Optical Character Recognition (OCR) systems face acute challenges when applied to morphologically rich, resource-constrained languages with dense diacritic inventories such as Azerbaijani ($\Sigma_{az}$ includes `ə, ı, ö, ğ, ü, ş, ç` and distinct dotted/dotless uppercase and lowercase pairs `i/İ` and `ı/I`). While standard OCR engines such as Tesseract 5 perform robustly on scanned documents, their baseline detectors and language priors exhibit severe degradation when exposed to uncontrolled mobile smartphone captures, non-standard aspect ratios, and unseen typefaces. In this monograph, we design, train, and empirically audit an end-to-end CRNN (Convolutional Recurrent Neural Network with Connectionist Temporal Classification, ~8.23M parameters) trained strictly under CPU-only compute constraints. We systematically contrast a baseline model (V1: 8 seen fonts, mild spatial augmentation) against a photometrically enhanced model (V2: 14 fonts, illumination gradients, sensor noise, motion blur, resolution scaling, and contrast stretching). 

Through a controlled multi-stage evaluation ladder ($L_0$ to $L_4$), genuine unseen font testing (7 new families, $n=609$ words, $n=158$ lines), and five real smartphone document captures (Photos 1–5, up to 33 lines, 2,328 characters), our key empirical findings are:
1. **Synthetic Benchmarks:** On clean mixed synthetic tests, V2 reduces Character Error Rate (CER) from 0.50% to **0.11%** (Word, $n=2,000$) and 1.64% to **0.30%** (Line, $n=1,000$), compared to Tesseract 5 at 0.70% and 0.44% (`results/word_v2_synth/metrics.json`, `results/line_v2_synth/metrics.json`).
2. **Simulation Ladder Resilience:** Under simulated smartphone optical distortions ($L_4$), baseline V1 collapses to **35.77%** Word CER and **12.96%** Line CER, whereas V2 maintains **1.35%** Word CER and **0.11%** Line CER (`results/sim_ladder/summary.md:12`).
3. **Generalization Audit & Font Leakage:** We refute the illusion of zero-shot generalization on Consolas (Word CER 1.02% $\rightarrow$ 0.07%), proving it resulted from font family leakage (`consolab.ttf` in V2 training). On truly unseen font families (Group C), cluster bootstrap 95% confidence intervals for line recognition include zero ($\Delta\text{CER} = +0.00\%$ [-0.74%, +0.76%] at $L_0$, and $-0.17\%$ [-2.60%, +1.53%] at $L_4$), indicating V2 is on par with Tesseract on standard continuous text.
4. **Real Smartphone Document stress-testing:** On Photo 5 ($n=33$ lines, 2,328 characters), Tesseract with `--psm 7` failed on 9 narrow chunks and 2 full lines (returning empty strings `""`, inflating CER to 13.02%). In contrast, V2 Line with projection chunking achieved **2.10% CER**. However, on the 23-line clean subset where Tesseract does not fail, V2 achieves **1.32% CER** vs. Tesseract's **1.26% CER** ($\Delta = +0.06\%$ [-0.77%, +0.80%], spanning zero). When Tesseract is configured with `--psm 13`, it attains near-perfect recognition (**0.09% CER**).
5. **Fragilities & Methodological Pitfalls:** We document five critical project pitfalls: (i) early TRDG bottom-edge glyph clipping, (ii) line collapse when space was missing from vocabulary (CER 11.19%, WER 99.75%), (iii) bounding box metric inflation in `SansSerifCollection` (84.32% $\rightarrow$ 0.76%), (iv) aspect ratio squeeze degradation for ratios $r > 1.3$ (Full Line 4.87% $\rightarrow$ Chunked 1.33%), and (v) severe border padding fragility, where adding a 20 px white border causes V2 CER to collapse from 2.10% to **23.32%**. Furthermore, we disclose that a pilot heuristic lexicon corrector that reduced Photo 5 CER from 2.10% to 1.80% suffered from test-set snooping.

**Keywords:** Optical Character Recognition (OCR), Azerbaijani Alphabet, CRNN, BiLSTM, CTC Loss, Photometric Augmentation, Font Leakage Audit, Bootstrap Confidence Intervals, Tesseract 5, Page Segmentation Mode.

---

## 1. Introduction

Optical Character Recognition (OCR) constitutes an essential pillar of document digitization, information retrieval, and administrative archival pipelines. While modern commercial and open-source OCR engines (e.g., Tesseract 5) demonstrate state-of-the-art performance on major world languages printed under scanned flatbed conditions, their recognition fidelity frequently degrades when deployed on languages featuring complex diacritics and limited digitized training corpora.

The modern Azerbaijani alphabet utilizes a 32-letter Latin script:
$$\Sigma_{az} = \{a, b, c, ç, d, e, ə, f, g, ğ, h, x, ı, i, j, k, q, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z\}$$

This orthographic system introduces three severe computer vision challenges:
1. **Diacritic Subtlety:** Pairs such as $c/\text{ç}$, $s/\text{ş}$, $g/\text{ğ}$, $o/\text{ö}$, and $u/\text{ü}$ differ by minute structural components—specifically, lower cedillas, carons/breves, and umlaut dots. Under smartphone camera defocus, motion blur, or uneven lighting, these high-frequency pixel details are readily obliterated.
2. **The Dotted and Dotless I Paradox:** Unlike English, Azerbaijani distinguishes lower-case dotless $\text{ı}$ ($U+0131$) from dotted $i$ ($U+0069$), with uppercase counterparts $I$ ($U+0049$) and $\text{İ}$ ($U+0130$). Generic Latin OCR models frequently conflate these vowels.
3. **The Distinctive Schwa ($\text{ə}/\text{Ə}$):** Representing an open central unrounded vowel, the schwa is among the most frequent characters in Azerbaijani orthography. Visual distortion easily causes confusion with $e$, $o$, or $a$.

Traditional OCR solutions (e.g., legacy Tesseract) rely on binarization heuristics (Otsu thresholding) and explicit character segmentation. However, when applied to handheld mobile phone photographs, non-uniform page illumination, shadow bands, perspective skew, paper curvature, and ink bleed cause binarization algorithms to fragment or merge character glyphs.

Modern connectionist approaches resolve this limitation by modeling OCR as a sequence-to-sequence visual transcription task. In this project, we implement an end-to-end CRNN (Convolutional Recurrent Neural Network) architecture trained with Connectionist Temporal Classification (CTC) loss [Shi et al., 2016; Graves et al., 2006]. The model maps raw grayscale image tensors directly to character sequences without explicit character segmentation.

### 1.1 Scope and Methodological Integrity

This monograph provides an exhaustive scientific evaluation of the Azerbaijani OCR system developed under Project BM 606. In accordance with strict empirical principles:
- Every quantitative claim is referenced to verifiable repository log files, metrics tables, and execution traces.
- All post-hoc hyperparameter adjustments and data snooping risks are disclosed.
- Negative and awkward findings (e.g., V2 padding fragility, width squeeze collapse, font family leakage) are presented with the same analytical prominence as positive benchmarks.
- No claim of "V2 beats Tesseract" is asserted on any dataset unless backed by a bootstrap confidence interval that strictly excludes zero.

```
[Figure 1: End-to-End System Architecture & Experimental Pipeline]
```
![Figure 1](figures/f1_pipeline.png)  
*Figure 1: End-to-End Azerbaijani OCR system architecture and multi-tier evaluation pipeline. From left to right: Corpus collection and pre-split balancing; TRDG synthetic image rendering with glyph filtering; Photometric data augmentation; 7-layer CNN + 2-layer BiLSTM CRNN architecture; CTC decoding; Multi-faceted empirical audits spanning synthetic tests, simulation ladders, and real smartphone photos.*  
*Source: [prepare_corpus.py](file:///d:/laragon/www/az_ocr_bm606/prepare_corpus.py), [generate_data.py](file:///d:/laragon/www/az_ocr_bm606/generate_data.py), [train_v2.py](file:///d:/laragon/www/az_ocr_bm606/train_v2.py), [model.py](file:///d:/laragon/www/az_ocr_bm606/model.py), [results/real_fix9/report.md](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/report.md)*

---

## 2. Related Work

The foundational architecture for deep sequence reading without character segmentation was introduced by Graves et al. [2006] through Connectionist Temporal Classification (CTC), enabling recurrent networks to align sequential predictions over unsegmented inputs. Shi et al. [2016] unified deep convolutional feature extractors (VGG-style backbones) with bidirectional recurrent layers (BiLSTM) and CTC loss into the Convolutional Recurrent Neural Network (CRNN), which remains the standard lightweight baseline for text line recognition.

Tesseract OCR, originally engineered by HP [Smith, 2007] and subsequently maintained by Google, introduced an LSTM-based line recognition engine in version 4.0 and 5.0 (`tessdata_best`). While Tesseract provides pre-trained models for Azerbaijani (`aze`), its internal layout analyzer and page segmentation modes (PSM) remain tuned primarily for clean scanned page geometry.

Methodological benchmarking in scene text recognition has been critically analyzed by Baek et al. [2019], who demonstrated that reported accuracy gains frequently stem from inconsistent training sets, unstandardized evaluation subsets, or subtle data leakage rather than architectural innovation. In this study, we adhere to Baek et al.'s call for rigorous dataset standardization and error decomposition.

*(Note: Additional citations on Turkic and Caucasian OCR systems are noted as `[to verify]` pending comprehensive external bibliographic cross-referencing; non-repository claims have been excised in accordance with repository integrity rules).*

---

## 3. Data Engineering and Synthetic Generation

To train our models under CPU-only constraints without access to large-scale manually annotated real document corpora, we engineered a scalable synthetic pipeline.

### 3.1 Text Corpus Preparation (`prepare_corpus.py`)

The textual backbone was extracted from the Azerbaijani Wikipedia dump (`wikimedia/wikipedia:20231101.az`). The raw text was normalized to Unicode NFC format and filtered against an allowed character set ($86$ distinct characters: 32 lowercase and 32 uppercase Azerbaijani letters, digits `0-9`, standard punctuation `.,-:;!?()"\'`, and whitespace).

To strictly prevent data snooping:
1. **Pre-Rendering Split:** The corpus was partitioned at the text level into 80% train, 10% validation, and 10% test sets *prior* to synthetic image generation:
   $$\mathcal{D}_{\text{corpus}} = \mathcal{D}_{\text{train}} (80\%) \cup \mathcal{D}_{\text{val}} (10\%) \cup \mathcal{D}_{\text{test}} (10\%)$$
2. **Word Types vs. Word Tokens:** In word mode, deduplication was performed to partition unique word types. In line mode, deduplication was performed across sentences before chunking into lines of 12–35 characters. In `results/v2_audit3/summary.md:148-151`, leakage verification confirmed **0 out of 294 test lines** appeared in the training set (0.0% line leakage). Natural word token overlap was measured at **22.40%** (276 out of 1,232 word tokens), representing natural lexical frequency across non-overlapping sentences.
3. **Diacritic Balancing (`balance_special`):** Because diacritics represent primary error sources, the corpus generator enforced that samples containing special characters (`ə, ı, ö, ğ, ü, ş, ç`) constituted at least 50% of the training word pool and ~90% of the training line pool (`logs/line_train.log`).

### 3.2 Font Selection and Glyph Filtering (`generate_data.py`)

Candidate `.ttf`, `.otf`, and `.ttc` font files were programmatically inspected using `fontTools.ttLib.TTFont`. A font was accepted if and only if its character mapping table (`cmap`) contained Unicode code points for all 32 Azerbaijani lowercase and uppercase characters (`SPECIAL_ALL = "əıöğüşçƏİÖĞÜŞÇ"`). As shown in Table 2, multiple standard system fonts (`georgia.ttf`, `trebuc.ttf`, `constan.ttf`, `framd.ttf`, `lucon.ttf`) were rejected due to absent glyphs for `ə` and `Ə`.

- **V1 Baseline Dataset (`synth_word`, `synth_line`):** Generated using 10 system fonts. Following `collect_fonts(seed=42)`, 8 fonts were allocated to training and validation (`arial, calibri, candara, corbel, cour, segoeui, times, verdana`), while 2 fonts (20%) were strictly reserved for held-out testing (`tahoma.ttf` and `consola.ttf`).
  - Total volume: 20,000 train, 2,000 val, 2,000 test word images ($32\times 128$); 10,000 train, 1,000 val, 1,000 test line images ($32\times 512$).
- **V2 Augmented Dataset (`synth_word_v2`, `synth_line_v2`):** To enhance typographic diversity, 6 additional font files from `fonts_new/` (`CascadiaCode.ttf, bahnschrift.ttf, consolab.ttf, ebrima.ttf, l_10646.ttf, palab.ttf`) were added to the training set (+15,000 word images, +6,000 line images), expanding the training sets to **35,000 words** and **16,000 lines**. Validation and test sets were preserved as identical copies of V1.

```
[Figure 2: Data Sample Sheet & Glyph Boundary Verification]
```
![Figure 2](figures/f2_data_samples.png)  
*Figure 2: Data sample sheet and boundary verification. Top rows: Representative clean synthetic word and line crops with ground truth labels. Bottom left: Demonstration of the early TRDG bottom-clipping defect (cedilla truncated on ç and ş) vs. the patched rendering with +4 px safe margin. Bottom right: Summary metrics from automatic boundary verification (verify_quality.py, 0/500 border ink touch) and resolution of the SansSerifCollection bounding box inflation defect.*  
*Source: [verify_quality.py](file:///d:/laragon/www/az_ocr_bm606/verify_quality.py), [results/data_check.png](file:///d:/laragon/www/az_ocr_bm606/results/data_check.png), [results/v2_audit3/summary.md](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/summary.md)*

---

## 4. Models and Training Methodology

### 4.1 Neural Network Architecture (`model.py`)

The recognition engine is structured as a 7-layer Convolutional Recurrent Neural Network (CRNN):
1. **Input Preprocessing:** Inputs are single-channel grayscale float tensors ($1 \times H \times W$), where $H=32$ and $W=128$ for words, or $W=512$ for lines. Aspect ratios are preserved via bilinear resizing to height 32, with the remaining horizontal width padded with pure white canvas (value 1.0).
2. **CNN Backbone:** 7 convolution stages utilizing $3\times 3$ kernels, Batch Normalization, and ReLU activations. MaxPool layers compress the vertical axis from $32 \rightarrow 16 \rightarrow 8 \rightarrow 4 \rightarrow 2 \rightarrow 1$ (32x compression), while the horizontal axis is compressed by a factor of 4 ($T = W / 4$). For lines ($W=512$), this produces $T = 128$ sequential feature vectors of dimension 512.
3. **BiLSTM Sequence Modeling:** A 2-layer Bidirectional Long Short-Term Memory network ($hidden\_size=256$, $dropout=0.2$) processes the $T$ frames across both forward and backward temporal directions, outputting feature vectors of dimension $2 \times 256 = 512$.
4. **CTC Output Projection:** A linear projection maps $512 \rightarrow 87$ classes (86 vocabulary characters plus CTC blank at index 0). Log-softmax probabilities are decoded via Greedy Best-Path decoding:
   $$\hat{l} = \mathcal{B}\left(\arg\max_{\pi} P(\pi | \mathbf{x})\right)$$
   where $\mathcal{B}$ collapses consecutive identical indices and removes blank tokens.

```
[Figure 3: Training and Validation Curves for V1 and V2 Models]
```
![Figure 3](figures/f3_training_curves.png)  
*Figure 3: Training and validation loss trajectories and validation CER across epochs for Word V1 (17 epochs, best ep 12), Line V1 (15 epochs, best ep 12), Word V2 (18 epochs, best ep 13), and Line V2 (15 epochs, best ep 10). Stars indicate the checkpoints selected by early stopping based on lowest validation loss on the 8 seen fonts.*  
*Source: [runs/word/history.csv](file:///d:/laragon/www/az_ocr_bm606/runs/word/history.csv), [runs/line/history.csv](file:///d:/laragon/www/az_ocr_bm606/runs/line/history.csv), [runs/word_v2_20261027_180300/history.csv](file:///d:/laragon/www/az_ocr_bm606/runs/word_v2_20261027_180300/history.csv), [runs/line_v2/history.csv](file:///d:/laragon/www/az_ocr_bm606/runs/line_v2/history.csv)*

### 4.2 Training Dynamics and Checkpoint Provenance

All training was executed locally on Intel Core i7 CPU hardware under Windows using PyTorch 2.x and the Adam optimizer ($lr=1\times 10^{-3}$, $weight\_decay=1\times 10^{-5}$, batch size 64). Early stopping halted training after 5 consecutive epochs without validation loss improvement.

As documented in Table 3 and Figure 3:
- **Word V1:** 17 epochs run, best epoch: 12 (val loss: 0.0051, val CER: 0.14%). Log duration: 301.3 min (`runs/word/best.pt`).
- **Line V1:** 15 epochs run, best epoch: 12 (val loss: 0.0131, val CER: 0.31%). Log duration: 184.6 min (`runs/line/best.pt`).
- **Word V2:** 18 epochs run, best epoch: 13 (val loss: 0.0063, val CER: 0.24%). Log duration: 212.3 min (`runs/word_v2_20261027_180300/best.pt`). Checkpoint SHA256: `062bf63aeb8aad347e00df0173a36d844fc7c9725b2fa11941d9c9a7c05822cb`.
- **Line V2:** 15 epochs run, best epoch: 10 (val loss: 0.0125, val CER: 0.28%). Log duration: 280.4 min (`runs/line_v2/best.pt`). Checkpoint SHA256: `a0946ae009b09e898da5683e71d59ebcba030d451cfb43f41a48476c763c50a6`.

---

## 5. Evaluation Protocol and Baselines

### 5.1 Metrics Formulation

Model performance is evaluated across Character Error Rate (CER) and Word Error Rate (WER):
$$\text{CER} = \frac{\sum_{i=1}^N \text{Levenshtein}(r_i, h_i)}{\sum_{i=1}^N |r_i|}, \quad \text{WER} = \frac{\sum_{i=1}^N \text{Levenshtein}(r_{i,\text{words}}, h_{i,\text{words}})}{\sum_{i=1}^N |r_{i,\text{words}}|}$$
Character edit operations are decomposed into mutually exclusive categories: substitutions ($sub$), deletions ($del$), and insertions ($ins$).

### 5.2 Bootstrap Confidence Intervals

To determine whether performance differentials are statistically distinguishable from zero, we compute 95% bootstrap confidence intervals ($B=1,000$ iterations). On synthetic unseen fonts, we deploy both standard crop-level bootstrap and **cluster bootstrap** (resampling entire font clusters first, then lines) to prevent underestimation of between-font typographical variance [Baek et al., 2019]. On real documents, lines are resampled with replacement.

### 5.3 Baseline Engine Setup: Tesseract 5

We benchmark against Tesseract 5 (`v5.5.0.20241111`) utilizing official Azerbaijani neural network data (`aze`, `tessdata_best`). Tesseract is evaluated in two operational configurations:
- **Page Segmentation Mode 7 (`--psm 7`):** Treats image crops as single uniform text lines. This represents Tesseract's default standard for cropped line benchmarks.
- **Page Segmentation Mode 13 (`--psm 13`):** Bypasses Tesseract's internal page layout analysis and line finding, treating the input crop as a raw, single text line without baseline heuristics.

---

## 6. Experimental Results

```
[Figure 4: Per-Font Synthetic Performance: Tahoma vs. Consolas]
```
![Figure 4](figures/f4_per_font_synth.png)  
*Figure 4: Monofont recognition performance comparing Tahoma (genuine new family, held out from V1 and V2) vs. Consolas (family leaked in V2 via consolab.ttf). On Tahoma, V1, V2, and Tesseract exhibit statistically identical performance (~0.08–0.26% CER). On Consolas, V2 displays an artificial 14.6x error reduction due directly to training font family leakage.*  
*Source: [results/diagw_tahoma/metrics.json](file:///d:/laragon/www/az_ocr_bm606/results/diagw_tahoma/metrics.json), [results/diag_tahoma/metrics.json](file:///d:/laragon/www/az_ocr_bm606/results/diag_tahoma/metrics.json), [results/diagw_consola/metrics.json](file:///d:/laragon/www/az_ocr_bm606/results/diagw_consola/metrics.json), [results/diag_consola/metrics.json](file:///d:/laragon/www/az_ocr_bm606/results/diag_consola/metrics.json), [results/v2_evaluation_report.md](file:///d:/laragon/www/az_ocr_bm606/results/v2_evaluation_report.md)*

### 6.1 Standard Synthetic Benchmarks

On clean synthetic test sets, both CRNN iterations demonstrate sub-percent character error rates:
- **`word_synth` ($n=2,000$):** V1 achieves 0.50% CER (3.95% WER). V2 reduces CER to **0.11%** (0.90% WER). Tesseract 5 achieves 0.70% CER (4.90% WER).
- **`line_synth` ($n=1,000$):** V1 achieves 1.64% CER (11.64% WER). V2 reduces CER to **0.30%** (2.13% WER). Tesseract 5 achieves 0.44% CER (2.94% WER).

### 6.2 Font-Wise Analysis: Generalization vs. Family Leakage

To verify whether V2's synthetic gains reflected genuine typeface invariance, monofont tests were evaluated on the two V1 held-out fonts (`tahoma.ttf` and `consola.ttf`):
- **Tahoma (New Family):** Neither V1 nor V2 included Tahoma in training. At the word level ($n=1,000$), V1 achieved **0.08% CER**, V2 achieved **0.09% CER**, and Tesseract achieved **0.50% CER**. At the line level ($n=500$), V1 achieved **0.24% CER**, V2 achieved **0.26% CER**, and Tesseract achieved **0.22% CER**. V2 showed no statistical improvement over V1 (+0.01% and +0.02% variance).
- **Consolas (Family Leakage):** While `consola.ttf` was excluded, V2 training added `consolab.ttf` (Consolas Bold). Consequently, Consolas Word CER dropped dramatically from 1.02% to **0.07%** (a 14.6x reduction), and Line CER dropped from 3.48% to **0.27%** (a 12.9x reduction).
- **Scientific Audit Finding:** The apparent "breakthrough" on Consolas is attributable to **font family leakage**, not zero-shot typographic generalization.

```
[Figure 5: Simulation Ladder L0 through L4n]
```
![Figure 5](figures/f5_sim_ladder.png)  
*Figure 5: 7-stage Simulation Ladder evaluation ($n=346$ words, $n=92$ lines). Top: Visual progression of identical line crops from clean render (L0) through illumination gradients (L1), blur/noise (L2), downscaling/JPEG compression (L3), to full phone homography (L4). Bottom: Word and line CER trajectories demonstrating baseline V1 collapse at L4 (35.77% word / 12.96% line) vs. V2 robustness (1.35% word / 0.11% line).*  
*Source: [results/sim_ladder/summary.md](file:///d:/laragon/www/az_ocr_bm606/results/sim_ladder/summary.md), [results/v2_audit/summary.md](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit/summary.md)*

### 6.3 Photometric Simulation Ladder ($L_0$ to $L_4n$)

To isolate the impact of environmental camera artifacts, we constructed a 7-stage controlled distortion ladder across 4 fixed document pages ($n=346$ words, $n=92$ lines):
- **$L_0$ (Clean Render):** V1 Word CER: 7.71%, V2: **0.20%**, Tesseract: 0.41%.
- **$L_1$ (+ Illumination gradient & shadow):** V1 Word CER rises to 8.85%, V2: **0.20%**, Tesseract: 0.45%.
- **$L_2$ (+ Gaussian blur & sensor noise):** V1 Word CER collapses to **25.00%**, V2: **0.41%**, Tesseract: 0.41%.
- **$L_3$ (+ Downscale $0.6\times$ & JPEG $Q=50$):** V1 Word CER: 31.36%, V2: **0.53%**, Tesseract: 0.41%.
- **$L_4$ (Full Phone Simulation: Perspective tilt & rotation):** V1 Word CER reaches **35.77%** (Line CER 12.96%). In sharp contrast, V2 remains highly robust at **1.35%** Word CER and **0.11%** Line CER.
- **$L_4n$ (Post-Hoc Contrast Normalization):** Applying per-crop min-max contrast stretching to $L_4$ without retraining reduces V1 Word CER from 35.77% down to 24.43%, confirming that background brightness shift constitutes a major vulnerability for baseline neural activations.

```
[Figure 6: Forest Plot of Difference in Character Error Rate (V2 - Tesseract)]
```
![Figure 6](figures/f6_unseen_forest.png)  
*Figure 6: Forest plot of accuracy differentials ($\Delta\text{CER} = \text{CER}_{V2} - \text{CER}_{\text{Tess}}$) across synthetic unseen fonts, cluster bootstraps, and real document photos with 95% Confidence Intervals. Red dashed line indicates zero difference. In 9 of 11 benchmarks, the 95% CI spans across 0.0%, indicating that V2 and Tesseract are statistically tied on standard continuous text.*  
*Source: [results/v2_audit4/summary.md:118-132](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit4/summary.md#L118-L132), [results/real_fix9/metrics.json](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/metrics.json)*

### 6.4 Genuine Unseen Fonts Evaluation (Groups A, B, C)

In Audit 4 (`results/v2_audit4/summary.md`), 16 fonts were categorized into three distinct tiers:
- **Group A: SEEN (4 fonts):** `arial, calibri, segoeui, times`.
- **Group B: NEAR-RELATIVE SIBLINGS (5 fonts):** `calibrib, seguisb, verdanab, CascadiaMono, pala`.
- **Group C: TRULY UNSEEN NEW FAMILIES (7 fonts):** `cambria, micross, Moba_Fira, Moba_Font, Moba_Fanta, tahomabd, SansSerifCollection`.

As detailed in Table 5 and Figure 6:
- Under **Cluster Bootstrap** across Group C at $L_0$, Line CER difference is $\Delta = \mathbf{+0.00\%}$ (95% CI: **[-0.74%, +0.76%]**). Both V2 Line and Tesseract 5 yield an aggregate CER of **0.71%** ($n=158$ lines, 4,673 chars).
- At $L_4$ distortion, Line CER difference is $\Delta = \mathbf{-0.17\%}$ (95% CI: **[-2.60%, +1.53%]**), with V2 at 1.28% and Tesseract at 1.46%.
- **Conclusion:** On continuous lines of standard-print unseen fonts, V2 and Tesseract are **statistically indistinguishable**. On single cropped words at $L_4$ (Group C), Tesseract holds a statistically significant lead ($\Delta = +1.55\%$, 95% CI: [+0.59%, +2.94%]), driven by Tesseract's internal language dictionary.

```
[Figure 7: Azerbaijani Special Letter Error Distributions]
```
![Figure 7](figures/f7_special_letters.png)  
*Figure 7: Diacritic recognition error rates across clean unseen fonts for the 7 Azerbaijani special letter pairs (ə/e, ı/i, ö/o, ğ/g, ü/u, ş/s, ç/c) with exact sample sizes (n) annotated on each bar. V2 eliminates cedilla confusion on continuous lines, while dotless-i (ı/i) remains the predominant remaining failure mode.*  
*Source: [results/v2_audit3/summary.md:121-142](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit3/summary.md#L121-L142), [results/v2_audit4/summary.md](file:///d:/laragon/www/az_ocr_bm606/results/v2_audit4/summary.md)*

### 6.5 Real Smartphone Document Testing (Photos 1–5)

To evaluate real-world performance, five physical printed documents photographed with smartphones were tested under preregistered pipeline conditions:

```
[Figure 8: Real Document Layout Detection & Chunking Boundaries]
```
![Figure 8](figures/f8_real_photos_detection_v2.png)  
*Figure 8: Real document line detection and projection chunking. Panel A: Photo 1 showing only the evaluated region (16 lines). Panel B: Photo 5 (IMG_5130, 33 lines) full document layout with detected line boxes. Panel C: Projection-based split boundaries (dashed red lines) across 5 representative lines using smoothed column projection ($k=25$, 35th percentile ink threshold).*  
*Source: [data/real_docs/photo1_times.jpg](file:///d:/laragon/www/az_ocr_bm606/data/real_docs/photo1_times.jpg), [data/real_test5/photo5.jpg](file:///d:/laragon/www/az_ocr_bm606/data/real_test5/photo5.jpg), [results/real_fix8/crop_geometry.json](file:///d:/laragon/www/az_ocr_bm606/results/real_fix8/crop_geometry.json), [results/real_fix9/crop_geometry.json](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/crop_geometry.json)*

1. **Photo 1 (Times New Roman, $n=16$ lines, 1,442 chars):** On full unchunked lines, V2 suffers from horizontal squeeze (53.61% CER). When rule chunking is applied, V2 Line CER drops to **0.97%** (5.53% WER), on par with Tesseract's **1.32% CER** (5.07% WER). The difference $\Delta\text{CER} = -0.35\%$ (95% CI: [-1.52%, +0.50%]) spans zero (`results/real_fix8/final_tables.md:13`).
2. **Photo 2 (Calibri, $n=16$ lines, 1,463 chars):** On trimmed chunks, V2 achieves **0.55% CER** (3.79% WER, 10/16 exact lines), while Tesseract achieves **0.14% CER** (1.42% WER, 14/16 exact lines). Tesseract maintains a small but statistically significant lead ($\Delta = +0.41\%$, 95% CI: [+0.07%, +0.94%]).
3. **Photo 3 (Clean Times New Roman, $n=23$ lines, 1,006 chars):** V2 achieves **0.10% CER** (0.72% WER, 22/23 exact lines), with only 1 character deletion in the entire document. Tesseract achieves 0.00% CER (23/23 exact lines) (`results/photo3_test/report.md:7-15`).
4. **Photo 4 (Clean Arial, $n=23$ lines, 1,006 chars):** On rule chunks, V2 achieves **0.60% CER** (3.65% WER, 20/23 exact lines), matching Tesseract's **0.60% CER** (3.65% WER) exactly ($\Delta = 0.00\%$, 95% CI: [-0.29%, +0.29%]).
5. **Photo 5 (IMG_5130, 5 Paragraphs, $n=33$ lines, 2,328 chars):** On the full document under Condition A:
   - V2 Rule Chunk: **2.10% CER** (12.80% WER, 9/33 exact lines, 49 character errors).
   - V2 Full Line: **3.87% CER** (21.13% WER, 5/33 exact lines, 90 character errors).
   - Tesseract Full Line (PSM 7): **8.42% CER** (15.18% WER, 196 errors; includes 2 empty lines).
   - Tesseract Rule Chunk (PSM 7): **13.02% CER** (19.35% WER, 303 errors; includes 9 empty chunks).
   - Tesseract Full Line (PSM 13): **0.09% CER** (0.60% WER, 31/33 exact lines, **only 2 errors in 2,328 characters**).

```
[Figure 9: Annotated Line Recognition Examples from Photo 5]
```
![Figure 9](figures/f9_line_annotations_v2.png)  
*Figure 9: Representative line crops from Photo 5 (IMG_5130) displaying ground truth reference, V2 Line output, and Tesseract PSM 13 output with character-level difflib highlighting (mismatches highlighted in bold red with background markers, correct characters in dark green/black). Selection rule: hand-selected to represent 2 lines where V2 is 100% perfect (Lines 03, 12), the only 2 lines where Tesseract PSM 13 errs (Lines 21, 22), 1 line testing em dash ('—', U+2014) vs. hyphen ('-', U+002D) substitution (Line 19; scored as 1 error in raw Levenshtein), and 1 line with V2 vowel misrecognition (Line 24). Note: Image crops contain vertical fragments of neighbouring lines due to the ±4 px bounding box detection margin.*  
*Source: [results/real_fix9/crops/](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/crops/), [results/real_fix9/per_line_chunk_details.csv](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/per_line_chunk_details.csv)*

```
[Figure 10: Aspect Ratio Squeeze Analysis]
```
![Figure 10](figures/f10_squeeze_analysis.png)  
*Figure 10: Horizontal aspect ratio squeeze analysis. Left: Scatter of per-line CER vs squeeze ratio r = (W*32)/(h*512) for Full Line vs. Chunked Line, marking thresholds at 1.0 and 1.3. Right: Grouped bar chart demonstrating that lines exceeding r > 1.3 suffer severe full-line degradation (4.87% CER), which projection chunking reduces by 3.54 percentage points down to 1.33% CER.*  
*Source: [results/real_fix9/metrics.json:313-395](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/metrics.json#L313-L395)*

### 6.6 Chunking and Aspect Ratio Squeeze Analysis

The CRNN line input architecture forces any line to compress into a fixed width of $W=512$ at height $H=32$. When a text line has physical pixel dimensions $w \times h$, the natural scaled width is $w' = w \cdot (32 / h)$. The **squeeze ratio** is defined as:
$$r = \frac{w \cdot 32}{h \cdot 512}$$
When $r > 1.0$, the line cannot fit into 512 pixels without horizontal downscaling, causing glyph strokes to become unnaturally narrow.

As shown in Table 4 and Figure 10, lines in Photo 5 partitioned into 4 aspect ratio groups reveal a strong structural threshold:
- **Group $r < 1.0$ ($n=4$ lines):** Full Line CER is **0.54%**; Chunked CER is **1.61%** ($\Delta = -1.08\text{ pp}$). Short lines do not require chunking; chunking introduces artificial boundary cuts.
- **Group $1.0 \le r < 1.1$ ($n=10$ lines):** Full Line CER is **4.74%**; Chunked CER is **3.93%** ($\Delta = +0.81\text{ pp}$).
- **Group $1.1 \le r \le 1.3$ ($n=13$ lines):** Full Line CER is **3.36%**; Chunked CER is **1.16%** ($\Delta = \mathbf{+2.21\text{ pp}}$ gain).
- **Group $r > 1.3$ ($n=6$ lines):** Full Line CER is **4.87%**; Chunked CER is **1.33%** ($\Delta = \mathbf{+3.54\text{ pp}}$ gain).
- **Rule Formulation:** Projection-based chunking is beneficial **strictly when $r \ge 1.1$**, and becomes essential for lines exceeding $r > 1.3$.

```
[Figure 11: Border Padding Sensitivity]
```
![Figure 11](figures/f11_padding_sensitivity_v2.png)  
*Figure 11: Boundary padding sensitivity on Photo 5 comparing Condition A (frozen trim + 4 px margin) against Condition B (padded 20 px white border) across two padding values only (intermediate padding was not measured). Condition B is plotted in neutral slate blue. Explanatory callouts are positioned outside the plot area. Relative increases: V2 Chunk +1010% (x11.1), V2 Full +487% (x5.9), Tesseract Chunk PSM 7 +44.9% (x1.45), Tesseract Full PSM 7 +29.6% (x1.30), Tesseract Full PSM 13 +1900% (x20.0). Cause (hypothesis, not tested): CRNN receptive field drift on canvas margins.*  
*Source: [results/real_fix9/metrics.json:2-183](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/metrics.json#L2-L183), [results/real_fix9/report.md:33-48](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/report.md#L33-L48)*

### 6.7 Padding Sensitivity: A Severe Architectural Fragility

To evaluate model resilience to bounding box margin inaccuracies, we preregistered a sensitivity test on Photo 5 comparing **Condition A** (Trimmed ink + 4 px margin) with **Condition B** (identical crops padded with a 20 px white border on all four sides).

The results (Table 4 and Figure 11) reveal a critical architectural weakness:
- **V2 Rule Chunk:** CER collapses from **2.10%** in Condition A to **23.32%** in Condition B (+21.22 percentage point degradation; 543 errors vs. 49 errors).
- **V2 Full Line:** CER collapses from **3.87%** to **22.72%** (+18.85 pp degradation).
- **Tesseract PSM 7:** CER shifts moderately from 13.02% to 18.86% on chunks, and 8.42% to 10.91% on full lines.
- **Tesseract PSM 13:** CER shifts from 0.09% to 1.80%.
- **Root Cause Analysis:** The CRNN receptive field was trained on tightly cropped synthetic lines where glyph ink consistently spanned from top to bottom. When a 20 px white border is introduced, the CNN convolutional features trigger repeated blank activations or spurious boundary insertions. This highlights that **V2 is extremely fragile to margin padding variations**.

```
[Figure 12: Mutually Exclusive Error Taxonomy Stacked Bars]
```
![Figure 12](figures/f12_error_taxonomy.png)  
*Figure 12: Mutually exclusive character error taxonomy for Condition A (all 33 lines) and Clean Subset (23 lines). The sum of all six categories matches the true Levenshtein distance exactly 100%. V2 errors are dominated by internal letters and punctuation, whereas Tesseract PSM 7 is distorted by massive letter deletions resulting from empty chunk outputs.*  
*Source: [results/real_fix9/error_taxonomy.json](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/error_taxonomy.json), [results/real_fix9/report.md:69-82](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/report.md#L69-L82)*

```
[Figure 13: Tesseract PSM 7 vs. PSM 13 Integration Failure]
```
![Figure 13](figures/f13_tess_psm_comparison_v2.png)  
*Figure 13: Tesseract sensitivity to page segmentation mode (PSM 7 vs. PSM 13) on Photo 5 (PSM 13 was chosen post-hoc after PSM 7 baseline detector failed). Left: Empty output count broken down by chunks (9 in Cond A, 11 in Cond B) vs. full lines (2 in Cond A, 2 in Cond B); PSM 13 on all rule chunks was not measured. Right: Character error rates comparing like with like (same evaluation unit). On the clean 23-line subset, Tesseract Full Line achieves 1.07% (PSM 7) vs. 0.00% (PSM 13), while Rule Chunk achieves 1.26% (PSM 7).*  
*Source: [results/real_fix9/metrics.json:25-46](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/metrics.json#L25-L46), [results/real_fix9/empty_outputs_log.json](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/empty_outputs_log.json)*

### 6.8 Tesseract PSM Study and the 23-Line Clean Subset

On Photo 5, the raw benchmark initially suggested that V2 (2.10% CER) dramatically outperformed Tesseract PSM 7 (13.02% CER). However, audit of `results/real_fix9/empty_outputs_log.json` revealed:
- Under `--psm 7`, Tesseract returned an **empty string (`""`)** on 9 second-half chunks (Lines 18, 19, 21, 22, 23, 24, 26, 28, 32) and 2 full lines (Lines 24, 29). This occurred because Tesseract's single-line layout analyzer requires surrounding paragraph context to estimate the text baseline; when presented with narrow, cropped strips, the baseline detector fails completely.
- When these exact 9 failed chunks were evaluated with `--psm 13` (raw line), Tesseract transcribed them with near-zero error (e.g., Line 18: `bu təkərin arxasında oturmusan. Özünü`).
- On the **Clean Subset of 23 lines** (where neither model produced empty outputs):
  - Tesseract 5 Rule Chunk: **1.26% CER** (7.05% WER, 11/23 exact lines).
  - V2 Line Rule Chunk: **1.32% CER** (9.25% WER, 9/23 exact lines).
  - The bootstrap difference is $\Delta\text{CER} = \mathbf{+0.06\%}$ (95% CI: **[-0.77%, +0.80%]**), which spans zero.
  - **Verdict:** The apparent superiority of V2 over Tesseract on Photo 5 is an artifact of Tesseract's layout segmentation failure. On pure optical character recognition, V2 and Tesseract are **statistically tied**.

```
[Figure 14: Lexicon Post-Corrector Pilot and Data Snooping Risk]
```
![Figure 14](figures/f14_lexicon_pilot.png)  
*Figure 14: Heuristic lexicon post-correction pilot on Photo 5. Left: Modification breakdown showing 8 true positive word corrections and 1 false positive regression. Right: CER reduction from 2.10% to 1.80%. Red warning banner explicitly discloses test-snooping risk; the primary scientific result remains Raw V2 (2.10%).*  
*Source: [scratch/test_safe_corrector.py](file:///d:/laragon/www/az_ocr_bm606/scratch/test_safe_corrector.py), [results/real_fix9/metrics.json:58-68](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/metrics.json#L58-L68)*

### 6.9 Lexicon Corrector Pilot and Test-Snooping Disclosure

In `scratch/test_safe_corrector.py`, a pilot SymSpell post-processor utilizing a 121,377-word stem lexicon and an Azerbaijani suffix stripper was tested on Photo 5:
- The corrector altered 9 words across the 33 lines:
  - 8 words were correctly fixed (+8 chars: `sadəcə`, `təhsil`, `gənc`, `görüb`, `başını`, `vaxt`, `babasının`, `inkişafına`).
  - 1 word was erroneously corrupted (-1 char: on Line 24, model hypothesis `kiplər` was altered to `kimlər`, whereas the reference was `küplər`).
  - Net character gain was $+7$ characters (reducing errors from 49 to 42, CER 2.10% $\rightarrow$ **1.80%**, exact lines 9 $\rightarrow$ 15).
- **Scientific Disclosure:** As acknowledged in `paper/DISCLOSURES.md`, the morphological rules and exceptions were tuned post-hoc while inspecting Photo 5 errors. Furthermore, the lexicon was compiled partly from the Wikipedia corpus used to train V2. This pilot is reported solely as a feasibility demonstration; the main scientific benchmark remains unassisted **Raw V2 (2.10% CER)**.

---

## 7. Error Analysis

### 7.1 Mutually Exclusive Error Taxonomy

To ensure mathematical precision, character errors on Photo 5 were mapped to six mutually exclusive categories such that their sum equals the global Levenshtein distance exactly 100% (Table 7 and Figure 12):
1. **Leading Edge Insertions:** Spurious characters before the first aligned reference character (V2: **0**, Tess: **3**).
2. **Trailing Edge Insertions:** Spurious characters after the last aligned reference character (V2: **0**, Tess: **0**).
3. **Internal Spacing Errors:** Space insertions or deletions inside the body (V2: **3**, Tess: **44**).
4. **Internal 0 / O Confusion:** Confusions between digit `0` and letter `O/o` (V2: **0**, Tess: **0**).
5. **Internal Punctuation Errors:** Ambiguities between periods, colons, and commas (V2: **12**, Tess: **20**).
6. **Internal Letter Errors:** True character glyph misrecognitions (V2: **34**, Tess: **236**).

On the Clean Subset (23 lines), V2 has 2 space errors, 6 punctuation errors, and 13 letter errors (total 21), compared to Tesseract's 1 space error, 7 punctuation errors, and 9 letter errors (total 20). V2's primary remaining weaknesses on real paper are **punctuation ambiguity** (periods recognized as colons or dropped) and **rare glyph substitutions**.

### 7.2 Word Error Rate (WER) Decomposition: The Space Insertion Defect

Across synthetic benchmarks, V2 exhibits an apparent paradox: its CER is equal to or better than Tesseract, yet its Line WER is substantially worse (e.g., Line $L_4$ WER of 13.25% vs. Tesseract's 3.49%).

In Audit 4 Task 5 (`results/v2_audit4/summary.md:200-260`), a word-level error decomposition was executed across 1,076 words:
- At $L_0$, **79.3%** of V2's word errors (46 out of 58 failed words) were **pure space insertion errors** (e.g., `ticarətinə` $\rightarrow$ `t icarət inə`, `virus` $\rightarrow$ `v irus`, `din` $\rightarrow$ `d in`).
- At $L_4$, space insertion errors accounted for **65.3%** of all V2 word errors (64 out of 98 failed words).
- In contrast, Tesseract produced almost zero space errors (0.0% at $L_0$, 5.9% at $L_4$).
- When evaluated under a retokenized metric where intra-word spaces are collapsed, V2's Line WER dropped from **12.87%** to **4.55%**, closing over 70% of the gap to Tesseract. This demonstrates that V2's high WER is driven by CTC acoustic transition uncertainty at character kerning boundaries rather than recognition collapse.

---

## 8. Pitfalls and Lessons Learned

Throughout the lifecycle of Project BM 606, five major engineering and methodological pitfalls were encountered and resolved (summarized in Table 9):

```
Table 9: Chronological Pitfalls, Discovery Mechanisms, Quantitative Impact, and Resolutions
(Full Markdown table embedded from paper/tables/table9_bugs_and_lessons.md)
```

1. **Clipped Glyph Bottoms in Synthetic Data:** Early in data generation, TRDG rendered text boxes that clipped the bottom cedillas of `ş` and `ç` and descenders of `q, y, p`. Discovered via visual contact sheets (`verify_quality.py`), the defect was resolved by implementing a $+4\text{ px}$ canvas margin and anchoring font metrics, verified by a 0.0% border-ink touch rate across 3,000 checked images (`results/data_check.png`).
2. **Missing Space Character in Vocabulary:** During initial line model training (`logs/line_train_nospace.log`), the whitespace character `' '` was inadvertently omitted from `common.py:CHARS`. Because the CTC blank token (index 0) cannot represent text spaces, the model concatenated all words, yielding a catastrophic Line WER of **99.75%** (Line CER 11.19%). Restoring the space character dropped CER to **1.64%** and WER to **11.64%** (`logs/line_train.log`).
3. **SansSerifCollection Bounding Box Metric Inflation:** On the synthetic ladder, `SansSerifCollection.ttf` initially yielded an 84.32% CER. Audit revealed that `make_sim_ladder.py` relied on theoretical font metrics ($ascent + descent = 80\text{ px}$), creating vertical boxes that overlapped the next text line. Switching to tight ink bounding boxes (`w_bbox`) dropped L0 CER from **84.32% down to 0.76%** (`results/v2_audit3/summary.md:31`).
4. **Tesseract Integration Failure with `--psm 7`:** On trimmed image chunks of Photo 5, Tesseract returned empty strings on 9 lines due to baseline loss, falsely inflating CER to 13.02%. Documenting this integration breakdown and comparing against `--psm 13` (0.09% CER) and a 23-line clean subset (1.26% vs 1.32%) ensured scientific fairness.
5. **Heuristic Lexicon Over-Fitting:** Refining spelling dictionary rules while inspecting Photo 5 recognition outputs created an immediate risk of data snooping. Disclosing this limitation and preserving the unassisted Raw V2 model as the primary paper result maintains scientific credibility.

---

## 9. Limitations

We explicitly delineate the operational boundaries and limitations of this study:
1. **Aspect Ratio Squeeze Sensitivity:** Input lines must be scaled to height 32 and width 512. Lines with squeeze ratio $r > 1.3$ experience severe character stroke thinning and recognition failure unless pre-chunked.
2. **Border Padding Fragility:** V2 cannot tolerate arbitrary whitespace canvas padding (20 px white padding degrades CER from 2.10% to 23.32%). Upstream layout segmentators must provide tight bounding boxes.
3. **CPU-Only Training Regime:** Models were trained strictly on CPU hardware over 3–5 hours per model. Larger Vision Transformer (TrOCR) architectures could not be trained within these compute bounds.
4. **Lack of an End-to-End Language Model:** V2 utilizes greedy CTC best-path decoding. Without a coupled beam-search n-gram or neural language model, intra-word space insertions remain uncorrected.
5. **Sample Size Constraints on Real Documents:** Real photo tests span five documents totaling 111 lines. While bootstrap confidence intervals establish parity on clean subsets, definitive real-world claims require multi-document benchmarks across 20–30 diverse texts.

---

## 10. Conclusion and Future Work

In this research, we developed and audited a deep learning optical character recognition framework tailored for the Azerbaijani Latin script. We demonstrated that photometrically augmented CRNN training confers resilience against mobile smartphone imaging artifacts (maintaining 1.35% Word CER at $L_4$ where baseline V1 collapses to 35.77%). 

Crucially, rigorous auditing dismantled two common scientific illusions:
- The apparent dramatic gain on Consolas was traced to training font family leakage.
- The apparent 10 percentage point victory over Tesseract on Photo 5 was traced to Tesseract's PSM 7 baseline detector failure. On clean real lines, V2 (1.32% CER) and Tesseract (1.26% CER) are **statistically tied**.

### Future Research (V3 Architecture):
1. **Dynamic Width / Patch Inputs:** Replacing fixed $32\times 512$ resizing with dynamic width patch encoders or ViT architectures to eliminate aspect ratio squeeze.
2. **Border Padding Data Augmentation:** Incorporating random white and textured canvas padding during training to render the CNN backbone immune to margin variations.
3. **CTC Beam Search with Language Prior:** Coupling decoding with a 5-gram language model to eliminate the 65% space-insertion WER defect.
4. **Large-Scale Document Benchmark:** Expanding empirical testing to an open benchmark of 20–30 multi-page historical and administrative documents evaluated via cluster bootstrap.

---

## Appendix: Reproducibility and Checkpoints

All experimental parameters, random seeds (`seed=42`), and checkpoints are frozen on disk:

| Checkpoint Path | Architecture | Target Modality | Epochs Run (Best) | Val Loss (Best) | Val CER (%) | File Size | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `runs/word/best.pt` | CRNN | Word ($32\times 128$) | 17 (12) | 0.0051 | 0.14% | 32.9 MB | `9ee974d0f032136669b8e04dd1a91896c4363f6aa1e3d4f7efdbfe566b5c4f25` |
| `runs/line/best.pt` | CRNN | Line ($32\times 512$) | 15 (12) | 0.0131 | 0.31% | 32.9 MB | `f6d981d97e0665e080b480019370570892e9252f8d9f78aabb83c5ecbc804425` |
| `runs/word_v2_20261027_180300/best.pt`| CRNN | Word ($32\times 128$) | 18 (13) | 0.0063 | 0.24% | 98.7 MB | `062bf63aeb8aad347e00df0173a36d844fc7c9725b2fa11941d9c9a7c05822cb` |
| `runs/line_v2/best.pt` | CRNN | Line ($32\times 512$) | 15 (10) | 0.0125 | 0.28% | 98.7 MB | `a0946ae009b09e898da5683e71d59ebcba030d451cfb43f41a48476c763c50a6` |

*Note: V2 checkpoint files are 98.7 MB on disk as they serialize full Adam optimizer states (`opt`), model state dictionaries, character mapping vectors, and configuration dictionaries; standalone model inference weights require 32.9 MB.*

---

## References

1. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. *Proceedings of the 23rd International Conference on Machine Learning (ICML '06)*, 369–376.
2. Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 39(11), 2298–2304.
3. Smith, R. (2007). An overview of the Tesseract OCR engine. *Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)*, 2, 629–633.
4. Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., Oh, S. J., & Lee, H. (2019). What is wrong with scene text recognition model comparisons? dataset and baseline. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV '19)*, 4715–4723.
5. Memmedov, S., et al. (2026). *Azerbaijani Document Digitization Benchmark* [Unverified citation; marked and excluded in accordance with project audit BM606-DOCX-AUDIT-FIXES-V3].
