# Azerbaijani OCR Project: Independent Audit Report

**Date of Audit:** October 23, 2026  
**Auditor:** Automated Codebase Verification Agent  
**Repository:** `d:\laragon\www\az_ocr_bm606`  
**Scope:** Strict read-only audit of 10 verification claims, codebase integrity, experimental methodology, and gap analysis against the project proposal.

---

## 1. Master Claims Verification Table

| Claim # | Claim Description | Status | Real Value (Measured / Verified) | Source File & Location |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Data sizes:**<br>• synth_word: train/val/test 20k / 2k / 2k<br>• synth_line: 10k / 1k / 1k<br>• synth_word_v2: train 35k, val/test 2k / 2k<br>• synth_line_v2: train 16k, val/test 1k / 1k | **CONFIRMED** | • `synth_word`: train 20,000 img / 20,000 tsv; val 2,000 img / 2,000 tsv; test 2,000 img / 2,000 tsv<br>• `synth_line`: train 10,000 img / 10,000 tsv; val 1,000 img / 1,000 tsv; test 1,000 img / 1,000 tsv<br>• `synth_word_v2`: train 35,000 img / 35,000 tsv; val 2,000 img / 2,000 tsv; test 2,000 img / 2,000 tsv<br>• `synth_line_v2`: train 16,000 img / 16,000 tsv; val 1,000 img / 1,000 tsv; test 1,000 img / 1,000 tsv | `data/synth_word/`<br>`data/synth_line/`<br>`data/synth_word_v2/`<br>`data/synth_line_v2/`<br>(`images/*.png`, `labels.tsv`) |
| **2** | **v2 val/test identity & leakage:**<br>• v2 val/test identical to v1 val/test<br>• Word-set leakage count (val/test in train) for v1 and v2 | **CONFIRMED** | • `labels.tsv` SHA256 hashes match identically between v1 and v2 for all val and test splits (`match=True`). Image hashes match identically.<br>• Word-set leakage: **0** val/test labels in train for v1 (0.00%); **0** val/test labels in train for v2 (0.00%).<br>• Line-set leakage: **0** overlapping lines for v1 and v2. | `data/synth_word/` & `data/synth_word_v2/`<br>`data/synth_line/` & `data/synth_line_v2/`<br>(`labels.tsv` & `images/`) |
| **3** | **Fonts:**<br>• 10 fonts in `fonts/`<br>• Train vs test split (`collect_fonts('fonts', 42)`)<br>• `fonts_new/` contents | **CONFIRMED** | • Exactly 10 fonts in `fonts/`: `arial.ttf`, `calibri.ttf`, `candara.ttf`, `consola.ttf`, `corbel.ttf`, `cour.ttf`, `segoeui.ttf`, `tahoma.ttf`, `times.ttf`, `verdana.ttf`<br>• `collect_fonts('fonts', 42)` returns:<br>  - Train (8): `candara.ttf`, `times.ttf`, `cour.ttf`, `segoeui.ttf`, `verdana.ttf`, `corbel.ttf`, `arial.ttf`, `calibri.ttf`<br>  - Test (2): `tahoma.ttf`, `consola.ttf`<br>• `fonts_new/` contains 6 fonts: `bahnschrift.ttf`, `CascadiaCode.ttf`, `consolab.ttf`, `ebrima.ttf`, `l_10646.ttf`, `palab.ttf` | `fonts/`<br>`fonts_new/`<br>[generate_data.py:28-36](file:///d:/laragon/www/az_ocr_bm606/generate_data.py#L28-L36) |
| **4** | **Special letters in line-corpus:**<br>• Percentage containing (ə ı ö ğ ü ş ç) across train / val / test splits | **CONFIRMED** | • `corpus_line/train.txt`: **89.79%** (9,338 / 10,400 lines)<br>• `corpus_line/val.txt`: **96.24%** (92,059 / 95,659 lines)<br>• `corpus_line/test.txt`: **96.04%** (92,104 / 95,906 lines)<br>*(Note: `corpus_word` has 49.55% train, 48.36% val, 50.44% test)* | `data/corpus_line/train.txt`<br>`data/corpus_line/val.txt`<br>`data/corpus_line/test.txt` |
| **5** | **Model architecture & training logs:**<br>• Param count: 8.23M<br>• Input sizes: word 32x128, line 32x512<br>• Optimizer & learning rate<br>• Epochs run & best epoch for `runs/word` and `runs/line` | **CONFIRMED** | • CRNN parameters: exactly **8,225,815** (8.23M)<br>• Input sizes: `word` (32, 128), `line` (32, 512)<br>• Optimizer: `Adam(lr=1e-3, weight_decay=1e-5)` with gradient norm clip 5.0<br>• `runs/word`: 17 epochs run (stopped by early stopping, patience 5); Best epoch: **12** (`val_loss = 0.0051`, `val_cer = 0.14%`, train time 301.3 min)<br>• `runs/line`: 15 epochs run (`--epochs 15`); Best epoch: **12** (`val_loss = 0.0131`, `val_cer = 0.31%`, train time 184.6 min) | [common.py:18](file:///d:/laragon/www/az_ocr_bm606/common.py#L18)<br>[model.py:10-52](file:///d:/laragon/www/az_ocr_bm606/model.py#L10-L52)<br>[train.py:99-147](file:///d:/laragon/www/az_ocr_bm606/train.py#L99-L147)<br>[logs/word_train_rerun.log:1-22](file:///d:/laragon/www/az_ocr_bm606/logs/word_train_rerun.log#L1-L22)<br>[logs/line_train.log:1-17](file:///d:/laragon/www/az_ocr_bm606/logs/line_train.log#L1-L17) |
| **6** | **Results in `results/*/metrics.json`:**<br>• word_synth (0.50/3.95)<br>• line_synth (1.64/11.64)<br>• word Tahoma (0.08/0.70)<br>• word Consolas (1.02/8.10)<br>• line Tahoma (0.24/1.74)<br>• line Consolas (3.48/23.76)<br>• Tess word (0.70/4.90), line (0.44/2.94)<br>• Ladder word CER L0..L4n (7.71, 8.85, 25.00, 31.36, 35.77, 24.43)<br>• Ladder line CER L0..L4n (1.87, 4.10, 13.84, 19.14, 12.96, 7.43) | **CONFIRMED** | All claimed values match exact recorded metrics:<br>• `word_synth`: CER = 0.50%, WER = 3.95%, n = 2,000<br>• `line_synth`: CER = 1.64%, WER = 11.64%, n = 1,000<br>• `diagw_tahoma`: CER = 0.08%, WER = 0.70%, n = 1,000<br>• `diagw_consola`: CER = 1.02%, WER = 8.10%, n = 1,000<br>• `diag_tahoma`: CER = 0.24%, WER = 1.74%, n = 500<br>• `diag_consola`: CER = 3.48%, WER = 23.76%, n = 500<br>• `tess_word_synth`: CER = 0.70%, WER = 4.90%, n = 2,000<br>• `tess_line_synth`: CER = 0.44%, WER = 2.94%, n = 1,000<br>• Ladder Word CER: L0=7.71%, L1=8.85%, L2=25.00%, L3=31.36%, L4=35.77%, L4n=24.43% (n=346)<br>• Ladder Line CER: L0=1.87%, L1=4.10%, L2=13.84%, L3=19.14%, L4=12.96%, L4n=7.43% (n=92)<br>*(Additional: `line_synth_nospace` was CER 11.19%, WER 99.75%)* | `results/word_synth/metrics.json`<br>`results/line_synth/metrics.json`<br>`results/diagw_tahoma/metrics.json`<br>`results/diagw_consola/metrics.json`<br>`results/diag_tahoma/metrics.json`<br>`results/diag_consola/metrics.json`<br>`results/tess_word_synth/metrics.json`<br>`results/tess_line_synth/metrics.json`<br>`results/sim_ladder/summary.md` |
| **7** | **Data augmentation:**<br>• List transforms and probabilities in `train.py`<br>• Verify if `train_v2.py` adds blur, Gaussian noise, JPEG, downscale, brightness/contrast, gradient, contrast stretch | **CONFIRMED** | • `train.py` augmentations:<br>  1. Rotation $\pm 5^\circ$ ($p = 0.5$)<br>  2. Gaussian blur $k \in \{3, 5\}$ ($p = 0.3$)<br>  3. Salt & Pepper noise 0.5% ($p = 0.3$)<br>  4. Perspective warp ($p = 0.3$)<br>  *(Gaussian sensor noise, JPEG, downscale, brightness/contrast, gradient, and contrast stretch are all absent: $p = 0$)*<br>• `train_v2.py`: Differs exactly as claimed, implementing all 7 new transformations with specified probabilities. | [train.py:21-40](file:///d:/laragon/www/az_ocr_bm606/train.py#L21-L40)<br>[train_v2.py:26-148](file:///d:/laragon/www/az_ocr_bm606/train_v2.py#L26-L148) |
| **8** | **Code review:**<br>• Identify bugs and methodological issues (leakage, val model selection vs test reporting, vocabulary mismatch, preprocessing mismatch, bias). | **CONFIRMED / DOCUMENTED** | 8 specific bugs and methodological weaknesses identified (detailed in Section 2). Highlights:<br>• Word corpus stripped all digits and punctuation (`prepare_corpus.py:14`).<br>• Preprocessing pads with fixed 255 canvas without background normalization (`common.py:84-86`).<br>• Val split shares fonts with train; test uses unseen fonts (`generate_data.py:88`). | [prepare_corpus.py:14](file:///d:/laragon/www/az_ocr_bm606/prepare_corpus.py#L14)<br>[common.py:41](file:///d:/laragon/www/az_ocr_bm606/common.py#L41)<br>[common.py:84-86](file:///d:/laragon/www/az_ocr_bm606/common.py#L84-L86)<br>[generate_data.py:88](file:///d:/laragon/www/az_ocr_bm606/generate_data.py#L88) |
| **9** | **Timing comparability (`ms_per_image`):**<br>• Is `ms_per_image` comparable between CRNN and Tesseract? Describe what each includes. | **CONFIRMED (Incomparable)** | **NOT comparable.**<br>• CRNN: Measures only the GPU/CPU forward pass (`model(x)`) in batches of 128. Excludes image disk read (`cv2.imread`), preprocessing, tensor transformation, DataLoader collation, and CTC decoding.<br>• Tesseract: Measures end-to-end wall-clock time sequentially per sample, including disk read, RGB conversion, external OS process spawning (`tesseract.exe`), OCR engine init, and text extraction. | [evaluate.py:32-52](file:///d:/laragon/www/az_ocr_bm606/evaluate.py#L32-L52)<br>[evaluate.py:61-66](file:///d:/laragon/www/az_ocr_bm606/evaluate.py#L61-L66) |
| **10** | **Course proposal gap analysis:**<br>• Status of TrOCR, beam search, word-segmentation error, real-photo test, preprocessing ablation, cost metric. | **CONFIRMED** | None of the 6 items are implemented or fully completed:<br>• TrOCR: Not implemented (only CRNN).<br>• Beam search: Not implemented (only greedy CTC).<br>• Word-segmentation error: Not implemented.<br>• Real-photo test: Not implemented (only synthetic / simulated).<br>• Preprocessing ablation: Incomplete (only 50-sample smoke test).<br>• Cost metric: Incomplete (no cloud/dollar cost or FLOPs). | [model.py](file:///d:/laragon/www/az_ocr_bm606/model.py)<br>[common.py:43-49](file:///d:/laragon/www/az_ocr_bm606/common.py#L43-L49)<br>[run_all.ps1:48,67](file:///d:/laragon/www/az_ocr_bm606/run_all.ps1#L48)<br>[run_ablation.py](file:///d:/laragon/www/az_ocr_bm606/run_ablation.py) |

---

## 2. Problems Found (Ordered by Severity)

### High Severity

#### 1. Complete Absence of Digits and Punctuation in Word Training Corpus
* **File & Line:** [prepare_corpus.py:14](file:///d:/laragon/www/az_ocr_bm606/prepare_corpus.py#L14)
* **Code:** `WORD_RE = re.compile("[" + AZ_LOWER + AZ_UPPER + "]+")`
* **Fact:** The regex filter strictly retained only Azerbaijani alphabet letters, dropping all numbers (`0-9`) and punctuation (`.,:;!?()"'`). Consequently, `data/synth_word/train/labels.tsv` contains **0.00%** digits and **0.00%** punctuation. While `common.CHARS` and the CRNN linear projection layer have output slots for digits and punctuation, the word model was never trained to recognize them.
* **Impact:** When evaluated on natural or simulated text containing punctuation or numbers (such as `data/sim_preview` or `sim_ladder`), the word model fails or outputs spurious letters (e.g., `(BMTQAK` -> `ÇBMTQAK`, `1927-ci` -> `Izzzci`, `tikmək,` -> `tikmək`). This is a primary driver of the word model's high CER (7.71% at L0, rising to 25.00% at L2).

#### 2. Naive White Canvas Padding Without Illumination Normalization
* **File & Line:** [common.py:84-86](file:///d:/laragon/www/az_ocr_bm606/common.py#L84-L86)
* **Code:**
  ```python
  canvas = np.full((H, W) + img.shape[2:], 255, np.uint8)
  canvas[:, :new_w] = img
  x = canvas.astype(np.float32) / 255.0
  ```
* **Fact:** The image is padded on the right with fixed 255 (pure white) and scaled simply by dividing by 255.0. No min-max contrast stretching, standardization (mean/std subtraction), or adaptive thresholding is applied.
* **Impact:** When encountering photos with off-white, yellowed, or shadowed paper (background ~180-210), this produces an artificial high-contrast vertical edge where image meets padding. Furthermore, the learned CNN feature activations expect background activations near 1.0; a darkened background causes total feature collapse. This is proven by the fact that simply applying per-crop contrast normalization (L4n) reduces word CER from 35.77% down to 24.43% and line CER from 12.96% down to 7.43% without any model retraining.

#### 3. Font Split Asymmetry: Validation Set Does Not Measure Font Generalization
* **File & Line:** [generate_data.py:85-88](file:///d:/laragon/www/az_ocr_bm606/generate_data.py#L85-L88)
* **Code:**
  ```python
  tr_fonts, te_fonts = collect_fonts(args.fonts, args.seed)
  ...
  for split, n, fonts in [("train", args.n_train, tr_fonts), ("val", args.n_val, tr_fonts), ("test", args.n_test, te_fonts)]:
  ```
* **Fact:** Both `train` and `val` splits were generated using the 8 training fonts (`candara`, `times`, `cour`, `segoeui`, `verdana`, `corbel`, `arial`, `calibri`), whereas `test` exclusively used the 2 unseen test fonts (`tahoma`, `consola`).
* **Impact:** In `train.py:139-142`, early stopping and checkpoint saving (`best.pt`) are driven exclusively by `val_loss`. Because `val` contains the same fonts as `train`, `val_loss` and `val_cer` (which reached 0.14%) only reflect generalization to unseen word combinations in *seen* typography, not generalization to unseen fonts. Overfitting to font geometry cannot be detected by early stopping.

---

### Medium Severity

#### 4. Silent Character Dropping in Vocabulary Encoding
* **File & Line:** [common.py:40-41](file:///d:/laragon/www/az_ocr_bm606/common.py#L40-L41)
* **Code:**
  ```python
  def encode(self, text: str):
      return [self.c2i[c] for c in text if c in self.c2i]
  ```
* **Fact:** `Vocab.encode` silently drops any character not present in `self.c2i` without raising an error or logging a warning.
* **Impact:** This silent failure mechanism was the exact root cause of the earlier line-model bug: when `common.CHARS` lacked the space character, all spaces were silently deleted from the training labels, resulting in a line model with 99.75% WER (`results/line_synth_nospace`). Any unicode anomaly or out-of-alphabet character in future datasets will be silently discarded rather than surfaced.

#### 5. Lack of Unicode Normalization in Dataset Loading
* **File & Line:** [common.py:130-141](file:///d:/laragon/www/az_ocr_bm606/common.py#L130-L141)
* **Code:** `load_labels(data_dir)` reads lines and splits by `\t` without calling `unicodedata.normalize("NFC", ...)`.
* **Fact:** While `evaluate.py:64` normalizes Tesseract output via `normalize()`, the ground truth labels loaded via `load_labels` and the CRNN hypotheses generated via `Vocab.decode_greedy` do not undergo NFC normalization during evaluation.
* **Impact:** If text in external or future test sets uses decomposed characters (NFD), Levenshtein alignment in `align()` will penalize single visual glyphs as multiple edits, distorting CER/WER.

#### 6. CTC Sequence Length Fixed to Full Feature Map Dimension
* **File & Line:** [train.py:76-77](file:///d:/laragon/www/az_ocr_bm606/train.py#L76-L77)
* **Code:** `loss = crit(out.float(), y, torch.full((B,), T, dtype=torch.long), yl)`
* **Fact:** `train.py` supplies `torch.full((B,), T, ...)` as the `input_lengths` parameter to `torch.nn.CTCLoss`, where $T = W / 4$ (32 for word, 128 for line).
* **Impact:** Shorter words that occupy only a small fraction of the image canvas are treated as having length $T$. The model is forced to dedicate sequence time-steps to emitting CTC blanks over the padded right canvas. While the model learns to do this, standard practice calculates true unpadded feature widths $\lceil \text{new\_w} / 4 \rceil$ to avoid wasting capacity and gradient propagation on blank canvas padding.

---

### Low Severity

#### 7. Non-Standard WER Behavior on Over-Segmented Hypotheses
* **File & Line:** [common.py:124-127](file:///d:/laragon/www/az_ocr_bm606/common.py#L124-L127)
* **Code:**
  ```python
  e2, _ = align(r.split(), h.split())
  we += e2
  wn += len(r.split())
  ```
* **Fact:** When evaluating single words (`word_synth`), `len(r.split()) == 1`. If the OCR model predicts an extraneous space inside a word (e.g., hypothesis has 2 words for 1 reference word), the edit distance is 2 (1 substitution + 1 insertion), while reference word count is 1.
* **Impact:** In edge cases with frequent space insertions, WER mathematically exceeds 100%.

#### 8. Asymmetric Sample Counts in Line Corpus Generation
* **File & Line:** [prepare_corpus.py:122-124](file:///d:/laragon/www/az_ocr_bm606/prepare_corpus.py#L122-L124)
* **Code:** `if k == "train" and args.max_items: lines = balance_special(lines, int(args.max_items * 0.8), ...)`
* **Fact:** The `--max-items` argument was only applied to cap `corpus_line/train.txt` (yielding 10,400 lines), while `val.txt` (95,659 lines) and `test.txt` (95,906 lines) were left uncapped.
* **Impact:** Does not cause model degradation because `generate_data.py` explicitly samples fixed sample sizes (1,000 for val/test), but the raw corpus text files exhibit a heavily distorted 10k / 95k / 95k split rather than 80% / 10% / 10%.

---

## 3. Detailed Analysis of Specific Claims

### Claim 8: Factors Influencing Reported Numbers (Optimism vs. Pessimism)

| Factor | Direction of Bias | Mechanism & Evidence |
| :--- | :---: | :--- |
| **Synthetic TRDG Test Set (`word_synth`, `line_synth`)** | **Highly Optimistic** | Images have clean, uniform, pure-white backgrounds with crisp synthetic typography. Reported test CER (0.50% word, 1.64% line) reflects near-ideal scanning conditions and does not generalize to real mobile phone photos. |
| **Evaluation on Unseen Fonts (`diagw_consola`, `diag_consola`)** | **Realistic Degradation** | Models were evaluated on held-out test fonts (`tahoma`, `consola`). On Tahoma, word CER is 0.08%; on monospace Consolas, CER jumps to 1.02% (and line Consolas CER is 3.48%). This properly penalizes font mismatch. |
| **Simulated Camera Ladder (`sim_ladder`)** | **Strong Pessimistic Bias for Word Model** | Word model was never exposed to digits or punctuation during training due to `WORD_RE` filtering. In `sim_ladder` (derived from natural lines), words containing punctuation and numbers suffer heavy substitution errors (7.71% baseline CER at L0 with zero distortion). |
| **Preprocessing Discontinuity** | **Strong Pessimistic Bias Under Shadows** | Padding with pure white (`255`) on darker paper backgrounds severely degrades CNN feature maps without model retraining, exaggerating distortion penalties (35.77% CER at L4). |

### Claim 9: Timing Comparability Between CRNN and Tesseract

The metric `ms_per_image` reported in `results/*/metrics.json` is **fundamentally not comparable** between the CRNN models and Tesseract:

1. **CRNN Measurement ([evaluate.py:32-52](file:///d:/laragon/www/az_ocr_bm606/evaluate.py#L32-L52)):**
   * **Inference Mode:** Batched processing (`batch_size=128`).
   * **What is timed:** Strictly the model tensor forward pass (`model(x)`) with device synchronization (`torch.cuda.synchronize()`).
   * **What is excluded:** Disk I/O (`cv2.imread`), preprocessing pipeline (`preprocess`), memory allocation, DataLoader batch collation, and CTC greedy decoding (`Vocab.decode_greedy`).
   * **Result:** Reflects amortized accelerator compute throughput (~4.5 - 5.2 ms/word image, ~19 - 22 ms/line image).

2. **Tesseract Measurement ([evaluate.py:61-66](file:///d:/laragon/www/az_ocr_bm606/evaluate.py#L61-L66)):**
   * **Inference Mode:** Strictly sequential, single-image execution (`batch_size=1`).
   * **What is timed:** Total wall-clock duration of the loop.
   * **What is included:** Disk read (`Image.open(path)`), PIL color conversion, spawning an external operating system process (`pytesseract` executing `tesseract.exe` via CLI), process startup/shutdown, loading linguistic dictionaries and language data files, execution of Tesseract OCR pipeline, and string post-processing (`normalize`).
   * **Result:** Reflects per-sample OS process overhead (~92 - 105 ms/image), dominated by process invocation rather than core OCR engine execution time.

### Claim 10: Proposal Gap Analysis (Unimplemented Features)

Relative to the original research/course proposal, the following 6 components are confirmed to be **unimplemented or incomplete**:

1. **TrOCR Architecture:**
   * *Status:* **Not Implemented.**
   * *Evidence:* The codebase only defines and trains CRNN (`model.py`: VGG-style Conv2d + 2-layer Bidirectional LSTM + Linear projection). No Vision Transformer (ViT) or Transformer decoder architecture exists.
2. **Beam Search Decoding & Language Model:**
   * *Status:* **Not Implemented.**
   * *Evidence:* Only greedy argmax decoding is implemented ([common.py:43-49](file:///d:/laragon/www/az_ocr_bm606/common.py#L43-L49)). There is no CTC beam search decoder, and no n-gram or neural language model integration.
3. **Word Segmentation Error Evaluation:**
   * *Status:* **Not Implemented.**
   * *Evidence:* All evaluation scripts (`evaluate.py`, `run_sim_ladder.py`) take pre-cropped word images or pre-cropped line images as input. No text detection module (e.g., CRAFT, DBNet, bounding box IoU) or word segmentation error metric exists.
4. **Real-World Photo Test Set:**
   * *Status:* **Not Implemented.**
   * *Evidence:* Evaluated only on synthetic TRDG images and simulated phone renders (`sim_preview`, `sim_ladder`). In `run_all.ps1:48, 55, 67`, the script explicitly checks for `data\real_word` and `data\real_line` and prints `"No real word test set ... skipped"`.
5. **Preprocessing Ablation Study:**
   * *Status:* **Incomplete.**
   * *Evidence:* `run_ablation.py` was created, but only a 50-sample smoke test (`results/smoke`) was ever executed. A full factorial ablation table (grayscale vs RGB, Otsu vs Adaptive vs None, Gaussian vs Median vs None) across the test benchmark was never computed.
6. **Cost Metric:**
   * *Status:* **Incomplete.**
   * *Evidence:* `metrics.json` records latency (`ms_per_image`), parameter count (`8.23M`), and peak GPU memory (`gpu_mem_mb`). No financial compute cost metric (dollar cost per 1,000 pages, cloud inference expense, or FLOPs/MACs) is defined or logged.

---
*Audit completed with zero modifications to existing project code, configuration, or environment.*
