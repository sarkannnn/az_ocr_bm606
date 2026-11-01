"""Update paper/paper_v4.md and build paper/paper_v4.docx:
- Abstract: restore $L_4$, $n=158$, $n=111$ and summarize all 6 strengths of CRNN V2.
- Section 1.1 (Contributions): 5 contributions highlighting CPU speed (2.74x-3.34x), uppercase special character superiority (2.46% vs 3.69%), 4-way i/ı/İ/I accuracy, audit protocol, cross-mode PSM 7/13 analysis, and intrinsic CTC confidence calibration (AUROC = 0.848, 0.47% hybrid CER).
- Section 6.8.1: fix '$ ext{CER}' -> '$\\text{CER}'.
- Section 10 (Conclusion): 'Where CRNN V2 Excels and What Remains Open' summarizing all 6 confirmed areas of superiority alongside open challenges.
- Rebuild paper/paper_v4.docx with all 15 tables and 15 figures.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_V4_MD = ROOT / "paper" / "paper_v4.md"


def update_markdown():
    md = PAPER_V4_MD.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_abstract = r"""## Abstract

Optical character recognition (OCR) for Azerbaijani is challenging under smartphone captures because of its dense diacritic inventory (ə, ı, ö, ğ, ü, ş, ç). We present a CPU-trained Convolutional Recurrent Neural Network (CRNN, 8.23M parameters, [CPU MODEL], 3–5 h training) together with a transparent audit protocol that exposes pitfalls commonly inflating OCR results: font-family leakage, bounding-box bugs, missing space tokens, and segmentation-mode sensitivity. Under simulated mobile distortion ($), photometric augmentation prevents model collapse, keeping 1.35% word and 0.11% line Character Error Rate (CER) where the baseline reaches 35.77% and 12.96% (2 of 4 ladder fonts are family-leaked in V2). On synthetic lines, V2 lowers CER from 1.64% to 0.30%; the audit attributes this gain to Consolas family leakage, while unseen Tahoma is unchanged (0.24% vs. 0.26%). Across 7 truly unseen font families (=158$ lines), V2 and Tesseract 5 both reach 0.71% line CER (cluster bootstrap 95% CI [-0.74%, +0.76%]; no difference detected). On five document photos (=111$ lines), V2 with rule-based chunking reaches 0.10% to 2.10% CER; on a clean 23-line subset of Photo 5, Tesseract with PSM 13 reaches 0.00% CER, ahead of V2 (1.32%). Our contributions are a practical CPU-trained model and a reproducible audit protocol; removing border-padding fragility and closing the real-document gap remain the main open challenges.

**Keywords:** Optical Character Recognition (OCR), Azerbaijani Alphabet, CRNN, CTC Loss, Photometric Augmentation, Font-Family Leakage, Evaluation Pitfalls, Cluster Bootstrap, Tesseract 5, Page Segmentation Mode."""

    new_abstract = r"""## Abstract

Optical character recognition (OCR) for Azerbaijani is challenging under smartphone captures because of its dense diacritic inventory (ə, ı, ö, ğ, ü, ş, ç) and four-way dotted/dotless `i/ı/İ/I` distinction. We present a CPU-trained Convolutional Recurrent Neural Network (CRNN, 8.23M parameters, `Intel Core i5-7200U @ 2.50 GHz`, 3–5 h training) together with a transparent audit protocol that exposes pitfalls commonly inflating OCR results: font-family leakage, bounding-box bugs, missing space tokens, and segmentation-mode sensitivity. On CPU inference across real document lines, CRNN V2 runs at **155.4 ms/line** (`batch=1`) and **127.6 ms/line** (`batch=16`)—achieving a **2.74$\times$ to 3.34$\times$ speedup** over Tesseract 5 `--psm 13` (426.0 ms/line) and up to **4.05$\times$** over `--psm 7` (516.5 ms/line). Under simulated mobile distortion ($L_4$), photometric augmentation prevents model collapse, keeping 1.35% word and 0.11% line Character Error Rate (CER) where the baseline reaches 35.77% and 12.96% (2 of 4 ladder fonts are family-leaked in V2). Across $n=50,122$ non-space synthetic characters, CRNN V2 outperforms Tesseract 5 on uppercase special Azerbaijani letters (`Ə, İ, Ö, Ğ, Ü, Ş, Ç, I`: **2.46% vs. 3.69%** error), standard lowercase (`a–z`: **0.24% vs. 0.46%**), and standard uppercase (`A–Z`: **0.74% vs. 1.14%**), while resolving the 4-way `i/ı/İ/I` paradox at 99.64% (`i`), 99.41% (`ı`), 97.30% (`İ`), and 95.45% (`I`) accuracy. Across 7 truly unseen font families ($n=158$ lines), V2 and Tesseract 5 both reach 0.71% line CER at $L_0$ (cluster bootstrap 95% CI [-0.74%, +0.76%]) and 1.28% vs. 1.46% at $L_4$. On five real document photos ($n=111$ lines), V2 with rule-based chunking reaches 0.10% to 2.10% CER (outperforming Tesseract on Photo 1 at **0.97% vs. 1.32%** under both PSM 7 and PSM 13, while Tesseract PSM 13 reaches 0.00% on the clean 23-line subset of Photo 5). Finally, we show that raw CTC posteriors provide a zero-snooping error detector (**AUROC = 0.848**), enabling a confidence-gated hybrid cascade (80% CRNN V2 + 20% Tesseract fallback) that cuts real-document CER from **1.08% to 0.47%** at a **2.00$\times$ CPU speedup**.

**Keywords:** Optical Character Recognition (OCR), Azerbaijani Alphabet, CRNN, CTC Loss, Photometric Augmentation, Font-Family Leakage, Evaluation Pitfalls, Cluster Bootstrap, Confidence Calibration, Tesseract 5, Page Segmentation Mode."""

    assert old_abstract in md, "old_abstract not found!"
    md = md.replace(old_abstract, new_abstract, 1)

    old_contrib = r"""### 1.1 Contributions

In this study, we present a complete engineering implementation and audit-driven evaluation of an Azerbaijani OCR system developed under Project BM 606. Rather than presenting idealized benchmarks, we provide an empirical accounting of genuine strengths alongside critical methodological pitfalls. Our four primary contributions are:

1. **A Practical, Low-Resource Recognition Engine:** We design an 8.23M-parameter CRNN (7-layer CNN + 2-layer BiLSTM + CTC) trained entirely under CPU compute constraints ([CPU MODEL]) within 3–5 hours per model, achieving 0.10% to 2.10% CER across five real smartphone document captures.
2. **Resilience to Photometric Camera Distortions:** We demonstrate that a calibrated photometric data augmentation pipeline (illumination gradients, sensor noise, resolution downscaling, and contrast stretching) prevents model collapse under simulated smartphone imaging, maintaining 1.35% word and 0.11% line CER under $L_4$ distortion where a baseline model collapses to 35.77% and 12.96% CER.
3. **A Reproducible Audit Protocol Exposing Common Pitfalls:** We document, quantify, and provide engineering resolutions for five project pitfalls that commonly inflate or distort OCR results: font-family leakage (Consolas 1.02% $\rightarrow$ 0.07%), bounding box metric inflation (84.32% $\rightarrow$ 0.76%), CTC missing-space vocabulary collapse (11.19% $\rightarrow$ 1.64%), aspect ratio squeeze degradation ($r > 1.3$), and canvas border padding fragility (2.10% $\rightarrow$ 23.32%).
4. **Diagnostic Decomposition of Baseline Engines:** We show that apparent 10 percentage point victories over Tesseract 5 on real document crops stem from layout segmentation failure (PSM 7 baseline loss causing empty outputs) rather than character recognition defects; on clean subsets, properly configured Tesseract (PSM 13) achieves 0.00% CER, establishing a transparent, fair benchmark for future low-resource OCR development."""

    new_contrib = r"""### 1.1 Contributions

In this study, we present a complete engineering implementation and audit-driven evaluation of an Azerbaijani OCR system developed under Project BM 606. Rather than presenting idealized benchmarks, we provide an empirical accounting of genuine strengths alongside critical methodological pitfalls. Our five primary contributions are:

1. **A Fast, Low-Resource CPU Recognition Engine:** We design an 8.23M-parameter CRNN (7-layer CNN + 2-layer BiLSTM + CTC; 31.41 MiB FP32 / 15.72 MiB FP16 pure model weights) trained entirely on CPU (`Intel Core i5-7200U @ 2.50 GHz`) within 3–5 hours per model. On CPU line inference, CRNN V2 runs **2.74$\times$ to 3.34$\times$ faster** than Tesseract 5 `--psm 13` (155.4 ms unbatched / 127.6 ms batched vs. 426.0 ms/line) and up to **4.05$\times$ faster** than `--psm 7` (516.5 ms/line), achieving 0.10% to 2.10% CER across five real smartphone document captures with **zero empty-line failures**.
2. **Resilience to Photometric Distortions and Orthographic Superiority on Azerbaijani Specials:** We show that photometric data augmentation prevents model collapse under simulated smartphone imaging ($L_4$: 1.35% word and 0.11% line CER vs. V1's 35.77% and 12.96%). Across $n=50,122$ non-space characters, CRNN V2 outperforms Tesseract 5 on **uppercase special Azerbaijani letters** (`Ə, İ, Ö, Ğ, Ü, Ş, Ç, I`: **2.46% vs. 3.69%** error), **standard lowercase** (`0.24% vs. 0.46%`), and **standard uppercase** (`0.74% vs. 1.14%`), achieves **0.00% error** on unseen distorted `ə/e`, `ğ/g`, and `ş/s` pairs at $L_4$, and resolves the 4-way `i/ı/İ/I` paradox with 99.64% (`i`), 99.41% (`ı`), 97.30% (`İ`), and 95.45% (`I`) accuracy.
3. **A Reproducible Audit Protocol Exposing Five Common Pitfalls:** We document, quantify, and provide engineering resolutions for five project pitfalls that commonly inflate or distort OCR results: font-family leakage (Consolas 1.02% $\rightarrow$ 0.07%), bounding box metric inflation (84.32% $\rightarrow$ 0.76%), CTC missing-space vocabulary collapse (11.19% $\rightarrow$ 1.64%), aspect ratio squeeze degradation ($r > 1.3$), and canvas border padding fragility (2.10% $\rightarrow$ 23.32%).
4. **Comprehensive Cross-Mode Diagnostic of Baseline Engines (Photos 1–5):** We evaluate both `--psm 7` and `--psm 13` across all five real document captures ($n=111$ lines), proving that while `--psm 13` resolves Photo 5's narrow-strip empty-output failures (0.09% full-line CER, and 0.00% on the clean 23-line subset), `--psm 13` produces the **exact same CER as `--psm 7`** on rule-based chunks across Photos 1–4 (`1.32%`, `0.14%`, `0.00%`, `0.60%`), where CRNN V2 outperforms Tesseract on Photo 1 (**0.97% vs. 1.32%**).
5. **Intrinsic CTC Confidence Calibration and Hybrid Cascade Deployment:** We demonstrate that raw CTC posterior probabilities provide a strictly zero-snooping error predictor on real photos (**AUROC = 0.8475** mean-char / **0.8413** min-char), enabling a confidence-gated hybrid cascade (routing only the lowest-confidence 20% of lines to Tesseract) that reduces aggregate real-photo CER from **1.08% to 0.47%** while retaining a **2.00$\times$ CPU speedup**."""

    assert old_contrib in md, "old_contrib not found!"
    md = md.replace(old_contrib, new_contrib, 1)

    # Fix broken '$ ext{CER}' in Section 6.8.1
    md = md.replace(r"($ ext{CER} = 0.00\%$)", r"($\text{CER} = 0.00\%$)")
    md = md.replace(r"($ ext{CER} > 0\%$)", r"($\text{CER} > 0\%$)")

    old_concl = r"""### What We Find and What Remains Open

Our transparent audit resolved several common evaluation illusions:
- The apparent dramatic synthetic gain on Consolas was traced directly to font-family leakage (`consolab.ttf` in training).
- On standard clean print of 7 truly unseen font families (Group C), no difference was detected between V2 and Tesseract 5 (0.71% vs. 0.71% Line CER; 95% CI [-0.74%, +0.76%]).
- On real smartphone captures, V2 with projection chunking achieves 0.10% to 2.10% CER. However, on clean full-line text without layout failures, a properly configured Tesseract with PSM 13 achieves **0.00% CER** (23/23 exact lines on Photo 5), outperforming V2 Line (1.32% chunk, 3.87% full line on the full 33 lines). Closing this performance gap remains the primary open challenge."""

    new_concl = r"""### Where CRNN V2 Excels and What Remains Open

Our comprehensive evaluation establishes **six concrete operational advantages** of CRNN V2 alongside transparent boundaries where Tesseract 5 retains an edge:

1. **CPU Inference Throughput (Section 4.3, Table 3b):** On identical CPU hardware (`Intel Core i5-7200U @ 2.50 GHz`), CRNN V2 Line runs **2.74$\times$ faster** unbatched (155.35 ms vs. 425.99 ms/line) and **3.34$\times$ faster** batched (127.59 ms/line) than Tesseract 5 `--psm 13`, and up to **4.05$\times$ faster** than `--psm 7` (516.53 ms/line), within a compact 31.41 MiB FP32 (15.72 MiB FP16) parameter footprint.
2. **Uppercase Special Azerbaijani Letters & Standard Alphabet Accuracy (Section 6.3.1, Table 6b):** Across $n=50,122$ non-space characters, CRNN V2 achieves lower error rates than Tesseract 5 on **uppercase special Azerbaijani characters** (`Ə, İ, Ö, Ğ, Ü, Ş, Ç, I`: **2.46% vs. 3.69%**), **standard lowercase** (`a–z`: **0.24% vs. 0.46%**), and **standard uppercase** (`A–Z`: **0.74% vs. 1.14%**), while resolving the 4-way `i/ı/İ/I` paradox at 99.64% (`i`), 99.41% (`ı`), 97.30% (`İ`), and 95.45% (`I`) accuracy with zero `İ/I` cross-confusions.
3. **Unseen Distorted Special Pairs at $L_4$ (Section 6.3, Table 6):** Under heavy simulated camera distortion ($L_4$) on unseen fonts, CRNN V2 achieves **0.00% error** on `ə/e` (vs. Tesseract 1.02%–2.03%), `ğ/g` (vs. 5.00%), and `ş/s` (vs. 1.79%).
4. **Real Document Photo 1 Across Both PSM 7 and PSM 13 (Section 6.4.1, Table 4b):** On Photo 1 ($n=16$ lines), CRNN V2 Rule Chunk achieves **0.97% CER**, outperforming Tesseract 5 under both `--psm 7` (**1.32%**) and `--psm 13` (**1.32%** chunk, **2.22%** full line).
5. **Zero Empty-Line Collapse & Zero Leading-Edge Noise (Sections 6.4 & 7.1, Tables 7 & 8):** Under standard single-line mode (`--psm 7`), CRNN V2 produces **0 empty outputs** and **0 leading-edge noise insertions**, whereas Tesseract suffers 9 empty chunks (13.02% CER) and 3 leading margin insertions on Photo 5.
6. **Seen-Font Statistical Lead & Intrinsic Confidence-Gated Hybrid Synergy (Sections 6.2 & 6.8.1, Tables 5 & 8b):** CRNN V2 holds a statistically significant lead on seen-font words (0.81% vs. 2.39% CER, 95% CI `[-2.68%, -0.76%]`), and its raw CTC posteriors predict real-document line errors without a lexicon (**AUROC = 0.848**), enabling an 80% CRNN + 20% Tesseract cascade that cuts real-photo CER from **1.08% to 0.47%** at a **2.00$\times$ CPU speedup**.

At the same time, our audit resolved several evaluation illusions and defined the remaining engineering challenges:
- The apparent synthetic gain on Consolas was traced to font-family leakage (`consolab.ttf` in training).
- Across 7 truly unseen font families (Group C), V2 and Tesseract 5 are statistically tied at $L_0$ (0.71% vs. 0.71% Line CER; 95% CI `[-0.74%, +0.76%]`) and $L_4$ (1.28% vs. 1.46%).
- On unchunked full lines where aspect ratio squeeze exceeds $r > 1.3$ or on clean full-line text without layout failures (Photo 5 Clean 23), Tesseract `--psm 13` achieves **0.00% CER** (23/23 exact lines) compared to V2's 1.32% chunk / 3.87% full-line CER. Eliminating fixed-width squeeze and border-padding fragility remains the primary architectural frontier."""

    assert old_concl in md, "old_concl not found!"
    md = md.replace(old_concl, new_concl, 1)

    PAPER_V4_MD.write_text(md, encoding="utf-8")
    print(f"Updated {PAPER_V4_MD} ({PAPER_V4_MD.stat().st_size:,} bytes).")


def build_docx_v4():
    v4_builder = ROOT / "paper" / "scratch" / "build_monograph_docx_v4.py"
    subprocess.run([sys.executable, str(v4_builder)], check=True)


if __name__ == "__main__":
    update_markdown()
    build_docx_v4()
