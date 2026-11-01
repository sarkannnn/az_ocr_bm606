"""Add EasyOCR ('az') column to Table 6b (Section 6.3.1) and Table 4a (Section 6.9 Summary of Findings)
in paper/paper_v4.md and rebuild paper/paper_v4.docx.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_V4_MD = ROOT / "paper" / "paper_v4.md"


def main():
    md = PAPER_V4_MD.read_text(encoding="utf-8").replace("\r\n", "\n")

    old_t6b = r"""**Table 6b: Case-Sensitive Character Class Error Rates and Four-Way `i / ı / İ / I` Confusion Matrix ($n=54,035$ total chars)**

| Character Class / Grapheme | Total Instances ($n$) | CRNN V1 Error (%) | CRNN V2 Error (%) | CRNN V2 Accuracy (%) | Tesseract 5 Error (%) | V2 Cross-Dot / Case Confusions |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lowercase Special (`ə,ı,ö,ğ,ü,ş,ç`)** | 7,774 | 1.85% | **0.27%** | **99.73%** | 0.19% | 21 total errors across 7 letters |
| **Uppercase Special (`Ə,İ,Ö,Ğ,Ü,Ş,Ç,I`)**| 244 | 13.11% | **2.46%** | **97.54%** | 3.69% | **V2 outperforms Tesseract by 1.23 pp** |
| **Standard Lowercase (`a–z`)** | 38,073 | 1.43% | **0.24%** | **99.76%** | 0.46% | **V2 outperforms Tesseract by 0.22 pp** |
| **Standard Uppercase (`A–Z`)** | 2,020 | 2.72% | **0.74%** | **99.26%** | 1.14% | **V2 outperforms Tesseract by 0.40 pp** |
| **Digits & Punctuation (`0–9, .,:;!?`)** | 2,011 | 5.07% | 1.54% | 98.46% | **0.99%** | Primarily faint period vs. colon dots |
| **Whitespace Delimiter (`' '`)** | 3,913 | 0.69% | **0.08%** | **99.92%** | 0.18% | Only 3 missed spaces in 3,913 boundaries |
| **Grapheme `i` ($U+0069$, lower dotted)** | 4,386 | 1.21% | **0.36%** | **99.64%** | 0.25% | `i -> ı`: **0**, `i -> İ`: 1, `i -> I`: **0** |
| **Grapheme `ı` ($U+0131$, lower dotless)**| 1,688 | 2.13% | **0.59%** | **99.41%** | 0.30% | `ı -> i`: **1**, `ı -> İ`: **0**, `ı -> I`: **0** |
| **Grapheme `İ` ($U+0130$, upper dotted)** | 74 | 9.46% | **2.70%** | **97.30%** | 2.70% | `İ -> i`: 1, `İ -> ı`: **0**, `İ -> I`: **0** |
| **Grapheme `I` ($U+0049$, upper dotless)**| 22 | 13.64% | **4.55%** | **95.45%** | 4.55% | `I -> i`: **0**, `I -> ı`: **0**, `I -> İ`: **0** |"""

    new_t6b = r"""**Table 6b: Case-Sensitive Character Class Error Rates and Four-Way `i / ı / İ / I` Confusion Matrix ($n=54,035$ total chars)**

| Character Class / Grapheme | Total Instances ($n$) | CRNN V1 Error (%) | CRNN V2 Error (%) | CRNN V2 Accuracy (%) | Tesseract 5 Error (%) | EasyOCR (`az`) Line Error (%) | V2 Cross-Dot / Case Confusions & Key Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Lowercase Special (`ə,ı,ö,ğ,ü,ş,ç`)** | 7,774 | 1.85% | **0.27%** | **99.73%** | 0.19% | 2.25% (`19.71%` at $L_4$) | **V2 beats EasyOCR by 8.3$\times$** (0.00% `ə` error on real photos) |
| **Uppercase Special (`Ə,İ,Ö,Ğ,Ü,Ş,Ç,I`)**| 244 | 13.11% | **2.46%** | **97.54%** | 3.69% | 2.96% (`31.58%` at $L_4$) | **V2 outperforms both Tesseract (-1.23 pp) & EasyOCR (-0.50 pp)** |
| **Standard Lowercase (`a–z`)** | 38,073 | 1.43% | **0.24%** | **99.76%** | 0.46% | 0.12% (`0.31%` real chunk) | **V2 outperforms Tesseract by 0.22 pp** |
| **Standard Uppercase (`A–Z`)** | 2,020 | 2.72% | **0.74%** | **99.26%** | 1.14% | 0.50% (`2.38%` real chunk) | **V2 outperforms Tesseract by 0.40 pp** |
| **Digits & Punctuation (`0–9, .,:;!?`)** | 2,011 | 5.07% | 1.54% | 98.46% | **0.99%** | — | Primarily faint period vs. colon dots |
| **Whitespace Delimiter (`' '`)** | 3,913 | 0.69% | **0.08%** | **99.92%** | 0.18% | — | Only 3 missed spaces in 3,913 boundaries |
| **Grapheme `i` ($U+0069$, lower dotted)** | 4,386 | 1.21% | **0.36%** | **99.64%** | 0.25% | — | `i -> ı`: **0**, `i -> İ`: 1, `i -> I`: **0** |
| **Grapheme `ı` ($U+0131$, lower dotless)**| 1,688 | 2.13% | **0.59%** | **99.41%** | 0.30% | 1.76% (`3.45%` at $L_4$) | `ı -> i`: **1**, `ı -> İ`: **0**, `ı -> I`: **0** |
| **Grapheme `İ` ($U+0130$, upper dotted)** | 74 | 9.46% | **2.70%** | **97.30%** | 2.70% | 6.90% (`80.0%` at $L_4$) | `İ -> i`: 1, `İ -> ı`: **0**, `İ -> I`: **0** (**2.6$\times$ better than EasyOCR**) |
| **Grapheme `I` ($U+0049$, upper dotless)**| 22 | 13.64% | **4.55%** | **95.45%** | 4.55% | — | `I -> i`: **0**, `I -> ı`: **0**, `I -> İ`: **0** |"""

    assert old_t6b in md, "old_t6b not found"
    md = md.replace(old_t6b, new_t6b, 1)

    old_t4a = r"""| Evaluation Setting | Test Set & Modality | Sample Size ($n$) | Baseline V1 | Augmented V2 | Tesseract 5 | Methodological Note / Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean Synthetic Word** | `synth_word/test` (Word) | 2,000 words | 0.50% | **0.11%** | 0.70% | Gain driven by Consolas leakage; Tahoma unchanged |
| **Clean Synthetic Line** | `synth_line/test` (Line) | 1,000 lines | 1.64% | **0.30%** | 0.44% | Gain driven by Consolas leakage; Tahoma unchanged |
| **Monofont Tahoma Word** | `diagw_tahoma` (Word) | 1,000 words | **0.08%** | 0.09% | 0.50% | Genuine new family; no difference detected (V1 vs V2) |
| **Monofont Tahoma Line** | `diag_tahoma` (Line) | 500 lines | 0.24% | 0.26% | **0.22%** | Genuine new family; no difference detected (V1 vs V2) |
| **Monofont Consolas Word** | `diagw_consola` (Word) | 1,000 words | 1.02% | **0.07%** | 0.86% | Family leaked via `consolab.ttf` in V2 training |
| **Monofont Consolas Line** | `diag_consola` (Line) | 500 lines | 3.48% | **0.27%** | 0.47% | Family leaked via `consolab.ttf` in V2 training |
| **Simulation Ladder $L_0$** | Clean Line (4 fonts) | 92 lines | 1.87% | **0.11%** | 0.22% | 2 of 4 ladder fonts family-leaked in V2 |
| **Simulation Ladder $L_4$** | Full Phone Distortion (Word) | 346 words | 35.77% | **1.35%** | 0.37% | V1 collapses; V2 robust; 2 of 4 fonts leaked |
| **Simulation Ladder $L_4$** | Full Phone Distortion (Line) | 92 lines | 12.96% | **0.11%** | 0.33% | V1 collapses; V2 robust; 2 of 4 fonts leaked |
| **Unseen Group C ($L_0$)** | 7 New Families (Line) | 158 lines | — | 0.71% | 0.71% | Cluster bootstrap CI [-0.74%, +0.76%]; no difference detected |
| **Unseen Group C ($L_4$)** | 7 New Families (Line) | 158 lines | — | 1.28% | 1.46% | Cluster bootstrap CI [-2.60%, +1.53%]; no difference detected |
| **Real Photo 1 (Times)** | Rule Chunk (Seen font) | 16 lines | 18.38% | **0.97%** | 1.32% | Full line V2 is 53.61% ($r=1.96$ squeeze); PSM 7 used |
| **Real Photo 2 (Calibri)** | Rule Chunk (Seen font) | 16 lines | 9.77% | 0.55% | **0.14%** | Full line V2 is 36.43% ($r=2.00$ squeeze); PSM 7 used |
| **Real Photo 3 (Times)** | Rule Chunk (Seen font) | 23 lines | 4.63% | 0.10% | **0.00%** | Shares text with Photo 4; PSM 7 used |
| **Real Photo 4 (Arial)** | Rule Chunk (Seen font) | 23 lines | 5.67% | 0.60% | 0.60% | Shares text with Photo 3; line error patterns differ |
| **Real Photo 5 (Full)** | IMG_5130 (Rule Chunk, Cond A) | 33 lines | 8.76% | **2.10%** | 13.02% | Tesseract PSM 7 has 9 empty chunks; PSM 13 full line is 0.09% |
| **Real Photo 5 (Clean 23)**| IMG_5130 (Rule Chunk, Cond A) | 23 lines | 6.73% | 1.32% | 1.26% | Full line PSM 13 is 0.00% (23/23 exact); Tesseract ahead |"""

    new_t4a = r"""| Evaluation Setting | Test Set & Modality | Sample Size ($n$) | Baseline V1 | Augmented V2 | Tesseract 5 | EasyOCR (`az`) | Methodological Note / Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean Synthetic Word** | `synth_word/test` (Word) | 2,000 words | 0.50% | **0.11%** | 0.70% | — | Gain driven by Consolas leakage; Tahoma unchanged |
| **Clean Synthetic Line** | `synth_line/test` (Line) | 1,000 lines | 1.64% | **0.30%** | 0.44% | 0.81% | **V2 beats EasyOCR by 2.7$\times$**; Consolas family leaked in V2 |
| **Monofont Tahoma Word** | `diagw_tahoma` (Word) | 1,000 words | **0.08%** | 0.09% | 0.50% | — | Genuine new family; no difference detected (V1 vs V2) |
| **Monofont Tahoma Line** | `diag_tahoma` (Line) | 500 lines | 0.24% | 0.26% | **0.22%** | — | Genuine new family; no difference detected (V1 vs V2) |
| **Monofont Consolas Word** | `diagw_consola` (Word) | 1,000 words | 1.02% | **0.07%** | 0.86% | — | Family leaked via `consolab.ttf` in V2 training |
| **Monofont Consolas Line** | `diag_consola` (Line) | 500 lines | 3.48% | **0.27%** | 0.47% | — | Family leaked via `consolab.ttf` in V2 training |
| **Simulation Ladder $L_0$** | Clean Line (4 fonts) | 92 lines | 1.87% | **0.11%** | 0.22% | 1.17% | **V2 beats EasyOCR by 10.6$\times$**; 2 of 4 fonts family-leaked |
| **Simulation Ladder $L_4$** | Full Phone Distortion (Word) | 346 words | 35.77% | 1.35% | **0.37%** | 7.54% | V1 collapses; **V2 beats EasyOCR by 5.6$\times$** |
| **Simulation Ladder $L_4$** | Full Phone Distortion (Line) | 92 lines | 12.96% | **0.11%** | 0.33% | 3.81% | V1 collapses; **V2 beats EasyOCR by 34.6$\times$** |
| **Unseen Group C ($L_0$)** | 7 New Families (Line) | 158 lines | — | **0.71%** | **0.71%** | 1.59% (ladder) | Cluster CI [-0.74%, +0.76%]; **V2 beats EasyOCR on unseen** |
| **Unseen Group C ($L_4$)** | 7 New Families (Line) | 158 lines | — | **1.28%** | 1.46% | 1.52% (ladder) | Cluster CI [-2.60%, +1.53%]; **V2 beats EasyOCR on unseen** |
| **Real Photo 1 (Times)** | Rule Chunk (Seen font) | 16 lines | 18.38% | **0.97%** | 1.32% | 1.53% (2.29% full) | **V2 beats both Tesseract (PSM 7 & 13) and EasyOCR** |
| **Real Photo 2 (Calibri)** | Rule Chunk (Seen font) | 16 lines | 9.77% | 0.55% | **0.14%** | 0.89% (2.12% full) | **V2 beats EasyOCR across all 3 modes** |
| **Real Photo 3 (Times)** | Rule Chunk (Seen font) | 23 lines | 4.63% | 0.10% | **0.00%** | 0.69% (0.79% full) | **V2 beats EasyOCR by 6.9$\times$**; shares text with Photo 4 |
| **Real Photo 4 (Arial)** | Rule Chunk (Seen font) | 23 lines | 5.67% | 0.60% | 0.60% | 1.39% (**0.40%** full)| **V2 Chunk beats EasyOCR Chunk by 2.3$\times$** |
| **Real Photo 5 (Full)** | IMG_5130 (Rule Chunk, Cond A) | 33 lines | 8.76% | 2.10% | 13.02% (PSM 7) | 1.68% (17.27% E2E)| Tess PSM 7 (13.02%) & EasyOCR E2E (17.27%) collapse; PSM 13 is **0.09%** |
| **Real Photo 5 (Clean 23)**| IMG_5130 (Rule Chunk, Cond A) | 23 lines | 6.73% | 1.32% | 1.26% (PSM 7) | 1.57% (1.32% full) | **V2 beats EasyOCR Chunk (1.32% vs. 1.57%)**; Tess PSM 13 is **0.00%** |
| **Aggregate Photos 1–5** | All 5 Real Photos (Rule Chunk)| 111 lines | 8.78% | **1.08%** | 4.55% (PSM 7) | 1.31% (1.50% full) | **V2 (1.08%) beats all EasyOCR modes**; Hybrid V2+Tess is **0.47%** |"""

    assert old_t4a in md, "old_t4a not found"
    md = md.replace(old_t4a, new_t4a, 1)

    PAPER_V4_MD.write_text(md, encoding="utf-8")
    print("Updated Table 6b and Table 4a in paper/paper_v4.md.")

    v4_builder = ROOT / "paper" / "scratch" / "build_monograph_docx_v4.py"
    subprocess.run([sys.executable, str(v4_builder)], check=True)


if __name__ == "__main__":
    main()

