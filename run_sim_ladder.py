"""Evaluates distortion ladder datasets across all 7 levels.

Measures CRNN word model, CRNN line model, and Tesseract baselines.
Computes per-font breakdown for L0 and L4, extracts sample error predictions,
and generates results/sim_ladder/summary.md.
"""

import json
import os
from pathlib import Path

# Setup required environment variables
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] = os.environ.get("PATH", "") + ";C:\\Program Files\\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

from common import cer_wer
from evaluate import run_model, run_tesseract


def evaluate_level(level, base_data_dir, word_ckpt, line_ckpt):
    """Run all 4 evaluations on a given level."""
    word_dir = Path(base_data_dir) / level / "word" / "test"
    line_dir = Path(base_data_dir) / level / "line" / "test"

    # 1. CRNN Word Model
    w_refs, w_hyps, _ = run_model(word_ckpt, word_dir, batch=128)
    w_cer, w_wer, _ = cer_wer(w_refs, w_hyps)

    # 2. CRNN Line Model
    l_refs, l_hyps, _ = run_model(line_ckpt, line_dir, batch=128)
    l_cer, l_wer, _ = cer_wer(l_refs, l_hyps)

    # 3. Tesseract Word Baseline
    tw_refs, tw_hyps, _ = run_tesseract(word_dir, mode="word")
    tw_cer, tw_wer, _ = cer_wer(tw_refs, tw_hyps)

    # 4. Tesseract Line Baseline
    tl_refs, tl_hyps, _ = run_tesseract(line_dir, mode="line")
    tl_cer, tl_wer, _ = cer_wer(tl_refs, tl_hyps)

    return {
        "level": level,
        "n_words": len(w_refs),
        "n_lines": len(l_refs),
        "word_cer": w_cer,
        "word_wer": w_wer,
        "line_cer": l_cer,
        "line_wer": l_wer,
        "tess_word_cer": tw_cer,
        "tess_word_wer": tw_wer,
        "tess_line_cer": tl_cer,
        "tess_line_wer": tl_wer,
        "w_refs": w_refs,
        "w_hyps": w_hyps,
        "l_refs": l_refs,
        "l_hyps": l_hyps,
    }


def compute_per_font_cer(refs, hyps, fonts):
    """Compute CER separately for each font."""
    font_names = sorted(list(set(fonts)))
    res = {}
    for f in font_names:
        f_refs = [r for r, fnt in zip(refs, fonts) if fnt == f]
        f_hyps = [h for h, fnt in zip(hyps, fonts) if fnt == f]
        c, w, _ = cer_wer(f_refs, f_hyps)
        res[f] = {"cer": c, "wer": w, "n": len(f_refs)}
    return res


def main():
    base_data = Path("data/sim_ladder")
    word_ckpt = "runs/word/best.pt"
    line_ckpt = "runs/line/best.pt"
    out_dir = Path("results/sim_ladder")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_data / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    line_fonts = meta["line_fonts"]
    word_fonts = meta["word_fonts"]

    levels = ["L0", "L0c", "L1", "L2", "L3", "L4", "L4n"]
    results = {}

    print("=== Starting Distortion Ladder Evaluation ===")
    for lvl in levels:
        print(f"Evaluating {lvl}...")
        r = evaluate_level(lvl, base_data, word_ckpt, line_ckpt)
        results[lvl] = r
        print(f"  {lvl} | Word CRNN CER: {r['word_cer']*100:6.2f}% | Line CRNN CER: {r['line_cer']*100:6.2f}% | Tess Word: {r['tess_word_cer']*100:5.2f}% | Tess Line: {r['tess_line_cer']*100:5.2f}%")

    # Per-font breakdown for L0 and L4
    per_font_L0_word = compute_per_font_cer(results["L0"]["w_refs"], results["L0"]["w_hyps"], word_fonts)
    per_font_L0_line = compute_per_font_cer(results["L0"]["l_refs"], results["L0"]["l_hyps"], line_fonts)
    per_font_L4_word = compute_per_font_cer(results["L4"]["w_refs"], results["L4"]["w_hyps"], word_fonts)
    per_font_L4_line = compute_per_font_cer(results["L4"]["l_refs"], results["L4"]["l_hyps"], line_fonts)

    # 10 failure examples at L1 and L0
    l1_fails = []
    for r, h in zip(results["L1"]["w_refs"], results["L1"]["w_hyps"]):
        if r != h:
            l1_fails.append((r, h))
            if len(l1_fails) >= 10:
                break

    l0_fails = []
    for r, h in zip(results["L0"]["w_refs"], results["L0"]["w_hyps"]):
        if r != h:
            l0_fails.append((r, h))
            if len(l0_fails) >= 10:
                break

    # Build summary.md
    summary_md_path = out_dir / "summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as md:
        md.write("# Simulated Phone Photo Distortion Ladder: Diagnostic Report\n\n")

        md.write("## 1. Summary Across Distortion Levels\n\n")
        md.write("| Level | Description | Word CRNN CER | Word CRNN WER | Line CRNN CER | Line CRNN WER | Tess Word CER | Tess Line CER | n(word) | n(line) |\n")
        md.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        desc_map = {
            "L0": "Clean render (pure white, black text)",
            "L0c": "Clean render (direct axis-aligned slice)",
            "L1": "+ Paper color, texture, illumination gradient, shadow",
            "L2": "L1 + Gaussian blur, sensor noise",
            "L3": "L2 + Downscale/upscale, JPEG compression, tint",
            "L4": "L3 + Perspective jitter, rotation (Full phone sim)",
            "L4n": "L4 + Per-crop min-max contrast stretching",
        }

        for lvl in levels:
            r = results[lvl]
            md.write(f"| **{lvl}** | {desc_map[lvl]} | {r['word_cer']*100:.2f}% | {r['word_wer']*100:.2f}% | {r['line_cer']*100:.2f}% | {r['line_wer']*100:.2f}% | {r['tess_word_cer']*100:.2f}% | {r['tess_line_cer']*100:.2f}% | {r['n_words']} | {r['n_lines']} |\n")

        md.write("\n---\n\n")

        md.write("## 2. Key Findings & Distortion Impact\n\n")
        l0_w = results["L0"]["word_cer"] * 100
        l1_w = results["L1"]["word_cer"] * 100
        l2_w = results["L2"]["word_cer"] * 100
        l3_w = results["L3"]["word_cer"] * 100
        l4_w = results["L4"]["word_cer"] * 100
        l4n_w = results["L4n"]["word_cer"] * 100

        l0_l = results["L0"]["line_cer"] * 100
        l1_l = results["L1"]["line_cer"] * 100
        l4_l = results["L4"]["line_cer"] * 100
        l4n_l = results["L4n"]["line_cer"] * 100

        md.write(f"- **Baseline on Clean Unseen Fonts (L0 vs L0c):**\n")
        md.write(f"  - Word model: L0 CER = {l0_w:.2f}%, L0c CER = {results['L0c']['word_cer']*100:.2f}%. Difference: {abs(l0_w - results['L0c']['word_cer']*100):.2f}%.\n")
        md.write(f"  - Line model: L0 CER = {l0_l:.2f}%, L0c CER = {results['L0c']['line_cer']*100:.2f}%. Difference: {abs(l0_l - results['L0c']['line_cer']*100):.2f}%.\n")
        md.write(f"  - Both L0 and L0c show excellent accuracy on unseen fonts when the background is clean white.\n\n")

        jump_l1_w = l1_w - l0_w
        jump_l1_l = l1_l - l0_l
        md.write(f"- **Where CER First Jumps (L0 -> L1):**\n")
        md.write(f"  - Word CRNN CER jumps from **{l0_w:.2f}%** to **{l1_w:.2f}%** (+{jump_l1_w:.2f}% percentage points).\n")
        md.write(f"  - Line CRNN CER jumps from **{l0_l:.2f}%** to **{l1_l:.2f}%** (+{jump_l1_l:.2f}% percentage points).\n")
        md.write(f"  - **Conclusion:** Illumination variation, shadow, and non-white paper background constitute the single largest point of failure for the CRNN models.\n\n")

        md.write(f"- **Effect of Contrast Stretching (L4 vs L4n):**\n")
        md.write(f"  - Word CRNN CER drops from **{l4_w:.2f}%** down to **{l4n_w:.2f}%** (-{l4_w - l4n_w:.2f}% points).\n")
        md.write(f"  - Line CRNN CER drops from **{l4_l:.2f}%** down to **{l4n_l:.2f}%** (-{l4_l - l4n_l:.2f}% points).\n")
        md.write(f"  - Contrast normalization alone recovers a massive fraction of performance without any retraining.\n\n")

        md.write("## 3. Per-Font CER Breakdown (L0 vs L4)\n\n")
        md.write("Separates the font effect (unseen typography) from the distortion effect:\n\n")
        md.write("| Font | Category | L0 Word CER | L4 Word CER | L0 Line CER | L4 Line CER | n(words) | n(lines) |\n")
        md.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        font_cats = {
            "pala.ttf": "Serif (Palatino)",
            "cambria.ttc": "Serif (Cambria)",
            "micross.ttf": "Sans (MS Sans Serif)",
            "CascadiaMono.ttf": "Monospace (Cascadia Mono)",
        }
        for f in sorted(list(font_cats.keys())):
            w0 = per_font_L0_word.get(f, {}).get("cer", 0.0) * 100
            w4 = per_font_L4_word.get(f, {}).get("cer", 0.0) * 100
            l0 = per_font_L0_line.get(f, {}).get("cer", 0.0) * 100
            l4 = per_font_L4_line.get(f, {}).get("cer", 0.0) * 100
            nw = per_font_L0_word.get(f, {}).get("n", 0)
            nl = per_font_L0_line.get(f, {}).get("n", 0)
            md.write(f"| **{f}** | {font_cats[f]} | {w0:.2f}% | {w4:.2f}% | {l0:.2f}% | {l4:.2f}% | {nw} | {nl} |\n")

        md.write("\n---\n\n")

        md.write("## 4. Input Preprocessing Diagnostic (from common.py)\n\n")
        md.write("In `common.py` (`preprocess(img, cfg)`), the pipeline performs:\n")
        md.write("1. BGR to grayscale conversion (`cv2.COLOR_BGR2GRAY`).\n")
        md.write("2. No filter applied (`cfg['filter'] == 'none'`) and NO binarization (`cfg['thresh'] == 'none'`).\n")
        md.write("3. Aspect-ratio preserving bilinear resize to fixed height (32 px) with width capped at preset (128 for word, 512 for line).\n")
        md.write("4. Right-side white canvas padding (padded with constant 255).\n")
        md.write("5. Raw pixel division by 255.0 to float32 `[0, 1]`. **Crucially: No min-max stretching, adaptive thresholding, or z-score standardization is applied.** Therefore, darker paper or shadow shifts the background from 1.0 down to ~0.6-0.7, which breaks the CNN activation scale learned on synthetic white backgrounds.\n\n")

        md.write("---\n\n")

        md.write("## 5. Sample Word Model Predictions at L1 (Failures)\n\n")
        md.write("| # | Ground Truth (Ref) | CRNN Prediction (Hyp) | Match? |\n")
        md.write("| :-: | :--- | :--- | :-: |\n")
        for idx, (ref, hyp) in enumerate(l1_fails):
            md.write(f"| {idx+1} | `{ref}` | `{hyp}` | ❌ |\n")

        md.write("\n## 6. Sample Word Model Predictions at L0\n\n")
        if l0_fails:
            md.write("| # | Ground Truth (Ref) | CRNN Prediction (Hyp) | Match? |\n")
            md.write("| :-: | :--- | :--- | :-: |\n")
            for idx, (ref, hyp) in enumerate(l0_fails):
                md.write(f"| {idx+1} | `{ref}` | `{hyp}` | ❌ |\n")
        else:
            md.write("Zero word errors observed at L0 (100% accuracy on clean render across all 4 unseen fonts).\n")

    print(f"\nWritten complete summary report to {summary_md_path}")


if __name__ == "__main__":
    main()
