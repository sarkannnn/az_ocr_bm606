"""Audit 2 script: Contact sheets for word crops, per-font Tesseract evaluation, and crop bug diagnosis.
"""

import os
import sys
import json
import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Setup environment
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] = os.environ.get("PATH", "") + ";C:\\Program Files\\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

from common import cer_wer, load_labels
from evaluate import run_model, run_tesseract

V1_WORD_CKPT = "runs/word/best.pt"
V1_LINE_CKPT = "runs/line/best.pt"
V2_WORD_CKPT = "runs/word_v2_20261027_180300/best.pt"
V2_LINE_CKPT = "runs/line_v2/best.pt"

def build_contact_sheet(word_items, line_items, out_path, font_name, n_words=30, n_lines=3, seed=42):
    random.seed(seed)
    sampled_words = random.sample(word_items, min(n_words, len(word_items)))
    sampled_lines = random.sample(line_items, min(n_lines, len(line_items))) if line_items else []

    cols = 5
    rows = math.ceil(len(sampled_words) / cols)
    card_w = 260
    card_h = 105
    header_h = 60
    
    line_section_h = (len(sampled_lines) * 70 + 40) if sampled_lines else 0
    total_w = cols * card_w + 40
    total_h = header_h + line_section_h + rows * card_h + 30

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    lbl_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
    hdr_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)

    draw.text((20, 15), f"Crop Verification Contact Sheet: {font_name} (L0)", fill=(0, 40, 120), font=title_font)
    
    y_offset = header_h

    # Draw lines if any
    if sampled_lines:
        draw.text((20, y_offset), "Sample Full Line Crops (Reference):", fill=(30, 30, 30), font=hdr_font)
        y_offset += 25
        for idx, (lp, l_lbl) in enumerate(sampled_lines):
            l_im = Image.open(lp)
            draw.text((25, y_offset), f"Line {idx+1}: {l_lbl}", fill=(60, 60, 60), font=lbl_font)
            y_offset += 18
            canvas.paste(l_im, (25, y_offset))
            y_offset += l_im.height + 10
        y_offset += 15

    # Draw words
    draw.text((20, y_offset), f"Sample Word Crops ({len(sampled_words)} samples):", fill=(30, 30, 30), font=hdr_font)
    y_offset += 25

    word_start_y = y_offset
    for idx, (wp, w_lbl) in enumerate(sampled_words):
        r = idx // cols
        c = idx % cols
        x_cell = 20 + c * card_w
        y_cell = word_start_y + r * card_h

        draw.text((x_cell, y_cell), f"[{idx+1}] {w_lbl}", fill=(0, 80, 0), font=lbl_font)
        
        w_im = Image.open(wp)
        # Paste below label
        canvas.paste(w_im, (x_cell, y_cell + 20))
        # Draw bounding rect around the crop area
        draw.rectangle([x_cell, y_cell + 20, x_cell + w_im.width, y_cell + 20 + w_im.height], outline=(200, 200, 200), width=1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved contact sheet to {out_path}")

def eval_per_font(ckpt, data_dir, fonts, is_word=True, is_tess=False):
    font_names = sorted(list(set(fonts)))
    mode = "word" if is_word else "line"
    if is_tess:
        refs, hyps, _ = run_tesseract(data_dir, mode=mode)
    else:
        refs, hyps, _ = run_model(ckpt, data_dir, batch=128)
        
    res = {}
    for f in font_names:
        f_refs = [r for r, fnt in zip(refs, fonts) if fnt == f]
        f_hyps = [h for h, fnt in zip(hyps, fonts) if fnt == f]
        c, w, _ = cer_wer(f_refs, f_hyps)
        res[f] = {"cer": c, "wer": w, "n": len(f_refs), "refs": f_refs, "hyps": f_hyps}
    return res

def main():
    out_dir = Path("results/v2_audit2")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1 & 2: Building Contact Sheets ===")
    with open("data/sim_unseen/metadata.json", encoding="utf-8") as f:
        unseen_meta = json.load(f)

    u_w_items = load_labels("data/sim_unseen/L0/word/test")
    u_l_items = load_labels("data/sim_unseen/L0/line/test")

    # Group by font
    sans_w = [(p, lbl) for (p, lbl), fnt in zip(u_w_items, unseen_meta["word_fonts"]) if fnt == "SansSerifCollection.ttf"]
    sans_l = [(p, lbl) for (p, lbl), fnt in zip(u_l_items, unseen_meta["line_fonts"]) if fnt == "SansSerifCollection.ttf"]

    ink_w = [(p, lbl) for (p, lbl), fnt in zip(u_w_items, unseen_meta["word_fonts"]) if fnt == "Inkfree.ttf"]
    ink_l = [(p, lbl) for (p, lbl), fnt in zip(u_l_items, unseen_meta["line_fonts"]) if fnt == "Inkfree.ttf"]

    moba_w = [(p, lbl) for (p, lbl), fnt in zip(u_w_items, unseen_meta["word_fonts"]) if fnt == "Moba_Fira.ttf"]
    moba_l = [(p, lbl) for (p, lbl), fnt in zip(u_l_items, unseen_meta["line_fonts"]) if fnt == "Moba_Fira.ttf"]

    build_contact_sheet(sans_w, sans_l, out_dir / "sansserif_words.png", "SansSerifCollection.ttf", n_words=30, n_lines=3)
    build_contact_sheet(ink_w, ink_l, out_dir / "inkfree_words.png", "Inkfree.ttf", n_words=20, n_lines=0)
    build_contact_sheet(moba_w, moba_l, out_dir / "mobafira_words.png", "Moba_Fira.ttf", n_words=20, n_lines=0)

    print("\n=== Step 3: Per-Font Tesseract & Model Evaluation ===")
    with open("data/sim_ladder/metadata.json", encoding="utf-8") as f:
        ladder_meta = json.load(f)

    # 1. Sim Ladder per-font evaluation (L0 and L4)
    ladder_results = {}
    for lvl in ["L0", "L4"]:
        w_dir = f"data/sim_ladder/{lvl}/word/test"
        l_dir = f"data/sim_ladder/{lvl}/line/test"
        w_fonts = ladder_meta["word_fonts"]
        l_fonts = ladder_meta["line_fonts"]

        print(f"Evaluating sim_ladder {lvl}...")
        w_v1 = eval_per_font(V1_WORD_CKPT, w_dir, w_fonts, is_word=True, is_tess=False)
        w_v2 = eval_per_font(V2_WORD_CKPT, w_dir, w_fonts, is_word=True, is_tess=False)
        w_tess = eval_per_font(None, w_dir, w_fonts, is_word=True, is_tess=True)

        l_v1 = eval_per_font(V1_LINE_CKPT, l_dir, l_fonts, is_word=False, is_tess=False)
        l_v2 = eval_per_font(V2_LINE_CKPT, l_dir, l_fonts, is_word=False, is_tess=False)
        l_tess = eval_per_font(None, l_dir, l_fonts, is_word=False, is_tess=True)

        ladder_results[lvl] = {
            "w_v1": w_v1, "w_v2": w_v2, "w_tess": w_tess,
            "l_v1": l_v1, "l_v2": l_v2, "l_tess": l_tess,
        }

    # 2. Sim Unseen per-font evaluation (L0 and L4)
    unseen_results = {}
    for lvl in ["L0", "L4"]:
        w_dir = f"data/sim_unseen/{lvl}/word/test"
        l_dir = f"data/sim_unseen/{lvl}/line/test"
        w_fonts = unseen_meta["word_fonts"]
        l_fonts = unseen_meta["line_fonts"]

        print(f"Evaluating sim_unseen {lvl}...")
        w_v1 = eval_per_font(V1_WORD_CKPT, w_dir, w_fonts, is_word=True, is_tess=False)
        w_v2 = eval_per_font(V2_WORD_CKPT, w_dir, w_fonts, is_word=True, is_tess=False)
        w_tess = eval_per_font(None, w_dir, w_fonts, is_word=True, is_tess=True)

        l_v1 = eval_per_font(V1_LINE_CKPT, l_dir, l_fonts, is_word=False, is_tess=False)
        l_v2 = eval_per_font(V2_LINE_CKPT, l_dir, l_fonts, is_word=False, is_tess=False)
        l_tess = eval_per_font(None, l_dir, l_fonts, is_word=False, is_tess=True)

        unseen_results[lvl] = {
            "w_v1": w_v1, "w_v2": w_v2, "w_tess": w_tess,
            "l_v1": l_v1, "l_v2": l_v2, "l_tess": l_tess,
        }

    # Save complete evaluation output as JSON for analysis
    all_data = {
        "ladder": ladder_results,
        "unseen": unseen_results,
    }
    
    # Save a sanitized version without refs/hyps
    sanitized = {}
    for scope in ["ladder", "unseen"]:
        sanitized[scope] = {}
        for lvl in ["L0", "L4"]:
            sanitized[scope][lvl] = {}
            for k in ["w_v1", "w_v2", "w_tess", "l_v1", "l_v2", "l_tess"]:
                sanitized[scope][lvl][k] = {}
                for fnt, metrics in all_data[scope][lvl][k].items():
                    sanitized[scope][lvl][k][fnt] = {
                        "cer": metrics["cer"],
                        "wer": metrics["wer"],
                        "n": metrics["n"],
                    }

    with open(out_dir / "per_font_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    print("\n=== All computations complete ===")

if __name__ == "__main__":
    main()
