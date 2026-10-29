import cv2
import numpy as np
import sys, os
from pathlib import Path
import json

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"
import pytesseract
import torch

from common import make_cfg, preprocess, Vocab, cer_wer, align, normalize
from model import CRNN
from test_real_doc2 import REAL_LINES, IMG_PATH, load_model, infer, run_tess_crop

OUT_DIR = Path("results/real_test2/crops_masked")
OUT_DIR.mkdir(parents=True, exist_ok=True)

img = cv2.imread(IMG_PATH)
h_img, w_img = img.shape[:2]

# Load models
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model("runs/line/best.pt")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model("runs/line_v2/best.pt")

# Fix line 0 coordinates
lines_coords = list(REAL_LINES)
# line 0: ("TƏQDİMAT", 23, 48, 300, 485)
lines_coords[0] = ("TƏQDİMAT", 23, 48, 300, 485)

# Masking logic
# For each line, crop with 5px margin top and bottom (4-6px margin)
margin_y = 5
margin_x = 5

new_crops = []
saved_paths = []

for idx, (gt, y0, y1, x0, x1) in enumerate(lines_coords):
    # Find boundary with previous and next lines
    # previous line bottom:
    prev_y1 = lines_coords[idx-1][2] if idx > 0 else 0
    next_y0 = lines_coords[idx+1][1] if idx < len(lines_coords)-1 else h_img
    
    # Interline cutoffs
    top_cutoff = max(0, y0 - margin_y)
    bot_cutoff = min(h_img, y1 + margin_y)
    
    # If the previous line is close, don't cross the midpoint
    if idx > 0 and prev_y1 > top_cutoff:
        split_top = (prev_y1 + y0) // 2
    else:
        split_top = top_cutoff
        
    if idx < len(lines_coords)-1 and next_y0 < bot_cutoff:
        split_bot = (y1 + next_y0) // 2
    else:
        split_bot = bot_cutoff
        
    crop_y0 = max(0, y0 - margin_y)
    crop_y1 = min(h_img, y1 + margin_y)
    crop_x0 = max(0, x0 - margin_x)
    crop_x1 = min(w_img, x1 + margin_x)
    
    crop = img[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    
    # Coordinates of line inside crop:
    # line is from (y0 - crop_y0) to (y1 - crop_y0)
    in_y0 = y0 - crop_y0
    in_y1 = y1 - crop_y0
    in_split_top = split_top - crop_y0
    in_split_bot = split_bot - crop_y0
    
    # Masking: any pixel above in_split_top is masked to white (255, 255, 255)
    if in_split_top > 0:
        crop[:in_split_top, :] = 255
    # Any pixel below in_split_bot is masked to white
    if in_split_bot < crop.shape[0]:
        crop[in_split_bot:, :] = 255
        
    crop_path = OUT_DIR / f"masked_line_{idx:02d}.png"
    cv2.imwrite(str(crop_path), crop)
    saved_paths.append(str(crop_path))
    new_crops.append((gt, crop))

print(f"Generated {len(new_crops)} masked crops in {OUT_DIR}")
print("First 5 crop paths:")
for p in saved_paths[:5]:
    print(" ", p)

# Now evaluate V1, V2, Tesseract on new crops
refs = []
v1_preds = []
v2_preds = []
tes_preds = []

for gt, crop in new_crops:
    p1 = infer(v1_line_m, v1_line_cfg, v1_line_voc, crop, dev)
    p2 = infer(v2_line_m, v2_line_cfg, v2_line_voc, crop, dev)
    pt = run_tess_crop(crop, is_word=False)
    
    refs.append(normalize(gt))
    v1_preds.append(normalize(p1))
    v2_preds.append(normalize(p2))
    tes_preds.append(normalize(pt))

print("\n--- NEW RESULTS ON 21 MASKED LINE CROPS ---")
for name, hyps in [("V1", v1_preds), ("V2", v2_preds), ("Tesseract", tes_preds)]:
    c, w, _ = cer_wer(refs, hyps)
    tot_c = sum(len(r) for r in refs)
    tot_w = sum(len(r.split()) for r in refs)
    err_c = sum(align(r, h)[0] for r, h in zip(refs, hyps))
    err_w = sum(align(r.split(), h.split())[0] for r, h in zip(refs, hyps))
    exact = sum(1 for r, h in zip(refs, hyps) if r == h)
    exact_s = sum(1 for r, h in zip(refs, hyps) if r.strip() == h.strip())
    print(f"{name:10s} CER={c*100:6.2f}% ({err_c:3d}/{tot_c:3d}), WER={w*100:6.2f}% ({err_w:3d}/{tot_w:3d}), exact={exact}/21 (exact_strip={exact_s}/21)")

print("\nLine by line comparison for V2 and Tesseract:")
for i, (r, p2, pt) in enumerate(zip(refs, v2_preds, tes_preds)):
    e2 = align(r, p2)[0]
    et = align(r, pt)[0]
    if e2 > 0 or et > 0:
        print(f"L{i:02d} GT:   '{r}'")
        if e2 > 0:
            print(f"    V2:   '{p2}' (CER: {e2}/{len(r)})")
        if et > 0:
            print(f"    Tess: '{pt}' (CER: {et}/{len(r)})")
