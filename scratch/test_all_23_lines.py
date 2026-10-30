import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import os
from pathlib import Path
import json

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

sys.path.insert(0, str(Path.cwd()))
from common import load_model_checkpoint, infer_single_crop, align, normalize

# Load models
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint('runs/line_v2/best.pt')
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint('runs/line/best.pt')

img_path = r'C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793351399184.jpg'
img = cv2.imread(img_path)

# Let's get the exact line boxes
data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0})

print(f"Total lines: {len(lines)}")

# Save all line crops into a test folder to inspect
out_crops = Path('results/photo3_test/crops')
out_crops.mkdir(parents=True, exist_ok=True)

# For each line, let's also compute the ink-extent bounds (vertical) and common/padded horizontal bounds
# With a 4 px margin as in real_fix6:
crop_records = []
for idx, l in enumerate(lines):
    # Padding: vertical 3-4 px
    y0 = max(0, l['y0'] - 4)
    y1 = min(img.shape[0], l['y1'] + 4)
    x0 = max(0, l['x0'] - 6)
    x1 = min(img.shape[1], l['x1'] + 6)
    
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(str(out_crops / f"line_{idx:02d}.png"), crop)
    
    p_v2 = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, crop, dev)
    p_v1 = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, crop, dev)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    p_tess = normalize(' '.join(pytesseract.image_to_string(pil_crop, lang='aze', config='--psm 7').split()))
    
    crop_records.append({
        'idx': idx,
        'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1,
        'h': y1 - y0, 'w': x1 - x0,
        'tess_full': p_tess,
        'v1_full': p_v1,
        'v2_full': p_v2,
    })
    print(f"--- Line {idx:02d} ---")
    print(f"TESS: {p_tess}")
    print(f"V2:   {p_v2}")
    print(f"V1:   {p_v1}")

with open('results/photo3_test/initial_crops_info.json', 'w', encoding='utf-8') as f:
    json.dump(crop_records, f, ensure_ascii=False, indent=2)
