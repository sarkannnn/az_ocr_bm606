import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_model_checkpoint, infer_single_crop, align, normalize

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

# Load models
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")
v2_word_m, v2_word_cfg, v2_word_voc, dev = load_model_checkpoint("runs/word_v2_20261027_180300/best.pt")

img = cv2.imread('data/real_test3/photo3.jpg')

# Let's test a couple of well-isolated lines and words
test_samples = [
    # text, y0, y1, x0, x1
    ("Çöldə isə nəvəsi Murad səhərin sərin havasında babasının köhnə velosipedini təmizləyirdi", 1110, 1205, 260, 2395),
    ("unudulmaqda olan xalçaçılıq sənəti hər keçən gün daha da zəifləyir.", 1346, 1434, 255, 1750),
    ("Babası qorxurdu ki, o öləndən sonra bu qədim sənət də onunla birlikdə torpağa", 2517, 2583, 215, 2465),
    ("babasına kömək edəcək və bu sənətin yerə düşməsinə imkan verməyəcək.", 2684, 2746, 215, 2055),
]

print("=== LINE CROPS TEST ===")
for ref, y0, y1, x0, x1 in test_samples:
    crop = img[y0:y1, x0:x1]
    p_v2 = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, crop, dev)
    p_v1 = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, crop, dev)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    p_tess = normalize(" ".join(pytesseract.image_to_string(pil_crop, lang="aze", config="--psm 7").split()))
    
    c_err2 = align(ref, p_v2)[0]
    c_err1 = align(ref, p_v1)[0]
    c_errt = align(ref, p_tess)[0]
    print(f"\nREF:  {ref}")
    print(f"V2:   {p_v2} (CER: {c_err2/len(ref)*100:.1f}%)")
    print(f"TESS: {p_tess} (CER: {c_errt/len(ref)*100:.1f}%)")
    print(f"V1:   {p_v1} (CER: {c_err1/len(ref)*100:.1f}%)")
