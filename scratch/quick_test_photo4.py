import sys
import os
from pathlib import Path
import cv2
from PIL import Image
import pytesseract
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import load_model_checkpoint, infer_single_crop, align, normalize

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

# Load models
v2_word_m, v2_word_cfg, v2_word_voc, dev = load_model_checkpoint("runs/word_v2_20261027_180300/best.pt")
v1_word_m, v1_word_cfg, v1_word_voc, dev = load_model_checkpoint("runs/word/best.pt")

img_path = Path('data/real_test4/photo4.jpg')
img = cv2.imread(str(img_path))
data = pytesseract.image_to_data(Image.open(img_path), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

# Target words with Azerbaijani special letters
words_to_test = []
seen = set()
for _, r in data.iterrows():
    w = str(r.text).strip()
    if any(c in w for c in 'əıöğüşçƏİÖĞÜŞÇ') and len(w) >= 4 and r.width >= 35 and r.height >= 20 and r.conf > 70:
        if w not in seen:
            seen.add(w)
            words_to_test.append((w, int(r.top), int(r.top + r.height), int(r.left), int(r.left + r.width)))
    if len(words_to_test) >= 20:
        break

print(f"Testing {len(words_to_test)} words from Calibri Document (Photo 4)...")
print(f"{'#':2s} | {'Reference':15s} | {'V2 Word':15s} | {'Tesseract':15s} | {'V1 Word':15s} | V2 CER")
print("-" * 75)
v2_errs, tes_errs, v1_errs = 0, 0, 0
tot_chars = 0
for i, (ref, y0, y1, x0, x1) in enumerate(words_to_test):
    crop = img[max(0, y0-2):min(img.shape[0], y1+2), max(0, x0-2):min(img.shape[1], x1+2)]
    p_v2 = infer_single_crop(v2_word_m, v2_word_cfg, v2_word_voc, crop, dev)
    p_v1 = infer_single_crop(v1_word_m, v1_word_cfg, v1_word_voc, crop, dev)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    p_tess = normalize(" ".join(pytesseract.image_to_string(pil_crop, lang="aze", config="--psm 8").split()))
    
    cer2 = align(ref, p_v2)[0]
    cert = align(ref, p_tess)[0]
    cer1 = align(ref, p_v1)[0]
    
    v2_errs += cer2
    tes_errs += cert
    v1_errs += cer1
    tot_chars += len(ref)

    status = "100% DÜZ" if ref == p_v2 else f"Xəta ({cer2/len(ref)*100:.1f}%)"
    print(f"{i+1:02d} | {ref:15s} | {p_v2:15s} | {p_tess:15s} | {p_v1:15s} | {status}")

print("-" * 75)
print(f"OVERALL WORD CER -> V2: {v2_errs/tot_chars*100:.2f}%, Tesseract: {tes_errs/tot_chars*100:.2f}%, V1: {v1_errs/tot_chars*100:.2f}%")
