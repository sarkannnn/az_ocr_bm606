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

img = cv2.imread('data/real_test3/photo3.jpg')
data = pytesseract.image_to_data(Image.open('data/real_test3/photo3.jpg'), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

# Pick words with special Azerbaijani chars
words_to_test = []
for idx, row in data.iterrows():
    w = str(row.text).strip()
    if any(c in w for c in 'əıöğüşçƏİÖĞÜŞÇ') and len(w) >= 4 and row.width >= 40 and row.height >= 20:
        x0, y0 = int(row.left), int(row.top)
        x1, y1 = int(row.left + row.width), int(row.top + row.height)
        words_to_test.append((w, y0, y1, x0, x1))
    if len(words_to_test) >= 15:
        break

print(f"Testing {len(words_to_test)} words with Azerbaijani diacritics...")
print(f"{'#':2s} | {'Reference':15s} | {'V2 Word':15s} | {'Tesseract':15s} | {'V1 Word':15s} | V2 CER")
print("-" * 75)
for i, (ref, y0, y1, x0, x1) in enumerate(words_to_test):
    # Add a small 2px margin
    crop = img[max(0, y0-2):min(img.shape[0], y1+2), max(0, x0-2):min(img.shape[1], x1+2)]
    p_v2 = infer_single_crop(v2_word_m, v2_word_cfg, v2_word_voc, crop, dev)
    p_v1 = infer_single_crop(v1_word_m, v1_word_cfg, v1_word_voc, crop, dev)
    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    p_tess = normalize(" ".join(pytesseract.image_to_string(pil_crop, lang="aze", config="--psm 8").split()))
    
    cer2 = align(ref, p_v2)[0] / max(len(ref), 1)
    print(f"{i+1:02d} | {ref:15s} | {p_v2:15s} | {p_tess:15s} | {p_v1:15s} | {cer2*100:5.1f}%")
