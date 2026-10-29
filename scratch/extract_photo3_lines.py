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

img_path = Path('data/real_test3/photo3.jpg')
img = cv2.imread(str(img_path))
h_img, w_img = img.shape[:2]

data = pytesseract.image_to_data(Image.open(img_path), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

# Reconstruct paragraph lines
lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'y0': y0, 'y1': y1, 'x0': x0, 'x1': x1, 'words': len(grp)})

# Filter clear middle body lines
clean_lines = [l for l in lines if l['words'] >= 6 and 1100 <= l['y0'] <= 2700]
print(f"Total candidate body lines: {len(clean_lines)}")
for i, l in enumerate(clean_lines):
    print(f"L{i+1:02d}: (y={l['y0']}:{l['y1']}, x={l['x0']}:{l['x1']}) {l['text']}")
