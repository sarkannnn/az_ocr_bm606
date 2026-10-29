import sys
import os
from pathlib import Path
from PIL import Image
import pytesseract
import pandas as pd
import cv2

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

img_path = Path('data/real_test3/photo3.jpg')
im = Image.open(img_path)
w_img, h_img = im.size
print(f"Image size: {w_img}x{h_img}")

data = pytesseract.image_to_data(im, lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0, 'words': len(grp)})

# Filter lines that have at least 3 words and reasonable width
body_lines = [l for l in lines if l['words'] >= 4 and l['w'] > 500]
print(f"Total reconstructed body lines: {len(body_lines)}")
for i, l in enumerate(body_lines[:25]):
    print(f"[{i+1:02d}] (y={l['y0']}:{l['y1']}, x={l['x0']}:{l['x1']}) {l['text']}")
