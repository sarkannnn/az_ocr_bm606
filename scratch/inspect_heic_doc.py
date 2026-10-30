import cv2
import os
import sys
from PIL import Image
import pytesseract
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

img_p = Path('data/real_test5/photo5.jpg')
data = pytesseract.image_to_data(Image.open(img_p), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0, 'words': len(grp)})

print(f"Total reconstructed lines: {len(lines)}")
for i, l in enumerate(lines):
    print(f"[{i+1:02d}] (y={l['y0']}:{l['y1']}, x={l['x0']}:{l['x1']}, w={l['w']}, h={l['h']}) {l['text']}")
