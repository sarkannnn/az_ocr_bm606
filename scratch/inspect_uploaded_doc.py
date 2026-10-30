import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

img_path = r'C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793351399184.jpg'
img = cv2.imread(img_path)

data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0, 'words': len(grp)})

print(f"Total reconstructed lines: {len(lines)}")
for i, l in enumerate(lines):
    print(f"[{i+1:02d}] (x={l['x0']}:{l['x1']}, y={l['y0']}:{l['y1']}, h={l['h']}, w={l['w']}) {l['text']}")
