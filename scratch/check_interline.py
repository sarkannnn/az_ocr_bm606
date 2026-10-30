import sys
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
os.environ['TESSDATA_PREFIX'] = r'D:\laragon\www\az_ocr_bm606\tessdata'

img1 = cv2.imread('data/real_docs/photo1_times.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Check L00-L01 boundary:
# L00 y: [1106, 1195]
# L01 y: [1194, 1275]
print("=== L00-L01 Boundary Check ===")
c1 = img1[1194:1275, 200:2620]
d1 = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB)), lang='aze', output_type=pytesseract.Output.DATAFRAME)
d1 = d1[d1.text.notnull() & (d1.text.str.strip() != '')]

for _, r in d1.head(10).iterrows():
    act_top = 1194 + int(r['top'])
    act_bot = act_top + int(r['height'])
    print(f"{r['text']:12s} top_in_crop={r['top']:2d}, act_top={act_top:4d}, act_bot={act_bot:4d}, in_L00_crop={(act_top <= 1195)}")

# Check L01-L02 boundary:
# L01 y: [1194, 1275]
# L02 y: [1273, 1348]
print("\n=== L01-L02 Boundary Check ===")
c2 = img1[1273:1348, 200:2620]
d2 = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB)), lang='aze', output_type=pytesseract.Output.DATAFRAME)
d2 = d2[d2.text.notnull() & (d2.text.str.strip() != '')]

for _, r in d2.head(10).iterrows():
    act_top = 1273 + int(r['top'])
    act_bot = act_top + int(r['height'])
    print(f"{r['text']:12s} top_in_crop={r['top']:2d}, act_top={act_top:4d}, act_bot={act_bot:4d}, in_L01_crop={(act_top <= 1275)}")

# Save high resolution zoomed images with +/- 15 px context
# L00-L01: shared y in [1194, 1195], context y in [1179, 1210]
out_dir = Path("results/real_fix7/edge_inspection")
out_dir.mkdir(parents=True, exist_ok=True)

vis_01 = img1[1179:1211, 200:1600].copy()
# Draw L00 bottom at y=1195 (rel y = 16) in red
cv2.line(vis_01, (0, 16), (vis_01.shape[1], 16), (0, 0, 255), 1)
# Draw L01 top at y=1194 (rel y = 15) in green
cv2.line(vis_01, (0, 15), (vis_01.shape[1], 15), (0, 255, 0), 1)
cv2.imwrite(str(out_dir / "p1_boundary_L00_L01.png"), vis_01)

# L01-L02: shared y in [1273, 1275], context y in [1258, 1290]
vis_12 = img1[1258:1291, 200:1600].copy()
# Draw L01 bottom at y=1275 (rel y = 17) in red
cv2.line(vis_12, (0, 17), (vis_12.shape[1], 17), (0, 0, 255), 1)
# Draw L02 top at y=1273 (rel y = 15) in green
cv2.line(vis_12, (0, 15), (vis_12.shape[1], 15), (0, 255, 0), 1)
cv2.imwrite(str(out_dir / "p1_boundary_L01_L02.png"), vis_12)

print("\nInterline images saved successfully.")
