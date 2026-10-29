import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
from test_real_doc2 import REAL_LINES, IMG_PATH

img = cv2.imread(IMG_PATH)
h_img, w_img = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Total lines: {len(REAL_LINES)}")

# 1. Line check
# For each line, check:
# a) The exact bounding box (y0..y1, x0..x1)
# b) The actual crop extracted in test_real_doc2: max(0, y0-2)..min(h_img, y1+2), max(0, x0-3)..min(w_img, x1+3)

print("\n--- LINE CROPS VERTICAL CLIPPING CHECK ---")
top_clipped_raw = 0
bot_clipped_raw = 0
top_clipped_act = 0
bot_clipped_act = 0

for idx, (gt, y0, y1, x0, x1) in enumerate(REAL_LINES):
    # Let's inspect margin around line
    ym0 = max(0, y0 - 10)
    ym1 = min(h_img, y1 + 10)
    xm0 = max(0, x0 - 5)
    xm1 = min(w_img, x1 + 5)
    
    sub = gray[ym0:ym1, xm0:xm1]
    _, bin_inv = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Coordinates inside sub:
    # Raw box:
    ry0 = y0 - ym0
    ry1 = y1 - ym0
    
    # Actual crop box (used in test_real_doc2):
    ay0 = max(0, y0 - 2) - ym0
    ay1 = min(h_img, y1 + 2) - ym0
    
    # Connected ink across boundary:
    # A stroke crosses top if there is ink at row y and directly above at row y-1 (same x column)
    # To be robust against single noise pixels, stroke width >= 2 or count > 5 px
    top_cross_raw = np.sum((bin_inv[ry0, :] > 0) & (bin_inv[ry0 - 1, :] > 0)) if ry0 > 0 else 0
    bot_cross_raw = np.sum((bin_inv[ry1 - 1, :] > 0) & (bin_inv[ry1, :] > 0)) if ry1 < sub.shape[0] else 0
    
    top_cross_act = np.sum((bin_inv[ay0, :] > 0) & (bin_inv[ay0 - 1, :] > 0)) if ay0 > 0 else 0
    bot_cross_act = np.sum((bin_inv[ay1 - 1, :] > 0) & (bin_inv[ay1, :] > 0)) if ay1 < sub.shape[0] else 0
    
    # Also check total ink outside top/bottom within 2px
    top_leak_2px = np.sum(bin_inv[max(0, ay0-2):ay0, :]) / 255.0
    bot_leak_2px = np.sum(bin_inv[ay1:min(sub.shape[0], ay1+2), :]) / 255.0
    
    print(f"L{idx:02d} [{y0}:{y1}]: raw_top_cross={top_cross_raw:2d}, raw_bot_cross={bot_cross_raw:2d} | act_top_cross={top_cross_act:2d}, act_bot_cross={bot_cross_act:2d} | gt='{gt[:28]}'")

