import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Load crop geometry
with open("results/real_fix6/crop_geometry.json", "r", encoding="utf-8") as f:
    geom = json.load(f)

# Chunk splitting functions
def find_split_points(crop_bgr, n_chunks):
    if n_chunks <= 1:
        return []
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 160).astype(np.float32)
    proj = ink.sum(axis=0)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth_proj = np.convolve(proj, kernel, mode='same')
    
    w = crop_bgr.shape[1]
    nominal_w = w / n_chunks
    search_radius = int(nominal_w * 0.35)
    splits = []
    for k in range(1, n_chunks):
        ideal_x = int(k * nominal_w)
        x0 = max(15, ideal_x - search_radius)
        x1 = min(w - 15, ideal_x + search_radius)
        window = smooth_proj[x0:x1]
        min_val = window.min()
        candidates = np.where(window == min_val)[0] + x0
        best_x = candidates[np.argmin(np.abs(candidates - ideal_x))]
        splits.append(int(best_x))
    return splits

def split_into_chunks(crop_bgr, n_chunks):
    splits = find_split_points(crop_bgr, n_chunks)
    chunks = []
    prev_x = 0
    for sx in splits:
        chunks.append(crop_bgr[:, prev_x:sx])
        prev_x = sx
    chunks.append(crop_bgr[:, prev_x:])
    return chunks

# Test robust trim rule
removed_tall_components = []

def robust_trim_chunk_to_ink(chunk_bgr, chunk_tag=""):
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY)
    h_chunk, w_chunk = gray.shape
    ink = (gray < 160).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    clean_ink = np.zeros_like(ink)
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_left_or_right = (x == 0) or (x + w >= w_chunk)
        is_tall_shadow = touches_left_or_right and (h >= 0.70 * h_chunk)
        is_small_speck = (area < 8)
        
        if is_small_speck or is_tall_shadow:
            if h > 6:
                reason = "shadow_stripe" if is_tall_shadow else "area_lt_8"
                removed_tall_components.append({
                    "chunk": chunk_tag,
                    "component_id": i,
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area),
                    "chunk_w": int(w_chunk), "chunk_h": int(h_chunk),
                    "reason": reason
                })
        else:
            clean_ink[labels == i] = 1
            
    col_counts = clean_ink.sum(axis=0)
    valid_cols = np.where(col_counts > 0)[0]
    if len(valid_cols) == 0:
        return chunk_bgr, 0, w_chunk
    x0 = max(0, int(valid_cols[0]) - 4)
    x1 = min(w_chunk, int(valid_cols[-1]) + 5)
    return chunk_bgr[:, x0:x1], x0, x1

# Load raw images
img1 = cv2.imread("data/real_docs/photo1_times.jpg")
img2_deskewed = cv2.imread("results/real_fix6/photo2_deskewed.jpg")

print("Testing Photo 1 and Photo 2 chunks with robust trim rule...")
for p_key, img, p_tag in [("photo1", img1, "p1"), ("photo2", img2_deskewed, "p2")]:
    for item in geom[p_key]:
        l_idx = item["line_idx"]
        y0, y1 = item["y0"], item["y1"]
        x0, x1 = item["x0"], item["x1"]
        c = img[y0:y1, x0:x1]
        h_c, w_c = c.shape[:2]
        rule_n = max(1, int(np.ceil((w_c * 32) / (h_c * 512))))
        raw_chunks = split_into_chunks(c, rule_n)
        
        for ch_i, raw_ch in enumerate(raw_chunks):
            tag = f"{p_tag}_L{l_idx:02d}_ch{ch_i:02d}"
            trimmed, cut_x0, cut_x1 = robust_trim_chunk_to_ink(raw_ch, tag)
            if ch_i == 0 and l_idx in [0, 3, 6, 7, 12]:
                print(f"[{tag}] raw_w={raw_ch.shape[1]} -> trimmed_w={trimmed.shape[1]} (cut left {cut_x0} px, cut right {raw_ch.shape[1]-cut_x1} px)")

print(f"\nTotal removed components taller than 6 px: {len(removed_tall_components)}")
for r in removed_tall_components:
    print(r)
