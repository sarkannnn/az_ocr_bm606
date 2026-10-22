import random
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from common import load_labels, SPECIAL_ALL

# Target characters for cedilla and descender verification
TARGET_CHARS = set("şçğəqypŞÇĞƏQYP")

random.seed(42)

splits = [
    ("synth_word/train", "data/synth_word/train"),
    ("synth_word/val",   "data/synth_word/val"),
    ("synth_word/test",  "data/synth_word/test"),
    ("synth_line/train", "data/synth_line/train"),
    ("synth_line/val",   "data/synth_line/val"),
    ("synth_line/test",  "data/synth_line/test"),
]

def check_edge_ink(img_path, threshold=128):
    """Returns (top_touch, bottom_touch) if text ink (< threshold) reaches row 0 or row -1."""
    with Image.open(img_path) as im:
        arr = np.array(im.convert("L"))
    top_touch = bool(np.any(arr[0, :] < threshold))
    bottom_touch = bool(np.any(arr[-1, :] < threshold))
    return top_touch, bottom_touch

print("=" * 75)
print("AUTOMATIC BOUNDARY VERIFICATION (500 RANDOM IMAGES PER SET)")
print("Checking for text ink touching row 0 (top) or row -1 (bottom)")
print("=" * 75)

summary_results = []

for name, p in splits:
    split_path = Path(p)
    if not split_path.exists():
        print(f"Skipping {name}: path {p} not found")
        continue
    items = load_labels(p)
    n_sample = min(500, len(items))
    sample_indices = random.sample(range(len(items)), n_sample)
    
    top_flags = 0
    bottom_flags = 0
    either_flags = 0
    
    for idx in sample_indices:
        img_path, _ = items[idx]
        top, bottom = check_edge_ink(img_path)
        if top:
            top_flags += 1
        if bottom:
            bottom_flags += 1
        if top or bottom:
            either_flags += 1
            
    top_pct = (top_flags / n_sample) * 100
    bottom_pct = (bottom_flags / n_sample) * 100
    either_pct = (either_flags / n_sample) * 100
    
    summary_results.append({
        "split": name,
        "n_sample": n_sample,
        "top_flags": top_flags,
        "top_pct": top_pct,
        "bottom_flags": bottom_flags,
        "bottom_pct": bottom_pct,
        "either_flags": either_flags,
        "either_pct": either_pct,
    })
    
    print(f"{name:18s} | N={n_sample:3d} | Top ink: {top_flags:2d} ({top_pct:4.1f}%) | "
          f"Bottom ink: {bottom_flags:2d} ({bottom_pct:4.1f}%) | Either: {either_flags:2d} ({either_pct:4.1f}%)")

print("=" * 75)
print("\n" + "=" * 75)
print("CONTACT SHEET GENERATION: WORDS & LINES WITH [ş, ç, ğ, ə, q, y, p]")
print("=" * 75)

Path("results").mkdir(parents=True, exist_ok=True)

# Generate contact sheet with samples rich in targeted letters
active_splits = [(name, p) for name, p in splits if Path(p).exists()]
ncols = 5
nrows = len(active_splits)
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.2 * nrows))

for row_idx, (name, p) in enumerate(active_splits):
    items = load_labels(p)
    # Filter items that contain any target chars
    target_items = [it for it in items if any(c in TARGET_CHARS for c in it[1])]
    if not target_items:
        target_items = items
    # Sort by number of target characters descending, then sample
    target_items.sort(key=lambda it: sum(1 for c in it[1] if c in TARGET_CHARS), reverse=True)
    # Pick from top 100 rich samples
    top_pool = target_items[:min(100, len(target_items))]
    samples = random.sample(top_pool, min(ncols, len(top_pool)))
    
    for col_idx, (img_path, label) in enumerate(samples):
        ax = axes[row_idx, col_idx] if nrows > 1 else axes[col_idx]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        # Highlight present target chars
        found_targets = "".join(sorted(set(c for c in label if c in TARGET_CHARS)))
        ax.set_title(f"{label}\n[targets: {found_targets}]", fontsize=9, pad=4)
        
    left_ax = axes[row_idx, 0] if nrows > 1 else axes[0]
    left_ax.text(-0.2, 0.5, name, transform=left_ax.transAxes,
                 fontsize=11, va="center", ha="right", weight="bold")

plt.suptitle("Azərbaycan OCR Dataset Quality Check — Descenders & Cedillas Verification\n"
             "Testing Letters: ş, ç, ğ, ə, q, y, p (Full visibility without edge clipping)",
             fontsize=13, y=0.99)
plt.tight_layout()
out_path = Path("results/data_check.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Contact sheet saved to {out_path} ({out_path.stat().st_size} bytes)")
print("=" * 75)
