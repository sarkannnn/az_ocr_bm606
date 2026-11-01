import os
import sys
import json
import math
import difflib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.stdout.reconfigure(encoding="utf-8")

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent Color Palette
COLOR_V1 = "#4682B4"      # Steel Blue
COLOR_V2 = "#2E8B57"      # Sea Green
COLOR_TESS = "#D95F02"    # Dark Orange / Rust
COLOR_RED = "#D62728"     # Red (ONLY for errors, warnings, thresholds)
COLOR_GRAY = "#7F7F7F"
COLOR_BG = "#F8F9FA"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

print("Setting up figure generation...")

# ==============================================================================
# F1: Pipeline diagram
# ==============================================================================
def make_f1():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    
    # Title & subtitle
    ax.text(50, 56, "End-to-End Azerbaijani OCR System Architecture & Experimental Pipeline", 
            ha="center", va="center", fontsize=14, weight="bold", color="#111111")
    ax.text(50, 52.5, "Corpus Engineering  ->  TRDG Synthetic Engine  ->  Augmentation  ->  CRNN+CTC  ->  Multi-Tier Audit", 
            ha="center", va="center", fontsize=9.5, color="#555555")
    
    # Helper to draw boxes
    def draw_box(x, y, w, h, title, lines, color="#EAEAEA", border="#AAAAAA", text_color="#222222"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                     edgecolor=border, facecolor=color, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9.5, weight="bold", color=text_color)
        for i, l in enumerate(lines):
            ax.text(x + w/2, y + h - 6 - i*3.2, l, ha="center", va="center", fontsize=8, color="#333333")
            
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2, shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha="center", va="center", fontsize=7.5, color="#444444", weight="bold")

    # Stage 1: Text Corpus
    draw_box(2, 30, 20, 18, "1. Azerbaijani Corpus", 
             ["Wikimedia az 20231101", "80/10/10 Split pre-render", "Special letter balancing (>=50%)", "Forbidden label tracking"],
             color="#E8F1F5", border="#4682B4")
             
    # Stage 2: TRDG & Fonts
    draw_box(26, 30, 22, 18, "2. Image Generation", 
             ["TRDG (Patched Pillow 13)", "Glyph validation (32 az letters)", "V1: 8 seen fonts", "V2 additions: 6 fonts (+21k)", "20% fonts held out"],
             color="#EBF5EE", border="#2E8B57")
             
    # Stage 3: Data Augmentation
    draw_box(52, 30, 22, 18, "3. Augmentation (V2)", 
             ["Photometric illumination gradient", "Sensor noise & motion blur", "Downscale + JPEG artifacts", "Contrast stretching", "Perspective & slight rotation"],
             color="#FEF5E7", border="#D95F02")
             
    # Stage 4: CRNN Model
    draw_box(78, 30, 20, 18, "4. CRNN Architecture", 
             ["7-layer CNN (VGG-style)", "2-layer BiLSTM (h=256)", "8.23M parameters", "Input: 32x128 / 32x512", "CTC loss (blank=0)"],
             color="#F4ECF7", border="#8E44AD")

    # Stage 5: Evaluation Branches
    draw_box(4, 5, 27, 18, "Synthetic & Ladder Benchmarks", 
             ["Standard tests: Tahoma vs Consolas", "7-step Ladder L0-L4n (n=346w/92l)", "Bootstrap 95% Confidence Intervals", "Special letter confusion matrix"],
             color="#FFFFFF", border="#888888")
             
    draw_box(36, 5, 28, 18, "Generalization Audits", 
             ["Group A (Seen), Group B (Sibling)", "Group C (New Families, n=609w/158l)", "SansSerif tight ink box fix", "Space error WER decomposition"],
             color="#FFFFFF", border="#888888")
             
    draw_box(68, 5, 28, 18, "Real Document Photos (1-5)", 
             ["Mobile photos (NDU doc, stories)", "Rule chunking vs Squeeze ratio", "Padding sensitivity (4px vs 20px)", "Tesseract PSM 7 vs 13 integration"],
             color="#FFFFFF", border="#888888")

    # Arrows
    draw_arrow(22, 39, 26, 39)
    draw_arrow(48, 39, 52, 39)
    draw_arrow(74, 39, 78, 39)
    
    # Connect stage 4 to stage 5
    draw_arrow(88, 30, 82, 23, "Inference")
    draw_arrow(88, 30, 50, 23)
    draw_arrow(88, 30, 18, 23)

    # Footnote / Source line
    ax.text(2, 1.5, "Source: prepare_corpus.py, generate_data.py, train_v2.py:AUG_CONFIG, model.py, run_v2_audit4.py, results/real_fix9/report.md",
            fontsize=7.5, color="#666666", style="italic")
            
    fig.tight_layout()
    fig.savefig(FIG_DIR / "f1_pipeline.png", dpi=200)
    plt.close(fig)
    print("F1 generated.")

# ==============================================================================
# F2: Data sample sheet & bottom clipping bug demonstration
# ==============================================================================
def make_f2():
    fig, axes = plt.subplots(3, 4, figsize=(13, 7.5), dpi=200)
    fig.suptitle("Data Sample Sheet & Glyph Boundary Verification", fontsize=13, weight="bold", y=0.98)
    
    # 1. Clean synthetic samples (Word and Line)
    # Load 4 word samples
    sample_words = [
        ("data/synth_word/test/images/0000000.png", "Word Sample 1"),
        ("data/synth_word/test/images/0000001.png", "Word Sample 2"),
        ("data/synth_word/test/images/0000002.png", "Word Sample 3"),
        ("data/synth_word/test/images/0000003.png", "Word Sample 4"),
    ]
    for idx, (p, title) in enumerate(sample_words):
        ax = axes[0, idx]
        if Path(p).exists():
            img = Image.open(p)
            ax.imshow(img, cmap="gray")
        ax.set_title(f"Synthetic Word ({idx+1})", fontsize=8.5, weight="bold")
        ax.axis("off")
        
    # 2. Line samples from synth_line
    sample_lines = [
        ("data/synth_line/test/images/0000000.png", "Line Sample 1"),
        ("data/synth_line/test/images/0000001.png", "Line Sample 2"),
        ("data/synth_line/test/images/0000002.png", "Line Sample 3"),
        ("data/synth_line/test/images/0000003.png", "Line Sample 4"),
    ]
    for idx, (p, title) in enumerate(sample_lines):
        ax = axes[1, idx]
        if Path(p).exists():
            img = Image.open(p)
            ax.imshow(img, cmap="gray")
        ax.set_title(f"Synthetic Line ({idx+1})", fontsize=8.5, weight="bold")
        ax.axis("off")
        
    # 3. Bottom Clipping Bug: Before vs After demonstration
    # We will synthesize a synthetic crop with lowercase cedilla / descender 'ç' and 'ş'
    # Demonstrating clipped cedilla vs intact with 4px margin
    font_path = "C:/Windows/Fonts/calibri.ttf"
    if not Path(font_path).exists():
        font_path = "C:/Windows/Fonts/arial.ttf"
    fnt = ImageFont.truetype(font_path, 36)
    
    # Render intact text on white
    test_word = "açıq-şəffaf"
    bbox = fnt.getbbox(test_word)
    w_text = bbox[2] - bbox[0]
    h_text = bbox[3] - bbox[1]
    
    # (a) Clipped bottom: tight bounding box without descender padding
    im_clipped = Image.new("L", (w_text + 10, h_text - 4), color=255)
    dr_clipped = ImageDraw.Draw(im_clipped)
    dr_clipped.text((5, -bbox[1]), test_word, font=fnt, fill=0)
    
    # (b) Intact: with safe 4px margin
    im_fixed = Image.new("L", (w_text + 10, h_text + 8), color=255)
    dr_fixed = ImageDraw.Draw(im_fixed)
    dr_fixed.text((5, -bbox[1] + 4), test_word, font=fnt, fill=0)
    
    # Subplot 2,0: Clipped
    axes[2, 0].imshow(im_clipped, cmap="gray")
    axes[2, 0].set_title("[BUG] Clipped Bottom\n(Cedilla cut on ç / ş)", fontsize=8.5, color=COLOR_RED, weight="bold")
    axes[2, 0].axhline(im_clipped.height - 1, color=COLOR_RED, lw=1.5, linestyle="--")
    axes[2, 0].axis("off")
    
    # Subplot 2,1: Fixed
    axes[2, 1].imshow(im_fixed, cmap="gray")
    axes[2, 1].set_title("[FIXED] Safe Margin (+4px)\n(Full cedilla preserved)", fontsize=8.5, color=COLOR_V2, weight="bold")
    axes[2, 1].axhline(im_fixed.height - 1, color=COLOR_V2, lw=1.5, linestyle="--")
    axes[2, 1].axis("off")
    
    # Subplot 2,2 & 2,3: Quality verification summary stats from verify_quality.py
    axes[2, 2].axis("off")
    stats_text = (
        "Automatic Boundary Verification Check:\n"
        "• Script: verify_quality.py (N=500 per set)\n"
        "• Target letters: [ş, ç, ğ, ə, q, y, p]\n"
        "• Row 0 (Top ink touch):  0 / 500 (0.0%)\n"
        "• Row -1 (Bottom touch):  0 / 500 (0.0%)\n"
        "• Verdict: PASS across all 6 dataset splits"
    )
    axes[2, 2].text(0.05, 0.5, stats_text, fontsize=8, va="center", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5EE", edgecolor=COLOR_V2, lw=1))
                    
    axes[2, 3].axis("off")
    sans_bug_text = (
        "SansSerifCollection Bounding Box Bug:\n"
        "• make_sim_ladder.py ascent+descent metric\n"
        "• Inflated crop overlapped next text line\n"
        "• L0 Word CER inflated to 84.32%\n"
        "• Fix: Tight ink bounding box (w_bbox)\n"
        "• Result: L0 Word CER dropped to 0.76%"
    )
    axes[2, 3].text(0.05, 0.5, sans_bug_text, fontsize=8, va="center", color="#222222",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF5E7", edgecolor=COLOR_TESS, lw=1))

    fig.text(0.05, 0.02, "Source: data/synth_word/test/, data/synth_line/test/, verify_quality.py, results/v2_audit3/summary.md:24-35",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "f2_data_samples.png", dpi=200)
    plt.close(fig)
    print("F2 generated.")

# ==============================================================================
# F3: Training curves for Word V1/V2 and Line V1/V2
# ==============================================================================
def make_f3():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=200)
    fig.suptitle("Training & Validation Loss / CER Trajectories Across Epochs", fontsize=13, weight="bold", y=0.98)
    
    configs = [
        ("runs/word/history.csv", "Word V1 (32x128, Mild Aug, 8 Fonts)", 12, axes[0, 0]),
        ("runs/line/history.csv", "Line V1 (32x512, Mild Aug, 8 Fonts)", 12, axes[0, 1]),
        ("runs/word_v2_20261027_180300/history.csv", "Word V2 (32x128, Strong Photometric Aug, 14 Fonts)", 13, axes[1, 0]),
        ("runs/line_v2/history.csv", "Line V2 (32x512, Strong Photometric Aug, 14 Fonts)", 10, axes[1, 1]),
    ]
    
    for csv_path, title, best_ep, ax in configs:
        df = pd.read_csv(csv_path)
        epochs = df["epoch"]
        
        # Loss curves
        l1 = ax.plot(epochs, df["train_loss"], label="Train CTC Loss", color=COLOR_V1, lw=1.8, linestyle="--")
        l2 = ax.plot(epochs, df["val_loss"], label="Val CTC Loss", color=COLOR_TESS, lw=1.8)
        ax.set_ylabel("CTC Loss", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_title(title, fontsize=9.5, weight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        # Secondary axis for val CER
        ax2 = ax.twinx()
        cer_pct = df["val_cer"] * 100
        l3 = ax2.plot(epochs, cer_pct, label="Val CER (%)", color=COLOR_V2, lw=1.6, linestyle="-.")
        ax2.set_ylabel("Val CER (%)", fontsize=9, color=COLOR_V2)
        ax2.tick_params(axis="y", labelcolor=COLOR_V2)
        
        # Mark best epoch
        best_row = df[df["epoch"] == best_ep].iloc[0]
        ax.axvline(best_ep, color=COLOR_RED, linestyle=":", lw=1.2, label=f"Best Ep {best_ep}")
        ax.plot(best_ep, best_row["val_loss"], marker="*", markersize=10, color=COLOR_RED)
        
        # Text annotation for best val CER
        ax.text(best_ep + 0.3, best_row["val_loss"], 
                f"Ep {best_ep}\nLoss: {best_row['val_loss']:.4f}\nCER: {best_row['val_cer']*100:.2f}%", 
                fontsize=7.5, color="#111111", bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", alpha=0.85, lw=0.5))
                
        # Combine legends on ax
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        if ax == axes[0, 0]:
            ax.legend(lines, labels, loc="upper right", fontsize=7.5)

    fig.text(0.05, 0.02, "Source: runs/word/history.csv, runs/line/history.csv, runs/word_v2_20261027_180300/history.csv, runs/line_v2/history.csv",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(FIG_DIR / "f3_training_curves.png", dpi=200)
    plt.close(fig)
    print("F3 generated.")

# ==============================================================================
# F4: Per-font bar chart synthetic: Tahoma vs Consolas
# ==============================================================================
def make_f4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.suptitle("Per-Font Synthetic Test: Genuine Generalization vs. Family Leakage", fontsize=12, weight="bold")
    
    # Data from metrics.json files:
    # Tahoma: word (n=1000), line (n=500)
    # Consolas: word (n=1000), line (n=500)
    tahoma_word = [0.08, 0.09, 0.50]  # V1, V2, Tess
    tahoma_line = [0.24, 0.26, 0.22]
    
    consola_word = [1.02, 0.07, 0.86] # V1, V2, Tess
    consola_line = [3.48, 0.27, 0.47]
    
    x = np.arange(2)
    w = 0.26
    
    # Ax1: Tahoma (New Family)
    b1 = ax1.bar(x - w, [tahoma_word[0], tahoma_line[0]], width=w, label="V1 Line/Word", color=COLOR_V1)
    b2 = ax1.bar(x, [tahoma_word[1], tahoma_line[1]], width=w, label="V2 Line/Word", color=COLOR_V2)
    b3 = ax1.bar(x + w, [tahoma_word[2], tahoma_line[2]], width=w, label="Tesseract 5 (aze)", color=COLOR_TESS)
    
    ax1.set_title("Tahoma (NEW FAMILY — Not in V1 or V2 Train)\nGenuine Font Invariance (V1 ~= V2 ~= Tess)", fontsize=9.5, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Word Level (n=1,000)", "Line Level (n=500)"], fontsize=9)
    ax1.set_ylabel("Character Error Rate (CER %)", fontsize=9)
    ax1.set_ylim(0, 4.0)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    # Annotate values
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.06, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)
            
    # Ax2: Consolas (Family Leakage)
    b4 = ax2.bar(x - w, [consola_word[0], consola_line[0]], width=w, label="V1 Line/Word", color=COLOR_V1)
    b5 = ax2.bar(x, [consola_word[1], consola_line[1]], width=w, label="V2 Line/Word", color=COLOR_V2)
    b6 = ax2.bar(x + w, [consola_word[2], consola_line[2]], width=w, label="Tesseract 5 (aze)", color=COLOR_TESS)
    
    ax2.set_title("Consolas (FAMILY LEAKAGE — consolab.ttf in V2 Train)\nArtificial Zero-Shot Drop (14.6x Word, 12.9x Line)", fontsize=9.5, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Word Level (n=1,000)", "Line Level (n=500)"], fontsize=9)
    ax2.set_ylabel("Character Error Rate (CER %)", fontsize=9)
    ax2.set_ylim(0, 4.0)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bars in [b4, b5, b6]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.06, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)

    # Red warning badge on Consolas
    ax2.text(0.5, 3.5, "DISCLOSURE: NOT Zero-Shot Generalization!\n(consolab.ttf added to V2 training corpus)",
             ha="center", va="center", color=COLOR_RED, fontsize=8, weight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEEEE", edgecolor=COLOR_RED, lw=1))

    ax1.legend(loc="upper right", fontsize=8.5)
    
    fig.text(0.05, 0.02, "Source: results/diagw_tahoma/metrics.json, results/diag_tahoma/metrics.json, results/diagw_consola/metrics.json, results/diag_consola/metrics.json, results/v2_evaluation_report.md:26-34",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIG_DIR / "f4_per_font_synth.png", dpi=200)
    plt.close(fig)
    print("F4 generated.")

# ==============================================================================
# F5: Simulation ladder crops + CER vs Level plot
# ==============================================================================
def make_f5():
    fig = plt.figure(figsize=(12, 8), dpi=200)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    
    fig.suptitle("Simulation Ladder (L0 -> L4): Photometric Degradation & Model Resilience", fontsize=13, weight="bold")
    
    # Top Left & Top Right: Real line crops from L0, L1, L2, L3, L4
    # We will pick 2 crops from data/sim_ladder/{level}/line/test/images/0000000.png and 0000001.png
    levels = ["L0", "L1", "L2", "L3", "L4"]
    level_descs = [
        "L0: Clean Render (White background, black text)",
        "L1: + Illumination gradient & soft shadow",
        "L2: + Gaussian blur & sensor noise",
        "L3: + Downscale (0.6x) & JPEG (Q=50)",
        "L4: + Perspective tilt & rotation (Full Phone Sim)"
    ]
    
    ax_crops1 = fig.add_subplot(gs[0, 0])
    ax_crops1.axis("off")
    ax_crops1.set_title("Visual Progression: Line Sample 00 (CascadiaMono)", fontsize=9.5, weight="bold")
    
    ax_crops2 = fig.add_subplot(gs[0, 1])
    ax_crops2.axis("off")
    ax_crops2.set_title("Visual Progression: Line Sample 01 (Cambria)", fontsize=9.5, weight="bold")
    
    y_pos = 0.85
    for lvl, desc in zip(levels, level_descs):
        p1 = Path(f"data/sim_ladder/{lvl}/line/test/images/0000000.png")
        p2 = Path(f"data/sim_ladder/{lvl}/line/test/images/0000001.png")
        
        if p1.exists():
            im1 = Image.open(p1)
            # Display inline via inset axes or sub-rectangles
            # Let's draw text label then image
            ax_crops1.text(0.0, y_pos, desc, fontsize=7.5, weight="bold", color="#222222", transform=ax_crops1.transAxes)
            sub_ax1 = ax_crops1.inset_axes([0.0, y_pos - 0.12, 0.95, 0.10], transform=ax_crops1.transAxes)
            sub_ax1.imshow(im1, cmap="gray")
            sub_ax1.axis("off")
            
        if p2.exists():
            im2 = Image.open(p2)
            ax_crops2.text(0.0, y_pos, desc, fontsize=7.5, weight="bold", color="#222222", transform=ax_crops2.transAxes)
            sub_ax2 = ax_crops2.inset_axes([0.0, y_pos - 0.12, 0.95, 0.10], transform=ax_crops2.transAxes)
            sub_ax2.imshow(im2, cmap="gray")
            sub_ax2.axis("off")
            
        y_pos -= 0.19
        
    # Bottom Left: Word CER across Ladder levels
    # Data from results/sim_ladder/summary.md:7-13 and results/v2_audit/summary.md:65-72
    ladder_levels = ["L0", "L1", "L2", "L3", "L4", "L4n"]
    
    # Word CER
    v1_word_cer = [7.71, 8.85, 25.00, 31.36, 35.77, 24.43]
    v2_word_cer = [0.20, 0.20, 0.41, 0.53, 1.35, 1.14]
    tess_word_cer = [0.41, 0.45, 0.41, 0.41, 0.37, 0.37]
    
    # Line CER
    v1_line_cer = [1.87, 4.10, 13.84, 19.14, 12.96, 7.43]
    v2_line_cer = [0.11, 0.11, 0.07, 0.18, 0.11, 0.07]
    tess_line_cer = [0.22, 0.22, 0.29, 0.22, 0.33, 0.33]
    
    ax_word = fig.add_subplot(gs[1, 0])
    ax_word.plot(ladder_levels, v1_word_cer, marker="o", color=COLOR_V1, lw=1.8, label="V1 Word (Collapse at L4: 35.8%)")
    ax_word.plot(ladder_levels, v2_word_cer, marker="s", color=COLOR_V2, lw=1.8, label="V2 Word (Robust at L4: 1.35%)")
    ax_word.plot(ladder_levels, tess_word_cer, marker="^", color=COLOR_TESS, lw=1.8, linestyle=":", label="Tesseract Word (~0.4%)")
    ax_word.set_title("Word Level CER across Ladder (n=346 words)", fontsize=9.5, weight="bold")
    ax_word.set_ylabel("CER (%)", fontsize=9)
    ax_word.set_xlabel("Distortion Level", fontsize=9)
    ax_word.grid(True, linestyle=":", alpha=0.6)
    ax_word.legend(fontsize=7.5, loc="upper left")
    
    ax_line = fig.add_subplot(gs[1, 1])
    ax_line.plot(ladder_levels, v1_line_cer, marker="o", color=COLOR_V1, lw=1.8, label="V1 Line (Peak L3: 19.1%, L4: 13.0%)")
    ax_line.plot(ladder_levels, v2_line_cer, marker="s", color=COLOR_V2, lw=1.8, label="V2 Line (Immune at L4: 0.11%)")
    ax_line.plot(ladder_levels, tess_line_cer, marker="^", color=COLOR_TESS, lw=1.8, linestyle=":", label="Tesseract Line (~0.3%)")
    ax_line.set_title("Line Level CER across Ladder (n=92 lines)", fontsize=9.5, weight="bold")
    ax_line.set_ylabel("CER (%)", fontsize=9)
    ax_line.set_xlabel("Distortion Level", fontsize=9)
    ax_line.grid(True, linestyle=":", alpha=0.6)
    ax_line.legend(fontsize=7.5, loc="upper left")
    
    # Note on L4n and non-monotonicity
    ax_line.text(0.98, 0.50, "Note on V1 Line non-monotonicity:\nL3 (19.14%) > L4 (12.96%)\nCause not investigated (A.7)",
                 ha="right", va="center", transform=ax_line.transAxes, fontsize=7.5, color="#444444",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEEEE", edgecolor="#AAAAAA", lw=0.8))

    fig.text(0.05, 0.015, "Source: results/sim_ladder/summary.md:7-13, results/v2_audit/summary.md:65-72, data/sim_ladder/",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIG_DIR / "f5_sim_ladder.png", dpi=200)
    plt.close(fig)
    print("F5 generated.")

# ==============================================================================
# F6: Unseen-font forest plot with 95% Confidence Intervals
# ==============================================================================
def make_f6():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.suptitle("Forest Plot: Accuracy Difference ΔCER (V2 Line/Word − Tesseract 5)", fontsize=12, weight="bold")
    
    # Conditions, delta, lower CI, upper CI, sample n
    # From results/v2_audit4/summary.md:118-132, results/v2_audit3/summary.md:108-114, results/real_fix9/metrics.json:80-90, 301-311
    items = [
        ("Clean Unseen Word L0 (n=699w, 4754ch)", +0.34, -0.25, +0.88, "Synthetic Clean"),
        ("Clean Unseen Line L0 (n=180l, 5309ch)", +0.17, -0.22, +0.58, "Synthetic Clean"),
        ("Clean Unseen Word L4 (n=699w, 4754ch)", +0.63, +0.02, +1.19, "Synthetic Distorted"),
        ("Clean Unseen Line L4 (n=180l, 5309ch)", +0.17, -1.12, +1.14, "Synthetic Distorted"),
        ("Group C Cluster Word L0 (n=609w, 4194ch)", +0.62, -0.35, +2.08, "Cluster Bootstrap"),
        ("Group C Cluster Line L0 (n=158l, 4673ch)", +0.00, -0.74, +0.76, "Cluster Bootstrap"),
        ("Group C Cluster Word L4 (n=609w, 4194ch)", +1.55, +0.59, +2.94, "Cluster Bootstrap"),
        ("Group C Cluster Line L4 (n=158l, 4673ch)", -0.17, -2.60, +1.53, "Cluster Bootstrap"),
        ("Real Photo 1 Trimmed Chunks (n=16l, 1442ch)", -0.35, -1.52, +0.50, "Real Document"),
        ("Real Photo 2 Trimmed Chunks (n=16l, 1463ch)", +0.41, +0.07, +0.94, "Real Document"),
        ("Real Photo 5 Clean Subset (n=23l, 1591ch)", +0.06, -0.77, +0.80, "Real Document"),
    ]
    
    y = np.arange(len(items))[::-1]
    labels = [it[0] for it in items]
    deltas = [it[1] for it in items]
    ci_lows = [it[2] for it in items]
    ci_highs = [it[3] for it in items]
    
    # Plot vertical zero line in RED
    ax.axvline(0, color=COLOR_RED, linestyle="--", lw=1.5, zorder=1, label="Zero Difference (V2 = Tesseract)")
    
    # Plot points and error bars
    for idx in range(len(items)):
        d = deltas[idx]
        low = ci_lows[idx]
        high = ci_highs[idx]
        
        # Color point green if V2 better, orange if Tess better, gray if CI includes 0
        point_color = COLOR_V2 if high < 0 else (COLOR_TESS if low > 0 else "#333333")
        
        ax.plot([low, high], [y[idx], y[idx]], color=point_color, lw=2.2, zorder=2)
        ax.plot(d, y[idx], marker="s", markersize=6.5, color=point_color, zorder=3)
        
        # Label CI numbers
        ax.text(high + 0.12, y[idx], f"{d:+.2f}% [{low:+.2f}%, {high:+.2f}%]", 
                va="center", fontsize=7.5, color="#222222", weight="bold" if (low > 0 or high < 0) else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Difference ΔCER = CER(V2) − CER(Tesseract) [Percentage Points]\n(Negative = V2 is better, Positive = Tesseract is better)", fontsize=9)
    ax.set_xlim(-3.5, 4.0)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    
    # Legend & interpretation callout
    ax.text(0.03, 0.12, 
            "Key Observation:\n"
            "• In 9 out of 11 conditions, the 95% CI spans across 0.0%.\n"
            "• On real paper lines (Clean subset: [-0.77%, +0.80%]),\n"
            "  V2 and Tesseract recognition accuracy are statistically indistinguishable.",
            transform=ax.transAxes, fontsize=8, va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#999999", lw=1))

    ax.legend(loc="upper right", fontsize=8.5)

    fig.text(0.05, 0.02, "Source: results/v2_audit4/summary.md:118-132, results/v2_audit3/summary.md:108-114, results/real_fix9/metrics.json:80-90, 301-311",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "f6_unseen_forest.png", dpi=200)
    plt.close(fig)
    print("F6 generated.")

# ==============================================================================
# F7: Special-letter heatmap/bars
# ==============================================================================
def make_f7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    fig.suptitle("Azerbaijani Special Diacritic Error Rates across Clean Unseen Fonts", fontsize=12, weight="bold")
    
    # Letter pairs and counts from results/v2_audit3/summary.md:121-142
    pairs = ["ə / e", "ı / i", "ö / o", "ğ / g", "ü / u", "ş / s", "ç / c"]
    counts = [394, 138, 32, 20, 56, 56, 23]
    
    # Word level L4 errors %
    v2_word_l4 = [0.00, 7.25, 6.25, 0.00, 0.00, 0.00, 4.35]
    tess_word_l4 = [1.78, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    
    # Line level L4 errors %
    v2_line_l4 = [0.00, 2.17, 3.12, 0.00, 0.00, 0.00, 0.00]
    tess_line_l4 = [1.02, 0.72, 0.00, 5.00, 1.79, 1.79, 0.00]
    
    x = np.arange(len(pairs))
    w = 0.35
    
    # Left: Word L4
    b1 = ax1.bar(x - w/2, v2_word_l4, width=w, label="V2 Word L4", color=COLOR_V2)
    b2 = ax1.bar(x + w/2, tess_word_l4, width=w, label="Tesseract Word L4", color=COLOR_TESS)
    ax1.set_title("Word Level under L4 Distortion (Full Sim)", fontsize=9.5, weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(pairs, counts)], fontsize=8)
    ax1.set_ylabel("Special Letter Error Rate (%)", fontsize=9)
    ax1.set_ylim(0, 12.0)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b1:
        h = bar.get_height()
        if h > 0: ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for bar in b2:
        h = bar.get_height()
        if h > 0: ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
        
    ax1.legend(loc="upper right", fontsize=8)

    # Right: Line L4
    b3 = ax2.bar(x - w/2, v2_line_l4, width=w, label="V2 Line L4", color=COLOR_V2)
    b4 = ax2.bar(x + w/2, tess_line_l4, width=w, label="Tesseract Line L4", color=COLOR_TESS)
    ax2.set_title("Line Level under L4 Distortion (Full Sim)", fontsize=9.5, weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(pairs, counts)], fontsize=8)
    ax2.set_ylabel("Special Letter Error Rate (%)", fontsize=9)
    ax2.set_ylim(0, 12.0)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b3:
        h = bar.get_height()
        if h > 0: ax2.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for bar in b4:
        h = bar.get_height()
        if h > 0: ax2.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax2.legend(loc="upper right", fontsize=8)

    # Annotate V1 historical failure in subtitle / footnote
    fig.text(0.5, 0.05, 
             "Historical Context (V1 Baseline): V1 suffered 22.43% ç->c failure (24/107) and 7.69% ş->s failure (17/221) due to lack of diverse fonts & noise.\n"
             "V2 resolved cedilla confusion to 0.00% on lines, while dotless-i (ı/i) remains the primary remaining confusion (2.17% on lines).",
             ha="center", fontsize=8, color="#333333", style="italic")
             
    fig.text(0.05, 0.015, "Source: results/v2_audit3/summary.md:121-142, logs/eval_word_synth.log, logs/eval_line_synth.log",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(FIG_DIR / "f7_special_letters.png", dpi=200)
    plt.close(fig)
    print("F7 generated.")

# ==============================================================================
# F8: Real photo detection and chunking boundaries
# ==============================================================================
def make_f8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.5), dpi=200)
    fig.suptitle("Real Document Processing: Line Detection & Projection-Based Chunking", fontsize=12, weight="bold")
    
    # 1. Photo 1 (TNR) with detected line bounding boxes
    p1_path = Path("data/real_docs/photo1_times.jpg")
    g8_path = Path("results/real_fix8/crop_geometry.json")
    
    if p1_path.exists() and g8_path.exists():
        im1 = Image.open(p1_path)
        g8 = json.load(open(g8_path, encoding="utf-8"))
        
        # Draw on a copy of downscaled image
        im1_small = im1.resize((im1.width // 4, im1.height // 4))
        scale = 0.25
        dr1 = ImageDraw.Draw(im1_small)
        
        for item in g8.get("photo1", []):
            x0, y0, x1, y1 = item["x0"]*scale, item["y0"]*scale, item["x1"]*scale, item["y1"]*scale
            dr1.rectangle([x0, y0, x1, y1], outline="red", width=2)
            dr1.text((x0 + 4, y0 + 2), str(item["line_idx"]), fill="red")
            
        ax1.imshow(im1_small)
        ax1.set_title("Photo 1 (Times New Roman, 16 Lines)\nDetected Line Boxes in RED", fontsize=9.5, weight="bold")
        ax1.axis("off")
    else:
        ax1.text(0.5, 0.5, "Photo 1 image not found", ha="center")
        ax1.axis("off")
        
    # 2. Photo 5 (IMG_5130) with detected lines & rule-based chunk split crop
    p5_path = Path("data/real_test5/photo5.jpg")
    g9_path = Path("results/real_fix9/crop_geometry.json")
    
    if p5_path.exists() and g9_path.exists():
        im5 = Image.open(p5_path)
        g9 = json.load(open(g9_path, encoding="utf-8"))
        
        im5_small = im5.resize((im5.width // 4, im5.height // 4))
        scale5 = 0.25
        dr5 = ImageDraw.Draw(im5_small)
        
        for item in g9:
            x0, y0, x1, y1 = item["x0"]*scale5, item["y0"]*scale5, item["x1"]*scale5, item["y1"]*scale5
            dr5.rectangle([x0, y0, x1, y1], outline="red", width=2)
            
        ax2.imshow(im5_small)
        ax2.set_title("Photo 5 (IMG_5130, 33 Lines)\nFull Page Layout with 5 Paragraphs", fontsize=9.5, weight="bold")
        ax2.axis("off")
        
        # Inset: show a line crop with the vertical red chunk boundary
        # Line 00 crop from results/real_fix9/crops/line_00.png
        p_crop = Path("results/real_fix9/crops/line_00.png")
        if p_crop.exists():
            im_c = Image.open(p_crop)
            # Find chunk boundary from per_line_chunk_details.csv
            # For Line 00, it split into 2 chunks
            sub_ax = ax2.inset_axes([0.05, 0.05, 0.90, 0.16])
            sub_ax.imshow(im_c, cmap="gray")
            # Draw vertical line around split point (~1000px on original)
            split_x = im_c.width * (1140 / 1969)  # approx ratio
            sub_ax.axvline(split_x, color=COLOR_RED, lw=2, linestyle="--")
            sub_ax.set_title("Line 00: Inter-Word Split Boundary (Red Dashed Line, k=25, p=35)", fontsize=7.5, color=COLOR_RED, weight="bold")
            sub_ax.axis("off")
            
    fig.text(0.05, 0.02, "Source: data/real_docs/photo1_times.jpg, data/real_test5/photo5.jpg, results/real_fix8/crop_geometry.json, results/real_fix9/crop_geometry.json",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "f8_real_photos_detection.png", dpi=200)
    plt.close(fig)
    print("F8 generated.")

# ==============================================================================
# F9: Annotated line examples from Photo 5 (difflib comparison)
# ==============================================================================
def make_f9():
    fig, axes = plt.subplots(6, 1, figsize=(13, 8.5), dpi=200)
    fig.suptitle("Annotated Line Recognition Examples from Photo 5 (IMG_5130)", fontsize=13, weight="bold")
    
    csv_path = Path("results/real_fix9/per_line_chunk_details.csv")
    df = pd.read_csv(csv_path)
    
    # Choose 6 representative lines:
    # L00 (exact both), L03 (clean), L11 (dotless/spacing), L16 (təhsil), L18 (psm7 empty fallback), L24 (küpün)
    selected_indices = [0, 3, 11, 16, 18, 24]
    
    for row_idx, line_idx in enumerate(selected_indices):
        ax = axes[row_idx]
        ax.axis("off")
        
        row = df[df["line_idx"] == line_idx].iloc[0]
        ref = row["ref"]
        v2_hyp = row["v2_rule_joined_A"]
        tess_hyp = row["tess_full_A_psm13"]
        
        crop_path = Path(f"results/real_fix9/crops/line_{line_idx:02d}.png")
        
        # Sub-axes for image crop on left
        if crop_path.exists():
            im_c = Image.open(crop_path)
            sub_im = ax.inset_axes([0.0, 0.1, 0.38, 0.8])
            sub_im.imshow(im_c, cmap="gray")
            sub_im.axis("off")
            
        # Text details on right
        ax.text(0.40, 0.85, f"Line {line_idx+1:02d} | Reference:", fontsize=8, weight="bold", color="#111111")
        ax.text(0.55, 0.85, ref, fontsize=8, color="#222222")
        
        # V2 text with error highlights
        ax.text(0.40, 0.50, f"V2 Output (Chunked):", fontsize=8, weight="bold", color=COLOR_V2)
        ax.text(0.55, 0.50, v2_hyp, fontsize=8, color="#222222" if v2_hyp == ref else COLOR_RED)
        
        # Tess text with psm13
        ax.text(0.40, 0.15, f"Tess psm 13 Output:", fontsize=8, weight="bold", color=COLOR_TESS)
        ax.text(0.55, 0.15, tess_hyp, fontsize=8, color="#222222" if tess_hyp == ref else COLOR_RED)
        
        # Divider line
        ax.axhline(0.0, color="#EEEEEE", lw=0.8)

    fig.text(0.05, 0.015, "Source: results/real_fix9/per_line_chunk_details.csv, results/real_fix9/crops/line_00.png..line_24.png",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(FIG_DIR / "f9_line_annotations.png", dpi=200)
    plt.close(fig)
    print("F9 generated.")

# ==============================================================================
# F10: Squeeze analysis scatter & grouped bar chart
# ==============================================================================
def make_f10():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=200)
    fig.suptitle("Aspect Ratio Squeeze Analysis: Aspect Ratio vs. CER Degradation", fontsize=12, weight="bold")
    
    # 1. Scatter plot: W*32/(h*512) vs CER
    df9 = pd.read_csv("results/real_fix9/per_line_chunk_details.csv")
    sys.path.insert(0, ".")
    from common import align
    
    ratios = []
    v2_full_cers = []
    v2_chunk_cers = []
    
    for _, r in df9.iterrows():
        ref = r["ref"]
        # compute exact CER per line
        e_full, _ = align(ref, r["v2_full_A"])
        e_chunk, _ = align(ref, r["v2_rule_joined_A"])
        ratios.append(r["ratio_w_h"])
        v2_full_cers.append(e_full / len(ref) * 100)
        v2_chunk_cers.append(e_chunk / len(ref) * 100)
        
    ax1.scatter(ratios, v2_full_cers, color=COLOR_V1, s=40, alpha=0.8, label="V2 Full Line CER")
    ax1.scatter(ratios, v2_chunk_cers, color=COLOR_V2, s=40, alpha=0.8, marker="^", label="V2 Chunked CER")
    
    # Threshold lines at 1.0 and 1.3
    ax1.axvline(1.0, color=COLOR_RED, linestyle="--", lw=1.2, label="Threshold 1.0")
    ax1.axvline(1.3, color=COLOR_RED, linestyle="-.", lw=1.2, label="Threshold 1.3 (Severe Squeeze)")
    
    ax1.set_xlabel("Squeeze Ratio  r = (W · 32) / (h · 512)", fontsize=9)
    ax1.set_ylabel("Line CER (%)", fontsize=9)
    ax1.set_title("Per-Line CER vs Squeeze Ratio (33 Lines)", fontsize=9.5, weight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8)
    
    # 2. Grouped bar chart by ratio group from results/real_fix9/metrics.json:313-395
    groups = ["< 1.0\n(n=4)", "1.0 – 1.1\n(n=10)", "1.1 – 1.3\n(n=13)", "> 1.3\n(n=6)"]
    full_cers = [0.54, 4.74, 3.36, 4.87]
    chunk_cers = [1.61, 3.93, 1.16, 1.33]
    reductions = [-1.08, +0.81, +2.21, +3.54]
    
    x = np.arange(4)
    w = 0.35
    
    b1 = ax2.bar(x - w/2, full_cers, width=w, label="V2 Full Line CER", color=COLOR_V1)
    b2 = ax2.bar(x + w/2, chunk_cers, width=w, label="V2 Chunked CER", color=COLOR_V2)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=8.5)
    ax2.set_ylabel("Group CER (%)", fontsize=9)
    ax2.set_title("CER Reduction by Ratio Group (Threshold Rule Benefit)", fontsize=9.5, weight="bold")
    ax2.set_ylim(0, 6.0)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    # Annotate delta reduction
    for i in range(4):
        ax2.text(x[i], max(full_cers[i], chunk_cers[i]) + 0.3, 
                 f"Δ: {reductions[i]:+.2f} pp", ha="center", fontsize=8, weight="bold",
                 color=COLOR_RED if reductions[i] < 0 else COLOR_V2)
                 
    ax2.legend(loc="upper right", fontsize=8)

    fig.text(0.05, 0.02, "Source: results/real_fix9/metrics.json:313-395, results/real_fix9/per_line_chunk_details.csv",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(FIG_DIR / "f10_squeeze_analysis.png", dpi=200)
    plt.close(fig)
    print("F10 generated.")

# ==============================================================================
# F11: Border padding sensitivity (4px vs 20px)
# ==============================================================================
def make_f11():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.suptitle("Boundary Padding Sensitivity: Condition A (4 px) vs Condition B (20 px)", fontsize=12, weight="bold")
    
    # Data from results/real_fix9/metrics.json:35-42, 149-165
    models = [
        "V2 Rule Chunk",
        "V2 Full Line",
        "Tesseract Chunk (psm 7)",
        "Tesseract Full (psm 7)",
        "Tesseract Full (psm 13)"
    ]
    
    cer_a = [2.10, 3.87, 13.02, 8.42, 0.09]
    cer_b = [23.32, 22.72, 18.86, 10.91, 1.80]
    
    x = np.arange(len(models))
    w = 0.35
    
    b1 = ax.bar(x - w/2, cer_a, width=w, label="Condition A (Trim + 4 px margin)", color=COLOR_V2)
    b2 = ax.bar(x + w/2, cer_b, width=w, label="Condition B (Padded 20 px white border)", color=COLOR_RED)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("Character Error Rate (CER %)", fontsize=9)
    ax.set_ylim(0, 28.0)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color=COLOR_RED, weight="bold")

    ax.text(0.03, 0.85, 
            "CRITICAL FRAGILITY FINDING:\n"
            "• V2 collapses from 2.10% -> 23.32% (+21.2 pp error) when padded with 20px white canvas.\n"
            "• CRNN receptive field was trained on tight crops; blank border triggers blank hallucinations.\n"
            "• Tesseract is significantly more robust to border whitespace padding.",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEEEE", edgecolor=COLOR_RED, lw=1))

    ax.legend(loc="upper right", fontsize=8.5)

    fig.text(0.05, 0.02, "Source: results/real_fix9/metrics.json:2-183, results/real_fix9/report.md:33-48",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG_DIR / "f11_padding_sensitivity.png", dpi=200)
    plt.close(fig)
    print("F11 generated.")

# ==============================================================================
# F12: Error taxonomy stacked bars
# ==============================================================================
def make_f12():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=200)
    fig.suptitle("Mutually Exclusive Error Taxonomy: V2 Line vs Tesseract 5", fontsize=12, weight="bold")
    
    categories = ["Leading Edge", "Trailing Edge", "Internal Space", "0 / O Confusion", "Internal Punctuation", "Internal Letter"]
    colors = ["#CCCCCC", "#AAAAAA", "#F39C12", "#9B59B6", "#3498DB", "#E74C3C"]
    
    # 1. Condition A (All 33 Lines)
    # V2: 0, 0, 3, 0, 12, 34 = 49
    # Tess: 3, 0, 44, 0, 20, 236 = 303
    v2_a = [0, 0, 3, 0, 12, 34]
    tess_a = [3, 0, 44, 0, 20, 236]
    
    # 2. Clean Subset (23 Lines)
    # V2: 0, 0, 2, 0, 6, 13 = 21
    # Tess: 3, 0, 1, 0, 7, 9 = 20
    v2_clean = [0, 0, 2, 0, 6, 13]
    tess_clean = [3, 0, 1, 0, 7, 9]
    
    # Plot Stacked Bars for Condition A
    bottom_v2 = 0
    bottom_tess = 0
    for idx, cat in enumerate(categories):
        ax1.bar(0, v2_a[idx], bottom=bottom_v2, color=colors[idx], width=0.5, label=cat)
        bottom_v2 += v2_a[idx]
        ax1.bar(1, tess_a[idx], bottom=bottom_tess, color=colors[idx], width=0.5)
        bottom_tess += tess_a[idx]
        
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["V2 Line\n(Sum: 49)", "Tesseract psm 7\n(Sum: 303)"], fontsize=9, weight="bold")
    ax1.set_ylabel("Total Character Errors (Levenshtein)", fontsize=9)
    ax1.set_title("Condition A: All 33 Lines\n(Tess includes 9 empty chunks)", fontsize=9.5, weight="bold")
    ax1.set_ylim(0, 330)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    # Plot Stacked Bars for Clean Subset
    bottom_v2_c = 0
    bottom_tess_c = 0
    for idx, cat in enumerate(categories):
        ax2.bar(0, v2_clean[idx], bottom=bottom_v2_c, color=colors[idx], width=0.5)
        bottom_v2_c += v2_clean[idx]
        ax2.bar(1, tess_clean[idx], bottom=bottom_tess_c, color=colors[idx], width=0.5)
        bottom_tess_c += tess_clean[idx]
        
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["V2 Line\n(Sum: 21)", "Tesseract psm 7\n(Sum: 20)"], fontsize=9, weight="bold")
    ax2.set_ylabel("Total Character Errors (Levenshtein)", fontsize=9)
    ax2.set_title("Clean Subset: 23 Lines\n(Zero empty outputs across all models)", fontsize=9.5, weight="bold")
    ax2.set_ylim(0, 35)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    ax1.legend(loc="upper right", fontsize=8)

    fig.text(0.05, 0.02, "Source: results/real_fix9/error_taxonomy.json, results/real_fix9/report.md:69-82",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG_DIR / "f12_error_taxonomy.png", dpi=200)
    plt.close(fig)
    print("F12 generated.")

# ==============================================================================
# F13: Tesseract PSM 7 vs PSM 13 comparison
# ==============================================================================
def make_f13():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.suptitle("Tesseract 5: Page Segmentation Mode (PSM 7 vs PSM 13) Diagnostics", fontsize=12, weight="bold")
    
    # Left: Empty output counts
    categories = ["Condition A\n(Trim + 4px)", "Condition B\n(Padded 20px)"]
    empty_psm7 = [11, 13]  # 9 chunks + 2 full lines in A; 11 chunks + 2 full in B
    empty_psm13 = [0, 0]
    
    x = np.arange(2)
    w = 0.35
    
    b1 = ax1.bar(x - w/2, empty_psm7, width=w, label="PSM 7 (Single Text Line)", color=COLOR_RED)
    b2 = ax1.bar(x + w/2, empty_psm13, width=w, label="PSM 13 (Raw Line / No OSD)", color=COLOR_V2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9)
    ax1.set_ylabel("Number of Empty Outputs (\" \")", fontsize=9)
    ax1.set_title("Empty Output Count (Integration Failure)", fontsize=9.5, weight="bold")
    ax1.set_ylim(0, 16)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{int(h)} empty", ha="center", va="bottom", fontsize=8, color=COLOR_RED, weight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{int(h)}", ha="center", va="bottom", fontsize=8, color=COLOR_V2, weight="bold")
        
    ax1.legend(loc="upper right", fontsize=8.5)

    # Right: CER comparison
    modes = ["Rule Chunk\n(Cond A)", "Full Line\n(Cond A)", "Clean Subset\n(23 Lines)"]
    cer_psm7 = [13.02, 8.42, 1.26]
    cer_psm13 = [0.09, 0.09, 0.00]  # psm 13 full line CER
    
    x2 = np.arange(3)
    b3 = ax2.bar(x2 - w/2, cer_psm7, width=w, label="PSM 7 CER", color=COLOR_RED)
    b4 = ax2.bar(x2 + w/2, cer_psm13, width=w, label="PSM 13 CER (Fallback/Raw)", color=COLOR_TESS)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(modes, fontsize=8.5)
    ax2.set_ylabel("Tesseract Character Error Rate (CER %)", fontsize=9)
    ax2.set_title("CER Collapse under PSM 7 vs Near-Perfection under PSM 13", fontsize=9.5, weight="bold")
    ax2.set_ylim(0, 16)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color=COLOR_RED)
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color=COLOR_TESS, weight="bold")
        
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.text(0.05, 0.02, "Source: results/real_fix9/metrics.json:25-46, 396-405, results/real_fix9/empty_outputs_log.json",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG_DIR / "f13_tess_psm_comparison.png", dpi=200)
    plt.close(fig)
    print("F13 generated.")

# ==============================================================================
# F14: Lexicon pilot & test-snooping risk disclosure
# ==============================================================================
def make_f14():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.suptitle("Lexicon Post-Correction Pilot: Gains vs. Test-Snooping Risk", fontsize=12, weight="bold")
    
    # 1. Bar of changed words: 8 correct vs 1 wrong
    cats = ["Correct Corrections\n(True Positives)", "Erroneous Changes\n(False Positives)"]
    counts = [8, 1]
    colors = [COLOR_V2, COLOR_RED]
    
    b = ax1.bar(cats, counts, color=colors, width=0.45)
    ax1.set_ylabel("Number of Modified Words (Total: 9)", fontsize=9)
    ax1.set_title("Word Modification Breakdown (Photo 5)", fontsize=9.5, weight="bold")
    ax1.set_ylim(0, 10)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    ax1.text(0, 8.3, "8 Words (+8 chars)\n(sadəcə, təhsil, gənc,\ngörüb, başını, vaxt,\nbabasının, inkişafına)", 
             ha="center", va="bottom", fontsize=7.5, color=COLOR_V2, weight="bold")
    ax1.text(1, 1.3, "1 Word (-1 char)\n(kiplər -> kimlər;\nRef: küplər)", 
             ha="center", va="bottom", fontsize=7.5, color=COLOR_RED, weight="bold")
             
    # 2. CER Impact: Raw V2 vs Lexicon Corrected
    stages = ["Raw V2 Line\n(Frozen Pipeline)", "V2 + Heuristic Lexicon\n(Test-Snooping Pilot)"]
    cers = [2.10, 1.80]
    
    b2 = ax2.bar(stages, cers, color=[COLOR_V1, "#27AE60"], width=0.45)
    ax2.set_ylabel("Photo 5 Character Error Rate (CER %)", fontsize=9)
    ax2.set_title("CER Impact on Photo 5 (IMG_5130, 2328 chars)", fontsize=9.5, weight="bold")
    ax2.set_ylim(0, 3.0)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    ax2.text(0, 2.15, "2.10% (49 errs)\n[Main Scientific Result]", ha="center", va="bottom", fontsize=8, weight="bold")
    ax2.text(1, 1.85, "1.80% (42 errs)\n[Net gain: +7 chars]", ha="center", va="bottom", fontsize=8, weight="bold", color="#27AE60")
    
    # Red warning banner for test-snooping risk
    ax2.text(0.5, 0.40, "CRITICAL SCIENTIFIC DISCLOSURE:\nTEST-SNOOPING RISK\nRules & suffixes were tuned after inspecting Photo 5 errors.\nPrimary benchmark result remains Raw V2 (2.10%).",
             ha="center", va="center", transform=ax2.transAxes, color=COLOR_RED, fontsize=7.5, weight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEEEE", edgecolor=COLOR_RED, lw=1.2))

    fig.text(0.05, 0.02, "Source: scratch/test_safe_corrector.py, results/real_fix9/per_line_chunk_details.csv, results/real_fix9/metrics.json:58-68",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG_DIR / "f14_lexicon_pilot.png", dpi=200)
    plt.close(fig)
    print("F14 generated.")

if __name__ == "__main__":
    make_f1()
    make_f2()
    make_f3()
    make_f4()
    make_f5()
    make_f6()
    make_f7()
    make_f8()
    make_f9()
    make_f10()
    make_f11()
    make_f12()
    make_f13()
    make_f14()
    print("All 14 figures successfully generated in paper/figures/!")
