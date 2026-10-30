"""
Regenerate Figures F8, F9, F11, F13 with suffix _v2 for BM 606 Azerbaijani OCR Paper.
Adheres strictly to all user constraints:
- Do NOT run new experiments, do NOT modify existing results.
- Every plotted number comes from existing files (metrics.json, CSVs).
- F13: Checked metrics.json; omitted psm 13 on rule chunks / labeled 'not measured'.
       Compares like with like with units in labels; shows clean subset (full psm7 1.07%, chunk psm7 1.26%, full psm13 0.00%).
       Splits empty counts into chunks vs full lines.
       Title: "Tesseract sensitivity to page segmentation mode" (no 'collapse'); note psm 13 chosen post-hoc.
- F11: Text box moved OUTSIDE plot area, no bar/label coverage.
       States actual relative increases for all 5 bars.
       Marks cause as "hypothesis, not tested".
       Neutral color for Condition B (not red).
       Notes "two padding values only (4 px, 20 px)".
- F9: Character-level difflib highlighting (mismatched in red, correct in dark green/black).
      Selection rule for 6 lines stated in caption (hand-selected: 2 V2 perfect, 2 Tess errs, 1 em-dash, 1 V2 err).
      Notes crops contain vertical fragments of neighbouring lines.
      States em dash ('—', U+2014) vs hyphen ('-', U+002D) counts as 1 error in raw Levenshtein.
- F8: Zoomed crop in separate panel (not on top of photo).
      Draws split boundaries as dashed lines on 5 lines.
      Bigger line numbers.
      Caption notes Photo 1 shows only the evaluated region (16 lines).
"""

import sys
import os
import json
import difflib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parent.parent
FIG_DIR = WORKSPACE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent Palette
COLOR_V1 = "#4682B4"      # Steel Blue
COLOR_V2 = "#2E8B57"      # Sea Green
COLOR_TESS = "#D95F02"    # Dark Orange / Rust
COLOR_NEUTRAL_B = "#4A6572" # Slate / Dark Neutral for Condition B
COLOR_RED = "#D62728"     # Red (ONLY for errors, warnings, thresholds)
COLOR_GREEN = "#1B5E20"   # Dark Green for correct matches
COLOR_GRAY = "#666666"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8


# ==============================================================================
# F8_v2: Real photo detection and chunking boundaries
# ==============================================================================
def make_f8_v2():
    print("Generating F8_v2...")
    # 3-panel layout: Panel A (Photo 1), Panel B (Photo 5), Panel C (Zoomed split crops)
    fig = plt.figure(figsize=(17, 8.8), dpi=200)
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.45], wspace=0.22)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    fig.suptitle("Figure 8: Real Document Processing: Line Detection & Projection-Based Chunking Boundaries",
                 fontsize=13, weight="bold", y=0.98)
    
    # Panel A: Photo 1 (TNR, 16 lines)
    p1_path = WORKSPACE / "data" / "real_docs" / "photo1_times.jpg"
    g8_path = WORKSPACE / "results" / "real_fix8" / "crop_geometry.json"
    
    if p1_path.exists() and g8_path.exists():
        im1 = Image.open(p1_path)
        g8 = json.load(open(g8_path, encoding="utf-8"))
        
        scale1 = 0.32
        im1_disp = im1.resize((int(im1.width * scale1), int(im1.height * scale1)), Image.Resampling.LANCZOS)
        dr1 = ImageDraw.Draw(im1_disp)
        
        for item in g8.get("photo1", []):
            x0 = item["x0"] * scale1
            y0 = item["y0"] * scale1
            x1 = item["x1"] * scale1
            y1 = item["y1"] * scale1
            dr1.rectangle([x0, y0, x1, y1], outline="#E74C3C", width=3)
            # Bigger line numbers with solid background badge
            l_num = str(item["line_idx"] + 1)
            dr1.rectangle([x0 - 2, y0 - 18, x0 + 26, y0 + 1], fill="#C0392B")
            dr1.text((x0 + 3, y0 - 17), f"{int(l_num):02d}", fill="white")
            
        ax1.imshow(im1_disp)
        ax1.set_title("Panel A: Photo 1 (Times New Roman)\nEvaluated Region of 16 Lines", fontsize=10, weight="bold")
        ax1.axis("off")
        ax1.text(0.5, -0.04, "Note: Photo 1 shows only the evaluated region (16 lines)",
                 ha="center", va="top", transform=ax1.transAxes, fontsize=8, color="#555555", style="italic")
    else:
        ax1.text(0.5, 0.5, "Photo 1 image not found", ha="center")
        ax1.axis("off")

    # Panel B: Photo 5 (IMG_5130, 33 lines)
    p5_path = WORKSPACE / "data" / "real_test5" / "photo5.jpg"
    g9_path = WORKSPACE / "results" / "real_fix9" / "crop_geometry.json"
    
    if p5_path.exists() and g9_path.exists():
        im5 = Image.open(p5_path)
        g9 = json.load(open(g9_path, encoding="utf-8"))
        
        scale5 = 0.32
        im5_disp = im5.resize((int(im5.width * scale5), int(im5.height * scale5)), Image.Resampling.LANCZOS)
        dr5 = ImageDraw.Draw(im5_disp)
        
        for item in g9:
            x0 = item["x0"] * scale5
            y0 = item["y0"] * scale5
            x1 = item["x1"] * scale5
            y1 = item["y1"] * scale5
            dr5.rectangle([x0, y0, x1, y1], outline="#E74C3C", width=2)
            # Bigger line numbers on left
            l_num = str(item["line_idx"] + 1)
            dr5.rectangle([x0 - 24, y0, x0 - 1, y0 + 14], fill="#C0392B")
            dr5.text((x0 - 22, y0 + 1), f"{int(l_num):02d}", fill="white")
            
        ax2.imshow(im5_disp)
        ax2.set_title("Panel B: Photo 5 (IMG_5130, 33 Lines)\nFull Page Layout (5 Paragraphs)", fontsize=10, weight="bold")
        ax2.axis("off")
        ax2.text(0.5, -0.04, "Tesseract image_to_data grouping (33 line boxes)",
                 ha="center", va="top", transform=ax2.transAxes, fontsize=8, color="#555555", style="italic")

    # Panel C: Zoomed split boundaries on 5 lines
    ax3.set_title("Panel C: Projection Split Boundaries (5 Lines)\nInter-Word Valleys (k=25, 35th percentile ink)", 
                  fontsize=10, weight="bold")
    ax3.axis("off")
    
    split_lines_data = [
        (0, 1186, "Line 01 (w=1969 px): Split at x=1186 px between 'qədim' and 'sənət'"),
        (1, 1238, "Line 02 (w=2185 px): Split at x=1238 px between 'özündə' and 'yaşadır.'"),
        (2, 1202, "Line 03 (w=2098 px): Split at x=1202 px between 'sahələri' and 'nəsildən-nəslə,'"),
        (3, 1257, "Line 04 (w=2161 px): Split at x=1257 px between 'gəlib' and 'çatmışdır.'"),
        (4, 1058, "Line 05 (w=1982 px): Split at x=1058 px between 'dərin' and 'fəlsəfə'")
    ]
    
    sub_height = 0.150
    sub_spacing = 0.045
    
    for idx, (l_idx, split_x, desc) in enumerate(split_lines_data):
        crop_path = WORKSPACE / "results" / "real_fix9" / "crops" / f"line_{l_idx:02d}.png"
        y_pos = 1.0 - (idx + 1) * (sub_height + sub_spacing) + 0.015
        
        crop_ax = ax3.inset_axes([0.02, y_pos, 0.96, sub_height])
        if crop_path.exists():
            im_c = Image.open(crop_path)
            crop_ax.imshow(im_c, cmap="gray")
            # Draw dashed red vertical line at exact split point
            crop_ax.axvline(split_x, color=COLOR_RED, lw=2.2, linestyle="--")
            
            # Annotate Chunk 1 and Chunk 2 clearly
            crop_ax.text(split_x * 0.40, im_c.height * 0.85, "◄ Chunk 1", color="#1B5E20", 
                         fontsize=8, weight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
            crop_ax.text(split_x + (im_c.width - split_x) * 0.40, im_c.height * 0.85, "Chunk 2 ►", color="#0D47A1", 
                         fontsize=8, weight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
                         
        crop_ax.set_title(desc, fontsize=8, weight="bold", loc="left", pad=3)
        crop_ax.set_xticks([])
        crop_ax.set_yticks([])
        for spine in crop_ax.spines.values():
            spine.set_edgecolor("#999999")
            spine.set_linewidth(0.8)

    # Footnote & caption info
    fig.text(0.04, 0.015,
             "Source: data/real_docs/photo1_times.jpg, data/real_test5/photo5.jpg, results/real_fix8/crop_geometry.json, "
             "results/real_fix9/crop_geometry.json, results/real_fix9/crops/line_00.png..line_04.png\n"
             "Caption: Figure 8: Real document line detection and projection chunking. Panel A shows Photo 1 (only the evaluated region of 16 lines is displayed). "
             "Panel B shows Photo 5 full page layout. Panel C displays projection-based split boundaries (dashed red lines) across 5 representative lines in a dedicated separate panel.",
             fontsize=7.5, color="#555555", style="italic")
             
    fig.savefig(FIG_DIR / "f8_real_photos_detection_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("F8_v2 generated successfully.")


# ==============================================================================
# F9_v2: Annotated line examples from Photo 5 (character-level difflib)
# ==============================================================================
def make_f9_v2():
    print("Generating F9_v2...")
    fig, axes = plt.subplots(6, 1, figsize=(15, 9.8), dpi=200)
    fig.suptitle("Figure 9: Annotated Line Recognition Examples from Photo 5 (IMG_5130)", fontsize=13, weight="bold", y=0.99)
    
    csv_path = WORKSPACE / "results" / "real_fix9" / "per_line_chunk_details.csv"
    df = pd.read_csv(csv_path)
    
    selected_indices = [2, 11, 18, 20, 21, 23]
    
    def render_difflib_text(ax, x_start, y_pos, label_prefix, label_color, ref_str, hyp_str):
        ax.text(x_start, y_pos, label_prefix, fontsize=7.8, weight="bold", color=label_color, fontfamily="Consolas")
        
        x_char_start = x_start + 0.058
        char_step = 0.0086
        
        matcher = difflib.SequenceMatcher(None, ref_str, hyp_str)
        curr_x = x_char_start
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                segment = hyp_str[j1:j2]
                for ch in segment:
                    col = COLOR_GREEN if hyp_str == ref_str else "#222222"
                    ax.text(curr_x, y_pos, ch, fontsize=7.8, color=col, fontfamily="Consolas")
                    curr_x += char_step
            elif tag in ('replace', 'insert'):
                segment = hyp_str[j1:j2]
                for ch in segment:
                    # Highlight mismatched character in bold red with background marker
                    ax.text(curr_x, y_pos, ch, fontsize=8.0, weight="bold", color=COLOR_RED, fontfamily="Consolas",
                            bbox=dict(facecolor="#FFD6D6", edgecolor=COLOR_RED, boxstyle="round,pad=0.04", lw=0.6))
                    curr_x += char_step
            elif tag == 'delete':
                del_str = f"[{ref_str[i1:i2]}]"
                for ch in del_str:
                    ax.text(curr_x, y_pos, ch, fontsize=7.2, weight="bold", color="#C0392B", fontfamily="Consolas")
                    curr_x += char_step * 0.9

    for row_idx, l_idx in enumerate(selected_indices):
        ax = axes[row_idx]
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        
        row = df[df["line_idx"] == l_idx].iloc[0]
        ref = row["ref"]
        v2_hyp = row["v2_rule_joined_A"]
        tess_hyp = row["tess_full_A_psm13"]
        
        # Sub-axis for cropped image on left (compact width to leave generous room for text)
        crop_path = WORKSPACE / "results" / "real_fix9" / "crops" / f"line_{l_idx:02d}.png"
        crop_ax = ax.inset_axes([0.005, 0.12, 0.22, 0.76])
        if crop_path.exists():
            im_c = Image.open(crop_path)
            crop_ax.imshow(im_c, cmap="gray")
        crop_ax.set_xticks([])
        crop_ax.set_yticks([])
        for spine in crop_ax.spines.values():
            spine.set_edgecolor("#AAAAAA")
            spine.set_linewidth(0.8)
            
        # Category tag
        if l_idx in [2, 11]:
            tag = "[V2 100% PERFECT]"
            tag_color = "#1E7E34"
        elif l_idx in [20, 21]:
            tag = "[TESSERACT PSM 13 ERROR]"
            tag_color = "#D95F02"
        elif l_idx == 18:
            tag = "[EM-DASH (—) vs HYPHEN (-) SUBSTITUTION TEST]"
            tag_color = "#6A1B9A"
        else:
            tag = "[V2 GLYPH & PUNCTUATION ERROR]"
            tag_color = "#C0392B"
            
        ax.text(0.24, 0.88, f"Line {l_idx+1:02d}  {tag}", fontsize=8.5, weight="bold", color=tag_color)
        
        # 1. Reference Line
        ax.text(0.24, 0.64, "REF: ", fontsize=7.8, weight="bold", color="#111111", fontfamily="Consolas")
        curr_x = 0.24 + 0.058
        for ch in ref:
            ax.text(curr_x, 0.64, ch, fontsize=7.8, color="#111111", fontfamily="Consolas")
            curr_x += 0.0086
            
        # 2. V2 Output with difflib highlighting
        render_difflib_text(ax, 0.24, 0.40, "V2:  ", COLOR_V2, ref, v2_hyp)
        
        # 3. Tesseract Output with difflib highlighting
        render_difflib_text(ax, 0.24, 0.16, "TESS:", COLOR_TESS, ref, tess_hyp)
        
        # Divider line
        ax.axhline(0.02, color="#E0E0E0", lw=0.8)

    # Caption and explicit disclosures
    fig.text(0.02, 0.015,
             "Source: results/real_fix9/per_line_chunk_details.csv, results/real_fix9/crops/line_02,11,18,20,21,23.png\n"
             "Selection Rule: Chosen by hand to inspect boundary behaviors: includes 2 lines where V2 is 100% perfect (Lines 03, 12), the only 2 lines where Tesseract PSM 13 errs (Lines 21, 22),\n"
             "1 line testing em-dash ('—', U+2014) vs hyphen ('-', U+002D) substitution (Line 19), and 1 line with V2 vowel misrecognition (Line 24). In raw Levenshtein, '—' vs '-' counts as 1 error.\n"
             "Note: Image crops contain vertical fragments of neighbouring lines (ascenders/descenders) due to the ±4 px bounding box detection margin.",
             fontsize=7.5, color="#555555", style="italic")
             
    fig.savefig(FIG_DIR / "f9_line_annotations_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("F9_v2 generated successfully.")


# ==============================================================================
# F11_v2: Border padding sensitivity (4px vs 20px)
# ==============================================================================
def make_f11_v2():
    print("Generating F11_v2...")
    fig = plt.figure(figsize=(11.0, 7.5), dpi=200)
    gs = GridSpec(2, 1, height_ratios=[3.2, 1.25], hspace=0.35)
    
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    
    fig.suptitle("Figure 11: Boundary Padding Sensitivity: Condition A (4 px) vs. Condition B (20 px)", 
                 fontsize=12, weight="bold", y=0.98)
    
    models = [
        "V2 Rule Chunk\n[Chunk unit]",
        "V2 Full Line\n[Full Line unit]",
        "Tesseract Chunk (PSM 7)\n[Chunk unit]",
        "Tesseract Full (PSM 7)\n[Full Line unit]",
        "Tesseract Full (PSM 13)\n[Full Line unit]"
    ]
    
    cer_a = [2.10, 3.87, 13.02, 8.42, 0.09]
    cer_b = [23.32, 22.72, 18.86, 10.91, 1.80]
    
    rel_inc = ["+1010% (x11.1)", "+487% (x5.9)", "+44.9% (x1.45)", "+29.6% (x1.30)", "+1900% (x20.0)"]
    pp_diff = ["+21.22 pp", "+18.85 pp", "+5.84 pp", "+2.49 pp", "+1.71 pp"]
    
    x = np.arange(len(models))
    w = 0.35
    
    b1 = ax.bar(x - w/2, cer_a, width=w, label="Condition A: Tight Trim + 4 px Margin", color=COLOR_V2)
    b2 = ax.bar(x + w/2, cer_b, width=w, label="Condition B: Padded 20 px White Border", color=COLOR_NEUTRAL_B)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("Character Error Rate (CER %)", fontsize=9.5)
    ax.set_ylim(0, 30.0)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for idx, bar in enumerate(b1):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color="#1B5E20")
    for idx, bar in enumerate(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4, f"{h:.2f}%\n({pp_diff[idx]})", ha="center", va="bottom", fontsize=7.5, color=COLOR_NEUTRAL_B, weight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, h + 3.0, rel_inc[idx], ha="center", va="bottom", fontsize=7, color="#333333", style="italic")

    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Tested under two padding values only (4 px margin vs. 20 px border; intermediate padding was not measured)",
                 fontsize=8.5, color="#555555", style="italic", pad=8)

    # Separate Text Box OUTSIDE plot area in ax_text
    box_rect = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.03",
                                      facecolor="#F8F9FA", edgecolor="#CCCCCC", linewidth=1.0)
    ax_text.add_patch(box_rect)
    
    callout_text = (
        "EMPIRICAL AUDIT & RELATIVE INCREASES ACROSS ALL FIVE CONDITIONS:\n"
        "• V2 Rule Chunk: 2.10% -> 23.32% (+21.22 pp, x11.1 relative increase / +1010%)\n"
        "• V2 Full Line: 3.87% -> 22.72% (+18.85 pp, x5.9 relative increase / +487%)\n"
        "• Tesseract Chunk (PSM 7): 13.02% -> 18.86% (+5.84 pp, x1.45 relative increase / +44.9%)\n"
        "• Tesseract Full (PSM 7): 8.42% -> 10.91% (+2.49 pp, x1.30 relative increase / +29.6%)\n"
        "• Tesseract Full (PSM 13): 0.09% -> 1.80% (+1.71 pp, x20.0 relative increase / +1900%)\n"
        "Cause (Hypothesis, not tested): CRNN receptive field was trained exclusively on tightly cropped synthetic lines without canvas borders;\n"
        "introducing a 20 px white margin causes convolutional features to drift into blank activations, triggering blank hallucinations."
    )
    ax_text.text(0.04, 0.50, callout_text, fontsize=8, va="center", color="#222222", linespacing=1.35)

    fig.text(0.04, 0.01, "Source: results/real_fix9/metrics.json:2-183, results/real_fix9/report.md:33-48",
             fontsize=7.5, color="#666666", style="italic")
             
    fig.savefig(FIG_DIR / "f11_padding_sensitivity_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("F11_v2 generated successfully.")


# ==============================================================================
# F13_v2: Tesseract sensitivity to page segmentation mode
# ==============================================================================
def make_f13_v2():
    print("Generating F13_v2...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=200)
    fig.suptitle("Figure 13: Tesseract Sensitivity to Page Segmentation Mode (PSM 7 vs. PSM 13)",
                 fontsize=12.5, weight="bold", y=0.98)
    
    # -------------------------------------------------------------
    # Panel 1: Empty Output Count (Split Chunks vs Full Lines)
    # -------------------------------------------------------------
    categories = ["Condition A\n(Trim + 4 px)", "Condition B\n(Padded 20 px)"]
    x = np.arange(len(categories))
    w = 0.22
    
    b_c7 = ax1.bar(x - w, [9, 11], width=w, label="PSM 7 Chunks [Chunk unit]", color="#C0392B")
    b_f7 = ax1.bar(x, [2, 2], width=w, label="PSM 7 Full Lines [Full Line unit]", color="#E67E22")
    b_f13 = ax1.bar(x + w, [0, 0], width=w, label="PSM 13 Full Lines [Full Line unit]", color=COLOR_V2)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=9, weight="bold")
    ax1.set_ylabel("Number of Empty Outputs (\" \")", fontsize=9.5)
    ax1.set_title("Empty Output Count (Integration Failure Breakdown)\nChunks vs. Full Lines", fontsize=10, weight="bold")
    ax1.set_ylim(0, 14.0)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for bar in b_c7:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{int(h)} empty", ha="center", va="bottom", fontsize=8, color="#C0392B", weight="bold")
    for bar in b_f7:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{int(h)} empty", ha="center", va="bottom", fontsize=8, color="#E67E22", weight="bold")
    for bar in b_f13:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{int(h)}", ha="center", va="bottom", fontsize=8, color=COLOR_V2, weight="bold")

    ax1.text(0.5, 0.88, "* Note: PSM 13 was run on failed chunks as fallback;\nstand-alone run on all 33 rule chunks was not measured.",
             ha="center", va="center", transform=ax1.transAxes, fontsize=7.5, color="#555555", style="italic",
             bbox=dict(facecolor="#F8F9FA", edgecolor="#CCCCCC", boxstyle="round,pad=0.3"))

    ax1.legend(loc="upper left", fontsize=8)

    # -------------------------------------------------------------
    # Panel 2: Character Error Rate (Comparing Like with Like)
    # -------------------------------------------------------------
    groups = [
        "Full Line\n(All 33 Lines)\n[Full Line unit]",
        "Full Line\n(Clean 23 Lines)\n[Full Line unit]",
        "Rule Chunk\n(All 33 Lines)\n[Chunk unit]",
        "Rule Chunk\n(Clean 23 Lines)\n[Chunk unit]"
    ]
    
    x2 = np.arange(len(groups))
    w2 = 0.35
    
    psm7_cers = [8.42, 1.07, 13.02, 1.26]
    psm13_cers = [0.09, 0.00, np.nan, np.nan]
    
    b3 = ax2.bar(x2 - w2/2, psm7_cers, width=w2, label="PSM 7 CER (%)", color="#E67E22")
    b4 = ax2.bar(x2 + w2/2, [0 if np.isnan(v) else v for v in psm13_cers], width=w2, label="PSM 13 CER (%)", color=COLOR_V2)
    
    ax2.set_xticks(x2)
    ax2.set_xticklabels(groups, fontsize=8.2)
    ax2.set_ylabel("Tesseract Character Error Rate (CER %)", fontsize=9.5)
    ax2.set_title("Comparing Like with Like (Same Evaluation Unit)\nClean Subset vs. All 33 Lines", fontsize=10, weight="bold")
    ax2.set_ylim(0, 16.5)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    
    for idx, bar in enumerate(b3):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color="#D35400", weight="bold")
        
    for idx, bar in enumerate(b4):
        if np.isnan(psm13_cers[idx]):
            bar.set_visible(False)
            ax2.text(bar.get_x() + bar.get_width()/2, 0.8, "not measured\nin metrics.json", ha="center", va="bottom", 
                     fontsize=7.5, color="#888888", style="italic", rotation=90)
        else:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, color=COLOR_V2, weight="bold")

    ax2.legend(loc="upper right", fontsize=8.5)

    # Footnote & disclosures
    fig.text(0.04, 0.015,
             "Source: results/real_fix9/metrics.json:25-46, 246-265, 289-299, results/real_fix9/empty_outputs_log.json\n"
             "Methodological Note: PSM 13 was evaluated post-hoc after PSM 7 returned empty outputs. PSM 13 bypasses page layout analysis and treats crops as raw lines.\n"
             "Rule chunk PSM 13 across all 33 lines is absent from metrics.json (hence omitted / marked 'not measured'). Comparing like with like reveals clean full-line parity (1.07% vs 0.00%).",
             fontsize=7.5, color="#555555", style="italic")
             
    fig.savefig(FIG_DIR / "f13_tess_psm_comparison_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("F13_v2 generated successfully.")


if __name__ == "__main__":
    make_f8_v2()
    make_f9_v2()
    make_f11_v2()
    make_f13_v2()
    print("All 4 requested _v2 figures successfully created!")
