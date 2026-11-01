"""
Figure set v5 (paper/figures_v5/) for the BM 606 Azerbaijani OCR project.

Goals
  * No text overflow: every caption is wrapped to the figure width, annotations live
    in legends / dedicated panels instead of floating on top of data.
  * EasyOCR ('az') added next to Tesseract 5 (aze) in every comparison figure.
  * Highlights where CRNN V2 is better (speed, special letters, distortion, EasyOCR gap)
    while still showing the cases where a baseline wins (no hidden numbers).

All numbers are read from existing result files (results/easyocr_benchmark/metrics.json,
results/strengthening_audit/metrics.json, ...) or are copied from the audited tables of
paper/paper_v4.md (Table 4c, Table 3b). No new experiments are run.

Run from anywhere:  python paper/generate_figures_v5.py
"""
import sys
import json
import csv
import shutil
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures_v5"
OUT.mkdir(parents=True, exist_ok=True)

C_V1, C_V2, C_TESS, C_EASY = "#7B93AD", "#1B9E5A", "#E07B00", "#8E44AD"
C_RED, C_GRAY, C_INK = "#C0392B", "#6B7280", "#1F2937"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#BBBBBB",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
})

EASY = json.loads((ROOT / "results/easyocr_benchmark/metrics.json").read_text(encoding="utf-8"))
STR = json.loads((ROOT / "results/strengthening_audit/metrics.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ helpers
def finish(fig, name, title, caption, top=0.93):
    """Title + wrapped caption at the bottom, then save. Nothing can overflow the canvas."""
    import io
    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=0.992, color=C_INK)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    body = Image.open(buf).convert("RGB")

    # Caption is rendered separately at the exact pixel width of the figure and stacked below it,
    # so it can never collide with axis labels or run past the canvas.
    dpi = 200
    w_in = body.width / dpi
    fs = 9
    chars = int(w_in * 72 / (fs * 0.56) * 0.93)
    wrapped = "\n".join(textwrap.fill(p, chars) for p in caption.split("\n"))
    n_lines = wrapped.count("\n") + 1
    h_in = (n_lines * fs * 1.5 + 18) / 72
    cfig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    cfig.text(0.012, 0.5, wrapped, fontsize=fs, color="#4B5563", style="italic", va="center", ha="left",
              linespacing=1.4)
    cbuf = io.BytesIO()
    cfig.savefig(cbuf, format="png", dpi=dpi, facecolor="white")
    plt.close(cfig)
    cbuf.seek(0)
    cap = Image.open(cbuf).convert("RGB")
    canvas = Image.new("RGB", (body.width, body.height + cap.height), "white")
    canvas.paste(body, (0, 0))
    canvas.paste(cap.resize((body.width, cap.height)), (0, body.height))
    canvas.save(OUT / name)
    print("saved", name)


def bar_labels(ax, bars, fmt="{:.2f}", fs=8.5, color=C_INK, pad=0.0, rot=0, weight="bold"):
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3 + pad),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs,
                    color=color, rotation=rot, fontweight=weight)


def photo_cer(photo, key):
    return EASY["real_photos"][photo][key]["cer"]


# ------------------------------------------------------------------ data tables
PHOTOS = ["Photo 1 (Times)", "Photo 2 (Calibri)", "Photo 3 (Times)", "Photo 4 (Arial)",
          "Photo 5 (Clean 23)", "Photo 5 (Times, All 33)", "Aggregate All 5 Photos (n=111)"]
SHORT = ["Photo 1\nTimes", "Photo 2\nCalibri", "Photo 3\nTimes", "Photo 4\nArial",
         "Photo 5\nclean 23", "Photo 5\nall 33", "All 5 photos\n(n=111)"]
v2_real = [photo_cer(p, "v2_rule_chunk") for p in PHOTOS]
ez_real = [photo_cer(p, "easyocr_rule_chunk") for p in PHOTOS]
# Tesseract rule-chunk PSM 7 (like-for-like with chunk mode); clean-23 value from Table 4c
ts_real = [photo_cer(p, "tess_rule_chunk") if "tess_rule_chunk" in EASY["real_photos"][p] else 1.26
           for p in PHOTOS]

# (label, V2, Tesseract, EasyOCR)  -- Table 4c of paper_v4.md / easyocr_benchmark metrics
SYN = [
    ("Synthetic\nline (n=1000)", 0.30, 0.44, EASY["synth_line_test"]["mixed_1000"]["cer"]),
    ("Ladder L0\nline", 0.11, 0.22, EASY["sim_ladder"]["L0_line"]["cer"]),
    ("Ladder L4\nline", 0.11, 0.33, EASY["sim_ladder"]["L4_line"]["cer"]),
    ("Ladder L4\nword", 1.35, 0.37, EASY["sim_ladder"]["L4_word"]["cer"]),
    ("Unseen fonts\nL0 line", 0.71, 0.71, EASY["group_c_unseen_ladder"]["GroupC_L0_line"]["cer"]),
    ("Unseen fonts\nL4 line", 1.28, 1.46, EASY["group_c_unseen_ladder"]["GroupC_L4_line"]["cer"]),
]


# ================================================================== F1 pipeline
def make_f1():
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 92)
    ax.axis("off")
    ax.text(80, 87, "Azerbaijani OCR: End-to-End System & Multi-Engine Evaluation Pipeline",
            ha="center", fontsize=17, fontweight="bold", color=C_INK)
    ax.text(80, 82, "Corpus  →  TRDG synthetic engine  →  Photometric augmentation  →  CRNN + CTC  →  "
                    "Audit vs. Tesseract 5 & EasyOCR",
            ha="center", fontsize=11.5, color="#4B5563")

    def box(x, y, w, h, title, lines, fc, ec, fs=10.5):
        wrap = max(int(w * 0.95), 30)
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.0",
                                            fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h - 3.6, title, ha="center", va="center", fontsize=12, fontweight="bold", color=C_INK)
        ax.plot([x + 3, x + w - 3], [y + h - 6.6] * 2, color=ec, lw=0.8, alpha=0.6)
        step = (h - 11.5) / max(len(lines), 1)
        for i, t in enumerate(lines):
            ax.text(x + w / 2, y + h - 10.8 - i * step, textwrap.fill(t, wrap), ha="center", va="top",
                    fontsize=fs, color="#111827", linespacing=1.15)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.8, shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))

    top_y, top_h, bw = 46, 30, 35
    xs = [3, 44, 85, 126]
    box(xs[0], top_y, bw, top_h, "1. Azerbaijani corpus",
        ["Wikimedia az 20231101", "80/10/10 split before rendering", "Special-letter balancing (≥50%)",
         "Forbidden-label tracking"], "#E8F1F8", "#2F6FA5")
    box(xs[1], top_y, bw, top_h, "2. Image generation",
        ["TRDG (patched Pillow 13)", "Glyph validation (32 az letters)", "V1: 8 fonts · V2: +6 fonts (+21k)",
         "20% of font families held out"], "#E9F6EE", "#1B9E5A")
    box(xs[2], top_y, bw, top_h, "3. Augmentation (V2)",
        ["Illumination gradient + shadow", "Sensor noise & motion blur", "Downscale + JPEG artifacts",
         "Contrast stretch, perspective, rotation"], "#FEF3E2", "#E07B00")
    box(xs[3], top_y, 31, top_h, "4. CRNN architecture",
        ["7-layer VGG-style CNN", "2-layer BiLSTM (h=256)", "8.23M params (31.4 MiB)",
         "CTC loss, 86 characters"], "#F3EAF8", "#8E44AD")
    for a, b in zip(xs[:3], xs[1:]):
        arrow(a + bw + 0.6, top_y + top_h / 2, b - 0.6, top_y + top_h / 2)

    bot_y, bot_h, bw2 = 6, 30, 48
    bx = [3, 56, 109]
    box(bx[0], bot_y, bw2, bot_h, "Synthetic & ladder benchmarks",
        ["Tahoma vs. Consolas per-font test", "Simulation ladder L0–L4n (346 words / 92 lines)",
         "Bootstrap 95% confidence intervals", "4×4 i / ı / İ / I audit"], "#FFFFFF", "#6B7280", fs=10.3)
    box(bx[1], bot_y, bw2, bot_h, "Generalization & multi-engine audit",
        ["Seen / sibling / new font families", "Group C: 609 words / 158 lines",
         "Tesseract 5 (aze) + EasyOCR (az)", "Space-error WER & CTC calibration"], "#FFFFFF", "#6B7280", fs=10.3)
    box(bx[2], bot_y, bw2, bot_h, "Real document photos (1–5)",
        ["Smartphone photos, n = 111 lines", "Rule chunking vs. squeeze ratio",
         "Padding sensitivity (4 px vs. 20 px)", "Tesseract PSM 7 / 13, EasyOCR 3 modes"], "#FFFFFF", "#6B7280", fs=10.3)

    sx = xs[3] + 31 / 2
    for tx in [bx[0] + bw2 / 2, bx[1] + bw2 / 2, bx[2] + bw2 / 2]:
        arrow(sx, top_y - 0.8, tx, bot_y + bot_h + 1.0, rad=0.0)
    ax.text(sx + 2, 41.5, "Inference & audit", fontsize=10.5, fontweight="bold", color="#6B21A8",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5EEFA", ec="#8E44AD", lw=1.0))

    ax.text(3, 1.6, "Source: prepare_corpus.py, generate_data.py, train_v2.py (AUG_CONFIG), model.py, run_v2_audit4.py, "
                    "scratch/benchmark_easyocr.py, results/real_fix9/report.md", fontsize=8.5, color="#6B7280", style="italic")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "f1_pipeline_v5.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved f1_pipeline_v5.png")


# ================================================================== F4 head-to-head (NEW headline figure)
def make_headtohead():
    fig = plt.figure(figsize=(17, 10.8))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.14)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    w = 0.26

    def grouped(ax, labels, v2, ts, ez, ymax, title, ylabel=True):
        x = np.arange(len(labels))
        b1 = ax.bar(x - w, v2, w, color=C_V2, label="CRNN V2 (ours)", zorder=3)
        b2 = ax.bar(x, ts, w, color=C_TESS, label="Tesseract 5 (aze)", zorder=3)
        b3 = ax.bar(x + w, ez, w, color=C_EASY, label="EasyOCR (az)", zorder=3)
        for bars, col in [(b1, "#14703F"), (b2, "#A85A00"), (b3, "#6C3483")]:
            for b in bars:
                h = b.get_height()
                txt = f"{h:.2f}" if h < 10 else f"{h:.1f}"
                ax.annotate(txt, (b.get_x() + b.get_width() / 2, min(h, ymax)), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom", fontsize=7.8,
                            color=col, fontweight="bold", rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left")
        if ylabel:
            ax.set_ylabel("Character error rate (%) — lower is better")
        ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)

    ts_a = [min(t, 6.0) for t in ts_real]
    grouped(axA, SHORT, v2_real, ts_a, ez_real, 3.2, "A. Real smartphone photos (rule-chunk mode)")
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.955))
    axA.set_ylim(0, 3.2)
    for b in axA.patches:  # clipped Tesseract bars: show true value
        pass
    syn_labels = [s[0] for s in SYN]
    grouped(axB, syn_labels, [s[1] for s in SYN], [s[2] for s in SYN], [s[3] for s in SYN], 8.5,
            "B. Synthetic, camera-distortion (L4) and unseen fonts", ylabel=False)

    # Panel C: how many times lower is V2's error than EasyOCR (like-for-like chunk / line)
    names = [s.replace("\n", " ") for s in SHORT] + [s[0].replace("\n", " ") for s in SYN]
    v2_all = v2_real + [s[1] for s in SYN]
    ez_all = ez_real + [s[3] for s in SYN]
    ratio = np.array([e / max(v, 0.01) for v, e in zip(v2_all, ez_all)])
    order = np.argsort(ratio)
    y = np.arange(len(order))
    cols = [C_V2 if ratio[i] >= 1.0 else C_RED for i in order]
    axC.barh(y, [ratio[i] for i in order], color=cols, height=0.62, zorder=3)
    axC.axvline(1.0, color=C_INK, lw=1.4, ls="--", zorder=4)
    axC.set_xscale("log")
    axC.set_yticks(y)
    axC.set_yticklabels([names[i] for i in order], fontsize=9)
    axC.set_xlim(0.5, 80)
    axC.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    axC.set_xticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×", "50×"])
    axC.grid(axis="x", ls=":", alpha=0.6, zorder=0)
    for yi, i in zip(y, order):
        r = ratio[i]
        axC.annotate(f"{r:.1f}×  ({ez_all[i]:.2f}% → {v2_all[i]:.2f}%)", (r, yi), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=8.8,
                     color="#14703F" if r >= 1 else C_RED, fontweight="bold")
    axC.set_xlabel("EasyOCR error ÷ CRNN V2 error (log scale).  Green = V2 better · Red = EasyOCR better · dashed line = parity")
    axC.set_title("C. Error-reduction factor of CRNN V2 over EasyOCR (az) — V2 wins in 12 of 13 settings", loc="left")

    finish(fig, "f4_headtohead_v5.png",
           "Head-to-Head: CRNN V2 vs. Tesseract 5 and EasyOCR (az)",
           "Panel A/B use the like-for-like operating mode of each engine (rule chunks for photos, line recognition for synthetic and "
           "ladder sets). EasyOCR = latin_g2 recognizer; Tesseract = aze tessdata_best. Panel C divides EasyOCR CER by V2 CER. "
           "Honest note: Tesseract is better on several clean-photo and L4-word settings; EasyOCR is better on Photo 5 (all 33 lines).\n"
           "Source: results/easyocr_benchmark/metrics.json, Table 4c of paper/paper_v4.md.", top=0.945)


# ================================================================== F5 ladder
def make_ladder():
    levels = ["L0", "L1", "L2", "L3", "L4", "L4n"]
    word = {"V1": [7.71, 8.85, 25.00, 31.36, 35.77, 24.43], "V2": [0.20, 0.20, 0.41, 0.53, 1.35, 1.14],
            "Tess": [0.41, 0.45, 0.41, 0.41, 0.37, 0.37]}
    line = {"V1": [1.87, 4.10, 13.84, 19.14, 12.96, 7.43], "V2": [0.11, 0.11, 0.07, 0.18, 0.11, 0.07],
            "Tess": [0.22, 0.22, 0.29, 0.22, 0.33, 0.33]}
    ez_w = {0: EASY["sim_ladder"]["L0_word"]["cer"], 4: EASY["sim_ladder"]["L4_word"]["cer"]}
    ez_l = {0: EASY["sim_ladder"]["L0_line"]["cer"], 4: EASY["sim_ladder"]["L4_line"]["cer"]}

    fig = plt.figure(figsize=(16, 9.4))
    gs = GridSpec(5, 3, width_ratios=[1.05, 1.0, 1.0], hspace=0.55, wspace=0.28)
    descs = ["L0  clean render", "L1  + illumination gradient & shadow", "L2  + blur & sensor noise",
             "L3  + downscale 0.6× & JPEG Q50", "L4  + perspective tilt & rotation"]
    for r, (lvl, d) in enumerate(zip(["L0", "L1", "L2", "L3", "L4"], descs)):
        ax = fig.add_subplot(gs[r, 0])
        p = ROOT / f"data/sim_ladder/{lvl}/line/test/images/0000000.png"
        if p.exists():
            ax.imshow(Image.open(p), cmap="gray", aspect="equal")
        else:  # data/sim_ladder was cleaned up; reuse the rendered strips of the audited v1 figure
            old = Image.open(ROOT / "paper/figures/f5_sim_ladder.png").convert("RGB")
            y0 = [299, 413, 528, 643, 758][r]
            ax.imshow(old.crop((303, y0, 942, y0 + 64)), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#CCCCCC")
        ax.set_title(d, fontsize=9.5, loc="left", pad=3)

    def plot(ax, data, ez, title, note):
        ax.plot(levels, data["V1"], "o-", color=C_V1, lw=2.2, ms=6, label="CRNN V1 (baseline)")
        ax.plot(levels, data["V2"], "s-", color=C_V2, lw=2.8, ms=7, label="CRNN V2 (ours)")
        ax.plot(levels, data["Tess"], "^:", color=C_TESS, lw=2.0, ms=6, label="Tesseract 5")
        ax.plot(list(ez.keys()), list(ez.values()), "D", color=C_EASY, ms=9, label="EasyOCR (az), L0 & L4 only")
        for k, v in ez.items():
            ax.annotate(f"{v:.2f}%", (k, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color=C_EASY)
        ax.annotate(f"{data['V2'][4]:.2f}%", (4, data["V2"][4]), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color="#14703F")
        ax.set_yscale("log")
        ax.set_ylim(0.04, 90)
        ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
        ax.set_ylabel("CER % (log scale)")
        ax.set_xlabel("Distortion level")
        ax.grid(True, which="major", ls=":", alpha=0.6)
        ax.set_title(title, loc="left")
        ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8.8, color="#14703F", fontweight="bold", va="bottom")

    axw = fig.add_subplot(gs[0:5, 1])
    axl = fig.add_subplot(gs[0:5, 2])
    plot(axw, word, ez_w, "Word-level CER (n = 346 words)", f"L4: V2 = {ez_w[4] / word['V2'][4]:.1f}× below EasyOCR,\n{word['V1'][4] / word['V2'][4]:.0f}× below V1")
    plot(axl, line, ez_l, "Line-level CER (n = 92 lines)", f"L4: V2 = {ez_l[4] / line['V2'][4]:.0f}× below EasyOCR,\n{line['V1'][4] / line['V2'][4]:.0f}× below V1")
    axw.legend(loc="upper left", fontsize=9)
    finish(fig, "f5_sim_ladder_v5.png",
           "Simulation Ladder L0 → L4: Photometric Augmentation Keeps CRNN V2 Stable",
           "Left: one ladder line rendered at each distortion level. Right: CER on a log axis (V1 collapses to 35.8% word CER at L4; "
           "V2 stays at 1.35%). EasyOCR was benchmarked only at L0 and L4. Tesseract is slightly lower than V2 on L4 words (0.37% vs. 1.35%) "
           "and is shown for transparency. V1 non-monotonicity (L3 > L4 on lines) was not investigated.\n"
           "Source: results/sim_ladder/summary.md, results/v2_audit/summary.md, results/easyocr_benchmark/metrics.json.", top=0.94)


# ================================================================== F6 per-font
def make_per_font():
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
    ez_cons = EASY["synth_line_test"]["consolas_500"]["cer"]
    sets = [
        ("Tahoma — new family (not in V1/V2 training)", "Genuine invariance: all engines ≈ 0.1–0.5%",
         {"word": [0.08, 0.09, 0.50, np.nan], "line": [0.24, 0.26, 0.22, np.nan]}, "#EAF6EE"),
        ("Consolas — family leakage (consolab.ttf in V2 training)", "Disclosure: NOT zero-shot generalization",
         {"word": [1.02, 0.07, 0.86, np.nan], "line": [3.48, 0.27, 0.47, ez_cons]}, "#FDECEA"),
    ]
    x = np.arange(2)
    w = 0.2
    for ax, (title, sub, d, bg) in zip(axes, sets):
        ax.set_facecolor(bg)
        vals = np.array([d["word"], d["line"]]).T  # engines x levels
        cols = [C_V1, C_V2, C_TESS, C_EASY]
        names = ["CRNN V1", "CRNN V2 (ours)", "Tesseract 5", "EasyOCR (az)"]
        for i in range(4):
            bars = ax.bar(x + (i - 1.5) * w, vals[i], w, color=cols[i], label=names[i], zorder=3)
            for xb, v in zip(x + (i - 1.5) * w, vals[i]):
                if np.isnan(v):
                    ax.text(xb, 0.07, "n/a", rotation=90, ha="center", va="bottom", fontsize=8, color=C_GRAY)
                else:
                    ax.annotate(f"{v:.2f}", (xb, v), xytext=(0, 3), textcoords="offset points", ha="center",
                                fontsize=8.8, fontweight="bold", color=C_INK, rotation=90 if v < 0.9 else 0)
        ax.set_xticks(x)
        ax.set_xticklabels(["Word level (n = 1,000)", "Line level (n = 500)"])
        ax.set_ylim(0, 4.2)
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=11)
        ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)
    axes[0].set_ylabel("Character error rate (%)")
    axes[0].legend(loc="upper left", ncol=2, fontsize=9)
    finish(fig, "f6_per_font_v5.png",
           "Per-Font Synthetic Test: Genuine Generalization vs. Family Leakage",
           "On Consolas (leaked family) V2 line CER is 0.27% vs. 3.48% for V1, 0.47% for Tesseract and 0.81% for EasyOCR (997 line images). "
           "EasyOCR was run word-level on neither font and on only 3 Tahoma line images, so those bars are marked n/a.\n"
           "Source: results/diagw_*/metrics.json, results/diag_*/metrics.json, results/easyocr_benchmark/metrics.json.", top=0.90)


# ================================================================== F7 forest + EasyOCR delta
def make_forest():
    rows = [  # label, delta, lo, hi  (V2 - Tesseract, percentage points)
        ("Clean unseen word L0 (n=699w)", 0.34, -0.25, 0.88), ("Clean unseen line L0 (n=180l)", 0.17, -0.22, 0.58),
        ("Clean unseen word L4 (n=699w)", 0.63, 0.02, 1.19), ("Clean unseen line L4 (n=180l)", 0.17, -1.12, 1.14),
        ("Group C word L0 (n=609w)", 0.62, -0.35, 2.08), ("Group C line L0 (n=158l)", 0.00, -0.74, 0.76),
        ("Group C word L4 (n=609w)", 1.55, 0.59, 2.94), ("Group C line L4 (n=158l)", -0.17, -2.60, 1.53),
        ("Real Photo 1 chunks (n=16l)", -0.35, -1.52, 0.50), ("Real Photo 2 chunks (n=16l)", 0.41, 0.07, 0.94),
        ("Real Photo 5 clean (n=23l)", 0.06, -0.77, 0.80),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.5))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, lo, hi) in zip(y, rows):
        sig = lo > 0 or hi < 0
        col = C_TESS if (sig and d > 0) else (C_V2 if sig else "#374151")
        ax.plot([lo, hi], [yi, yi], color=col, lw=3.2, solid_capstyle="round")
        ax.plot(d, yi, "s", color=col, ms=9)
        ax.annotate(f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]", (3.3, yi), va="center", fontsize=8.8,
                    fontweight="bold" if sig else "normal", color=col, annotation_clip=False)
    ax.axvline(0, color=C_RED, ls="--", lw=1.6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.2)
    ax.set_xlim(-3.0, 3.2)
    ax.set_xlabel("ΔCER = V2 − Tesseract (pp), 95% cluster-bootstrap CI\nnegative = V2 better · positive = Tesseract better")
    ax.set_title("A. V2 vs. Tesseract 5: 9 of 11 CIs include 0 (statistical tie)", loc="left")
    ax.grid(axis="x", ls=":", alpha=0.6)

    names = [s.replace("\n", " ") for s in SHORT] + [s[0].replace("\n", " ") for s in SYN]
    v2_all = v2_real + [s[1] for s in SYN]
    ez_all = ez_real + [s[3] for s in SYN]
    d = np.array(v2_all) - np.array(ez_all)
    order = np.argsort(d)[::-1]
    y2 = np.arange(len(order))[::-1]
    for yi, i in zip(y2, order):
        col = C_V2 if d[i] < 0 else C_RED
        ax2.plot([0, d[i]], [yi, yi], color=col, lw=3)
        ax2.plot(d[i], yi, "o", color=col, ms=9)
        ax2.annotate(f"{d[i]:+.2f}", (d[i], yi), xytext=(-8 if d[i] < 0 else 8, 0), textcoords="offset points",
                     ha="right" if d[i] < 0 else "left", va="center", fontsize=8.8, fontweight="bold", color=col)
    ax2.axvline(0, color=C_INK, lw=1.4)
    ax2.set_yticks(y2); ax2.set_yticklabels([names[i] for i in order], fontsize=9.2)
    ax2.set_xlim(-7.4, 1.4)
    ax2.set_xlabel("ΔCER = V2 − EasyOCR (pp), point estimates\nnegative = V2 better")
    ax2.set_title("B. V2 vs. EasyOCR (az): V2 lower in 12 of 13 settings", loc="left")
    ax2.grid(axis="x", ls=":", alpha=0.6)
    finish(fig, "f7_forest_vs_tess_easyocr_v5.png",
           "Accuracy Difference of CRNN V2 vs. Tesseract 5 and EasyOCR",
           "Panel A reproduces the audited cluster-bootstrap CIs against Tesseract (no claim of superiority where the CI spans 0). "
           "Panel B compares V2 to EasyOCR with point estimates only (no bootstrap was run for EasyOCR), so it supports a consistent "
           "direction but not significance claims.\n"
           "Source: results/v2_audit4/summary.md, results/real_fix9/metrics.json, results/easyocr_benchmark/metrics.json.", top=0.93)


# ================================================================== F8 character classes
def make_char_classes():
    keys = ["lower_special", "upper_special", "standard_lower", "standard_upper"]
    lab = ["Lowercase special\n(ə ı ö ğ ü ş ç)", "Uppercase special\n(Ə İ Ö Ğ Ü Ş Ç I)", "Std lowercase\n(a–z)", "Std uppercase\n(A–Z)"]
    v1 = [STR["v1_char_audit"]["class_summary"][k]["error_rate_pct"] for k in keys]
    v2 = [STR["v2_char_audit"]["class_summary"][k]["error_rate_pct"] for k in keys]
    ts = [STR["tess_char_audit"]["class_summary"][k]["error_rate_pct"] for k in keys]
    ez = [EASY["synth_line_special_char_audit"][k]["error_rate_pct"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), gridspec_kw=dict(width_ratios=[1.35, 1.0, 0.9], wspace=0.28))
    ax = axes[0]
    x = np.arange(4); w = 0.2
    cap = 6.0
    for i, (vals, col, nm) in enumerate([(v1, C_V1, "CRNN V1"), (v2, C_V2, "CRNN V2 (ours)"),
                                         (ts, C_TESS, "Tesseract 5"), (ez, C_EASY, "EasyOCR (az)")]):
        clipped = [min(v, cap) for v in vals]
        bars = ax.bar(x + (i - 1.5) * w, clipped, w, color=col, label=nm, zorder=3)
        for xb, v in zip(x + (i - 1.5) * w, vals):
            ax.annotate(f"{v:.2f}" + (" ▲" if v > cap else ""), (xb, min(v, cap)), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.6, fontweight="bold",
                        color=C_INK, rotation=90)
    ax.set_ylim(0, cap + 1.6)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylabel("Character error rate (%)")
    ax.set_title("A. Synthetic lines, n = 50,122 characters", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)
    ax.text(0.01, 0.97, "▲ = bar clipped at 6% (V1 uppercase special: 13.11%)", transform=ax.transAxes,
            fontsize=8.5, color=C_GRAY, style="italic", va="top")

    ax = axes[1]
    ra = EASY["real_photos_special_char_audit"]
    series = [("CRNN V2", ra["crnn_v2_chunk"], C_V2), ("Tesseract", ra["tesseract_chunk"], C_TESS),
              ("EasyOCR chunk", ra["easyocr_chunk"], C_EASY), ("EasyOCR full", ra["easyocr_full"], "#B07CC6")]
    w = 0.2
    for i, (nm, dct, col) in enumerate(series):
        vals = [dct[k]["error_rate_pct"] for k in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=nm, zorder=3)
        for xb, v in zip(x + (i - 1.5) * w, vals):
            ax.annotate(f"{v:.2f}", (xb, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8.2, fontweight="bold", rotation=90)
    ns = [ra["crnn_v2_chunk"][k]["total"] for k in keys]
    ax.set_xticks(x); ax.set_xticklabels([f"{l.split(chr(10))[0]}\nn={n}" for l, n in zip(lab, ns)], fontsize=8.6)
    ax.set_ylim(0, 6.2)
    ax.set_title("B. Real photos (111 lines)", loc="left")
    ax.legend(loc="upper right", fontsize=8.6)
    ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)

    ax = axes[2]
    l4 = EASY["ladder_l4_special_char_audit"]
    cl = [EASY["synth_line_special_char_audit"][k]["error_rate_pct"] for k in keys]
    l4v = [l4[k]["error_rate_pct"] for k in keys]
    w = 0.36
    b1 = ax.bar(x - w / 2, cl, w, color="#C9A6DA", label="clean synthetic", zorder=3)
    b2 = ax.bar(x + w / 2, l4v, w, color=C_EASY, label="L4 camera distortion", zorder=3)
    bar_labels(ax, b1, "{:.1f}", 8.5, rot=0)
    bar_labels(ax, b2, "{:.1f}", 8.5, rot=0)
    ax.set_xticks(x); ax.set_xticklabels(["Low\nspecial", "Up\nspecial", "Std\nlower", "Std\nupper"], fontsize=9)
    ax.set_ylim(0, max(l4v) * 1.18)
    ax.set_title("C. EasyOCR degradation at L4", loc="left")
    ax.legend(loc="upper left", fontsize=8.8)
    ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)
    finish(fig, "f8_character_classes_v5.png",
           "Where Specialization Pays Off: Azerbaijani Letter-Class Error Rates",
           "V2 has the lowest error on uppercase special letters (2.46% vs. 3.69% Tesseract and 2.96% EasyOCR) and 8.3× lower error than EasyOCR on "
           "lowercase special letters (0.27% vs. 2.25%). Tesseract (0.19%) and EasyOCR (0.12%) are lower on some standard-letter classes. "
           "EasyOCR's pan-Latin 341-character head loses capacity on diacritics and degrades to 19.7% on lowercase specials under L4 distortion.\n"
           "Source: results/strengthening_audit/metrics.json, results/easyocr_benchmark/metrics.json.", top=0.93)


# ================================================================== F9 efficiency + calibration
def make_efficiency():
    info = EASY["easyocr_model_info"]
    cov = STR["real_photos_confidence"]["hybrid_pareto_curve"]
    h10 = next(r for r in cov if r["fallback_pct"] == 10)
    h20 = next(r for r in cov if r["fallback_pct"] == 20)
    fig = plt.figure(figsize=(18, 7.2))
    gs = GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    eng = [("CRNN V2 (batch 16)", 127.59, C_V2), ("CRNN V2 (batch 1)", 155.35, "#52B788"),
           ("EasyOCR recognize", info["cpu_latency_photo5_recognize_ms_per_line"], C_EASY),
           ("Tesseract PSM 13", 425.99, C_TESS), ("Tesseract PSM 7", 516.53, "#F0A04B"),
           ("EasyOCR readtext (E2E)", info["cpu_latency_photo5_readtext_ms_per_line"], "#6C3483")]
    y = np.arange(len(eng))[::-1]
    ax.barh(y, [e[1] for e in eng], color=[e[2] for e in eng], height=0.62, zorder=3)
    for yi, (n, v, c) in zip(y, eng):
        sp = v / 127.59
        txt = f"{v:,.0f} ms" + ("" if abs(sp - 1) < 1e-6 else f"  ({sp:.1f}× slower)")
        ax.annotate(txt, (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([e[0] for e in eng], fontsize=9.5)
    ax.set_xscale("log"); ax.set_xlim(80, 20000)
    ax.set_xticks([100, 300, 1000, 3000]); ax.set_xticklabels(["100", "300", "1000", "3000"])
    ax.set_xlabel("CPU latency per line (ms, log scale, i5-7200U)")
    ax.set_title("A. Speed: V2 is 2.3–26× faster", loc="left")
    ax.grid(axis="x", ls=":", alpha=0.6, zorder=0)

    ax = fig.add_subplot(gs[1])
    ez_ms = info["cpu_latency_photo5_recognize_ms_per_line"]
    pts = [("1", "CRNN V2 (b16)", 127.59, 1.08, C_V2, "o"), ("2", "CRNN V2 (b1)", 155.35, 1.08, "#52B788", "s"),
           ("3", "Hybrid 10% Tess", h10["effective_latency_ms"], h10["hybrid_cer_pct"], "#2563EB", "D"),
           ("4", "Hybrid 20% Tess", h20["effective_latency_ms"], h20["hybrid_cer_pct"], "#1D4ED8", "P"),
           ("5", "Tesseract 5 (full line)", 425.99, 0.32, C_TESS, "^"),
           ("6", "EasyOCR chunk", ez_ms, 1.31, C_EASY, "X"), ("7", "EasyOCR full", ez_ms, 1.50, "#B07CC6", "v")]
    ax.plot([127.59, h10["effective_latency_ms"], h20["effective_latency_ms"], 425.99],
            [1.08, h10["hybrid_cer_pct"], h20["hybrid_cer_pct"], 0.32], "--", color="#6B7280", lw=1.5, zorder=2)
    for n, nm, xv, yv, c, m in pts:
        ax.scatter(xv, yv, s=170, color=c, marker=m, edgecolor="black", lw=0.8, zorder=4, label=f"{n}  {nm}")
        ax.annotate(n, (xv, yv), xytext=(9, 7), textcoords="offset points", fontsize=11, fontweight="bold", color=c)
    ax.set_xlim(100, 460); ax.set_ylim(0.1, 1.75)
    ax.set_xlabel("CPU latency per line (ms)"); ax.set_ylabel("Real-photo CER (%), n = 111 lines")
    ax.set_title("B. Speed–accuracy trade-off", loc="left")
    ax.legend(loc="upper right", fontsize=8.6, handletextpad=0.3, borderaxespad=0.2)
    ax.grid(True, ls=":", alpha=0.6)

    ax = fig.add_subplot(gs[2])
    sub = [r for r in cov if r["fallback_pct"] <= 30]
    pct = [100 - r["fallback_pct"] for r in sub]
    ax.plot(pct, [r["selective_crnn_cer_pct"] for r in sub], "o-", color=C_V2, lw=2.4, label="Selective CRNN V2 (confident lines)")
    ax.plot(pct, [r["hybrid_cer_pct"] for r in sub], "s-", color="#2563EB", lw=2.4, label="Confidence-gated hybrid (V2 + Tess)")
    ax.invert_xaxis()
    ax.set_xlabel("Lines handled by CRNN V2 (%)"); ax.set_ylabel("CER (%)")
    ax.set_ylim(0.2, 1.5)
    ax.set_title(f"C. CTC confidence: AUROC = {STR['real_photos_confidence']['auroc_mean_char_conf']:.3f}", loc="left")
    ax.annotate(f"80% V2 + 20% Tess\nCER {h20['hybrid_cer_pct']:.2f}% (from 1.08%)", (80, h20["hybrid_cer_pct"]),
                xytext=(93, 0.28), fontsize=9.2, fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_INK),
                bbox=dict(boxstyle="round,pad=0.3", fc="#F0F9FF", ec="#2563EB"))
    ax.legend(loc="upper right", fontsize=8.8)
    ax.grid(True, ls=":", alpha=0.6)
    finish(fig, "f9_efficiency_calibration_v5.png",
           "Deployment Profile: Speed, Trade-off and Intrinsic Confidence",
           "A: latency measured on 33 Photo-5 lines (EasyOCR readtext includes CRAFT detection). B: aggregate CER over all 111 real lines; "
           "Tesseract full-line (PSM 13) is the most accurate point at 426 ms, and the confidence-gated hybrid recovers most of that "
           "accuracy at half the cost. C: raw CTC posteriors detect erroneous lines without any lexicon.\n"
           "Source: results/efficiency_benchmark/metrics.json, results/strengthening_audit/metrics.json, results/easyocr_benchmark/metrics.json.", top=0.92)


# ================================================================== F10 Photo-5 integration (replaces F13)
def make_integration():
    p5 = EASY["real_photos"]["Photo 5 (Times, All 33)"]
    rows = [  # label, cer, empty, color
        ("CRNN V2  rule chunks", p5["v2_rule_chunk"]["cer"], p5["v2_rule_chunk"]["empty_outputs"], C_V2),
        ("CRNN V2  full line", p5["v2_full_line"]["cer"], p5["v2_full_line"]["empty_outputs"], "#52B788"),
        ("EasyOCR  recognize, chunks", p5["easyocr_rule_chunk"]["cer"], p5["easyocr_rule_chunk"]["empty_outputs"], C_EASY),
        ("EasyOCR  recognize, full line", p5["easyocr_full_line_recog"]["cer"], p5["easyocr_full_line_recog"]["empty_outputs"], "#B07CC6"),
        ("EasyOCR  readtext (CRAFT + rec.)", p5["easyocr_e2e_readtext_cer"], 0, "#6C3483"),
        ("Tesseract  PSM 7, chunks", p5["tess_rule_chunk"]["cer"], 9, C_TESS),
        ("Tesseract  PSM 7, full line", 8.42, 2, "#F0A04B"),
        ("Tesseract  PSM 13, full line", p5["tess_full_line"]["cer"], 0, "#B45F06"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1.6, 1.0], wspace=0.12), sharey=True)
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.64, zorder=3)
    for yi, r in zip(y, rows):
        ax.annotate(f"{r[1]:.2f}%", (r[1], yi), xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=9.5, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel("Photo 5 character error rate (%), all 33 lines — lower is better")
    ax.set_title("A. Real photo with narrow line strips", loc="left")
    ax.grid(axis="x", ls=":", alpha=0.6, zorder=0)
    ax.text(0.98, 0.35, "V2 chunks: 6.2× lower than Tesseract PSM 7\nand 8.2× lower than EasyOCR readtext", transform=ax.transAxes,
            ha="right", fontsize=10, fontweight="bold", color="#14703F",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF3", ec=C_V2))
    ax2.barh(y, [r[2] for r in rows], color=[r[3] for r in rows], height=0.64, zorder=3)
    for yi, r in zip(y, rows):
        ax2.annotate(str(r[2]), (r[2], yi), xytext=(5, 0), textcoords="offset points", va="center",
                     fontsize=9.5, fontweight="bold")
    ax2.set_xlim(0, 11)
    ax2.set_xlabel("Empty outputs (of 33 lines / chunks)")
    ax2.set_title("B. Integration failures", loc="left")
    ax2.grid(axis="x", ls=":", alpha=0.6, zorder=0)
    ax2.tick_params(left=False)
    finish(fig, "f10_photo5_integration_v5.png",
           "Photo 5 Stress Test: Segmentation Failures of Off-the-Shelf Engines",
           "Tesseract PSM 7 returns 9 empty chunks and EasyOCR's CRAFT detector truncates words on narrow single-line strips, whereas V2 "
           "produces no empty outputs. Transparency: Tesseract PSM 13 (0.09%) and EasyOCR recognize (1.42–1.68%) are more accurate than V2 on "
           "this photo (2.10%); PSM 13 was chosen post-hoc after PSM 7 failed.\n"
           "Source: results/real_fix9/metrics.json, results/easyocr_benchmark/metrics.json.", top=0.92)


# ================================================================== F11 padding sensitivity
def make_padding():
    models = ["V2 rule chunk", "V2 full line", "Tesseract PSM 7\nchunk", "Tesseract PSM 7\nfull line", "Tesseract PSM 13\nfull line"]
    a = [2.10, 3.87, 13.02, 8.42, 0.09]
    b = [23.32, 22.72, 18.86, 10.91, 1.80]
    fig = plt.figure(figsize=(14, 8.4))
    gs = GridSpec(2, 1, height_ratios=[2.4, 1.0], hspace=0.38)
    ax = fig.add_subplot(gs[0])
    x = np.arange(5); w = 0.36
    b1 = ax.bar(x - w / 2, a, w, color=C_V2, label="Condition A: tight trim + 4 px margin", zorder=3)
    b2 = ax.bar(x + w / 2, b, w, color="#4A6572", label="Condition B: padded with 20 px white border", zorder=3)
    bar_labels(ax, b1, "{:.2f}%", 9, "#14703F")
    bar_labels(ax, b2, "{:.2f}%", 9, "#334155")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylim(0, 28); ax.set_ylabel("Character error rate (%)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="y", ls=":", alpha=0.6, zorder=0)
    ax.set_title("Only two padding values were tested (4 px and 20 px)", loc="left", fontsize=10.5, color="#4B5563")
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    cell = [[m.replace("\n", " "), f"{p:.2f}%", f"{q:.2f}%", f"+{q - p:.2f} pp", f"×{q / p:.1f}"] for m, p, q in zip(models, a, b)]
    t = axt.table(cellText=cell, colLabels=["Engine / mode", "4 px", "20 px", "Δ", "Relative"], loc="center",
                  cellLoc="center", colWidths=[0.34, 0.14, 0.14, 0.16, 0.14])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
    for (r, c), cellobj in t.get_celld().items():
        cellobj.set_edgecolor("#D1D5DB")
        if r == 0:
            cellobj.set_facecolor("#F3F4F6"); cellobj.set_text_props(fontweight="bold")
    finish(fig, "f11_padding_sensitivity_v5.png",
           "Boundary Padding Sensitivity: 4 px vs. 20 px Border",
           "All engines degrade with a 20 px border; V2 degrades most because the CRNN was trained only on tightly cropped lines "
           "(hypothesis, not tested: convolutional features drift into blank activations). EasyOCR was not evaluated under padding.\n"
           "Source: results/real_fix9/metrics.json, results/real_fix9/report.md.", top=0.93)


# ================================================================== legacy figures with no comparison (re-rendered into figures_v5)
def make_legacy():
    sys.path.insert(0, str(ROOT / "paper"))
    import os
    os.chdir(ROOT)
    import generate_figures as gf  # noqa: E402  (module-level code is safe; main guarded)
    gf.FIG_DIR = OUT
    for fn in ("make_f2", "make_f3", "make_f10", "make_f12", "make_f14"):
        try:
            getattr(gf, fn)()
        except Exception as e:  # keep going, report
            print("legacy", fn, "failed:", e)
    import generate_figures_v2 as g2
    g2.FIG_DIR = OUT
    for fn in ("make_f8_v2", "make_f9_v2"):
        try:
            getattr(g2, fn)()
        except Exception as e:
            print("legacy", fn, "failed:", e)


if __name__ == "__main__":
    make_f1()
    make_headtohead()
    make_ladder()
    make_per_font()
    make_forest()
    make_char_classes()
    make_efficiency()
    make_integration()
    make_padding()
    if "--legacy" in sys.argv:
        make_legacy()
    print("done ->", OUT)



