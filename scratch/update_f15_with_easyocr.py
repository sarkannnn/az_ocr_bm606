"""Update Figure 15 (paper/figures/f15_efficiency_and_calibration.png) to include EasyOCR (az)
in Panel A (CPU Speed-Accuracy Pareto Frontier across all 111 real document lines)
and Panel B (Character Class Error Rate Comparison).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STRENGTH_JSON = ROOT / "results" / "strengthening_audit" / "metrics.json"
EASY_JSON = ROOT / "results" / "easyocr_benchmark" / "metrics.json"
FIG15_PATH = ROOT / "paper" / "figures" / "f15_efficiency_and_calibration.png"


def main():
    s_data = json.loads(STRENGTH_JSON.read_text(encoding="utf-8"))
    e_data = json.loads(EASY_JSON.read_text(encoding="utf-8"))

    v1_char = s_data["v1_char_audit"]["class_summary"]
    v2_char = s_data["v2_char_audit"]["class_summary"]
    t_char = s_data["tess_char_audit"]["class_summary"]
    e_char = e_data["synth_line_special_char_audit"]

    ctc_cal = s_data["real_photos_confidence"]
    cov = ctc_cal["hybrid_pareto_curve"]
    auroc_min = ctc_cal["auroc_min_char_conf"]
    auroc_mean = ctc_cal["auroc_mean_char_conf"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9), dpi=300)

    # Panel A: CPU Latency (ms/line) vs Real-Photo Aggregate CER (%) across all 5 photos (n=111)
    ax0 = axes[0]
    crnn_b16_ms = 127.59
    crnn_b1_ms = 155.35
    tess_p13_ms = 425.99
    easy_rec_ms = e_data["easyocr_model_info"]["cpu_latency_photo5_recognize_ms_per_line"]  # 362.78 ms

    h10 = next(r for r in cov if r["fallback_pct"] == 10)
    h20 = next(r for r in cov if r["fallback_pct"] == 20)
    h25 = next(r for r in cov if r["fallback_pct"] == 25)

    ax0.plot([crnn_b16_ms, h10["effective_latency_ms"], h20["effective_latency_ms"], h25["effective_latency_ms"], tess_p13_ms],
             [1.08, h10["hybrid_cer_pct"], h20["hybrid_cer_pct"], h25["hybrid_cer_pct"], 0.32],
             linestyle="--", color="#475569", linewidth=1.6, zorder=2, label="CRNN+Tess Pareto Frontier")

    pts = [
        ("CRNN V2 (batch=16)\n128 ms | 1.08% CER", crnn_b16_ms, 1.08, "#16a34a", "o", 95, (-10, 22)),
        ("CRNN V2 (batch=1)\n155 ms | 1.08% CER", crnn_b1_ms, 1.08, "#22c55e", "s", 80, (14, -26)),
        (f"Hybrid (10% Tess)\n{h10['effective_latency_ms']:.0f} ms | {h10['hybrid_cer_pct']:.2f}% CER", h10["effective_latency_ms"], h10["hybrid_cer_pct"], "#0284c7", "D", 80, (12, 4)),
        (f"Hybrid (20% Tess)\n{h20['effective_latency_ms']:.0f} ms | {h20['hybrid_cer_pct']:.2f}% CER", h20["effective_latency_ms"], h20["hybrid_cer_pct"], "#2563eb", "P", 105, (-15, -36)),
        ("Tesseract 5 (Full)\n426 ms | 0.32% CER", tess_p13_ms, 0.32, "#dc2626", "^", 95, (-95, 18)),
        (f"EasyOCR ['az'] (Chunk)\n{easy_rec_ms:.0f} ms | 1.31% CER", easy_rec_ms, 1.31, "#9333ea", "X", 95, (-65, -34)),
        (f"EasyOCR ['az'] (Full)\n{easy_rec_ms:.0f} ms | 1.50% CER", easy_rec_ms, 1.50, "#a855f7", "v", 85, (-65, 14)),
    ]
    for lbl, x, y, col, mk, sz, offset in pts:
        ax0.scatter([x], [y], c=col, marker=mk, s=sz, zorder=4, edgecolors="black", linewidths=0.7)
        ax0.annotate(lbl, (x, y), xytext=offset, textcoords="offset points",
                     fontsize=7.3, fontweight="bold", color=col,
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.7),
                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, alpha=0.92, lw=0.7))

    ax0.set_title("A. CPU Speed–Accuracy Trade-off (Photos 1–5, n=111)")
    ax0.set_xlabel("CPU Inference Latency per Line (ms, lower is faster)")
    ax0.set_ylabel("Aggregate Real-Photo CER (%, lower is better)")
    ax0.set_xlim(95, 485)
    ax0.set_ylim(0.10, 1.72)
    ax0.grid(True, linestyle=":", alpha=0.6)

    # Panel B: Character Error Rate by Character Class (Lower Special, Upper Special, Std Lower, Std Upper)
    ax1 = axes[1]
    cats = ["Lower Special\n(ə,ı,ö,ğ,ü,ş,ç)", "Upper Special\n(Ə,İ,Ö,Ğ,Ü,Ş,Ç,I)", "Std Lower\n(a–z)", "Std Upper\n(A–Z)"]
    keys = ["lower_special", "upper_special", "standard_lower", "standard_upper"]
    v1_vals = [v1_char[k]["error_rate_pct"] for k in keys]
    v2_vals = [v2_char[k]["error_rate_pct"] for k in keys]
    t_vals = [t_char[k]["error_rate_pct"] for k in keys]
    e_vals = [e_char[k]["error_rate_pct"] for k in keys]

    x = np.arange(len(cats))
    w = 0.20
    b1 = ax1.bar(x - 1.5 * w, v1_vals, w, label="CRNN V1", color="#94a3b8", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x - 0.5 * w, v2_vals, w, label="CRNN V2 (Ours)", color="#16a34a", edgecolor="black", linewidth=0.6)
    b3 = ax1.bar(x + 0.5 * w, t_vals, w, label="Tesseract 5", color="#2563eb", edgecolor="black", linewidth=0.5)
    b4 = ax1.bar(x + 1.5 * w, e_vals, w, label="EasyOCR ('az')", color="#9333ea", edgecolor="black", linewidth=0.5)

    for bars, col in [(b1, "#475569"), (b2, "#15803d"), (b3, "#1d4ed8"), (b4, "#7e22ce")]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.18, f"{h:.2f}%",
                     ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=col, rotation=25)

    ax1.set_title("B. Character Error Rate by Orthographic Class")
    ax1.set_ylabel("Character Error Rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, fontsize=8.2)
    ax1.set_ylim(0, max(v1_vals) * 1.24)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Panel C: CTC Confidence Calibration & Selective OCR Curve on Real Photos (n=111 lines)
    ax2 = axes[2]
    sub_cov = [r for r in cov if r["fallback_pct"] <= 30]
    pcts = [100 - r["fallback_pct"] for r in sub_cov]
    sel_cers = [r["selective_crnn_cer_pct"] for r in sub_cov]
    hyb_cers = [r["hybrid_cer_pct"] for r in sub_cov]

    ax2.plot(pcts, sel_cers, "o-", color="#16a34a", linewidth=2.0, markersize=5,
             label="Selective CRNN V2 (High-Conf Subset)")
    ax2.plot(pcts, hyb_cers, "s-", color="#2563eb", linewidth=2.0, markersize=5,
             label="Confidence-Gated Hybrid (V2 + Tess Fallback)")

    ax2.annotate(f"80% Retained\nSelective CER: {h20['selective_crnn_cer_pct']:.2f}%\nHybrid CER: {h20['hybrid_cer_pct']:.2f}%",
                 (80, h20["hybrid_cer_pct"]), xytext=(76, 0.82),
                 arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.0),
                 fontsize=7.8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#2563eb", lw=0.8))

    ax2.text(0.04, 0.08,
             f"CTC Error Detection (n=111 lines):\n"
             f"• AUROC (min-char conf): {auroc_min:.3f}\n"
             f"• AUROC (mean-char conf): {auroc_mean:.3f}\n"
             f"• Exact lines mean p_min: {ctc_cal['exact_mean_min_conf']:.3f}\n"
             f"• Error lines mean p_min: {ctc_cal['error_mean_min_conf']:.3f}",
             transform=ax2.transAxes, fontsize=7.6,
             bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#16a34a", lw=0.8))

    ax2.set_title("C. Intrinsic CTC Confidence Calibration (Photos 1–5)")
    ax2.set_xlabel("Percentage of Lines Processed by CRNN V2 (%)")
    ax2.set_ylabel("Real-Document CER (%)")
    ax2.set_xlim(102, 68)  # 100% -> 70% left to right
    ax2.set_ylim(0.25, 1.22)
    ax2.legend(loc="upper left", frameon=True, fontsize=7.6)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIG15_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved {FIG15_PATH} ({FIG15_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
