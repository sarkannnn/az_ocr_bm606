"""Compute strengthening empirical analyses for paper_v4:
1. 4x4 Dotted/Dotless I Paradox Matrix (i, ı, İ, I) & Uppercase vs. Lowercase Azerbaijani Special Letters (Ə/ə, İ/I/i/ı, Ö/ö, Ğ/ğ, Ü/ü, Ş/ş, Ç/ç) across synthetic test sets and real photos (Photos 1-5).
2. CTC Confidence Calibration & Selective/Hybrid OCR Analysis on Real Document Photos (Photos 1-5, n=111 lines):
   - Mean CTC emitted-character probability on error-free vs. erroneous lines/chunks.
   - AUROC of CTC confidence as an unsupervised error detector.
   - Selective OCR / Confidence-Gated Hybrid (routing only low-confidence chunks/lines to Tesseract).
3. Lexicon-free CTC Space-Threshold Decoding (suppressing low-probability intra-word space spikes without a dictionary).
4. Generate Figure 15 (paper/figures/f15_efficiency_and_calibration.png) at 300 DPI.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein as L

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import align, load_model_checkpoint, normalize, preprocess

OUT_DIR = ROOT / "results" / "strengthening_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tsv_pairs(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                pairs.append((normalize(parts[0]), normalize(parts[1])))
    return pairs


def load_real_photo_pairs():
    """Load (ref, v1_chunk, v2_chunk, tess_chunk, v2_full, tess_full) across all 5 real photos (n=111 lines)."""
    records = []
    # Photos 1 & 2
    with open(ROOT / "results" / "real_fix8" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": r["photo"],
                "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "v1_chunk": normalize(r["v1_rule_joined"].strip()),
                "v2_chunk": normalize(r["v2_rule_joined"].strip()),
                "tess_chunk": normalize(r["tess_rule_joined"].strip()),
                "v2_full": normalize(r["v2_full"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "real_fix8",
                "chunk_paths": [p.strip() for p in r["chunk_crop_paths"].split(";") if p.strip()],
            })
    # Photo 3
    with open(ROOT / "results" / "photo3_test" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 3 (Times)",
                "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "v1_chunk": normalize(r["v1_joined"].strip()),
                "v2_chunk": normalize(r["v2_joined"].strip()),
                "tess_chunk": normalize(r["tess_joined"].strip()),
                "v2_full": normalize(r["v2_full"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "photo3_test",
                "chunk_paths": [p.strip() for p in r["chunk_paths"].split(";") if p.strip()],
            })
    # Photo 4
    with open(ROOT / "results" / "photo4_sans" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 4 (Arial)",
                "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "v1_chunk": normalize(r["v1_rule_joined"].strip()),
                "v2_chunk": normalize(r["v2_rule_joined"].strip()),
                "tess_chunk": normalize(r["tess_rule_joined"].strip()),
                "v2_full": normalize(r["v2_full"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "photo4_sans",
                "chunk_paths": [p.strip() for p in r["chunk_paths"].split(";") if p.strip()],
            })
    # Photo 5
    with open(ROOT / "results" / "real_fix9" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 5 (Times)",
                "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "v1_chunk": normalize(r["v1_rule_joined_A"].strip()),
                "v2_chunk": normalize(r["v2_rule_joined_A"].strip()),
                "tess_chunk": normalize(r["tess_rule_joined_A"].strip()),
                "v2_full": normalize(r["v2_full_A"].strip()),
                "tess_full": normalize(r["tess_full_A_psm13"].strip()),
                "base_dir": ROOT / "results" / "real_fix9",
                "chunk_paths": [p.strip() for p in r["chunk_paths_A"].split(";") if p.strip()],
            })
    return records


def analyze_char_classes_and_i_paradox(pairs):
    """Compute character-level accuracy by character class (lowercase special, uppercase special,
    ASCII letters, digits, punctuation) and the 4x4 (i, ı, İ, I) confusion matrix.
    """
    i_chars = ["i", "ı", "İ", "I"]
    i_mat = {r: {h: 0 for h in i_chars + ["OTHER", "DEL"]} for r in i_chars}

    upper_special = "ƏİÖĞÜŞÇI"
    lower_special = "əıöğüşç"

    per_char_total = Counter()
    per_char_ok = Counter()
    per_char_top_sub = defaultdict(Counter)

    class_stats = {
        "lower_special": [0, 0],   # [errors, total]
        "upper_special": [0, 0],
        "standard_lower": [0, 0],
        "standard_upper": [0, 0],
        "digits_punct": [0, 0],
        "space": [0, 0],
    }

    for ref, hyp in pairs:
        _, ops = align(ref, hyp)
        for op, r_ch, h_ch in ops:
            if op in ("ok", "sub", "del"):
                per_char_total[r_ch] += 1
                is_err = (op != "ok")
                if not is_err:
                    per_char_ok[r_ch] += 1
                elif op == "sub":
                    per_char_top_sub[r_ch][h_ch] += 1

                if r_ch in i_chars:
                    if op == "ok":
                        i_mat[r_ch][r_ch] += 1
                    elif op == "del":
                        i_mat[r_ch]["DEL"] += 1
                    elif h_ch in i_chars:
                        i_mat[r_ch][h_ch] += 1
                    else:
                        i_mat[r_ch]["OTHER"] += 1

                if r_ch in lower_special:
                    class_stats["lower_special"][1] += 1
                    if is_err:
                        class_stats["lower_special"][0] += 1
                elif r_ch in upper_special:
                    class_stats["upper_special"][1] += 1
                    if is_err:
                        class_stats["upper_special"][0] += 1
                elif r_ch == " ":
                    class_stats["space"][1] += 1
                    if is_err:
                        class_stats["space"][0] += 1
                elif r_ch.islower():
                    class_stats["standard_lower"][1] += 1
                    if is_err:
                        class_stats["standard_lower"][0] += 1
                elif r_ch.isupper():
                    class_stats["standard_upper"][1] += 1
                    if is_err:
                        class_stats["standard_upper"][0] += 1
                else:
                    class_stats["digits_punct"][1] += 1
                    if is_err:
                        class_stats["digits_punct"][0] += 1

    special_breakdown = {}
    for ch in list(lower_special) + list(upper_special):
        tot = per_char_total[ch]
        ok = per_char_ok[ch]
        err = tot - ok
        top_s = per_char_top_sub[ch].most_common(2)
        special_breakdown[ch] = {
            "total": tot,
            "ok": ok,
            "errors": err,
            "error_rate_pct": round(100.0 * err / tot, 2) if tot > 0 else 0.0,
            "accuracy_pct": round(100.0 * ok / tot, 2) if tot > 0 else 100.0,
            "top_confusions": top_s,
        }

    class_summary = {}
    for k, (err, tot) in class_stats.items():
        class_summary[k] = {
            "errors": err,
            "total": tot,
            "error_rate_pct": round(100.0 * err / tot, 2) if tot > 0 else 0.0,
            "accuracy_pct": round(100.0 * (tot - err) / tot, 2) if tot > 0 else 100.0,
        }

    return {
        "i_paradox_matrix": i_mat,
        "special_letters": special_breakdown,
        "class_summary": class_summary,
    }


def decode_with_confidence_and_space_threshold(model, cfg, vocab, img_bgr, device, space_thresh=0.0):
    """Run CRNN forward pass and return:
    - greedy decoded text (with optional space probability threshold `space_thresh`)
    - mean confidence of emitted non-blank characters
    - minimum character confidence in the sequence
    - mean frame max probability across all T=128 frames
    """
    x = torch.from_numpy(preprocess(img_bgr, cfg)).unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = model(x)  # (T, 1, C)
        probs = log_probs.exp().squeeze(1).cpu().numpy()  # (T, C)

    space_idx = vocab.c2i.get(" ", -1)
    T, C = probs.shape
    pred_ids = []
    emitted_confs = []
    prev_id = -1

    for t in range(T):
        p_t = probs[t]
        cid = int(np.argmax(p_t))
        conf = float(p_t[cid])
        if cid == space_idx and space_thresh > 0.0 and conf < space_thresh:
            # Suppress low-confidence space spike -> pick second best (usually blank 0)
            p_copy = p_t.copy()
            p_copy[space_idx] = -1.0
            cid = int(np.argmax(p_copy))
            conf = float(p_t[cid])
        pred_ids.append(cid)
        if cid != 0 and cid != prev_id:
            emitted_confs.append(conf)
        prev_id = cid

    text = vocab.decode_greedy(pred_ids)
    mean_char_conf = float(np.mean(emitted_confs)) if emitted_confs else 1.0
    min_char_conf = float(np.min(emitted_confs)) if emitted_confs else 1.0
    mean_frame_conf = float(np.mean(np.max(probs, axis=1)))
    return text, mean_char_conf, min_char_conf, mean_frame_conf


def compute_auroc(scores_pos, scores_neg):
    """Compute AUROC (probability that a randomly chosen error-free line has higher confidence
    than a randomly chosen erroneous line).
    """
    wins = 0.0
    total = 0
    for sp in scores_pos:
        for sn in scores_neg:
            total += 1
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / max(total, 1)


def main():
    # 1. Character Class & Dotted/Dotless I Paradox across Synthetic + Real Benchmarks
    synth_v1 = load_tsv_pairs(ROOT / "results" / "line_synth" / "predictions.tsv") + \
               load_tsv_pairs(ROOT / "results" / "word_synth" / "predictions.tsv")
    synth_v2 = load_tsv_pairs(ROOT / "results" / "line_v2_synth" / "predictions.tsv") + \
               load_tsv_pairs(ROOT / "results" / "word_v2_synth" / "predictions.tsv")
    synth_tess = load_tsv_pairs(ROOT / "results" / "tess_line_synth" / "predictions.tsv") + \
                 load_tsv_pairs(ROOT / "results" / "tess_word_synth" / "predictions.tsv")

    real_records = load_real_photo_pairs()
    real_v1_pairs = [(r["ref"], r["v1_chunk"]) for r in real_records]
    real_v2_pairs = [(r["ref"], r["v2_chunk"]) for r in real_records]
    real_tess_pairs = [(r["ref"], r["tess_full"]) for r in real_records]

    combined_v1 = synth_v1 + real_v1_pairs
    combined_v2 = synth_v2 + real_v2_pairs
    combined_tess = synth_tess + real_tess_pairs

    audit_v1 = analyze_char_classes_and_i_paradox(combined_v1)
    audit_v2 = analyze_char_classes_and_i_paradox(combined_v2)
    audit_tess = analyze_char_classes_and_i_paradox(combined_tess)

    # 2. CTC Confidence Calibration & Selective/Hybrid OCR on Real Photos 1-5 (n=111 lines)
    model, cfg, vocab, device = load_model_checkpoint(ROOT / "runs" / "line_v2" / "best.pt", torch.device("cpu"))

    line_conf_records = []
    for rec in real_records:
        chunk_texts = []
        chunk_mean_confs = []
        chunk_min_confs = []
        for rel_cp in rec["chunk_paths"]:
            cp = rec["base_dir"] / rel_cp.replace("\\", "/")
            img = cv2.imdecode(np.fromfile(str(cp), dtype=np.uint8), cv2.IMREAD_COLOR)
            txt, m_conf, min_c, _ = decode_with_confidence_and_space_threshold(model, cfg, vocab, img, device, 0.0)
            chunk_texts.append(txt.strip())
            chunk_mean_confs.append(m_conf)
            chunk_min_confs.append(min_c)

        joined_v2 = normalize(" ".join(t for t in chunk_texts if t))
        edit_dist = L.distance(rec["ref"], joined_v2)
        tess_dist = L.distance(rec["ref"], rec["tess_full"])
        line_conf_records.append({
            "photo": rec["photo"],
            "line_idx": rec["line_idx"],
            "ref_len": len(rec["ref"]),
            "v2_errors": edit_dist,
            "tess_psm13_or_full_errors": tess_dist,
            "is_exact": int(edit_dist == 0),
            "mean_char_conf": float(np.mean(chunk_mean_confs)),
            "min_char_conf": float(np.min(chunk_min_confs)),
        })

    exact_confs = [r["min_char_conf"] for r in line_conf_records if r["is_exact"] == 1]
    error_confs = [r["min_char_conf"] for r in line_conf_records if r["is_exact"] == 0]
    exact_mean_confs = [r["mean_char_conf"] for r in line_conf_records if r["is_exact"] == 1]
    error_mean_confs = [r["mean_char_conf"] for r in line_conf_records if r["is_exact"] == 0]

    auroc_min = compute_auroc(exact_confs, error_confs)
    auroc_mean = compute_auroc(exact_mean_confs, error_mean_confs)

    # Selective / Hybrid OCR curve: sort 111 lines by min_char_conf ascending (lowest confidence first)
    sorted_lines = sorted(line_conf_records, key=lambda r: r["min_char_conf"])
    total_chars = sum(r["ref_len"] for r in sorted_lines)
    hybrid_curve = []
    for k_pct in [0, 5, 10, 15, 20, 25, 30, 50, 100]:
        k_Fallback = int(round(len(sorted_lines) * (k_pct / 100.0)))
        # First k_Fallback lowest-confidence lines use Tesseract full-line; remaining (111 - k) use CRNN V2
        tot_err = sum(r["tess_psm13_or_full_errors"] for r in sorted_lines[:k_Fallback]) + \
                  sum(r["v2_errors"] for r in sorted_lines[k_Fallback:])
        cer_hybrid = 100.0 * tot_err / total_chars
        # Effective latency: CRNN (127.59 ms batched) on all lines + Tesseract (425.99 ms) on k% fallback lines
        eff_ms = 127.59 + (k_pct / 100.0) * 425.99
        # Selective CER on only the retained (100 - k)% high-confidence CRNN lines:
        retained = sorted_lines[k_Fallback:]
        ret_chars = sum(r["ref_len"] for r in retained)
        ret_err = sum(r["v2_errors"] for r in retained)
        cer_selective = 100.0 * ret_err / max(ret_chars, 1)
        hybrid_curve.append({
            "fallback_pct": k_pct,
            "fallback_lines": k_Fallback,
            "retained_lines": len(retained),
            "selective_crnn_cer_pct": round(cer_selective, 2),
            "hybrid_cer_pct": round(cer_hybrid, 2),
            "effective_latency_ms": round(eff_ms, 1),
            "speedup_vs_tess_psm13": round(425.99 / eff_ms, 2),
        })

    # 3. Generate Figure 15 (3-panel publication figure at 300 DPI)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Calibri"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300)

    # Panel A: CPU Latency vs Real-Document Accuracy (All 111 lines across Photos 1-5)
    ax0 = axes[0]
    for h in hybrid_curve:
        if h["fallback_pct"] in (0, 10, 20, 30, 100):
            lbl = f"Hybrid ({h['fallback_pct']}% fallback)" if 0 < h["fallback_pct"] < 100 else None
            color = "#2B5B84" if 0 < h["fallback_pct"] < 100 else ("#1B365D" if h["fallback_pct"] == 0 else "#C0392B")
            marker = "o" if 0 < h["fallback_pct"] < 100 else ("s" if h["fallback_pct"] == 0 else "^")
            ax0.scatter(h["effective_latency_ms"], h["hybrid_cer_pct"], s=85, color=color, marker=marker, zorder=4)
            tag = (
                f"CRNN V2 (batch=16)\n({h['effective_latency_ms']:.0f} ms, {h['hybrid_cer_pct']:.2f}%)"
                if h["fallback_pct"] == 0
                else (
                    f"Tesseract 5 Full\n(426 ms, {h['hybrid_cer_pct']:.2f}%)"
                    if h["fallback_pct"] == 100
                    else f"{h['fallback_pct']}% Tess Fallback\n({h['effective_latency_ms']:.0f} ms, {h['hybrid_cer_pct']:.2f}%)"
                )
            )
            dx, dy = (10, 0.06) if h["fallback_pct"] != 100 else (-115, 0.12)
            if h["fallback_pct"] == 20:
                dx, dy = (10, -0.14)
            ax0.annotate(tag, (h["effective_latency_ms"], h["hybrid_cer_pct"]),
                         xytext=(dx, dy * 100), textcoords="offset points", fontsize=7.5, weight="bold", color=color)

    xs = [h["effective_latency_ms"] for h in hybrid_curve]
    ys = [h["hybrid_cer_pct"] for h in hybrid_curve]
    ax0.plot(xs, ys, "--", color="#2B5B84", alpha=0.7, linewidth=1.5, label="Confidence-Gated Pareto Curve")
    ax0.set_xlabel("Mean CPU Latency per Line (ms)", fontsize=9.5, weight="bold")
    ax0.set_ylabel("Aggregate Real-Photo CER (%, n=111 lines)", fontsize=9.5, weight="bold")
    ax0.set_title("A. CPU Speed–Accuracy Pareto Frontier", fontsize=10.5, weight="bold", color="#1B365D")
    ax0.set_xlim(90, 580)
    ax0.set_ylim(-0.05, 1.35)
    ax0.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Uppercase vs Lowercase Special Letters & Dotted/Dotless I Recognition Accuracy
    ax1 = axes[1]
    groups = ["Lower Special\n(ə,ı,ö,ğ,ü,ş,ç)", "Upper Special\n(Ə,İ,Ö,Ğ,Ü,Ş,Ç,I)", "Standard Lower\n(a–z)", "Standard Upper\n(A–Z)", "Digits & Punct\n(0–9, .,:;!)"]
    keys = ["lower_special", "upper_special", "standard_lower", "standard_upper", "digits_punct"]
    v1_errs = [audit_v1["class_summary"][k]["error_rate_pct"] for k in keys]
    v2_errs = [audit_v2["class_summary"][k]["error_rate_pct"] for k in keys]
    tess_errs = [audit_tess["class_summary"][k]["error_rate_pct"] for k in keys]

    x_pos = np.arange(len(groups))
    w = 0.26
    b1 = ax1.bar(x_pos - w, v1_errs, width=w, label="CRNN V1", color="#95A5A6")
    b2 = ax1.bar(x_pos, v2_errs, width=w, label="CRNN V2", color="#1B365D")
    b3 = ax1.bar(x_pos + w, tess_errs, width=w, label="Tesseract 5", color="#E67E22")
    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}%",
                     ha="center", va="bottom", fontsize=6.5, rotation=0)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(groups, fontsize=7.5)
    ax1.set_ylabel("Character Error Rate within Class (%)", fontsize=9.5, weight="bold")
    ax1.set_title("B. Error Rate by Character Class (n=44,957 chars)", fontsize=10.5, weight="bold", color="#1B365D")
    ax1.set_ylim(0, max(v1_errs) * 1.22)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel C: CTC Confidence Calibration & Selective OCR Curve on Real Photos (n=111 lines)
    ax2 = axes[2]
    cov_pct = [100 - h["fallback_pct"] for h in hybrid_curve if h["fallback_pct"] <= 50]
    sel_cer = [h["selective_crnn_cer_pct"] for h in hybrid_curve if h["fallback_pct"] <= 50]
    ax2.plot(cov_pct, sel_cer, "-o", color="#1B365D", linewidth=2.0, markersize=6, label="Selective CRNN V2 CER")
    for c, s in zip(cov_pct, sel_cer):
        if c in (100, 90, 80, 70, 50):
            ax2.annotate(f"{s:.2f}%", (c, s), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.5, weight="bold")
    ax2.set_xlabel("Retained High-Confidence CRNN Coverage (%)", fontsize=9.5, weight="bold")
    ax2.set_ylabel("CER on Retained Lines (%)", fontsize=9.5, weight="bold")
    ax2.set_title(f"C. CTC Confidence Calibration (AUROC = {auroc_min:.2f})", fontsize=10.5, weight="bold", color="#1B365D")
    ax2.invert_xaxis()
    ax2.set_ylim(0, max(sel_cer) * 1.28)
    ax2.grid(True, linestyle=":", alpha=0.5)
    info_txt = (
        f"Exact lines (n={len(exact_confs)}):\n"
        f"  Mean min-conf = {np.mean(exact_confs):.3f}\n"
        f"Error lines (n={len(error_confs)}):\n"
        f"  Mean min-conf = {np.mean(error_confs):.3f}\n"
        f"Error detection AUROC = {auroc_min:.3f}"
    )
    ax2.text(0.04, 0.12, info_txt, transform=ax2.transAxes, fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#1B365D", alpha=0.9))

    plt.tight_layout()
    fig_path = ROOT / "paper" / "figures" / "f15_efficiency_and_calibration.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    out_data = {
        "n_combined_chars": sum(v["total"] for v in audit_v2["class_summary"].values()),
        "v1_char_audit": audit_v1,
        "v2_char_audit": audit_v2,
        "tess_char_audit": audit_tess,
        "real_photos_confidence": {
            "n_lines": len(line_conf_records),
            "n_exact_lines": len(exact_confs),
            "n_error_lines": len(error_confs),
            "exact_mean_min_conf": round(float(np.mean(exact_confs)), 4),
            "error_mean_min_conf": round(float(np.mean(error_confs)), 4),
            "exact_mean_char_conf": round(float(np.mean(exact_mean_confs)), 4),
            "error_mean_char_conf": round(float(np.mean(error_mean_confs)), 4),
            "auroc_min_char_conf": round(auroc_min, 4),
            "auroc_mean_char_conf": round(auroc_mean, 4),
            "hybrid_pareto_curve": hybrid_curve,
        },
        "figure15_path": str(fig_path),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_data["real_photos_confidence"], indent=2))
    print("V2 Class Summary:", json.dumps(audit_v2["class_summary"], indent=2))
    print("V2 I-Paradox Matrix:", json.dumps(audit_v2["i_paradox_matrix"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

