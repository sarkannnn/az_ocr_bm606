import os
import sys
import time
import json
import math
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common import (
    load_model_checkpoint, normalize, cer_wer,
    preprocess, split_line_into_chunks, robust_trim_chunk
)
import pytesseract
from PIL import Image
from scripts.eval_ablation_models import build_genuine_295_benchmark

def cv2_imread_utf8(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def decode_single_crop_with_confidence(model, cfg, vocab, img, device):
    x = preprocess(img, cfg)
    x_tensor = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        out = model(x_tensor)  # (T, 1, C)
        probs = out.exp().squeeze(1).cpu().numpy()  # (T, C)

    T, C = probs.shape
    emitted_chars = []
    emitted_confs = []
    prev_idx = -1

    for t in range(T):
        p_t = probs[t]
        c_idx = int(np.argmax(p_t))
        conf = float(p_t[c_idx])

        if c_idx != 0 and c_idx != prev_idx:  # 0 is blank
            ch = vocab.i2c.get(c_idx, "")
            emitted_chars.append(ch)
            emitted_confs.append(conf)
        prev_idx = c_idx

    text = "".join(emitted_chars)
    return text, emitted_confs

def infer_line_smart_with_confidence(model, cfg, vocab, line_bgr, device):
    """Runs official infer_line_smart while tracking exact emitted character confidences."""
    h_c, w_c = line_bgr.shape[:2]
    if h_c == 0 or w_c == 0:
        return "", 0.0, 0.0, []
    target_h = cfg.get("height", 32)
    target_w = cfg.get("width", 512)
    squeeze_ratio = (w_c * target_h) / (h_c * target_w)
    n_chunks = max(1, math.ceil(squeeze_ratio)) if squeeze_ratio > 1.05 else 1
    raw_chunks = split_line_into_chunks(line_bgr, n_chunks)
    preds = []
    all_confs = []
    for ch in raw_chunks:
        ch_trimmed = robust_trim_chunk(ch)
        p, confs = decode_single_crop_with_confidence(model, cfg, vocab, ch_trimmed, device)
        p = p.strip()
        if p:
            preds.append(p)
            all_confs.extend(confs)

    full_text = " ".join(preds)
    if all_confs:
        mean_c = float(np.mean(all_confs))
        min_c = float(np.min(all_confs))
    else:
        mean_c = 0.0
        min_c = 0.0
    return full_text, mean_c, min_c, all_confs

def compute_auroc_manual(scores_exact, scores_error):
    """Computes Area Under ROC Curve via Mann-Whitney U statistic:
    Probability that an exact-match (error-free) line has higher confidence
    than an erroneous line.
    """
    n_pos = len(scores_exact)
    n_neg = len(scores_error)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    wins = 0.0
    for sp in scores_exact:
        for sn in scores_error:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return float(wins / (n_pos * n_neg))

def mcnemar_test_2x2(b, c):
    """Exact two-sided binomial McNemar test on discordant pairs (b, c)."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_val = 2.0 * binom.cdf(k, n, 0.5)
    return float(min(1.0, p_val))

def main():
    print("=" * 80)
    print("AZOCR-V2: STAGE 2.9 CONFIDENCE, AUROC & CASCADE AUDIT (295 LINES)")
    print("=" * 80)

    out_dir = BASE_DIR / "results" / "confidence_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    csv_path = out_dir / "confidence_295_lines.csv"
    if csv_path.exists():
        print(f"[+] Mövcud {csv_path} faylından 295 sətirlik qeydlər dərhal oxunur...")
        import csv
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                records.append({
                    "idx": int(r["idx"]),
                    "doc_id": r["doc_id"],
                    "img_code": r["img_code"],
                    "line_idx": int(r["line_idx"]),
                    "is_real": int(r["is_real"]),
                    "ref": r["ref"],
                    "ref_len": int(r["ref_len"]),
                    "ref_words": int(r["ref_words"]),
                    "crnn_pred": r["crnn_pred"],
                    "crnn_exact": int(r["crnn_exact"]),
                    "crnn_char_errors": float(r["crnn_char_errors"]),
                    "crnn_word_errors": float(r["crnn_word_errors"]),
                    "tess_pred": r["tess_pred"],
                    "tess_exact": int(r["tess_exact"]),
                    "tess_char_errors": float(r["tess_char_errors"]),
                    "tess_word_errors": float(r["tess_word_errors"]),
                    "mean_conf": float(r["mean_conf"]),
                    "min_conf": float(r["min_conf"])
                })
    else:
        # 1. Load CRNN-V2 Model
        ckpt_v2 = BASE_DIR / "runs" / "line_v2" / "best.pt"
        device = torch.device("cpu")
        print(f"[+] CRNN-V2 checkpoint yüklənir: {ckpt_v2}")
        model, cfg, vocab, _ = load_model_checkpoint(ckpt_v2, device=device)
        model.eval()

        # 2. Load 295 Benchmark Lines
        samples = build_genuine_295_benchmark()
        print(f"[+] 295 benchmark satırı yükləndi (55 real + 240 simulyasiya).")
        print("[+] 295 satır üzrə CTC inam skorları və proqnozlar hesablanır...")

        for i, (doc_id, img_code, line_idx, img_p, gt) in enumerate(samples):
            img_bgr = cv2_imread_utf8(img_p)
            raw_pred, mean_conf, min_conf, conf_list = infer_line_smart_with_confidence(model, cfg, vocab, img_bgr, device)
            
            norm_gt = normalize(gt.replace("—", "-"))
            norm_crnn = normalize(raw_pred.replace("—", "-"))
            is_exact = int(norm_gt == norm_crnn)
            
            # Also run Tesseract PSM 7 for cascade comparison
            pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            raw_tess = pytesseract.image_to_string(pil_img, lang="aze", config="--psm 7").strip()
            cleaned_tess = " ".join(raw_tess.split())
            norm_tess = normalize(cleaned_tess.replace("—", "-"))
            tess_is_exact = int(norm_gt == norm_tess)
            
            c_err_crnn, _ = align(norm_gt, norm_crnn)
            w_err_crnn, _ = align(norm_gt.split(), norm_crnn.split())
            c_err_tess, _ = align(norm_gt, norm_tess)
            w_err_tess, _ = align(norm_gt.split(), norm_tess.split())
            
            is_real = int(doc_id in ["B1", "B2", "B3"])
            
            records.append({
                "idx": i,
                "doc_id": doc_id,
                "img_code": img_code,
                "line_idx": line_idx,
                "is_real": is_real,
                "ref": norm_gt,
                "ref_len": len(norm_gt),
                "ref_words": len(norm_gt.split()),
                "crnn_pred": norm_crnn,
                "crnn_exact": is_exact,
                "crnn_char_errors": c_err_crnn,
                "crnn_word_errors": w_err_crnn,
                "tess_pred": norm_tess,
                "tess_exact": tess_is_exact,
                "tess_char_errors": c_err_tess,
                "tess_word_errors": w_err_tess,
                "mean_conf": mean_conf,
                "min_conf": min_conf
            })

    # 3. AUROC Analysis
    def evaluate_auroc_subset(sub_records, name):
        exact_min = [r["min_conf"] for r in sub_records if r["crnn_exact"] == 1]
        err_min = [r["min_conf"] for r in sub_records if r["crnn_exact"] == 0]
        exact_mean = [r["mean_conf"] for r in sub_records if r["crnn_exact"] == 1]
        err_mean = [r["mean_conf"] for r in sub_records if r["crnn_exact"] == 0]
        
        auroc_min = compute_auroc_manual(exact_min, err_min)
        auroc_mean = compute_auroc_manual(exact_mean, err_mean)
        
        return {
            "name": name,
            "total_lines": len(sub_records),
            "exact_lines": len(exact_min),
            "error_lines": len(err_min),
            "exact_rate_pct": round(len(exact_min) / len(sub_records) * 100, 2),
            "mean_min_conf_exact": round(float(np.mean(exact_min)), 4) if exact_min else 0.0,
            "mean_min_conf_error": round(float(np.mean(err_min)), 4) if err_min else 0.0,
            "mean_mean_conf_exact": round(float(np.mean(exact_mean)), 4) if exact_mean else 0.0,
            "mean_mean_conf_error": round(float(np.mean(err_mean)), 4) if err_mean else 0.0,
            "auroc_min_conf": round(auroc_min, 4),
            "auroc_mean_conf": round(auroc_mean, 4)
        }

    auroc_all = evaluate_auroc_subset(records, "Overall (295 Lines)")
    auroc_real = evaluate_auroc_subset([r for r in records if r["is_real"] == 1], "Real Physical (55 Lines)")
    auroc_sim = evaluate_auroc_subset([r for r in records if r["is_real"] == 0], "Simulated Phone (240 Lines)")

    print("\n--- AUROC ERROR DETECTION METRICS ---")
    for a in [auroc_all, auroc_real, auroc_sim]:
        print(f"[{a['name']}]: Exact={a['exact_lines']}/{a['total_lines']} ({a['exact_rate_pct']}%)")
        print(f"  AUROC (Min Char Conf):  {a['auroc_min_conf']:.4f}")
        print(f"  AUROC (Mean Char Conf): {a['auroc_mean_conf']:.4f}")
        print(f"  Exact lines mean min_conf: {a['mean_min_conf_exact']:.4f} vs Error lines: {a['mean_min_conf_error']:.4f}")

    # 4. Cascade Evaluation on 295 lines
    # Sort lines by min_conf ascending (lowest confidence first)
    sorted_by_min_conf = sorted(records, key=lambda r: r["min_conf"])
    total_benchmark_chars = sum(r["ref_len"] for r in records)
    total_benchmark_words = sum(r["ref_words"] for r in records)

    cascade_sweep = []
    # Historical CPU latencies from Stage 2.8:
    t_crnn_ms = 85.71
    t_tess_ms = 128.40

    for fallback_pct in [0, 5, 10, 15, 20, 25, 30, 50, 100]:
        k_fallback = int(round(len(sorted_by_min_conf) * (fallback_pct / 100.0)))
        fallback_set = set(r["idx"] for r in sorted_by_min_conf[:k_fallback])
        
        casc_refs = []
        casc_preds = []
        casc_exact = 0
        
        for r in records:
            casc_refs.append(r["ref"])
            if r["idx"] in fallback_set:
                pred = r["tess_pred"]
            else:
                pred = r["crnn_pred"]
            casc_preds.append(pred)
            if pred == r["ref"]:
                casc_exact += 1
                
        cer_ratio, wer_ratio, _ = cer_wer(casc_refs, casc_preds)
        cer = cer_ratio * 100.0
        wer = wer_ratio * 100.0
        exact_rate = casc_exact / len(records) * 100.0
        eff_latency = t_crnn_ms + (fallback_pct / 100.0) * t_tess_ms
        
        # Cutoff min_conf threshold for this fallback percentage
        cutoff_min_conf = sorted_by_min_conf[k_fallback - 1]["min_conf"] if k_fallback > 0 else 0.0
        
        cascade_sweep.append({
            "fallback_pct": fallback_pct,
            "fallback_lines": k_fallback,
            "cutoff_min_conf": round(cutoff_min_conf, 4),
            "cer_pct": round(cer, 3),
            "wer_pct": round(wer, 2),
            "exact_match_lines": f"{casc_exact}/295",
            "exact_match_pct": round(exact_rate, 2),
            "effective_cpu_latency_ms": round(eff_latency, 2)
        })

    print("\n--- CONFIDENCE-GATED HYBRID CASCADE SWEEP (295 LINES) ---")
    for cs in cascade_sweep:
        print(f"Fallback {cs['fallback_pct']:3d}% ({cs['fallback_lines']:2d} lines, cutoff<={cs['cutoff_min_conf']:.3f}): "
              f"CER={cs['cer_pct']:.3f}% | WER={cs['wer_pct']:.2f}% | Exact={cs['exact_match_lines']} ({cs['exact_match_pct']}%) | Latency={cs['effective_cpu_latency_ms']:.1f} ms")

    # 5. Head-to-Head Comparison: CRNN Raw vs Tesseract PSM 7 vs 20% Cascade
    crnn_raw_stat = cascade_sweep[0]  # 0%
    casc_20_stat = next(cs for cs in cascade_sweep if cs["fallback_pct"] == 20)
    tess_stat = cascade_sweep[-1]  # 100%

    print("\n--- HEAD-TO-HEAD COMPARISON: CRNN RAW VS TESSERACT VS 20% CASCADE ---")
    print(f"CRNN-V2 Raw:       CER={crnn_raw_stat['cer_pct']}%, WER={crnn_raw_stat['wer_pct']}%, Exact={crnn_raw_stat['exact_match_lines']}, Latency={crnn_raw_stat['effective_cpu_latency_ms']} ms")
    print(f"Tesseract PSM 7:   CER={tess_stat['cer_pct']}%, WER={tess_stat['wer_pct']}%, Exact={tess_stat['exact_match_lines']}, Latency={tess_stat['effective_cpu_latency_ms']} ms")
    print(f"20% Fallback Casc: CER={casc_20_stat['cer_pct']}%, WER={casc_20_stat['wer_pct']}%, Exact={casc_20_stat['exact_match_lines']}, Latency={casc_20_stat['effective_cpu_latency_ms']} ms")

    # 6. Real Paired McNemar Table on 295 Benchmark Lines (Exact Match)
    a_lines = sum(1 for r in records if r["crnn_exact"] == 1 and r["tess_exact"] == 1)
    b_lines = sum(1 for r in records if r["crnn_exact"] == 1 and r["tess_exact"] == 0)
    c_lines = sum(1 for r in records if r["crnn_exact"] == 0 and r["tess_exact"] == 1)
    d_lines = sum(1 for r in records if r["crnn_exact"] == 0 and r["tess_exact"] == 0)

    p_mcnemar_lines = mcnemar_test_2x2(b_lines, c_lines)

    print("\n--- REAL PAIRED MCNEMAR TABLE (LINE-LEVEL EXACT MATCH, n=295) ---")
    print(f"  Both Correct (a):             {a_lines}")
    print(f"  CRNN Correct, Tess Wrong (b): {b_lines}")
    print(f"  Tess Correct, CRNN Wrong (c): {c_lines}")
    print(f"  Both Wrong (d):               {d_lines}")
    print(f"  Discordant pairs (b + c):     {b_lines + c_lines}")
    print(f"  McNemar Exact p-value:        {p_mcnemar_lines:.4f} (Statistically Significant at 0.05: {p_mcnemar_lines < 0.05})")

    # 7. Save CSV and JSON
    import csv
    csv_path = out_dir / "confidence_295_lines.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx", "doc_id", "img_code", "line_idx", "is_real", "ref_len", "ref_words",
            "crnn_exact", "crnn_char_errors", "crnn_word_errors",
            "tess_exact", "tess_char_errors", "tess_word_errors",
            "mean_conf", "min_conf", "ref", "crnn_pred", "tess_pred"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"\n[+] Satır bazlı inam və nəticə qeydləri CSV faylına yazıldı: {csv_path}")

    output_json = {
        "status": "COMPLETED",
        "benchmark_lines": 295,
        "auroc_results": {
            "overall_295": auroc_all,
            "real_55": auroc_real,
            "sim_240": auroc_sim
        },
        "cascade_provenance_audit": {
            "manuscript_threshold": "20% fallback (Table 8b, Section 6.8.1)",
            "historical_calibration_data": "Photos 1-5 (n=111 real physical lines)",
            "methodological_critique": "The 20% fallback threshold was selected post-hoc by sweeping [0, 5, 10, 15, 20, 25, 30%] directly on the test documents (Photos 1-5). It was not tuned on an independent validation set. In this audit, we freeze the 20% threshold without adaptation to the benchmark."
        },
        "cascade_evaluation_295": {
            "parameter_sweep": cascade_sweep,
            "head_to_head": {
                "crnn_v2_raw": crnn_raw_stat,
                "tesseract_5_psm7": tess_stat,
                "cascade_20pct_fallback": casc_20_stat,
                "accuracy_comparison": f"On the 295 benchmark lines, CRNN-V2 Raw achieves {crnn_raw_stat['cer_pct']}% CER ({crnn_raw_stat['exact_match_lines']} exact lines). Tesseract PSM 7 achieves {tess_stat['cer_pct']}% CER ({tess_stat['exact_match_lines']} exact lines). The 20% cascade achieves {casc_20_stat['cer_pct']}% CER ({casc_20_stat['exact_match_lines']} exact lines)."
            }
        },
        "mcnemar_paired_analysis": {
            "benchmark_line_level": {
                "total_lines": 295,
                "both_correct_a": a_lines,
                "crnn_correct_tess_wrong_b": b_lines,
                "tess_correct_crnn_wrong_c": c_lines,
                "both_wrong_d": d_lines,
                "discordant_pairs": b_lines + c_lines,
                "mcnemar_exact_p_value": round(p_mcnemar_lines, 4),
                "is_significant_at_05": bool(p_mcnemar_lines < 0.05)
            },
            "historical_table6b_missing_data_disclosure": {
                "sample_size": 244,
                "missing_data": "Individual character-by-character paired binary prediction vector for the 244 historical uppercase specials was not preserved in raw logs (only aggregate error totals: V2=6, Tess=9). Consequently, exact discordant pair count (b, c) is bounded within [0, 6] shared errors, yielding p in [0.2500, 0.6072], none of which are statistically significant."
            }
        },
        "stage2_8_report_corrections": {
            "unit_correction": "In Stage 2.8 report Table 1, '2.210 ms' represented 2,209.9 ms (2.21 seconds). It has been corrected to '2.21 s (2,210 ms)'.",
            "tesseract_api_speed_caveat": "The ~60-65 ms/line direct API figure was an unmeasured empirical estimate derived by subtracting the 64.8 ms subprocess spawn overhead from 127.5 ms. It is an unverified theoretical lower bound, not an actual C++ API benchmark.",
            "standard_deviation_clarification": "The +/- 23.58 ms std for CRNN and +/- 15.32 ms for Tesseract reflect LINE-TO-LINE variation across lines of varying lengths and aspect ratios. The REPETITION-TO-REPETITION run standard deviation across the 3 benchmark passes is much tighter: +/- 0.95 s (+/- 3.2 ms/line) for CRNN and +/- 0.25 s (+/- 0.8 ms/line) for Tesseract."
        }
    }

    json_path = out_dir / "confidence_cascade_audit_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Yekun JSON faylı yazıldı: {json_path}")

if __name__ == "__main__":
    main()
