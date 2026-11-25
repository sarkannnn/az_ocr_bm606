"""Task 3: Retrospective Frozen-Threshold Confidence Cascade Evaluation.
Determines fixed confidence thresholds strictly on D1-D5 (111 lines) and applies them
unchanged to B1-B15 (295 lines).
"""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(r"d:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(ROOT))

from common import load_model_checkpoint, normalize, align
from scripts.audit_confidence_cascade_295 import decode_single_crop_with_confidence

OUT_DIR = ROOT / "FINAL_REMAINING_CHECKS"
OUT_DIR.mkdir(exist_ok=True)


def load_d1_d5():
    records = []
    with open(ROOT / "results" / "real_fix8" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": r["photo"], "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "real_fix8",
                "chunk_paths": [p.strip() for p in r["chunk_crop_paths"].split(";") if p.strip()],
            })
    with open(ROOT / "results" / "photo3_test" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 3 (Times)", "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "photo3_test",
                "chunk_paths": [p.strip() for p in r["chunk_paths"].split(";") if p.strip()],
            })
    with open(ROOT / "results" / "photo4_sans" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 4 (Arial)", "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "tess_full": normalize(r["tess_full"].strip()),
                "base_dir": ROOT / "results" / "photo4_sans",
                "chunk_paths": [p.strip() for p in r["chunk_paths"].split(";") if p.strip()],
            })
    with open(ROOT / "results" / "real_fix9" / "per_line_chunk_details.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            records.append({
                "photo": "Photo 5 (Times)", "line_idx": int(r["line_idx"]),
                "ref": normalize(r["ref"].strip()),
                "tess_full": normalize(r["tess_full_A_psm13"].strip()),
                "base_dir": ROOT / "results" / "real_fix9",
                "chunk_paths": [p.strip() for p in r["chunk_paths_A"].split(";") if p.strip()],
            })
    return records


def main():
    device = torch.device("cpu")
    model, cfg, vocab, _ = load_model_checkpoint(ROOT / "runs" / "line_v2" / "best.pt", device=device)
    model.eval()

    d1_d5_raw = load_d1_d5()
    assert len(d1_d5_raw) == 111, f"Expected 111 D1-D5 lines, got {len(d1_d5_raw)}"

    d_rows = []
    for rec in d1_d5_raw:
        preds, all_confs = [], []
        for rel_cp in rec["chunk_paths"]:
            cp = rec["base_dir"] / rel_cp.replace("\\", "/")
            img = cv2.imdecode(np.fromfile(str(cp), dtype=np.uint8), cv2.IMREAD_COLOR)
            txt, confs = decode_single_crop_with_confidence(model, cfg, vocab, img, device)
            txt = txt.strip()
            if txt:
                preds.append(txt)
                all_confs.extend(confs)
        hyp = normalize(" ".join(preds))
        ref = rec["ref"]
        c_err, _ = align(ref, hyp)
        t_err, _ = align(ref, rec["tess_full"])
        d_rows.append({
            "ref_len": len(ref),
            "crnn_err": c_err,
            "tess_err": t_err,
            "is_error": int(c_err > 0),
            "min_conf": float(np.min(all_confs)) if all_confs else 0.0,
            "mean_conf": float(np.mean(all_confs)) if all_confs else 0.0,
        })

    # Load B1-B15 (295 lines)
    b_rows = []
    with open(ROOT / "results" / "confidence_audit" / "confidence_295_lines.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b_rows.append({
                "ref_len": int(r["ref_len"]),
                "ref_words": int(r["ref_words"]),
                "crnn_exact": int(r["crnn_exact"]),
                "crnn_c_err": round(float(r["crnn_char_errors"]) * int(r["ref_len"])),
                "crnn_w_err": round(float(r["crnn_word_errors"]) * int(r["ref_words"])),
                "tess_exact": int(r["tess_exact"]),
                "tess_c_err": round(float(r["tess_char_errors"]) * int(r["ref_len"])),
                "tess_w_err": round(float(r["tess_word_errors"]) * int(r["ref_words"])),
                "min_conf": float(r["min_conf"]),
                "mean_conf": float(r["mean_conf"]),
            })
    assert len(b_rows) == 295

    b_total_chars = sum(r["ref_len"] for r in b_rows)
    b_total_words = sum(r["ref_words"] for r in b_rows)
    d_total_chars = sum(r["ref_len"] for r in d_rows)

    T_CRNN = 85.71
    T_TESS = 128.40

    def eval_rule(rule_name, metric_key, threshold, d_k_override=None):
        if threshold is None:
            # 0% or 100% baseline
            routed_b = 0 if d_k_override == 0 else len(b_rows)
            routed_d = 0 if d_k_override == 0 else len(d_rows)
            b_flags = [d_k_override != 0] * len(b_rows)
            d_flags = [d_k_override != 0] * len(d_rows)
        else:
            b_flags = [r[metric_key] <= threshold for r in b_rows]
            d_flags = [r[metric_key] <= threshold for r in d_rows]
            routed_b = sum(b_flags)
            routed_d = sum(d_flags)

        d_err = sum(r["tess_err"] if fl else r["crnn_err"] for r, fl in zip(d_rows, d_flags))
        b_c_err = sum(r["tess_c_err"] if fl else r["crnn_c_err"] for r, fl in zip(b_rows, b_flags))
        b_w_err = sum(r["tess_w_err"] if fl else r["crnn_w_err"] for r, fl in zip(b_rows, b_flags))
        b_exact = sum(r["tess_exact"] if fl else r["crnn_exact"] for r, fl in zip(b_rows, b_flags))

        ratio_b = routed_b / len(b_rows)
        eff_lat = T_CRNN + ratio_b * T_TESS if routed_b > 0 else T_CRNN
        if d_k_override == len(b_rows):
            # Tesseract-only standalone vs CRNN+Tess 100% cascade
            pass
        return {
            "rule_name": rule_name,
            "confidence_metric": metric_key,
            "frozen_threshold_from_d1_d5": f"{threshold:.6f}" if threshold is not None else "N/A",
            "d1_d5_routed_lines": routed_d,
            "d1_d5_routing_ratio_pct": round(100.0 * routed_d / len(d_rows), 2),
            "d1_d5_cascade_cer_pct": round(100.0 * d_err / d_total_chars, 3),
            "b1_b15_routed_lines": routed_b,
            "b1_b15_routing_ratio_pct": round(100.0 * ratio_b, 2),
            "b1_b15_cer_pct": round(100.0 * b_c_err / b_total_chars, 3),
            "b1_b15_wer_pct": round(100.0 * b_w_err / b_total_words, 2),
            "b1_b15_exact_matches": b_exact,
            "b1_b15_exact_match_pct": round(100.0 * b_exact / len(b_rows), 2),
            "b1_b15_effective_latency_ms": round(eff_lat, 2),
            "speedup_vs_tesseract_standalone": round(T_TESS / eff_lat, 2),
            "evaluation_status": "RETROSPECTIVE_FROZEN_THRESHOLD",
        }

    out_rows = [eval_rule("Baseline_CRNN_V2_Only (0% routing)", "none", None, d_k_override=0)]

    # Calibration percentiles on D1-D5 min_conf
    sorted_min = sorted(r["min_conf"] for r in d_rows)
    for pct in [5, 10, 15, 20, 25, 30]:
        k = int(round(len(d_rows) * (pct / 100.0)))
        thr = sorted_min[k - 1]
        out_rows.append(eval_rule(f"D1-D5_{pct}pct_min_conf_cutoff", "min_conf", thr))

    # Youden-J optimal on D1-D5 for min_conf and mean_conf
    n_pos = sum(r["is_error"] for r in d_rows)
    n_neg = len(d_rows) - n_pos
    for mkey in ["min_conf", "mean_conf"]:
        best_j, best_t = -1.0, 0.0
        for cand in sorted(set(r[mkey] for r in d_rows)):
            tp = sum(1 for r in d_rows if r[mkey] <= cand and r["is_error"] == 1)
            fp = sum(1 for r in d_rows if r[mkey] <= cand and r["is_error"] == 0)
            j = (tp / n_pos) - (fp / n_neg)
            if j > best_j:
                best_j, best_t = j, cand
        out_rows.append(eval_rule(f"D1-D5_YoudenJ_optimal_{mkey} (J={best_j:.3f})", mkey, best_t))

    # Tesseract standalone anchor
    t_row = eval_rule("Baseline_Tesseract5_PSM7_Standalone (100%)", "none", None, d_k_override=len(b_rows))
    t_row["b1_b15_effective_latency_ms"] = round(T_TESS, 2)
    t_row["speedup_vs_tesseract_standalone"] = 1.00
    out_rows.append(t_row)

    out_csv = OUT_DIR / "FROZEN_THRESHOLD_CASCADE.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    for r in out_rows:
        print(f"{r['rule_name']:42s} | thr={r['frozen_threshold_from_d1_d5']:>8s} | B_route={r['b1_b15_routed_lines']:3d} ({r['b1_b15_routing_ratio_pct']:5.2f}%) | CER={r['b1_b15_cer_pct']:.3f}% | WER={r['b1_b15_wer_pct']:.2f}% | Exact={r['b1_b15_exact_matches']}/295 ({r['b1_b15_exact_match_pct']:.2f}%) | Lat={r['b1_b15_effective_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
