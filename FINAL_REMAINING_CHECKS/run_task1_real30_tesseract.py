"""Task 1: Real30 Tesseract v5.4.0.20240606 Fairness Evaluation (PSM7 & PSM13, Original vs +10px White Margin)."""
import csv
import os
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

ROOT = Path(r"d:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(ROOT))

from common import align, normalize
from scripts.audit_real30_methodology import segment_lines_with_boxes, cv2_imread_utf8

TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESS_EXE

PHOTOS_DIR = ROOT / "Real30_GPU18_Final_Evaluation" / "extracted_photos"
VA_CSV = ROOT / "results" / "real30_evaluation" / "verified_alignment.csv"
GT_CSV = ROOT / "data" / "real30_evaluation" / "ground_truth_lines.csv"
OUT_DIR = ROOT / "FINAL_REMAINING_CHECKS"
OUT_DIR.mkdir(exist_ok=True)


def nfc(s) -> str:
    if pd.isna(s):
        return ""
    return unicodedata.normalize("NFC", str(s).strip())


def main():
    tess_ver = str(pytesseract.get_tesseract_version())
    print(f"Tesseract version: {tess_ver}")

    df_va = pd.read_csv(VA_CSV)
    df_gt = pd.read_csv(GT_CSV)
    assert len(df_va) == 690
    assert len(df_gt) == 421

    crops_data = []
    for idx in range(1, 31):
        pid = f"P{idx:02d}"
        img_bgr = cv2_imread_utf8(PHOTOS_DIR / f"{pid}.jpg")
        crops, boxes = segment_lines_with_boxes(img_bgr)
        va_page = df_va[df_va["page_id"] == pid].sort_values("crop_id")
        assert len(crops) == len(va_page)
        for crop_bgr, (_, va_row) in zip(crops, va_page.iterrows()):
            crops_data.append({
                "page_id": pid,
                "crop_id": int(va_row["crop_id"]),
                "category": va_row["verified_crop_category"],
                "gt_text": nfc(va_row["verified_ground_truth_text"]),
                "crop_bgr": crop_bgr,
            })

    # Page GT concatenation for Page-Primary CER/WER
    page_gt_text = {}
    for pid, grp in df_gt.groupby("page_id"):
        lines = [nfc(t) for t in grp.sort_values("line_no")["ground_truth_nfc"] if nfc(t)]
        page_gt_text[pid] = " ".join(lines)

    conditions = [
        ("PSM7_original_crop", 7, 0),
        ("PSM13_original_crop", 13, 0),
        ("PSM7_plus_10px_white", 7, 10),
        ("PSM13_plus_10px_white", 13, 10),
    ]

    summary_rows = []
    for cond_name, psm, pad in conditions:
        t0 = time.perf_counter()
        latencies = []
        clean_c_err = clean_c_len = clean_w_err = clean_w_len = clean_exact = clean_empty = 0
        noise_total = noise_empty = noise_fp = 0
        page_preds = {f"P{i:02d}": [] for i in range(1, 31)}

        for item in crops_data:
            img = item["crop_bgr"]
            if pad > 0:
                img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ts = time.perf_counter()
            raw = pytesseract.image_to_string(pil_img, lang="aze", config=f"--psm {psm}").strip()
            latencies.append((time.perf_counter() - ts) * 1000.0)
            pred = normalize(" ".join(raw.split()).replace("—", "-"))
            page_preds[item["page_id"]].append(pred)

            if item["category"] == "genuine_single_line":
                gt = normalize(item["gt_text"].replace("—", "-"))
                ce, _ = align(gt, pred)
                we, _ = align(gt.split(), pred.split())
                clean_c_err += ce
                clean_c_len += len(gt)
                clean_w_err += we
                clean_w_len += len(gt.split())
                if gt == pred:
                    clean_exact += 1
                if not pred:
                    clean_empty += 1
            elif item["category"] == "noise":
                noise_total += 1
                if not pred:
                    noise_empty += 1
                else:
                    noise_fp += 1

        page_c_err = page_c_len = page_w_err = page_w_len = 0
        for pid, gt_full in page_gt_text.items():
            gt_n = normalize(gt_full.replace("—", "-"))
            hyp_n = normalize(" ".join(p for p in page_preds[pid] if p).replace("—", "-"))
            ce, _ = align(gt_n, hyp_n)
            we, _ = align(gt_n.split(), hyp_n.split())
            page_c_err += ce
            page_c_len += len(gt_n)
            page_w_err += we
            page_w_len += len(gt_n.split())

        elapsed_s = time.perf_counter() - t0
        row = {
            "benchmark": "Real30 (690 crops; 325 Clean325; 314 Noise)",
            "condition": cond_name,
            "psm": psm,
            "padding_px": pad,
            "tesseract_version": tess_ver,
            "clean325_cer_pct": round(100.0 * clean_c_err / clean_c_len, 2),
            "clean325_wer_pct": round(100.0 * clean_w_err / clean_w_len, 2),
            "clean325_exact_matches": clean_exact,
            "clean325_exact_pct": round(100.0 * clean_exact / 325, 2),
            "clean325_empty_outputs": clean_empty,
            "noise_total": noise_total,
            "noise_empty": noise_empty,
            "noise_fp_count": noise_fp,
            "noise_fp_rate_pct": round(100.0 * noise_fp / noise_total, 2),
            "page_primary_cer_pct": round(100.0 * page_c_err / page_c_len, 2),
            "page_primary_wer_pct": round(100.0 * page_w_err / page_w_len, 2),
            "mean_ms_per_crop": round(float(np.mean(latencies)), 2),
            "total_seconds": round(elapsed_s, 1),
            "evaluation_status": "POST_HOC_FAIRNESS_BENCHMARK",
        }
        summary_rows.append(row)
        print(f"Done {cond_name}: Clean325 CER={row['clean325_cer_pct']}%, Exact={clean_exact}/325, Empty={clean_empty}, NoiseFP={row['noise_fp_rate_pct']}%, PageCER={row['page_primary_cer_pct']}%, MeanMs={row['mean_ms_per_crop']}ms ({elapsed_s:.1f}s)")

    out_csv = OUT_DIR / "REAL30_TESSERACT_FAIRNESS.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
