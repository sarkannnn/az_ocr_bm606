"""Task 2: Real30 Clean325 Padding Sensitivity (0, 4, 8, 12, 20 px) across Group D, RandomPadding, VariableWidth (9 official models).
Keeps Canvas-Border Padding and Tight-Ink Margin protocols strictly separate.
"""
import csv
import io
import math
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"d:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(ROOT))

from common import Vocab, CHARS, DEFAULT_CFG, align, normalize, infer_line_smart, infer_single_crop, split_line_into_chunks
from model import CRNN
from scripts.audit_real30_methodology import segment_lines_with_boxes, cv2_imread_utf8
from scripts.eval_padding_robustness import trim_chunk_with_margin

ZIP_PATH = ROOT / "AzOCR_Zenodo_v1_Release" / "AzOCR_v1_Models_18Official.zip"
PHOTOS_DIR = ROOT / "Real30_GPU18_Final_Evaluation" / "extracted_photos"
VA_CSV = ROOT / "results" / "real30_evaluation" / "verified_alignment.csv"
OUT_DIR = ROOT / "FINAL_REMAINING_CHECKS"
OUT_DIR.mkdir(exist_ok=True)

TARGET_RUNS = [
    ("Group_D_V2Clean_PhotoAug", "Group_D_V2Clean_PhotoAug_seed42", 42),
    ("Group_D_V2Clean_PhotoAug", "Group_D_V2Clean_PhotoAug_seed43", 43),
    ("Group_D_V2Clean_PhotoAug", "Group_D_V2Clean_PhotoAug_seed44", 44),
    ("Exp_RandomPadding_GroupD", "Exp_RandomPadding_GroupD_seed42", 42),
    ("Exp_RandomPadding_GroupD", "Exp_RandomPadding_GroupD_seed43", 43),
    ("Exp_RandomPadding_GroupD", "Exp_RandomPadding_GroupD_seed44", 44),
    ("Exp_VariableWidth_GroupD", "Exp_VariableWidth_GroupD_seed42", 42),
    ("Exp_VariableWidth_GroupD", "Exp_VariableWidth_GroupD_seed43", 43),
    ("Exp_VariableWidth_GroupD", "Exp_VariableWidth_GroupD_seed44", 44),
]
PADDINGS = [0, 4, 8, 12, 20]


def nfc(s) -> str:
    if pd.isna(s):
        return ""
    return unicodedata.normalize("NFC", str(s).strip())


def count_chunks(img_bgr: np.ndarray, cfg: dict) -> int:
    h_c, w_c = img_bgr.shape[:2]
    target_h = cfg.get("height", 32)
    target_w = cfg.get("width", 512)
    squeeze_ratio = (w_c * target_h) / (h_c * target_w)
    return max(1, math.ceil(squeeze_ratio)) if squeeze_ratio > 1.05 else 1


def infer_tight_ink_margin(model, cfg, vocab, line_bgr, device, margin_px: int):
    n_chunks = count_chunks(line_bgr, cfg)
    raw_chunks = split_line_into_chunks(line_bgr, n_chunks)
    preds = []
    for ch in raw_chunks:
        ch_trimmed = trim_chunk_with_margin(ch, margin=margin_px)
        p = infer_single_crop(model, cfg, vocab, ch_trimmed, device).strip()
        if p:
            preds.append(p)
    return " ".join(preds), n_chunks


def main():
    torch.set_num_threads(4)
    device = torch.device("cpu")

    df_va = pd.read_csv(VA_CSV)
    clean_items = []
    for idx in range(1, 31):
        pid = f"P{idx:02d}"
        img_bgr = cv2_imread_utf8(PHOTOS_DIR / f"{pid}.jpg")
        crops, _ = segment_lines_with_boxes(img_bgr)
        va_page = df_va[df_va["page_id"] == pid].sort_values("crop_id")
        for crop_bgr, (_, va_row) in zip(crops, va_page.iterrows()):
            if va_row["verified_crop_category"] == "genuine_single_line":
                clean_items.append((pid, int(va_row["crop_id"]), nfc(va_row["verified_ground_truth_text"]), crop_bgr))
    assert len(clean_items) == 325

    rows = []
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for grp_id, run_id, seed in TARGET_RUNS:
            ckpt_bytes = zf.read(f"{run_id}/best.pt")
            ck = torch.load(io.BytesIO(ckpt_bytes), map_location=device, weights_only=False)
            cfg = ck.get("cfg", DEFAULT_CFG)
            vocab = Vocab(ck.get("chars", CHARS))
            in_ch = 1 if cfg.get("gray", True) else 3
            model = CRNN(in_ch=in_ch, num_classes=vocab.size).to(device)
            model.load_state_dict(ck["model"])
            model.eval()
            print(f"[+] Loaded {run_id}")

            for proto in ["Canvas_Border_Padding", "Tight_Ink_Margin"]:
                for pad in PADDINGS:
                    t0 = time.perf_counter()
                    c_err = c_len = w_err = w_len = exact = total_chunks = 0
                    for _, _, gt_raw, crop_bgr in clean_items:
                        if proto == "Canvas_Border_Padding":
                            img_in = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255]) if pad > 0 else crop_bgr
                            n_ch = count_chunks(img_in, cfg)
                            pred_raw = infer_line_smart(model, cfg, vocab, img_in, device)
                        else:
                            pred_raw, n_ch = infer_tight_ink_margin(model, cfg, vocab, crop_bgr, device, pad)
                        total_chunks += n_ch
                        gt = normalize(gt_raw.replace("—", "-"))
                        pred = normalize(pred_raw.replace("—", "-"))
                        ce, _ = align(gt, pred)
                        we, _ = align(gt.split(), pred.split())
                        c_err += ce
                        c_len += len(gt)
                        w_err += we
                        w_len += len(gt.split())
                        if gt == pred:
                            exact += 1

                    dt = time.perf_counter() - t0
                    rows.append({
                        "protocol": proto,
                        "group_id": grp_id,
                        "run_id": run_id,
                        "seed": seed,
                        "padding_px": pad,
                        "clean325_lines": 325,
                        "total_gt_chars": c_len,
                        "total_chunks": total_chunks,
                        "cer_pct": round(100.0 * c_err / c_len, 2),
                        "wer_pct": round(100.0 * w_err / w_len, 2),
                        "exact_matches": exact,
                        "exact_match_pct": round(100.0 * exact / 325, 2),
                        "mean_ms_per_crop": round(1000.0 * dt / 325, 2),
                        "evaluation_status": "POST_HOC_PADDING_SENSITIVITY",
                    })
                    print(f"  {proto:21s} pad={pad:2d}px -> CER={rows[-1]['cer_pct']:5.2f}%, Exact={exact:2d}/325, Chunks={total_chunks}, {rows[-1]['mean_ms_per_crop']:.1f}ms")

    out_csv = OUT_DIR / "REAL30_PADDING_SENSITIVITY.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
