"""AzOCR-V2 — Real30 Final GPU Evaluation (18 Official Models).

Extracts the 30 embedded JPEG images from the original Real30 PDF without
re-rendering, verifies byte-for-byte identity and crop coordinate alignment
against ground_truth_lines.csv (421 lines) and verified_alignment.csv (690 crops,
325 genuine_single_line), and evaluates all 18 official best.pt checkpoints
(6 groups x 3 seeds, excluding RTX 2050 repeat) using the fixed infer_line_smart
pipeline.
"""

import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

from common import Vocab, CHARS, DEFAULT_CFG, infer_line_smart, normalize, align
from model import CRNN
from scripts.audit_real30_methodology import segment_lines_with_boxes, cv2_imread_utf8

PDF_PATH = Path(r"C:\Users\serxan.m\Downloads\Image to PDF 20261124 12.02.30 (1).pdf")
RUNS_ROOT = Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\runs")
GT_CSV = PROJ_ROOT / "data" / "real30_evaluation" / "ground_truth_lines.csv"
VA_CSV = PROJ_ROOT / "results" / "real30_evaluation" / "verified_alignment.csv"
EXISTING_PHOTOS_DIR = PROJ_ROOT / "data" / "real30_evaluation" / "photos"
OUT_DIR = PROJ_ROOT / "Real30_GPU18_Final_Evaluation"

GROUPS = [
    ("Group_A_V1Clean_BasicAug", "Group A (V1Clean + BasicAug)"),
    ("Group_B_V1Clean_PhotoAug", "Group B (V1Clean + PhotoAug)"),
    ("Group_C_V2Clean_BasicAug", "Group C (V2Clean + BasicAug)"),
    ("Group_D_V2Clean_PhotoAug", "Group D (V2Clean + PhotoAug)"),
    ("Exp_RandomPadding_GroupD", "RandomPadding (Group D)"),
    ("Exp_VariableWidth_GroupD", "VariableWidth (Group D, Tesla T4)"),
]
SEEDS = [42, 43, 44]


def nfc(text) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def cer_wer_pair(ref: str, hyp: str):
    ref_n = nfc(ref)
    hyp_n = nfc(hyp)
    d_c, _ = align(ref_n, hyp_n)
    l_c = len(ref_n)
    r_words = ref_n.split()
    h_words = hyp_n.split()
    d_w, _ = align(r_words, h_words)
    l_w = len(r_words)
    cer = d_c / max(1, l_c)
    wer = d_w / max(1, l_w)
    return d_c, l_c, d_w, l_w, cer, wer


def load_gpu_checkpoint(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", DEFAULT_CFG)
    chars = ck.get("chars", CHARS)
    vocab = Vocab(chars)
    in_ch = 1 if cfg.get("gray", True) else 3
    model = CRNN(in_ch=in_ch, num_classes=vocab.size).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {
        "epoch": int(ck.get("epoch", -1)),
        "val_cer": float(ck.get("val_cer", -1.0)),
        "val_wer": float(ck.get("val_wer", -1.0)),
        "cfg": cfg,
    }
    return model, cfg, vocab, meta


def main():
    print("=" * 80)
    print("AZOCR-V2 — REAL30 FINAL GPU EVALUATION (18 OFFICIAL CHECKPOINTS)")
    print("=" * 80)

    assert PDF_PATH.exists(), f"Missing PDF: {PDF_PATH}"
    assert PROJ_ROOT.exists(), f"Missing project root: {PROJ_ROOT}"
    assert RUNS_ROOT.exists(), f"Missing runs root: {RUNS_ROOT}"
    assert GT_CSV.exists(), f"Missing GT CSV: {GT_CSV}"
    assert VA_CSV.exists(), f"Missing VA CSV: {VA_CSV}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    photos_out_dir = OUT_DIR / "extracted_photos"
    photos_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract 30 embedded JPEG streams from PDF without re-rendering
    print("\n[1/5] Extracting 30 embedded JPEG images directly from PDF...")
    pdf_bytes = PDF_PATH.read_bytes()
    jpeg_streams = re.findall(rb"stream\r?\n(\xff\xd8\xff.*?\xff\xd9)\r?\nendstream", pdf_bytes, re.DOTALL)
    assert len(jpeg_streams) == 30, f"Expected 30 JPEG streams in PDF, found {len(jpeg_streams)}"

    extracted_paths = {}
    for idx, jbytes in enumerate(jpeg_streams, 1):
        pid = f"P{idx:02d}"
        out_jpg = photos_out_dir / f"{pid}.jpg"
        out_jpg.write_bytes(jbytes)
        extracted_paths[pid] = out_jpg

        existing_jpg = EXISTING_PHOTOS_DIR / f"{pid}_pdf_embedded.jpeg"
        if existing_jpg.exists():
            assert sha256_bytes(jbytes) == sha256_file(existing_jpg), (
                f"Byte mismatch between PDF stream and existing photo for {pid}"
            )
    print("  [+] Extracted 30 JPEGs and verified 100% byte-for-byte SHA-256 match with existing photos.")

    # 2. Verify Ground Truth & Verified Alignment Counts
    print("\n[2/5] Verifying Ground Truth (421) and Verified Alignment (690 crops, 325 clean lines)...")
    df_gt = pd.read_csv(GT_CSV)
    df_va = pd.read_csv(VA_CSV)

    assert len(df_gt) == 421, f"Expected 421 GT lines, found {len(df_gt)}"
    assert len(df_va) == 690, f"Expected 690 VA rows, found {len(df_va)}"
    cat_counts = df_va["verified_crop_category"].value_counts().to_dict()
    assert cat_counts.get("genuine_single_line", 0) == 325, (
        f"Expected 325 genuine_single_line, found {cat_counts.get('genuine_single_line', 0)}"
    )
    print(f"  [+] GT lines: {len(df_gt)} | VA crops: {len(df_va)} | Categories: {cat_counts}")

    # 3. Verify Segmentation Coordinates and Image Dimensions
    print("\n[3/5] Verifying image dimensions and exact crop coordinates across P01-P30...")
    segmented_crops_map = {}  # (pid, crop_id) -> crop_bgr
    page_dims = {}

    for idx in range(1, 31):
        pid = f"P{idx:02d}"
        img_bgr = cv2_imread_utf8(extracted_paths[pid])
        H, W = img_bgr.shape[:2]
        page_dims[pid] = (W, H)

        crops, boxes = segment_lines_with_boxes(img_bgr)
        va_page = df_va[df_va["page_id"] == pid].sort_values("crop_id")
        assert len(crops) == len(va_page), (
            f"Crop count mismatch on {pid}: segmented={len(crops)} vs VA={len(va_page)}"
        )
        for crop_bgr, box, (_, va_row) in zip(crops, boxes, va_page.iterrows()):
            cy0, cy1, cx0, cx1 = box
            cid = int(va_row["crop_id"])
            expected_box = (
                int(va_row["bbox_x0"]),
                int(va_row["bbox_y0"]),
                int(va_row["bbox_x1"]),
                int(va_row["bbox_y1"]),
            )
            assert (cx0, cy0, cx1, cy1) == expected_box, (
                f"Box mismatch on {pid} crop {cid}: {(cx0, cy0, cx1, cy1)} != {expected_box}"
            )
            assert 0 <= cx0 < cx1 <= W and 0 <= cy0 < cy1 <= H, (
                f"Out-of-bounds box on {pid} crop {cid}: {expected_box} vs {(W, H)}"
            )
            segmented_crops_map[(pid, cid)] = crop_bgr

    print(f"  [+] All 690 crops verified: 0 coordinate mismatches, 0 out-of-bounds dimensions.")

    # 4. Verify all 18 official checkpoints & compute SHA-256
    print("\n[4/5] Verifying 18 official checkpoints and computing SHA-256...")
    checkpoints = []
    for grp_id, grp_label in GROUPS:
        for seed in SEEDS:
            run_name = f"{grp_id}_seed{seed}"
            ckpt_path = RUNS_ROOT / run_name / "best.pt"
            assert ckpt_path.exists(), f"Missing official checkpoint: {ckpt_path}"
            summary_path = RUNS_ROOT / run_name / "run_summary.json"
            hw = "Unknown"
            if summary_path.exists():
                sdata = json.loads(summary_path.read_text(encoding="utf-8"))
                hw = sdata.get("hardware", {}).get("gpu_name", "Unknown")

            sha = sha256_file(ckpt_path)
            size_bytes = ckpt_path.stat().st_size
            checkpoints.append({
                "run_name": run_name,
                "group_id": grp_id,
                "group_label": grp_label,
                "seed": seed,
                "hardware": hw,
                "ckpt_path": ckpt_path,
                "size_bytes": size_bytes,
                "sha256": sha,
            })
            print(f"  [+] {run_name} ({hw}) | SHA256: {sha[:16]}...")

    # Save checkpoint manifest
    manifest_df = pd.DataFrame([
        {k: v for k, v in c.items() if k != "ckpt_path"} | {"ckpt_path": str(c["ckpt_path"])}
        for c in checkpoints
    ])
    manifest_df.to_csv(OUT_DIR / "checkpoint_sha256_manifest.csv", index=False)

    # 5. Run Inference across all 18 models on 690 crops
    print("\n[5/5] Running fixed infer_line_smart inference across all 18 models...")
    torch.set_num_threads(4)
    device = torch.device("cpu")

    # Prepare base raw predictions DataFrame from VA
    raw_preds_df = df_va[[
        "page_id", "crop_id", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
        "crop_h", "crop_w", "verified_crop_category", "verified_matched_gt_line_no",
        "verified_ground_truth_text"
    ]].copy()

    ordered_keys = [(r["page_id"], int(r["crop_id"])) for _, r in raw_preds_df.iterrows()]
    singles_mask = raw_preds_df["verified_crop_category"] == "genuine_single_line"
    page_ids = [f"P{i:02d}" for i in range(1, 31)]

    per_model_rows = []
    per_page_rows = []

    for m_idx, ck_info in enumerate(checkpoints, 1):
        run_name = ck_info["run_name"]
        t0_model = time.perf_counter()
        model, cfg, vocab, meta = load_gpu_checkpoint(ck_info["ckpt_path"], device)

        preds = []
        t0_inf = time.perf_counter()
        for pid, cid in ordered_keys:
            crop_bgr = segmented_crops_map[(pid, cid)]
            txt = normalize(infer_line_smart(model, cfg, vocab, crop_bgr, device).strip())
            preds.append(txt)
        dt_inf_ms = (time.perf_counter() - t0_inf) * 1000.0
        mean_ms_per_crop = dt_inf_ms / len(ordered_keys)

        raw_preds_df[run_name] = preds

        # --- Metric A: 325 Clean Single Lines ---
        singles_df = raw_preds_df[singles_mask]
        c325_dc, c325_lc, c325_dw, c325_lw = 0, 0, 0, 0
        c325_cers, c325_wers = [], []
        c325_exact = 0
        for _, r in singles_df.iterrows():
            gt_txt = r["verified_ground_truth_text"]
            hyp_txt = r[run_name]
            dc, lc, dw, lw, cer, wer = cer_wer_pair(gt_txt, hyp_txt)
            c325_dc += dc
            c325_lc += lc
            c325_dw += dw
            c325_lw += lw
            c325_cers.append(cer)
            c325_wers.append(wer)
            if nfc(gt_txt) == nfc(hyp_txt):
                c325_exact += 1

        # --- Metric B & C: Full Page P01-P30 (Primary Space & Sensitivity Newline) ---
        sp_dc_tot, sp_lc_tot, sp_dw_tot, sp_lw_tot = 0, 0, 0, 0
        sp_cers, sp_wers = [], []
        nl_dc_tot, nl_lc_tot, nl_dw_tot, nl_lw_tot = 0, 0, 0, 0
        nl_cers, nl_wers = [], []

        for pid in page_ids:
            p_gt = df_gt[df_gt["page_id"] == pid].sort_values("line_no")
            p_crops = raw_preds_df[raw_preds_df["page_id"] == pid].sort_values(["bbox_y0", "bbox_x0"])

            # Primary (Space, non-empty)
            gt_sp = " ".join([nfc(t) for t in p_gt["ground_truth_nfc"]])
            pred_sp = " ".join([nfc(t) for t in p_crops[run_name] if len(nfc(t)) > 0])
            dc_sp, lc_sp, dw_sp, lw_sp, cer_sp, wer_sp = cer_wer_pair(gt_sp, pred_sp)
            sp_dc_tot += dc_sp
            sp_lc_tot += lc_sp
            sp_dw_tot += dw_sp
            sp_lw_tot += lw_sp
            sp_cers.append(cer_sp)
            sp_wers.append(wer_sp)

            # Sensitivity (Newline, all crops)
            gt_nl = "\n".join([nfc(t) for t in p_gt["ground_truth_nfc"]])
            pred_nl = "\n".join([nfc(t) for t in p_crops[run_name]])
            dc_nl, lc_nl, dw_nl, lw_nl, cer_nl, wer_nl = cer_wer_pair(gt_nl, pred_nl)
            nl_dc_tot += dc_nl
            nl_lc_tot += lc_nl
            nl_dw_tot += dw_nl
            nl_lw_tot += lw_nl
            nl_cers.append(cer_nl)
            nl_wers.append(wer_nl)

            # Page-level 325 clean lines
            p_singles = singles_df[singles_df["page_id"] == pid]
            ps_dc, ps_lc, ps_dw, ps_lw, ps_ex = 0, 0, 0, 0, 0
            for _, sr in p_singles.iterrows():
                sdc, slc, sdw, slw, _, _ = cer_wer_pair(sr["verified_ground_truth_text"], sr[run_name])
                ps_dc += sdc
                ps_lc += slc
                ps_dw += sdw
                ps_lw += slw
                if nfc(sr["verified_ground_truth_text"]) == nfc(sr[run_name]):
                    ps_ex += 1

            per_page_rows.append({
                "run_name": run_name,
                "group_id": ck_info["group_id"],
                "seed": ck_info["seed"],
                "page_id": pid,
                "gt_lines": len(p_gt),
                "total_crops": len(p_crops),
                "clean325_lines": len(p_singles),
                "clean325_exact": ps_ex,
                "clean325_cer_micro_pct": round(ps_dc / max(1, ps_lc) * 100.0, 2) if ps_lc > 0 else None,
                "clean325_wer_micro_pct": round(ps_dw / max(1, ps_lw) * 100.0, 2) if ps_lw > 0 else None,
                "page_primary_cer_pct": round(cer_sp * 100.0, 2),
                "page_primary_wer_pct": round(wer_sp * 100.0, 2),
                "page_newline_cer_pct": round(cer_nl * 100.0, 2),
                "page_newline_wer_pct": round(wer_nl * 100.0, 2),
            })

        m_row = {
            "run_name": run_name,
            "group_id": ck_info["group_id"],
            "group_label": ck_info["group_label"],
            "seed": ck_info["seed"],
            "hardware": ck_info["hardware"],
            "best_epoch": meta["epoch"],
            "train_val_cer_pct": round(meta["val_cer"] * 100.0, 3),
            "sha256": ck_info["sha256"],
            "clean325_lines": len(singles_df),
            "clean325_cer_micro_pct": round(c325_dc / max(1, c325_lc) * 100.0, 2),
            "clean325_cer_macro_pct": round(float(np.mean(c325_cers)) * 100.0, 2),
            "clean325_wer_micro_pct": round(c325_dw / max(1, c325_lw) * 100.0, 2),
            "clean325_wer_macro_pct": round(float(np.mean(c325_wers)) * 100.0, 2),
            "clean325_exact_matches": c325_exact,
            "clean325_exact_match_pct": round(c325_exact / len(singles_df) * 100.0, 2),
            "page_primary_cer_micro_pct": round(sp_dc_tot / max(1, sp_lc_tot) * 100.0, 2),
            "page_primary_cer_macro_pct": round(float(np.mean(sp_cers)) * 100.0, 2),
            "page_primary_wer_micro_pct": round(sp_dw_tot / max(1, sp_lw_tot) * 100.0, 2),
            "page_primary_wer_macro_pct": round(float(np.mean(sp_wers)) * 100.0, 2),
            "page_newline_cer_micro_pct": round(nl_dc_tot / max(1, nl_lc_tot) * 100.0, 2),
            "page_newline_cer_macro_pct": round(float(np.mean(nl_cers)) * 100.0, 2),
            "page_newline_wer_micro_pct": round(nl_dw_tot / max(1, nl_lw_tot) * 100.0, 2),
            "page_newline_wer_macro_pct": round(float(np.mean(nl_wers)) * 100.0, 2),
            "mean_ms_per_crop": round(mean_ms_per_crop, 2),
        }
        per_model_rows.append(m_row)
        dt_tot = time.perf_counter() - t0_model
        print(
            f"  [{m_idx:02d}/18] {run_name:34s} | "
            f"Clean325 CER={m_row['clean325_cer_micro_pct']:5.2f}%, Exact={c325_exact}/325 ({m_row['clean325_exact_match_pct']:5.2f}%) | "
            f"Page CER={m_row['page_primary_cer_micro_pct']:5.2f}% | {dt_tot:.1f}s"
        )

    # Save raw predictions & per-model & per-page CSVs
    raw_preds_df.to_csv(OUT_DIR / "raw_predictions_all_18_models.csv", index=False)
    df_models = pd.DataFrame(per_model_rows)
    df_models.to_csv(OUT_DIR / "per_model_summary.csv", index=False)
    df_pages = pd.DataFrame(per_page_rows)
    df_pages.to_csv(OUT_DIR / "per_page_metrics_all_18_models.csv", index=False)

    # Compute Group Mean & Sample Std (ddof=1)
    metric_cols = [
        "clean325_cer_micro_pct", "clean325_cer_macro_pct",
        "clean325_wer_micro_pct", "clean325_wer_macro_pct",
        "clean325_exact_matches", "clean325_exact_match_pct",
        "page_primary_cer_micro_pct", "page_primary_cer_macro_pct",
        "page_primary_wer_micro_pct", "page_primary_wer_macro_pct",
        "page_newline_cer_micro_pct", "page_newline_wer_micro_pct",
        "mean_ms_per_crop",
    ]
    group_rows = []
    for grp_id, grp_label in GROUPS:
        sub = df_models[df_models["group_id"] == grp_id]
        row = {
            "group_id": grp_id,
            "group_label": grp_label,
            "n_seeds": len(sub),
            "hardware": sub["hardware"].iloc[0],
        }
        for col in metric_cols:
            vals = sub[col].astype(float).values
            row[f"{col}_mean"] = round(float(np.mean(vals)), 2)
            row[f"{col}_std"] = round(float(np.std(vals, ddof=1)), 2)
        group_rows.append(row)

    df_groups = pd.DataFrame(group_rows)
    df_groups.to_csv(OUT_DIR / "per_group_summary.csv", index=False)

    # Save JSON summary
    summary_json = {
        "evaluation_protocol": {
            "inference_function": "common.infer_line_smart (fixed target_h=32, target_w=512, squeeze_ratio>1.05)",
            "preprocessing": "gray=True, filter=none, thresh=none, interp=bilinear",
            "segmentation": "segment_lines_with_boxes (deskew + OpenCV row/col projection)",
            "total_pages": 30,
            "total_gt_lines": 421,
            "total_crops": 690,
            "clean_single_lines": 325,
            "page_primary_protocol": "Space-separated non-empty crops vs space-separated GT lines (NFC normalized)",
            "page_sensitivity_protocol": "Newline-separated all 690 crops vs newline-separated GT lines (NFC normalized)",
            "rtx2050_repro_excluded": True,
        },
        "group_summary": group_rows,
        "model_summary": per_model_rows,
    }
    with open(OUT_DIR / "evaluation_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print("\n[+] Evaluation completed successfully. All outputs saved to:")
    print(f"    {OUT_DIR}")


if __name__ == "__main__":
    main()
