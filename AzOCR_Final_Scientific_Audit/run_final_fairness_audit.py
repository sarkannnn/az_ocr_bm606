"""AzOCR-V2 — Final Baseline Fairness & Reproducibility Audit.

Compares the 18 official GPU checkpoints (evaluated in Real30_GPU18_Final_Evaluation)
against the 4 existing baselines (CRNN-V2 Raw, Tesseract 5.4 PSM 7, EasyOCR, PaddleOCR)
using identical P01-P30 photos, 421 GT lines, 690 fixed crops, 325 genuine_single_line,
Unicode NFC normalization, and Levenshtein alignment.
"""

import csv
import hashlib
import json
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

from common import align

GPU18_DIR = PROJ_ROOT / "Real30_GPU18_Final_Evaluation"
GT_CSV = PROJ_ROOT / "data" / "real30_evaluation" / "ground_truth_lines.csv"
VA_CSV = PROJ_ROOT / "results" / "real30_evaluation" / "verified_alignment.csv"
OUT_DIR = PROJ_ROOT / "AzOCR_Final_Scientific_Audit"
CODE_SNAP_DIR = OUT_DIR / "code_snapshot"

BASELINES = [
    ("crnn_v2_raw", "Baseline_CRNN-V2_Raw", "CPU (Local)", "infer_line_smart (target_h=32, target_w=512, runs/line_v2/best.pt)"),
    ("tesseract5_psm7", "Baseline_Tesseract5_PSM7", "CPU (Local)", "Tesseract v5.4.0.20240606 (--psm 7 -l aze)"),
    ("easyocr", "Baseline_EasyOCR", "CPU (Local)", "EasyOCR v1.7.2 (lang=['az'])"),
    ("paddleocr", "Baseline_PaddleOCR", "CPU (Local)", "PaddleX PP-OCRv6_medium_rec"),
]

GROUPS = [
    ("Group_A_V1Clean_BasicAug", "Group A (V1Clean + BasicAug)"),
    ("Group_B_V1Clean_PhotoAug", "Group B (V1Clean + PhotoAug)"),
    ("Group_C_V2Clean_BasicAug", "Group C (V2Clean + BasicAug)"),
    ("Group_D_V2Clean_PhotoAug", "Group D (V2Clean + PhotoAug)"),
    ("Exp_RandomPadding_GroupD", "RandomPadding (Group D)"),
    ("Exp_VariableWidth_GroupD", "VariableWidth (Group D, Tesla T4)"),
]
SEEDS = [42, 43, 44]
TARGET_CHARS = ["ə", "Ə", "ı", "i", "İ", "I"]


def nfc(text) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


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
    return d_c, l_c, d_w, l_w, d_c / max(1, l_c), d_w / max(1, l_w)


def compute_model_metrics(df_gt: pd.DataFrame, df_preds: pd.DataFrame, col_name: str):
    """Compute Clean325, Full-Page Primary/Newline, and 314-Noise metrics identically."""
    page_ids = [f"P{i:02d}" for i in range(1, 31)]
    singles_df = df_preds[df_preds["verified_crop_category"] == "genuine_single_line"]
    noise_df = df_preds[df_preds["verified_crop_category"] == "noise"]

    # 1. Clean 325
    c325_dc, c325_lc, c325_dw, c325_lw = 0, 0, 0, 0
    c325_cers, c325_wers = [], []
    c325_exact = 0
    per_page_clean = {}

    for pid in page_ids:
        p_s = singles_df[singles_df["page_id"] == pid]
        p_dc, p_lc = 0, 0
        for _, r in p_s.iterrows():
            dc, lc, dw, lw, cer, wer = cer_wer_pair(r["verified_ground_truth_text"], r[col_name])
            c325_dc += dc; c325_lc += lc; c325_dw += dw; c325_lw += lw
            p_dc += dc; p_lc += lc
            c325_cers.append(cer); c325_wers.append(wer)
            if nfc(r["verified_ground_truth_text"]) == nfc(r[col_name]):
                c325_exact += 1
        per_page_clean[pid] = (p_dc, p_lc)

    # 2. Full Page Primary (Space) & Sensitivity (Newline)
    sp_dc_tot, sp_lc_tot, sp_dw_tot, sp_lw_tot = 0, 0, 0, 0
    sp_cers, sp_wers = [], []
    nl_dc_tot, nl_lc_tot, nl_dw_tot, nl_lw_tot = 0, 0, 0, 0
    nl_cers, nl_wers = [], []
    per_page_sp = {}

    for pid in page_ids:
        p_gt = df_gt[df_gt["page_id"] == pid].sort_values("line_no")
        p_crops = df_preds[df_preds["page_id"] == pid].sort_values(["bbox_y0", "bbox_x0"])

        gt_sp = " ".join([nfc(t) for t in p_gt["ground_truth_nfc"]])
        pred_sp = " ".join([nfc(t) for t in p_crops[col_name] if len(nfc(t)) > 0])
        dc_sp, lc_sp, dw_sp, lw_sp, cer_sp, wer_sp = cer_wer_pair(gt_sp, pred_sp)
        sp_dc_tot += dc_sp; sp_lc_tot += lc_sp; sp_dw_tot += dw_sp; sp_lw_tot += lw_sp
        sp_cers.append(cer_sp); sp_wers.append(wer_sp)
        per_page_sp[pid] = (dc_sp, lc_sp)

        gt_nl = "\n".join([nfc(t) for t in p_gt["ground_truth_nfc"]])
        pred_nl = "\n".join([nfc(t) for t in p_crops[col_name]])
        dc_nl, lc_nl, dw_nl, lw_nl, cer_nl, wer_nl = cer_wer_pair(gt_nl, pred_nl)
        nl_dc_tot += dc_nl; nl_lc_tot += lc_nl; nl_dw_tot += dw_nl; nl_lw_tot += lw_nl
        nl_cers.append(cer_nl); nl_wers.append(wer_nl)

    # 3. Noise 314
    noise_lens = [len(nfc(t)) for t in noise_df[col_name]]
    noise_fp_count = sum(1 for l in noise_lens if l > 0)
    noise_total_chars = sum(noise_lens)
    total_output_chars = sum(len(nfc(t)) for t in df_preds[col_name])

    summary = {
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
        "page_newline_wer_micro_pct": round(nl_dw_tot / max(1, nl_lw_tot) * 100.0, 2),
        "noise314_fp_count": noise_fp_count,
        "noise314_fp_rate_pct": round(noise_fp_count / len(noise_df) * 100.0, 2),
        "noise314_total_chars": noise_total_chars,
        "noise314_mean_chars": round(noise_total_chars / len(noise_df), 2),
        "total_output_chars_690": total_output_chars,
        "noise_share_of_output_pct": round(noise_total_chars / max(1, total_output_chars) * 100.0, 2),
    }
    return summary, per_page_clean, per_page_sp


def paired_bootstrap_ci(page_ids, stats_a, stats_b, n_boot=10000, seed=42):
    """Paired bootstrap over 30 pages for micro CER difference (A - B, in percentage points)."""
    rng = np.random.default_rng(seed)
    n = len(page_ids)
    dc_a = np.array([stats_a[p][0] for p in page_ids], dtype=np.float64)
    lc_a = np.array([stats_a[p][1] for p in page_ids], dtype=np.float64)
    dc_b = np.array([stats_b[p][0] for p in page_ids], dtype=np.float64)
    lc_b = np.array([stats_b[p][1] for p in page_ids], dtype=np.float64)

    obs_diff = (dc_a.sum() / max(1.0, lc_a.sum()) - dc_b.sum() / max(1.0, lc_b.sum())) * 100.0
    boot_diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cer_a = dc_a[idx].sum() / max(1.0, lc_a[idx].sum())
        cer_b = dc_b[idx].sum() / max(1.0, lc_b[idx].sum())
        boot_diffs[b] = (cer_a - cer_b) * 100.0

    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    return round(float(obs_diff), 2), round(float(lo), 2), round(float(hi), 2)


def main():
    print("=" * 80)
    print("AZOCR-V2 — FINAL BASELINE FAIRNESS & REPRODUCIBILITY AUDIT")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CODE_SNAP_DIR.mkdir(parents=True, exist_ok=True)

    df_gt = pd.read_csv(GT_CSV)
    df_va = pd.read_csv(VA_CSV)
    df_raw18 = pd.read_csv(GPU18_DIR / "raw_predictions_all_18_models.csv")
    df_gpu_models = pd.read_csv(GPU18_DIR / "per_model_summary.csv")

    # Verify strict crop-for-crop alignment
    assert len(df_va) == 690 and len(df_raw18) == 690
    assert (df_va["page_id"] == df_raw18["page_id"]).all()
    assert (df_va["crop_id"] == df_raw18["crop_id"]).all()
    assert (df_va["bbox_x0"] == df_raw18["bbox_x0"]).all()
    assert (df_va["bbox_y1"] == df_raw18["bbox_y1"]).all()
    assert (df_va["verified_crop_category"] == df_raw18["verified_crop_category"]).all()
    print("[+] Verified 100% crop-for-crop equivalence (421 GT, 690 crops, 325 genuine_single_line).")

    # Merge baseline columns into unified predictions frame
    df_all = df_raw18.copy()
    for col, _, _, _ in BASELINES:
        df_all[col] = df_va[col]

    page_ids = [f"P{i:02d}" for i in range(1, 31)]
    comparison_rows = []
    page_clean_dict = {}
    page_sp_dict = {}
    noise_rows = []

    # 1. Compute Baseline metrics directly from raw predictions
    print("\n[1/4] Computing exact metrics for 4 Baselines...")
    for col, disp_name, hw, proto in BASELINES:
        summ, p_clean, p_sp = compute_model_metrics(df_gt, df_all, col)
        page_clean_dict[disp_name] = p_clean
        page_sp_dict[disp_name] = p_sp
        comparison_rows.append({
            "entry_type": "Baseline",
            "model_or_group": disp_name,
            "hardware": hw,
            "inference_protocol": proto,
            **{k: summ[k] for k in [
                "clean325_lines", "clean325_cer_micro_pct", "clean325_cer_macro_pct",
                "clean325_wer_micro_pct", "clean325_wer_macro_pct",
                "clean325_exact_matches", "clean325_exact_match_pct",
                "page_primary_cer_micro_pct", "page_primary_cer_macro_pct",
                "page_primary_wer_micro_pct", "page_primary_wer_macro_pct",
                "page_newline_cer_micro_pct", "page_newline_wer_micro_pct",
                "noise314_fp_rate_pct", "noise314_total_chars"
            ]}
        })
        noise_rows.append({
            "entry_type": "Baseline",
            "model_name": disp_name,
            "noise314_count": 314,
            "noise_fp_count": summ["noise314_fp_count"],
            "noise_fp_rate_pct": summ["noise314_fp_rate_pct"],
            "noise_total_chars": summ["noise314_total_chars"],
            "noise_mean_chars": summ["noise314_mean_chars"],
            "total_output_chars_690": summ["total_output_chars_690"],
            "noise_share_of_output_pct": summ["noise_share_of_output_pct"],
        })
        print(f"  {disp_name:25s} | Clean325 CER={summ['clean325_cer_micro_pct']:5.2f}%, Exact={summ['clean325_exact_matches']}/325 | Page CER={summ['page_primary_cer_micro_pct']:5.2f}% | NoiseFP={summ['noise314_fp_rate_pct']:5.2f}% ({summ['noise314_total_chars']} chars)")

    # 2. Compute / Verify 18 GPU models
    print("\n[2/4] Computing & verifying metrics for 18 GPU Checkpoints...")
    gpu_model_summaries = {}
    for _, gm in df_gpu_models.iterrows():
        rname = gm["run_name"]
        summ, p_clean, p_sp = compute_model_metrics(df_gt, df_all, rname)
        # Cross-verify against Real30_GPU18_Final_Evaluation
        assert summ["clean325_cer_micro_pct"] == gm["clean325_cer_micro_pct"], f"Mismatch on {rname}"
        assert summ["page_primary_cer_micro_pct"] == gm["page_primary_cer_micro_pct"], f"Mismatch on {rname}"
        page_clean_dict[rname] = p_clean
        page_sp_dict[rname] = p_sp
        gpu_model_summaries[rname] = summ

        comparison_rows.append({
            "entry_type": "GPU_Model",
            "model_or_group": rname,
            "hardware": gm["hardware"],
            "inference_protocol": "infer_line_smart (target_h=32, target_w=512)",
            **{k: summ[k] for k in [
                "clean325_lines", "clean325_cer_micro_pct", "clean325_cer_macro_pct",
                "clean325_wer_micro_pct", "clean325_wer_macro_pct",
                "clean325_exact_matches", "clean325_exact_match_pct",
                "page_primary_cer_micro_pct", "page_primary_cer_macro_pct",
                "page_primary_wer_micro_pct", "page_primary_wer_macro_pct",
                "page_newline_cer_micro_pct", "page_newline_wer_micro_pct",
                "noise314_fp_rate_pct", "noise314_total_chars"
            ]}
        })
        noise_rows.append({
            "entry_type": "GPU_Model",
            "model_name": rname,
            "noise314_count": 314,
            "noise_fp_count": summ["noise314_fp_count"],
            "noise_fp_rate_pct": summ["noise314_fp_rate_pct"],
            "noise_total_chars": summ["noise314_total_chars"],
            "noise_mean_chars": summ["noise314_mean_chars"],
            "total_output_chars_690": summ["total_output_chars_690"],
            "noise_share_of_output_pct": summ["noise_share_of_output_pct"],
        })

    # Add Group Mean ± SD rows to comparison_rows and build 3-seed averaged page dictionaries
    for grp_id, grp_label in GROUPS:
        sub = [gpu_model_summaries[f"{grp_id}_seed{s}"] for s in SEEDS]
        hw = df_gpu_models[df_gpu_models["group_id"] == grp_id]["hardware"].iloc[0]
        row = {
            "entry_type": "GPU_Group_Mean",
            "model_or_group": f"{grp_id} (Mean ± SD)",
            "hardware": hw,
            "inference_protocol": "3-seed Mean ± Sample SD (ddof=1), infer_line_smart",
            "clean325_lines": 325,
        }
        for k in [
            "clean325_cer_micro_pct", "clean325_cer_macro_pct",
            "clean325_wer_micro_pct", "clean325_wer_macro_pct",
            "clean325_exact_matches", "clean325_exact_match_pct",
            "page_primary_cer_micro_pct", "page_primary_cer_macro_pct",
            "page_primary_wer_micro_pct", "page_primary_wer_macro_pct",
            "page_newline_cer_micro_pct", "page_newline_wer_micro_pct",
            "noise314_fp_rate_pct", "noise314_total_chars"
        ]:
            vals = np.array([s[k] for s in sub], dtype=float)
            row[k] = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
        comparison_rows.append(row)

        # Group 3-seed mean per-page stats for bootstrap
        grp_clean, grp_sp = {}, {}
        for pid in page_ids:
            grp_clean[pid] = (
                float(np.mean([page_clean_dict[f"{grp_id}_seed{s}"][pid][0] for s in SEEDS])),
                float(np.mean([page_clean_dict[f"{grp_id}_seed{s}"][pid][1] for s in SEEDS])),
            )
            grp_sp[pid] = (
                float(np.mean([page_sp_dict[f"{grp_id}_seed{s}"][pid][0] for s in SEEDS])),
                float(np.mean([page_sp_dict[f"{grp_id}_seed{s}"][pid][1] for s in SEEDS])),
            )
        page_clean_dict[grp_id] = grp_clean
        page_sp_dict[grp_id] = grp_sp

    pd.DataFrame(comparison_rows).to_csv(OUT_DIR / "BASELINE_COMPARISON.csv", index=False)
    print(f"  [+] Saved {OUT_DIR / 'BASELINE_COMPARISON.csv'}")

    # 3. Statistical Tests: Paired Seed Differences & 30-Page Paired Bootstrap CIs
    print("\n[3/4] Running Paired Seed Differences and 30-Page Paired Bootstrap CIs...")
    stat_rows = []

    seed_pairs = [
        ("Exp_RandomPadding_GroupD", "Group_D_V2Clean_PhotoAug", "RandomPadding vs Group D"),
        ("Group_B_V1Clean_PhotoAug", "Group_A_V1Clean_BasicAug", "Group B (PhotoAug) vs Group A (BasicAug)"),
        ("Group_D_V2Clean_PhotoAug", "Group_C_V2Clean_BasicAug", "Group D (PhotoAug) vs Group C (BasicAug)"),
    ]
    seed_metrics = [
        "clean325_cer_micro_pct", "clean325_wer_micro_pct",
        "clean325_exact_match_pct", "page_primary_cer_micro_pct", "page_primary_wer_micro_pct"
    ]
    for grp_a, grp_b, label in seed_pairs:
        for met in seed_metrics:
            diffs = [
                gpu_model_summaries[f"{grp_a}_seed{s}"][met] - gpu_model_summaries[f"{grp_b}_seed{s}"][met]
                for s in SEEDS
            ]
            mean_d = float(np.mean(diffs))
            sd_d = float(np.std(diffs, ddof=1))
            t_stat, p_val = stats.ttest_rel(
                [gpu_model_summaries[f"{grp_a}_seed{s}"][met] for s in SEEDS],
                [gpu_model_summaries[f"{grp_b}_seed{s}"][met] for s in SEEDS],
            )
            stat_rows.append({
                "test_type": "Paired_Seed_Diff_n3",
                "comparison": label,
                "metric": met,
                "unit_or_scope": "3 paired seeds (42,43,44); df=2",
                "seed42_or_obs_diff": round(diffs[0], 2),
                "seed43_or_ci95_low": round(diffs[1], 2),
                "seed44_or_ci95_high": round(diffs[2], 2),
                "mean_diff": round(mean_d, 2),
                "sd_diff_ddof1": round(sd_d, 2),
                "t_stat": round(float(t_stat), 4) if not np.isnan(t_stat) else "N/A",
                "p_value_two_sided": round(float(p_val), 4) if not np.isnan(p_val) else "N/A",
                "note": "Exploratory df=2; negative CER/WER diff means first group is better (lower error).",
            })

    boot_pairs = [
        ("Exp_RandomPadding_GroupD", "Group_D_V2Clean_PhotoAug", "RandomPadding (3-seed mean) vs Group D (3-seed mean)"),
        ("Group_B_V1Clean_PhotoAug", "Group_A_V1Clean_BasicAug", "Group B (3-seed mean) vs Group A (3-seed mean)"),
        ("Group_D_V2Clean_PhotoAug", "Group_C_V2Clean_BasicAug", "Group D (3-seed mean) vs Group C (3-seed mean)"),
        ("Exp_RandomPadding_GroupD", "Baseline_CRNN-V2_Raw", "RandomPadding (3-seed mean) vs Baseline CRNN-V2 Raw"),
        ("Exp_RandomPadding_GroupD", "Baseline_Tesseract5_PSM7", "RandomPadding (3-seed mean) vs Baseline Tesseract 5 PSM 7"),
        ("Exp_RandomPadding_GroupD", "Baseline_EasyOCR", "RandomPadding (3-seed mean) vs Baseline EasyOCR"),
        ("Exp_RandomPadding_GroupD", "Baseline_PaddleOCR", "RandomPadding (3-seed mean) vs Baseline PaddleOCR"),
        ("Group_D_V2Clean_PhotoAug", "Baseline_CRNN-V2_Raw", "Group D (3-seed mean) vs Baseline CRNN-V2 Raw"),
        ("Group_D_V2Clean_PhotoAug", "Baseline_Tesseract5_PSM7", "Group D (3-seed mean) vs Baseline Tesseract 5 PSM 7"),
        ("Group_D_V2Clean_PhotoAug", "Baseline_EasyOCR", "Group D (3-seed mean) vs Baseline EasyOCR"),
        ("Group_D_V2Clean_PhotoAug", "Baseline_PaddleOCR", "Group D (3-seed mean) vs Baseline PaddleOCR"),
    ]
    for a_key, b_key, label in boot_pairs:
        obs_c, lo_c, hi_c = paired_bootstrap_ci(page_ids, page_clean_dict[a_key], page_clean_dict[b_key])
        stat_rows.append({
            "test_type": "Paired_Bootstrap_30Pages_10k",
            "comparison": label,
            "metric": "clean325_cer_micro_pct_diff",
            "unit_or_scope": "30 pages (P01-P30), 10000 bootstrap resamples",
            "seed42_or_obs_diff": obs_c,
            "seed43_or_ci95_low": lo_c,
            "seed44_or_ci95_high": hi_c,
            "mean_diff": obs_c,
            "sd_diff_ddof1": "N/A",
            "t_stat": "N/A",
            "p_value_two_sided": "N/A",
            "note": "Captures page-sampling variance ONLY; does NOT capture training-seed variability.",
        })
        obs_p, lo_p, hi_p = paired_bootstrap_ci(page_ids, page_sp_dict[a_key], page_sp_dict[b_key])
        stat_rows.append({
            "test_type": "Paired_Bootstrap_30Pages_10k",
            "comparison": label,
            "metric": "page_primary_cer_micro_pct_diff",
            "unit_or_scope": "30 pages (P01-P30), 10000 bootstrap resamples",
            "seed42_or_obs_diff": obs_p,
            "seed43_or_ci95_low": lo_p,
            "seed44_or_ci95_high": hi_p,
            "mean_diff": obs_p,
            "sd_diff_ddof1": "N/A",
            "t_stat": "N/A",
            "p_value_two_sided": "N/A",
            "note": "Captures page-sampling variance ONLY; does NOT capture training-seed variability.",
        })

    pd.DataFrame(stat_rows).to_csv(OUT_DIR / "STATISTICAL_TESTS.csv", index=False)
    print(f"  [+] Saved {OUT_DIR / 'STATISTICAL_TESTS.csv'}")

    # 4. Character Error Analysis & 314 Noise Crop Breakdown
    print("\n[4/4] Computing Character Error Analysis (Clean325) and 314 Noise Crop False-Positives...")
    singles_df = df_all[df_all["verified_crop_category"] == "genuine_single_line"]
    gt_char_counts = {ch: sum(nfc(t).count(ch) for t in singles_df["verified_ground_truth_text"]) for ch in TARGET_CHARS}

    char_rows = []
    # Analyze all 4 baselines + 6 representative/all 18 GPU models
    models_to_char_analyze = [(col, disp) for col, disp, _, _ in BASELINES] + [
        (rname, rname) for rname in df_gpu_models["run_name"].tolist()
    ]
    for col, disp in models_to_char_analyze:
        # Track target char hits and substitutions
        char_hits = {ch: 0 for ch in TARGET_CHARS}
        char_subs = {ch: Counter() for ch in TARGET_CHARS}
        for _, r in singles_df.iterrows():
            gt_s = nfc(r["verified_ground_truth_text"])
            hyp_s = nfc(r[col])
            _, ops = align(gt_s, hyp_s)
            for op_type, rc, hc in ops:
                if rc in TARGET_CHARS:
                    if op_type == "ok":
                        char_hits[rc] += 1
                    elif op_type == "sub":
                        char_subs[rc][hc] += 1
                    elif op_type == "del":
                        char_subs[rc]["<DEL>"] += 1

        # Combine with noise row
        n_info = next(nr for nr in noise_rows if nr["model_name"] == disp)
        for ch in TARGET_CHARS:
            denom = gt_char_counts[ch]
            acc = round(char_hits[ch] / denom * 100.0, 2) if denom > 0 else "N/A"
            top_err = "; ".join([f"{k}:{v}" for k, v in char_subs[ch].most_common(3)]) if denom > 0 else "N/A"
            char_rows.append({
                "model_name": disp,
                "target_char": ch,
                "gt_count_clean325": denom,
                "correct_count": char_hits[ch] if denom > 0 else "N/A",
                "accuracy_pct": acc,
                "top_error_confusions": top_err,
                "noise314_fp_rate_pct": n_info["noise_fp_rate_pct"],
                "noise314_total_chars": n_info["noise_total_chars"],
                "noise_share_of_output_pct": n_info["noise_share_of_output_pct"],
            })

    pd.DataFrame(char_rows).to_csv(OUT_DIR / "CHARACTER_ERROR_ANALYSIS.csv", index=False)
    print(f"  [+] Saved {OUT_DIR / 'CHARACTER_ERROR_ANALYSIS.csv'}")

    # 5. Reproducibility Manifest & Code Snapshot
    manifest_rows = []
    code_sources = [
        (PROJ_ROOT / "model.py", "model.py"),
        (PROJ_ROOT / "common.py", "common.py"),
        (PROJ_ROOT / "train_v2.py", "train_v2.py"),
        (PROJ_ROOT / "staging" / "AzOCR_GPU_Training_Package" / "dataset.py", "dataset.py"),
        (Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\code\train_gpu.py"), "train_gpu.py"),
        (Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\code\run_experiments.py"), "run_experiments.py"),
        (PROJ_ROOT / "requirements.txt", "requirements.txt"),
    ]
    for src, dest_name in code_sources:
        if src.exists():
            dst = CODE_SNAP_DIR / dest_name
            shutil.copy2(src, dst)
            manifest_rows.append({
                "artifact_type": "code_snapshot",
                "name": dest_name,
                "source_path": str(src),
                "size_bytes": src.stat().st_size,
                "sha256": sha256_file(src),
            })

    for data_src, name in [
        (GT_CSV, "ground_truth_lines.csv"),
        (VA_CSV, "verified_alignment.csv"),
        (GPU18_DIR / "per_model_summary.csv", "gpu18_per_model_summary.csv"),
        (GPU18_DIR / "checkpoint_sha256_manifest.csv", "gpu18_checkpoint_sha256_manifest.csv"),
    ]:
        manifest_rows.append({
            "artifact_type": "data_or_result",
            "name": name,
            "source_path": str(data_src),
            "size_bytes": data_src.stat().st_size,
            "sha256": sha256_file(data_src),
        })

    # Environment specs
    env_specs = [
        ("python", sys.version.split()[0]),
        ("pytorch", "2.14.1+cpu (local eval)"),
        ("opencv", "4.10.0"),
        ("tesseract", "v5.4.0.20240606 (--psm 7 -l aze)"),
        ("easyocr", "1.7.2"),
        ("paddleocr_paddlex", "3.7.0 / 3.7.2 (PP-OCRv6_medium_rec)"),
        ("model_selection_criterion", "Validation CER on synthetic validation split (no Real30 tuning)"),
        ("real30_device_provenance", "EXIF Make/Model stripped in PDF; no per-page phone model mapping verified"),
    ]
    for k, v in env_specs:
        manifest_rows.append({
            "artifact_type": "environment_or_protocol",
            "name": k,
            "source_path": v,
            "size_bytes": 0,
            "sha256": "N/A",
        })

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "REPRODUCIBILITY_MANIFEST.csv", index=False)
    print(f"  [+] Saved {OUT_DIR / 'REPRODUCIBILITY_MANIFEST.csv'} and code_snapshot/")
    print("\n[+] Final Scientific Audit computation finished cleanly.")


if __name__ == "__main__":
    main()
