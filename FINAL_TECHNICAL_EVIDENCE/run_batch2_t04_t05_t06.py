"""Batch 2: T04 (Augmentation Spec), T05 (RandomPadding), T06 (VariableWidth Audit + Exploratory Native Inference)."""

import io
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

from common import Vocab, CHARS, DEFAULT_CFG, align, normalize, preprocess, robust_trim_chunk
from model import CRNN
from scripts.audit_real30_methodology import segment_lines_with_boxes, cv2_imread_utf8

OUT_DIR = PROJ_ROOT / "FINAL_TECHNICAL_EVIDENCE"
MODELS_ZIP = PROJ_ROOT / "AzOCR_Zenodo_v1_Release" / "AzOCR_v1_Models_18Official.zip"
VA_CSV = PROJ_ROOT / "results" / "real30_evaluation" / "verified_alignment.csv"
GT_CSV = PROJ_ROOT / "data" / "real30_evaluation" / "ground_truth_lines.csv"
PHOTOS_DIR = PROJ_ROOT / "data" / "real30_evaluation" / "photos"
GPU18_SUMMARY = PROJ_ROOT / "Real30_GPU18_Final_Evaluation" / "per_model_summary.csv"


def write_t04():
    rows = [
        ("Rotation", "augment_basic / augment_photo", "0.50 (Basic) / 0.50 (Photo)", "-5.0 deg", "+5.0 deg", "Continuous Uniform(-5.0, 5.0)", "1 (both)", "Yes", "Yes", "Groups A-D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L46-49, L73-77 (dataset.py)", "fc267378", "Identical between GPU dataset.py and CPU train_v2.py"),
        ("Perspective", "augment_basic / augment_photo", "0.30 (Basic) / 0.30 (Photo)", "-0.02W, -0.08H", "+0.02W, +0.08H", "Continuous Uniform(-1, 1) * [0.02W, 0.08H] per corner", "4 (Basic) / 2 (Photo)", "Yes", "Yes", "Groups A-D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L60-65, L79-84 (dataset.py)", "fc267378", "Order differs: last in BasicAug, 2nd in PhotoAug"),
        ("Illumination gradient", "augment_photo", "0.50", "0.70 (low_val)", "1.00", "Continuous Uniform(0.7, 1.0) at random angle Uniform(0, 2pi)", "3", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L86-104 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("Shadow band", "augment_photo (nested in background_tint)", "0.25 effective (0.50 * 0.50)", "depth=0.12, bw=0.2", "depth=0.28, bw=0.5", "Nested 0.50 sub-prob; d0~U(-0.5,0.5), bw~U(0.2,0.5), depth~U(0.12,0.28)", "3b", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L98-103 (dataset.py)", "fc267378", "Gaussian attenuation band inside illumination gradient"),
        ("Brightness/contrast", "augment_photo", "0.60", "alpha=0.60, beta=-40.0", "alpha=1.10, beta=+20.0", "Continuous Uniform(0.6, 1.1) & Uniform(-40.0, 20.0)", "4", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L106-110 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("Gaussian blur", "augment_basic / augment_photo", "0.30 (Basic) / 0.60 (Photo)", "k=3 (Basic) / sigma=0.5 (Photo)", "k=5 (Basic) / sigma=2.0 (Photo)", "Discrete {3,5} (Basic) / Continuous U(0.5, 2.0) (Photo)", "2 (Basic) / 5 (Photo)", "Yes", "Yes", "All 6 Groups (strength differs)", "dataset.py (GPU) / train_v2.py (CPU)", "L50-53, L112-115 (dataset.py)", "fc267378", "Stronger and continuous in PhotoAug vs discrete in BasicAug"),
        ("Motion blur", "augment_photo", "0.15", "ksize=3", "ksize=7", "Discrete odd integer in [3, 7], angle~U(0, pi)", "6", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L117-132 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("Sensor noise", "augment_photo", "0.60", "sigma=3.0", "sigma=12.0", "Additive Normal(0, n_sigma), n_sigma~U(3.0, 12.0)", "7", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L134-138 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("Salt-and-pepper noise", "augment_basic / augment_photo", "0.30 (Basic) / 0.20 (Photo)", "0.005 black + 0.005 white", "0.005 black + 0.005 white", "Pixel mask U(0,1): <0.005 -> 0, >0.995 -> 255", "3 (Basic) / 8 (Photo)", "Yes", "Yes", "All 6 Groups (prob differs)", "dataset.py (GPU) / train_v2.py (CPU)", "L54-59, L140-145 (dataset.py)", "fc267378", "Higher prob in BasicAug (0.30) than PhotoAug (0.20)"),
        ("Downscale/upscale", "augment_photo", "0.40", "scale=0.50", "scale=0.90", "Continuous U(0.5, 0.9); INTER_AREA down, INTER_LINEAR up", "9", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L147-152 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("JPEG compression", "augment_photo", "0.40", "quality=40", "quality=90", "Integer int(U(40, 90)) via cv2.imencode('.jpg')", "10", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L154-158 (dataset.py)", "fc267378", "PhotoAug-only"),
        ("Contrast stretch", "augment_photo", "0.50", "min=0", "max=255", "Grayscale cv2.normalize(NORM_MINMAX, 0, 255) -> BGR", "11", "No", "Yes", "Groups B, D, RandomPadding, VariableWidth", "dataset.py (GPU) / train_v2.py (CPU)", "L160-164 (dataset.py)", "fc267378", "PhotoAug-only"),
    ]
    cols = [
        "transform", "function_name", "probability", "min_param", "max_param",
        "sampling_distribution", "execution_order", "basic_aug_member", "photo_aug_member",
        "applicable_groups", "source_file", "source_lines", "git_commit", "notes"
    ]
    pd.DataFrame(rows, columns=cols).to_csv(OUT_DIR / "AUGMENTATION_FULL_SPECIFICATION_VERIFIED.csv", index=False)
    print("[T04] Saved AUGMENTATION_FULL_SPECIFICATION_VERIFIED.csv (12 transforms).")


def write_t05():
    md = """# AzOCR-V2 — RandomPadding Technical Specification (Task T05)

**Source Files:** `AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py` (lines 179, 207–216, 236–237), `Exp_RandomPadding_GroupD_seed{42,43,44}/config.yaml`

| Parameter | Verified Value | Code Evidence |
|---|---|---|
| **Minimum Padding** | `0` px per side | `self.rng.integers(0, self.random_padding_max + 1)` (`dataset.py:209-212`) |
| **Maximum Padding** | `12` px per side | `random_padding: 12` in `Exp_RandomPadding_GroupD_seed*/config.yaml:17` |
| **Horizontal Padding** | Independent `p_left`, `p_right` $\\in \\{0, 1, \\dots, 12\\}$ | `dataset.py:211-212` |
| **Vertical Padding** | Independent `p_top`, `p_btm` $\\in \\{0, 1, \\dots, 12\\}$ | `dataset.py:209-210` |
| **Random Sampling Distribution** | Discrete Uniform over `{0, 1, ..., 12}` independently per side | `np.random.Generator.integers(0, 13)` |
| **Application Probability** | `1.0` (applied to 100% of training samples when `train=True` and `random_padding_max > 0`; each side independently draws 0 px with $p = 1/13 \\approx 7.69\%$) | `if self.train and self.random_padding_max > 0:` (`dataset.py:208`) |
| **Padding Color** | Pure white `[255, 255, 255]` (BGR) | `cv2.copyMakeBorder(..., cv2.BORDER_CONSTANT, value=[255, 255, 255])` (`dataset.py:213-216`) |
| **Resize Order** | **1st:** Random border padding -> **2nd:** `augment_photo` -> **3rd:** `preprocess` resize to `H=32` | `dataset.py:208-237` |
| **Trim Behavior** | None during training (`robust_trim_chunk` is applied only at inference inside `infer_line_smart`) | `common.py:268-322` |
| **Fixed-Width Interaction** | `dynamic_width: false`; padded+augmented image is proportionally resized to `H=32` (clamped at `new_w <= 512`) and right-padded with white to `W=512` (`pad_alignment='left'`) | `dataset.py:235-237`, `common.py:103-111` |
"""
    (OUT_DIR / "RANDOM_PADDING_IMPLEMENTATION.md").write_text(md, encoding="utf-8")
    print("[T05] Saved RANDOM_PADDING_IMPLEMENTATION.md.")


def nfc(text) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def cer_wer_pair(ref: str, hyp: str):
    rn, hn = nfc(ref), nfc(hyp)
    dc, _ = align(rn, hn)
    dw, _ = align(rn.split(), hn.split())
    return dc, len(rn), dw, len(rn.split())


def infer_native_variable_width(model, cfg, vocab, line_bgr, device):
    """Post-hoc exploratory native variable-width single-pass inference (no chunking, no 512 cap)."""
    trimmed = robust_trim_chunk(line_bgr)
    h, w = trimmed.shape[:2]
    if h == 0 or w == 0:
        return ""
    H = cfg.get("height", 32)
    calc_w = max(32, ((round(w * H / max(h, 1)) + 3) // 4) * 4)
    dyn_cfg = dict(cfg)
    dyn_cfg["width"] = calc_w
    x = preprocess(trimmed, dyn_cfg)
    x_t = torch.from_numpy(x)[None].to(device)
    with torch.no_grad():
        out = model(x_t)
        ids = out.argmax(2).T.cpu().numpy()[0]
        return vocab.decode_greedy(ids)


def run_t06():
    df_va = pd.read_csv(VA_CSV)
    df_gt = pd.read_csv(GT_CSV)
    df_gpu18 = pd.read_csv(GPU18_SUMMARY)

    # Reconstruct exact 690 crops from photos
    crops_map = {}
    for pid in sorted(df_va["page_id"].unique()):
        img_bgr = cv2_imread_utf8(PHOTOS_DIR / f"{pid}_pdf_embedded.jpeg")
        crops, _ = segment_lines_with_boxes(img_bgr)
        va_p = df_va[df_va["page_id"] == pid].sort_values("crop_id")
        for c, (_, r) in zip(crops, va_p.iterrows()):
            crops_map[(pid, int(r["crop_id"]))] = c

    ordered_keys = [(r["page_id"], int(r["crop_id"])) for _, r in df_va.iterrows()]
    singles_mask = df_va["verified_crop_category"] == "genuine_single_line"
    page_ids = [f"P{i:02d}" for i in range(1, 31)]

    torch.set_num_threads(4)
    device = torch.device("cpu")
    results_rows = []

    with zipfile.ZipFile(MODELS_ZIP, "r") as z:
        for seed in [42, 43, 44]:
            rname = f"Exp_VariableWidth_GroupD_seed{seed}"
            ck = torch.load(io.BytesIO(z.read(f"{rname}/best.pt")), map_location=device, weights_only=False)
            cfg = ck.get("cfg", DEFAULT_CFG)
            vocab = Vocab(ck.get("chars", CHARS))
            model = CRNN(in_ch=1, num_classes=vocab.size).to(device)
            model.load_state_dict(ck["model"])
            model.eval()

            # Fixed infer_line_smart row from Real30_GPU18_Final_Evaluation
            orig_row = df_gpu18[df_gpu18["run_name"] == rname].iloc[0]
            results_rows.append({
                "run_name": rname,
                "inference_mode": "Official_Fixed512_infer_line_smart",
                "evaluation_status": "OFFICIAL_REAL30",
                "clean325_cer_micro_pct": orig_row["clean325_cer_micro_pct"],
                "clean325_wer_micro_pct": orig_row["clean325_wer_micro_pct"],
                "clean325_exact_matches": orig_row["clean325_exact_matches"],
                "clean325_exact_match_pct": orig_row["clean325_exact_match_pct"],
                "page_primary_cer_micro_pct": orig_row["page_primary_cer_micro_pct"],
                "page_primary_wer_micro_pct": orig_row["page_primary_wer_micro_pct"],
                "mean_ms_per_crop": orig_row["mean_ms_per_crop"],
            })

            # Post-hoc exploratory native variable-width inference
            t0 = time.perf_counter()
            preds = [normalize(infer_native_variable_width(model, cfg, vocab, crops_map[k], device).strip()) for k in ordered_keys]
            dt_ms = (time.perf_counter() - t0) * 1000.0 / len(ordered_keys)

            df_tmp = df_va[["page_id", "crop_id", "bbox_y0", "bbox_x0", "verified_crop_category", "verified_ground_truth_text"]].copy()
            df_tmp["pred"] = preds

            s_df = df_tmp[singles_mask]
            c_dc, c_lc, c_dw, c_lw, c_ex = 0, 0, 0, 0, 0
            for _, sr in s_df.iterrows():
                dc, lc, dw, lw = cer_wer_pair(sr["verified_ground_truth_text"], sr["pred"])
                c_dc += dc; c_lc += lc; c_dw += dw; c_lw += lw
                if nfc(sr["verified_ground_truth_text"]) == nfc(sr["pred"]):
                    c_ex += 1

            p_dc, p_lc, p_dw, p_lw = 0, 0, 0, 0
            for pid in page_ids:
                p_gt = df_gt[df_gt["page_id"] == pid].sort_values("line_no")
                p_cr = df_tmp[df_tmp["page_id"] == pid].sort_values(["bbox_y0", "bbox_x0"])
                gt_sp = " ".join([nfc(t) for t in p_gt["ground_truth_nfc"]])
                pr_sp = " ".join([nfc(t) for t in p_cr["pred"] if len(nfc(t)) > 0])
                dc, lc, dw, lw = cer_wer_pair(gt_sp, pr_sp)
                p_dc += dc; p_lc += lc; p_dw += dw; p_lw += lw

            results_rows.append({
                "run_name": rname,
                "inference_mode": "Exploratory_Native_VariableWidth_SinglePass",
                "evaluation_status": "POST_HOC_EXPLORATORY",
                "clean325_cer_micro_pct": round(c_dc / max(1, c_lc) * 100.0, 2),
                "clean325_wer_micro_pct": round(c_dw / max(1, c_lw) * 100.0, 2),
                "clean325_exact_matches": c_ex,
                "clean325_exact_match_pct": round(c_ex / len(s_df) * 100.0, 2),
                "page_primary_cer_micro_pct": round(p_dc / max(1, p_lc) * 100.0, 2),
                "page_primary_wer_micro_pct": round(p_dw / max(1, p_lw) * 100.0, 2),
                "mean_ms_per_crop": round(dt_ms, 2),
            })
            print(f"[T06] {rname} Native: Clean325 CER={results_rows[-1]['clean325_cer_micro_pct']}%, Exact={c_ex}/325, Page CER={results_rows[-1]['page_primary_cer_micro_pct']}%")

    df_res = pd.DataFrame(results_rows)
    df_res.to_csv(OUT_DIR / "VARIABLE_WIDTH_INFERENCE_RESULTS.csv", index=False)

    md = """# AzOCR-V2 — VariableWidth Technical & Inference Audit (Task T06)

## 1. Eight Direct Technical Answers (from `dataset.py`, `train_gpu.py`, `common.py`)

1. **Dynamic width in training?** **Yes.** `OCRDataset(..., dynamic_width=True)` computes `calc_w = ((max(16, round(w * 32 / h)) + 3) // 4) * 4` with no 512-px clamp (`dataset.py:226-234`).
2. **Width-bucketing applied?** **No.** `DataLoader` uses `shuffle=True` without a width-bucketed sampler (`train_gpu.py:243`).
3. **Batch padding?** `collate_fn_dynamic` (`dataset.py:253-273`) pads all samples in the random batch with `1.0` (white) on the right to `max_w = max(32, ((max(widths) + 3) // 4) * 4)`.
4. **CTC sequence length?** `time_steps = out.size(0)` (`max_w // 4` after two `(2,2)` MaxPool layers), passed identically via `torch.full((batch_size,), time_steps, dtype=torch.long)` (`train_gpu.py:77-78`).
5. **Does Real30 inference use 32×512 across all models?** **Yes.** All 18 checkpoints store `cfg={'height': 32, 'width': 512}`, and `evaluate_real30_gpu18.py` runs `common.infer_line_smart` identically across all 18 models.
6. **Are VariableWidth checkpoints converted to fixed width during inference?** **Yes**, during Real30/R59 official evaluation, `infer_line_smart` targets `32×512` per chunk (`common.py:307-322`). The CNN+BiLSTM weights themselves remain width-agnostic.
7. **Is smart chunking applied?** **Yes**, when `squeeze_ratio = (w_c * 32) / (h_c * 512) > 1.05`, `infer_line_smart` splits the line into `ceil(squeeze_ratio)` chunks at word gaps and trims each chunk (`common.py:313-322`).
8. **Is native variable-width inference possible?** **Yes.** Feeding `(1, 1, 32, W_dyn)` (`W_dyn = max(32, ((round(w_c * 32 / h_c) + 3) // 4) * 4)`) in a single pass works without modifying any weights.

## 2. Post-Hoc Exploratory Native Variable-Width vs. Official Fixed-512 `infer_line_smart`

*(Note: Official Real30 results remain unchanged; native single-pass results below are strictly **post-hoc exploratory**.)*

See `VARIABLE_WIDTH_INFERENCE_RESULTS.csv` for per-seed comparison.
"""
    (OUT_DIR / "VARIABLE_WIDTH_TECHNICAL_AUDIT.md").write_text(md, encoding="utf-8")
    print("[T06] Saved VARIABLE_WIDTH_INFERENCE_RESULTS.csv and VARIABLE_WIDTH_TECHNICAL_AUDIT.md.")


if __name__ == "__main__":
    write_t04()
    write_t05()
    run_t06()
