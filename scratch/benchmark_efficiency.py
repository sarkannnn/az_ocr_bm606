"""Benchmark model footprint, checkpoint composition, and CPU inference latency
for CRNN (V1/V2) vs. Tesseract 5 (aze) on real document line crops (Photo 5, n=33 lines).
"""
import io
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common import infer_single_crop, load_model_checkpoint, preprocess

# Setup ASCII tessdata path for Tesseract on Windows
_td = Path(os.environ.get("TEMP", ".")) / "az_tessdata"
_td.mkdir(exist_ok=True)
shutil.copy(ROOT / "tessdata" / "aze.traineddata", _td / "aze.traineddata")
os.environ["TESSDATA_PREFIX"] = str(_td)
cmd = os.environ.get("TESSERACT_CMD") or str(ROOT / "scratch" / "_tesseract" / "tesseract.exe")
if Path(cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = cmd

OUT = ROOT / "results" / "efficiency_benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def inspect_checkpoint(rel_path: str):
    p = ROOT / rel_path
    ck = torch.load(p, map_location="cpu", weights_only=False)
    keys = list(ck.keys()) if isinstance(ck, dict) else ["raw_state_dict"]
    model_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    buf_fp32 = io.BytesIO()
    torch.save({"model": model_sd, "cfg": ck.get("cfg", {})}, buf_fp32)
    fp32_mb = len(buf_fp32.getvalue()) / (1024 * 1024)

    model_fp16 = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in model_sd.items()}
    buf_fp16 = io.BytesIO()
    torch.save({"model": model_fp16, "cfg": ck.get("cfg", {})}, buf_fp16)
    fp16_mb = len(buf_fp16.getvalue()) / (1024 * 1024)

    has_opt = isinstance(ck, dict) and ("opt" in ck or "optimizer" in ck)
    return {
        "path": rel_path,
        "disk_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "keys": keys,
        "includes_optimizer_state": has_opt,
        "weights_only_fp32_mb": round(fp32_mb, 2),
        "weights_only_fp16_mb": round(fp16_mb, 2),
    }


def main():
    ckpts = [
        "runs/word/best.pt",
        "runs/line/best.pt",
        "runs/word_v2_20261027_180300/best.pt",
        "runs/line_v2/best.pt",
    ]
    ckpt_info = [inspect_checkpoint(c) for c in ckpts]
    tess_mb = round((ROOT / "tessdata" / "aze.traineddata").stat().st_size / (1024 * 1024), 2)

    # Load Photo 5 full line crops (n=33)
    crops_dir = ROOT / "results" / "real_fix9" / "crops"
    crop_paths = sorted(crops_dir.glob("*.png"))
    imgs_bgr = [cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR) for p in crop_paths]
    imgs_pil = [Image.open(p).convert("RGB") for p in crop_paths]

    model, cfg, vocab, device = load_model_checkpoint(ROOT / "runs" / "line_v2" / "best.pt", torch.device("cpu"))
    n_params = sum(p.numel() for p in model.parameters())

    # Warm-up CRNN
    for im in imgs_bgr[:3]:
        _ = infer_single_crop(model, cfg, vocab, im, device)

    # 1. CRNN sequential (batch=1) over 3 repeats of 33 lines
    t0 = time.perf_counter()
    repeats = 3
    for _ in range(repeats):
        for im in imgs_bgr:
            _ = infer_single_crop(model, cfg, vocab, im, device)
    crnn_b1_sec = (time.perf_counter() - t0) / repeats
    crnn_b1_ms_per_line = 1000.0 * crnn_b1_sec / len(imgs_bgr)

    # 2. CRNN batched (batch=16) over 3 repeats of 33 lines
    t0 = time.perf_counter()
    for _ in range(repeats):
        tensors = [torch.from_numpy(preprocess(im, cfg)).unsqueeze(0) for im in imgs_bgr]
        batch_t = torch.cat(tensors, dim=0)
        with torch.no_grad():
            for i in range(0, len(imgs_bgr), 16):
                out = model(batch_t[i : i + 16])
                preds = out.argmax(2).permute(1, 0).cpu().numpy()
                for seq in preds:
                    _ = vocab.decode_greedy(seq)
    crnn_b16_sec = (time.perf_counter() - t0) / repeats
    crnn_b16_ms_per_line = 1000.0 * crnn_b16_sec / len(imgs_bgr)

    # 3. Tesseract PSM 7 and PSM 13 over 33 lines
    t0 = time.perf_counter()
    for pil_im in imgs_pil:
        _ = pytesseract.image_to_string(pil_im, lang="aze", config="--psm 7")
    tess_psm7_sec = time.perf_counter() - t0
    tess_psm7_ms_per_line = 1000.0 * tess_psm7_sec / len(imgs_pil)

    t0 = time.perf_counter()
    for pil_im in imgs_pil:
        _ = pytesseract.image_to_string(pil_im, lang="aze", config="--psm 13")
    tess_psm13_sec = time.perf_counter() - t0
    tess_psm13_ms_per_line = 1000.0 * tess_psm13_sec / len(imgs_pil)

    summary = {
        "cpu_processor": platform.processor(),
        "n_lines_benchmarked": len(imgs_bgr),
        "crnn_parameters": n_params,
        "checkpoints": ckpt_info,
        "tesseract_aze_traineddata_mb": tess_mb,
        "latency": {
            "crnn_line_v2_batch1": {
                "ms_per_line": round(crnn_b1_ms_per_line, 2),
                "lines_per_sec": round(1000.0 / crnn_b1_ms_per_line, 2),
                "speedup_vs_tess_psm13": round(tess_psm13_ms_per_line / crnn_b1_ms_per_line, 2),
            },
            "crnn_line_v2_batch16": {
                "ms_per_line": round(crnn_b16_ms_per_line, 2),
                "lines_per_sec": round(1000.0 / crnn_b16_ms_per_line, 2),
                "speedup_vs_tess_psm13": round(tess_psm13_ms_per_line / crnn_b16_ms_per_line, 2),
            },
            "tesseract_psm7": {
                "ms_per_line": round(tess_psm7_ms_per_line, 2),
                "lines_per_sec": round(1000.0 / tess_psm7_ms_per_line, 2),
            },
            "tesseract_psm13": {
                "ms_per_line": round(tess_psm13_ms_per_line, 2),
                "lines_per_sec": round(1000.0 / tess_psm13_ms_per_line, 2),
            },
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

