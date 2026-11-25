"""AzOCR-V2: Aşama 2.3 — Padding Robustness Testi (15 Sənəd, 295 Sətir).

Bu betik mövcud CRNN-V2 checkpoint'i (runs/line_v2/best.pt) ilə sabit 295 sətirlik
rəsmi benchmark üzərində padding (0, 4, 8, 12, 20 px) dəyərlərinin təsirini
rəsmi infer_line_smart() borusu ilə ölçür.

İki Ayrı Eksperiment:
1. Əsas Eksperiment — Görüntüyə Əlavə Edilmiş Ağ Haşiyə (Canvas Border Padding +P px):
   Giriş sətir kəsiminə hər 4 tərəfdən +P px ağ fon əlavə edilir və rəsmi infer_line_smart()
   ön-emalı ilə test edilir. Referans +0 px rəsmi 0.30% CER ilə tam uyğunlaşır.
   Həmçinin sətirlərin bloklara bölünmə (segmentation / chunk count) dəyişməsi izlənir.
2. Ayrı Əlavə Eksperiment — Mürəkkəb Kənarı Haşiyəsi (Trim-to-Ink Margin P px):
   İnference zamanı robust_trim_chunk() funksiyasının daxili mürəkkəb təhlükəsizlik
   haşiyəsinin (0, 4, 8, 12, 20 px) təsiri ölçülür (əsas təcrübəyə qarışdırılmır).

Nəticələr: results/padding_robustness/
"""
import os
import sys
import time
import json
import torch
import cv2
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common import load_model_checkpoint, infer_line_smart, infer_single_crop, split_line_into_chunks, normalize, cer_wer
from scripts.eval_ablation_models import build_genuine_295_benchmark, cv2_imread_utf8


def trim_chunk_with_margin(chunk_bgr: np.ndarray, margin: int = 4) -> np.ndarray:
    """Mürəkkəb sərhədinə qədər sıxır və dəqiq `margin` piksel haşiyə saxlayır."""
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY) if chunk_bgr.ndim == 3 else chunk_bgr
    h_chunk, w_chunk = gray.shape
    if h_chunk < 5 or w_chunk < 5:
        return chunk_bgr
    sigma = min(25, max(5, h_chunk // 2))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    ink = ((bg.astype(np.int16) - gray.astype(np.int16)) > 18).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_x = w_chunk
    max_x = 0
    found = False
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        touches_border = (x == 0) or (x + w >= w_chunk)
        if area < 8 or (touches_border and h >= 0.70 * h_chunk):
            continue
        found = True
        if x < min_x:
            min_x = x
        if x + w > max_x:
            max_x = x + w
    if not found:
        return chunk_bgr
    x0 = max(0, min_x - margin)
    x1 = min(w_chunk, max_x + margin)
    return chunk_bgr[:, x0:x1]


def infer_line_with_custom_margin(model, cfg, vocab, line_bgr, device, margin_px: int) -> str:
    h_c, w_c = line_bgr.shape[:2]
    target_h = cfg.get("height", 32)
    target_w = cfg.get("width", 512)
    squeeze_ratio = (w_c * target_h) / (h_c * target_w)
    n_chunks = max(1, int(np.ceil(squeeze_ratio))) if squeeze_ratio > 1.05 else 1
    raw_chunks = split_line_into_chunks(line_bgr, n_chunks)
    preds = []
    for ch in raw_chunks:
        ch_trimmed = trim_chunk_with_margin(ch, margin=margin_px)
        p = infer_single_crop(model, cfg, vocab, ch_trimmed, device).strip()
        if p:
            preds.append(p)
    return " ".join(preds)


def main():
    print("=" * 80)
    print("AZOCR-V2: RƏSMİ INFER_LINE_SMART PADDING TESTİ (295 SƏTİR BENCHMARK)")
    print("=" * 80)

    device = torch.device("cpu")
    v2_model, v2_cfg, v2_vocab, _ = load_model_checkpoint("runs/line_v2/best.pt", device=device)

    samples = build_genuine_295_benchmark()
    print(f"Sabit test bazası: {len(samples)} sətir (B1-B15 rəsmi benchmark)")

    pad_values = [0, 4, 8, 12, 20]
    out_dir = BASE_DIR / "results" / "padding_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_added = {}
    results_margin = {}
    refs = [gt for tag, code, idx, cp, gt in samples]

    print("\n--- 1. ƏSAS EKSPERİMENT: Əlavə Edilmiş Ağ Haşiyə (Canvas Border Padding +P px) ---")
    print("    Protokol: cv2.copyMakeBorder(..., P, P, P, P) -> Rəsmi infer_line_smart()")
    for p in pad_values:
        t0 = time.perf_counter()
        hyps = []
        chunk_counts = []
        for tag, code, idx, cp, gt in samples:
            img = cv2_imread_utf8(cp)
            if p > 0:
                padded = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            else:
                padded = img
            h_c, w_c = padded.shape[:2]
            sq = (w_c * 32) / (h_c * 512)
            n_chunks = max(1, int(np.ceil(sq))) if sq > 1.05 else 1
            chunk_counts.append(n_chunks)

            pred = normalize(infer_line_smart(v2_model, v2_cfg, v2_vocab, padded, device).replace("—", "-"))
            hyps.append(pred)

        lat = (time.perf_counter() - t0) * 1000.0 / len(samples)
        cer, wer, _ = cer_wer(refs, hyps)
        exact = sum(1 for r, h in zip(refs, hyps) if r == h)
        mean_chunks = float(np.mean(chunk_counts))

        results_added[f"pad_{p:02d}px"] = {
            "pad_px": p,
            "raw_cer_pct": round(cer * 100, 2),
            "raw_wer_pct": round(wer * 100, 2),
            "exact_lines": f"{exact}/{len(samples)} ({exact/len(samples)*100:.1f}%)",
            "mean_chunks": round(mean_chunks, 2),
            "latency_ms": round(lat, 1)
        }
        print(f"  [+] Pad +{p:2d} px | CER: {cer*100:5.2f}% | WER: {wer*100:5.2f}% | Exact: {exact:3d}/295 ({exact/len(samples)*100:4.1f}%) | Orta chunk sayı: {mean_chunks:.2f} | {lat:.1f} ms/sətir")

    print("\n--- 2. AYRI EKSPERİMENT: Mürəkkəb Kənarı Haşiyəsi (Trim-to-Ink Margin P px) ---")
    print("    Protokol: Sabit kəsim -> robust_trim_chunk(margin=P px) daxili tənzimləməsi")
    for p in pad_values:
        t0 = time.perf_counter()
        hyps = []
        for tag, code, idx, cp, gt in samples:
            img = cv2_imread_utf8(cp)
            pred = normalize(infer_line_with_custom_margin(v2_model, v2_cfg, v2_vocab, img, device, margin_px=p).replace("—", "-"))
            hyps.append(pred)
        lat = (time.perf_counter() - t0) * 1000.0 / len(samples)
        cer, wer, _ = cer_wer(refs, hyps)
        exact = sum(1 for r, h in zip(refs, hyps) if r == h)

        results_margin[f"margin_{p:02d}px"] = {
            "margin_px": p,
            "raw_cer_pct": round(cer * 100, 2),
            "raw_wer_pct": round(wer * 100, 2),
            "exact_lines": f"{exact}/{len(samples)} ({exact/len(samples)*100:.1f}%)",
            "latency_ms": round(lat, 1)
        }
        print(f"  [+] Margin {p:2d} px | CER: {cer*100:5.2f}% | WER: {wer*100:5.2f}% | Exact: {exact:3d}/295 ({exact/len(samples)*100:4.1f}%) | {lat:.1f} ms/sətir")

    full_payload = {
        "benchmark": "15-doc smartphone benchmark (295 lines)",
        "model": "CRNN-V2 (runs/line_v2/best.pt)",
        "pad_values": pad_values,
        "canvas_border_padding_smart_pipeline": results_added,
        "trim_ink_margin_padding_ablation": results_margin,
    }

    json_path = out_dir / "padding_robustness_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2, ensure_ascii=False)

    md_path = out_dir / "padding_robustness_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AzOCR-V2 — Padding Robustness Eksperimental Hesabatı\n\n")
        f.write("Sabit Test Bazası: **15 Müstəqil Smartfon Sənədi (`B1`–`B15`, cəmi 295 sətir)**\n")
        f.write("Model: **CRNN-V2 Xam CTC Çıxışı (`runs/line_v2/best.pt`)**\n\n")

        f.write("## 1. Əsas Eksperiment: Əlavə Edilmiş Ağ Haşiyə (Canvas Border Padding + Rəsmi infer_line_smart())\n\n")
        f.write("Kəsimin kənarlarına birbaşa $+P\\text{ px}$ ağ fon əlavə edilir və rəsmi borudan keçirilir.\n\n")
        f.write("| Əlavə Haşiyə (Padding) | Xam (Raw) CER (%) ↓ | Xam (Raw) WER (%) ↓ | Tam Düzgün Sətir (Exact) ↑ | Orta Blok (Chunk) Sayı | Gecikmə (ms) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: :---: :---: |\n")
        for k, v in results_added.items():
            f.write(f"| **+{v['pad_px']} px** | **{v['raw_cer_pct']:.2f}%** | {v['raw_wer_pct']:.2f}% | {v['exact_lines']} | {v['mean_chunks']:.2f} | {v['latency_ms']} ms |\n")

        f.write("\n## 2. Ayrı Eksperiment: Mürəkkəb Kənarı Haşiyəsi (Trim-to-Ink Margin Padding)\n\n")
        f.write("Ön-emal zamanı mürəkkəb sərhədindən sonra saxlanılan təhlükəsizlik haşiyəsinin tənzimlənməsi.\n\n")
        f.write("| Saxlanılan Haşiyə (Margin) | Xam (Raw) CER (%) ↓ | Xam (Raw) WER (%) ↓ | Tam Düzgün Sətir (Exact) ↑ | Gecikmə (ms) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for k, v in results_margin.items():
            f.write(f"| **{v['margin_px']} px** | **{v['raw_cer_pct']:.2f}%** | {v['raw_wer_pct']:.2f}% | {v['exact_lines']} | {v['latency_ms']} ms |\n")

    print(f"\n[+] Hesabat yeniləndi:")
    print(f"    JSON: {json_path}")
    print(f"    MD  : {md_path}")


if __name__ == "__main__":
    main()
