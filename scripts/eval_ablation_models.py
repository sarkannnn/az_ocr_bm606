"""15 Belgelik Benchmark Üzerinde 2x2 Faktöriyel Ablasyon Değerlendirmesi (Düzeltilmiş 295 Sətir).

Bu betik aşağıdaki 4 modeli eyni 295 sətirlik rəsmi smartfon benchmark'ında (IMG_5121 - IMG_5135)
müəllif qiymətləndirmə protokolu (infer_line_smart + robust_trim_chunk) ilə qiymətləndirir:
- Group A: 8 V1 Font, V1 Baza Augmentasiyası -> runs/line/best.pt
- Group B: 8 V1 Font, V2 Fotorealistik Aug   -> runs/ablation_group_b_aug/best.pt
- Group C: 14 V2 Font, V1 Baza Augmentasiyası -> runs/ablation_group_c_v1aug/best.pt
- Group D: 14 V2 Font, V2 Fotorealistik Aug  -> runs/line_v2/best.pt

Çıxış: results/ablation_2x2_results.json və terminal/hesabat cədvəli.
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

from common import load_model_checkpoint, infer_line_smart, normalize, cer_wer
from postprocess import AzerbaijaniPostProcessor
from generate_and_test_unseen import P1_REFS, P2_REFS, P3_REFS, GENERATED_PHONE_DOCS


def cv2_imread_utf8(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


MODELS_CONFIG = {
    "Group_A (8 V1 Fonts, V1 Basic Aug - Baseline)": BASE_DIR / "runs" / "line" / "best.pt",
    "Group_B (8 V1 Fonts, V2 Photo Aug)": BASE_DIR / "runs" / "ablation_group_b_aug" / "best.pt",
    "Group_C (14 V2 Fonts, V1 Basic Aug)": BASE_DIR / "runs" / "ablation_group_c_v1aug" / "best.pt",
    "Group_D (14 V2 Fonts, V2 Photo Aug - Full V2)": BASE_DIR / "runs" / "line_v2" / "best.pt",
}


def build_genuine_295_benchmark():
    samples = []
    # B1 (IMG_5121): 16 real physical smartphone lines
    for i, ref in enumerate(P1_REFS):
        p = BASE_DIR / "results" / "real_fix8" / "crops_times" / f"line_{i:02d}.png"
        samples.append(("B1", "IMG_5121", i, p, normalize(ref.replace("—", "-"))))

    # B2 (IMG_5122): 16 real physical smartphone lines
    for i, ref in enumerate(P2_REFS):
        p = BASE_DIR / "results" / "real_fix8" / "crops_calibri" / f"line_{i:02d}.png"
        samples.append(("B2", "IMG_5122", i, p, normalize(ref.replace("—", "-"))))

    # B3 (IMG_5123): 23 real physical smartphone lines
    for i, ref in enumerate(P3_REFS):
        p = BASE_DIR / "results" / "photo3_test" / "crops" / f"line_{i:02d}.png"
        samples.append(("B3", "IMG_5123", i, p, normalize(ref.replace("—", "-"))))

    # B4..B15 (IMG_5124..IMG_5135): 12 docs x 20 lines = 240 lines
    for doc in GENERATED_PHONE_DOCS[3:]:
        c_dir = BASE_DIR / "data" / "phone_photos_15" / "crops" / doc["img_code"]
        for i, ref in enumerate(doc["lines"]):
            p = c_dir / f"line_{i:02d}.png"
            samples.append((doc.get("short_name", doc["img_code"]), doc["img_code"], i, p, normalize(ref.replace("—", "-"))))

    return samples


def main():
    print("=" * 80)
    print("AZOCR-V2: DƏQİQ 2x2 FAKTÖRİYEL ABLASYON ANALİZİ (15 SƏNƏD, 295 SƏTİR)")
    print("=" * 80)
    device = torch.device("cpu")
    pp = AzerbaijaniPostProcessor()

    samples = build_genuine_295_benchmark()
    print(f"Toplam qiymətləndirilən sənəd/sətir sayı: {len(samples)} sətir (16 + 16 + 23 + 240)")

    for tag, code, idx, cp, gt in samples:
        if not cp.exists():
            raise FileNotFoundError(f"Crop faylı tapılmadı: {cp}")

    all_results = {}

    for m_label, m_path in MODELS_CONFIG.items():
        print(f"\n[+] Qiymətləndirilir: {m_label}")
        if not m_path.exists():
            print(f"    [XƏTA] Checkpoint tapılmadı: {m_path}")
            continue

        model, cfg, vocab, _ = load_model_checkpoint(str(m_path), device=device)

        refs = []
        raw_hyps = []
        pp_hyps = []
        latencies = []

        for tag, code, idx, cp, gt in samples:
            img_bgr = cv2_imread_utf8(cp)
            refs.append(gt)

            t0 = time.perf_counter()
            raw_pred = normalize(infer_line_smart(model, cfg, vocab, img_bgr, device).replace("—", "-"))
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            raw_hyps.append(raw_pred)

            if pp and pp.loaded:
                toks = [pp.correct_word(t, max_edit_distance=1) for t in raw_pred.split()]
                pp_pred = normalize(pp.clean_punctuation(" ".join(toks)))
            else:
                pp_pred = raw_pred
            pp_hyps.append(pp_pred)

        raw_cer, raw_wer, _ = cer_wer(refs, raw_hyps)
        pp_cer, pp_wer, _ = cer_wer(refs, pp_hyps)
        raw_exact = sum(1 for r, h in zip(refs, raw_hyps) if r == h)
        pp_exact = sum(1 for r, h in zip(refs, pp_hyps) if r == h)
        mean_lat = np.mean(latencies)

        all_results[m_label] = {
            "raw_cer_pct": round(raw_cer * 100, 2),
            "raw_wer_pct": round(raw_wer * 100, 2),
            "raw_exact": f"{raw_exact}/{len(samples)} ({raw_exact/len(samples)*100:.1f}%)",
            "pp_cer_pct": round(pp_cer * 100, 2),
            "pp_wer_pct": round(pp_wer * 100, 2),
            "pp_exact": f"{pp_exact}/{len(samples)} ({pp_exact/len(samples)*100:.1f}%)",
            "latency_ms_per_line": round(mean_lat, 1),
        }

        print(f"    Raw CER: {raw_cer*100:.2f}%, WER: {raw_wer*100:.2f}%, Exact: {raw_exact}/{len(samples)}")
        print(f"    PP  CER: {pp_cer*100:.2f}%, WER: {pp_wer*100:.2f}%, Exact: {pp_exact}/{len(samples)}")

    out_json = BASE_DIR / "results" / "ablation_2x2_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Nəticələr yazıldı: {out_json}")


if __name__ == "__main__":
    main()
