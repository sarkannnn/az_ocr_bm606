"""15 Belgelik Benchmark Üzerinde 2x2 Faktöriyel Ablasyon Değerlendirmesi.

Bu betik aşağıdaki 4 modeli aynı 295 sətirlik rəsmi smartfon benchmark'ında (IMG_5121 - IMG_5135)
eyni protokolla qiymətləndirir:
- Grup A: V1 (Standart font, Augmentasiya YOK) -> runs/line/best.pt
- Grup B: Standart font, Augmentasiya VAR      -> runs/ablation_group_b_aug/best.pt
- Grup C: Geniş font, Augmentasiya YOK         -> runs/ablation_group_c_noaug/best.pt
- Grup D: V2 (Geniş font, Augmentasiya VAR)    -> runs/line_v2/best.pt

Çıxış: results/ablation_2x2_results.json və Markdown cədvəli.
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

from common import load_model_checkpoint, preprocess, normalize, cer_wer
from postprocess import AzerbaijaniPostProcessor

MODELS_CONFIG = {
    "Group_A (Std Font, No Aug - V1)": BASE_DIR / "runs" / "line" / "best.pt",
    "Group_B (Std Font, V2 Aug)": BASE_DIR / "runs" / "ablation_group_b_aug" / "best.pt",
    "Group_C (Wide Font, No Aug)": BASE_DIR / "runs" / "ablation_group_c_noaug" / "best.pt",
    "Group_D (Wide Font, V2 Aug - V2)": BASE_DIR / "runs" / "line_v2" / "best.pt",
}


def infer_crop(model, cfg, vocab, img_bgr, device):
    x_np = preprocess(img_bgr, cfg)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred_idx = out.argmax(dim=2).squeeze(1).cpu().numpy().tolist()
    pred_str = vocab.decode_greedy(pred_idx)
    return normalize(pred_str.replace("—", "-"))


def main():
    print("=" * 75)
    print("AZOCR-V2: 2x2 FAKTÖRİYEL ABLASYON ANALİZİ (15 BELGE, 295 SATIR)")
    print("=" * 75)
    device = torch.device("cpu")
    pp = AzerbaijaniPostProcessor()

    crops_base = BASE_DIR / "data" / "phone_photos_15" / "crops"
    from generate_and_test_unseen import P1_REFS, P2_REFS, P3_REFS, GENERATED_PHONE_DOCS

    doc_refs = {
        "IMG_5121": P1_REFS,
        "IMG_5122": P2_REFS,
        "IMG_5123": P3_REFS,
    }
    for doc in GENERATED_PHONE_DOCS:
        doc_refs[doc["img_code"]] = doc["refs"]

    all_samples = []
    for img_code in sorted(doc_refs.keys()):
        refs = doc_refs[img_code]
        crop_dir = crops_base / img_code
        for idx, ref in enumerate(refs):
            cp = crop_dir / f"line_{idx:02d}.png"
            if cp.exists():
                all_samples.append((str(cp), normalize(ref.replace("—", "-"))))

    print(f"Toplam qiymətləndirilən sətir sayı: {len(all_samples)} (Gözlənən: 295)")

    results = {}
    for name, ckpt in MODELS_CONFIG.items():
        if not ckpt.exists():
            print(f"[-] {name} kontrol nöqtəsi tapılmadı ({ckpt.name}). Keçilir...")
            continue

        print(f"\n[+] Qiymətləndirilir: {name}...")
        model, cfg, vocab, _ = load_model_checkpoint(ckpt, device=device)

        refs, raw_hyps, pp_hyps = [], [], []
        t0 = time.perf_counter()
        for cp, ref in all_samples:
            img = cv2.imread(cp)
            raw = infer_crop(model, cfg, vocab, img, device)
            toks = [pp.correct_word(t, max_edit_distance=1) for t in raw.split()] if pp.loaded else raw.split()
            v2_pp = normalize(pp.clean_punctuation(" ".join(toks)))

            refs.append(ref)
            raw_hyps.append(raw)
            pp_hyps.append(v2_pp)

        dur_ms = (time.perf_counter() - t0) * 1000.0 / len(all_samples)
        raw_cer, raw_wer, raw_emr = cer_wer(refs, raw_hyps)
        pp_cer, pp_wer, pp_emr = cer_wer(refs, pp_hyps)

        exact_raw = sum(1 for r, h in zip(refs, raw_hyps) if r == h)
        exact_pp = sum(1 for r, h in zip(refs, pp_hyps) if r == h)

        results[name] = {
            "raw_cer_pct": round(raw_cer * 100, 2),
            "raw_wer_pct": round(raw_wer * 100, 2),
            "raw_exact": f"{exact_raw}/{len(all_samples)} ({exact_raw/len(all_samples)*100:.1f}%)",
            "pp_cer_pct": round(pp_cer * 100, 2),
            "pp_wer_pct": round(pp_wer * 100, 2),
            "pp_exact": f"{exact_pp}/{len(all_samples)} ({exact_pp/len(all_samples)*100:.1f}%)",
            "latency_ms_per_line": round(dur_ms, 1),
        }
        print(f"    Raw CER: {results[name]['raw_cer_pct']}%, Exact Lines: {results[name]['raw_exact']}")

    out_json = BASE_DIR / "results" / "ablation_2x2_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nNəticələr qeyd edildi: {out_json}")


if __name__ == "__main__":
    main()
