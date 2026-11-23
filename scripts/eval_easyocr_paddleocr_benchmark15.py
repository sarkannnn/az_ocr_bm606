"""15 Belgelik Benchmark Üzerinde EasyOCR ve PaddleOCR Değerlendirmesi.

Bu betik 15 müstəqil smartfon sənəd fotosunun (295 sətir) EasyOCR (və mövcuddursa PaddleOCR)
çıxışlarını qiymətləndirir və makalənin Table 12 cədvəlinə əlavə ediləcək metrikaları çıxarır.
"""
import os
import sys
import time
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common import normalize, cer_wer

def main():
    print("=" * 75)
    print("EASYOCR & PADDLEOCR 15-SƏNƏDLİK BENCHMARK TESTİ (295 SƏTİR)")
    print("=" * 75)

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

    print(f"Qiymətləndiriləcək sətir sayı: {len(all_samples)}")

    results = {}

    # 1. EasyOCR Testi
    try:
        import easyocr
        print("\n[+] EasyOCR yüklənir (CPU, lang=['az'])...")
        reader = easyocr.Reader(['az'], gpu=False)

        refs, hyps = [], []
        t0 = time.perf_counter()
        for cp, ref in all_samples:
            res = reader.recognize(cp, detail=0)
            pred = normalize(" ".join(res).replace("—", "-")) if res else ""
            refs.append(ref)
            hyps.append(pred)

        total_sec = time.perf_counter() - t0
        cer, wer, emr = cer_wer(refs, hyps)
        exact = sum(1 for r, h in zip(refs, hyps) if r == h)

        results["EasyOCR_az"] = {
            "status": "COMPLETED",
            "cer_pct": round(cer * 100, 2),
            "wer_pct": round(wer * 100, 2),
            "exact_lines": f"{exact}/{len(all_samples)} ({exact/len(all_samples)*100:.1f}%)",
            "latency_ms_per_line": round(total_sec * 1000.0 / len(all_samples), 1)
        }
        print(f"EasyOCR Nəticələri: CER={results['EasyOCR_az']['cer_pct']}%, WER={results['EasyOCR_az']['wer_pct']}%, Exact={results['EasyOCR_az']['exact_lines']}")
    except Exception as e:
        print(f"[-] EasyOCR işlədilmədi: {e}")
        results["EasyOCR_az"] = {"status": "FAILED", "reason": str(e)}

    # 2. PaddleOCR Testi
    try:
        from paddleocr import PaddleOCR
        print("\n[+] PaddleOCR yüklənir...")
        ocr = PaddleOCR(use_angle_cls=False, lang='latin', use_gpu=False)

        refs, hyps = [], []
        t0 = time.perf_counter()
        for cp, ref in all_samples:
            res = ocr.ocr(cp, cls=False)
            pred = ""
            if res and res[0]:
                pred = " ".join([line[1][0] for line in res[0]])
            pred = normalize(pred.replace("—", "-"))
            refs.append(ref)
            hyps.append(pred)

        total_sec = time.perf_counter() - t0
        cer, wer, emr = cer_wer(refs, hyps)
        exact = sum(1 for r, h in zip(refs, hyps) if r == h)

        results["PaddleOCR"] = {
            "status": "COMPLETED",
            "cer_pct": round(cer * 100, 2),
            "wer_pct": round(wer * 100, 2),
            "exact_lines": f"{exact}/{len(all_samples)} ({exact/len(all_samples)*100:.1f}%)",
            "latency_ms_per_line": round(total_sec * 1000.0 / len(all_samples), 1)
        }
    except Exception as e:
        print(f"[-] PaddleOCR işlədilmədi: {e}")
        results["PaddleOCR"] = {
            "status": "NOT_INSTALLED",
            "reason": "PaddleOCR mühitdə quraşdırılmayıb və ya Azərbaycan Latin xüsusi hərfləri (ə, ı, ö, ğ, ü, ş, ç) rəsmi modeldə məhduddur.",
            "error": str(e)
        }

    out_json = BASE_DIR / "results" / "baselines_benchmark15.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nBaseline nəticələri qeyd edildi: {out_json}")


if __name__ == "__main__":
    main()
