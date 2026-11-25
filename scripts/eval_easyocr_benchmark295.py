"""15 Belgelik Rəsmi Benchmark (295 Sətir) Üzərində EasyOCR Dəyərləndirməsi.

Bu betik EasyOCR 1.7.2 modelini heç bir yenidən təlim aparmadan,
rəsmi 15 telefon sənədi benchmarkının 295 sətirinin tamamında qiymətləndirir:
- 55 sətir real fiziki smartfon fotoları (B1, B2, B3)
- 240 sətir simulyasiya edilmiş smartfon fotoları (B4 - B15)

Metrikalar: CER, WER, Exact Match, Hərf-hərf xəta analizi (Azərbaycan diakritikləri).
Nəticələr: results/easyocr_295/ qovluğuna qeyd olunur.
"""
import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common import normalize, cer_wer
from scripts.eval_ablation_models import build_genuine_295_benchmark

def char_audit(refs, hyps):
    lower_special = set("əıöğüşç")
    upper_special = set("ƏİÖĞÜŞÇI")
    i_family = set("iıİI")
    
    stats = {
        "lower_special": [0, 0],
        "upper_special": [0, 0],
        "i_family": [0, 0],
        "schwa": [0, 0],
        "standard_lower": [0, 0],
        "standard_upper": [0, 0],
        "digits_punct": [0, 0]
    }
    
    import difflib
    for r, h in zip(refs, hyps):
        sm = difflib.SequenceMatcher(None, r, h)
        # Count target occurrences
        for ch in r:
            if ch in lower_special:
                stats["lower_special"][1] += 1
            elif ch in upper_special:
                stats["upper_special"][1] += 1
            if ch in i_family:
                stats["i_family"][1] += 1
            if ch in "əƏ":
                stats["schwa"][1] += 1
            if 'a' <= ch <= 'z':
                stats["standard_lower"][1] += 1
            elif 'A' <= ch <= 'Z':
                stats["standard_upper"][1] += 1
            elif ch.isdigit() or ch in ".,:;!?-_()\"'":
                stats["digits_punct"][1] += 1

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'delete'):
                sub_r = r[i1:i2]
                for ch in sub_r:
                    if ch in lower_special:
                        stats["lower_special"][0] += 1
                    elif ch in upper_special:
                        stats["upper_special"][0] += 1
                    if ch in i_family:
                        stats["i_family"][0] += 1
                    if ch in "əƏ":
                        stats["schwa"][0] += 1
                    if 'a' <= ch <= 'z':
                        stats["standard_lower"][0] += 1
                    elif 'A' <= ch <= 'Z':
                        stats["standard_upper"][0] += 1
                    elif ch.isdigit() or ch in ".,:;!?-_()\"'":
                        stats["digits_punct"][0] += 1

    out = {}
    for k, (err, tot) in stats.items():
        acc = round(100.0 * (1.0 - err / max(tot, 1)), 2)
        err_pct = round(100.0 * err / max(tot, 1), 2)
        out[k] = {"errors": err, "total": tot, "accuracy_pct": acc, "error_rate_pct": err_pct}
    return out


def main():
    print("=" * 80)
    print("AZOCR-V2: EASYOCR 15-SƏNƏDLİK (295 SƏTİR) RƏSMİ BENCHMARK DEĞERLENDİRMƏSİ")
    print("=" * 80)
    
    import easyocr
    print("[+] EasyOCR modeli yüklənir (dil konfiqurasiyası: ['az'], CPU)...")
    t_load0 = time.perf_counter()
    reader = easyocr.Reader(['az'], gpu=False, verbose=False)
    load_time = time.perf_counter() - t_load0
    print(f"[+] EasyOCR uğurla yükləndi ({load_time:.2f} saniyə).")

    # Model info
    rec_params = sum(p.numel() for p in reader.recognizer.parameters())
    det_params = sum(p.numel() for p in reader.detector.parameters())
    model_dir = Path.home() / ".EasyOCR" / "model"
    rec_mb = round((model_dir / "latin_g2.pth").stat().st_size / (1024 * 1024), 2) if (model_dir / "latin_g2.pth").exists() else 0
    det_mb = round((model_dir / "craft_mlt_25k.pth").stat().st_size / (1024 * 1024), 2) if (model_dir / "craft_mlt_25k.pth").exists() else 0

    print(f"    - EasyOCR Versiyası: {easyocr.__version__}")
    print(f"    - Recognizer: latin_g2 ({rec_params:,} parametr, {rec_mb} MB)")
    print(f"    - Detector: craft_mlt_25k ({det_params:,} parametr, {det_mb} MB)")
    print(f"    - Toplam Parametr: {rec_params + det_params:,} ({(rec_mb + det_mb):.1f} MB)")
    print(f"    - Dil Kodu: 'az' (EasyOCR daxilində ümumi 'latin_g2' modelinə yönləndirilir)")

    samples = build_genuine_295_benchmark()
    print(f"\n[+] Toplam 295 sətir hazırlandı (55 real + 240 simulyasiya).")

    all_records = []
    latencies = []
    
    print("\n[+] EasyOCR ilə sətir-sətir tanınma başlayır...")
    t_start = time.perf_counter()
    
    for i, (tag, code, idx, cp, gt) in enumerate(samples):
        t0 = time.perf_counter()
        res = reader.recognize(str(cp), detail=0)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        
        pred = normalize(" ".join(res).replace("—", "-")) if res else ""
        
        c_err, w_err, _ = cer_wer([gt], [pred])
        exact = 1 if gt == pred else 0
        
        is_real = 1 if tag in ("B1", "B2", "B3") else 0
        
        # Map IMG_5121 -> B1, IMG_5122 -> B2, ..., IMG_5135 -> B15
        b_num = int(code.replace("IMG_", "")) - 5120
        b_tag = f"B{b_num}"
        
        all_records.append({
            "doc_id": b_tag,
            "img_code": code,
            "line_idx": idx,
            "crop_path": str(cp),
            "is_real": is_real,
            "ground_truth": gt,
            "easyocr_pred": pred,
            "exact_match": exact,
            "line_cer_pct": round(c_err * 100, 2),
            "line_wer_pct": round(w_err * 100, 2),
            "latency_ms": round(dt, 1)
        })
        
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"    İrəliləyiş: {i+1}/{len(samples)} sətir tamamlandı (orta sürət: {np.mean(latencies):.1f} ms/sətir)...")

    total_time = time.perf_counter() - t_start
    print(f"[+] Bütün 295 sətir tamamlandı ({total_time:.2f} saniyə).")

    # Aggregate metrics
    all_refs = [r["ground_truth"] for r in all_records]
    all_preds = [r["easyocr_pred"] for r in all_records]
    
    overall_cer, overall_wer, _ = cer_wer(all_refs, all_preds)
    overall_exact = sum(r["exact_match"] for r in all_records)
    
    real_refs = [r["ground_truth"] for r in all_records if r["is_real"] == 1]
    real_preds = [r["easyocr_pred"] for r in all_records if r["is_real"] == 1]
    real_cer, real_wer, _ = cer_wer(real_refs, real_preds)
    real_exact = sum(r["exact_match"] for r in all_records if r["is_real"] == 1)

    sim_refs = [r["ground_truth"] for r in all_records if r["is_real"] == 0]
    sim_preds = [r["easyocr_pred"] for r in all_records if r["is_real"] == 0]
    sim_cer, sim_wer, _ = cer_wer(sim_refs, sim_preds)
    sim_exact = sum(r["exact_match"] for r in all_records if r["is_real"] == 0)

    # Per-doc metrics
    doc_metrics = {}
    doc_keys = [f"B{k}" for k in range(1, 16)]
    for dk in doc_keys:
        d_refs = [r["ground_truth"] for r in all_records if r["doc_id"] == dk]
        d_preds = [r["easyocr_pred"] for r in all_records if r["doc_id"] == dk]
        d_cer, d_wer, _ = cer_wer(d_refs, d_preds)
        d_exact = sum(1 for r, p in zip(d_refs, d_preds) if r == p)
        d_lats = [r["latency_ms"] for r in all_records if r["doc_id"] == dk]
        doc_metrics[dk] = {
            "n_lines": len(d_refs),
            "cer_pct": round(d_cer * 100, 2),
            "wer_pct": round(d_wer * 100, 2),
            "accuracy_char_pct": round((1.0 - d_cer) * 100, 2),
            "exact_match": f"{d_exact}/{len(d_refs)} ({d_exact/len(d_refs)*100:.1f}%)",
            "mean_latency_ms": round(float(np.mean(d_lats)), 1)
        }

    # Special characters audit
    audit = char_audit(all_refs, all_preds)
    real_audit = char_audit(real_refs, real_preds)
    sim_audit = char_audit(sim_refs, sim_preds)

    out_data = {
        "easyocr_info": {
            "version": easyocr.__version__,
            "language_requested": ["az"],
            "model_recognizer": "latin_g2",
            "model_detector": "craft_mlt_25k",
            "recognizer_params": rec_params,
            "detector_params": det_params,
            "total_params": rec_params + det_params,
            "model_disk_mb": round(rec_mb + det_mb, 2),
            "is_native_azerbaijani_model": False,
            "note": "EasyOCR lacks a dedicated Azerbaijani OCR model. The 'az' tag maps to generalist 'latin_g2' Latin recognizer with no Azerbaijani lexicon/LM."
        },
        "overall_295": {
            "n_lines": len(all_records),
            "total_chars": sum(len(r) for r in all_refs),
            "cer_pct": round(overall_cer * 100, 2),
            "accuracy_char_pct": round((1.0 - overall_cer) * 100, 2),
            "wer_pct": round(overall_wer * 100, 2),
            "accuracy_word_pct": round((1.0 - overall_wer) * 100, 2),
            "exact_match": f"{overall_exact}/{len(all_records)} ({overall_exact/len(all_records)*100:.1f}%)",
            "mean_latency_ms_per_line": round(float(np.mean(latencies)), 1)
        },
        "subsets": {
            "real_photos_55": {
                "n_lines": len(real_refs),
                "total_chars": sum(len(r) for r in real_refs),
                "cer_pct": round(real_cer * 100, 2),
                "accuracy_char_pct": round((1.0 - real_cer) * 100, 2),
                "wer_pct": round(real_wer * 100, 2),
                "exact_match": f"{real_exact}/{len(real_refs)} ({real_exact/len(real_refs)*100:.1f}%)"
            },
            "simulated_photos_240": {
                "n_lines": len(sim_refs),
                "total_chars": sum(len(r) for r in sim_refs),
                "cer_pct": round(sim_cer * 100, 2),
                "accuracy_char_pct": round((1.0 - sim_cer) * 100, 2),
                "wer_pct": round(sim_wer * 100, 2),
                "exact_match": f"{sim_exact}/{len(sim_refs)} ({sim_exact/len(sim_refs)*100:.1f}%)"
            }
        },
        "per_document": doc_metrics,
        "special_characters_audit": audit,
        "real_special_characters_audit": real_audit,
        "sim_special_characters_audit": sim_audit
    }

    out_dir = BASE_DIR / "results" / "easyocr_295"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "easyocr_295_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "easyocr_295_predictions.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    # Generate Markdown Report
    report_path = out_dir / "easyocr_295_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EasyOCR 15-Sənədlik (295 Sətir) Rəsmi Benchmark Qiymətləndirmə Hesabatı\n\n")
        f.write("Bu hesabat AzOCR layihəsinin rəsmi 15 telefon sənədi benchmarkı (295 sətir) üzərində ")
        f.write("**EasyOCR 1.7.2** modelinin birbaşa qiymətləndirmə nəticələrini təqdim edir.\n\n")
        
        f.write("## 1. Model Məlumatı və Azərbaycan Dili Dəstəyi\n\n")
        f.write(f"- **EasyOCR Versiyası:** `{easyocr.__version__}`\n")
        f.write(f"- **Konfiqurasiya Edilmiş Dil:** `['az']`\n")
        f.write(f"- **İstifadə Edilən Tanıyıcı (Recognizer):** `latin_g2` ({rec_params:,} parametr, {rec_mb} MB)\n")
        f.write(f"- **İstifadə Edilən Detektor (Detector):** `craft_mlt_25k` ({det_params:,} parametr, {det_mb} MB)\n")
        f.write(f"- **Doğma Azərbaycan Modeli Mövcuddurmu?:** **XEYR.** EasyOCR kitabxanasında Azərbaycan dili üçün ayrıca ixtisaslaşmış təlim çəkisi yoxdur. `'az'` dil kodu birbaşa 341 simvolluq ümumi `latin_g2` Latın modelinə yönləndirilir. Bu səbəbdən modeldə Azərbaycan dili üçün lüğət, n-gram və ya morfoloji dil modeli iştirak etmir.\n\n")

        f.write("## 2. Ümumi Nəticələr və Alt-Qrup Bölgüsü\n\n")
        f.write("| Qrup | Sətir Sayı | Simvol Sayı | CER (%) ↓ | Simvol Dəqiqliyi (%) ↑ | WER (%) ↓ | Tam Düzgün Sətir (Exact Match) ↑ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Bütün Benchmark (B1–B15)** | **295** | {sum(len(r) for r in all_refs):,} | **{overall_cer*100:.2f}%** | {out_data['overall_295']['accuracy_char_pct']:.2f}% | **{overall_wer*100:.2f}%** | **{out_data['overall_295']['exact_match']}** |\n")
        f.write(f"| Real Telefon Fotoları (B1–B3) | 55 | {sum(len(r) for r in real_refs):,} | **{real_cer*100:.2f}%** | {out_data['subsets']['real_photos_55']['accuracy_char_pct']:.2f}% | **{real_wer*100:.2f}%** | **{out_data['subsets']['real_photos_55']['exact_match']}** |\n")
        f.write(f"| Simulyasiya Fotoları (B4–B15) | 240 | {sum(len(r) for r in sim_refs):,} | **{sim_cer*100:.2f}%** | {out_data['subsets']['simulated_photos_240']['accuracy_char_pct']:.2f}% | **{sim_wer*100:.2f}%** | **{out_data['subsets']['simulated_photos_240']['exact_match']}** |\n\n")

        f.write("## 3. Digər Modellərlə Müqayisə (B1–B15, n = 295 Sətir)\n\n")
        f.write("| Model / Alqoritm | CER (%) ↓ | WER (%) ↓ | Exact Match ↑ | Orta Latensiya (ms) ↓ | Model Həcmi |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write("| **CRNN-V2 + SymSpell (Təklif olunan)** | **0.29%** | **1.37%** | **259/295 (87.8%)** | 86.4 ms | 98.7 MB |\n")
        f.write("| CRNN-V2 (Xam CTC) | **0.30%** | **1.64%** | **253/295 (85.8%)** | 86.3 ms | 98.7 MB |\n")
        f.write("| Tesseract 5 LSTM (`aze`, `--psm 7`) | **0.31%** | **1.55%** | **250/295 (84.7%)** | 182.2 ms | 5.99 MB |\n")
        f.write("| Tesseract 5 LSTM (`aze`, `--psm 13`) | **0.47%** | **2.40%** | **230/295 (78.0%)** | 141.4 ms | 5.99 MB |\n")
        f.write(f"| **EasyOCR 1.7.2 (`az` / `latin_g2`)** | **{overall_cer*100:.2f}%** | **{overall_wer*100:.2f}%** | **{out_data['overall_295']['exact_match']}** | {out_data['overall_295']['mean_latency_ms_per_line']} ms | 94.0 MB |\n")
        f.write("| CRNN-V1 (Baza — Augmentasiyasız) | **12.37%** | **42.49%** | 37/295 (12.5%) | 85.2 ms | 32.9 MB |\n\n")

        f.write("## 4. B1–B15 Sənədləri Üzrə Təfərrüatlı Nəticələr\n\n")
        f.write("| Sənəd Kodu | Tip | Şrift | Təhrif Dərəcəsi | Sətir | CER (%) | WER (%) | Exact Match | Latensiya (ms) |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        doc_info = {
            "B1": ("Real Telefon (IMG_5121)", "Times New Roman", "Real L2"),
            "B2": ("Real Telefon (IMG_5122)", "Calibri", "Real L2"),
            "B3": ("Real Telefon (IMG_5123)", "Times New Roman", "Real L3"),
            "B4": ("Simulyasiya (IMG_5124)", "Arial", "L2"),
            "B5": ("Simulyasiya (IMG_5125)", "Tahoma", "L1"),
            "B6": ("Simulyasiya (IMG_5126)", "Cambria", "L2"),
            "B7": ("Simulyasiya (IMG_5127)", "MS Sans Serif", "L3"),
            "B8": ("Simulyasiya (IMG_5128)", "Tahoma", "L4"),
            "B9": ("Simulyasiya (IMG_5129)", "Consolas", "L1"),
            "B10": ("Simulyasiya (IMG_5130)", "Cambria", "L2"),
            "B11": ("Simulyasiya (IMG_5131)", "Verdana", "L3"),
            "B12": ("Simulyasiya (IMG_5132)", "Courier New", "L2"),
            "B13": ("Simulyasiya (IMG_5133)", "Segoe UI", "L3"),
            "B14": ("Simulyasiya (IMG_5134)", "MS Sans Serif", "L4"),
            "B15": ("Simulyasiya (IMG_5135)", "Candara", "L4"),
        }
        
        for dk in doc_keys:
            info = doc_info.get(dk, ("-", "-", "-"))
            m = doc_metrics[dk]
            f.write(f"| **{dk}** | {info[0]} | {info[1]} | {info[2]} | {m['n_lines']} | {m['cer_pct']}% | {m['wer_pct']}% | {m['exact_match']} | {m['mean_latency_ms']} ms |\n")

        f.write("\n## 5. Xüsusi Qrafemlər və Diakritik Hərflər Auditi\n\n")
        f.write("| Qrafem / Simvol Kateqoriyası | Nümunə Sayı (n) | EasyOCR Xəta Sayı | EasyOCR Xəta (%) | EasyOCR Dəqiqlik (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for k, name in [
            ("lower_special", "Kiçik Xüsusi Hərflər (ə,ı,ö,ğ,ü,ş,ç)"),
            ("upper_special", "Böyük Xüsusi Hərflər (Ə,İ,Ö,Ğ,Ü,Ş,Ç,I)"),
            ("i_family", "4-Tərəfli i / ı / İ / I Ailəsi"),
            ("schwa", "Ə hərfi (ə / Ə)"),
            ("standard_lower", "Standart Kiçik Hərflər (a–z)"),
            ("standard_upper", "Standart Böyük Hərflər (A–Z)"),
            ("digits_punct", "Rəqəmlər və Durğu İşarələri")
        ]:
            item = audit[k]
            f.write(f"| {name} | {item['total']} | {item['errors']} | {item['error_rate_pct']}% | {item['accuracy_pct']}% |\n")

    print(f"\n[+] Nəticələr qeyd edildi:")
    print(f"    - JSON: {json_path}")
    print(f"    - CSV:  {csv_path}")
    print(f"    - MD:   {report_path}")

    print("\n" + "=" * 80)
    print("XÜLASƏ:")
    print(f"EasyOCR 295 Sətir: CER = {overall_cer*100:.2f}%, WER = {overall_wer*100:.2f}%, Exact = {out_data['overall_295']['exact_match']}")
    print(f"  - Real Fotolar (55 sətir): CER = {real_cer*100:.2f}%, WER = {real_wer*100:.2f}%, Exact = {out_data['subsets']['real_photos_55']['exact_match']}")
    print(f"  - Simulyasiya (240 sətir): CER = {sim_cer*100:.2f}%, WER = {sim_wer*100:.2f}%, Exact = {out_data['subsets']['simulated_photos_240']['exact_match']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
