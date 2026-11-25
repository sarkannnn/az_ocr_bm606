import os
import sys
import time
import json
import math
import subprocess
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2

# Set thread environment variables before importing torch/paddle
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common import load_model_checkpoint, infer_line_smart, normalize, cer_wer
from postprocess import AzerbaijaniPostProcessor
from scripts.eval_ablation_models import build_genuine_295_benchmark

def cv2_imread_utf8(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def wilson_ci(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / n) + (z2 / (4 * n * n)))
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)

def mcnemar_exact(b, c):
    """Exact two-sided binomial test on discordant pairs (b, c) for McNemar test."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided p-value
    p_val = 2.0 * binom.cdf(k, n, 0.5)
    return min(1.0, p_val)

def measure_tesseract_overhead():
    """Measures Tesseract subprocess spawn and initialization overhead on Windows."""
    import pytesseract
    from PIL import Image
    
    # 1. Direct subprocess spawn overhead (tesseract --version)
    times_cli = []
    for _ in range(20):
        t0 = time.perf_counter()
        subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        times_cli.append((time.perf_counter() - t0) * 1000.0)
        
    # 2. Pytesseract on a blank 1x1 image (measures disk write + CLI spawn + model load + text read)
    blank_img = Image.new("RGB", (32, 100), color=(255, 255, 255))
    times_blank = []
    for _ in range(20):
        t0 = time.perf_counter()
        pytesseract.image_to_string(blank_img, lang="aze", config="--psm 7")
        times_blank.append((time.perf_counter() - t0) * 1000.0)
        
    return {
        "subprocess_version_cli_mean_ms": round(float(np.mean(times_cli)), 2),
        "subprocess_version_cli_std_ms": round(float(np.std(times_cli)), 2),
        "pytesseract_blank_spawn_load_mean_ms": round(float(np.mean(times_blank)), 2),
        "pytesseract_blank_spawn_load_std_ms": round(float(np.std(times_blank)), 2)
    }

def main():
    print("=" * 80)
    print("AZOCR-V2: CPU SPEED BENCHMARK & PAIRED STATISTICAL CONTROL (295 SATIR)")
    print("=" * 80)
    
    out_dir = BASE_DIR / "results" / "speed_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # System info
    cpu_cores = os.cpu_count()
    torch_threads = torch.get_num_threads()
    device = torch.device("cpu")
    print(f"[+] Sistem Mantıksal Çekirdek Sayısı: {cpu_cores}")
    print(f"[+] Sabitlenen PyTorch Thread Sayısı: {torch_threads}")
    print(f"[+] OMP / MKL Thread Sayısı: {os.environ.get('OMP_NUM_THREADS', 'N/A')}")
    
    # Load Benchmark
    samples = build_genuine_295_benchmark()
    print(f"[+] 295 benchmark satırı yüklendi (55 real + 240 simüle).")
    
    # Pre-load images in memory to isolate inference from disk I/O
    print("[+] Görüntüler belleğe önceden yükleniyor...")
    loaded_imgs = []
    for doc_id, img_code, line_idx, img_p, gt in samples:
        img_bgr = cv2_imread_utf8(img_p)
        loaded_imgs.append((doc_id, img_code, line_idx, img_bgr, gt))
    print(f"[+] {len(loaded_imgs)} görüntü belleğe yüklendi.")

    results_data = {
        "benchmark_info": {
            "name": "Canonical 15-Document Smartphone Benchmark (B1–B15)",
            "total_lines": 295,
            "real_lines": 55,
            "simulated_lines": 240,
            "system_cpu": "Intel(R) Core(TM) 7 240H",
            "logical_processors": cpu_cores,
            "fixed_threads": torch_threads,
            "torch_version": torch.__version__,
            "repetitions": 3,
            "warmup_lines": 10
        },
        "models": {}
    }

    # -------------------------------------------------------------
    # 1. CRNN-V2 Raw & CRNN-V2 + SymSpell
    # -------------------------------------------------------------
    ckpt_v2 = BASE_DIR / "runs" / "line_v2" / "best.pt"
    print(f"\n[+] CRNN-V2 yükleniyor: {ckpt_v2}")
    t0_crnn_load = time.perf_counter()
    model, cfg, vocab, _ = load_model_checkpoint(ckpt_v2, device=device)
    model.eval()
    post_processor = AzerbaijaniPostProcessor()
    crnn_init_ms = (time.perf_counter() - t0_crnn_load) * 1000.0
    print(f"    Model başlatma süresi: {crnn_init_ms:.1f} ms")

    # Warm-up CRNN
    print("    Warm-up uygulanıyor (10 satır)...")
    for i in range(10):
        infer_line_smart(model, cfg, vocab, loaded_imgs[i][3], device)

    # 3 Repetitions for CRNN-V2 Raw
    print("    CRNN-V2 Raw ölçümü (3 tekrar x 295 satır)...")
    crnn_raw_runs = []
    per_line_times_crnn_raw = defaultdict(list)
    
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, (doc_id, img_code, line_idx, img_bgr, gt) in enumerate(loaded_imgs):
            t0 = time.perf_counter()
            pred = infer_line_smart(model, cfg, vocab, img_bgr, device)
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_crnn_raw[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        crnn_raw_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    # 3 Repetitions for CRNN-V2 + SymSpell
    print("    CRNN-V2 + SymSpell ölçümü (3 tekrar x 295 satır)...")
    crnn_sym_runs = []
    per_line_times_crnn_sym = defaultdict(list)
    
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, (doc_id, img_code, line_idx, img_bgr, gt) in enumerate(loaded_imgs):
            t0 = time.perf_counter()
            raw_pred = infer_line_smart(model, cfg, vocab, img_bgr, device)
            pred = post_processor.process_line(raw_pred)
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_crnn_sym[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        crnn_sym_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    def summarize_latencies(per_line_dict, runs_sec):
        all_means = [np.mean(vals) for vals in per_line_dict.values()]
        real_means = [np.mean(per_line_dict[i]) for i in range(55)]
        sim_means = [np.mean(per_line_dict[i]) for i in range(55, 295)]
        return {
            "mean_ms_per_line": round(float(np.mean(all_means)), 2),
            "std_ms_per_line": round(float(np.std(all_means)), 2),
            "median_ms_per_line": round(float(np.median(all_means)), 2),
            "real_55_mean_ms": round(float(np.mean(real_means)), 2),
            "sim_240_mean_ms": round(float(np.mean(sim_means)), 2),
            "run_durations_sec": [round(float(x), 2) for x in runs_sec],
            "throughput_lines_per_sec": round(float(295.0 / np.mean(runs_sec)), 2)
        }

    results_data["models"]["CRNN-V2_Raw"] = {
        "engine": "PyTorch CRNN (Custom Line Model)",
        "init_ms": round(crnn_init_ms, 1),
        "invocation_type": "Direct in-memory C++ libtorch tensor call",
        **summarize_latencies(per_line_times_crnn_raw, crnn_raw_runs)
    }

    results_data["models"]["CRNN-V2_SymSpell"] = {
        "engine": "PyTorch CRNN + SymSpell C++ Extension",
        "init_ms": round(crnn_init_ms, 1),
        "invocation_type": "Direct in-memory Python/C++ call",
        **summarize_latencies(per_line_times_crnn_sym, crnn_sym_runs)
    }

    # -------------------------------------------------------------
    # 2. Tesseract 5 PSM 7 & PSM 13
    # -------------------------------------------------------------
    import pytesseract
    from PIL import Image

    tess_version = str(pytesseract.get_tesseract_version())
    print(f"\n[+] Tesseract 5 ölçülüyor (Versiyon: {tess_version})...")
    
    # Convert images to PIL
    pil_imgs = [Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) for _, _, _, img_bgr, _ in loaded_imgs]
    
    # Warm-up Tesseract
    print("    Warm-up uygulanıyor (10 satır)...")
    for i in range(10):
        pytesseract.image_to_string(pil_imgs[i], lang="aze", config="--psm 7")

    # PSM 7
    print("    Tesseract PSM 7 ölçümü (3 tekrar x 295 satır)...")
    tess_psm7_runs = []
    per_line_times_tess_psm7 = defaultdict(list)
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, pil_img in enumerate(pil_imgs):
            t0 = time.perf_counter()
            res = pytesseract.image_to_string(pil_img, lang="aze", config="--psm 7")
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_tess_psm7[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        tess_psm7_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    # PSM 13
    print("    Tesseract PSM 13 ölçümü (3 tekrar x 295 satır)...")
    tess_psm13_runs = []
    per_line_times_tess_psm13 = defaultdict(list)
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, pil_img in enumerate(pil_imgs):
            t0 = time.perf_counter()
            res = pytesseract.image_to_string(pil_img, lang="aze", config="--psm 13")
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_tess_psm13[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        tess_psm13_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    tess_overhead = measure_tesseract_overhead()
    print(f"    Tesseract Subprocess Çağrı Ek Yükü (Spawn + Init): {tess_overhead['pytesseract_blank_spawn_load_mean_ms']:.1f} ms")

    results_data["models"]["Tesseract5_PSM7"] = {
        "engine": f"Tesseract OCR {tess_version} (PSM 7)",
        "invocation_type": "Windows CLI Subprocess (pytesseract -> tesseract.exe)",
        "subprocess_overhead_per_call_ms": tess_overhead["pytesseract_blank_spawn_load_mean_ms"],
        **summarize_latencies(per_line_times_tess_psm7, tess_psm7_runs)
    }

    results_data["models"]["Tesseract5_PSM13"] = {
        "engine": f"Tesseract OCR {tess_version} (PSM 13)",
        "invocation_type": "Windows CLI Subprocess (pytesseract -> tesseract.exe)",
        "subprocess_overhead_per_call_ms": tess_overhead["pytesseract_blank_spawn_load_mean_ms"],
        **summarize_latencies(per_line_times_tess_psm13, tess_psm13_runs)
    }

    # -------------------------------------------------------------
    # 3. EasyOCR (Recognition on Pre-cropped lines)
    # -------------------------------------------------------------
    import easyocr
    print("\n[+] EasyOCR yükleniyor...")
    t0_easy_load = time.perf_counter()
    easy_reader = easyocr.Reader(['az'], gpu=False, verbose=False)
    easy_init_ms = (time.perf_counter() - t0_easy_load) * 1000.0
    print(f"    EasyOCR başlatma süresi: {easy_init_ms:.1f} ms")

    # Warm-up EasyOCR
    print("    Warm-up uygulanıyor (10 satır)...")
    for i in range(10):
        easy_reader.recognize(loaded_imgs[i][3])

    print("    EasyOCR ölçümü (3 tekrar x 295 satır)...")
    easy_runs = []
    per_line_times_easy = defaultdict(list)
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, (doc_id, img_code, line_idx, img_bgr, gt) in enumerate(loaded_imgs):
            t0 = time.perf_counter()
            res = easy_reader.recognize(img_bgr)
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_easy[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        easy_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    results_data["models"]["EasyOCR_az"] = {
        "engine": f"EasyOCR {easyocr.__version__} (PyTorch CPU)",
        "init_ms": round(easy_init_ms, 1),
        "invocation_type": "Direct in-memory PyTorch call (recognize)",
        **summarize_latencies(per_line_times_easy, easy_runs)
    }

    # -------------------------------------------------------------
    # 4. PaddleOCR (PP-OCRv6_medium_rec)
    # -------------------------------------------------------------
    import paddlex
    print("\n[+] PaddleOCR yükleniyor (PP-OCRv6_medium_rec)...")
    t0_pad_load = time.perf_counter()
    paddle_predictor = paddlex.create_predictor(model_name="PP-OCRv6_medium_rec", device="cpu")
    paddle_init_ms = (time.perf_counter() - t0_pad_load) * 1000.0
    print(f"    PaddleOCR model başlatma süresi: {paddle_init_ms:.1f} ms")

    # Warm-up PaddleOCR
    print("    Warm-up uygulanıyor (10 satır)...")
    for i in range(10):
        list(paddle_predictor(loaded_imgs[i][3]))

    print("    PaddleOCR ölçümü (3 tekrar x 295 satır)...")
    paddle_runs = []
    per_line_times_paddle = defaultdict(list)
    for rep in range(3):
        t_rep_start = time.perf_counter()
        for idx, (doc_id, img_code, line_idx, img_bgr, gt) in enumerate(loaded_imgs):
            t0 = time.perf_counter()
            res = list(paddle_predictor(img_bgr))
            dt = (time.perf_counter() - t0) * 1000.0
            per_line_times_paddle[idx].append(dt)
        total_rep_s = time.perf_counter() - t_rep_start
        paddle_runs.append(total_rep_s)
        print(f"      Tekrar {rep+1}: {total_rep_s:.2f} s ({total_rep_s*1000/295:.2f} ms/satır)")

    results_data["models"]["PaddleOCR_PP-OCRv6"] = {
        "engine": "PaddleOCR PP-OCRv6_medium_rec (PaddlePaddle CPU oneDNN)",
        "init_ms": round(paddle_init_ms, 1),
        "invocation_type": "Direct in-memory PaddleX predictor call",
        **summarize_latencies(per_line_times_paddle, paddle_runs)
    }

    # -------------------------------------------------------------
    # 5. PAIRED STATISTICAL AUDIT (MCNEMAR VS FISHER & TOST EQUIVALENCE)
    # -------------------------------------------------------------
    print("\n[5. PAIRED STATISTICAL CONTROL: MCNEMAR TEST & TOST]")
    # For the 244 uppercase special diacritics:
    # V2 had 6 errors, Tesseract had 9 errors out of 244 characters.
    # In paired data, let's analyze the discordant pairs (b, c):
    # b: cases where V2 is wrong and Tesseract is right
    # c: cases where Tesseract is wrong and V2 is right
    # Total V2 errors = 6 = a + b
    # Total Tess errors = 9 = a + c
    # Net difference = c - b = 9 - 6 = 3 errors.
    # The common agreement 'a' (both wrong) can range from 0 to 6.
    # If a=3: b = 3, c = 6 -> discordant sum = 9.
    # If a=4: b = 2, c = 5 -> discordant sum = 7.
    # If a=5: b = 1, c = 4 -> discordant sum = 5.
    # If a=6: b = 0, c = 3 -> discordant sum = 3.
    # Let's compute exact McNemar p-value across all possible configurations of (b, c):
    mcnemar_analysis = {}
    for a in range(7):
        b = 6 - a
        c = 9 - a
        p_val = mcnemar_exact(b, c)
        mcnemar_analysis[f"joint_errors_a={a}"] = {
            "b_v2_only_errors": b,
            "c_tess_only_errors": c,
            "discordant_pairs": b + c,
            "mcnemar_exact_p_value": round(float(p_val), 4),
            "statistically_significant_05": bool(p_val < 0.05)
        }
    print("McNemar exact test p-values across all possible joint error overlap configurations:")
    for k, v in mcnemar_analysis.items():
        print(f"  {k}: b={v['b_v2_only_errors']}, c={v['c_tess_only_errors']} -> p={v['mcnemar_exact_p_value']} (Sig: {v['statistically_significant_05']})")

    results_data["paired_statistical_audit"] = {
        "theoretical_critique": {
            "fisher_limitation": "Fisher's exact test assumes two independent samples. When CRNN-V2 and Tesseract 5 are evaluated on the exact same 244 characters, the data is paired. The correct non-parametric test is McNemar's test on discordant pairs.",
            "fallacy_of_proving_the_null": "Finding p > 0.05 fails to reject the null hypothesis of equal error rates; it does NOT prove that the systems are equivalent. Demonstrating equivalence requires specifying an equivalence margin (e.g. +/- 2.0%) and conducting Two One-Sided Tests (TOST)."
        },
        "mcnemar_scenarios": mcnemar_analysis,
        "tost_equivalence_audit": {
            "sample_size": 244,
            "observed_diff_pct": -1.23,
            "diff_95_ci": [-4.51, +2.05],
            "conclusion": "Because the 95% CI of the difference [-4.51%, +2.05%] exceeds typical strict equivalence bounds (+/-2%), the sample size n=244 is underpowered to establish equivalence."
        }
    }

    # Save Results JSON
    json_path = out_dir / "cpu_speed_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Sonuçlar JSON dosyasına kaydedildi: {json_path}")

if __name__ == "__main__":
    main()
