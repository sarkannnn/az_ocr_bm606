"""Evaluate V2, V1, and Tesseract on the user's third uploaded real phone photo.
Image: IMG_5121.HEIC (converted to photo3.jpg)
Document: Azerbaijani story 'Usta Əli və Xalçaçılıq Sənəti'
"""
import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd

from common import align, infer_single_crop, load_model_checkpoint, normalize

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

IMG_PATH = Path("data/real_test3/photo3.jpg")
OUT_DIR = Path("results/real_test3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = OUT_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load models
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_word_m, v1_word_cfg, v1_word_voc, dev = load_model_checkpoint("runs/word/best.pt")
v2_word_m, v2_word_cfg, v2_word_voc, dev = load_model_checkpoint("runs/word_v2_20261027_180300/best.pt")

infer = infer_single_crop


def run_tess_crop(img, is_word=False):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    psm = 8 if is_word else 7
    txt = pytesseract.image_to_string(pil_img, lang="aze", config=f"--psm {psm}")
    return normalize(" ".join(txt.split()))


doc_img = cv2.imread(str(IMG_PATH))
h_img, w_img = doc_img.shape[:2]

# Ground truth lines from the document photo
REAL_LINES = [
    ("Çöldə isə nəvəsi Murad səhərin sərin havasında babasının köhnə velosipedini təmizləyirdi", 1110, 1205, 260, 2395),
    ("Murad şəhərdə oxusa da, hər tətil otağında kəndə, bu torpağın qoxusuna qayıdırdı. Amma bu", 1189, 1279, 260, 2455),
    ("dəfə qayıdışı başqa idi. O, hiss edirdi ki, babası təkdir və onun daşıdığı bu ağır yük —", 1266, 1356, 258, 2275),
    ("unudulmaqda olan xalçaçılıq sənəti hər keçən gün daha da zəifləyir. Murad əlindəki dəsmalla", 1346, 1434, 255, 2475),
    ("velosipedin sükanını silə-silə başını qaldırıb emalatxananın açıq qapısından içəri baxdı.", 1425, 1509, 252, 2335),
    ("Babasının əyilmiş belini, dəzgah arxasında təkbaşına necə mücadilə etdiyini görəndə qəlbində", 1505, 1585, 250, 2498),
    ("qəribə bir sızıltı oldu. O, ayaqqabılarını çıxarıb içəri daxil oldu, ehtiyatla babasının çiyinlərinə", 1585, 1662, 248, 2505),
    ("— Baba, səhər tezdən yenə başladın? Bir az dincəlsənə, — deyə Murad incə, lakin qayğılı səslə", 1788, 1854, 240, 2490),
    ("Usta Əli təəccüblə geri döndü, üzündəki qırışlar yumşaldı və dərin, mənalı gözləri ilə nəvəsinə", 1990, 2049, 235, 2535),
    ("baxdı. O, təbəssüm edərək dedi: — Ay bala, usta o qoca olar ki, əlləri işləməyəndə ruhu da", 2070, 2127, 232, 2415),
    ("dincəlir. Bu xalça sadəcə ipdən toxunmur. Görürsən bu naxışları? Hər ilmə bir zamandır, bir", 2151, 2208, 230, 2478),
    ("tariximizdir. Biz getsək də, bu naxışlar qalacaq. Amma naxışların dili olmasa, heç kim bizim kim", 2231, 2291, 228, 2605),
    ("Murad babasının dediklərinin mənasını tam dərk etməsə də, onun səsindəki həsrəti və qorxunu", 2434, 2495, 220, 2558),
    ("hiss edirdi. Babası qorxurdu ki, o öləndən sonra bu qədim sənət də onunla birlikdə torpağa", 2517, 2583, 218, 2465),
    ("basdırılacaq. Həmin an Murad daxilində qəti bir qərar verdi: bu xalçanın son ilməsinə qədər", 2600, 2671, 215, 2500),
    ("babasına kömək edəcək və bu sənətin yerə düşməsinə imkan verməyəcək.", 2684, 2746, 215, 2055),
]

# Extract 20 precise words with Azerbaijani characters directly via image_to_data
data = pytesseract.image_to_data(Image.open(IMG_PATH), lang="aze", output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != "")]

# Filter high-confidence words that contain Azerbaijani letters
target_words = []
seen = set()
for _, r in data.iterrows():
    w = str(r.text).strip()
    if any(c in w for c in "əıöğüşçƏİÖĞÜŞÇ") and len(w) >= 4 and r.width >= 35 and r.height >= 20 and r.conf > 70:
        if w not in seen:
            seen.add(w)
            target_words.append((w, int(r.top), int(r.top + r.height), int(r.left), int(r.left + r.width)))
    if len(target_words) >= 20:
        break

REAL_WORDS = target_words


def eval_set(items, is_word=False):
    prefix = "word" if is_word else "line"
    m_v1 = v1_word_m if is_word else v1_line_m
    cfg_v1 = v1_word_cfg if is_word else v1_line_cfg
    voc_v1 = v1_word_voc if is_word else v1_line_voc

    m_v2 = v2_word_m if is_word else v2_line_m
    cfg_v2 = v2_word_cfg if is_word else v2_line_cfg
    voc_v2 = v2_word_voc if is_word else v2_line_voc

    results = []
    tot_chars = 0
    tot_words = 0
    v1_char_err, v2_char_err, tes_char_err = 0, 0, 0
    v1_word_err, v2_word_err, tes_word_err = 0, 0, 0

    for idx, (gt, y0, y1, x0, x1) in enumerate(items):
        pad_y = 2 if is_word else 3
        pad_x = 2 if is_word else 5
        crop = doc_img[max(0, y0-pad_y):min(h_img, y1+pad_y), max(0, x0-pad_x):min(w_img, x1+pad_x)]
        crop_fname = f"{prefix}_{idx:02d}.png"
        cv2.imwrite(str(CROPS_DIR / crop_fname), crop)

        p_v1 = infer(m_v1, cfg_v1, voc_v1, crop, dev)
        p_v2 = infer(m_v2, cfg_v2, voc_v2, crop, dev)
        p_tes = run_tess_crop(crop, is_word=is_word)

        gt_n = normalize(gt.strip())
        p1_n = normalize(p_v1.strip())
        p2_n = normalize(p_v2.strip())
        pt_n = normalize(p_tes.strip())

        c_err1, _ = align(gt_n, p1_n)
        c_err2, _ = align(gt_n, p2_n)
        c_err_t, _ = align(gt_n, pt_n)

        w_err1, _ = align(gt_n.split(), p1_n.split())
        w_err2, _ = align(gt_n.split(), p2_n.split())
        w_err_t, _ = align(gt_n.split(), pt_n.split())

        n_c = len(gt_n)
        n_w = len(gt_n.split())
        tot_chars += n_c
        tot_words += n_w

        v1_char_err += c_err1
        v2_char_err += c_err2
        tes_char_err += c_err_t

        v1_word_err += w_err1
        v2_word_err += w_err2
        tes_word_err += w_err_t

        results.append({
            "idx": idx,
            "crop": crop_fname,
            "gt": gt_n,
            "v1": p1_n,
            "v2": p2_n,
            "tes": pt_n,
            "v1_cer": (c_err1 / n_c),
            "v2_cer": (c_err2 / n_c),
            "tes_cer": (c_err_t / n_c),
        })

    summary = {
        "n_items": len(items),
        "total_chars": tot_chars,
        "total_words": tot_words,
        "v1_cer": (v1_char_err / tot_chars * 100) if tot_chars else 0,
        "v2_cer": (v2_char_err / tot_chars * 100) if tot_chars else 0,
        "tes_cer": (tes_char_err / tot_chars * 100) if tot_chars else 0,
        "v1_wer": (v1_word_err / tot_words * 100) if tot_words else 0,
        "v2_wer": (v2_word_err / tot_words * 100) if tot_words else 0,
        "tes_wer": (tes_word_err / tot_words * 100) if tot_words else 0,
        "details": results,
    }
    return summary


print("Evaluating 16 lines on Photo 3 (IMG_5121)...")
lines_summary = eval_set(REAL_LINES, is_word=False)
print(f"Line CER -> V1: {lines_summary['v1_cer']:.2f}%, V2: {lines_summary['v2_cer']:.2f}%, Tesseract: {lines_summary['tes_cer']:.2f}%")

print(f"Evaluating {len(REAL_WORDS)} words on Photo 3 (IMG_5121)...")
words_summary = eval_set(REAL_WORDS, is_word=True)
print(f"Word CER -> V1: {words_summary['v1_cer']:.2f}%, V2: {words_summary['v2_cer']:.2f}%, Tesseract: {words_summary['tes_cer']:.2f}%")

# Save json
with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump({"lines": lines_summary, "words": words_summary}, f, ensure_ascii=False, indent=2)

# Generate Markdown report
md = []
md.append("# Üçüncü Real Sənəd Fotosu Üzərində Sınaq Nəticələri (Photo 3 — IMG_5121)")
md.append(f"**Giriş şəkli:** `IMG_5121.HEIC` (3024x4032 yüksək keyfiyyətli kamera fotosu)\n")
md.append("Bu foto Azərbaycan dilində çap olunmuş klassik bədii nəsr ('Usta Əli və Xalçaçılıq Sənəti') səhifəsini əks etdirir və xüsusi diakritik simvollarla (`ə, ı, ö, ğ, ü, ş, ç`) son dərəcə zəngindir.\n")
md.append("## 1. Ümumi Nəticələr Xülasəsi\n")
md.append("| Kateqoriya | Model | CER (%) | WER (%) | Düzgün Tanınan / Cəmi |")
md.append("|---|---|---|---|---|")
md.append(f"| **Sətirlər (16 sətir)** | **V1 Line** | {lines_summary['v1_cer']:.2f}% | {lines_summary['v1_wer']:.2f}% | {sum(1 for d in lines_summary['details'] if d['v1'] == d['gt'])} / {lines_summary['n_items']} |")
md.append(f"| | **V2 Line** | **{lines_summary['v2_cer']:.2f}%** | **{lines_summary['v2_wer']:.2f}%** | **{sum(1 for d in lines_summary['details'] if d['v2'] == d['gt'])} / {lines_summary['n_items']}** |")
md.append(f"| | **Tesseract (aze)** | {lines_summary['tes_cer']:.2f}% | {lines_summary['tes_wer']:.2f}% | {sum(1 for d in lines_summary['details'] if d['tes'] == d['gt'])} / {lines_summary['n_items']} |")
md.append("|---|---|---|---|---|")
md.append(f"| **Sözlər ({words_summary['n_items']} söz)** | **V1 Word** | {words_summary['v1_cer']:.2f}% | {words_summary['v1_wer']:.2f}% | {sum(1 for d in words_summary['details'] if d['v1'] == d['gt'])} / {words_summary['n_items']} |")
md.append(f"| | **V2 Word** | **{words_summary['v2_cer']:.2f}%** | **{words_summary['v2_wer']:.2f}%** | **{sum(1 for d in words_summary['details'] if d['v2'] == d['gt'])} / {words_summary['n_items']}** |")
md.append(f"| | **Tesseract (aze)** | {words_summary['tes_cer']:.2f}% | {words_summary['tes_wer']:.2f}% | {sum(1 for d in words_summary['details'] if d['tes'] == d['gt'])} / {words_summary['n_items']} |")

md.append("\n## 2. Diakritik Söz Nümunələri (Müqayisə Cədvəli)\n")
md.append("| № | Orijinal Söz (GT) | V2 Word (Bizim) | Tesseract (aze) | V1 Word | V2 Dəqiqliyi |")
md.append("|---|---|---|---|---|---|")
for d in words_summary["details"]:
    status = "DÜZ (100%)" if d['gt'] == d['v2'] else f"Yaxın ({d['v2_cer']*100:.1f}%)"
    md.append(f"| {d['idx']+1} | `{d['gt']}` | **`{d['v2']}`** | `{d['tes']}` | `{d['v1']}` | {status} |")

with open(OUT_DIR / "evaluation.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"Report saved to {OUT_DIR / 'evaluation.md'}")
