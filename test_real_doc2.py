"""Evaluate V2, V1, and Tesseract on the user's second uploaded real phone photo.
Image: media_1793194879421.jpg
"""
import os
import sys
import json
from pathlib import Path
import cv2
from PIL import Image
import pytesseract

from common import align, infer_single_crop, load_model_checkpoint, normalize

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

IMG_PATH = r"C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793194879421.jpg"
OUT_DIR = Path("results/real_test2")
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

doc_img = cv2.imread(IMG_PATH)
h_img, w_img = doc_img.shape[:2]

# 2. Define ground-truth lines and words from the user's second photo
# Coordinates are (label, y0, y1, x0, x1)
REAL_LINES = [
    ("TƏQDİMAT", 23, 48, 365, 485),
    ("Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi", 149, 173, 2, 495),
    ("hazırlanması.", 194, 214, 50, 165),
    ("Əsas səhifənin daha informativ və funksional qurulması.", 211, 233, 50, 485),
    ("çıxarılması.", 255, 275, 53, 150),
    ("qurulması.", 294, 314, 55, 145),
    ("axtarış sisteminin tətbiq edilməsi.", 352, 374, 58, 315),
    ("Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi", 387, 411, 2, 455),
    ("qurulması.", 431, 452, 60, 150),
    ("elektron mühitdə idarə olunması.", 487, 507, 62, 315),
    ("imkanlarının yaradılması.", 545, 566, 64, 260),
    ("saytda göstərilməsi.", 603, 624, 65, 215),
    ("Müəlliflər üçün elektron xidmətlərin genişləndirilməsi", 638, 659, 15, 445),
    ("Onlayn məqalə təqdimetmə imkanlarının təkmilləşdirilməsi.", 658, 680, 65, 505),
    ("Nəşr və xidmət haqları barədə məlumatların ayrıca təqdim edilməsi.", 695, 718, 68, 580),
    ("Məqalə qəbulu müddətləri və redaksiya elanlarının operativ yayımlanması.", 714, 736, 68, 625),
    ("Xarici müəlliflər üçün ingilis dilində məlumatların əlçatanlığının artırılması.", 732, 754, 68, 680),
    ("ingilis, rus) daxil edilməsi və bazada ayrıca saxlanılması.", 770, 792, 70, 495),
    ("qərarının yüklənməsi imkanının yaradılması.", 826, 848, 70, 405),
    ("tərəfindən verilməsi imkanının yaradılması.", 863, 885, 70, 405),
    ("Naxçıvan Dövlət Universiteti", 913, 933, 555, 730),
]

REAL_WORDS = [
    ("TƏQDİMAT", 23, 48, 365, 485),
    ("Veb-saytın", 149, 173, 2, 85),
    ("dizaynının", 149, 173, 90, 182),
    ("istifadəçi", 149, 173, 210, 305),
    ("interfeysinin", 149, 173, 310, 418),
    ("yenilənməsi", 149, 173, 420, 490),
    ("akademik", 170, 192, 208, 298),
    ("standartlarına", 170, 192, 305, 435),
    ("informativ", 211, 233, 235, 328),
    ("funksional", 211, 233, 355, 448),
    ("Elektron", 387, 411, 2, 78),
    ("jurnal", 387, 411, 80, 138),
    ("idarəetmə", 387, 411, 142, 240),
    ("sisteminin", 387, 411, 245, 335),
    ("təkmilləşdirilməsi", 387, 411, 340, 452),
    ("platformasının", 408, 430, 235, 365),
    ("Məqalələrin", 446, 468, 65, 162),
    ("redaksiya", 446, 468, 275, 358),
    ("redaktor", 446, 468, 415, 488),
    ("müəllif", 446, 468, 510, 568),
    ("resenzent", 505, 526, 225, 315),
    ("yaradılması.", 545, 566, 168, 256),
    ("Onlayn", 658, 680, 68, 142),
    ("elanlarının", 714, 736, 375, 475),
    ("operativ", 714, 736, 480, 555),
    ("ingilis", 732, 754, 280, 338),
    ("əlçatanlığının", 732, 754, 480, 600),
    ("saxlanılması.", 770, 792, 388, 490),
    ("Universiteti", 913, 933, 658, 730),
    ("Naxçıvan", 913, 933, 555, 618),
]

def eval_set(items, is_word=False):
    prefix = "word" if is_word else "line"
    results = []
    
    total_chars = 0
    total_words = 0
    v1_char_err = 0
    v2_char_err = 0
    tes_char_err = 0
    
    v1_word_err = 0
    v2_word_err = 0
    tes_word_err = 0

    m_v1 = v1_word_m if is_word else v1_line_m
    cfg_v1 = v1_word_cfg if is_word else v1_line_cfg
    voc_v1 = v1_word_voc if is_word else v1_line_voc

    m_v2 = v2_word_m if is_word else v2_line_m
    cfg_v2 = v2_word_cfg if is_word else v2_line_cfg
    voc_v2 = v2_word_voc if is_word else v2_line_voc

    for idx, (gt, y0, y1, x0, x1) in enumerate(items):
        crop = doc_img[max(0, y0-2):min(h_img, y1+2), max(0, x0-3):min(w_img, x1+3)]
        crop_fname = f"{prefix}_{idx:02d}.png"
        cv2.imwrite(str(CROPS_DIR / crop_fname), crop)
        
        pred_v1 = infer(m_v1, cfg_v1, voc_v1, crop, dev)
        pred_v2 = infer(m_v2, cfg_v2, voc_v2, crop, dev)
        pred_tes = run_tess_crop(crop, is_word=is_word)
        
        gt_n = normalize(gt)
        p1_n = normalize(pred_v1)
        p2_n = normalize(pred_v2)
        pt_n = normalize(pred_tes)
        
        c_err1 = align(gt_n, p1_n)[0]
        c_err2 = align(gt_n, p2_n)[0]
        c_err_t = align(gt_n, pt_n)[0]
        n_c = max(len(gt_n), 1)
        
        w_gt = gt_n.split()
        w_p1 = p1_n.split()
        w_p2 = p2_n.split()
        w_pt = pt_n.split()
        n_w = max(len(w_gt), 1)
        
        w_err1 = 0 if w_p1 == w_gt else 1
        w_err2 = 0 if w_p2 == w_gt else 1
        w_err_t = 0 if w_pt == w_gt else 1
        
        total_chars += n_c
        total_words += n_w
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
        "total_chars": total_chars,
        "total_words": total_words,
        "v1_cer": (v1_char_err / total_chars * 100) if total_chars else 0,
        "v2_cer": (v2_char_err / total_chars * 100) if total_chars else 0,
        "tes_cer": (tes_char_err / total_chars * 100) if total_chars else 0,
        "v1_wer": (v1_word_err / total_words * 100) if total_words else 0,
        "v2_wer": (v2_word_err / total_words * 100) if total_words else 0,
        "tes_wer": (tes_word_err / total_words * 100) if total_words else 0,
        "details": results
    }
    return summary

print("Evaluating 21 lines...")
lines_summary = eval_set(REAL_LINES, is_word=False)
print(f"Line CER -> V1: {lines_summary['v1_cer']:.2f}%, V2: {lines_summary['v2_cer']:.2f}%, Tesseract: {lines_summary['tes_cer']:.2f}%")

print("Evaluating 30 words...")
words_summary = eval_set(REAL_WORDS, is_word=True)
print(f"Word CER -> V1: {words_summary['v1_cer']:.2f}%, V2: {words_summary['v2_cer']:.2f}%, Tesseract: {words_summary['tes_cer']:.2f}%")

# Save json
with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump({"lines": lines_summary, "words": words_summary}, f, ensure_ascii=False, indent=2)

# Generate Markdown report
md = []
md.append("# İkinci Real Sənəd Fotosu Üzərində Sınaq Nəticələri (Photo 2)")
md.append(f"**Giriş şəkli:** `media_1793194879421.jpg` ({w_img}x{h_img})\n")
md.append("Bu foto birinci fotoya nisbətən daha düz bucaqdan və daha aydın işıqlandırma altında çəkilmişdir.\n")
md.append("## 1. Ümumi Nəticələr Xülasəsi\n")
md.append("| Kateqoriya | Model | CER (%) | WER (%) | Düzgün Tanınan / Cəmi |")
md.append("|---|---|---|---|---|")
md.append(f"| **Sətirlər (21 sətir)** | **V1 Line** | {lines_summary['v1_cer']:.2f}% | {lines_summary['v1_wer']:.2f}% | {sum(1 for d in lines_summary['details'] if d['v1'] == d['gt'])} / {lines_summary['n_items']} |")
md.append(f"| | **V2 Line** | **{lines_summary['v2_cer']:.2f}%** | **{lines_summary['v2_wer']:.2f}%** | **{sum(1 for d in lines_summary['details'] if d['v2'] == d['gt'])} / {lines_summary['n_items']}** |")
md.append(f"| | **Tesseract (aze)** | {lines_summary['tes_cer']:.2f}% | {lines_summary['tes_wer']:.2f}% | {sum(1 for d in lines_summary['details'] if d['tes'] == d['gt'])} / {lines_summary['n_items']} |")
md.append("|---|---|---|---|---|")
md.append(f"| **Sözlər (30 söz)** | **V1 Word** | {words_summary['v1_cer']:.2f}% | {words_summary['v1_wer']:.2f}% | {sum(1 for d in words_summary['details'] if d['v1'] == d['gt'])} / {words_summary['n_items']} |")
md.append(f"| | **V2 Word** | **{words_summary['v2_cer']:.2f}%** | **{words_summary['v2_wer']:.2f}%** | **{sum(1 for d in words_summary['details'] if d['v2'] == d['gt'])} / {words_summary['n_items']}** |")
md.append(f"| | **Tesseract (aze)** | {words_summary['tes_cer']:.2f}% | {words_summary['tes_wer']:.2f}% | {sum(1 for d in words_summary['details'] if d['tes'] == d['gt'])} / {words_summary['n_items']} |")

md.append("\n## 2. Sətir Nümunələri (Müqayisə Cədvəli)\n")
md.append("| № | Orijinal Mətn (GT) | V1 Line | V2 Line | Tesseract (aze) |")
md.append("|---|---|---|---|---|")
for d in lines_summary["details"]:
    md.append(f"| {d['idx']+1} | `{d['gt']}` | `{d['v1']}` | **`{d['v2']}`** | `{d['tes']}` |")

md.append("\n## 3. Söz Nümunələri (Müqayisə Cədvəli)\n")
md.append("| № | Orijinal Söz (GT) | V1 Word | V2 Word | Tesseract (aze) |")
md.append("|---|---|---|---|---|")
for d in words_summary["details"]:
    md.append(f"| {d['idx']+1} | `{d['gt']}` | `{d['v1']}` | **`{d['v2']}`** | `{d['tes']}` |")

with open(OUT_DIR / "evaluation.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"Report saved to {OUT_DIR / 'evaluation.md'}")
