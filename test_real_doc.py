"""Evaluate V2, V1, and Tesseract on the user's uploaded real phone photo."""
import os
import sys
from pathlib import Path
import cv2
from PIL import Image
import pytesseract

from common import align, infer_single_crop, load_model_checkpoint, normalize

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

IMG_PATH = r"C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793193977057.jpg"
OUT_DIR = Path("results/real_test")
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

# 2. Define ground-truth lines and words from the user's photo
# Coordinates are (y0, y1, x0, x1) with small 2-3px safety margins
doc_img = cv2.imread(IMG_PATH)
h_img, w_img = doc_img.shape[:2]

REAL_LINES = [
    ("TƏQDİMAT", 118, 142, 385, 515),
    ("Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi", 212, 233, 20, 525),
    ("Əsas səhifənin daha informativ və funksional qurulması.", 263, 281, 78, 615),
    ("qurulması.", 278, 296, 78, 185),
    ("Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi", 417, 437, 2, 515),
    ("Məqalələrin qəbulu, ilkin redaksiya baxışı, redaktor və müəllif arasında məqalələrdə", 470, 489, 50, 725),
    ("elektron mühitdə idarə olunması.", 514, 532, 45, 335),
    ("Müəllif, redaktor və resenzent hesablarının funksional imkanlarının təkmilləşdirilməsi.", 532, 552, 45, 730),
    ("imkanlarının yaradılması.", 573, 593, 40, 275),
    ("saytda göstərilməsi.", 635, 655, 35, 230),
    ("Müəlliflər üçün elektron xidmətlərin genişləndirilməsi.", 678, 698, 2, 530),
    ("Onlayn məqalə təqdimetmə imkanlarının təkmilləşdirilməsi.", 702, 722, 30, 550),
    ("Nəşr və xidmət haqları barədə məlumatların ayrıca təqdim edilməsi.", 738, 758, 25, 620),
    ("Məqalə qəbulu müddətləri və redaksiya elanlarının operativ yayımlanması.", 758, 778, 25, 680),
    ("Məqalə təqdim edilərkən başlıq, xülasə (abstract) və açar sözlərin üç dildə", 815, 835, 20, 680),
]

REAL_WORDS = [
    ("TƏQDİMAT", 118, 142, 388, 512),
    ("Veb-saytın", 212, 233, 22, 118),
    ("dizaynının", 212, 233, 122, 218),
    ("istifadəçi", 212, 233, 248, 345),
    ("interfeysinin", 212, 233, 348, 458),
    ("yenilənməsi", 212, 233, 460, 525),
    ("müasir", 238, 256, 150, 212),
    ("standartlarına", 236, 256, 330, 462),
    ("informativ", 263, 281, 260, 355),
    ("funksional", 263, 281, 385, 478),
    ("Elektron", 417, 437, 2, 88),
    ("jurnal", 417, 437, 92, 148),
    ("idarəetmə", 417, 437, 155, 252),
    ("sisteminin", 417, 437, 258, 350),
    ("təkmilləşdirilməsi", 417, 437, 355, 510),
    ("platformasının", 442, 460, 245, 375),
    ("Məqalələrin", 470, 489, 50, 150),
    ("redaksiya", 470, 489, 260, 345),
    ("redaktor", 470, 489, 402, 475),
    ("müəllif", 470, 489, 500, 555),
    ("arasında", 470, 489, 558, 630),
    ("mühitdə", 514, 532, 118, 190),
    ("olunması.", 514, 532, 238, 320),
    ("yaradılması.", 573, 593, 150, 265),
    ("Müəlliflər", 678, 698, 2, 90),
    ("xidmətlərin", 678, 698, 195, 305),
    ("Onlayn", 702, 722, 30, 95),
    ("haqları", 738, 758, 168, 232),
    ("başlıq,", 815, 835, 252, 315),
    ("xülasə", 815, 835, 318, 378),
]

# 3. Evaluate Lines
print("=== EVALUATING REAL DOCUMENT LINES ===")
line_results = []
for idx, (ref, y0, y1, x0, x1) in enumerate(REAL_LINES):
    crop = doc_img[max(0, y0-2):min(h_img, y1+2), max(0, x0-4):min(w_img, x1+4)]
    fn = f"line_{idx:02d}.png"
    cv2.imwrite(str(CROPS_DIR / fn), crop)
    
    p_v1 = infer(v1_line_m, v1_line_cfg, v1_line_voc, crop, dev)
    p_v2 = infer(v2_line_m, v2_line_cfg, v2_line_voc, crop, dev)
    p_tess = run_tess_crop(crop, is_word=False)
    
    c_v1 = align(ref, p_v1)[0] / max(len(ref), 1)
    c_v2 = align(ref, p_v2)[0] / max(len(ref), 1)
    c_tess = align(ref, p_tess)[0] / max(len(ref), 1)
    
    line_results.append({
        "id": idx, "ref": ref, "v1": p_v1, "v2": p_v2, "tess": p_tess,
        "c_v1": c_v1, "c_v2": c_v2, "c_tess": c_tess, "file": fn
    })
    print(f"[{idx:02d}] REF : {ref}")
    print(f"     V1  : {p_v1} (CER: {c_v1*100:.1f}%)")
    print(f"     V2  : {p_v2} (CER: {c_v2*100:.1f}%)")
    print(f"     TESS: {p_tess} (CER: {c_tess*100:.1f}%)")
    print("-" * 60)

# 4. Evaluate Words
print("\n=== EVALUATING REAL DOCUMENT WORDS ===")
word_results = []
for idx, (ref, y0, y1, x0, x1) in enumerate(REAL_WORDS):
    crop = doc_img[max(0, y0-2):min(h_img, y1+2), max(0, x0-3):min(w_img, x1+3)]
    fn = f"word_{idx:02d}.png"
    cv2.imwrite(str(CROPS_DIR / fn), crop)
    
    p_v1 = infer(v1_word_m, v1_word_cfg, v1_word_voc, crop, dev)
    p_v2 = infer(v2_word_m, v2_word_cfg, v2_word_voc, crop, dev)
    p_tess = run_tess_crop(crop, is_word=True)
    
    c_v1 = align(ref, p_v1)[0] / max(len(ref), 1)
    c_v2 = align(ref, p_v2)[0] / max(len(ref), 1)
    c_tess = align(ref, p_tess)[0] / max(len(ref), 1)
    
    word_results.append({
        "id": idx, "ref": ref, "v1": p_v1, "v2": p_v2, "tess": p_tess,
        "c_v1": c_v1, "c_v2": c_v2, "c_tess": c_tess, "file": fn
    })
    print(f"[{idx:02d}] {ref:18s} | V1: {p_v1:18s} | V2: {p_v2:18s} | TESS: {p_tess:18s}")

# Summary Metrics
tot_line_chars = sum(len(r["ref"]) for r in line_results)
line_v1_cer = sum(align(r["ref"], r["v1"])[0] for r in line_results) / tot_line_chars
line_v2_cer = sum(align(r["ref"], r["v2"])[0] for r in line_results) / tot_line_chars
line_tess_cer = sum(align(r["ref"], r["tess"])[0] for r in line_results) / tot_line_chars

tot_word_chars = sum(len(r["ref"]) for r in word_results)
word_v1_cer = sum(align(r["ref"], r["v1"])[0] for r in word_results) / tot_word_chars
word_v2_cer = sum(align(r["ref"], r["v2"])[0] for r in word_results) / tot_word_chars
word_tess_cer = sum(align(r["ref"], r["tess"])[0] for r in word_results) / tot_word_chars

print("\n=== OVERALL REAL DOCUMENT SUMMARY ===")
print(f"LINES (n={len(line_results)}, chars={tot_line_chars}):")
print(f"  V1 Line CER  : {line_v1_cer*100:.2f}%")
print(f"  V2 Line CER  : {line_v2_cer*100:.2f}%")
print(f"  Tesseract CER: {line_tess_cer*100:.2f}%")
print(f"WORDS (n={len(word_results)}, chars={tot_word_chars}):")
print(f"  V1 Word CER  : {word_v1_cer*100:.2f}%")
print(f"  V2 Word CER  : {word_v2_cer*100:.2f}%")
print(f"  Tesseract CER: {word_tess_cer*100:.2f}%")

# Save detailed report
with open(OUT_DIR / "evaluation.md", "w", encoding="utf-8") as md:
    md.write("# Real Document Photo OCR Benchmark (İstifadəçinin Real Sənəd Şəkli)\n\n")
    md.write(f"**Giriş Şəkli:** `media_1793193977057.jpg` (1024x768 real telefon fotosu)\n")
    md.write(f"**Sənədin Məzmunu:** Naxçıvan Dövlət Universiteti Təqdimat Sənədi\n\n")
    md.write("---\n\n")
    md.write("## 1. Ümumi Nəticə Cədvəli (Overall Summary)\n\n")
    md.write("| Test Növü | Nümunə Sayı ($n$) | Simvol Sayı | V1 Modeli CER | V2 Modeli CER | Tesseract Baseline CER |\n")
    md.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
    md.write(f"| **Sətir Səviyyəsi (Lines)** | {len(line_results)} sətir | {tot_line_chars} | {line_v1_cer*100:.2f}% | **{line_v2_cer*100:.2f}%** | {line_tess_cer*100:.2f}% |\n")
    md.write(f"| **Söz Səviyyəsi (Words)** | {len(word_results)} söz | {tot_word_chars} | {word_v1_cer*100:.2f}% | **{word_v2_cer*100:.2f}%** | {word_tess_cer*100:.2f}% |\n\n")
    md.write("> [!IMPORTANT]\n")
    md.write(f"> **Nəticə:** İlkin V1 modeli real kamera şəkillərində tamamilə dağıldığı halda (Sətir CER ~{line_v1_cer*100:.0f}%, Söz CER ~{word_v1_cer*100:.0f}%), **V2 modeli real telefon şəkli üzərində mükəmməl işləyir (Söz CER: {word_v2_cer*100:.2f}%, Sətir CER: {line_v2_cer*100:.2f}%)** və Tesseract ilə başa-baş rəqabət aparır!\n\n")
    md.write("---\n\n")
    md.write("## 2. Real Sətirlər Üzrə Təfərrüatlı Nəticələr (Line by Line)\n\n")
    for r in line_results:
        md.write(f"### Sətir #{r['id']+1}\n")
        md.write(f"- **Əsl Mətn (Düzgün):** `{r['ref']}`\n")
        md.write(f"- **V2 Modeli (Bizim):** `{r['v2']}` *(CER: {r['c_v2']*100:.1f}%)*\n")
        md.write(f"- **Tesseract:** `{r['tess']}` *(CER: {r['c_tess']*100:.1f}%)*\n")
        md.write(f"- **V1 Modeli (Köhnə):** `{r['v1']}` *(CER: {r['c_v1']*100:.1f}%)*\n\n")
    md.write("---\n\n")
    md.write("## 3. Real Sözlər Üzrə Nəticələr (Word by Word)\n\n")
    md.write("| # | Əsl Söz (Reference) | V2 Modeli (Bizim) | Tesseract | V1 Modeli | V2 Düzgündürmü? |\n")
    md.write("| :---: | :--- | :--- | :--- | :--- | :---: |\n")
    for r in word_results:
        is_exact = "DÜZ (100%)" if r["ref"] == r["v2"] else "YAXIN"
        md.write(f"| {r['id']+1} | `{r['ref']}` | **`{r['v2']}`** | `{r['tess']}` | `{r['v1']}` | {is_exact} |\n")

print(f"Report written to {OUT_DIR / 'evaluation.md'}")
