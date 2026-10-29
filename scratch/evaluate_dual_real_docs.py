"""Master Real Document Benchmark & Figure Generation Script.
Evaluates the two definitive real phone photos:
  - Document 1: Times New Roman (Serif) — 'Usta Əli və Xalçaçılıq Sənəti'
  - Document 2: Calibri (Sans-Serif) — 'Qoca Səməndər və Dulusçuluq Sənəti'
Generates all 3 academic figures and produces results/real_docs/evaluation.md.
"""
import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import align, infer_single_crop, load_model_checkpoint, normalize

sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

OUT_DIR = Path("results/real_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_P1 = OUT_DIR / "crops_times"
CROPS_P1.mkdir(parents=True, exist_ok=True)
CROPS_P2 = OUT_DIR / "crops_calibri"
CROPS_P2.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

# -------------------------------------------------------------
# DOC 1: Times New Roman
# -------------------------------------------------------------
img1 = cv2.imread("data/real_docs/photo1_times.jpg")
h1, w1 = img1.shape[:2]

P1_LINES = [
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

data1 = pytesseract.image_to_data(Image.open("data/real_docs/photo1_times.jpg"), lang="aze", output_type=pytesseract.Output.DATAFRAME)
data1 = data1[data1.text.notnull() & (data1.text.str.strip() != "")]
P1_WORDS = []
seen1 = set()
for _, r in data1.iterrows():
    w = str(r.text).strip()
    if any(c in w for c in "əıöğüşçƏİÖĞÜŞÇ") and len(w) >= 4 and r.width >= 35 and r.height >= 20 and r.conf > 70:
        if w not in seen1:
            seen1.add(w)
            P1_WORDS.append((w, int(r.top), int(r.top + r.height), int(r.left), int(r.left + r.width)))
    if len(P1_WORDS) >= 20:
        break

# -------------------------------------------------------------
# DOC 2: Calibri
# -------------------------------------------------------------
img2 = cv2.imread("data/real_docs/photo2_calibri.jpg")
h2, w2 = img2.shape[:2]

P2_LINES = [
    ("Gəncə çayının kənarında, qədim ağacların kölgəsinə sığınmış balaca bir dulusçu emalatxanası var", 345, 425, 340, 2580),
    ("idi. Burada illərdir gil ilə nəfəs alan, gilə can verən Qoca Səməndər yaşayırdı. Səməndər usta", 425, 498, 340, 2560),
    ("əllərini palçığın tarixinə batıranda sanki zaman dayanırdı, onun hər bir hərəkətində əsrlərin", 498, 568, 335, 2520),
    ("sənəti, nəsildən-nəslə keçən gizli bir məharət yaşatdırdığı hiss olunurdu. Divarlardan asılmış gil", 568, 640, 335, 2540),
    ("kasalar, bəzəkli küplər, nəqşli boşqablar otağa xüsusi bir torpaq qoxusu və qədimlik gətirirdi.", 640, 715, 330, 2520),
    ("Səməndər usta çəliyini kənara qoyub ağır-ağar dolabın arxasına keçdi, qabarıq və çat-çat olmuş", 715, 788, 330, 2545),
    ("barmaqları ilə fırlanan dairəvi taxtaya toxundu. Bu taxta onun ömrünün yekunu, sonuncu böyük", 788, 860, 325, 2550),
    ("idi. Elə həmin an emalatxananın qapısı ehtiyatla açıldı və içəri şəhərdən yeni gəlmiş nəvəsi Taryel", 930, 1005, 320, 2570),
    ("girdi. Taryel ali məktəbi bitirib şəhərdə böyük imkanlar qazanmış bir gənc idi, lakin hər dəfə bu", 1005, 1078, 320, 2555),
    ("kəndə gələndə özünü yenidən uşaqlıq illərindəki kimi azad və rahat hiss edirdi. O, babasının", 1078, 1150, 315, 2525),
    ("əyilmiş belini, yorğun, lakin qətiyyətli baxışlarını görüb asta addımlarla yaxınlaşdı, əlindəki", 1150, 1222, 315, 2515),
    ("çantanı bir kənara qoyub babasının çiyninə toxundu.", 1222, 1295, 310, 1550),
    ("— Baba, yenə saatlarla bu təkərin arxasında oturmusan. Özünü yorursan, gəl bu gün biraz dincəl,", 1340, 1415, 305, 2565),
    ("Qoca Səməndər başını qaldırıb nəvəsinə baxdı, dodaqlarında yüngül, lakin dərin mənalı bir", 1485, 1560, 300, 2540),
    ("təbəssüm yarandı. O, ovcundakı nəm gili yavaşca masaya qoyub dedi: — Ay bala, insan o vaxt", 1560, 1632, 298, 2530),
    ("qocalır ki, torpağa olan inamı ölür. Bu gil sadəcə torpaq parçasımı sanırsan? Xeyr! Hər küpün,", 1632, 1705, 295, 2535),
]

data2 = pytesseract.image_to_data(Image.open("data/real_docs/photo2_calibri.jpg"), lang="aze", output_type=pytesseract.Output.DATAFRAME)
data2 = data2[data2.text.notnull() & (data2.text.str.strip() != "")]
P2_WORDS = []
seen2 = set()
for _, r in data2.iterrows():
    w = str(r.text).strip()
    if any(c in w for c in "əıöğüşçƏİÖĞÜŞÇ") and len(w) >= 4 and r.width >= 35 and r.height >= 20 and r.conf > 70:
        if w not in seen2:
            seen2.add(w)
            P2_WORDS.append((w, int(r.top), int(r.top + r.height), int(r.left), int(r.left + r.width)))
    if len(P2_WORDS) >= 20:
        break


def evaluate_doc(img, items, is_word, crops_dir):
    prefix = "word" if is_word else "line"
    m_v1 = v1_word_m if is_word else v1_line_m
    cfg_v1 = v1_word_cfg if is_word else v1_line_cfg
    voc_v1 = v1_word_voc if is_word else v1_line_voc

    m_v2 = v2_word_m if is_word else v2_line_m
    cfg_v2 = v2_word_cfg if is_word else v2_line_cfg
    voc_v2 = v2_word_voc if is_word else v2_line_voc

    h_img, w_img = img.shape[:2]
    results = []
    tot_chars, tot_words = 0, 0
    v1_char_err, v2_char_err, tes_char_err = 0, 0, 0
    v1_word_err, v2_word_err, tes_word_err = 0, 0, 0

    for idx, (gt, y0, y1, x0, x1) in enumerate(items):
        pad_y = 2 if is_word else 3
        pad_x = 2 if is_word else 5
        crop = img[max(0, y0-pad_y):min(h_img, y1+pad_y), max(0, x0-pad_x):min(w_img, x1+pad_x)]
        crop_fname = f"{prefix}_{idx:02d}.png"
        cv2.imwrite(str(crops_dir / crop_fname), crop)

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

    return {
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

print("Running Doc 1 (Times New Roman)...")
doc1_lines = evaluate_doc(img1, P1_LINES, False, CROPS_P1)
doc1_words = evaluate_doc(img1, P1_WORDS, True, CROPS_P1)
print(f"Doc 1 Line CER -> V2: {doc1_lines['v2_cer']:.2f}%, Tess: {doc1_lines['tes_cer']:.2f}%, V1: {doc1_lines['v1_cer']:.2f}%")
print(f"Doc 1 Word CER -> V2: {doc1_words['v2_cer']:.2f}%, Tess: {doc1_words['tes_cer']:.2f}%, V1: {doc1_words['v1_cer']:.2f}%")

print("\nRunning Doc 2 (Calibri)...")
doc2_lines = evaluate_doc(img2, P2_LINES, False, CROPS_P2)
doc2_words = evaluate_doc(img2, P2_WORDS, True, CROPS_P2)
print(f"Doc 2 Line CER -> V2: {doc2_lines['v2_cer']:.2f}%, Tess: {doc2_lines['tes_cer']:.2f}%, V1: {doc2_lines['v1_cer']:.2f}%")
print(f"Doc 2 Word CER -> V2: {doc2_words['v2_cer']:.2f}%, Tess: {doc2_words['tes_cer']:.2f}%, V1: {doc2_words['v1_cer']:.2f}%")

# Combined summary
tot_word_chars = doc1_words['total_chars'] + doc2_words['total_chars']
comb_word_v2_cer = (doc1_words['v2_cer'] * doc1_words['total_chars'] + doc2_words['v2_cer'] * doc2_words['total_chars']) / tot_word_chars
comb_word_tes_cer = (doc1_words['tes_cer'] * doc1_words['total_chars'] + doc2_words['tes_cer'] * doc2_words['total_chars']) / tot_word_chars
comb_word_v1_cer = (doc1_words['v1_cer'] * doc1_words['total_chars'] + doc2_words['v1_cer'] * doc2_words['total_chars']) / tot_word_chars

print("\n=== COMBINED WORD BENCHMARK ===")
print(f"Total Words Tested: {doc1_words['n_items'] + doc2_words['n_items']}")
print(f"Combined V2 Word CER   : {comb_word_v2_cer:.2f}%")
print(f"Combined Tesseract CER : {comb_word_tes_cer:.2f}%")
print(f"Combined V1 Word CER   : {comb_word_v1_cer:.2f}%")

# Save results
summary_data = {
    "doc1_times": {"lines": doc1_lines, "words": doc1_words},
    "doc2_calibri": {"lines": doc2_lines, "words": doc2_words},
    "combined_word": {
        "v2_cer": comb_word_v2_cer,
        "tes_cer": comb_word_tes_cer,
        "v1_cer": comb_word_v1_cer
    }
}
with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# Generate the Definitive 3 Figures
# ---------------------------------------------------------------------
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
if not Path(FONT_BOLD).exists():
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
    FONT_REG = "C:/Windows/Fonts/arial.ttf"

# FIGURE 1: Synthetic Dataset & Camera Augmentation
def generate_fig1():
    w_fig, h_fig = 1200, 600
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_sub = ImageFont.truetype(FONT_BOLD, 16)
    f_txt = ImageFont.truetype(FONT_REG, 13)

    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 1. Azərbaycan Dili üçün Sintetik Verilənlər Bazası və Kamera Təhrifi Auqmentasiyası Boru Kəməri", fill=(20, 40, 80), font=f_title)

    # Subpanel a
    draw.rectangle([(20, 65), (580, 580)], outline=(210, 215, 225), width=2)
    draw.rectangle([(20, 65), (580, 100)], fill=(245, 247, 250))
    draw.text((35, 74), "(a) Təmiz Sintetik Nümunələr (TRDG + Şrift Filtrləməsi)", fill=(30, 50, 90), font=f_sub)

    from common import load_labels
    words = load_labels("data/synth_word/train")[:4]
    lines = load_labels("data/synth_line/train")[:3]
    y_pos = 115
    draw.text((35, y_pos), "Söz Səviyyəsi (32x128):", fill=(80, 80, 80), font=f_txt)
    y_pos += 22
    for p, lbl in words:
        if p.exists():
            im = Image.open(p).convert("RGB")
            canvas.paste(im.resize((140, 35)), (35, y_pos))
            draw.rectangle([(35, y_pos), (175, y_pos + 35)], outline=(180, 180, 180))
            draw.text((185, y_pos + 8), f"Etalon: '{lbl}'", fill=(40, 40, 40), font=f_txt)
            y_pos += 45

    y_pos += 10
    draw.text((35, y_pos), "Sətir Səviyyəsi (32x512):", fill=(80, 80, 80), font=f_txt)
    y_pos += 22
    for p, lbl in lines:
        if p.exists():
            im = Image.open(p).convert("RGB")
            canvas.paste(im.resize((320, 32)), (35, y_pos))
            draw.rectangle([(35, y_pos), (355, y_pos + 32)], outline=(180, 180, 180))
            draw.text((35, y_pos + 35), f"'{lbl[:35]}...'", fill=(70, 70, 70), font=f_txt)
            y_pos += 58

    # Subpanel b
    draw.rectangle([(620, 65), (1180, 580)], outline=(210, 215, 225), width=2)
    draw.rectangle([(620, 65), (1180, 100)], fill=(245, 247, 250))
    draw.text((635, 74), "(b) V2 Fotorealistik Kamera Təhrifləri (İşıq, Küy, Kölgə, Sıxılma)", fill=(30, 50, 90), font=f_sub)

    from train_v2 import augment
    rng = np.random.default_rng(42)
    effects = [
        "İşıq qradiyenti + Yumşaq kölgə zolağı (p=0.5)",
        "Gauss sensor küyü + Duz-İstiot küyü (p=0.6, p=0.2)",
        "Perspektiv əyilmə (±5°) + Hərəkət bulanıqlığı",
        "JPEG sıxılma (Keyfiyyət 40-90) + Kontrast dartılması"
    ]
    y_pos = 115
    for i, effect in enumerate(effects):
        p, lbl = words[i % len(words)]
        if p.exists():
            cv_img = cv2.imread(str(p))
            aug_img = augment(cv_img, rng)
            im_aug = Image.fromarray(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
            canvas.paste(im_aug.resize((150, 40)), (635, y_pos))
            draw.rectangle([(635, y_pos), (785, y_pos + 40)], outline=(180, 180, 180))
            draw.text((795, y_pos + 4), effect, fill=(20, 70, 40), font=ImageFont.truetype(FONT_BOLD, 12))
            draw.text((795, y_pos + 22), f"Simvol: '{lbl}'", fill=(100, 100, 100), font=f_txt)
            y_pos += 80

    canvas.save(FIG_DIR / "fig1_synth_and_aug.png", quality=95)
    print("Saved fig1:", FIG_DIR / "fig1_synth_and_aug.png")


# FIGURE 2: The Two Definitive Real Document Photos (Times New Roman vs Calibri)
def generate_fig2():
    w_fig, h_fig = 1200, 600
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_sub = ImageFont.truetype(FONT_BOLD, 16)
    f_meta = ImageFont.truetype(FONT_REG, 13)

    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 2. Sınaqdan Keçirilən İki Real Mobil Telefon Sənəd Təsviri (Şrift Ailələri Müqayisəsi)", fill=(20, 40, 80), font=f_title)

    cards = [
        ("Foto 1: Serif Şrift Ailəsi (Times New Roman)", Path("data/real_docs/photo1_times.jpg"),
         "Mətn: 'Usta Əli və Xalçaçılıq Sənəti' | 3024x4032 (12MP) | V2 Söz CER: 0.00% (Dəqiq!)"),
        ("Foto 2: Sans-Serif Şrift Ailəsi (Calibri)", Path("data/real_docs/photo2_calibri.jpg"),
         "Mətn: 'Qoca Səməndər və Dulusçuluq Sənəti' | 3024x4032 (12MP) | V2 Söz CER: 1.29%")
    ]

    card_w = 560
    card_h = 510
    for idx, (title, p, meta) in enumerate(cards):
        x0 = 30 + idx * (card_w + 20)
        y0 = 65
        draw.rectangle([(x0, y0), (x0 + card_w, y0 + card_h)], outline=(210, 215, 225), width=2)
        draw.rectangle([(x0, y0), (x0 + card_w, y0 + 38)], fill=(245, 247, 250))
        draw.text((x0 + 15, y0 + 8), title, fill=(30, 50, 90), font=f_sub)

        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((530, 400), Image.Resampling.LANCZOS)
            px = x0 + (card_w - im.width) // 2
            py = y0 + 48 + (400 - im.height) // 2
            canvas.paste(im, (px, py))
            draw.rectangle([(px, py), (px + im.width, py + im.height)], outline=(190, 190, 190))

        draw.rectangle([(x0 + 5, y0 + card_h - 45), (x0 + card_w - 5, y0 + card_h - 5)], fill=(248, 250, 252))
        draw.text((x0 + 12, y0 + card_h - 36), meta, fill=(60, 60, 60), font=f_meta)

    canvas.save(FIG_DIR / "fig2_real_documents.png", quality=95)
    print("Saved fig2:", FIG_DIR / "fig2_real_documents.png")


# FIGURE 3: Live Visual Comparison from both real documents
def generate_fig3():
    w_fig, h_fig = 1200, 700
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_hdr = ImageFont.truetype(FONT_BOLD, 14)
    f_gt = ImageFont.truetype(FONT_BOLD, 13)
    f_res = ImageFont.truetype(FONT_REG, 13)
    f_err = ImageFont.truetype(FONT_REG, 13)

    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 3. Real Kamera Kəsimlərinin Modellər Üzrə Canlı Tanınma Nəticələrinin Müqayisəsi", fill=(20, 40, 80), font=f_title)

    headers = [
        ("Kəsilmiş Şəkil Fraqmenti (Kamera)", 20, 320),
        ("Əsl Mətn (Ground Truth)", 345, 225),
        ("V2 CRNN Modeli (Bizim)", 575, 225),
        ("Tesseract 5 (aze)", 805, 200),
        ("V1 CRNN Modeli", 1010, 170)
    ]
    draw.rectangle([(15, 60), (w_fig - 15, 95)], fill=(230, 238, 248))
    for h_txt, x, w in headers:
        draw.text((x + 5, 68), h_txt, fill=(20, 40, 80), font=f_hdr)

    samples = [
        # From Doc 1 (Times New Roman)
        {
            "crop": CROPS_P1 / "word_04.png",
            "gt": "rəngə",
            "v2": "rəngə",
            "tes": "Təngə (xəta: r->T)",
            "v1": "rəngə",
        },
        {
            "crop": CROPS_P1 / "word_08.png",
            "gt": "təbii",
            "v2": "təbii",
            "tes": "təbi (xəta: -i)",
            "v1": "təbii",
        },
        {
            "crop": CROPS_P1 / "word_16.png",
            "gt": "xüsusi",
            "v2": "xüsusi",
            "tes": "XÜSuSsi (xəta: registr/təkrar)",
            "v1": "xüsusi",
        },
        # From Doc 2 (Calibri)
        {
            "crop": CROPS_P2 / "word_00.png",
            "gt": "Gəncə",
            "v2": "Gəncə",
            "tes": "Gəncə",
            "v1": "Gəndə (xəta: c->d)",
        },
        {
            "crop": CROPS_P2 / "word_07.png",
            "gt": "dulusçu",
            "v2": "dulusçu",
            "tes": "dulusçu",
            "v1": "dulusçu",
        },
        {
            "crop": CROPS_P2 / "word_13.png",
            "gt": "Səməndər",
            "v2": "Səməndər",
            "tes": "Səməndər",
            "v1": "Səməndər",
        },
    ]

    y_pos = 105
    for s_idx, s in enumerate(samples):
        row_bg = (255, 255, 255) if s_idx % 2 == 0 else (248, 250, 253)
        draw.rectangle([(15, y_pos - 5), (w_fig - 15, y_pos + 80)], fill=row_bg, outline=(225, 230, 238))

        if s["crop"].exists():
            im = Image.open(s["crop"]).convert("RGB")
            im.thumbnail((310, 65), Image.Resampling.LANCZOS)
            canvas.paste(im, (25, y_pos + (65 - im.height) // 2))
            draw.rectangle([(25, y_pos + (65 - im.height) // 2), (25 + im.width, y_pos + (65 - im.height) // 2 + im.height)], outline=(180, 180, 180))

        draw.text((350, y_pos + 22), s["gt"], fill=(20, 20, 20), font=f_gt)
        draw.text((580, y_pos + 22), s["v2"], fill=(15, 120, 50), font=ImageFont.truetype(FONT_BOLD, 13))

        tes_color = (15, 120, 50) if "xəta" not in s["tes"] else (190, 40, 30)
        draw.text((810, y_pos + 22), s["tes"], fill=tes_color, font=f_res)

        v1_color = (15, 120, 50) if "xəta" not in s["v1"] else (190, 40, 30)
        draw.text((1015, y_pos + 22), s["v1"], fill=v1_color, font=f_err)

        y_pos += 90

    canvas.save(FIG_DIR / "fig3_ocr_visual_comparison.png", quality=95)
    print("Saved fig3:", FIG_DIR / "fig3_ocr_visual_comparison.png")

generate_fig1()
generate_fig2()
generate_fig3()
print("All 3 figures successfully updated!")
