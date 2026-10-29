"""Generate academic figures for the research report.
Figure 1: Synthetic Dataset & Camera Augmentation Pipeline
Figure 2: Real Document Phone Photos (Photo 1, Photo 2, Photo 3)
Figure 3: Visual OCR Comparison (Cropped Image strips vs. GT vs. V2 vs. Tesseract vs. V1)
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Try to find standard fonts
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REG = "C:/Windows/Fonts/segoeui.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"

if not Path(FONT_BOLD).exists():
    FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
    FONT_REG = "C:/Windows/Fonts/arial.ttf"
    FONT_MONO = "C:/Windows/Fonts/cour.ttf"


# ---------------------------------------------------------------------
# Figure 1: Synthetic Dataset Generation & Realistic Camera Augmentations
# ---------------------------------------------------------------------
def make_fig1():
    print("Generating Figure 1: Synthetic Pipeline & Augmentations...")
    # Load sample images from data_check or generate clean contact
    w_fig, h_fig = 1200, 600
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_sub = ImageFont.truetype(FONT_BOLD, 16)
    f_txt = ImageFont.truetype(FONT_REG, 13)

    # Title Banner
    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 1. Azərbaycan Dili üçün Sintetik Məlumat Bazası və Fotorealistik Kamera Auqmentasiyası", fill=(20, 40, 80), font=f_title)

    # Sub-panels
    # Left: Clean synthetic samples (Word & Line)
    draw.rectangle([(20, 65), (580, 580)], outline=(210, 215, 225), width=2)
    draw.rectangle([(20, 65), (580, 100)], fill=(245, 247, 250))
    draw.text((35, 74), "(a) Təmiz Sintetik Nümunələr (TRDG + Şrift Filtrləməsi)", fill=(30, 50, 90), font=f_sub)

    # Load 4 clean samples from data/synth_word/train and data/synth_line/train
    from common import load_labels
    words = load_labels("data/synth_word/train")[:4]
    lines = load_labels("data/synth_line/train")[:3]

    y_pos = 115
    draw.text((35, y_pos), "Söz Səviyyəsi (32x128):", fill=(80, 80, 80), font=f_txt)
    y_pos += 22
    for p, lbl in words:
        if p.exists():
            im = Image.open(p).convert("RGB")
            # Draw sample with border
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

    # Right: Augmented camera samples
    draw.rectangle([(620, 65), (1180, 580)], outline=(210, 215, 225), width=2)
    draw.rectangle([(620, 65), (1180, 100)], fill=(245, 247, 250))
    draw.text((635, 74), "(b) V2 Kamera Təhrifi Auqmentasiyaları (İşıq, Kölgə, Küy, Bucaqlı)", fill=(30, 50, 90), font=f_sub)

    # Generate 4 augmented examples on the fly using train_v2 augment
    from train_v2 import augment
    rng = np.random.default_rng(42)
    y_pos = 115
    effects = [
        "İşıq qradiyenti + Yumşaq kölgə zolağı (p=0.5)",
        "Gauss sensor küyü + Duz-İstiot küyü (p=0.6, p=0.2)",
        "Perspektiv əyilmə (±5°) + Hərəkət bulanıqlığı",
        "JPEG sıxılma (Keyfiyyət 40-90) + Kontrast dartılması"
    ]
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
    print("Saved:", FIG_DIR / "fig1_synth_and_aug.png")


# ---------------------------------------------------------------------
# Figure 2: The Three Tested Real Document Photos
# ---------------------------------------------------------------------
def make_fig2():
    print("Generating Figure 2: Tested Real Document Photos...")
    w_fig, h_fig = 1200, 580
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_sub = ImageFont.truetype(FONT_BOLD, 15)
    f_cap = ImageFont.truetype(FONT_REG, 12)

    # Title Banner
    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 2. Sınaqdan Keçirilən Real Telefon Sənəd Təsvirləri (Giriş Fotoları)", fill=(20, 40, 80), font=f_title)

    photos = [
        ("Foto 1: NDU Təqdimatı (Bucaqlı)", Path(r"C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793193977057.jpg"),
         "1024x768 | Kölgəli & Bucaqlı | Sətir CER: 28.87% (V2)"),
        ("Foto 2: NDU Təqdimatı (Perpendikulyar)", Path(r"C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793194879421.jpg"),
         "768x1024 | Düz Bucaq & Aydın | Sətir CER: 5.02% (V2)"),
        ("Foto 3: Bədii Nəsr — 'Usta Əli'", Path("data/real_test3/photo3.jpg"),
         "3024x4032 (12MP) | Zəngin Diakritik | Söz CER: 0.00% (V2)")
    ]

    card_w = 360
    card_h = 490
    for idx, (title, p, meta) in enumerate(photos):
        x0 = 25 + idx * (card_w + 30)
        y0 = 65
        # Card container
        draw.rectangle([(x0, y0), (x0 + card_w, y0 + card_h)], outline=(210, 215, 225), width=2)
        draw.rectangle([(x0, y0), (x0 + card_w, y0 + 35)], fill=(245, 247, 250))
        draw.text((x0 + 12, y0 + 8), title, fill=(30, 50, 90), font=f_sub)

        if p.exists():
            im = Image.open(p).convert("RGB")
            # Fit inside 340 x 380 preserving aspect ratio
            im.thumbnail((340, 380), Image.Resampling.LANCZOS)
            px = x0 + (card_w - im.width) // 2
            py = y0 + 45 + (380 - im.height) // 2
            canvas.paste(im, (px, py))
            draw.rectangle([(px, py), (px + im.width, py + im.height)], outline=(190, 190, 190))

        # Metadata footer
        draw.rectangle([(x0 + 5, y0 + card_h - 45), (x0 + card_w - 5, y0 + card_h - 5)], fill=(248, 250, 252))
        draw.text((x0 + 10, y0 + card_h - 38), meta, fill=(60, 60, 60), font=f_cap)

    canvas.save(FIG_DIR / "fig2_real_documents.png", quality=95)
    print("Saved:", FIG_DIR / "fig2_real_documents.png")


# ---------------------------------------------------------------------
# Figure 3: Visual OCR Comparison (Cropped Image Strips vs Models)
# ---------------------------------------------------------------------
def make_fig3():
    print("Generating Figure 3: Visual OCR Comparison Panel...")
    w_fig, h_fig = 1200, 680
    canvas = Image.new("RGB", (w_fig, h_fig), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(FONT_BOLD, 22)
    f_hdr = ImageFont.truetype(FONT_BOLD, 14)
    f_gt = ImageFont.truetype(FONT_BOLD, 13)
    f_res = ImageFont.truetype(FONT_REG, 13)
    f_err = ImageFont.truetype(FONT_REG, 13)

    # Title Banner
    draw.rectangle([(0, 0), (w_fig, 50)], fill=(240, 244, 250))
    draw.text((20, 12), "Şəkil 3. Real Sənəd Kəsimlərinin Modellər Üzrə Tanınma Nəticələrinin Vizual Müqayisəsi", fill=(20, 40, 80), font=f_title)

    # Table Header
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

    # 6 Sample items from Real Test 2 and Real Test 3
    samples = [
        {
            "crop": Path("results/real_test2/crops/line_03.png"),
            "gt": "Əsas səhifənin daha informativ və funksional qurulması.",
            "v2": "Əsas səhifənin daha informativ və funksional qurulması.",
            "tes": "Əsas səhifənin daha informativ və funksional qurulması.",
            "v1": "\"Əəaşiəəhifəninidahainiförmiaitiyi...",
            "match": True
        },
        {
            "crop": Path("results/real_test2/crops/line_07.png"),
            "gt": "Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi",
            "v2": "Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi",
            "tes": "Elektron lurnal idarəetmə sisteminin...",
            "v1": "RiBlaktonijürmaiidaraatmaisistami...",
            "match": True
        },
        {
            "crop": Path("results/real_test2/crops/line_20.png"),
            "gt": "Naxçıvan Dövlət Universiteti",
            "v2": "Naxçıvan Dövlət Universiteti",
            "tes": "Naxçıvan Dövlət Universiteti",
            "v1": "idaxgiyanqöxiatüiixaritati",
            "match": True
        },
        {
            "crop": Path("results/real_test3/crops/word_00.png"),
            "gt": "Əlinin",
            "v2": "Əlinin",
            "tes": "Əlinin",
            "v1": "Əlinin",
            "match": True
        },
        {
            "crop": Path("results/real_test3/crops/word_04.png"),
            "gt": "rəngə",
            "v2": "rəngə",
            "tes": "Təngə (xəta: r->T)",
            "v1": "rəngə",
            "match": True
        },
        {
            "crop": Path("results/real_test3/crops/word_08.png"),
            "gt": "təbii",
            "v2": "təbii",
            "tes": "təbi (xəta: -i)",
            "v1": "təbii",
            "match": True
        },
    ]

    y_pos = 105
    for s_idx, s in enumerate(samples):
        row_bg = (255, 255, 255) if s_idx % 2 == 0 else (248, 250, 253)
        draw.rectangle([(15, y_pos - 5), (w_fig - 15, y_pos + 75)], fill=row_bg, outline=(225, 230, 238))

        # Crop Image
        if s["crop"].exists():
            im = Image.open(s["crop"]).convert("RGB")
            # Fit crop to 310 x 60
            im.thumbnail((310, 60), Image.Resampling.LANCZOS)
            canvas.paste(im, (25, y_pos + (60 - im.height) // 2))
            draw.rectangle([(25, y_pos + (60 - im.height) // 2), (25 + im.width, y_pos + (60 - im.height) // 2 + im.height)], outline=(180, 180, 180))

        # Ground truth
        draw.text((350, y_pos + 18), s["gt"][:30], fill=(20, 20, 20), font=f_gt)

        # V2 (Our model)
        draw.text((580, y_pos + 18), s["v2"][:30], fill=(15, 120, 50), font=ImageFont.truetype(FONT_BOLD, 13))

        # Tesseract
        tes_color = (15, 120, 50) if "xəta" not in s["tes"] else (190, 40, 30)
        draw.text((810, y_pos + 18), s["tes"][:28], fill=tes_color, font=f_res)

        # V1
        draw.text((1015, y_pos + 18), s["v1"][:24], fill=(160, 40, 40), font=f_err)

        y_pos += 85

    canvas.save(FIG_DIR / "fig3_ocr_visual_comparison.png", quality=95)
    print("Saved:", FIG_DIR / "fig3_ocr_visual_comparison.png")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    print("All 3 figures successfully generated in results/figures/")
