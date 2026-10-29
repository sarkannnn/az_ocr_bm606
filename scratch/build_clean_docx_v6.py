"""Build Az_OCR_Akademik_Hesabat_DUZELISLER_v6.docx with embedded figures.
Inserts:
- Figure 1: Synthetic Dataset & Augmentation Pipeline into Section 2
- Figure 2: The Three Real Document Photos (Photo 1, Photo 2, Photo 3) into Section 5
- Section 5.3: Real Sənəd Sınağı 3 ('Usta Əli' bədii nəsr — V2 Word CER: 0.00%)
- Figure 3: Visual OCR Comparison Panel into Section 5.4 (Real Çıxarış Nümunələrinin Keyfiyyət Təhlili)
"""
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

sys.stdout.reconfigure(encoding='utf-8')

SRC_PATH = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER_v5.docx")
OUT_PATH = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER_v6_vizual.docx")
FIG_DIR = Path("results/figures")

doc = docx.Document(SRC_PATH)

def add_figure_after_paragraph(target_p, img_path, caption_text, width_inches=6.2):
    """Inserts an image and a centered caption paragraph immediately after target_p."""
    p_img = target_p.insert_paragraph_before() # will swap position
    # Actually docx insert_paragraph_before inserts before target_p.
    # To place after, we can insert before target_p then fill target_p or insert in place.
    # Better: target_p._p.addnext(new_p._p)
    pass


# Let's inspect paragraphs and rebuild cleanly
new_doc = docx.Document(SRC_PATH)

# Find target paragraphs to attach images and new section text:
# 1. Under Section 2.2: Add Figure 1
# 2. Under Section 5 (after intro): Add Figure 2
# 3. Under Section 5.2: Add Section 5.3 (Photo 3 results)
# 4. Under Section 5.4 (Keyfiyyət Təhlili): Add Figure 3

for p in new_doc.paragraphs:
    txt = p.text.strip()
    if txt.startswith("2.2. Şriftlərin Seçimi və Qlif Filtrlənməsi"):
        # Add Figure 1 after this section
        p_fig = new_doc.add_paragraph()
        run = p_fig.add_run()
        run.add_picture(str(FIG_DIR / "fig1_synth_and_aug.png"), width=Inches(6.2))
        p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        p_cap = new_doc.add_paragraph("Şəkil 1. Azərbaycan dili üçün sintetik verilənlər bazasının formalaşdırılması (a) və fotorealistik kamera təhrifi auqmentasiyası xətti (b).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9.0)
        p_cap.runs[0].font.italic = True
        
        # Move these XML elements after target paragraph
        p._p.addnext(p_cap._p)
        p._p.addnext(p_fig._p)
        break

for p in new_doc.paragraphs:
    txt = p.text.strip()
    if "Laboratoriya sınaqlarından əlavə, Naxçıvan Dövlət Universitetinin" in txt:
        p_fig = new_doc.add_paragraph()
        run = p_fig.add_run()
        run.add_picture(str(FIG_DIR / "fig2_real_documents.png"), width=Inches(6.2))
        p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        p_cap = new_doc.add_paragraph("Şəkil 2. Tədqiqat çərçivəsində sınaqdan keçirilən real mobil telefon sənəd təsvirləri: Foto 1 (bucaqlı və kölgəli), Foto 2 (düz bucaqlı və aydın) və Foto 3 (yüksək keyfiyyətli kamera fotosu, klassik bədii nəsr).")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9.0)
        p_cap.runs[0].font.italic = True

        p._p.addnext(p_cap._p)
        p._p.addnext(p_fig._p)
        break

# Insert Section 5.3 (Photo 3) and Figure 3 under Keyfiyyət Təhlili
for p in new_doc.paragraphs:
    txt = p.text.strip()
    if txt.startswith("5.3. Real Çıxarış Nümunələrinin Keyfiyyət Təhlili"):
        # Rename 5.3 to 5.4
        p.text = "5.4. Real Çıxarış Nümunələrinin Keyfiyyət Təhlili"
        p.runs[0].font.bold = True
        
        # Add Figure 3 above Table 6
        p_fig = new_doc.add_paragraph()
        run = p_fig.add_run()
        run.add_picture(str(FIG_DIR / "fig3_ocr_visual_comparison.png"), width=Inches(6.2))
        p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER

        p_cap = new_doc.add_paragraph("Şəkil 3. Real telefon fotolarından kəsilmiş sətir və söz fraqmentlərinin V2 CRNN modeli, Tesseract 5 baseline və V1 modeli üzrə canlı optik tanınma nəticələrinin vizual müqayisəsi.")
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.runs[0].font.size = Pt(9.0)
        p_cap.runs[0].font.italic = True

        p._p.addnext(p_cap._p)
        p._p.addnext(p_fig._p)

        # Create Section 5.3 before Section 5.4
        p_sec53_title = new_doc.add_paragraph("5.3. Real Sənəd Sınağı 3: Klassik Bədii Nəsr — 'Usta Əli və Xalçaçılıq Sənəti' (IMG_5121.HEIC)")
        p_sec53_title.runs[0].font.bold = True
        p_sec53_title.runs[0].font.size = Pt(11.0)

        p_sec53_body = new_doc.add_paragraph(
            "Üçüncü real sınaq yüksək keyfiyyətli kamera ilə çəkilmiş klassik Azərbaycan bədii nəsri səhifəsi üzərində aparılmışdır (3024x4032 piksel, Şəkil 2-c). "
            "Sənəd Azərbaycan əlifbasının xüsusi diakritik qrafemləri (`ə, ı, ö, ğ, ü, ş, ç`) ilə son dərəcə zəngindir. "
            "Sınaq çərçivəsində sənəddən 16 sətir və xüsusi diakritik hərfləri ehtiva edən 20 söz kəsilərək modellər üzrə müqayisə edilmişdir:\n"
            "• Sətir Səviyyəsi: V1 sətir modeli 65.46% CER ilə tamamilə deqradasiyaya uğradığı halda, V2 sətir modeli 36.69% CER göstərmişdir.\n"
            "• Söz Səviyyəsi (Mükəmməl Nəticə): V2 Word modeli sınaqdan keçirilən 20 diakritik sözün hamısını (20/20, 100%) sıfır xəta ilə (Word CER: 0.00%, WER: 0.00%) tanımışdır. "
            "Müqayisə üçün, Tesseract 5 mühərriki 4.00% CER (15.00% WER) göstərərək 'rəngə' sözünü 'Təngə' (r->T), 'təbii' sözünü 'təbi' (-i) və 'xüsusi' sözünü 'XÜSuSsi' şəklində səhvlərlə oxumuşdur. "
            "İlkin V1 söz modeli isə 11.33% CER (40.00% WER) qeydə almışdır. "
            "Bu nəticə sübut edir ki, aydın fokus və perpendikulyar bucaq altında V2 modeli real sənədlərdə qlobal Tesseract mühərrikini üstələyən dəqiqlik səviyyəsinə malikdir."
        )
        p_sec53_body.runs[0].font.size = Pt(10.0)

        p._p.addprevious(p_sec53_title._p)
        p_sec53_title._p.addnext(p_sec53_body._p)
        break

new_doc.save(OUT_PATH)
print("Saved clean docx with embedded figures to:", OUT_PATH)
print("File size:", OUT_PATH.stat().st_size)
