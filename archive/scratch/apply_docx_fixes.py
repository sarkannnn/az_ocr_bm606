import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SRC_DOCX = Path("results/Az_OCR_Akademik_Hesabat.docx")
DST_DOCX = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER.docx")

# Make a fresh copy
shutil.copy2(SRC_DOCX, DST_DOCX)
doc = docx.Document(DST_DOCX)

print(f"Loaded {DST_DOCX}")

# Helper to style run
def set_run_font(run, name="Calibri", size=10.5, color=None, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color

# 1. Title Block & Author
# P0: Title
# P1: Author metadata
if len(doc.paragraphs) > 1:
    p0 = doc.paragraphs[0]
    p0.text = "Azərbaycan Dili üçün Dərin Öyrənmə Əsaslı Optik Simvol Tanıma (CRNN+CTC) Sistemi və Real Dünya Sənəd Təsvirlərində Qiymətləndirilməsi"
    for r in p0.runs:
        set_run_font(r, name="Calibri", size=18, bold=True, color=RGBColor(0x1B, 0x36, 0x5D))
        
    p1 = doc.paragraphs[1]
    p1.text = "Müəllif: [Müəllif Adı] / BM 606 Tədqiqat Qrupu | Layihə Kodu: BM 606 | Tarix: 29 Oktyabr 2026\nElmi Nəşrə Hazır Texniki Monoqrafiya (Fakt-Yoxlanılmış Düzəliş Nəşri)"
    for r in p1.runs:
        set_run_font(r, name="Calibri", size=10, italic=True, color=RGBColor(0x55, 0x55, 0x55))

# 2. Abstract & Key takeaways (P4 - P20)
# Update Abstract paragraph (P4)
for i, p in enumerate(doc.paragraphs[:15]):
    if "Xülasə (Abstract)" in p.text or "Bu tədqiqat işində" in p.text:
        p.text = (
            "Bu tədqiqat işində Azərbaycan dilinin xüsusi qrafemlərini (ə, ı, ö, ğ, ü, ş, ç) ehtiva edən çap sənədlərinin "
            "rəqəmsallaşdırılması üçün Konvolusion və İkiistiqamətli Təkrarlanan Neyron Şəbəkələrinə (CRNN) və "
            "Bağlantıçı Zaman Təsnifatına (CTC) əsaslanan dərin öyrənmə arxitekturası layihələndirilmiş, təlim edilmiş "
            "və hərtərəfli audit edilmişdir. Modelin birinci versiyasının (V1) real telefon fotoşəkillərində kəskin deqradasiyaya "
            "uğraması (sətir CER ~48–73%) aradan qaldırılaraq, fotorealistik kamera təhrifləri (işıq qradiyenti, kölgə zolaqları, "
            "hərəkət bluru, sensor küyü, JPEG sıxılması) ilə V2 modeli hazırlanmışdır. Sintetik qarışıq testlərdə V2 modeli 0.11% "
            "(söz) və 0.30% (sətir) CER nümayiş etdirmişdir. Real telefon fotolarında (Real Document 2) sənaye standartı olan "
            "Tesseract 5 LSTM modeli sətir səviyyəsində demək olar ki, qüsursuz nəticə göstərmişdir (CER 0.24%, 21 sətirdən 19-u tam dəqiq). "
            "V2 modeli sətir səviyyəsində 5.02% CER əldə etmişdir. Kəsilmiş söz səviyyəsində isə (n=30) V2 modeli 46.71% CER nümayiş "
            "etdirərək Tesseract-dan (50.99%) irəlidə görünmüşdür; lakin söz kəsimlərinin 60%-də aşkar edilmiş qonşu söz mürəkkəb müdaxiləsi "
            "(neighbour intrusion) səbəbindən bu üstünlük ilkin (preliminary) xarakter daşıyır."
        )
        for r in p.runs:
            set_run_font(r, name="Calibri", size=10.5)
        break

# Update Table 1: Augmentations
t1 = doc.tables[0]
t1_data = [
    ["Parametr / Təhrif Növü", "V1 Modeli (train.py)", "V2 Modeli (train_v2.py:AUG_CONFIG) [Faktiki Parametrlər]"],
    ["Fırlanma Bucağı (Rotation)", "± 5° kobud fırlanma (p=0.5)", "± 3° incə fırlanma (p=0.5) [YENİ BÖLGÜ]"],
    ["İşıqlanma / Kölgə [YENİ]", "Yoxdur (bərabər fon)", "Hamar işıq qradiyenti (p=0.5), sintetik kölgə zolaqları"],
    ["Sensor Küyü (Noise) [YENİ]", "Bəsit Duz-İstiot (p=0.3, ratio=0.005)", "Qauss sensor küyü (σ ∈ [3, 12], p=0.6) + Duz-istiot (p=0.2)"],
    ["Bulanıqlıq (Blur) [YENİ]", "Bəsit Qauss Bluru (k∈{3,5}, p=0.3)", "Qauss blur (σ ∈ [0.5, 2.0], p=0.6) + Hərəkət bluru (kernel 3–7, p=0.15)"],
    ["Ölçü Dəyişməsi (Downscale) [YENİ]", "Yoxdur", "0.5× – 0.9× süni rezolyusiya itkisi (p=0.4)"],
    ["JPEG Sıxılma Artefaktı [YENİ]", "Yoxdur", "Keyfiyyət dərəcəsi Q ∈ [40, 90] (p=0.4)"],
    ["Kontrast Təhrifi [YENİ]", "Yoxdur", "Parlaqlıq/kontrast (α∈[0.6, 1.1], β∈[-40, 20], p=0.6) + Kontrast dartılması (p=0.5)"],
]
for r_idx, row in enumerate(t1.rows):
    if r_idx < len(t1_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t1_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=9.5, bold=(r_idx==0))

# Update Table 2: Training Parameters
t2 = doc.tables[1]
t2_data = [
    ["Model", "Ümumi Epox", "Ən Yaxşı Epox", "Val Loss", "Val CER (%)*", "Təlim Vaxtı", "Checkpoint Ölçüsü"],
    ["V2 Word", "18 (Early Stop, pat=5)", "Epox 13", "0.0063", "0.24% (Seen fonts)", "212.3 dəq ümumi (154.1 dəq @ best)", "98.7 MB (xalis: 32.9 MB)"],
    ["V2 Line", "15 (Early Stop, pat=5)", "Epox 10", "0.0125", "0.28% (Seen fonts)", "280.4 dəq ümumi (186.9 dəq @ best)", "98.7 MB (xalis: 32.9 MB)"],
]
for r_idx, row in enumerate(t2.rows):
    if r_idx < len(t2_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t2_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=9.0, bold=(r_idx==0))

# Update Table 3: Synthetic Test Sets
t3 = doc.tables[2]
t3_data = [
    ["Dataset / Test Çoxluğu", "Dəqiq Şrift Tərkibi və Təsnifatı", "V1 Modeli CER", "V2 Modeli CER", "Tesseract 5 CER", "V1 -> V2 Dəyişimi"],
    ["Söz Testi (word_synth)", "1,001 tahoma (New) + 999 consola (Leakage)", "0.50%", "0.11%", "0.70%", "CER 4.55 dəfə azaldı"],
    ["Sətir Testi (line_synth)", "501 tahoma (New) + 499 consola (Leakage)", "1.64%", "0.30%", "0.44%", "CER 5.47 dəfə azaldı"],
    ["Söz, Tahoma (diagw_tahoma)", "1,000 tahoma.ttf — NEW FAMILY", "0.08%", "0.09%", "0.50%", "Dəyişmədi (+0.01% dalğalanma)"],
    ["Sətir, Tahoma (diag_tahoma)", "500 tahoma.ttf — NEW FAMILY", "0.24%", "0.26%", "0.22%", "Dəyişmədi (+0.02% dalğalanma)"],
    ["Söz, Consolas (diagw_consola)", "1,000 consola.ttf — FAMILY LEAKAGE (consolab)", "1.02%", "0.07%", "0.86%", "CER 14.57x azaldı (Ailə sızması)"],
    ["Sətir, Consolas (diag_consola)", "500 consola.ttf — FAMILY LEAKAGE (consolab)", "3.48%", "0.27%", "0.47%", "CER 12.89x azaldı (Ailə sızması)"],
]
for r_idx, row in enumerate(t3.rows):
    if r_idx < len(t3_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t3_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=9.0, bold=(r_idx==0))

# Update Table 4: Simulation Ladder
t4 = doc.tables[3]
t4_data = [
    ["Pillə", "Tətbiq Edilən Təhrif Növü", "V1 Word CER", "V2 Word CER", "V1 Line CER*", "V2 Line CER"],
    ["L0", "Təmiz render (Sıfır təhrif)", "7.71%", "0.20%", "1.87%", "0.00%"],
    ["L1", "İşıqlanma qradiyenti və kölgə", "8.85%", "0.20%", "4.10%", "0.00%"],
    ["L2", "Bulanıqlıq və sensor küyü", "25.00%", "0.41%", "13.84%", "0.04%"],
    ["L3", "Ölçü kiçiltmə (0.6×) və JPEG (Q=50)", "31.36%", "0.53%", "19.14%*", "0.04%"],
    ["L4", "Tam kamera təhrifi (Bütün faktorlar)", "35.77%", "1.35%", "12.96%*", "0.11%"],
    ["L4n", "L4 + Kontrast normallaşdırması", "24.43%", "1.14%", "7.43%", "0.09%"],
]
for r_idx, row in enumerate(t4.rows):
    if r_idx < len(t4_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t4_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=9.0, bold=(r_idx==0))

# Update Table 5: Real Document Test 2
t5 = doc.tables[4]
t5_data = [
    ["Kateqoriya", "Model", "CER (%)", "WER (%) [Standart]", "100% Dəqiq Tanınan Sayı / Cəmi"],
    ["Sətirlər (21 sətir, 837 simvol)", "V1 Line", "48.51%", "98.88% (88/89)", "0 / 21 (0.0%)"],
    ["", "V2 Line", "5.02%", "22.47% (20/89)", "7 / 21 tam raw (9/21 strip)"],
    ["", "Tesseract 5", "0.24%", "2.25% (2/89)", "19 / 21 tam qüsursuz (90.5%)"],
    ["Sözlər (30 söz, 304 simvol)**", "V1 Word", "90.13%", "100.00% (30/30)", "0 / 30 (0.0%)"],
    ["", "V2 Word", "46.71%", "90.00% (27/30)", "3 / 30 (10.0%)"],
    ["", "Tesseract 5", "50.99%", "146.67% (44/30)", "4 / 30 (13.3%)"],
]
for r_idx, row in enumerate(t5.rows):
    if r_idx < len(t5_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t5_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=9.0, bold=(r_idx==0))

# Update Table 6: Real Transcription Examples
t6 = doc.tables[5]
t6_data = [
    ["Orijinal Mətn (Ground Truth)", "V1 Line Modeli", "V2 Line Modeli", "Tesseract 5 (aze)"],
    ["TƏQDİMAT (Başlıq, x=300..485)", "jirəggitifTijj", " TəqDiMAT ı", "TƏQDİMAT (100% Dəqiq)"],
    ["hazırlanması.", "ihagiianmasiji", "hazırlanması.. (Nöqtə təkrarlanması)", "hazırlanması."],
    ["Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi", "RiBlaktonijürmaiidaraatmaisistamin...", "Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi (100%)", "Elektron lurnal idarəetmə sisteminin..."],
    ["Nəşr və xidmət haqları barədə məlumatların ayrıca təqdim edilməsi.", "iHaşrijaixidməthağlaribarəğamalüin...", "Nəşr və xldmət haqları barədə məlumatların... ('i'->'l')", "Nəşr və xidmət haqları barədə məlumatların..."],
    ["Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi", "y6bsaytinidizaynininipaistifadəği...", "y6b-saytın dizaynının və İstifadəçi interfeysinin yenilənməsi", "Vöb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi"],
]
for r_idx, row in enumerate(t6.rows):
    if r_idx < len(t6_data):
        for c_idx, cell in enumerate(row.cells):
            cell.text = t6_data[r_idx][c_idx]
            for p in cell.paragraphs:
                for r in p.runs:
                    set_run_font(r, size=8.5, bold=(r_idx==0))

# Update Paragraphs in Sec 5, 6, 7
for i, p in enumerate(doc.paragraphs):
    txt = p.text
    # Ladder caption & non-monotonicity note
    if "Cədvəl 4." in txt:
        p.text = "Cədvəl 4. 4-Şriftli Simulyasiya Nərdivanında (cambria.ttc, micross.ttf — 2 yeni ailə; pala.ttf, CascadiaMono.ttf — 2 yaxın qohum) Söz və Sətir CER (%) dinamikası. Qeyd: V1 sətir CER göstəricisi L3-də (19.14%) L4-dən (12.96%) yüksəkdir; bu qeyri-monotonluğun səbəbi araşdırılmamışdır."
    elif "Foto 2 üzrə" in txt or "Foto 2:" in txt or "Sınaq 2-nin təhlili" in txt:
        p.text = (
            "Foto 2 (media_1793194879421.jpg) üzrə sınaq nəticələri: Sənəd üfüqi çəkilişlə daha aydın təsvir olunmuşdur. "
            "Düzəldilmiş TƏQDİMAT kəsimi ilə (x=300..485) Tesseract 5 sətir səviyyəsində demək olar ki, qüsursuz nəticə nümayiş etdirmişdir: "
            "Sətir CER 0.24% (cəmi 2 hərf xətası: 'Vöb-saytın' və 'lurnal'), Sətir WER 2.25% (21 sətirdən 19-u 100% tam dəqiq). "
            "V2 modeli sətir səviyyəsində 5.02% CER və 22.47% standart WER əldə etmişdir (7 sətir tam raw bərabər, 9 sətir kənar boşluqsuz bərabər). "
            "Söz səviyyəsində V2 CER 46.71% vs Tesseract 50.99% təşkil edir. Kənar mürəkkəb analizi göstərmişdir ki, 30 söz kəsiminin "
            "18-də (60.0%) qonşu söz mürəkkəb müdaxiləsi (neighbour intrusion) mövcuddur; bu səbəbdən söz səviyyəsində V2-nin Tesseract-dan irəlidə olması "
            "ilkin (preliminary) xarakter daşıyır."
        )
    elif "Müşahidə: V2 modeli hətta" in txt:
        p.text = (
            "Müşahidə: V2 modeli sənədin rəsmi künc ştampını ('Naxçıvan Dövlət Universiteti') dəqiq tanımışdır (kənardakı tək boşluq xaric). "
            "Tesseract isə 'jurnal' sözünü 'lurnal', 'Veb-saytın' sözünü 'Vöb-saytın' kimi tanımış, lakin bütövlükdə sənəd sətirlərində 0.24% CER ilə "
            "V2-dən (5.02%) qat-qat yüksək ümumi sabitlik nümayiş etdirmişdir."
        )
    elif "6.1. Şrift Ailəsi Sızması" in txt:
        p.text = "6.1. Şrift Ailəsi Sızması və Ümumiləşmə Təhlili (Groups A, B, C Auditi)"
    elif "Bununla belə, təlimdə iştirak etməyən Cambria" in txt:
        p.text = (
            "Audit 4-ün tam klaster-bootstrap təhlili (95% CI) göstərdi ki: Qrup A (görünən 4 şrift) üzrə L0-da V2 CER 0.81% vs Tesseract 2.39% "
            "(fərq -1.58% [-2.68, -0.76]), lakin L4-də Tesseract ilə statistik bərabərdir (fərq -0.36% [-1.25, +0.50]). "
            "Qrup B (yaxın qohum 5 şrift) üzrə L0 və L4-də modellər arasında fərq statistik əhəmiyyətli deyildir. "
            "Qrup C (tamamilə yeni 7 font ailəsi) üzrə L4 səviyyəsində Tesseract V2 modelindən statistik dürüstlüklə üstündür: "
            "V2 CER 2.36% vs Tesseract 0.81% (Fərq Δ = +1.55% [+0.59%, +2.94%], sıfır intervala daxil deyil). "
            "Həmçinin qeyd edilməlidir ki, Moba_Fira monospace şriftidir və təlimdə CascadiaCode və consolab iştirak etmişdir. "
            "Qrup A və B-də klaster sayı az olduğu üçün (k=4, 5) bu qrupların intervalları yüksək dispersiyaya malikdir."
        )
    elif "6.2. Diakritik Qarışıqlıq" in txt:
        p.text = "6.2. Diakritik Qarışıqlıq və Hərf Matrisi Auditi"
    elif "'ı' -> 'i' qarışması:" in txt or "'ə' -> 'e' qarışması:" in txt:
        p.text = (
            "Audit 3 və Audit 4-ün dəqiq çaşqınlıq matrisi analizi göstərmişdir ki, V2 modelində 'ə' -> 'e' xətası 0.00% təşkil edir (0 / 394 instansiya). "
            "'ı' -> 'i' xətası isə V2 modelində 6.52%–7.25% (9–10 / 138 söz) təşkil edir (Tesseract: 0.0%–0.72%). "
            "V2 modelində 'ı/i' xətalarının 91.5%-i cəmi 3 şriftdə (Moba_Fanta, Moba_Font və SansSerifCollection) cəmlənmişdir. "
            "Əvvəlki sənəddə qeyd olunan 'ı->i V1 14.2% -> V2 0.8%' və 'ə->e V1 8.7% -> V2 0.4%' iddiaları təsdiqlənməmişdir (mənbəsizdir)."
        )
    elif "Durğu işarələri: V2 sətir modelində" in txt:
        p.text = (
            "Durğu işarələri və CTC boşluq xətaları: V2 sətir modelində sətir sonundakı tək nöqtə simvolu bəzən '..' kimi dekodlaşdırılır "
            "(məsələn, 'hazırlanması..', 'qurulması..'). "
            "Sətir səviyyəsində V2-nin söz xətalarının (WER) 65.3%–79.3%-i söz daxilinə səhvən boşluq daxil edilməsindən qaynaqlanır "
            "(L0-da 46/58 xəta = 79.3%, L4-də 64/98 xəta = 65.3%). Tesseract-da isə boşluq xətası demək olar ki, yoxdur (0.0% – 5.9%)."
        )
    elif "7. Nəticə və Gələcək" in txt:
        p.text = "7. Tədqiqat Məhdudiyyətləri, Nəticə və Gələcək İstiqamətlər"
    elif "Bu monoqrafiyada təqdim olunan Azərbaycan dili üçün" in txt:
        p.text = (
            "Tədqiqat Məhdudiyyətləri (Reproducibility & Scope Limitations):\n"
            "1. Təlim Skriptinin Qorunması: V2 verilənlər bazasına əlavə olunan 15,000 söz və 6,000 sətir scratch/build_v2_datasets.py skripti ilə "
            "generate_data.py TRDG generatoru (GeneratorFromStrings, bgs=[0,1], size=64/48) əsasında yaradılmışdır.\n"
            "2. Korpus Bölünməsi: Test dəstindəki sözlərin 22.40%-nin (276 / 1232) təlim lüğətində rast gəlinməsi token səviyyəli kəsişmədir; "
            "unikal söz formaları (types) üzrə verilənlər bazası düzgün bölünmüşdür.\n"
            "3. Təlim Qeyri-Sabitliyi: V2 sətir modelində təlimin son epoxlarında (epox 9-da val loss 0.72-yə, epox 13-də 0.65-ə, epox 15-də 0.48-ə sıçramışdır) "
            "itki dalğalanması müşahidə olunmuşdur; model 10-cu epoxda optimal nöqtəyə çatmışdır.\n"
            "4. Real Şəkil Üstünlüyü Hüdudları: Tesseract 5 sənəd sətirlərində 0.24% CER ilə layihə modelindən (5.02%) aydın şəkildə üstündür. "
            "Modelimizin Tesseract-dan üstünlüyü yalnız sintetik kəsilmiş sözlərdə və təhrifli söz sınaqlarında müşahidə olunur və ilkin xarakter daşıyır."
        )

# Remove Reference 5
# Look for paragraph starting with "5. Memmedov"
for i, p in enumerate(doc.paragraphs):
    if "5. Memmedov" in p.text:
        print(f"Removing Reference 5 from P{i}: {p.text}")
        p.text = "" # Cleared fictitious reference

doc.save(DST_DOCX)
print(f"Successfully saved corrected document to {DST_DOCX}")
