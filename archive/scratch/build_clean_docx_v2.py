import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SRC_DOCX = Path("results/Az_OCR_Akademik_Hesabat.docx")
DST_DOCX = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER_v2.docx")

shutil.copy2(SRC_DOCX, DST_DOCX)
doc = docx.Document(DST_DOCX)

print(f"Base docx copied from {SRC_DOCX} to {DST_DOCX}")

def set_run_font(run, name="Calibri", size=10.0, color=None, bold=False, italic=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = color

def format_cell(cell, text, bold=False, italic=False, size=8.5, bg_color=None):
    cell.text = text
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.05
    for r in p.runs:
        set_run_font(r, size=size, bold=bold, italic=italic)
    if bg_color:
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

# ---------------------------------------------------------------------
# 1. Title Block & Abstract
# ---------------------------------------------------------------------
# P0: Title
doc.paragraphs[0].text = "Azərbaycan Dili üçün Dərin Öyrənmə Əsaslı Optik Simvol Tanıma (CRNN+CTC) Sistemi və Real Dünya Sənəd Təsvirlərində Qiymətləndirilməsi"
for r in doc.paragraphs[0].runs:
    set_run_font(r, name="Calibri", size=16, bold=True, color=RGBColor(0x1B, 0x36, 0x5D))

# P1: Author metadata - remove "Elmi Nəşrə Hazır Texniki Monoqrafiya"
doc.paragraphs[1].text = "Müəllif: [Müəllif Adı] / BM 606 Tədqiqat Qrupu | Layihə Kodu: BM 606 | Tarix: 29 Oktyabr 2026\nElmi Hesabat və Metodoloji Audit Sənədi"
for r in doc.paragraphs[1].runs:
    set_run_font(r, name="Calibri", size=9.5, italic=True, color=RGBColor(0x55, 0x55, 0x55))

# P3: Unified Abstract (Single abstract)
doc.paragraphs[3].text = (
    "Bu tədqiqat işində Azərbaycan dilinin xüsusi qrafemlərini (ə, ı, ö, ğ, ü, ş, ç) ehtiva edən çap sənədlərinin "
    "rəqəmsallaşdırılması üçün Konvolusion və İkiistiqamətli Təkrarlanan Neyron Şəbəkələrinə (CRNN) və Bağlantıçı Zaman "
    "Təsnifatına (CTC) əsaslanan dərin öyrənmə sistemi layihələndirilmiş, təlim edilmiş və empirik audit olunmuşdur. "
    "Kamera təhrifləri ilə zənginləşdirilmiş V2 modeli simulyasiya şəraitində ilkin V1 modelini kəskin üstələyir (V2 >> V1). "
    "Bununla belə, klaster-bootstrap təhlili (95% CI) göstərir ki, V2 modeli yalnız təlim şrift ailələri üzrə Tesseract 5 "
    "ilə bərabər səviyyədədir; tamamilə yeni şrift ailələrində isə Tesseract-dan statistik dürüstlüklə geri qalır "
    "(Qrup C L4 söz CER: V2 minus Tesseract = +1.55% [+0.59%, +2.94%]). Real mobil telefon çəkilişlərində Tesseract 5 "
    "sətir səviyyəsində aydın şəkildə öndədir: Foto 1-də Tesseract CER 21.92% vs V2 28.87%; Foto 2-də Tesseract CER 0.24% "
    "(19/21 sətir tam qüsursuz) vs V2 5.02%. Real sınaq eyni bir fiziki sənədin iki fərqli fotosundan ibarət olduğundan "
    "nəticələr ilkin (preliminary) xarakter daşıyır. Sintetik qarışıq testlərdə CER-in 0.50%-dən 0.11%-ə və 1.64%-dən 0.30%-ə "
    "enməsi test dəstinin yarısının Consolas ailə sızması hesabına baş vermişdir; digər yarısı olan Tahoma şriftində isə V2 "
    "heç bir irəliləyiş göstərməmişdir (0.08% vs 0.09% söz; 0.24% vs 0.26% sətir)."
)
for r in doc.paragraphs[3].runs:
    set_run_font(r, name="Calibri", size=10, italic=True)

# P4: Delete duplicate abstract paragraph
p4 = doc.paragraphs[4]
if "Tədqiqat çərçivəsində ilkin baza modeli" in p4.text:
    p4._p.getparent().remove(p4._p)

# ---------------------------------------------------------------------
# 2. Text Paragraphs Update (Matching by content)
# ---------------------------------------------------------------------
for p in list(doc.paragraphs):
    txt = p.text.strip()
    
    # Sec 1.2: Replace character segmentation claim
    if txt.startswith("Ənənəvi OCR alətləri (məsələn, Tesseract OCR) xətt seqmentasiyası"):
        p.text = (
            "Müasir Tesseract 4 və 5 mühərrikləri ənənəvi simvol seqmentasiyasından imtina edərək bütöv sətir üzərində işləyən "
            "LSTM xətt tanıyıcısından (line recognizer) istifadə edir. Layihəmizdə təklif olunan CRNN (CNN+BiLSTM) arxitekturası da "
            "həm söz, həm sətir təsvirlərini əvvəlcədən simvol seqmentasiyası tələb etmədən bütöv ardıcıllıq kimi emal edir və "
            "xətti qrafem kəsim xətalarının qarşısını alır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 1.3: 7-pilləli -> 6-pilləli
    elif txt.startswith("7-Pilləli Simulyasiya Nərdivanı (L0–L4n):"):
        p.text = "6-Pilləli Simulyasiya Nərdivanı (L0–L4n): Modellərin dözümlülüyünün təcrid olunmuş faktorlar üzrə ölçülməsi."
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 1.3: eyni sənədin iki fotosu
    elif txt.startswith("Real Dünya Sənəd Verifikasiyası: İki fərqli real smartfon"):
        p.text = "Real Dünya Sənəd Verifikasiyası: Eyni sənədin iki fotosu üzərində V1, V2 və Tesseract 5 modellərinin müqayisəli qiymətləndirilməsi."
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 2.1: Lexical split
    elif "cümlə və söz səviyyəsində qəti bölünmüş" in txt or "text leakage tam alınmışdır" in txt:
        p.text = (
            "Sintetik verilənlərin formalaşdırılmasında təlim və test dəstləri arasında leksik sızmanın təhlili aparılmışdır. "
            "Sətir korpusunda 294 test sətrinin heç biri təlim dəstində təkrar olunmur (0/294). Söz korpusunda isə test söz tokenlərinin "
            "22.4%-i (276 / 1232) təlim lüğətində rast gəlinir; bu, dilin təbii leksik kəsişməsidir (sətir korpusu sətirlər üzrə, "
            "söz korpusu isə unikal söz tipləri üzrə bölünmüşdür). Yalnız V2-yə əlavə olunan 15,000 sözün 20%-i (3,000 söz) rəqəm və "
            "durğu işarələrini ehtiva edir; ilkin V1 təlim korpusunda rəqəm və durğu ümumiyyətlə olmamışdır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 4.1 Intro sentence
    elif txt.startswith("Modellərin müstəqil sintetik test çoxluqları üzərində sınağı V2-nin həm V1"):
        p.text = (
            "Standart sintetik testlərdə V2 modeli yalnız Consolas şriftində V1-dən irəlidədir (ailə sızması hesabına), Tahoma şriftində "
            "V1 ilə eyni səviyyədə qalır (0.08% vs 0.09% söz; 0.24% vs 0.26% sətir), Tahoma sətirlərində isə Tesseract V2-dən irəlidədir "
            "(0.22% vs 0.26%)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 4.2 Title
    elif txt.startswith("4.2. 7-Pilləli Simulyasiya Nərdivanı"):
        p.text = "4.2. 6-Pilləli Simulyasiya Nərdivanı (Simulation Ladder L0–L4n)"
        for r in p.runs:
            set_run_font(r, size=11, bold=True)

    # Sec 4.2 Caption
    elif txt.startswith("Cədvəl 4."):
        p.text = "Cədvəl 4. 4-Şriftli Simulyasiya Nərdivanında (cambria.ttc, micross.ttf — 2 yeni ailə; pala.ttf, CascadiaMono.ttf — 2 yaxın qohum) Söz və Sətir CER (%) dinamikası. Qeyd: V1 sətir CER göstəricisi L3-də (19.14%) L4-dən (12.96%) yüksəkdir; bu qeyri-monotonluğun səbəbi araşdırılmamışdır."
        for r in p.runs:
            set_run_font(r, size=9.5, italic=True)

    # Sec 5.1: Real Document 1 delete 5.26% claim
    elif "5.26% fərqlə üstələmişdir!" in txt:
        p.text = (
            "Birinci sınaqda kamera bucaq altında yerləşmiş və təbii kölgələr mövcud olmuşdur. Sətir səviyyəsində (15 sətir, 762 simvol) "
            "V1 CER 73.49%-dən V2-də 28.87%-ə enmişdir (Tesseract: 21.92%). Kəsilmiş 30 söz kəsimində isə koordinatların qeyri-dəqiq olması "
            "və qonşu simvolların kəsimə düşməsi səbəbindən söz səviyyəli nəticələr etibarsız sayılır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 5.2: Real Document 2
    elif txt.startswith("İkinci foto perpendikulyar və kölgəsiz çəkilmişdir."):
        p.text = (
            "İkinci foto (media_1793194879421.jpg) perpendikulyar və təmiz işıq altında çəkilmişdir. Düzəldilmiş TƏQDİMAT kəsimi ilə (x=300..485) "
            "Tesseract 5 sətir səviyyəsində demək olar ki, qüsursuz nəticə nümayiş etdirmişdir: Sətir CER 0.24% (cəmi 2 / 837 hərf xətası: 'Vöb-saytın' və 'lurnal'), "
            "Sətir WER 2.25% (21 sətirdən 19-u 100% tam dəqiq). V2 modeli sətir səviyyəsində 5.02% CER və 22.47% standart WER əldə etmişdir "
            "(7 sətir tam raw bərabər, 9 sətir kənar boşluqsuz bərabər). Söz səviyyəsində V2 CER 46.71% vs Tesseract 50.99% təşkil edir. "
            "Kənar mürəkkəb analizi göstərmişdir ki, 30 söz kəsiminin 18-də (60.0%) qonşu sözlərin mürəkkəb müdaxiləsi (neighbour intrusion >15px) "
            "mövcuddur; bu səbəbdən söz səviyyəsində V2-nin Tesseract-dan irəlidə olması ilkin (preliminary) xarakter daşıyır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 5.3 Observation
    elif txt.startswith("Müşahidə: V2 modeli hətta sənədin rəsmi künc ştampını"):
        p.text = (
            "Müşahidə: V2 modeli sənədin rəsmi künc ştampını ('Naxçıvan Dövlət Universiteti') dəqiq tanımışdır (kənardakı tək boşluq xaric). "
            "Tesseract isə 'jurnal' sözünü 'lurnal', 'Veb-saytın' sözünü 'Vöb-saytın' kimi tanımış, lakin bütövlükdə sənəd sətirlərində 0.24% CER ilə "
            "V2-dən (5.02%) qat-qat yüksək ümumi sabitlik nümayiş etdirmişdir."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.1 Heading
    elif txt == "6.1. Şrift Ailəsi Sızması (Font Leakage Auditi)":
        p.text = "6.1. Şrift Ailəsi Sızması və Ümumiləşmə Təhlili (Groups A, B, C Auditi)"
        for r in p.runs:
            set_run_font(r, size=11, bold=True)

    # Sec 6.1 Content & Group Table placeholder
    elif txt.startswith("Elmi dürüstlük prinsiplərinə uyğun olaraq, V2 təlim korpusuna əlavə olunmuş"):
        p.text = (
            "Audit 4 çərçivəsində şriftlər 3 qrupa ayrılaraq tam klaster-bootstrap metodologiyası (95% CI) ilə qiymətləndirilmişdir:\n"
            "• Qrup A (Seen, 4 şrift: arial, calibri, segoeui, times): Təlimdə tam iştirak edən baza ailələr.\n"
            "• Qrup B (Near-Relative, 5 şrift: pala, CascadiaMono, verdanab, seguisb, calibrib): Təlim şriftləri ilə ailə qohumluğu olan (leakage) şriftlər.\n"
            "• Qrup C (New Families, 7 şrift: cambria, micross, tahomabd, Moba_Fanta, Moba_Font, Moba_Fira, SansSerifCollection): Təlimdə heç bir forması iştirak etməyən tamamilə yeni şrift ailələri.\n\n"
            "Aşağıdakı Cədvəl 7-də bütün qruplar üzrə L0 və L4 səviyyələrində V1, V2 və Tesseract 5 göstəriciləri, o cümlədən model fərqinin (V2 - Tess) "
            "klaster-bootstrap 95% etibarlılıq intervalları təqdim olunur:"
        )
        for r in p.runs:
            set_run_font(r, size=10)
        p_sec61 = p

    # Sec 6.2 Diacritic Analysis
    elif txt.startswith("'ı' -> 'i' qarışması: V1-də 14.2%"):
        p.text = (
            "Audit 3 və Audit 4-ün çaşqınlıq matrisi analizi göstərir ki, V2 modelində 'ə' -> 'e' xətası 0.00% təşkil edir (0 / 394 instansiya). "
            "Dotless-i (ı/i/İ) üzrə V2 modelində qeydə alınan cəmi 106 xətanın analizi (results/v2_audit4/summary.md:141-174):\n"
            "• Birbaşa 'ı' <-> 'i' əvəzləməsi cəmi 5 halda baş vermişdir (4 ədəd 'i->ı', 1 ədəd 'ı->i').\n"
            "• Əsas xətalar 'i' -> 'ü' (58 hal) və 'ı' -> 'u' (20 hal) qarışıqlığından ibarətdir.\n"
            "• Bu xətaların 91.5%-i (97 / 106) cəmi 3 şriftdə cəmlənmişdir: Moba_Fanta (53 hal / 50.0%), Moba_Font (30 hal / 28.3%) və "
            "SansSerifCollection (14 hal / 13.2%). Əvvəlki layihə qaralamasında qeyd olunan 'ı->i 14.2% -> 0.8%' və 'ə->e 8.7% -> 0.4%' iddiaları "
            "təsdiqlənməmişdir (mənbəsizdir)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.2 Rare cases
    elif txt.startswith("'ə' -> 'e' qarışması: V1-də 8.7%"):
        p.text = (
            "Nadir Diakritik Xətalar (results/v2_audit4/summary.md:175-195):\n"
            "• ö -> o (2 hal): ref 'Köpək' -> hyp 'Kopək' (SansSerifCollection.ttf, Word L4); ref 'köynək-baju' -> hyp 'koynək.baju' (SansSerifCollection.ttf, Word L4).\n"
            "• o -> ə (1 hal): ref 'Goranboy,' -> hyp 'Gəranboy,' (cambria.ttc, Word L4).\n"
            "• c -> e (2 hal): ref 'dcmokratik' -> hyp 'demokratik' (cambria.ttc, Word L4 və Line L4).\n"
            "• ç -> s (1 hal): ref 'qaçan' -> hyp 'qasan' (verdanab.ttf, Word L4).\n"
            "• G -> C (2 hal): ref 'Gürcüstan' -> hyp 'Cürcüstan' (pala.ttf, Line L0 və Line L4)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.2 WER space error intro
    elif txt.startswith("Durğu işarələri: V2 sətir modelində sətir sonundakı tək nöqtə"):
        p.text = (
            "Durğu işarələri və CTC boşluq xətaları: V2 sətir modelində sətir sonundakı tək nöqtə simvolu bəzən '..' kimi dekodlaşdırılır "
            "(məsələn, 'hazırlanması..', 'qurulması..').\n"
            "Sətir WER Boşluq Təhlili (Groups B & C, eyni 1076 söz üzrə xəta növləri dekompozisiyası, results/v2_audit4/summary.md:205-212):\n"
            "V2 modelinin sətir söz xətalarının böyük hissəsi (L0-da 79.3%, L4-də 65.3%) xalis boşluq xətalarından (həm sözlərin birləşməsi, "
            "həm də söz daxilinə səhvən boşluq daxil edilməsi) qaynaqlanır. Tesseract-da isə boşluq xətası demək olar ki, yoxdur (0.0% – 5.9%)."
        )
        for r in p.runs:
            set_run_font(r, size=10)
        p_sec62 = p

    # Sec 7 Heading
    elif txt == "7. Nəticə və Gələcək Tədqiqatlar":
        p.text = "7. Tədqiqat Məhdudiyyətləri, Nəticə və Gələcək İstiqamətlər"
        for r in p.runs:
            set_run_font(r, size=12, bold=True)

    # Sec 7 Limitations
    elif txt.startswith("Bu monoqrafiyada təqdim olunan Azərbaycan dili üçün CRNN+CTC sistemi"):
        p.text = (
            "Tədqiqat Məhdudiyyətləri (Reproducibility & Scope Limitations):\n"
            "1. Render skriptinin provenansı: V2 verilənlər bazasına əlavə olunan 15,000 söz və 6,000 sətir scratch/build_v2_datasets.py:111, 153 "
            "skripti ilə generate_data.py:50-54 TRDG generatoru (GeneratorFromStrings, size=64 söz / size=48 sətir, bgs=[0,1], fonts_new) vasitəsilə generasiya edilmişdir.\n"
            "2. Val dəstinin məhdudiyyəti: Val dəsti yalnız V1-in 8 təlim şriftini ehtiva edir (synth_word_v2/val və synth_line_v2/val); "
            "modelin ən yaxşı checkpoint-i yalnız görünən şriftlər üzrə seçilmişdir (unseen generalizasiya üzrə deyil).\n"
            "3. Təlim itkisinin qeyri-sabitliyi: V2 sətir modelində təlimin son epoxlarında kəskin val-loss sıçrayışları "
            "(epox 9-da 0.72, epox 13-də 0.65, epox 15-də 0.48) baş vermiş və model 10-cu epoxdan sonra erkən dayandırılmışdır (patience=5).\n"
            "4. Foto 2 kəsimləri və etalon mənşəyi: Sənəd 2 kəsim qutuları Tesseract image_to_data çıxarışının təhlilindən sonra "
            "test_real_doc2.py:46-135 faylına əllə daxil edilmişdir (müstəqil deyil). Etalon mətnlər sənəd fotosundan əllə transkripsiya "
            "edilmişdir (müstəqil cüt yoxlama sənədləşdirilmədiyi üçün status: 'təsdiq gözləyir'). 21 sətir kəsiminin 17-də (81.0%) 2px marja "
            "daxilində şaquli kənarda davam edən mürəkkəb mövcuddur; 4-6px marja və ağ maskalama sınağı isə Otsu binarizasiyasını pozaraq "
            "(V2 9.08%, Tess 29.27%) nəticə verməmişdir, buna görə sətir CER göstəricisi kəsim keyfiyyəti baxımından tam təsdiqlənməmişdir.\n"
            "5. Söz kəsimi və qonşu müdaxiləsi: 30 söz kəsiminin 18-də (60.0%) qonşu sözlərin mürəkkəb müdaxiləsi (neighbour intrusion >15px) "
            "mövcuddur; kəsilmiş söz benchmarkı etibarsız sayılır.\n"
            "6. Real sınaq həcminin məhdudluğu: Yalnız eyni bir sənədin 2 fotoşəkli sınaqdan keçirilmişdir; 20-30 foto və ya 100+ sətirlik "
            "genişmiqyaslı real sənəd testi hələ aparılmamışdır.\n"
            "7. Təhrifli və qəliz şriftlərdə Tesseract-ın üstünlüyü: Tesseract 5 həm real sənəd sətirlərində (0.24% vs 5.02%), həm nərdivan L4 "
            "sözlərində (0.37% vs 1.35%), həm Tahoma sətirlərində (0.22% vs 0.26%), həm də qəliz əl/dekorativ şriftlərdə (Inkfree L0 söz: "
            "Tesseract 14.20% vs V2 30.69%) layihə modelindən üstündür.\n"
            "8. Test kəsimlərində generator xətası: Sintetik test kəsimlərinin bir hissəsində (24/1049 kəsim, məs. 'və' sözündə 11.8% mürəkkəb itkisi) "
            "TRDG generatorunun kənarları kəsmə xətası qeydə alınmışdır. Xüsusi hərflərin seçmə həcmi kiçikdir (ç üçün n=23, ş üçün n=56).\n"
            "9. Rəqəm və durğu işarələri: Yalnız V2-yə əlavə olunan 15,000 sözün 20%-i (3,000 söz) rəqəm və durğu işarələrini ehtiva edir; "
            "ilkin V1 təlim korpusunda rəqəm və durğu ümumiyyətlə olmamışdır."
        )
        for r in p.runs:
            set_run_font(r, size=9.5)

    # Sec 7 Language Model: endirəcəkdir -> endirə bilər
    elif "Dil Modeli İnteqrasiyası:" in txt:
        p.text = (
            "Dil Modeli İnteqrasiyası: Greedy Best-Path əvəzinə CTC Beam Search alqoritmi ilə birləşdirilmiş n-gram və ya Transformer "
            "əsaslı Azərbaycan dili lüğət korreksiyasının qoşulması sətir CER-ni 5%-dən 1%-in altına endirə bilər."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Reference 1 update year to 2017
    elif txt.startswith("1. Shi, B."):
        p.text = (
            "1. Shi, B., Bai, X., & Yao, C. (2017). An end-to-end trainable neural network for image-based sequence recognition and its application to "
            "scene text recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(11), 2298-2304."
        )
        for r in p.runs:
            set_run_font(r, size=9.5)

    # Reference 5 remove
    elif txt.startswith("5. Memmedov, S."):
        p._p.getparent().remove(p._p)

# ---------------------------------------------------------------------
# 3. Tables 1, 2, 3, 5, 6 Updates
# ---------------------------------------------------------------------
# Table 1: Augmentations
t1 = doc.tables[0]
while len(t1.rows) < 12:
    t1.add_row()

t1_headers = ["Parametr / Təhrif Növü", "V1 Modeli (train.py:21-40)", "V2 Modeli (train_v2.py:26-47)"]
for c_idx, h in enumerate(t1_headers):
    format_cell(t1.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF")

t1_rows = [
    ["Fırlanma Bucağı (Rotation)", "± 5° fırlanma (p=0.5) [train.py:25]", "± 5° fırlanma (p=0.5) [train_v2.py:27-28]"],
    ["Perspektiv Əyilmə (Perspective)", "± 2% en, ± 8% hündürlük (p=0.3) [train.py:35-39]", "± 2% en, ± 8% hündürlük (p=0.3) [train_v2.py:29]"],
    ["Qauss Bulanığı (Gaussian Blur)", "k ∈ {3, 5} (p=0.3) [train.py:28-29]", "σ ∈ [0.5, 2.0] (p=0.6) [train_v2.py:30-31]"],
    ["Duz-İstiot Küyü (Salt & Pepper)", "0.5% piksel (p=0.3) [train.py:30-34]", "p=0.2 [train_v2.py:36]"],
    ["Hərəkət Bulanığı (Motion Blur) [YENİ]", "Yoxdur", "kernel 3–7 (p=0.15) [train_v2.py:32-33]"],
    ["Sensor/Qauss Küyü (Sensor Noise) [YENİ]", "Yoxdur", "σ ∈ [3.0, 12.0] (p=0.6) [train_v2.py:34-35]"],
    ["Rezolyusiya İtkisi (Downscale) [YENİ]", "Yoxdur", "faktor 0.5× – 0.9× (p=0.4) [train_v2.py:37-38]"],
    ["JPEG Sıxılma Artefaktı [YENİ]", "Yoxdur", "Q ∈ [40, 90] (p=0.4) [train_v2.py:39-40]"],
    ["Parlaqlıq və Kontrast [YENİ]", "Yoxdur", "α ∈ [0.6, 1.1], β ∈ [-40, 20] (p=0.6) [train_v2.py:41-43]"],
    ["İşıq Qradiyenti və Kölgə Zolaqları [YENİ]", "Yoxdur", "qradiyent 0.7 – 1.0 (p=0.5) [train_v2.py:44-45]"],
    ["Kontrast Dartılması [YENİ]", "Yoxdur", "p=0.5 [train_v2.py:46]"],
]
for r_idx, row_data in enumerate(t1_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(t1.rows[r_idx].cells[c_idx], val)

# Table 2: Training parameters
t2 = doc.tables[1]
t2_headers = ["Model", "Ümumi Epox", "Ən Yaxşı Epox", "Val Loss", "Val CER (%)*", "Təlim Vaxtı", "Checkpoint Ölçüsü"]
for c_idx, h in enumerate(t2_headers):
    format_cell(t2.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF")

t2_rows = [
    ["V2 Word", "18 (Early Stop, pat=5)", "Epox 13", "0.0063 [history.csv:14]", "0.24% (Seen fonts)*", "212.3 dəq ümumi (154.1 dəq @ best)", "98.7 MB (xalis: 32.9 MB)"],
    ["V2 Line", "15 (Early Stop, pat=5)", "Epox 10", "0.0125 [history.csv:11]", "0.28% (Seen fonts)*", "280.4 dəq ümumi (186.9 dəq @ best)", "98.7 MB (xalis: 32.9 MB)"],
]
for r_idx, row_data in enumerate(t2_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(t2.rows[r_idx].cells[c_idx], val)

# Table 3: Synthetic test sets
t3 = doc.tables[2]
t3_headers = ["Dataset / Test Çoxluğu", "Dəqiq Şrift Tərkibi və Təsnifatı", "V1 Modeli CER", "V2 Modeli CER", "Tesseract 5 CER", "V1 -> V2 Dəyişimi"]
for c_idx, h in enumerate(t3_headers):
    format_cell(t3.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF")

t3_rows = [
    ["Söz Testi (word_synth)", "1,001 tahoma (New) + 999 consola (Leakage)", "0.50%", "0.11%", "0.70%", "CER 4.55x azaldı (Qazanc yalnız Consolas sızmasındandır, Tahoma dəyişməyib)"],
    ["Sətir Testi (line_synth)", "501 tahoma (New) + 499 consola (Leakage)", "1.64%", "0.30%", "0.44%", "CER 5.47x azaldı (Qazanc yalnız Consolas sızmasındandır, Tahoma dəyişməyib)"],
    ["Söz, Tahoma (diagw_tahoma)", "1,000 tahoma.ttf — NEW FAMILY", "0.08%", "0.09%", "0.50%", "Dəyişmədi (+0.01% dalğalanma)"],
    ["Sətir, Tahoma (diag_tahoma)", "500 tahoma.ttf — NEW FAMILY", "0.24%", "0.26%", "0.22%", "Dəyişmədi (+0.02%, Tesseract öndədir)"],
    ["Söz, Consolas (diagw_consola)", "1,000 consola.ttf — FAMILY LEAKAGE (consolab)", "1.02%", "0.07%", "0.86%", "CER 14.57x azaldı (Ailə sızması)"],
    ["Sətir, Consolas (diag_consola)", "500 consola.ttf — FAMILY LEAKAGE (consolab)", "3.48%", "0.27%", "0.47%", "CER 12.89x azaldı (Ailə sızması)"],
]
for r_idx, row_data in enumerate(t3_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(t3.rows[r_idx].cells[c_idx], val)

# Table 4: Simulation Ladder (Replace with 8-column table)
t4 = doc.tables[3]
t4_parent = t4._tbl.getparent()
t4_new = doc.add_table(rows=7, cols=8)
t4_new.alignment = WD_TABLE_ALIGNMENT.CENTER
t4._tbl.addnext(t4_new._tbl)
t4_parent.remove(t4._tbl)

t4_headers = ["Pillə", "Təhrif Növü", "V1 Word CER", "V2 Word CER", "Tess Word CER", "V1 Line CER", "V2 Line CER", "Tess Line CER"]
for c_idx, h in enumerate(t4_headers):
    format_cell(t4_new.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF", size=8.0)

t4_rows = [
    ["L0", "Təmiz render (Sıfır təhrif) [results/v2_audit/summary.md:66]", "7.71%", "0.20%", "0.41%", "1.87%", "0.11%", "0.22%"],
    ["L1", "İşıqlanma qradiyenti və kölgə [summary.md:68]", "8.85%", "0.20%", "0.45%", "4.10%", "0.11%", "0.22%"],
    ["L2", "Bulanıqlıq və sensor küyü [summary.md:69]", "25.00%", "0.41%", "0.41%", "13.84%", "0.07%", "0.29%"],
    ["L3", "Ölçü kiçiltmə (0.6x) və JPEG [summary.md:70]", "31.36%", "0.53%", "0.41%", "19.14%", "0.18%", "0.22%"],
    ["L4", "Tam kamera simulyasiyası [summary.md:71]", "35.77%", "1.35%", "0.37%", "12.96%", "0.11%", "0.33%"],
    ["L4n", "L4 + Kontrast dartılması [summary.md:72]", "24.43%", "1.14%", "0.37%", "7.43%", "0.07%", "0.33%"],
]
for r_idx, row_data in enumerate(t4_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(t4_new.rows[r_idx].cells[c_idx], val, size=8.0)

# Table 5: Real Document Test 2
t5 = doc.tables[4]
t5_headers = ["Kateqoriya", "Model", "CER (%)", "WER (%) [Standart]", "100% Dəqiq Tanınan Sayı / Cəmi"]
for c_idx, h in enumerate(t5_headers):
    format_cell(t5.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF")

t5_rows = [
    ["Sətirlər (21 sətir, 837 simvol)", "V1 Line", "48.51%", "98.88% (88/89)", "0 / 21 (0.0%)"],
    ["", "V2 Line", "5.02%", "22.47% (20/89)", "7 / 21 tam raw (9/21 strip)"],
    ["", "Tesseract 5", "0.24%", "2.25% (2/89)", "19 / 21 tam qüsursuz (90.5%)"],
    ["Sözlər (30 söz, 304 simvol)** [ETİBARSIZ]", "V1 Word", "90.13%", "100.00% (30/30)", "0 / 30 (0.0%) [Neighbour intrusion: 18/30]"],
    ["", "V2 Word", "46.71%", "90.00% (27/30)", "3 / 30 (10.0%) [Neighbour intrusion: 18/30]"],
    ["", "Tesseract 5", "50.99%", "146.67% (44/30)†", "4 / 30 (13.3%) [Neighbour intrusion: 18/30]"],
]
for r_idx, row_data in enumerate(t5_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(t5.rows[r_idx].cells[c_idx], val)

# Table 6: Real Transcriptions
t6 = doc.tables[5]
t6_headers = ["Orijinal Mətn (Ground Truth)", "V1 Line Modeli", "V2 Line Modeli", "Tesseract 5 (aze)"]
for c_idx, h in enumerate(t6_headers):
    format_cell(t6.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF")

t6_rows = [
    ["TƏQDİMAT (Başlıq, x=300..485)", "jirəggitifTijj", " TəqDiMAT ı", "TƏQDİMAT (100% Dəqiq)"],
    ["hazırlanması.", "ihagiianmasiji", "hazırlanması.. (Nöqtə təkrarlanması)", "hazırlanması."],
    ["Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi", "RiBlaktonijürmaiidaraatmaisistamin...", "Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi (100%)", "Elektron lurnal idarəetmə sisteminin..."],
    ["Nəşr və xidmət haqları barədə məlumatların ayrıca təqdim edilməsi.", "iHaşrijaixidməthağlaribarəğamalüin...", "Nəşr və xldmət haqları barədə məlumatların... ('i'->'l')", "Nəşr və xidmət haqları barədə məlumatların..."],
    ["Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi", "y6bsaytinidizaynininipaistifadəği...", "y6b-saytın dizaynının və İstifadəçi interfeysinin yenilənməsi", "Vöb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi"],
]
for r_idx, row_data in enumerate(t6_rows, start=1):
    if r_idx < len(t6.rows):
        for c_idx, val in enumerate(row_data):
            format_cell(t6.rows[r_idx].cells[c_idx], val)

# ---------------------------------------------------------------------
# 4. Insert Table 7 (Groups A, B, C) and Table 8 (WER Table)
# ---------------------------------------------------------------------
# Create Table 7
table7 = doc.add_table(rows=13, cols=9)
table7.alignment = WD_TABLE_ALIGNMENT.CENTER
t7_headers = ["Qrup", "Rejim və Səviyyə", "Nümunə (n)", "Simvol", "V1 CER (%)", "V2 CER [Cluster 95% CI]", "Tess CER [Cluster 95% CI]", "Δ (V2 - Tess) [Cluster 95% CI]", "0 CI-yə Daxildir?"]
for c_idx, h in enumerate(t7_headers):
    format_cell(table7.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF", size=7.5)

t7_rows = [
    ["Qrup A", "Word L0", "378", "2472", "7.36% [results/v2_audit4/summary.md:120]", "0.81% [0.32%, 1.57%]", "2.39% [1.39%, 3.83%]", "-1.58% [-2.68%, -0.76%]", "XEYR (V2 öndədir)"],
    ["Qrup A", "Line L0", "95", "2777", "0.29% [summary.md:121]", "0.11% [0.00%, 0.30%]", "0.29% [0.00%, 0.65%]", "-0.18% [-0.57%, +0.11%]", "BƏLİ (Bərabər)"],
    ["Qrup A", "Word L4", "378", "2472", "45.27% [summary.md:122]", "1.33% [0.58%, 2.20%]", "1.70% [0.87%, 2.84%]", "-0.36% [-1.25%, +0.50%]", "BƏLİ (Bərabər)"],
    ["Qrup A", "Line L4", "95", "2777", "13.83% [summary.md:123]", "1.15% [0.18%, 3.00%]", "1.62% [0.11%, 4.09%]", "-0.47% [-1.91%, +0.44%]", "BƏLİ (Bərabər)"],
    ["Qrup B", "Word L0", "436", "3012", "9.56% [summary.md:124]", "0.46% [0.07%, 0.97%]", "0.96% [0.30%, 1.89%]", "-0.50% [-1.28%, +0.09%]", "BƏLİ (Bərabər)"],
    ["Qrup B", "Line L0", "114", "3368", "2.35% [summary.md:125]", "0.30% [0.00%, 0.92%] ", "0.12% [0.00%, 0.36%]", "+0.18% [-0.31%, +0.88%]", "BƏLİ (Bərabər)"],
    ["Qrup B", "Word L4", "436", "3012", "41.33% [summary.md:126]", "0.73% [0.27%, 1.31%]", "1.10% [0.39%, 2.08%]", "-0.37% [-1.26%, +0.50%]", "BƏLİ (Bərabər)"],
    ["Qrup B", "Line L4", "114", "3368", "16.12% [summary.md:127]", "0.50% [0.06%, 1.17%]", "0.18% [0.00%, 0.44%]", "+0.33% [-0.18%, +1.06%]", "BƏLİ (Bərabər)"],
    ["Qrup C", "Word L0", "609", "4194", "9.73% [summary.md:128]", "1.62% [0.22%, 3.44%]", "1.00% [0.36%, 1.78%]", "+0.62% [-0.35%, +2.08%]", "BƏLİ (Bərabər)"],
    ["Qrup C", "Line L0", "158", "4673", "6.03% [summary.md:129]", "0.71% [0.08%, 1.68%]", "0.71% [0.19%, 1.46%]", "+0.00% [-0.74%, +0.76%]", "BƏLİ (Tam bərabər)"],
    ["Qrup C", "Word L4", "609", "4194", "41.49% [summary.md:130]", "2.36% [1.23%, 3.95%]", "0.81% [0.37%, 1.31%]", "+1.55% [+0.59%, +2.94%]", "XEYR (Tess öndədir)"],
    ["Qrup C", "Line L4", "158", "4673", "20.93% [summary.md:131]", "1.28% [0.16%, 2.75%]", "1.46% [0.37%, 3.72%]", "-0.17% [-2.60%, +1.53%]", "BƏLİ (Bərabər)"],
]
for r_idx, row_data in enumerate(t7_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(table7.rows[r_idx].cells[c_idx], val, size=7.5)

# Insert Table 7 right after p_sec61
p_sec61._p.addnext(table7._tbl)

# Add interpretive note below Table 7 via oxml
note_p_xml = (
    f'<w:p {nsdecls("w")}><w:pPr><w:jc w:val="both"/></w:pPr>'
    f'<w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="18"/><w:i/></w:rPr>'
    f'<w:t xml:space="preserve">Cədvəl 7 üzrə Mühüm Elmi Qeydlər: '
    f'1. Qrup A Word L0 üzrə V2 CER (0.81%, 20/2472 hərf) ilə Qrup C Word L4 üzrə Tesseract CER (0.81%, 34/4194 hərf) göstəricilərinin eyni olması sırf rəqəmsal təsadüfdür; hər iki göstərici yoxlanmış birbaşa saylardan qaynaqlanır. '
    f'2. Qrup C üzrə 4 müqayisədən yalnız 1-də (Word L4) fərq intervalı sıfırı əhatə etmir (+1.55% [+0.59%, +2.94%]); bu pillədə Tesseract V2 modelini birmənalı üstələyir. Digər 3 halda modellər statistik bərabərdir. '
    f'3. Qrup A (k=4) və Qrup B (k=5) klaster sayları az olduğu üçün bu qrupların intervalları yüksək dispersiyaya malikdir. Moba_Fira monospace şriftidir və təlimdə CascadiaCode/consolab iştirak etmişdir.</w:t>'
    f'</w:r></w:p>'
)
p_t7_note_elem = parse_xml(note_p_xml)
table7._tbl.addnext(p_t7_note_elem)

# Create WER Table
table_wer = doc.add_table(rows=5, cols=8)
table_wer.alignment = WD_TABLE_ALIGNMENT.CENTER
t_wer_headers = ["Səviyyə və Model", "Ümumi Söz (n)", "Xalis Boşluq Xətası", "Tək Hərf Xətası", "Çoxhərfli Xəta", "Qeyri-Boşluq Xətası", "Raw WER (%)", "Boşluq Xətasının Payı (%)"]
for c_idx, h in enumerate(t_wer_headers):
    format_cell(table_wer.rows[0].cells[c_idx], h, bold=True, bg_color="EAECEF", size=8.0)

t_wer_rows = [
    ["L0 V2 Modeli", "1076", "46 (4.3%)", "8 (0.7%)", "4 (0.4%)", "12 (1.1%)", "5.39%", "79.3%"],
    ["L0 Tesseract 5", "1076", "0 (0.0%)", "22 (2.0%)", "7 (0.7%)", "29 (2.7%)", "2.70%", "0.0%"],
    ["L4 V2 Modeli", "1076", "64 (5.9%)", "26 (2.4%)", "8 (0.7%)", "34 (3.2%)", "9.11%", "65.3%"],
    ["L4 Tesseract 5", "1076", "2 (0.2%)", "23 (2.1%)", "9 (0.8%)", "32 (3.0%)", "3.16%", "5.9%"],
]
for r_idx, row_data in enumerate(t_wer_rows, start=1):
    for c_idx, val in enumerate(row_data):
        format_cell(table_wer.rows[r_idx].cells[c_idx], val, size=8.0)

# Insert WER Table right after p_sec62
p_sec62._p.addnext(table_wer._tbl)

doc.save(DST_DOCX)
print(f"Successfully finalized clean docx v2 at: {DST_DOCX}")
