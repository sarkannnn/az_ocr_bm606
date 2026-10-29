import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

SRC_DOCX = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER_v3.docx")
DST_DOCX = Path("results/Az_OCR_Akademik_Hesabat_DUZELISLER_v4.docx")

shutil.copy2(SRC_DOCX, DST_DOCX)
doc = docx.Document(DST_DOCX)
print(f"Copied {SRC_DOCX} -> {DST_DOCX}")

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
# 1. Update Paragraphs
# ---------------------------------------------------------------------
for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()

    # Sec 1.2: Remove or update Tesseract 4/5 citation
    if txt.startswith("Müasir Tesseract 4 və 5 mühərrikləri ənənəvi simvol seqmentasiyasından"):
        p.text = (
            "Müasir Tesseract 4 və 5 mühərrikləri ənənəvi simvol seqmentasiyasından imtina edərək "
            "bütöv sətir üzərində işləyən LSTM xətt tanıyıcısından (line recognizer) istifadə edir (Tesseract OCR, 2018). "
            "Layihəmizdə təklif olunan CRNN (CNN+BiLSTM) arxitekturası da həm söz, həm sətir təsvirlərini "
            "əvvəlcədən simvol seqmentasiyası tələb etmədən bütöv ardıcıllıq kimi emal edir və xətti qrafem kəsim xətalarının qarşısını alır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 1.3: Add digits and punctuation absent from V1 training corpus
    elif txt.startswith("Azərbaycan Dili üçün İxtisaslaşmış CRNN Modeli: 87 simvollu lüğət"):
        p.text = (
            "Azərbaycan Dili üçün İxtisaslaşmış CRNN Modeli: 87 simvollu lüğət (32 kiçik, 32 böyük hərf, rəqəmlər, durğu işarələri və blank) "
            "üçün sıfırdan layihələndirilmiş, həm söz (32x128), həm sətir (32x512) səviyyəli yüngülçəkili end-to-end model "
            "(qeyd edilməlidir ki, ilkin V1 modelinin söz təlim korpusunda rəqəmlər və durğu işarələri ümumiyyətlə olmamışdır)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 2.1: Text leakage update (0/294 lines, 22.4% word tokens, split rules)
    elif txt.startswith("Modelin təlimində Azərbaycan dilli Vikipediya məqalələrindən"):
        p.text = (
            "Modelin təlimində Azərbaycan dilli Vikipediya məqalələrindən və rəsmi xəbər portallarından toplanmış zəngin korpusdan istifadə edilmişdir. "
            "Məlumat sızmasının (leakage) qarşısını almaq üçün sətir korpusu birbaşa sətirlər üzrə (split by lines) bölünmüşdür və test dəstindəki "
            "294 sətrin heç biri (0/294, 0.0%) təlim korpusunda iştirak etmir. Söz korpusu isə unikal söz tipləri (word types) üzrə bölünmüşdür; "
            "bununla belə, test dəstindəki söz tokenlərinin 22.4%-i (276/1232 token) təbii dil kəsişməsi olaraq təlim lüğətində mövcuddur. "
            "Həmçinin balance_special funksiyası ilə diakritikli simvolları ehtiva edən sözlərin payı ən azı 50% səviyyəsində saxlanılmışdır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 4.1: Tahoma lines difference nominal, not statistically significant
    elif txt.startswith("Standart sintetik testlərdə V2 modeli yalnız Consolas"):
        p.text = (
            "Standart sintetik testlərdə V2 modeli yalnız Consolas şriftində V1-dən irəlidədir (ailə sızması hesabına), "
            "Tahoma şriftində V1 ilə eyni səviyyədə qalır (0.08% vs 0.09% söz; 0.24% vs 0.26% sətir); "
            "Tahoma sətirlərində fərq nominaldır, statistik əhəmiyyətli deyil (Tesseract 0.22% vs V2 0.26%, n=500)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 4.2: Replace real camera with simulated camera; replace daha dayanıqlı with nominal
    elif txt.startswith("Nəticə göstərir ki, V1 modeli"):
        p.text = (
            "Nəticə göstərir ki, V1 modeli simulyasiya olunmuş kamera şəraitində (L4) 35.77% CER ilə yararsız vəziyyətə düşdüyü halda, "
            "V2 modeli sözdə 1.35% CER qoruyub saxlayır (müqayisə üçün: Tesseract 5 bu pillədə 0.37% CER nümayiş etdirir). "
            "Sətir səviyyəsində isə L4-də V2 modelinin CER göstəricisi (0.11%) Tesseract 5-dən (0.33%) nominal olaraq daha aşağıdır "
            "(0.11% vs 0.33%, CI hesablanmayıb, 4 şrift)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 5.1: Photo-1 line crops automatic and line #4 reference mismatch -> preliminary
    elif txt.startswith("Birinci sınaqda kamera bucaq altında yerləşmiş"):
        p.text = (
            "Birinci sınaqda kamera bucaq altında yerləşmiş və təbii kölgələr mövcud olmuşdur. "
            "Foto 1-in 15 sətir kəsimi avtomatik icra edildiyi və 4-cü sətirdə etalon uyğunsuzluğu (reference mismatch) aşkarlandığı üçün "
            "sətir səviyyəli nəticələr (Tesseract 21.92% vs V2 28.87%, V1 73.49%) ilkin (preliminary) xarakter daşıyır. "
            "Kəsilmiş 30 söz kəsimində isə koordinatların qeyri-dəqiq olması və qonşu simvolların kəsimə düşməsi səbəbindən söz səviyyəli nəticələr etibarsız sayılır."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.1: Group A note update with error split (172 digit/punct vs 10 letters)
    elif "Audit 4 çərçivəsində şriftlər V2-nin təlim siyahısına nəzərən" in txt:
        p.text = (
            "Audit 4 çərçivəsində şriftlər V2-nin təlim siyahısına nəzərən 3 qrupa ayrılaraq tam klaster-bootstrap metodologiyası (95% CI) "
            "ilə qiymətləndirilmişdir:\n"
            "• Qrup A (Seen, 4 şrift: arial.ttf, calibri.ttf, segoeui.ttf, times.ttf): Həm V1, həm V2 təlimində tam iştirak edən baza şriftlər.\n"
            "• Qrup B (Near-Relative, 5 şrift: calibrib.ttf, CascadiaMono.ttf, pala.ttf, seguisb.ttf, verdanab.ttf): Təlim şriftləri ilə birbaşa ailə qohumluğu olan (leakage) şriftlər.\n"
            "• Qrup C (New Families, 7 şrift: cambria.ttc, micross.ttf, Moba_Fanta.ttf, Moba_Fira.ttf, Moba_Font.ttf, SansSerifCollection.ttf, tahomabd.ttf): Təlimdə heç bir forması iştirak etməyən tamamilə yeni şrift ailələri.\n\n"
            "Qeyd: Qrup A şriftləri (`arial`, `calibri`, `segoeui`, `times`) V1-in öz təlim şriftləridir. Buna baxmayaraq, V1-in Qrup A Word L0 səviyyəsində "
            "7.36% CER göstərməsinin dəqiq səbəbi xəta bölgüsündən aydın olur: 2,472 simvolda qeydə alınan 182 xətanın 172-si (94.5%-i, 110 əvəzetmə və 62 silinmə) "
            "rəqəmlər və durğu işarələrindən qaynaqlanır (çünki V1 söz təlim korpusunda rəqəm və durğu 0% olmuşdur). Hərf xətaları isə cəmi 10 ədəddir (5.5%, xalis hərf CER: 0.44%). "
            "Eyni şriftlərin sətir modelində (Line L0) isə V1 durğu və rəqəmlərlə təlim edildiyi üçün CER cəmi 0.29% təşkil edir.\n\n"
            "Cədvəl 7. Şrift qrupları (A, B, C) üzrə L0 və L4 səviyyələrində V1, V2 və Tesseract 5 CER müqayisəsi və klaster-bootstrap 95% etibarlılıq intervalları (Qruplar V2-nin təlim siyahısına nəzərən müəyyən edilmişdir):"
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.2: Diacritic confusion - remove previous draft claims, add Tesseract rates
    elif txt.startswith("Audit 3 və Audit 4-ün çaşqınlıq matrisi analizi göstərir ki, V2 modelində 'ə' -> 'e' xətası 0.00%"):
        p.text = (
            "Audit 3 və Audit 4-ün çaşqınlıq matrisi analizi göstərir ki, V2 modelində 'ə' -> 'e' xətası 0.00% təşkil edir (0 / 394 instansiya); "
            "müqayisə üçün Tesseract 5 bu simvol üzrə ~2% xəta dərəcəsi (L0-da 2.03% [8/394], L4-də 1.78% [7/394]) nümayiş etdirir. "
            "Dotless-i (ı/i/İ) üzrə V2 modelində söz səviyyəsində xəta dərəcəsi L0-da 6.52% (9/138) və L4-də 7.25% (10/138) təşkil edir; "
            "Tesseract 5-də isə bu göstərici 0.00% – 0.72% aralığındadır (L0-da 0.72% [1/138], L4-də 0.00% [0/138]). "
            "Sətir səviyyəsində V2-nin dotless-i xətası L0-da 0.72%, L4-də 2.17%-dir. "
            "V2 modelində dotless-i xətalarının əksəriyyəti birbaşa 'ı'<->'i' deyil, 'i'->'ü' (58 hal) və 'ı'->'u' (20 hal) qarışıqlığından ibarətdir "
            "və bu xətaların 91.5%-i (97/106) cəmi 3 dar/qeyri-adi şriftdə cəmlənmişdir (Moba_Fanta: 53, Moba_Font: 30, SansSerifCollection: 14)."
        )
        for r in p.runs:
            set_run_font(r, size=10)

    # Sec 6.2: Rare diacritics - reconcile with counts, separate other cases, note dcmokratik typo, add Tesseract
    elif txt.startswith("Nadir Diakritik Xətalar"):
        p.text = (
            "Nadir Diakritik Xətalar (Audit 4 çaşqınlıq cədvəlindən dəqiq saylar və sətirlər):\n"
            "• ö -> o xətaları (ö üçün cəmi n=32 instansiya, V2-də 2 xəta [6.25%]; Tesseract-da 0 xəta):\n"
            "  - ref: 'Köpək' -> hyp: 'Kopək' | Model: V2, Qrup: C, Şrift: SansSerifCollection.ttf, Səviyyə: L4, Rejim: word\n"
            "  - ref: 'köynək-baju' -> hyp: 'koynək.baju' | Model: V2, Qrup: C, Şrift: SansSerifCollection.ttf, Səviyyə: L4, Rejim: word\n"
            "  (Tesseract 5 hər iki sözü 'Köpək' və 'köynək-baju' olaraq 100% dəqiq oxumuşdur).\n"
            "• o -> ə xətası (V2-də 1 xəta; Tesseract-da 0 xəta):\n"
            "  - ref: 'Goranboy,' -> hyp: 'Gəranboy,' | Model: V2, Qrup: C, Şrift: cambria.ttc, Səviyyə: L4, Rejim: word\n"
            "  (Tesseract 5 'Goranboy,' sözünü dəqiq tanımışdır).\n"
            "• ç -> s xətası (ç üçün cəmi n=23 instansiya, V2-də 1 xəta [4.35%]; Tesseract-da 0 xəta):\n"
            "  - ref: 'qaçan' -> hyp: 'qasan' | Model: V2, Qrup: B, Şrift: verdanab.ttf, Səviyyə: L4, Rejim: word\n"
            "  (Tesseract 5 'qaçan' sözünü tam dəqiq oxumuşdur; lakin Tesseract-da fərqli s->c və c->s əvəzləmələri müşahidə edilmişdir).\n"
            "• c -> e halları (Etalon mətndə hərf səhvi / reference typo):\n"
            "  - ref: 'dcmokratik' -> hyp: 'demokratik' | Həm V2, həm Tesseract 5 (cambria.ttc, word və line)\n"
            "  (Qeyd: Etalon korpusda 'demokratik' sözü səhvən 'dcmokratik' daxil edilmişdir; hər iki model təbii orfoqrafik 'e' qəbul edərək eyni xətanı vermişdir).\n"
            "• Digər Registr və Simvol Fərqləri:\n"
            "  - ref: 'C.' -> hyp: 'c.' | Həm V2, həm Tesseract (CascadiaMono.ttf, L0/L4, word)\n"
            "  - ref: 'Gürcüstan baş hakiminin Qarabağda' -> hyp: 'Cürcüstan baş hakiminin Qarabağda' | V2 (pala.ttf, L0/L4, line; G->C qarışması)"
        )
        for r in p.runs:
            set_run_font(r, size=9.5)

    # Reference 3: Official Tesseract LSTM documentation
    elif txt.startswith("3. Smith, R. (2007)"):
        p.text = (
            "3. Tesseract OCR. (2018). Neural nets in Tesseract 4.00. GitHub documentation: "
            "https://github.com/tesseract-ocr/tesseract/wiki/NeuralNetsInTesseract4.00"
        )
        for r in p.runs:
            set_run_font(r, size=9.5)

# ---------------------------------------------------------------------
# 2. Update Table 1: Salt-and-Pepper Code Quote and Pixel Fraction
# ---------------------------------------------------------------------
t1 = doc.tables[0]
format_cell(
    t1.rows[4].cells[0],
    "Duz-İstiot Küyü (Salt & Pepper)",
    bold=False
)
format_cell(
    t1.rows[4].cells[1],
    "img[m < 0.005] = 0, img[m > 0.995] = 255 (p=0.3)\n"
    "Hər növ 0.005 (0.5% istiot, 0.5% duz), cəmi 1.0% piksel"
)
format_cell(
    t1.rows[4].cells[2],
    "img[m < 0.005] = 0, img[m > 0.995] = 255 (p=0.2, salt_pepper_prob)\n"
    "Hər növ 0.005 (0.5% istiot, 0.5% duz), cəmi 1.0% piksel"
)

# ---------------------------------------------------------------------
# 3. Update Table 7 Caption if present
# ---------------------------------------------------------------------
for p in doc.paragraphs:
    if p.text.startswith("Cədvəl 7 üzrə Mühüm Elmi Qeydlər"):
        p.text = (
            "Cədvəl 7 üzrə Mühüm Elmi Qeydlər (Şrift qrupları V2 təlim siyahısına görə təyin edilmişdir):\n"
            "1. Qrup A şriftləri V1-in öz təlim şriftləridir; V1-in L0 söz xətalarının 94.5%-i (172/182) korpusda 0% olan rəqəm və durğu işarələridir (xalis hərf CER: 0.44%).\n"
            "2. Qrup A Word L0 üzrə V2 CER (0.81%, 20/2472 hərf) ilə Qrup C Word L4 üzrə Tesseract CER (0.81%, 34/4194 hərf) göstəricilərinin eyni olması sırf rəqəmsal təsadüfdür; hər iki göstərici yoxlanmış birbaşa saylardan qaynaqlanır.\n"
            "3. Qrup C üzrə 4 müqayisədən yalnız 1-də (Word L4) fərq intervalı sıfırı əhatə etmir (+1.55% [+0.59%, +2.94%]); bu pillədə Tesseract V2 modelini birmənalı üstələyir. Digər 3 halda modellər statistik bərabərdir.\n"
            "4. Qrup A (k=4) və Qrup B (k=5) klaster sayları az olduğu üçün bu qrupların intervalları yüksək dispersiyaya malikdir. Moba_Fira monospace şriftidir və təlimdə CascadiaCode/consolab iştirak etmişdir."
        )
        for r in p.runs:
            set_run_font(r, size=9.0, italic=True)

doc.save(DST_DOCX)
print(f"Successfully finalized {DST_DOCX}")
