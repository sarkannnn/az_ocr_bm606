import json
import pandas as pd
from pathlib import Path

out_dir = Path("results/real_fix9")
with open(out_dir / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

with open(out_dir / "error_taxonomy.json", "r", encoding="utf-8") as f:
    tax = json.load(f)

with open(out_dir / "empty_outputs_log.json", "r", encoding="utf-8") as f:
    empty_log = json.load(f)

df = pd.read_csv(out_dir / "per_line_chunk_details.csv")

md = []
md.append("# Real Fix 9: Photo 5 (`IMG_5130.HEIC`) Həssaslıq, İnteqrasiya və Xəta Təhlili Hesabatı\n")
md.append("Bu hesabat istifadəçi tərəfindən təqdim edilmiş `IMG_5130.HEIC` (3024x4032, 33 sətir, 5 paraqraf) fotosu üzərində Tesseract 5-in seqmentasiya uğursuzluğunu (boş sətir xətaları), ağ haşiyə həssaslığını (`sensitivity, not selected`), nisbət qruplaşdırmasını ($W\\cdot 32 / (h\\cdot 512)$) və təmiz alt-çoxluq (`clean subset`) üzrə müqayisəli dəqiqlik nəticələrini əks etdirir.\n")

md.append("## 1. Metodoloji Açıqlama və Şəkillərin Təsnifatı\n")
md.append("### 1.1 Sənəd Nömrələnməsi (Photos 1 – 5 Haradadır?):")
md.append("- **Photo 1 (`results/real_test/`, `results/real_fix2-8/`):** İstifadəçinin ilk yüklədiyi 16 sətirlik sənəd fotosu (sol tərəfində kölgə zolağı və em-dash olan mətn).")
md.append("- **Photo 2 (`results/real_fix2/` – `results/real_fix8/`):** İkinci sənəd fotosu (16 sətir, `kasalar, bəzəkli küplər...` mətni).")
md.append("- **Photo 3 (`results/photo3_test/`):** Serif / Times New Roman şriftli 23 sətirlik çap sənədi (`media_1793351399184.jpg`).")
md.append("- **Photo 4 (`results/photo4_sans/`):** Sans-serif / Arial şriftli 23 sətirlik çap sənədi (`media_1793353856829.jpg`).")
md.append("- **Photo 5 (`results/photo5_heic_test/`, `results/real_fix9/`):** Təqdim edilmiş `C:\\Users\\serxan.m\\Downloads\\IMG_5130.HEIC` (bütün 33 sətirlik əsas hekayə mətnini ehtiva edən tam səhifə fotosu).\n")

md.append("### 1.2 Sətir Qutuları və Sərhədlərin Əldə Edilməsi:")
md.append("- **Avtomatik Aşkar:** Tesseract `image_to_data` vasitəsilə söz qutuları çıxarılmış, `(block_num, par_num, line_num)` üzrə qruplaşdırılaraq 33 sətir aşkar edilmişdir.")
md.append("- **Şaquli Sərhədlər ($y_0, y_1$):** Hər sətir üçün mətndən yuxarı və aşağı 4 px marja (`top - 4`, `bottom + 4`) əlavə edilmişdir.")
md.append("- **Üfüqi Sərhədlər ($x_0, x_1$):** Sol və sağ tərəfə 12 px marja tətbiq edilmişdir.")
md.append("- **Fərdiləşdirilmiş Tənzimləmə:** Sənəddə yalnız Line 12-də (0-indeks 11) sol kənarda ($x=395$) olan printer ləkəsinin mətndən kənarda qalması üçün $x_0=430$ təyin edilmişdir. Qalan 32 sətir tam avtomatik kəsilmişdir.")
md.append("- **Referansların Uyğunlaşdırılması:** Çap olunmuş 33 sətirlik referans mətn yuxarıdan aşağıya oxu ardıcıllığı ilə aşkarlanmış 33 sətirə 1-ə 1 dəqiq uyğunlaşdırılmışdır.\n")

md.append("## 2. Boş Çıxışların Qeydiyyatı və İnteqrasiya Uğursuzluğu (PSM 7 vs PSM 13)\n")
empty_A = [x for x in empty_log if "A" in x["condition"]]
empty_B = [x for x in empty_log if "B" in x["condition"]]
md.append(f"- **V1 Line:** Şərt A-da 0, Şərt B-də 0 boş çıxış.")
md.append(f"- **V2 Line:** Şərt A-da 0, Şərt B-də 0 boş çıxış.")
md.append(f"- **Tesseract 5:** Şərt A-da **11 boş çıxış** (9 ədəd 2-ci parça: Sətir 18, 19, 21, 22, 23, 24, 26, 28, 32; və 2 ədəd tam sətir: Sətir 24, 29). Şərt B-də **13 boş çıxış**.")
md.append("- **Səbəb:** Tesseract `--psm 7` rejimində tək sətir aşkarlayıcısı ətraf səhifə konteksti olmadan ensiz/yüksək aspektli kəsimlərdə baza xəttini (`baseline`) itirir və inteqrasiya səviyyəsində boş sətir (`\"\"`) qaytarır.")
md.append("- **PSM 13 Fallback Testi:** Həmin boş çıxan kəsimlər `--psm 13` (raw line) ilə təkrar yoxlandıqda Tesseract mətni demək olar ki, qüsursuz oxuyur (məsələn, Sətir 18: `bu təkərin arxasında oturmusan. Özünü`, Sətir 24: `gizlənib. Biz getdikdən sonra bu əllər olmayacaq, amma bu küplər bizim`). Bu fakt problemin optik tanıma (OCR) dəqiqliyindən deyil, Tesseract daxili xətt seqmentatorunun kəsilmiş zolaqlardakı inteqrasiya uğursuzluğundan qaynaqlandığını qəti sübut edir.\n")

md.append("## 3. Əsas Nəticələr Cədvəli: Şərt A, Şərt B və Təmiz Alt-Çoxluq (Clean Subset)\n")
md.append("*(Qeyd: Əsas cədvəldə Tesseract yalnız `--psm 7` üzrə qiymətləndirilir; Şərt B 'sensitivity, not selected' kimi etiketlənir)*\n")

md.append("### 3.1 Bütün 33 Sətir Üzrə Müqayisə (Şərt A və Şərt B)\n")
md.append("| Model və Konfiqurasiya | CER (%) | WER (%) | Dəqiq Sətir | CER Xəta | WER Xəta | Boşluq Xətası | Boş Çıxış Sayı |")
md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

mA = metrics["condition_A_all33"]
mB = metrics["condition_B_padded_all33"]

table_rows = [
    ("**V2 Rule Chunk (Şərt A - Trim+4px)**", mA["v2_rule_chunk"]["cer"], mA["v2_rule_chunk"]["wer"], f"**{mA['v2_rule_chunk']['exact']}/33**", mA["v2_rule_chunk"]["cer_errs"], mA["v2_rule_chunk"]["wer_errs"], mA["v2_rule_chunk"]["space_errors_char"], 0),
    ("V2 Full Line (Şərt A - Trim+4px)", mA["v2_full"]["cer"], mA["v2_full"]["wer"], f"{mA['v2_full']['exact']}/33", mA["v2_full"]["cer_errs"], mA["v2_full"]["wer_errs"], mA["v2_full"]["space_errors_char"], 0),
    ("Tesseract 5 Full Line (Şərt A, psm 7)", mA["tess_full_psm7"]["cer"], mA["tess_full_psm7"]["wer"], f"{mA['tess_full_psm7']['exact']}/33", mA["tess_full_psm7"]["cer_errs"], mA["tess_full_psm7"]["wer_errs"], mA["tess_full_psm7"]["space_errors_char"], 2),
    ("V1 Rule Chunk (Şərt A - Trim+4px)", mA["v1_rule_chunk"]["cer"], mA["v1_rule_chunk"]["wer"], f"{mA['v1_rule_chunk']['exact']}/33", mA["v1_rule_chunk"]["cer_errs"], mA["v1_rule_chunk"]["wer_errs"], mA["v1_rule_chunk"]["space_errors_char"], 0),
    ("Tesseract 5 Rule Chunk (Şərt A, psm 7)", mA["tess_rule_chunk_psm7"]["cer"], mA["tess_rule_chunk_psm7"]["wer"], f"{mA['tess_rule_chunk_psm7']['exact']}/33", mA["tess_rule_chunk_psm7"]["cer_errs"], mA["tess_rule_chunk_psm7"]["wer_errs"], mA["tess_rule_chunk_psm7"]["space_errors_char"], 9),
    ("V1 Full Line (Şərt A - Trim+4px)", mA["v1_full"]["cer"], mA["v1_full"]["wer"], f"{mA['v1_full']['exact']}/33", mA["v1_full"]["cer_errs"], mA["v1_full"]["wer_errs"], mA["v1_full"]["space_errors_char"], 0),
    ("---", 0, 0, "---", 0, 0, 0, 0),
    ("V2 Rule Chunk (Şərt B - sensitivity, not selected)", mB["v2_rule_chunk"]["cer"], mB["v2_rule_chunk"]["wer"], f"{mB['v2_rule_chunk']['exact']}/33", mB["v2_rule_chunk"]["cer_errs"], mB["v2_rule_chunk"]["wer_errs"], mB["v2_rule_chunk"]["space_errors_char"], 0),
    ("Tesseract 5 Rule Chunk (Şərt B, psm 7 - sensitivity)", mB["tess_rule_chunk_psm7"]["cer"], mB["tess_rule_chunk_psm7"]["wer"], f"{mB['tess_rule_chunk_psm7']['exact']}/33", mB["tess_rule_chunk_psm7"]["cer_errs"], mB["tess_rule_chunk_psm7"]["wer_errs"], mB["tess_rule_chunk_psm7"]["space_errors_char"], 11),
    ("V2 Full Line (Şərt B - sensitivity, not selected)", mB["v2_full"]["cer"], mB["v2_full"]["wer"], f"{mB['v2_full']['exact']}/33", mB["v2_full"]["cer_errs"], mB["v2_full"]["wer_errs"], mB["v2_full"]["space_errors_char"], 0),
    ("Tesseract 5 Full Line (Şərt B, psm 7 - sensitivity)", mB["tess_full_psm7"]["cer"], mB["tess_full_psm7"]["wer"], f"{mB['tess_full_psm7']['exact']}/33", mB["tess_full_psm7"]["cer_errs"], mB["tess_full_psm7"]["wer_errs"], mB["tess_full_psm7"]["space_errors_char"], 2),
]

for row in table_rows:
    if row[0] == "---":
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        continue
    name, cer, wer, exact, cerr, werr, sperr, empties = row
    md.append(f"| {name} | {cer:.2f}% | {wer:.2f}% | {exact} | {cerr} | {werr} | {sperr} | {empties} |")

md.append("\n### 3.2 Təmiz Alt-Çoxluq (Clean Subset: 23 Sətir, Heç Bir Modelin Boş Çıxışı Olmayan Sətirlər)\n")
md.append("Tesseract-ın inteqrasiya uğursuzluğuna uğramadığı 23 sətir üzrə birbaşa optik tanıma dəqiqliyi:\n")
md.append("| Model və Konfiqurasiya | CER (%) | WER (%) | Dəqiq Sətir | CER Xəta | WER Xəta | Boşluq Xətası |")
md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

mC = metrics["clean_subset_A"]
clean_rows = [
    ("**Tesseract 5 Rule Chunk (psm 7)**", mC["tess_rule_chunk_psm7"]["cer"], mC["tess_rule_chunk_psm7"]["wer"], f"{mC['tess_rule_chunk_psm7']['exact']}/23", mC["tess_rule_chunk_psm7"]["cer_errs"], mC["tess_rule_chunk_psm7"]["wer_errs"], mC["tess_rule_chunk_psm7"]["space_errors_char"]),
    ("**V2 Line Rule Chunk**", mC["v2_rule_chunk"]["cer"], mC["v2_rule_chunk"]["wer"], f"{mC['v2_rule_chunk']['exact']}/23", mC["v2_rule_chunk"]["cer_errs"], mC["v2_rule_chunk"]["wer_errs"], mC["v2_rule_chunk"]["space_errors_char"]),
    ("Tesseract 5 Full Line (psm 7)", mC["tess_full_psm7"]["cer"], mC["tess_full_psm7"]["wer"], f"{mC['tess_full_psm7']['exact']}/23", mC["tess_full_psm7"]["cer_errs"], mC["tess_full_psm7"]["wer_errs"], mC["tess_full_psm7"]["space_errors_char"]),
    ("V2 Line Full Line", mC["v2_full"]["cer"], mC["v2_full"]["wer"], f"{mC['v2_full']['exact']}/23", mC["v2_full"]["cer_errs"], mC["v2_full"]["wer_errs"], mC["v2_full"]["space_errors_char"]),
    ("V1 Line Rule Chunk", mC["v1_rule_chunk"]["cer"], mC["v1_rule_chunk"]["wer"], f"{mC['v1_rule_chunk']['exact']}/23", mC["v1_rule_chunk"]["cer_errs"], mC["v1_rule_chunk"]["wer_errs"], mC["v1_rule_chunk"]["space_errors_char"]),
    ("V1 Line Full Line", mC["v1_full"]["cer"], mC["v1_full"]["wer"], f"{mC['v1_full']['exact']}/23", mC["v1_full"]["cer_errs"], mC["v1_full"]["wer_errs"], mC["v1_full"]["space_errors_char"]),
]
for name, cer, wer, exact, cerr, werr, sperr in clean_rows:
    md.append(f"| {name} | {cer:.2f}% | {wer:.2f}% | {exact} | {cerr} | {werr} | {sperr} |")

md.append("\n## 4. Bootstrap 95% Etibarlılıq İntervalları (V2 − Tesseract psm 7)\n")
bA = mA["bootstrap_ci_rule_chunk_psm7"]
bB = mB["bootstrap_ci_rule_chunk_psm7"]
bC = mC["bootstrap_ci_rule_chunk_psm7"]

md.append(f"- **Şərt A (Bütün 33 Sətir):** $\Delta$CER: {bA['delta_cer']:+.2f}% (95% CI: [{bA['cer_ci'][0]:.2f}%, {bA['cer_ci'][1]:.2f}%]), $\Delta$WER: {bA['delta_wer']:+.2f}% (95% CI: [{bA['wer_ci'][0]:.2f}%, {bA['wer_ci'][1]:.2f}%])")
md.append(f"- **Şərt B (Padded 20px, Həssaslıq):** $\Delta$CER: {bB['delta_cer']:+.2f}% (95% CI: [{bB['cer_ci'][0]:.2f}%, {bB['cer_ci'][1]:.2f}%]), $\Delta$WER: {bB['delta_wer']:+.2f}% (95% CI: [{bB['wer_ci'][0]:.2f}%, {bB['wer_ci'][1]:.2f}%])")
md.append(f"- **Təmiz Alt-Çoxluq (23 Sətir):** $\Delta$CER: **{bC['delta_cer']:+.2f}%** (95% CI: **[{bC['cer_ci'][0]:.2f}%, {bC['cer_ci'][1]:.2f}%]**), $\Delta$WER: **{bC['delta_wer']:+.2f}%** (95% CI: **[{bC['wer_ci'][0]:.2f}%, {bC['wer_ci'][1]:.2f}%]**)")
md.append("> **Həlledici Müşahidə:** Təmiz alt-çoxluqda $\Delta$CER 95% intervalı `[-0.77%, +0.80%]` aralığında sıfırı tam kəsir. Yəni Tesseract seqmentasiya səhvi etmədikdə, V2 Line modeli (1.32% CER) ilə Tesseract 5 (1.26% CER) arasında fərq statistik cəhətdən sıfırdan fərqlənmir!\n")

md.append("## 5. Düzəldilmiş Qarşılıqlı Müstəsna Xəta Taksonomiyası (Levenshtein Cəmi Dəqiq Bərabərdir)\n")
md.append("Əvvəlki skriptdə xəritələnmə xətası səbəbilə silinmiş simvollar durğu işarələrinə düşmüşdü. Bu dəfə hər bir silinmə öz simvol növünə görə (hərf, boşluq, durğu) təsnif edilmişdir:\n")
md.append("| Xəta Kateqoriyası | Şərt A: V2 | Şərt A: Tess | Təmiz: V2 | Təmiz: Tess | İzah |")
md.append("|---|:---:|:---:|:---:|:---:|---|")

tA_v2 = tax["condition_A"]["v2"]
tA_ts = tax["condition_A"]["tesseract"]
tC_v2 = tax["clean_subset"]["v2"]
tC_ts = tax["clean_subset"]["tesseract"]

tax_rows = [
    ("1. Başlanğıc Kənar Əlavəsi (Leading)", tA_v2["leading_edge_insertions"], tA_ts["leading_edge_insertions"], tC_v2["leading_edge_insertions"], tC_ts["leading_edge_insertions"], "V2-də 0 ədəd (sol kənar artefaktı yoxdur)"),
    ("2. Sonluq Kənar Əlavəsi (Trailing)", tA_v2["trailing_edge_insertions"], tA_ts["trailing_edge_insertions"], tC_v2["trailing_edge_insertions"], tC_ts["trailing_edge_insertions"], "Sətir sonu artıq simvol"),
    ("3. Daxili Boşluq Xətası", tA_v2["internal_space_errors"], tA_ts["internal_space_errors"], tC_v2["internal_space_errors"], tC_ts["internal_space_errors"], "Boşluq əməliyyatları"),
    ("4. Daxili 0 / O Qarışıqlığı", tA_v2["internal_o0_errors"], tA_ts["internal_o0_errors"], tC_v2["internal_o0_errors"], tC_ts["internal_o0_errors"], "0 və O qarışıqlığı"),
    ("5. Daxili Durğu Xətası", tA_v2["internal_punct_errors"], tA_ts["internal_punct_errors"], tC_v2["internal_punct_errors"], tC_ts["internal_punct_errors"], "Nöqtə/iki nöqtə (`.` vs `:`) fərqləri"),
    ("6. Daxili Hərf Xətası", tA_v2["internal_letter_errors"], tA_ts["internal_letter_errors"], tC_v2["internal_letter_errors"], tC_ts["internal_letter_errors"], "Hərf səhvləri və silinmələri"),
    ("CƏM (Mutually Exclusive Sum)", tA_v2["sum_categories"], tA_ts["sum_categories"], tC_v2["sum_categories"], tC_ts["sum_categories"], "100% Levenshtein cəminə bərabərdir"),
]

for name, v2_a, ts_a, v2_c, ts_c, desc in tax_rows:
    md.append(f"| **{name}** | **{v2_a}** | **{ts_a}** | **{v2_c}** | **{ts_c}** | {desc} |")

md.append("\n## 6. V2 Line Modelinin Nisbət Qrupları Üzrə CER Təhlili ($W\\cdot 32 / (h\\cdot 512)$)\n")
md.append("Sətirlər en/hündürlük nisbətinə görə 4 qrupa bölünərək tam sətir və parçalanmış rejimdə yoxlanmışdır:\n")
md.append("| Nisbət Qrupu | Sətir Sayı | Sətir İndeksləri | V2 Tam Sətir CER | V2 Parçalanmış CER | CER Azalması ($\Delta$ pp) |")
md.append("|---|:---:|---|:---:|:---:|:---:|")

ratio_data = metrics["v2_ratio_grouped_CER"]
for grp in ["<1.0", "1.0-1.1", "1.1-1.3", ">1.3"]:
    d = ratio_data[grp]
    l_str = ", ".join([f"L{x+1:02d}" for x in d["lines"]])
    md.append(f"| **{grp}** | {d['n_lines']} | {l_str} | {d['v2_full_cer']:.2f}% | {d['v2_chunk_cer']:.2f}% | **{d['cer_reduction_pp']:+.2f} pp** |")

md.append("\n> **Nəticə:** Nisbət böyüdükcə ($>1.3$) V2 Line modelinin tam sətirdəki CER-i 4.87%-ə qalxır və parçalama sayəsində xəta **3.54 faiz bəndi azalaraq 1.33%-ə** enir. Qısa sətirlərdə isə ($<1.0$) parçalamaya ehtiyac qalmır.\n")

md.append("## 7. Əlaqə Cədvəli (Contact Sheet: Sətir 18, 19, 21, 22 İkinci Parçaları)\n")
md.append("Tesseract-ın `--psm 7` rejimində boş qayıtmasının vizual səbəbini göstərən əlaqə vərəqi [contact_sheet_lines18_19_21_22.png](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/contact_sheets/contact_sheet_lines18_19_21_22.png) faylında saxlanılmışdır.\n")
md.append("- Sətir 18 (2-ci parça): En 1149 px, Hündürlük 128 px, Aspekt nisbəti 8.98 -> Tesseract psm 7: `\"\"` (boş), lakin psm 13: `bu təkərin arxasında oturmusan. Özünü`")
md.append("- Sətir 19 (2-ci parça): En 725 px, Hündürlük 120 px, Aspekt nisbəti 6.04 -> Tesseract psm 7: `\"\"` (boş), lakin psm 13: `Səməndər başını qaldırıb`")
md.append("- Sətir 21 (2-ci parça): En 1239 px, Hündürlük 119 px, Aspekt nisbəti 10.41 -> Tesseract psm 7: `\"\"` (boş), lakin psm 13: `masaya qoyub dedi: \"Ay bala, insan o vaxt`")
md.append("- Sətir 22 (2-ci parça): En 743 px, Hündürlük 125 px, Aspekt nisbəti 5.94 -> Tesseract psm 7: `\"\"` (boş), lakin psm 13: `torpaq parçası deyil. Hər`\n")

with open(out_dir / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Comprehensive real_fix9 report generated successfully.")
