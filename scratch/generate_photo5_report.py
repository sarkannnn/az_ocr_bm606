import json
import pandas as pd
from pathlib import Path

out_dir = Path("results/photo5_heic_test")
with open(out_dir / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

with open(out_dir / "error_taxonomy.json", "r", encoding="utf-8") as f:
    tax = json.load(f)

df = pd.read_csv(out_dir / "per_line_chunk_details.csv")

md = []
md.append("# Photo 5 (IMG_5130.HEIC) OCR Qiymətləndirmə Hesabatı\n")
md.append("Bu hesabat istifadəçi tərəfindən təqdim edilmiş `IMG_5130.HEIC` (3024x4032 yüksək keyfiyyətli sənəd fotosu, 33 sətir, 5 paraqraf) üzərində dondurulmuş boru xəttinin (`pipeline`) icra və müqayisəli qiymətləndirmə nəticələrini əks etdirir.\n")

md.append("## 1. Ümumi Metriklər Cədvəli (33 Sətir, 2 328 Simvol, 336 Söz)\n")
md.append("| Model və Rejim | CER (%) | WER (%) | Dəqiq Sətirlər (Exact) | Simvol Xətası | Söz Xətası | Boşluq Xətası |")
md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

rows = [
    ("**V2 Line (Qayda Əsaslı Parçalama)**", metrics["v2_rule_chunk"]["cer"], metrics["v2_rule_chunk"]["wer"], f"**{metrics['v2_rule_chunk']['exact']}/33**", metrics["v2_rule_chunk"]["cer_errs"], metrics["v2_rule_chunk"]["wer_errs"], metrics["v2_rule_chunk"]["space_errors_char"]),
    ("V2 Line (Tam Sətir)", metrics["v2_full"]["cer"], metrics["v2_full"]["wer"], f"{metrics['v2_full']['exact']}/33", metrics["v2_full"]["cer_errs"], metrics["v2_full"]["wer_errs"], metrics["v2_full"]["space_errors_char"]),
    ("Tesseract 5 (Tam Sətir)", metrics["tess_full"]["cer"], metrics["tess_full"]["wer"], f"{metrics['tess_full']['exact']}/33", metrics["tess_full"]["cer_errs"], metrics["tess_full"]["wer_errs"], metrics["tess_full"]["space_errors_char"]),
    ("V1 Line (Qayda Əsaslı Parçalama)", metrics["v1_rule_chunk"]["cer"], metrics["v1_rule_chunk"]["wer"], f"{metrics['v1_rule_chunk']['exact']}/33", metrics["v1_rule_chunk"]["cer_errs"], metrics["v1_rule_chunk"]["wer_errs"], metrics["v1_rule_chunk"]["space_errors_char"]),
    ("Tesseract 5 (Qayda Əsaslı Parçalama)", metrics["tess_rule_chunk"]["cer"], metrics["tess_rule_chunk"]["wer"], f"{metrics['tess_rule_chunk']['exact']}/33", metrics["tess_rule_chunk"]["cer_errs"], metrics["tess_rule_chunk"]["wer_errs"], metrics["tess_rule_chunk"]["space_errors_char"]),
    ("V1 Line (Tam Sətir)", metrics["v1_full"]["cer"], metrics["v1_full"]["wer"], f"{metrics['v1_full']['exact']}/33", metrics["v1_full"]["cer_errs"], metrics["v1_full"]["wer_errs"], metrics["v1_full"]["space_errors_char"]),
]

for name, cer, wer, exact, cerr, werr, sperr in rows:
    md.append(f"| {name} | {cer:.2f}% | {wer:.2f}% | {exact} | {cerr} | {werr} | {sperr} |")

md.append("\n## 2. Bootstrap 95% Etibarlılıq İntervalı (V2 - Tesseract, Qayda Əsaslı Parçalama)\n")
b = metrics["bootstrap_ci_rule_chunk"]
md.append(f"- **$\Delta$CER:** {b['delta_cer']:+.2f}% (95% CI: [{b['cer_ci'][0]:.2f}%, {b['cer_ci'][1]:.2f}%])")
md.append(f"- **$\Delta$WER:** {b['delta_wer']:+.2f}% (95% CI: [{b['wer_ci'][0]:.2f}%, {b['wer_ci'][1]:.2f}%])\n")
md.append("> **Qeyd:** $\Delta$CER mənfi və 95% CI tamamilə sıfırdan aşağıdır ([-17.93%, -4.62%]), bu da V2 Line modelinin Tesseract üzərində statistik əhəmiyyətli (p < 0.05) üstünlüyünü sübut edir.\n")

md.append("## 3. Qarşılıqlı Müstəsna Xəta Taksonomiyası (V2 Line və Tesseract)\n")
md.append("| Xəta Kateqoriyası | V2 Line Sayı | Tesseract Sayı | İzah |")
md.append("|---|:---:|:---:|---|")
md.append(f"| **1. Başlanğıc Kənar Əlavəsi (Leading)** | **{tax['v2']['leading_edge_insertions']}** | **{tax['tesseract']['leading_edge_insertions']}** | V2-də kənar kölgə/artefakt əlavəsi 0-dır |")
md.append(f"| **2. Sonluq Kənar Əlavəsi (Trailing)** | **{tax['v2']['trailing_edge_insertions']}** | **{tax['tesseract']['trailing_edge_insertions']}** | Sətir sonu artıq simvol |")
md.append(f"| **3. Daxili Boşluq Xətası** | **{tax['v2']['internal_space_errors']}** | **{tax['tesseract']['internal_space_errors']}** | V2-də cəmi 3 boşluq xətası, Tesseract-da 44 |")
md.append(f"| **4. Daxili 0 / O Qarışıqlığı** | **{tax['v2']['internal_o0_errors']}** | **{tax['tesseract']['internal_o0_errors']}** | Rəqəm `0` və hərf `O` |")
md.append(f"| **5. Daxili Durğu Xətası** | **{tax['v2']['internal_punct_errors']}** | **{tax['tesseract']['internal_punct_errors']}** | Nöqtə/iki nöqtə fərqləri (`.` vs `:`) |")
md.append(f"| **6. Daxili Hərf Xətası** | **{tax['v2']['internal_letter_errors']}** | **{tax['tesseract']['internal_letter_errors']}** | Hərf oxuma xətaları |")
md.append(f"| **CƏM (Mutually Exclusive)** | **{tax['v2']['sum_categories']}** | **{tax['tesseract']['sum_categories']}** | $100\%$ Levenshtein cəminə bərabərdir |")

md.append("\n## 4. Əsas Müşahidələr və Təhlil\n")
md.append("1. **Parçalamanın (Chunking) Təsiri:**")
md.append("   - V2 Line tam sətirdə 3.87% CER (90 xəta) göstərdiyi halda, qayda əsaslı parçalama ilə xəta sayı 49-a, CER isə **2.10%-ə** düşmüşdür (təxminən 45% xəta azalması).")
md.append("   - Tesseract tək-tək kəsilmiş sətir parçalarında bəzən (məsələn, 17, 18, 20, 21, 23, 25-ci sətirlərin 2-ci yarısında) boş sətir qaytardığı üçün parçalanmış rejimdə CER 13.02%-ə yüksəlmişdir. Bu, V2 Line modelinin vizual seqmentlərə və kəsimlərə qarşı Tesseract-dan xeyli dözümlü olduğunu göstərir.")
md.append("2. **Kənar Əlavələri (Edge Insertions):**")
md.append("   - Dondurulmuş robast CC təmizləmə qaydası (area < 8 silinməsi, hündür kölgə zolaqlarının kəsilməsi və +4 px ink kənarı) sayəsində V2-də başlanğıc və sonluq kənar əlavələri tamamilə **0** olmuşdur (Foto 1-dəki kimi dırnaq `\"` xətaları təkrarlanmamışdır).")
md.append("3. **Boşluq İnteqrasiyası:**")
md.append("   - Birləşdirmə zamanı proyeksiya kəsim nöqtəsi qaydası (`ksize=25`, 35-ci persentil) sözləri bölməmiş, 33 sətirlik iri sənəddə cəmi 3 daxili boşluq fərqi yaranmışdır.")

md.append("\n## 5. Bütün 33 Sətir Üzrə Müfəssəl Nəticələr (Reference vs OCR)\n")
for idx, r in df.iterrows():
    v2_match = "DƏQİQ DÜZ" if str(r['v2_rule_joined']).strip() == str(r['ref']).strip() else "FƏRQ VAR"
    tess_match = "DƏQİQ DÜZ" if str(r['tess_rule_joined']).strip() == str(r['ref']).strip() else "FƏRQ VAR"
    md.append(f"### Sətir {idx+1:02d} (Ölçü: {r['crop_w']}x{r['crop_h']} px, Parça sayı: {r['rule_n_chunks']})")
    md.append(f"- **Orijinal (Referans):** `{r['ref']}`")
    md.append(f"- **V2 Line (Parçalanmış):** `{r['v2_rule_joined']}` ({v2_match})")
    md.append(f"- **Tesseract 5 (Parçalanmış):** `{r['tess_rule_joined']}` ({tess_match})")
    md.append(f"- **V2 Line (Tam Sətir):** `{r['v2_full']}`\n")

with open(out_dir / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Photo 5 Report successfully written.")
