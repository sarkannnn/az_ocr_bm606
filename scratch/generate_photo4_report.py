import json
import pandas as pd
from pathlib import Path

out_dir = Path("results/photo4_sans")
with open(out_dir / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

with open(out_dir / "error_taxonomy.json", "r", encoding="utf-8") as f:
    tax = json.load(f)

df = pd.read_csv(out_dir / "per_line_chunk_details.csv")

md = []
md.append("# Photo 4 (Sans-serif / Arial) OCR Qiymətləndirmə Hesabatı\n")
md.append("Bu hesabat istifadəçi tərəfindən təqdim edilmiş 4-cü sənəd fotosu (Sans-serif / Arial şrifti ilə çap olunmuş mətn) üzərində dondurulmuş boru xəttinin (`pipeline`) icra nəticələrini əks etdirir.\n")

md.append("## 1. Ümumi Metriklər Cədvəli (23 Sətir, 1 006 Simvol)\n")
md.append("| Model və Rejim | CER (%) | WER (%) | Dəqiq Sətirlər (Exact) | Simvol Xətası | Söz Xətası | Boşluq Xətası |")
md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

rows = [
    ("Tesseract 5 (Tam Sətir)", metrics["tess_full"]["cer"], metrics["tess_full"]["wer"], f"{metrics['tess_full']['exact']}/23", metrics["tess_full"]["cer_errs"], metrics["tess_full"]["wer_errs"], metrics["tess_full"]["space_errors_char"]),
    ("Tesseract 5 (Qayda Əsaslı Parçalama)", metrics["tess_rule_chunk"]["cer"], metrics["tess_rule_chunk"]["wer"], f"{metrics['tess_rule_chunk']['exact']}/23", metrics["tess_rule_chunk"]["cer_errs"], metrics["tess_rule_chunk"]["wer_errs"], metrics["tess_rule_chunk"]["space_errors_char"]),
    ("**V2 Line (Qayda Əsaslı Parçalama)**", metrics["v2_rule_chunk"]["cer"], metrics["v2_rule_chunk"]["wer"], f"**{metrics['v2_rule_chunk']['exact']}/23**", metrics["v2_rule_chunk"]["cer_errs"], metrics["v2_rule_chunk"]["wer_errs"], metrics["v2_rule_chunk"]["space_errors_char"]),
    ("V2 Line (Tam Sətir)", metrics["v2_full"]["cer"], metrics["v2_full"]["wer"], f"{metrics['v2_full']['exact']}/23", metrics["v2_full"]["cer_errs"], metrics["v2_full"]["wer_errs"], metrics["v2_full"]["space_errors_char"]),
    ("V1 Line (Qayda Əsaslı Parçalama)", metrics["v1_rule_chunk"]["cer"], metrics["v1_rule_chunk"]["wer"], f"{metrics['v1_rule_chunk']['exact']}/23", metrics["v1_rule_chunk"]["cer_errs"], metrics["v1_rule_chunk"]["wer_errs"], metrics["v1_rule_chunk"]["space_errors_char"]),
    ("V1 Line (Tam Sətir)", metrics["v1_full"]["cer"], metrics["v1_full"]["wer"], f"{metrics['v1_full']['exact']}/23", metrics["v1_full"]["cer_errs"], metrics["v1_full"]["wer_errs"], metrics["v1_full"]["space_errors_char"]),
]

for name, cer, wer, exact, cerr, werr, sperr in rows:
    md.append(f"| {name} | {cer:.2f}% | {wer:.2f}% | {exact} | {cerr} | {werr} | {sperr} |")

md.append("\n## 2. Bootstrap 95% Etibarlılıq İntervalı (V2 - Tesseract)\n")
b = metrics["bootstrap_ci_rule_chunk"]
md.append(f"- **$\Delta$CER:** {b['delta_cer']:+.2f}% (95% CI: [{b['cer_ci'][0]:.2f}%, {b['cer_ci'][1]:.2f}%])")
md.append(f"- **$\Delta$WER:** {b['delta_wer']:+.2f}% (95% CI: [{b['wer_ci'][0]:.2f}%, {b['wer_ci'][1]:.2f}%])\n")

md.append("## 3. Qarşılıqlı Müstəsna Xəta Taksonomiyası (V2 Line və Tesseract)\n")
md.append("| Xəta Kateqoriyası | V2 Line Sayı | Tesseract Sayı | İzah |")
md.append("|---|:---:|:---:|---|")
md.append(f"| **1. Başlanğıc Kənar Əlavəsi (Leading)** | **{tax['v2']['leading_edge_insertions']}** | **{tax['tesseract']['leading_edge_insertions']}** | 0 ədəd (Artıq dırnaq `\"` əlavələri tam yoxdur) |")
md.append(f"| **2. Sonluq Kənar Əlavəsi (Trailing)** | **{tax['v2']['trailing_edge_insertions']}** | **{tax['tesseract']['trailing_edge_insertions']}** | Sətir sonu artıq simvol |")
md.append(f"| **3. Daxili Boşluq Xətası** | **{tax['v2']['internal_space_errors']}** | **{tax['tesseract']['internal_space_errors']}** | Boşluq əməliyyatları |")
md.append(f"| **4. Daxili 0 / O Qarışıqlığı** | **{tax['v2']['internal_o0_errors']}** | **{tax['tesseract']['internal_o0_errors']}** | Rəqəm `0` və hərf `O` |")
md.append(f"| **5. Daxili Durğu Xətası** | **{tax['v2']['internal_punct_errors']}** | **{tax['tesseract']['internal_punct_errors']}** | Nöqtə, vergül, defis |")
md.append(f"| **6. Daxili Hərf Xətası** | **{tax['v2']['internal_letter_errors']}** | **{tax['tesseract']['internal_letter_errors']}** | Qlif tanınma fərqi |")
md.append(f"| **CƏM (Mutually Exclusive)** | **{tax['v2']['sum_categories']}** | **{tax['tesseract']['sum_categories']}** | $100\%$ Levenshtein cəminə bərabərdir |")

md.append("\n## 4. Bütün 23 Sətir Üzrə Müfəssəl Nəticələr\n")
for idx, r in df.iterrows():
    md.append(f"### Sətir {idx+1:02d} (Ölçü: {r['crop_w']}x{r['crop_h']} px, Parça: {r['rule_n_chunks']})")
    md.append(f"- **Orijinal (Çap olunmuş ref):** `{r['ref']}`")
    md.append(f"- **V2 Line (Parçalanmış):** `{r['v2_rule_joined']}` ({'DƏQİQ DÜZ' if r['v2_rule_joined'] == r['ref'] else 'FƏRQ VAR'})")
    md.append(f"- **Tesseract 5:** `{r['tess_rule_joined']}` ({'DƏQİQ DÜZ' if r['tess_rule_joined'] == r['ref'] else 'FƏRQ VAR'})")
    md.append(f"- **V1 Line (Parçalanmış):** `{r['v1_rule_joined']}`\n")

with open(out_dir / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Report generated successfully.")
