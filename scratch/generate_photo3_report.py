import json
import pandas as pd
from pathlib import Path

out_dir = Path("results/photo3_test")
with open(out_dir / "photo3_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

df = pd.read_csv(out_dir / "per_line_chunk_details.csv")

md = []
md.append("# Photo 3 (Yeni Sənəd) OCR Qiymətləndirmə Hesabatı\n")
md.append("Bu hesabat istifadəçi tərəfindən təqdim edilən 3-cü real çap olunmuş sənəd fotosu üzərində dondurulmuş OCR boru xəttinin (`pipeline`) icra nəticələrini əks etdirir.\n")

md.append("## 1. Ümumi Metriklər Cədvəli\n")
md.append("| Model & Rejim | CER (%) | WER (%) | Dəqiq Sətirlər (Exact) | Səhv Hərf/Simvol Sayı | Səhv Söz Sayı |")
md.append("|---|---|---|---|---|---|")

rows = [
    ("Tesseract 5 (Tam Sətir)", metrics["tess_full"]["cer"], metrics["tess_full"]["wer"], f"{metrics['tess_full']['exact']}/23", metrics["tess_full"]["cer_errs"], metrics["tess_full"]["wer_errs"]),
    ("Tesseract 5 (Qayda Əsaslı Chunk)", metrics["tess_rule_chunk"]["cer"], metrics["tess_rule_chunk"]["wer"], f"{metrics['tess_rule_chunk']['exact']}/23", metrics["tess_rule_chunk"]["cer_errs"], metrics["tess_rule_chunk"]["wer_errs"]),
    ("**V2 Line (Qayda Əsaslı Chunk)**", metrics["v2_rule_chunk"]["cer"], metrics["v2_rule_chunk"]["wer"], f"**{metrics['v2_rule_chunk']['exact']}/23**", metrics["v2_rule_chunk"]["cer_errs"], metrics["v2_rule_chunk"]["wer_errs"]),
    ("V2 Line (Tam Sətir)", metrics["v2_full"]["cer"], metrics["v2_full"]["wer"], f"{metrics['v2_full']['exact']}/23", metrics["v2_full"]["cer_errs"], metrics["v2_full"]["wer_errs"]),
    ("V1 Line (Qayda Əsaslı Chunk)", metrics["v1_rule_chunk"]["cer"], metrics["v1_rule_chunk"]["wer"], f"{metrics['v1_rule_chunk']['exact']}/23", metrics["v1_rule_chunk"]["cer_errs"], metrics["v1_rule_chunk"]["wer_errs"]),
    ("V1 Line (Tam Sətir)", metrics["v1_full"]["cer"], metrics["v1_full"]["wer"], f"{metrics['v1_full']['exact']}/23", metrics["v1_full"]["cer_errs"], metrics["v1_full"]["wer_errs"]),
]

for name, cer, wer, exact, cerr, werr in rows:
    md.append(f"| {name} | {cer:.2f}% | {wer:.2f}% | {exact} | {cerr} | {werr} |")

md.append("\n## 2. Bootstrap 95% Etibarlılıq İntervalları (V2 - Tesseract)\n")
b_chunk = metrics["bootstrap_ci"]["rule_chunk_v2_minus_tess"]
b_full = metrics["bootstrap_ci"]["full_line_v2_minus_tess"]
md.append(f"- **Qayda Əsaslı Bölünmə (Chunking):**")
md.append(f"  - $\\Delta$CER 95% CI: [{b_chunk['cer_ci'][0]:.2f}%, {b_chunk['cer_ci'][1]:.2f}%]")
md.append(f"  - $\\Delta$WER 95% CI: [{b_chunk['wer_ci'][0]:.2f}%, {b_chunk['wer_ci'][1]:.2f}%]")
md.append(f"- **Tam Sətir (Full Line):**")
md.append(f"  - $\\Delta$CER 95% CI: [{b_full['cer_ci'][0]:.2f}%, {b_full['cer_ci'][1]:.2f}%]")
md.append(f"  - $\\Delta$WER 95% CI: [{b_full['wer_ci'][0]:.2f}%, {b_full['wer_ci'][1]:.2f}%]\n")

md.append("## 3. V2 Xəta Taksonomiyası (Mutually Exclusive)\n")
with open(out_dir / "error_taxonomy.json", "r", encoding="utf-8") as f:
    tax = json.load(f)

md.append(f"- **Qayda Əsaslı V2 Xətaları cəmi:** {len(tax['rule_chunk_v2'])} ədəd")
for e in tax["rule_chunk_v2"]:
    md.append(f"  - Sətir {e['line_idx']+1}: Tip: `{e['op']}`, Kateqoriya: `{e['category']}`, Ref simvol: '{e['ref_char']}', Model simvol: '{e['pred_char']}'")

md.append(f"\n- **Tam Sətir V2 Xətaları cəmi:** {len(tax['full_line_v2'])} ədəd")
for e in tax["full_line_v2"]:
    md.append(f"  - Sətir {e['line_idx']+1}: Tip: `{e['op']}`, Kateqoriya: `{e['category']}`, Ref simvol: '{e['ref_char']}', Model simvol: '{e['pred_char']}'")

md.append("\n## 4. Sətirbəsətir Ətraflı Nəticələr (23 Sətir)\n")
for idx, r in df.iterrows():
    md.append(f"### Sətir {idx+1:02d} (Ölçü: {r['crop_w']}x{r['crop_h']} px, Parça sayı: {r['rule_n_chunks']})")
    md.append(f"- **Orijinal (Çap olunmuş mətndəki ref):** `{r['ref']}`")
    md.append(f"- **V2 (Qayda əsaslı chunk):** `{r['v2_joined']}` ({'DƏQİQ DÜZ' if r['v2_joined'] == r['ref'] else 'FƏRQ VAR'})")
    md.append(f"- **Tesseract 5:** `{r['tess_joined']}` ({'DƏQİQ DÜZ' if r['tess_joined'] == r['ref'] else 'FƏRQ VAR'})")
    md.append(f"- **V1 (Qayda əsaslı chunk):** `{r['v1_joined']}`\n")

with open(out_dir / "report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print("Report generated successfully.")
