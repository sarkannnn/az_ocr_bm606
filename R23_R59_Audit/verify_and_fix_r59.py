"""Verify R59_RAW_PREDICTIONS.csv against R59_FONT_BENCHMARK.csv and auto-generate
R59_CORRECTED_RESULTS.csv and R59_FINAL_VERIFICATION.md directly from raw predictions.
"""

import hashlib
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from fontTools.ttLib import TTFont

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

from common import align

OUT_DIR = PROJ_ROOT / "R23_R59_Audit"
RAW_CSV = OUT_DIR / "R59_RAW_PREDICTIONS.csv"
BENCH_CSV = OUT_DIR / "R59_FONT_BENCHMARK.csv"
MANIFEST_CSV = OUT_DIR / "R59_DATASET_MANIFEST.csv"

FONTS = ["DejaVuSans", "DejaVuSerif", "DejaVuSansMono", "Lato-Regular", "Raleway-Regular"]
TARGET_CHARS = ["ə", "Ə", "ı", "i", "İ"]

BASELINES = [
    ("Baseline_CRNN-V2_Raw", "Baseline"),
    ("Baseline_Tesseract5_PSM7", "Baseline"),
    ("Baseline_EasyOCR", "Baseline"),
    ("Baseline_PaddleOCR", "Baseline"),
]

GROUPS = [
    ("Group_A_V1Clean_BasicAug", "Group A (V1Clean + BasicAug)"),
    ("Group_B_V1Clean_PhotoAug", "Group B (V1Clean + PhotoAug)"),
    ("Group_C_V2Clean_BasicAug", "Group C (V2Clean + BasicAug)"),
    ("Group_D_V2Clean_PhotoAug", "Group D (V2Clean + PhotoAug)"),
    ("Exp_RandomPadding_GroupD", "RandomPadding (Group D)"),
    ("Exp_VariableWidth_GroupD", "VariableWidth (Group D, Tesla T4)"),
]
SEEDS = [42, 43, 44]


def nfc(text) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def compute_metrics(refs, hyps):
    dc_tot, lc_tot, dw_tot, lw_tot = 0, 0, 0, 0
    exact = 0
    char_hits = {c: 0 for c in TARGET_CHARS}
    char_gt = {c: 0 for c in TARGET_CHARS}
    for r, h in zip(refs, hyps):
        rn, hn = nfc(r), nfc(h)
        dc, ops = align(rn, hn)
        dc_tot += dc
        lc_tot += len(rn)
        dw, _ = align(rn.split(), hn.split())
        dw_tot += dw
        lw_tot += len(rn.split())
        if rn == hn:
            exact += 1
        for c in TARGET_CHARS:
            char_gt[c] += rn.count(c)
        for op, rc, _ in ops:
            if op == "ok" and rc in TARGET_CHARS:
                char_hits[rc] += 1
    out = {
        "n_lines": len(refs),
        "cer_micro_pct": round(dc_tot / max(1, lc_tot) * 100.0, 2),
        "wer_micro_pct": round(dw_tot / max(1, lw_tot) * 100.0, 2),
        "exact_matches": exact,
        "exact_match_pct": round(exact / max(1, len(refs)) * 100.0, 2),
    }
    for c in TARGET_CHARS:
        out[f"acc_{c}_pct"] = round(char_hits[c] / char_gt[c] * 100.0, 2) if char_gt[c] > 0 else "N/A"
    return out


def main():
    df_raw = pd.read_csv(RAW_CSV)
    df_bench = pd.read_csv(BENCH_CSV)
    df_man = pd.read_csv(MANIFEST_CSV)

    assert len(df_raw) == 200
    assert len(df_man) == 200

    # Verify all 200 image hashes
    for _, r in df_man.iterrows():
        p = OUT_DIR / r["image_rel_path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["image_sha256"], f"SHA mismatch on {p}"

    refs_all = df_raw["ground_truth_nfc"].tolist()
    fonts_all = df_raw["font_name"].tolist()

    all_models = BASELINES + [
        (f"{gid}_seed{s}", gid) for gid, _ in GROUPS for s in SEEDS
    ]

    recomputed_rows = []
    bench_mismatches = 0

    for m_name, grp in all_models:
        hyps_all = df_raw[m_name].tolist()
        for scope in ["ALL_5_FONTS"] + FONTS:
            if scope == "ALL_5_FONTS":
                r_sub, h_sub = refs_all, hyps_all
            else:
                r_sub = [r for r, f in zip(refs_all, fonts_all) if f == scope]
                h_sub = [h for h, f in zip(hyps_all, fonts_all) if f == scope]
            m = compute_metrics(r_sub, h_sub)
            row = {
                "entry_type": "Baseline" if grp == "Baseline" else "GPU_Model",
                "model_or_group": m_name,
                "group_id": grp,
                "font_name": scope,
                **m,
            }
            recomputed_rows.append(row)

            # Cross-check against R59_FONT_BENCHMARK.csv
            b_match = df_bench[(df_bench["model_name"] == m_name) & (df_bench["font_name"] == scope)]
            assert len(b_match) == 1
            b_row = b_match.iloc[0]
            for col in ["cer_micro_pct", "wer_micro_pct", "exact_matches", "exact_match_pct", "acc_ə_pct", "acc_Ə_pct", "acc_ı_pct", "acc_i_pct", "acc_İ_pct"]:
                if float(b_row[col]) != float(m[col]):
                    print(f"DIFF {m_name} [{scope}] {col}: bench={b_row[col]} vs raw={m[col]}")
                    bench_mismatches += 1

    df_recomp = pd.DataFrame(recomputed_rows)

    # Add 3-seed Group Mean ± SD rows for ALL_5_FONTS and each font
    group_rows = []
    for gid, glabel in GROUPS:
        for scope in ["ALL_5_FONTS"] + FONTS:
            sub = df_recomp[(df_recomp["group_id"] == gid) & (df_recomp["font_name"] == scope)]
            grow = {
                "entry_type": "GPU_Group_Mean_SD",
                "model_or_group": f"{glabel} (n=3)",
                "group_id": gid,
                "font_name": scope,
                "n_lines": int(sub["n_lines"].iloc[0]),
            }
            for col in ["cer_micro_pct", "wer_micro_pct", "exact_matches", "exact_match_pct", "acc_ə_pct", "acc_Ə_pct", "acc_ı_pct", "acc_i_pct", "acc_İ_pct"]:
                vals = sub[col].astype(float).values
                grow[col] = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}"
            group_rows.append(grow)

    df_corrected = pd.concat([df_recomp, pd.DataFrame(group_rows)], ignore_index=True)
    df_corrected.to_csv(OUT_DIR / "R59_CORRECTED_RESULTS.csv", index=False)

    # Auto-generate Markdown tables directly from df_corrected
    ov_baselines = df_corrected[(df_corrected["entry_type"] == "Baseline") & (df_corrected["font_name"] == "ALL_5_FONTS")]
    ov_groups = df_corrected[(df_corrected["entry_type"] == "GPU_Group_Mean_SD") & (df_corrected["font_name"] == "ALL_5_FONTS")]

    def df_to_md_table(df_sub, cols, headers):
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for _, r in df_sub.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    cols_ov = ["model_or_group", "cer_micro_pct", "wer_micro_pct", "exact_match_pct", "acc_ə_pct", "acc_Ə_pct", "acc_ı_pct", "acc_i_pct", "acc_İ_pct"]
    headers_ov = ["Model / Group", "CER (%)", "WER (%)", "Exact (%)", "ə (%)", "Ə (%)", "ı (%)", "i (%)", "İ (%)"]

    tbl_all = df_to_md_table(pd.concat([ov_baselines, ov_groups]), cols_ov, headers_ov)

    # Per-font CER pivot auto-generated
    pf_lines = ["| Model / Group | DejaVuSans | DejaVuSerif | DejaVuSansMono | Lato-Regular | Raleway-Regular |", "|---|---:|---:|---:|---:|---:|"]
    for _, b in ov_baselines.iterrows():
        mname = b["model_or_group"]
        vals = [str(df_corrected[(df_corrected["model_or_group"] == mname) & (df_corrected["font_name"] == fn)]["cer_micro_pct"].iloc[0]) for fn in FONTS]
        pf_lines.append(f"| {mname} | " + " | ".join(vals) + " |")
    for _, g in ov_groups.iterrows():
        gname = g["model_or_group"]
        vals = [str(df_corrected[(df_corrected["model_or_group"] == gname) & (df_corrected["font_name"] == fn)]["cer_micro_pct"].iloc[0]) for fn in FONTS]
        pf_lines.append(f"| {gname} | " + " | ".join(vals) + " |")
    tbl_per_font = "\n".join(pf_lines)

    # Font TTFont metadata verification
    font_lines = ["| Font File | Family | Subfamily | PostScript Name | License (from source/name table) | SHA-256 (Prefix) |", "|---|---|---|---|---|---|"]
    for fp in sorted((OUT_DIR / "fonts_open").glob("*.ttf")):
        ft = TTFont(str(fp), lazy=True)
        names = {r.nameID: r.toUnicode() for r in ft["name"].names if r.nameID in (1, 2, 6, 13)}
        lic_short = "SIL Open Font License 1.1" if "SIL Open Font License" in names.get(13, "") else "DejaVu / Bitstream Vera License (LICENSE_DEJAVU.txt)"
        fsha = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
        font_lines.append(f"| `{fp.name}` | {names.get(1)} | {names.get(2)} | `{names.get(6)}` | {lic_short} | `{fsha}` |")
    tbl_fonts = "\n".join(font_lines)

    md = f"""# AzOCR-V2 — R59 Final Verification Report (`R59_FINAL_VERIFICATION.md`)

**Verification Date:** 2026-11-25  
**Source Data:** `R59_RAW_PREDICTIONS.csv` (200 images × 22 OCR systems), `R59_DATASET_MANIFEST.csv`  
**Cross-Check Result:** Recomputed all 132 rows (`22 models × 6 scopes`) from raw predictions: **{bench_mismatches} mismatches** against `R59_FONT_BENCHMARK.csv`; **0/200** image SHA-256 mismatches.

---

## 1. Resolution of 16 vs. 18 Training Font Discrepancy

An audit of the training font directories and scripts resolved the 16 vs. 18 font label discrepancy:
- **Actual `.ttf` files on disk:** `fonts/` contains **10** files (`arial.ttf`, `calibri.ttf`, `candara.ttf`, `consola.ttf`, `corbel.ttf`, `cour.ttf`, `segoeui.ttf`, `tahoma.ttf`, `times.ttf`, `verdana.ttf`) and `fonts_new/` contains **6** files (`bahnschrift.ttf`, `CascadiaCode.ttf`, `consolab.ttf`, `ebrima.ttf`, `l_10646.ttf`, `palab.ttf`) — totaling **16 `.ttf` files** (15 unique font families, as `consola.ttf` and `consolab.ttf` are Regular/Bold weights of Consolas).
- **V1 vs. V2 Split:** V1 (`Group A/B`) used 8 training fonts from `fonts/` (holding out `consola` and `tahoma` at 20%). V2Clean (`Group C/D/RandomPadding/VariableWidth`) expanded coverage using the `16`-file pool (`fonts/` + `fonts_new/`). The string `"Geniş 18 font"` in `run_night_training.ps1` was a legacy/rounded script comment rather than 18 distinct font files on disk.
- **Zero Overlap Verified by Font Identifiers:** None of the 5 R59 open-license PostScript/Family identifiers (`DejaVuSans`, `DejaVuSerif`, `DejaVuSansMono`, `Lato-Regular`, `Raleway-Regular`) appear in the 16 training font identifiers.

---

## 2. Font Licenses & Text Authorship Status

{tbl_fonts}

- **Benchmark Sentences (40 lines):** Newly authored specifically for this evaluation benchmark, verified to have **0/40 overlap** with `corpus_line_clean` and `Real30`. **License Status:** Pending formal rights-holder/author CC0-1.0 sign-off prior to public Zenodo release (do not declare CC0 final until signed off by the corresponding author).

---

## 3. Auto-Generated Overall Results (`ALL_5_FONTS`, 200 Lines)

*(Every cell below is computed programmatically from `R59_RAW_PREDICTIONS.csv`; no manual transcription.)*

{tbl_all}

---

## 4. Auto-Generated Per-Font CER Micro (%)

{tbl_per_font}
"""
    (OUT_DIR / "R59_FINAL_VERIFICATION.md").write_text(md, encoding="utf-8")
    print(f"[+] Saved R59_CORRECTED_RESULTS.csv and R59_FINAL_VERIFICATION.md (bench_mismatches={bench_mismatches}).")


if __name__ == "__main__":
    main()
