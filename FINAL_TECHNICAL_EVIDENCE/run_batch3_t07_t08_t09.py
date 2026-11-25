"""Batch 3: T07 (Photo 3 CER), T08 (Legacy GT Counts), T09 (Dataset Counts & Fonts)."""

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd
from fontTools.ttLib import TTFont

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
OUT_DIR = PROJ_ROOT / "FINAL_TECHNICAL_EVIDENCE"


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s).strip()) if pd.notna(s) else ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_t07():
    md = """# AzOCR-V2 — Photo 3 (D3 / B3) CER Discrepancy Reconciliation: 0.10% vs. 0.89% (Task T07)

## 1. Root Cause Summary

The two reported CER values for **Photo 3 (`IMG_5123.jpg`, Times New Roman, 23 lines, 1,015 NFC chars)** are **both real** and originate from **different checkpoints/decoding/post-processing configurations** across two separate evaluations:

| Metric Value | Source File | Model / Key | Decoding / Post-Processing | Edit Dist / Denom | Protocol |
|---:|---|---|---|---|---|
| **0.10%** (`0.0985%`) | `results/photo3_test/photo3_metrics.json` | `v2_rule_chunk` (early CRNN-V2 diagnostic run) | Rule-based chunking, **raw CTC greedy (no SymSpell)** | `1 / 1,015` chars (`22/23` exact) | Diagnostic D3 test |
| **0.00%** | `results/unseen_benchmark/unseen_metrics.json` | `v2_raw` (`runs/line_v2/best.pt`) | `infer_line_smart`, **raw CTC greedy (no SymSpell)** | `0 / 1,015` chars (`23/23` exact) | B1–B15 benchmark (`B3`) |
| **0.89%** | `results/unseen_benchmark/unseen_metrics.json` | `v2_pp` (`runs/line_v2/best.pt`) | `infer_line_smart` + **SymSpell (`max_edit_distance=1`)** | `9 / 1,015` chars (`20/23` exact, WER `2.17%`) | B1–B15 benchmark (`B3`) |

## 2. Why SymSpell Turned 0.00% into 0.89% on Photo 3

As documented in `PHOTO_MAPPING_AUDIT.md` (Section 2), the physical A4 printout of D3/B3 contains **4 real typographical errors in the printed text itself** (`kimliyinir`, `tarixirə`, `hsduğu`, `ənənələnin`).
- **`v2_raw`** transcribed the printed page verbatim (`0 / 1,015` = **0.00% CER**, `23/23` exact).
- **`v2_pp`** applied SymSpell unigram dictionary correction (`data/az_lexicon.txt`) on top of `v2_raw`, altering those printed tokens toward dictionary words and thereby **introducing 9 character edits** relative to the verbatim printed ground truth (`9 / 1,015` = **0.89% CER**, `20/23` exact).
- **`0.10%`** is the earlier `v2_rule_chunk` raw result (`1 / 1,015` = `0.0985%`) from `results/photo3_test/photo3_metrics.json` before SymSpell post-processing was applied.

**Correct Interpretation:** Never mix `v2_rule_chunk` (`0.10%`), `v2_raw` (`0.00%`), and `v2_pp` (`0.89%`); always label whether SymSpell post-processing was enabled.
"""
    (OUT_DIR / "PHOTO3_CER_RECONCILIATION.md").write_text(md, encoding="utf-8")
    print("[T07] Saved PHOTO3_CER_RECONCILIATION.md.")


def run_t08():
    rows = []

    # 1. D1 and D2 from results/real_fix8/per_line_chunk_details.csv
    df_f8 = pd.read_csv(PROJ_ROOT / "results/real_fix8/per_line_chunk_details.csv")
    # Identify column names
    p_col = "photo"
    gt_col = "ref"

    for tag, did, fname, src_img in [
        ("photo 1", "D1", "photo1_times.jpg", PROJ_ROOT / "data/real_docs/photo1_times.jpg"),
        ("photo 2", "D2", "photo2_calibri.jpg", PROJ_ROOT / "data/real_docs/photo2_calibri.jpg"),
    ]:
        sub = df_f8[df_f8[p_col].astype(str).str.lower().str.contains(tag)]
        chars = sum(len(nfc(x)) for x in sub[gt_col])
        rows.append({
            "tier": "Diagnostic", "doc_id": did, "original_filename": fname,
            "line_count": len(sub), "char_count_nfc": chars,
            "normalization": "NFC, strip(), no newline",
            "gt_source": "results/real_fix8/per_line_chunk_details.csv",
            "file_sha256": sha256_file(src_img) if src_img.exists() else "N/A",
        })

    # 2. D3 from results/photo3_test/per_line_chunk_details.csv
    df_p3 = pd.read_csv(PROJ_ROOT / "results/photo3_test/per_line_chunk_details.csv")
    c_d3 = sum(len(nfc(x)) for x in df_p3["ref"])
    rows.append({
        "tier": "Diagnostic", "doc_id": "D3", "original_filename": "photo3_times.jpg",
        "line_count": len(df_p3), "char_count_nfc": c_d3,
        "normalization": "NFC, strip(), no newline",
        "gt_source": "results/photo3_test/per_line_chunk_details.csv",
        "file_sha256": sha256_file(PROJ_ROOT / "results/photo3_test/original_photo.jpg"),
    })

    # 3. D4 from results/photo4_sans/per_line_chunk_details.csv
    df_p4 = pd.read_csv(PROJ_ROOT / "results/photo4_sans/per_line_chunk_details.csv")
    c_d4 = sum(len(nfc(x)) for x in df_p4["ref"])
    rows.append({
        "tier": "Diagnostic", "doc_id": "D4", "original_filename": "photo4_sans.jpg",
        "line_count": len(df_p4), "char_count_nfc": c_d4,
        "normalization": "NFC, strip(), no newline",
        "gt_source": "results/photo4_sans/per_line_chunk_details.csv",
        "file_sha256": sha256_file(PROJ_ROOT / "results/photo4_sans/original_photo.jpg"),
    })

    # 4. D5 from results/real_fix9/per_line_chunk_details.csv
    df_p5 = pd.read_csv(PROJ_ROOT / "results/real_fix9/per_line_chunk_details.csv")
    c_d5 = sum(len(nfc(x)) for x in df_p5["ref"])
    rows.append({
        "tier": "Diagnostic", "doc_id": "D5", "original_filename": "IMG_5130.HEIC (photo5.jpg)",
        "line_count": len(df_p5), "char_count_nfc": c_d5,
        "normalization": "NFC, strip(), no newline",
        "gt_source": "results/real_fix9/per_line_chunk_details.csv",
        "file_sha256": sha256_file(PROJ_ROOT / "data/real_test5/photo5.jpg"),
    })

    # 5. B1-B15 from results/easyocr_295/easyocr_295_predictions.csv & PHOTO_MAPPING.csv
    df_pm = pd.read_csv(PROJ_ROOT / "PHOTO_MAPPING.csv")
    df_e295 = pd.read_csv(PROJ_ROOT / "results/easyocr_295/easyocr_295_predictions.csv")
    for _, pm in df_pm[df_pm["Tier"] == "Benchmark"].iterrows():
        did = pm["ID"]
        b_num = int(did.replace("B", ""))
        b_tag = f"B{b_num}"
        sub = df_e295[df_e295["doc_id"] == b_tag]
        chars = sum(len(nfc(x)) for x in sub["ground_truth"])
        img_p = PROJ_ROOT / "data/phone_photos_15" / f"{pm['Code_or_Name']}.jpg"
        rows.append({
            "tier": "Benchmark", "doc_id": did, "original_filename": f"{pm['Code_or_Name']}.jpg",
            "line_count": len(sub), "char_count_nfc": chars,
            "normalization": "NFC, strip(), no newline",
            "gt_source": "results/easyocr_295/easyocr_295_predictions.csv",
            "file_sha256": sha256_file(img_p) if img_p.exists() else pm["SHA256"],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "LEGACY_GT_CHARACTER_COUNTS.csv", index=False)

    d_tot = df_out[df_out["tier"] == "Diagnostic"]["char_count_nfc"].sum()
    b1_3_tot = df_out[df_out["doc_id"].isin(["B01", "B02", "B03"])]["char_count_nfc"].sum()
    b_tot = df_out[df_out["tier"] == "Benchmark"]["char_count_nfc"].sum()
    print(f"[T08] Saved LEGACY_GT_CHARACTER_COUNTS.csv (D1-D5={d_tot}, B1-B3={b1_3_tot}, B1-B15={b_tot}).")


def run_t09():
    # Verify exact corpus_line_clean and synth datasets
    tr_clean = [nfc(l) for l in (PROJ_ROOT / "data/corpus_line_clean/train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    va_clean = [nfc(l) for l in (PROJ_ROOT / "data/corpus_line_clean/val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    te_clean = [nfc(l) for l in (PROJ_ROOT / "data/corpus_line_clean/test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    set_tr, set_va, set_te = set(tr_clean), set(va_clean), set(te_clean)

    v1_tr_labels = [nfc(l.split("\t", 1)[1]) for l in (PROJ_ROOT / "data/synth_line_v1_clean/train/labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    v1_va_labels = [nfc(l.split("\t", 1)[1]) for l in (PROJ_ROOT / "data/synth_line_v1_clean/val/labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    v2_tr_labels = [nfc(l.split("\t", 1)[1]) for l in (PROJ_ROOT / "data/synth_line_v2_clean/train/labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    v2_va_labels = [nfc(l.split("\t", 1)[1]) for l in (PROJ_ROOT / "data/synth_line_v2_clean/val/labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]
    v2_te_labels = [nfc(l.split("\t", 1)[1]) for l in (PROJ_ROOT / "data/synth_line_v2_clean/test/labels.tsv").read_text(encoding="utf-8").splitlines() if l.strip()]

    leak_rows = [
        ("corpus_line_clean", "train", len(tr_clean), len(set_tr), len(set_tr & set_va), len(set_tr & set_te)),
        ("corpus_line_clean", "val", len(va_clean), len(set_va), 0, len(set_va & set_te)),
        ("corpus_line_clean", "test", len(te_clean), len(set_te), 0, 0),
        ("synth_line_v1_clean", "train", len(v1_tr_labels), len(set(v1_tr_labels)), len(set(v1_tr_labels) & set(v1_va_labels)), 0),
        ("synth_line_v1_clean", "val", len(v1_va_labels), len(set(v1_va_labels)), 0, 0),
        ("synth_line_v2_clean", "train", len(v2_tr_labels), len(set(v2_tr_labels)), len(set(v2_tr_labels) & set(v2_va_labels)), len(set(v2_tr_labels) & set(v2_te_labels))),
        ("synth_line_v2_clean", "val", len(v2_va_labels), len(set(v2_va_labels)), 0, len(set(v2_va_labels) & set(v2_te_labels))),
        ("synth_line_v2_clean", "test", len(v2_te_labels), len(set(v2_te_labels)), 0, 0),
    ]
    pd.DataFrame(leak_rows, columns=["dataset", "split", "total_samples", "unique_texts", "overlap_with_val", "overlap_with_test"]).to_csv(
        OUT_DIR / "DATASET_LEAKAGE_AUDIT.csv", index=False
    )

    # Font inventory
    font_rows = []
    for folder in ["fonts", "fonts_new"]:
        for fp in sorted((PROJ_ROOT / folder).glob("*.ttf")):
            ft = TTFont(str(fp), lazy=True)
            names = {r.nameID: r.toUnicode() for r in ft["name"].names if r.nameID in (1, 2, 6)}
            font_rows.append({
                "folder": folder, "filename": fp.name,
                "family": names.get(1), "subfamily": names.get(2),
                "postscript_name": names.get(6), "sha256": sha256_file(fp),
            })
    pd.DataFrame(font_rows).to_csv(OUT_DIR / "GPU_FONT_INVENTORY_VERIFIED.csv", index=False)

    md = rf"""# AzOCR-V2 — Dataset Count & Font Inventory Reconciliation (Task T09)

## 1. Exact Relationship Between Historical & Clean Dataset Counts

- **`10,400` corpus lines (`corpus_line/train.txt`):** Original training corpus line count. Also equals the exact `synth_line_v1/train` image count (`10,400`), whereas `10,000` was an approximate/rounded historical summary number in earlier notes.
- **Exact Leakage Breakdown (before cleaning, from `clean_dataset_summary.json`):**
  - `100` exact duplicates between `train` and `val`
  - `84` exact duplicates between `train` and `test`
  - `25` overlapping lines present in both sets ($100 + 84 - 25 = \mathbf{{159}}$ unique affected texts removed from `train`).
- **`10,241` V1Clean samples (`synth_line_v1_clean/train`):** $10,400 - 159 = \mathbf{{10,241}}$ samples (verified `0` train/val overlap after cleaning).
- **`16,000` historical V2 synthetic images (`synth_line_v2/train`):** Rendered from `10,176` unique corpus texts (`16,000` total images).
- **`15,754` V2Clean samples (`synth_line_v2_clean/train`):** Removing all images whose text matched the `159` leaked texts removed `246` images ($16,000 - 246 = \mathbf{{15,754}}$, verified `0` train/val and `0` train/test overlap).

## 2. Font Inventory Reconciliation (`14` vs `16 files / 15 families`)

- On disk, `fonts/` has **10** `.ttf` files and `fonts_new/` has **6** `.ttf` files = **16 `.ttf` files total**, representing **15 unique font families** (`consola.ttf` and `consolab.ttf` are Regular and Bold of Consolas).
- Historical V1 held out 20% of `fonts/` (`consola`, `tahoma`), leaving **8** training fonts.
- Historical V2 (`14` fonts in `ablation_2x2_results.json`) vs `16` files on disk reflects whether the 2 held-out fonts were excluded (`16 - 2 = 14`) vs counting all `16` `.ttf` files in `fonts/ + fonts_new/`.
"""
    (OUT_DIR / "DATASET_COUNT_RECONCILIATION.md").write_text(md, encoding="utf-8")
    print("[T09] Saved DATASET_COUNT_RECONCILIATION.md, DATASET_LEAKAGE_AUDIT.csv, GPU_FONT_INVENTORY_VERIFIED.csv.")


if __name__ == "__main__":
    run_t07()
    run_t08()
    run_t09()
