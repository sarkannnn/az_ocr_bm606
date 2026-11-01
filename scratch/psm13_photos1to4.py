"""Tesseract PSM 7 vs PSM 13 on the saved crops of Photos 1-4.

Re-runs Tesseract (aze, tessdata/) on the already-saved full-line crops and
rule-based chunks of Photos 1-4 (no model re-training, no re-detection) and
reports CER/WER for both page segmentation modes. PSM 7 is re-run as a
consistency check against the values stored in the original per-line CSVs.

Usage: python scratch/psm13_photos1to4.py
Output: results/psm13_photos1to4/metrics.json, per_line.csv, report.md
"""
import csv, json, os, sys
from pathlib import Path

import pytesseract
from PIL import Image
from rapidfuzz.distance import Levenshtein as L

ROOT = Path(__file__).resolve().parent.parent
# tessdata is copied to an ASCII-only path (Tesseract cannot open paths with non-ASCII characters here)
_td = Path(os.environ.get("TEMP", ".")) / "az_tessdata"
_td.mkdir(exist_ok=True)
import shutil; shutil.copy(ROOT / "tessdata" / "aze.traineddata", _td / "aze.traineddata")
os.environ["TESSDATA_PREFIX"] = str(_td)
cmd = os.environ.get("TESSERACT_CMD") or str(ROOT / "scratch" / "_tesseract" / "tesseract.exe")
if Path(cmd).exists():
    pytesseract.pytesseract.tesseract_cmd = cmd

OUT = ROOT / "results" / "psm13_photos1to4"
OUT.mkdir(parents=True, exist_ok=True)


def ocr(path, psm):
    img = Image.open(path).convert("RGB")
    return pytesseract.image_to_string(img, lang="aze", config=f"--psm {psm}").strip()


def metrics(refs, hyps):
    ce = sum(L.distance(r, h) for r, h in zip(refs, hyps))
    cn = sum(len(r) for r in refs)
    we = sum(L.distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    wn = sum(len(r.split()) for r in refs)
    exact = sum(r == h for r, h in zip(refs, hyps))
    return {"cer": 100 * ce / cn, "wer": 100 * we / wn, "char_errors": ce,
            "chars": cn, "exact_lines": exact, "n_lines": len(refs)}


def read(csvp):
    with open(csvp, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# (name, csv, base dir, full-crop pattern, chunk column, joined col, full col, row filter)
SETS = []
r8 = ROOT / "results" / "real_fix8"
rows8 = read(r8 / "per_line_chunk_details.csv")
for tag, name, crops in (("Photo 1", "Photo 1 (Times)", "crops_times"),
                         ("Photo 2", "Photo 2 (Calibri)", "crops_calibri")):
    rr = [r for r in rows8 if r["photo"].startswith(tag)]
    SETS.append((name, r8, rr, lambda r, c=crops: r8 / c / f"line_{int(r['line_idx']):02d}.png",
                 "chunk_crop_paths", "tess_rule_joined", "tess_full"))
for name, d in (("Photo 3 (Times)", "photo3_test"), ("Photo 4 (Arial)", "photo4_sans")):
    base = ROOT / "results" / d
    rr = read(base / "per_line_chunk_details.csv")
    SETS.append((name, base, rr, lambda r, b=base: b / "crops" / f"line_{int(r['line_idx']):02d}.png",
                 "chunk_paths", "tess_rule_joined" if d == "photo4_sans" else "tess_joined", "tess_full"))

summary, perline = {}, []
for name, base, rows, fullpath, chunkcol, joincol, fullcol in SETS:
    refs = [r["ref"].strip() for r in rows]
    res = {m: [] for m in ("full7", "full13", "chunk7", "chunk13")}
    for r in rows:
        fp = fullpath(r)
        res["full7"].append(ocr(fp, 7)); res["full13"].append(ocr(fp, 13))
        chunks = [base / p.strip() for p in r[chunkcol].split(";") if p.strip()]
        res["chunk7"].append(" ".join(ocr(c, 7) for c in chunks))
        res["chunk13"].append(" ".join(ocr(c, 13) for c in chunks))
    summary[name] = {
        "n_lines": len(refs),
        "tess_full_psm7": metrics(refs, res["full7"]),
        "tess_full_psm13": metrics(refs, res["full13"]),
        "tess_chunk_psm7": metrics(refs, res["chunk7"]),
        "tess_chunk_psm13": metrics(refs, res["chunk13"]),
        # consistency check: stored PSM 7 outputs from the original run
        "stored_tess_chunk_psm7": metrics(refs, [r[joincol].strip() for r in rows]),
        "stored_tess_full_psm7": metrics(refs, [r[fullcol].strip() for r in rows]),
    }
    for i, r in enumerate(rows):
        perline.append({"photo": name, "line": i, "ref": refs[i], **{k: res[k][i] for k in res}})
    print(name, {k: round(v["cer"], 2) for k, v in summary[name].items() if isinstance(v, dict)}, flush=True)

(OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
with open(OUT / "per_line.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(perline[0].keys())); w.writeheader(); w.writerows(perline)

ver = pytesseract.get_tesseract_version()
lines = [f"# Tesseract PSM 7 vs PSM 13 on Photos 1-4 (Tesseract {ver}, aze)\n",
         "CER (%) / WER (%) / exact lines. 'stored' = PSM 7 outputs saved in the original run (consistency check).\n",
         "| Photo | n | Chunk PSM 7 | Chunk PSM 13 | Full PSM 7 | Full PSM 13 | stored Chunk PSM 7 | stored Full PSM 7 |",
         "|---|---|---|---|---|---|---|---|"]
for name, s in summary.items():
    f = lambda m: f"{m['cer']:.2f} / {m['wer']:.2f} / {m['exact_lines']}"
    lines.append(f"| {name} | {s['n_lines']} | {f(s['tess_chunk_psm7'])} | {f(s['tess_chunk_psm13'])} | "
                 f"{f(s['tess_full_psm7'])} | {f(s['tess_full_psm13'])} | {f(s['stored_tess_chunk_psm7'])} | {f(s['stored_tess_full_psm7'])} |")
(OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))


