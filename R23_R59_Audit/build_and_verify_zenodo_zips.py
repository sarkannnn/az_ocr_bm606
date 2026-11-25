"""Build and rigorously verify the two final Zenodo release ZIP archives:
1. AzOCR_v1_Code_Benchmarks_Results.zip
2. AzOCR_v1_Models_18Official.zip
"""

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
DRAFT_DIR = PROJ_ROOT / "AzOCR_Zenodo_v1_DRAFT"
RUNS_ROOT = Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\runs")
RELEASE_DIR = PROJ_ROOT / "AzOCR_Zenodo_v1_Release"

GROUPS = [
    "Group_A_V1Clean_BasicAug",
    "Group_B_V1Clean_PhotoAug",
    "Group_C_V2Clean_BasicAug",
    "Group_D_V2Clean_PhotoAug",
    "Exp_RandomPadding_GroupD",
    "Exp_VariableWidth_GroupD",
]
SEEDS = [42, 43, 44]

FORBIDDEN_FILENAMES = {
    "ground_truth_lines.csv", "verified_alignment.csv", "raw_predictions.csv",
    "raw_predictions_all_18_models.csv", "last.pt",
    "arial.ttf", "calibri.ttf", "candara.ttf", "consola.ttf", "corbel.ttf",
    "cour.ttf", "segoeui.ttf", "tahoma.ttf", "times.ttf", "verdana.ttf",
    "bahnschrift.ttf", "CascadiaCode.ttf", "consolab.ttf", "ebrima.ttf",
    "l_10646.ttf", "palab.ttf",
}

SECRET_PATTERNS = [
    ("HuggingFace Token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("OpenAI Key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GitHub Token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z\-_]{30,}\b")),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Private Key Block", re.compile(r"-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----")),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def sha256_zip_entry(zf: zipfile.ZipFile, arcname: str) -> str:
    h = hashlib.sha256()
    with zf.open(arcname) as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Refresh DRAFT_SHA256_MANIFEST.csv (excluding itself)
    manifest_path = DRAFT_DIR / "manifests_and_licenses" / "DRAFT_SHA256_MANIFEST.csv"
    sha_rows = []
    for f in sorted(DRAFT_DIR.rglob("*")):
        if f.is_file() and f != manifest_path:
            rel = f.relative_to(DRAFT_DIR).as_posix()
            sha_rows.append({
                "relative_path": rel,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            })
    pd.DataFrame(sha_rows).to_csv(manifest_path, index=False)
    print(f"[+] Refreshed DRAFT_SHA256_MANIFEST.csv ({len(sha_rows)} files).")

    # 2. Build AzOCR_v1_Code_Benchmarks_Results.zip
    code_zip = RELEASE_DIR / "AzOCR_v1_Code_Benchmarks_Results.zip"
    if code_zip.exists():
        code_zip.unlink()

    code_src_map = {}
    with zipfile.ZipFile(code_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DRAFT_DIR.rglob("*")):
            if f.is_file():
                rel = f.relative_to(DRAFT_DIR).as_posix()
                z.write(f, arcname=rel)
                code_src_map[rel] = (f.stat().st_size, sha256_file(f))
    print(f"[+] Built {code_zip.name}: {code_zip.stat().st_size / (1024*1024):.2f} MB ({len(code_src_map)} files)")

    # 3. Build AzOCR_v1_Models_18Official.zip (18 official best.pt + MODEL_LICENSE_NOTICE.md + LICENSE_APACHE_2.0.txt)
    models_zip = RELEASE_DIR / "AzOCR_v1_Models_18Official.zip"
    if models_zip.exists():
        models_zip.unlink()

    models_src_map = {}
    with zipfile.ZipFile(models_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for extra_doc in ["MODEL_LICENSE_NOTICE.md", "LICENSE_APACHE_2.0.txt"]:
            doc_path = RELEASE_DIR / extra_doc
            z.write(doc_path, arcname=extra_doc)
            models_src_map[extra_doc] = (doc_path.stat().st_size, sha256_file(doc_path))
        for gid in GROUPS:
            for s in SEEDS:
                rname = f"{gid}_seed{s}"
                best_pt = RUNS_ROOT / rname / "best.pt"
                assert best_pt.exists(), f"Missing {best_pt}"
                arc = f"{rname}/best.pt"
                print(f"  Packing {arc}...")
                z.write(best_pt, arcname=arc)
                models_src_map[arc] = (best_pt.stat().st_size, sha256_file(best_pt))
    print(f"[+] Built {models_zip.name}: {models_zip.stat().st_size / (1024*1024):.2f} MB ({len(models_src_map)} files)")

    # 4. Re-open both ZIPs, run testzip(), verify EVERY internal file SHA-256, and scan for forbidden/secret items
    print("\n[+] Verifying integrity and scanning both ZIPs...")
    forbidden_hits = []
    secret_hits = []
    local_path_hits = []

    for zp, src_map in [(code_zip, code_src_map), (models_zip, models_src_map)]:
        with zipfile.ZipFile(zp, "r") as z:
            corrupt = z.testzip()
            if corrupt is not None:
                raise RuntimeError(f"Corrupt entry {corrupt} in {zp.name}")
            names = z.namelist()
            assert len(names) == len(src_map), f"Entry count mismatch in {zp.name}"
            for arc in names:
                # Verify SHA-256
                exp_sz, exp_sha = src_map[arc]
                info = z.getinfo(arc)
                assert info.file_size == exp_sz, f"Size mismatch on {arc}"
                got_sha = sha256_zip_entry(z, arc)
                assert got_sha == exp_sha, f"SHA-256 mismatch on {arc}"

                # Forbidden filename / path checks
                bname = Path(arc).name
                if bname in FORBIDDEN_FILENAMES or "reproducibility_check" in arc.lower() or "extracted_photos" in arc.lower():
                    forbidden_hits.append(f"{zp.name}:{arc}")
                if zp == code_zip and bname.endswith(".pt"):
                    forbidden_hits.append(f"{zp.name}:{arc} (.pt inside code zip)")
                if zp == models_zip and arc not in ("MODEL_LICENSE_NOTICE.md", "LICENSE_APACHE_2.0.txt") and not arc.endswith("/best.pt"):
                    forbidden_hits.append(f"{zp.name}:{arc} (unexpected file inside models zip)")
                if re.search(r"P\d{2}_pdf_embedded\.jpeg", arc):
                    forbidden_hits.append(f"{zp.name}:{arc}")

                # Secret scan on text files inside code zip
                if zp == code_zip and Path(arc).suffix.lower() in {".py", ".md", ".csv", ".json", ".jsonl", ".yaml", ".txt"}:
                    txt = z.read(arc).decode("utf-8", errors="ignore")
                    for sname, pat in SECRET_PATTERNS:
                        if pat.search(txt):
                            secret_hits.append(f"{arc} ({sname})")
                    if re.search(r"(serxan|C:[\\\\/]Users|D:[\\\\/]laragon)", txt, re.IGNORECASE):
                        local_path_hits.append(arc)

    assert not forbidden_hits, f"Forbidden files found: {forbidden_hits}"
    assert not secret_hits, f"Secret tokens found: {secret_hits}"
    assert not local_path_hits, f"Personal/local Windows paths found: {local_path_hits}"

    code_zip_sha = sha256_file(code_zip)
    models_zip_sha = sha256_file(models_zip)
    code_zip_mb = code_zip.stat().st_size / (1024 * 1024)
    models_zip_mb = models_zip.stat().st_size / (1024 * 1024)

    print(f"  {code_zip.name}: {code_zip_mb:.2f} MB | SHA256: {code_zip_sha}")
    print(f"  {models_zip.name}: {models_zip_mb:.2f} MB | SHA256: {models_zip_sha}")
    print(f"  Forbidden hits: {len(forbidden_hits)} | Secret hits: {len(secret_hits)} | Local username path refs in text: {len(local_path_hits)}")

    # 5. Write FINAL_PUBLICATION_LICENSE_AUDIT.md
    models_table_rows = "\n".join(
        f"| `{arc}` | {sz / (1024*1024):.2f} MB ({sz:,} bytes) | `{sha}` | Apache-2.0 ✅ |"
        for arc, (sz, sha) in models_src_map.items() if arc.endswith(".pt")
    )

    report = f"""# AzOCR-V2 — Final Publication License Audit (`FINAL_PUBLICATION_LICENSE_AUDIT.md`)

**Verification Date:** 2026-11-25  
**Release Folder:** `D:\\laragon\\www\\az_ocr_bm606\\AzOCR_Zenodo_v1_Release`  
**Zenodo Publication Status:** **STAGED LOCALLY — NOT PUBLISHED TO ZENODO**

---

## 1. Final Release ZIP Archives Summary

| Archive Name | Full Path | Size | File Count | Archive SHA-256 | Integrity (`testzip` + Entry SHA-256) |
|---|---|---:|---:|---|---|
| **`AzOCR_v1_Code_Benchmarks_Results.zip`** | `D:\\laragon\\www\\az_ocr_bm606\\AzOCR_Zenodo_v1_Release\\AzOCR_v1_Code_Benchmarks_Results.zip` | **{code_zip_mb:.2f} MB** ({code_zip.stat().st_size:,} bytes) | {len(code_src_map)} | `{code_zip_sha}` | ✅ 100% Passed ({len(code_src_map)}/{len(code_src_map)} matched) |
| **`AzOCR_v1_Models_18Official.zip`** | `D:\\laragon\\www\\az_ocr_bm606\\AzOCR_Zenodo_v1_Release\\AzOCR_v1_Models_18Official.zip` | **{models_zip_mb:.2f} MB** ({models_zip.stat().st_size:,} bytes) | {len(models_src_map)} (18 `best.pt` + Apache-2.0 + Notice) | `{models_zip_sha}` | ✅ 100% Passed ({len(models_src_map)}/{len(models_src_map)} matched) |

---

## 2. Scientific Co-Authorship vs. Legal Rights Ownership

- **Scientific Study Co-Authors:** Sarkan Mammadov, Abdul Jafarov, and **Tuba Karagül Yıldız** (co-author and academic supervisor).
- **Legal Rights Holders (Confirmed Open Release):** **Sarkan Mammadov** and **Abdul Jafarov** have confirmed open publication of:
  - Project source code under **MIT License** (`LICENSE_CODE_MIT.txt`)
  - R59 newly authored benchmark sentences, rendered PNGs, and result tables under **CC0 1.0 Universal** (`LICENSE_BENCHMARK_CC0_1.0.txt`)
  - All 18 official CRNN+CTC model checkpoints under **Apache License 2.0** (`LICENSE_APACHE_2.0.txt`, `MODEL_LICENSE_NOTICE.md`)
- **Strict Separation:** Tuba Karagül Yıldız is attributed strictly as scientific co-author and academic supervisor; no unconfirmed legal copyright ownership is created or implied.

---

## 3. Third-Party Components & Institutional IP Check

| Component | License / Terms | Handling in Zenodo Archives |
|---|---|---|
| **DejaVu Fonts** (`DejaVuSans.ttf`, `DejaVuSerif.ttf`, `DejaVuSansMono.ttf`) | DejaVu / Bitstream Vera License (`LICENSE_DEJAVU.txt`) | Original license & notices preserved verbatim |
| **Lato Font** (`Lato-Regular.ttf`) | SIL Open Font License 1.1 (`LICENSE_LATO_OFL1.1.txt`) | Original notice (`tyPoland Lukasz Dziedzic`, RFN "Lato") preserved verbatim |
| **Raleway Font** (`Raleway-Regular.ttf`) | SIL Open Font License 1.1 (`LICENSE_RALEWAY_OFL1.1.txt`) | Original notice (`Matt McInerney` et al., RFN "Raleway") preserved verbatim |
| **Proprietary Windows Fonts** (`fonts/`, `fonts_new/`) | Microsoft / Proprietary EULAs | **Excluded**; listed by SHA-256 only in `TRAINING_FONTS_ACQUISITION.md` |
| **Real30 Smartphone Photos & GTs** | Unconfirmed third-party/document rights | **Excluded** from both public archives |
| **Runtime Libraries** (PyTorch, OpenCV, NumPy, Pandas, Tesseract, EasyOCR, PaddleOCR) | BSD-3-Clause / Apache-2.0 | Referenced via `requirements.txt` only; no third-party binary weights redistributed |
| **Institutional / University IP Policy** | Governance check | Authors should verify institutional thesis/research IP rules prior to clicking Publish on Zenodo |

---

## 4. Security & Exclusion Scan Results

- **Personal / Local Windows Paths (`C:\\Users\\...`, `D:\\laragon\\...`):** ✅ **0 hits**
- **API Keys / Tokens / Passwords / Private Keys:** ✅ **0 hits**
- **Forbidden Files (`Real30` photos, `last.pt`, `reproducibility_check`, proprietary `.ttf`):** ✅ **0 hits**

---

## 5. `AzOCR_v1_Models_18Official.zip` Apache-2.0 Scope (18 Official `best.pt`)

| Archive Path | Uncompressed Size | Verified SHA-256 | License |
|---|---:|---|---|
{models_table_rows}
"""
    (RELEASE_DIR / "FINAL_PUBLICATION_LICENSE_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"\n[+] Saved {RELEASE_DIR / 'FINAL_PUBLICATION_LICENSE_AUDIT.md'}")


if __name__ == "__main__":
    main()
