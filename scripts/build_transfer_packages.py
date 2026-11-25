import os
import sys
import hashlib
import zipfile
import csv
from pathlib import Path

SOURCE_ROOT = Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results")
DEST_DIR = Path(r"D:\laragon\www\az_ocr_bm606\AzOCR_ChatGPT_Transfer")
REPO_ROOT = Path(r"d:\laragon\www\az_ocr_bm606")

GROUPS = [
    ("02_GROUP_A.zip", "Group_A_V1Clean_BasicAug"),
    ("03_GROUP_B.zip", "Group_B_V1Clean_PhotoAug"),
    ("04_GROUP_C.zip", "Group_C_V2Clean_BasicAug"),
    ("05_GROUP_D.zip", "Group_D_V2Clean_PhotoAug"),
    ("06_RANDOM_PADDING.zip", "Exp_RandomPadding_GroupD"),
    ("07_VARIABLE_WIDTH.zip", "Exp_VariableWidth_GroupD"),
]

SEEDS = [42, 43, 44]

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_zip_entry_sha256(zip_path, arcname):
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(arcname) as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
    return h.hexdigest()

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    print("=== Step 1: Building 01_METADATA.zip ===")
    meta_zip_path = DEST_DIR / "01_METADATA.zip"
    meta_files_to_add = [] # (abs_path, arcname)

    # 1. Summary files from runs/
    for fname in ["experiments_summary.json", "experiments_summary.md"]:
        p = SOURCE_ROOT / "runs" / fname
        if p.exists():
            meta_files_to_add.append((p, f"summary/{fname}"))

    # 2. Root files
    for fname in ["results_per_run.csv", "README.md"]:
        p = SOURCE_ROOT / fname
        if p.exists():
            meta_files_to_add.append((p, f"meta/{fname}"))

    # 3. logs
    for p in (SOURCE_ROOT / "logs").glob("*"):
        if p.is_file():
            meta_files_to_add.append((p, f"logs/{p.name}"))

    # 4. code/ files
    for p in (SOURCE_ROOT / "code").glob("*"):
        if p.is_file():
            meta_files_to_add.append((p, f"code/{p.name}"))

    # 5. Core repo files (model.py, common.py, requirements.txt)
    for fname in ["model.py", "common.py", "requirements.txt"]:
        p = REPO_ROOT / fname
        if p.exists():
            meta_files_to_add.append((p, f"codebase/{fname}"))

    # 6. Each of the 18 runs: config.yaml, run_summary.json, train_log.jsonl
    runs_dir = SOURCE_ROOT / "runs"
    for _, grp_name in GROUPS:
        for seed in SEEDS:
            run_name = f"{grp_name}_seed{seed}"
            run_path = runs_dir / run_name
            for item in ["config.yaml", "run_summary.json", "train_log.jsonl"]:
                ip = run_path / item
                if ip.exists():
                    meta_files_to_add.append((ip, f"runs_meta/{run_name}/{item}"))

    # 7. Reproducibility check files (config.yaml, run_summary.json, train_log.jsonl)
    repro_dir = SOURCE_ROOT / "reproducibility_check" / "Exp_VariableWidth_GroupD_seed44_RTX2050"
    if repro_dir.exists():
        for item in ["config.yaml", "run_summary.json", "train_log.jsonl"]:
            ip = repro_dir / item
            if ip.exists():
                meta_files_to_add.append((ip, f"reproducibility_check_meta/Exp_VariableWidth_GroupD_seed44_RTX2050/{item}"))

    # Write 01_METADATA.zip
    if meta_zip_path.exists():
        meta_zip_path.unlink()
    with zipfile.ZipFile(meta_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for src, arc in meta_files_to_add:
            # Verify no .pt files
            assert not src.name.endswith(".pt"), f"Found .pt file in metadata list: {src}"
            z.write(src, arcname=arc)
            orig_sha = get_sha256(src)
            manifest_rows.append({
                "package": "01_METADATA.zip",
                "archive_path": arc,
                "original_source_path": str(src),
                "file_type": "metadata",
                "size_bytes": src.stat().st_size,
                "sha256": orig_sha,
                "verified_match": "YES"
            })
    print(f"Created 01_METADATA.zip: {meta_zip_path.stat().st_size / (1024*1024):.2f} MB ({len(meta_files_to_add)} files)")

    # === Step 2: Building Group Archives (02 - 07) ===
    for zip_filename, grp_name in GROUPS:
        zip_path = DEST_DIR / zip_filename
        if zip_path.exists():
            zip_path.unlink()
        print(f"\nBuilding {zip_filename} for {grp_name}...")
        
        # Collect best.pt files
        files_to_pack = []
        for seed in SEEDS:
            run_name = f"{grp_name}_seed{seed}"
            best_pt = runs_dir / run_name / "best.pt"
            assert best_pt.exists(), f"Missing best.pt for {run_name}"
            files_to_pack.append((best_pt, f"{run_name}/best.pt"))

        # Write zip with compression
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for src, arc in files_to_pack:
                print(f"  Adding {arc} ({src.stat().st_size / (1024*1024):.2f} MB)...")
                z.write(src, arcname=arc)
                orig_sha = get_sha256(src)
                manifest_rows.append({
                    "package": zip_filename,
                    "archive_path": arc,
                    "original_source_path": str(src),
                    "file_type": "best_checkpoint",
                    "size_bytes": src.stat().st_size,
                    "sha256": orig_sha,
                    "verified_match": "PENDING"
                })

        zip_size_mb = zip_path.stat().st_size / (1024*1024)
        print(f"Created {zip_filename}: {zip_size_mb:.2f} MB (Limit: 450 MB, OK: {zip_size_mb < 450})")
        assert zip_size_mb < 450, f"Package {zip_filename} exceeds 450 MB!"

    # === Step 3: Register last.pt checkpoints in manifest (as unarchived source records) ===
    print("\nRegistering last.pt files in manifest...")
    for _, grp_name in GROUPS:
        for seed in SEEDS:
            run_name = f"{grp_name}_seed{seed}"
            last_pt = runs_dir / run_name / "last.pt"
            if last_pt.exists():
                sha = get_sha256(last_pt)
                manifest_rows.append({
                    "package": "NOT_ARCHIVED_SOURCE_ONLY",
                    "archive_path": "N/A",
                    "original_source_path": str(last_pt),
                    "file_type": "last_checkpoint",
                    "size_bytes": last_pt.stat().st_size,
                    "sha256": sha,
                    "verified_match": "SOURCE_RECORDED"
                })

    # Reproducibility run last.pt & best.pt
    repro_best = repro_dir / "best.pt"
    if repro_best.exists():
        manifest_rows.append({
            "package": "NOT_ARCHIVED_SOURCE_ONLY",
            "archive_path": "N/A",
            "original_source_path": str(repro_best),
            "file_type": "reproducibility_best_rtx2050",
            "size_bytes": repro_best.stat().st_size,
            "sha256": get_sha256(repro_best),
            "verified_match": "SOURCE_RECORDED"
        })
    repro_last = repro_dir / "last.pt"
    if repro_last.exists():
        manifest_rows.append({
            "package": "NOT_ARCHIVED_SOURCE_ONLY",
            "archive_path": "N/A",
            "original_source_path": str(repro_last),
            "file_type": "reproducibility_last_rtx2050",
            "size_bytes": repro_last.stat().st_size,
            "sha256": get_sha256(repro_last),
            "verified_match": "SOURCE_RECORDED"
        })

    # === Step 4: Verification of all ZIP files ===
    print("\n=== Step 4: Verifying Archive Integrity and Hashes ===")
    all_zips = [DEST_DIR / "01_METADATA.zip"] + [DEST_DIR / z for z, _ in GROUPS]
    for zp in all_zips:
        print(f"Testing zip integrity: {zp.name}...")
        with zipfile.ZipFile(zp, 'r') as z:
            corrupt = z.testzip()
            if corrupt is not None:
                raise RuntimeError(f"Corrupt entry {corrupt} in {zp.name}!")
        print(f"  {zp.name} integrity test PASSED.")

    # Check extracted hashes against original manifest hashes
    for row in manifest_rows:
        pkg = row["package"]
        if pkg != "NOT_ARCHIVED_SOURCE_ONLY":
            zp = DEST_DIR / pkg
            arc = row["archive_path"]
            extracted_sha = get_zip_entry_sha256(zp, arc)
            if extracted_sha != row["sha256"]:
                raise RuntimeError(f"Hash mismatch for {arc} in {pkg}!")
            row["verified_match"] = "YES"

    # === Step 5: Write SHA256_MANIFEST.csv ===
    manifest_csv = DEST_DIR / "SHA256_MANIFEST.csv"
    with open(manifest_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["package", "archive_path", "original_source_path", "file_type", "size_bytes", "sha256", "verified_match"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWritten {manifest_csv} ({len(manifest_rows)} entries).")

    # === Step 6: Write UPLOAD_ORDER.txt ===
    upload_order_txt = DEST_DIR / "UPLOAD_ORDER.txt"
    lines = [
        "================================================================================",
        "AzOCR-V2 GPU Training Results — ChatGPT Transfer Packages Upload Order",
        "Target Platform: ChatGPT (Standard 512 MB per-file limit, safe ceiling: 450 MB)",
        "================================================================================",
        "",
        "Total Packages to Upload: 7 ZIP Archives",
        "",
        "Step 1: Upload Metadata Package First",
        "--------------------------------------------------------------------------------",
        "  1. 01_METADATA.zip",
        "     - Content: Comprehensive summaries, CSV results, training logs, configs,",
        "                training code, codebase modules (model.py, common.py), and patches.",
        "     - Checkpoint files (.pt): 0 (Strictly excluded).",
        "     - Purpose: Enables immediate analysis of training curves, CER/WER tables,",
        "                hyperparameters, and architecture without waiting for heavy weights.",
        "",
        "Step 2: Upload Official Baseline Experiments (Clean Datasets)",
        "--------------------------------------------------------------------------------",
        "  2. 02_GROUP_A.zip",
        "     - Content: Group_A_V1Clean_BasicAug (seeds 42, 43, 44 best.pt).",
        "     - Baseline V1 clean data with standard affine/noise augmentation.",
        "",
        "  3. 03_GROUP_B.zip",
        "     - Content: Group_B_V1Clean_PhotoAug (seeds 42, 43, 44 best.pt).",
        "     - Baseline V1 clean data with realistic photographic distortions.",
        "",
        "Step 3: Upload V2 Improved Dataset Experiments",
        "--------------------------------------------------------------------------------",
        "  4. 04_GROUP_C.zip",
        "     - Content: Group_C_V2Clean_BasicAug (seeds 42, 43, 44 best.pt).",
        "     - V2 clean data with BasicAug.",
        "",
        "  5. 05_GROUP_D.zip",
        "     - Content: Group_D_V2Clean_PhotoAug (seeds 42, 43, 44 best.pt).",
        "     - V2 clean data with PhotoAug (Target benchmark configuration).",
        "",
        "Step 4: Upload Advanced Architectural & Preprocessing Experiments",
        "--------------------------------------------------------------------------------",
        "  6. 06_RANDOM_PADDING.zip",
        "     - Content: Exp_RandomPadding_GroupD (seeds 42, 43, 44 best.pt).",
        "     - Random padding on Group D configuration.",
        "",
        "  7. 07_VARIABLE_WIDTH.zip",
        "     - Content: Exp_VariableWidth_GroupD (seeds 42, 43, 44 best.pt).",
        "     - Variable width dynamic batching on Group D configuration (Tesla T4 official).",
        "",
        "================================================================================",
        "Standalone Verification Documents in Directory:",
        "--------------------------------------------------------------------------------",
        "  - SHA256_MANIFEST.csv : Complete cryptographic hash registry for all files and",
        "                          source checkpoints (including last.pt).",
        "  - TRANSFER_AUDIT.md   : Complete scientific reproducibility and package audit.",
        "================================================================================",
    ]
    with open(upload_order_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written {upload_order_txt}.")

if __name__ == '__main__':
    main()
