"""Assemble AzOCR_Zenodo_v1_DRAFT directory cleanly with strict license/privacy exclusions."""

import hashlib
import shutil
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
RUNS_SOURCE = Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results")
DRAFT_DIR = PROJ_ROOT / "AzOCR_Zenodo_v1_DRAFT"

GROUPS = [
    "Group_A_V1Clean_BasicAug",
    "Group_B_V1Clean_PhotoAug",
    "Group_C_V2Clean_BasicAug",
    "Group_D_V2Clean_PhotoAug",
    "Exp_RandomPadding_GroupD",
    "Exp_VariableWidth_GroupD",
]
SEEDS = [42, 43, 44]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    if DRAFT_DIR.exists():
        shutil.rmtree(DRAFT_DIR)
    DRAFT_DIR.mkdir(parents=True)

    # 1. Code
    code_dir = DRAFT_DIR / "code"
    eval_dir = code_dir / "evaluation"
    code_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    for src in [
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/model.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/common.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/dataset.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/train_gpu.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/run_experiments.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/train_v2.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/code_snapshot/requirements.txt",
    ]:
        shutil.copy2(src, code_dir / src.name)

    for src in [
        PROJ_ROOT / "Real30_GPU18_Final_Evaluation/evaluate_real30_gpu18.py",
        PROJ_ROOT / "AzOCR_Final_Scientific_Audit/run_final_fairness_audit.py",
        PROJ_ROOT / "R23_R59_Audit/build_and_eval_r59.py",
        PROJ_ROOT / "R23_R59_Audit/verify_and_fix_r59.py",
    ]:
        shutil.copy2(src, eval_dir / src.name)

    # 2. Configs & Logs (18 official runs ONLY; NO reproducibility_check)
    cfg_dir = DRAFT_DIR / "configs_and_logs"
    runs_meta_dir = cfg_dir / "runs_meta"
    runs_meta_dir.mkdir(parents=True)

    for src in [
        RUNS_SOURCE / "runs/experiments_summary.json",
        RUNS_SOURCE / "runs/experiments_summary.md",
        RUNS_SOURCE / "results_per_run.csv",
    ]:
        if src.exists():
            shutil.copy2(src, cfg_dir / src.name)

    for gid in GROUPS:
        for s in SEEDS:
            rname = f"{gid}_seed{s}"
            dst_r = runs_meta_dir / rname
            dst_r.mkdir(parents=True)
            for fname in ["config.yaml", "run_summary.json", "train_log.jsonl"]:
                fp = RUNS_SOURCE / "runs" / rname / fname
                if fp.exists():
                    shutil.copy2(fp, dst_r / fname)

    # 3. R59 Benchmark (Open-license fonts + 200 images + corrected results)
    r59_dir = DRAFT_DIR / "r59_benchmark"
    r59_dir.mkdir(parents=True)
    for fname in [
        "R59_CORRECTED_RESULTS.csv",
        "R59_RAW_PREDICTIONS.csv",
        "R59_DATASET_MANIFEST.csv",
        "R59_FINAL_VERIFICATION.md",
    ]:
        shutil.copy2(PROJ_ROOT / "R23_R59_Audit" / fname, r59_dir / fname)

    shutil.copytree(PROJ_ROOT / "R23_R59_Audit/fonts_open", r59_dir / "fonts_open")
    shutil.copytree(PROJ_ROOT / "R23_R59_Audit/licenses", r59_dir / "licenses")
    shutil.copytree(PROJ_ROOT / "R23_R59_Audit/r59_dataset", r59_dir / "r59_dataset")

    # 4. Results Tables (NO Real30 raw photos!)
    res_dir = DRAFT_DIR / "results_tables"
    res_dir.mkdir(parents=True)
    for src, dst_name in [
        (PROJ_ROOT / "Real30_GPU18_Final_Evaluation/per_model_summary.csv", "Real30_per_model_summary.csv"),
        (PROJ_ROOT / "Real30_GPU18_Final_Evaluation/per_group_summary.csv", "Real30_per_group_summary.csv"),
        (PROJ_ROOT / "Real30_GPU18_Final_Evaluation/per_page_metrics_all_18_models.csv", "Real30_per_page_metrics_all_18_models.csv"),
        (PROJ_ROOT / "Real30_GPU18_Final_Evaluation/FINAL_AUDIT.md", "Real30_FINAL_AUDIT.md"),
        (PROJ_ROOT / "AzOCR_Final_Scientific_Audit/BASELINE_COMPARISON.csv", "BASELINE_COMPARISON.csv"),
        (PROJ_ROOT / "AzOCR_Final_Scientific_Audit/STATISTICAL_TESTS.csv", "STATISTICAL_TESTS.csv"),
        (PROJ_ROOT / "AzOCR_Final_Scientific_Audit/CHARACTER_ERROR_ANALYSIS.csv", "CHARACTER_ERROR_ANALYSIS.csv"),
        (PROJ_ROOT / "AzOCR_Final_Scientific_Audit/FAIRNESS_AUDIT.md", "FAIRNESS_AUDIT.md"),
        (PROJ_ROOT / "R23_R59_Audit/R23_ABLATION_PLAN.md", "R23_ABLATION_PLAN.md"),
    ]:
        shutil.copy2(src, res_dir / dst_name)

    # 5. Manifests & Licenses (Training fonts acquisition instructions only; NO proprietary .ttf files)
    man_dir = DRAFT_DIR / "manifests_and_licenses"
    man_dir.mkdir(parents=True)
    shutil.copy2(PROJ_ROOT / "AzOCR_Final_Scientific_Audit/REPRODUCIBILITY_MANIFEST.csv", man_dir / "REPRODUCIBILITY_MANIFEST.csv")
    shutil.copy2(PROJ_ROOT / "Real30_GPU18_Final_Evaluation/checkpoint_sha256_manifest.csv", man_dir / "OFFICIAL_18_CHECKPOINTS_SHA256.csv")

    # Training fonts SHA-256 registry (no binary font files copied)
    tf_lines = [
        "# Training Fonts Acquisition & SHA-256 Registry",
        "",
        "Proprietary Windows/Microsoft fonts used during synthetic training are **NOT redistributed** in this open Zenodo archive.",
        "To reproduce the exact synthetic training datasets on a licensed Windows system, obtain the following fonts from standard Windows installations and verify their SHA-256 hashes:",
        "",
        "| Folder | Font File | SHA-256 |",
        "|---|---|---|",
    ]
    for folder in ["fonts", "fonts_new"]:
        for fp in sorted((PROJ_ROOT / folder).glob("*.ttf")):
            tf_lines.append(f"| `{folder}/` | `{fp.name}` | `{sha256_file(fp)}` |")
    (man_dir / "TRAINING_FONTS_ACQUISITION.md").write_text("\n".join(tf_lines) + "\n", encoding="utf-8")

    # Compute SHA-256 manifest of everything inside AzOCR_Zenodo_v1_DRAFT
    sha_rows = []
    for f in sorted(DRAFT_DIR.rglob("*")):
        if f.is_file():
            rel = f.relative_to(DRAFT_DIR).as_posix()
            sha_rows.append({
                "relative_path": rel,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_file(f),
            })
    pd.DataFrame(sha_rows).to_csv(man_dir / "DRAFT_SHA256_MANIFEST.csv", index=False)

    total_mb = sum(r["size_bytes"] for r in sha_rows) / (1024 * 1024)
    print(f"[+] Built AzOCR_Zenodo_v1_DRAFT: {len(sha_rows)} files, {total_mb:.2f} MB total.")


if __name__ == "__main__":
    main()
