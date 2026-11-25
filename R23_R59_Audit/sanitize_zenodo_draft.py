"""Sanitize personal/local Windows paths and finalize configurable relative paths
ONLY inside AzOCR_Zenodo_v1_DRAFT/ without touching any original files.
"""

import re
from pathlib import Path

DRAFT_DIR = Path(r"D:\laragon\www\az_ocr_bm606\AzOCR_Zenodo_v1_DRAFT")


def sanitize_text_paths(txt: str) -> str:
    # Replace specific long paths first
    replacements = [
        (r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\runs", "<TRAINING_RUNS_ROOT>/runs"),
        (r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\code", "<TRAINING_RUNS_ROOT>/code"),
        (r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results", "<TRAINING_RUNS_ROOT>"),
        (r"C:\Users\serxan.m\Downloads\Image to PDF 20261124 12.02.30 (1).pdf", "<PRIVATE_REAL30_SOURCE_PDF>"),
        (r"C:/Users/serxan.m/Downloads/Image to PDF 20261124 12.02.30 (1).pdf", "<PRIVATE_REAL30_SOURCE_PDF>"),
        ("file:///D:/laragon/www/az_ocr_bm606/Real30_GPU18_Final_Evaluation/", "./"),
        ("file:///D:/laragon/www/az_ocr_bm606/AzOCR_Final_Scientific_Audit/", "./"),
        ("file:///D:/laragon/www/az_ocr_bm606/AzOCR_Zenodo_v1_DRAFT/", "./"),
        ("file:///D:/laragon/www/az_ocr_bm606/", "./"),
        (r"D:\laragon\www\az_ocr_bm606\\", "./"),
        (r"D:\laragon\www\az_ocr_bm606", "<PROJECT_ROOT>"),
        ("D:/laragon/www/az_ocr_bm606/", "./"),
        ("D:/laragon/www/az_ocr_bm606", "<PROJECT_ROOT>"),
    ]
    for old, new in replacements:
        txt = txt.replace(old, new)
    # Catch-all for any remaining C:\Users\<user>
    txt = re.sub(r"C:[\\/]Users[\\/][^\\/,\s\"'`]+", "<USER_HOME>", txt, flags=re.IGNORECASE)
    return txt


def patch_eval_scripts():
    eval_dir = DRAFT_DIR / "code" / "evaluation"

    # 1. build_and_eval_r59.py
    p1 = eval_dir / "build_and_eval_r59.py"
    t1 = p1.read_text(encoding="utf-8")
    t1 = t1.replace(
        'PROJ_ROOT = Path(r"D:\\laragon\\www\\az_ocr_bm606")',
        'PROJ_ROOT = Path(os.environ.get("AZOCR_PROJ_ROOT", Path(__file__).resolve().parents[2]))'
    )
    t1 = t1.replace(
        'RUNS_ROOT = Path(r"C:\\Users\\serxan.m\\Desktop\\AzOCR_Trained_Runs_Results\\AzOCR_Trained_Runs_Results\\runs")',
        'RUNS_ROOT = Path(os.environ.get("AZOCR_RUNS_ROOT", PROJ_ROOT / "runs"))'
    )
    t1 = t1.replace(
        'pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"',
        'pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")'
    )
    p1.write_text(t1, encoding="utf-8")

    # 2. verify_and_fix_r59.py
    p2 = eval_dir / "verify_and_fix_r59.py"
    t2 = p2.read_text(encoding="utf-8")
    t2 = "import os\n" + t2
    t2 = t2.replace(
        'PROJ_ROOT = Path(r"D:\\laragon\\www\\az_ocr_bm606")',
        'PROJ_ROOT = Path(os.environ.get("AZOCR_PROJ_ROOT", Path(__file__).resolve().parents[2]))'
    )
    p2.write_text(t2, encoding="utf-8")

    # 3. evaluate_real30_gpu18.py
    p3 = eval_dir / "evaluate_real30_gpu18.py"
    t3 = p3.read_text(encoding="utf-8")
    t3 = "import os\n" + t3
    t3 = t3.replace(
        'PROJ_ROOT = Path(r"D:\\laragon\\www\\az_ocr_bm606")',
        'PROJ_ROOT = Path(os.environ.get("AZOCR_PROJ_ROOT", Path(__file__).resolve().parents[2]))'
    )
    t3 = t3.replace(
        'PDF_PATH = Path(r"C:\\Users\\serxan.m\\Downloads\\Image to PDF 20261124 12.02.30 (1).pdf")',
        'PDF_PATH = Path(os.environ.get("AZOCR_REAL30_PDF", PROJ_ROOT / "data" / "real30_evaluation" / "Real30_Source.pdf"))'
    )
    t3 = t3.replace(
        'RUNS_ROOT = Path(r"C:\\Users\\serxan.m\\Desktop\\AzOCR_Trained_Runs_Results\\AzOCR_Trained_Runs_Results\\runs")',
        'RUNS_ROOT = Path(os.environ.get("AZOCR_RUNS_ROOT", PROJ_ROOT / "runs"))'
    )
    p3.write_text(t3, encoding="utf-8")

    # 4. run_final_fairness_audit.py
    p4 = eval_dir / "run_final_fairness_audit.py"
    t4 = p4.read_text(encoding="utf-8")
    t4 = "import os\n" + t4
    t4 = t4.replace(
        'PROJ_ROOT = Path(r"D:\\laragon\\www\\az_ocr_bm606")',
        'PROJ_ROOT = Path(os.environ.get("AZOCR_PROJ_ROOT", Path(__file__).resolve().parents[2]))'
    )
    t4 = t4.replace(
        'Path(r"C:\\Users\\serxan.m\\Desktop\\AzOCR_Trained_Runs_Results\\AzOCR_Trained_Runs_Results\\code\\train_gpu.py")',
        'Path(os.environ.get("AZOCR_RUNS_CODE", PROJ_ROOT / "code")) / "train_gpu.py"'
    )
    t4 = t4.replace(
        'Path(r"C:\\Users\\serxan.m\\Desktop\\AzOCR_Trained_Runs_Results\\AzOCR_Trained_Runs_Results\\code\\run_experiments.py")',
        'Path(os.environ.get("AZOCR_RUNS_CODE", PROJ_ROOT / "code")) / "run_experiments.py"'
    )
    p4.write_text(t4, encoding="utf-8")


def patch_license_wording():
    # Ensure unconfirmed licenses are not presented as final in ZENODO_UPLOAD_PLAN.md and ZENODO_PACKAGE_AUDIT.md
    plan_p = DRAFT_DIR / "ZENODO_UPLOAD_PLAN.md"
    ptxt = plan_p.read_text(encoding="utf-8")
    ptxt = ptxt.replace(
        "Code: MIT; R59 Fonts: DejaVu & SIL OFL 1.1; R59 Text: CC0-1.0 *(pending author sign-off)*",
        "Code: *Pending rights-holder confirmation* (proposed MIT); R59 Fonts: DejaVu & SIL OFL 1.1; R59 Text: *Pending rights-holder confirmation* (proposed CC0-1.0)"
    )
    ptxt = ptxt.replace(
        "| ~1.57 GB | Apache-2.0 / MIT |",
        "| ~1.57 GB | *Pending rights-holder confirmation* (proposed Apache-2.0 / MIT) |"
    )
    plan_p.write_text(ptxt, encoding="utf-8")

    audit_p = DRAFT_DIR / "ZENODO_PACKAGE_AUDIT.md"
    atxt = audit_p.read_text(encoding="utf-8")
    atxt = atxt.replace(
        "MIT / Academic Open Source (`model.py`",
        "Pending rights-holder final license selection (proposed MIT) (`model.py`"
    )
    audit_p.write_text(atxt, encoding="utf-8")


def main():
    patch_eval_scripts()
    patch_license_wording()

    # Sanitize all remaining text files in DRAFT_DIR
    for f in sorted(DRAFT_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".py", ".md", ".csv", ".json", ".jsonl", ".yaml", ".txt"}:
            orig = f.read_text(encoding="utf-8", errors="ignore")
            cleaned = sanitize_text_paths(orig)
            if cleaned != orig:
                f.write_text(cleaned, encoding="utf-8")

    # Verify zero remaining hits
    pat = re.compile(r"(serxan|C:[\\\\/]Users|D:[\\\\/]laragon)", re.IGNORECASE)
    leftover = []
    for f in sorted(DRAFT_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".py", ".md", ".csv", ".json", ".jsonl", ".yaml", ".txt"}:
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if pat.search(txt):
                leftover.append(f.relative_to(DRAFT_DIR).as_posix())
    assert not leftover, f"Leftover local paths in: {leftover}"
    print("[+] Sanitized all personal/local Windows paths in AzOCR_Zenodo_v1_DRAFT (0 remaining hits).")


if __name__ == "__main__":
    main()
