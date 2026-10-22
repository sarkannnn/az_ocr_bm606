"""Addım 5: ön emal ablation-ı. Hər konfiqurasiya üçün qısa təlim + test CER -> results/ablation.csv

  python run_ablation.py --mode word --train-dir data/synth_word/train --val-dir data/synth_word/val \
      --test-dir data/synth_word/test --epochs 10
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    ("gri (baza)", []),
    ("RGB", ["--color", "rgb"]),
    ("gri + gaussian", ["--filter", "gaussian"]),
    ("gri + median", ["--filter", "median"]),
    ("gri + Otsu", ["--thresh", "otsu"]),
    ("gri + adaptiv", ["--thresh", "adaptive"]),
    ("gri + bicubic", ["--interp", "bicubic"]),
    ("gri + median + Otsu", ["--filter", "median", "--thresh", "otsu"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="word")
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="runs/ablation")
    a = ap.parse_args()
    rows = []
    for i, (name, flags) in enumerate(CONFIGS):
        run = Path(a.out) / f"cfg{i}"
        subprocess.run([sys.executable, "train.py", "--mode", a.mode, "--train-dir", a.train_dir,
                        "--val-dir", a.val_dir, "--out", str(run), "--epochs", str(a.epochs)] + flags, check=True)
        res_dir = Path("results") / f"ablation_cfg{i}"
        subprocess.run([sys.executable, "evaluate.py", "--ckpt", str(run / "best.pt"),
                        "--data", a.test_dir, "--out", str(res_dir)], check=True)
        m = json.loads((res_dir / "metrics.json").read_text())
        rows.append([name, f"{m['cer']*100:.2f}", f"{m['wer']*100:.2f}"])
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["konfiqurasiya", "CER %", "WER %"])
        w.writerows(rows)
    print("\n| Konfiqurasiya | CER % | WER % |\n|---|---|---|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


if __name__ == "__main__":
    main()
