"""Addım 5-6: CRNN + CTC təlimi (söz: 32x128, sətir: 32x512). Augmentation, dropout, weight decay, early stopping.

  python train.py --mode word --train-dir data/synth_word/train --val-dir data/synth_word/val --out runs/word
  python train.py --mode line --train-dir data/synth_line/train --val-dir data/synth_line/val --out runs/line
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import Vocab, make_cfg
from model import CRNN
from training_utils import OCRDataset, collate, run_epoch


def augment(img, rng):
    """±5° fırlatma, Gaussian blur, duz-istiot küyü, perspektiv əyilməsi."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:
        img = img.copy()
        m = rng.random(img.shape[:2])
        img[m < 0.005] = 0
        img[m > 0.995] = 255
    if rng.random() < 0.3:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jit = rng.uniform(-1, 1, (4, 2)) * np.array([w * 0.02, h * 0.08])
        M = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["word", "line"], required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--color", choices=["gray", "rgb"], default="gray")
    ap.add_argument("--filter", choices=["none", "gaussian", "median"], default="none")
    ap.add_argument("--thresh", choices=["none", "otsu", "adaptive"], default="none")
    ap.add_argument("--interp", choices=["bilinear", "bicubic"], default="bilinear")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_cfg(mode=a.mode, gray=(a.color == "gray"), filter=a.filter, thresh=a.thresh, interp=a.interp)
    vocab = Vocab()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = OCRDataset(a.train_dir, cfg, vocab, train=not a.no_aug, augment_fn=augment)
    va = OCRDataset(a.val_dir, cfg, vocab, train=False)
    kw = dict(collate_fn=collate, num_workers=a.workers, pin_memory=device.type == "cuda")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, **kw)

    model = CRNN(1 if cfg["gray"] else 3, vocab.size, dropout=a.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)  # L2
    crit = nn.CTCLoss(blank=0, zero_infinity=True)
    print(f"cfg={cfg} | cihaz={device} | parametr={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, a.epochs + 1):
        trl, _ = run_epoch(model, tl, crit, device, vocab, opt)
        vl_loss, vcer = run_epoch(model, vl, crit, device, vocab)
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=vl_loss, val_cer=vcer))
        print(f"epox {ep:3d} | train {trl:.4f} | val {vl_loss:.4f} | val CER {vcer*100:.2f}%")
        if vl_loss < best:
            best, bad = vl_loss, 0
            torch.save(dict(model=model.state_dict(), cfg=cfg, chars=vocab.chars, epoch=ep,
                            val_loss=vl_loss, val_cer=vcer, train_seconds=time.time() - t0), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Erkən dayandırma (doğrulama itkisi yaxşılaşmadı)")
                break

    train_sec = time.time() - t0
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_cer"])
        w.writeheader()
        w.writerows(hist)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train")
        plt.plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="val")
        plt.xlabel("epox"), plt.ylabel("CTC loss"), plt.legend(), plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=150)
    except ImportError:
        pass
    print(f"Təlim vaxtı: {train_sec/60:.1f} dəq | ən yaxşı val loss {best:.4f} | {out/'best.pt'}")


if __name__ == "__main__":
    main()
