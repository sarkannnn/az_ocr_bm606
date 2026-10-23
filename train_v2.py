"""Enhanced CRNN + CTC training script (v2) for Azerbaijani OCR.

Includes realistic phone-photo augmentations (illumination gradients, shadows,
Gaussian/sensor noise, motion blur, contrast stretching, JPEG compression,
downscaling, rotation, perspective), --resume support, last.pt saving,
and --save-aug-preview contact sheet generation.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from common import Vocab, cer_wer, load_labels, make_cfg, preprocess
from model import CRNN

AUG_CONFIG = {
    "rotation_prob": 0.5,
    "rotation_range": (-5.0, 5.0),
    "perspective_prob": 0.3,
    "gaussian_blur_prob": 0.6,
    "gaussian_blur_sigma": (0.5, 2.0),
    "motion_blur_prob": 0.15,
    "motion_blur_kernel": (3, 7),
    "sensor_noise_prob": 0.6,
    "sensor_noise_sigma": (3.0, 12.0),
    "salt_pepper_prob": 0.2,
    "downscale_prob": 0.4,
    "downscale_factor": (0.5, 0.9),
    "jpeg_prob": 0.4,
    "jpeg_quality": (40, 90),
    "brightness_contrast_prob": 0.6,
    "alpha_range": (0.6, 1.1),
    "beta_range": (-40.0, 20.0),
    "background_tint_prob": 0.5,
    "gradient_range": (0.7, 1.0),
    "contrast_stretch_prob": 0.5,
}


def augment(img, rng):
    """Photo-realistic data augmentation simulating phone cameras and diverse lighting."""
    h, w = img.shape[:2]

    # 1. Rotation +-5 deg (p: 0.5)
    if rng.random() < AUG_CONFIG["rotation_prob"]:
        deg = rng.uniform(*AUG_CONFIG["rotation_range"])
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Perspective jitter as before (p: 0.3)
    if rng.random() < AUG_CONFIG["perspective_prob"]:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jit = rng.uniform(-1, 1, (4, 2)).astype(np.float32) * np.array([w * 0.02, h * 0.08], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Background tint & smooth illumination gradient + soft dark band (p: 0.5)
    if rng.random() < AUG_CONFIG["background_tint_prob"]:
        img_f = img.astype(np.float32)
        gx = np.linspace(-1, 1, w, dtype=np.float32)
        gy = np.linspace(-1, 1, h, dtype=np.float32)
        X, Y = np.meshgrid(gx, gy)
        low_val = rng.uniform(*AUG_CONFIG["gradient_range"])
        angle = rng.uniform(0, 2 * np.pi)
        proj = np.cos(angle) * X + np.sin(angle) * Y
        p_min, p_max = proj.min(), proj.max()
        proj_norm = (proj - p_min) / max(p_max - p_min, 1e-5)
        grad = 1.0 - (1.0 - low_val) * proj_norm
        # Optional soft dark shadow band
        if rng.random() < 0.5:
            d0 = rng.uniform(-0.5, 0.5)
            dist = proj - d0
            bw = rng.uniform(0.2, 0.5)
            band = 1.0 - rng.uniform(0.12, 0.28) * np.exp(-(dist / bw) ** 2)
            grad *= band
        img = np.clip(img_f * grad[:, :, None], 0, 255).astype(np.uint8)

    # 4. Brightness / Contrast alpha 0.6-1.1, beta -40..+20 (p: 0.6)
    if rng.random() < AUG_CONFIG["brightness_contrast_prob"]:
        alpha = rng.uniform(*AUG_CONFIG["alpha_range"])
        beta = rng.uniform(*AUG_CONFIG["beta_range"])
        img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 5. Gaussian blur sigma 0.5-2.0 (p: 0.6)
    if rng.random() < AUG_CONFIG["gaussian_blur_prob"]:
        sigma = rng.uniform(*AUG_CONFIG["gaussian_blur_sigma"])
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    # 6. Motion blur kernel 3-7 (p: 0.15)
    if rng.random() < AUG_CONFIG["motion_blur_prob"]:
        ksize = int(rng.integers(AUG_CONFIG["motion_blur_kernel"][0], AUG_CONFIG["motion_blur_kernel"][1] + 1))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        m_angle = rng.uniform(0, np.pi)
        cx, cy = ksize // 2, ksize // 2
        for step in range(-ksize // 2, ksize // 2 + 1):
            kx = int(round(cx + step * np.cos(m_angle)))
            ky = int(round(cy + step * np.sin(m_angle)))
            if 0 <= kx < ksize and 0 <= ky < ksize:
                kernel[ky, kx] = 1.0
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

    # 7. Gaussian sensor noise sigma 3-12 (p: 0.6)
    if rng.random() < AUG_CONFIG["sensor_noise_prob"]:
        n_sigma = rng.uniform(*AUG_CONFIG["sensor_noise_sigma"])
        noise = rng.normal(0, n_sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. Salt-and-pepper noise as before (p: 0.2)
    if rng.random() < AUG_CONFIG["salt_pepper_prob"]:
        img = img.copy()
        m = rng.random(img.shape[:2])
        img[m < 0.005] = 0
        img[m > 0.995] = 255

    # 9. Downscale then upscale factor 0.5-0.9 (p: 0.4)
    if rng.random() < AUG_CONFIG["downscale_prob"]:
        scale = rng.uniform(*AUG_CONFIG["downscale_factor"])
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 10. JPEG compression quality 40-90 (p: 0.4)
    if rng.random() < AUG_CONFIG["jpeg_prob"]:
        q = int(rng.uniform(*AUG_CONFIG["jpeg_quality"]))
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 11. Contrast stretch min-max to 0-255 (p: 0.5)
    if rng.random() < AUG_CONFIG["contrast_stretch_prob"]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)

    return img


class OCRDataset(Dataset):
    def __init__(self, data_dir, cfg, vocab, train=False):
        self.items, self.cfg, self.vocab, self.train = load_labels(data_dir), cfg, vocab, train
        self.rng = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng()
        path, label = self.items[i]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if self.train:
            img = augment(img, self.rng)
        x = torch.from_numpy(preprocess(img, self.cfg))
        return x, torch.tensor(self.vocab.encode(label), dtype=torch.long), label


def collate(batch):
    xs, ys, labels = zip(*batch)
    return torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys]), list(labels)


def run_epoch(model, loader, crit, device, vocab, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot, n, refs, hyps = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y, yl, labels in loader:
            out = model(x.to(device))
            T, B, _ = out.shape
            loss = crit(out.float(), y, torch.full((B,), T, dtype=torch.long), yl)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            tot += loss.item() * B
            n += B
            if not train:
                for p, ref in zip(out.argmax(2).T.cpu().numpy(), labels):
                    hyps.append(vocab.decode_greedy(p))
                    refs.append(ref)
    cer = cer_wer(refs, hyps)[0] if refs else float("nan")
    return tot / max(n, 1), cer


def generate_aug_preview(data_dir, out_path, n_samples=24, seed=42):
    """Generate a contact sheet of 24 augmented samples with labels."""
    items = load_labels(data_dir)
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(items), size=min(n_samples, len(items)), replace=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = math.ceil(len(chosen_indices) / cols)
    card_w = 340
    card_h = 95
    canvas_w = cols * card_w + 30
    canvas_h = rows * card_h + 60

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    hdr_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    lbl_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)

    draw.text((20, 15), "V2 Augmentation Preview (24 Samples with Random Distortion)", fill=(0, 40, 120), font=hdr_font)

    for idx, item_idx in enumerate(chosen_indices):
        img_path, label = items[item_idx]
        raw_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        aug_bgr = augment(raw_bgr, rng)
        aug_rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)
        pil_aug = Image.fromarray(aug_rgb)

        # Scale to fit box
        max_h = 45
        if pil_aug.height > max_h:
            s = max_h / pil_aug.height
            pil_aug = pil_aug.resize((max(10, int(pil_aug.width * s)), max_h), Image.Resampling.LANCZOS)
        if pil_aug.width > (card_w - 20):
            s = (card_w - 20) / pil_aug.width
            pil_aug = pil_aug.resize((card_w - 20, max(8, int(pil_aug.height * s))), Image.Resampling.LANCZOS)

        r = idx // cols
        c = idx % cols
        x_cell = 20 + c * card_w
        y_cell = 50 + r * card_h

        draw.text((x_cell, y_cell), f"Label: {label}", fill=(40, 40, 40), font=lbl_font)
        canvas.paste(pil_aug, (x_cell, y_cell + 22))

    canvas.save(out_path)
    print(f"Saved augmentation preview contact sheet to {out_path}")


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
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training.")
    ap.add_argument("--save-aug-preview", type=str, default=None, help="Save augmentation preview contact sheet and exit.")
    a = ap.parse_args()

    if a.save_aug_preview:
        generate_aug_preview(a.train_dir, a.save_aug_preview, n_samples=24, seed=a.seed)
        return

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = make_cfg(mode=a.mode, gray=(a.color == "gray"), filter=a.filter, thresh=a.thresh, interp=a.interp)
    vocab = Vocab()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = OCRDataset(a.train_dir, cfg, vocab, train=not a.no_aug)
    va = OCRDataset(a.val_dir, cfg, vocab, train=False)
    kw = dict(collate_fn=collate, num_workers=a.workers, pin_memory=device.type == "cuda")
    tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(va, batch_size=a.batch_size, shuffle=False, **kw)

    model = CRNN(1 if cfg["gray"] else 3, vocab.size, dropout=a.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CTCLoss(blank=0, zero_infinity=True)

    start_epoch = 1
    best, bad = float("inf"), 0
    t0 = time.time()

    if a.resume:
        print(f"Resuming from checkpoint: {a.resume}")
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_epoch = ck.get("epoch", 0) + 1
        best = ck.get("val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, previous best val loss: {best:.4f}")

    print(f"cfg={cfg} | cihaz={device} | parametr={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    hist = []
    for ep in range(start_epoch, a.epochs + 1):
        trl, _ = run_epoch(model, tl, crit, device, vocab, opt)
        vl_loss, vcer = run_epoch(model, vl, crit, device, vocab)
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=vl_loss, val_cer=vcer))
        print(f"epox {ep:3d} | train {trl:.4f} | val {vl_loss:.4f} | val CER {vcer*100:.2f}%")

        # Always save last.pt every epoch
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), cfg=cfg, chars=vocab.chars, epoch=ep,
                        val_loss=vl_loss, val_cer=vcer, train_seconds=time.time() - t0), out / "last.pt")

        if vl_loss < best:
            best, bad = vl_loss, 0
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), cfg=cfg, chars=vocab.chars, epoch=ep,
                            val_loss=vl_loss, val_cer=vcer, train_seconds=time.time() - t0), out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("Erkən dayandırma (doğrulama itkisi yaxşılaşmadı)")
                break

    train_sec = time.time() - t0
    hist_file = out / "history.csv"
    write_header = not hist_file.exists() or start_epoch == 1
    with open(hist_file, "a" if not write_header else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_cer"])
        if write_header:
            w.writeheader()
        w.writerows(hist)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        all_hist = []
        with open(hist_file, newline="") as f:
            for row in csv.DictReader(f):
                all_hist.append({k: float(v) for k, v in row.items()})
        plt.figure(figsize=(6, 4))
        plt.plot([h["epoch"] for h in all_hist], [h["train_loss"] for h in all_hist], label="train")
        plt.plot([h["epoch"] for h in all_hist], [h["val_loss"] for h in all_hist], label="val")
        plt.xlabel("epox"), plt.ylabel("CTC loss"), plt.legend(), plt.tight_layout()
        plt.savefig(out / "loss_curve.png", dpi=150)
    except Exception:
        pass

    print(f"Təlim vaxtı: {train_sec/60:.1f} dəq | ən yaxşı val loss {best:.4f} | {out/'best.pt'}")


if __name__ == "__main__":
    main()
