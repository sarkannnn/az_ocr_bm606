#!/usr/bin/env python3
"""Azərbaycan dili üçün CRNN+CTC İnferens Skripti (Predict).

Tək şəkil və ya şəkillər qovluğu üzərində CRNN modeli ilə mətn tanıma aparır.
İstəyə görə Azərbaycan dili lüğət korreksiyasını (postprocess.py) tətbiq edir.

İstifadə nümunələri:
    python predict.py --image data/real_docs/crops/line_00.png
    python predict.py --image test_line.png --ckpt runs/line_v2/best.pt --postprocess
    python predict.py --image test_word.png --ckpt runs/word_v2_20261027_180300/best.pt
    python predict.py --dir data/synth_line_v2/test/images --limit 10
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

from common import infer_line_smart, load_model_checkpoint, segment_lines_opencv
from postprocess import AzerbaijaniPostProcessor

# Ensure UTF-8 output in Windows terminal
sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Azərbaycan dili OCR İnferens Skripti (CRNN + CTC)")
    parser.add_argument("--image", "-i", type=str, help="Oxunacaq şəkil faylının yolu")
    parser.add_argument("--dir", "-d", type=str, help="Oxunacaq şəkillər olan qovluğun yolu")
    parser.add_argument("--ckpt", "-c", type=str, default="runs/line_v2/best.pt",
                        help="Model çəkpointi (default: runs/line_v2/best.pt)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Lüğət əsaslı utancaq korreksiyanı (postprocess) söndür")
    parser.add_argument("--limit", type=int, default=20,
                        help="Qovluq rejimində maksimum emal olunacaq şəkil sayı")
    parser.add_argument("--out", "-o", type=str, help="Nəticələrin yazılacağı mətn faylı")

    args = parser.parse_args()

    if not args.image and not args.dir:
        parser.print_help()
        sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        # Fallback to runs/line/best.pt if line_v2 doesn't exist
        fallback = Path("runs/line/best.pt")
        if fallback.exists():
            ckpt_path = fallback
        else:
            print(f"XƏTA: Model faylı tapılmadı: {args.ckpt}", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Model yüklənir: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, vocab, device = load_model_checkpoint(ckpt_path, device=device)
    print(f"[*] Cihaz: {device.type.upper()} | Lüğət ölçüsü: {vocab.size} | Giriş ölçüsü: {cfg.get('h', 32)}x{cfg.get('w', 512)}")

    postprocessor = None
    if not args.no_postprocess:
        postprocessor = AzerbaijaniPostProcessor()
        if postprocessor.loaded:
            print("[*] Lüğət korreksiyası aktivləşdirildi (az_lexicon.txt yükləndi)")
        else:
            print("[!] Xəbərdarlıq: az_lexicon.txt tapılmadı, xam OCR çıxışı veriləcək")

    # Image collection
    images_to_process = []
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"XƏTA: Şəkil faylı tapılmadı: {p}", file=sys.stderr)
            sys.exit(1)
        images_to_process.append(p)
    elif args.dir:
        dp = Path(args.dir)
        if not dp.exists():
            print(f"XƏTA: Qovluq tapılmadı: {dp}", file=sys.stderr)
            sys.exit(1)
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".heic", ".heif"}
        all_imgs = [f for f in sorted(dp.iterdir()) if f.suffix.lower() in valid_exts]
        images_to_process.extend(all_imgs[:args.limit])
        print(f"[*] Qovluqda {len(all_imgs)} şəkil tapıldı, {len(images_to_process)} ədədi emal edilir...")

    results = []
    for img_path in images_to_process:
        img = cv2.imread(str(img_path))
        if img is None:
            try:
                import numpy as np
                from PIL import Image
                try:
                    import pillow_heif
                    pillow_heif.register_heif_opener()
                except ImportError:
                    pass
                pil_img = Image.open(img_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
        if img is None:
            print(f"[!] Oxunmadı (skip): {img_path}")
            continue

        line_crops = segment_lines_opencv(img) if cfg.get("width", 512) > 128 else [img]
        raw_lines = []
        final_lines = []
        for lc in line_crops:
            rp = infer_line_smart(model, cfg, vocab, lc, device)
            raw_lines.append(rp)
            if postprocessor and postprocessor.loaded:
                tokens = rp.split()
                corrected_tokens = [postprocessor.correct_word(t, max_edit_distance=1) for t in tokens]
                final_lines.append(postprocessor.clean_punctuation(" ".join(corrected_tokens)))
            else:
                final_lines.append(rp)
        raw_pred = "\n".join(raw_lines)
        final_pred = "\n".join(final_lines)

        results.append((img_path.name, raw_pred, final_pred))

        if len(images_to_process) == 1:
            print("-" * 50)
            print(f"Şəkil:      {img_path.name}")
            print(f"Xam OCR:    {raw_pred}")
            if postprocessor and postprocessor.loaded and raw_pred != final_pred:
                print(f"Düzəldilmiş: {final_pred}")
            print("-" * 50)
        else:
            print(f"[{img_path.name}] -> {final_pred}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for fname, raw, clean in results:
                f.write(f"{fname}\t{clean}\n")
        print(f"[*] Nəticələr saxlanıldı: {out_p}")


if __name__ == "__main__":
    main()

