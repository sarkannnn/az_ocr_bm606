"""Addım 7-8: CER, WER, xərc (vaxt/yaddaş/parametr), qarışıqlıq matrisi; Tesseract baseline.

  python evaluate.py --ckpt runs/word/best.pt --data data/synth_word/test --out results/word_synth
  python evaluate.py --ckpt runs/word/best.pt --data data/real_word/test  --out results/word_real
  python evaluate.py --baseline tesseract --mode word --data data/synth_word/test --out results/tess_word
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from common import CONFUSION_PAIRS, SPECIAL_LOWER, Vocab, cer_wer, load_labels, normalize


def run_model(ckpt_path, data_dir, batch=128):
    import torch
    from torch.utils.data import DataLoader
    from model import CRNN
    from train import OCRDataset, collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    cfg, vocab = ck["cfg"], Vocab(ck["chars"])
    model = CRNN(1 if cfg["gray"] else 3, vocab.size).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dl = DataLoader(OCRDataset(data_dir, cfg, vocab), batch_size=batch, collate_fn=collate, num_workers=2)
    refs, hyps, infer = [], [], 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for x, _, _, labels in dl:
            x = x.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer += time.perf_counter() - t
            for p, r in zip(out.argmax(2).T.cpu().numpy(), labels):
                hyps.append(vocab.decode_greedy(p))
                refs.append(r)
    extra = dict(ms_per_image=1000 * infer / max(len(refs), 1),
                 params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                 gpu_mem_mb=(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
                 train_seconds=ck.get("train_seconds"), epoch=ck.get("epoch"), cfg=cfg)
    return refs, hyps, extra


def run_tesseract(data_dir, mode):
    import pytesseract
    from PIL import Image
    if "aze" not in pytesseract.get_languages():
        raise SystemExit("Tesseract 'aze' dil paketi yoxdur (Ubuntu: sudo apt install tesseract-ocr-aze)")
    psm = 8 if mode == "word" else 7
    refs, hyps, t0 = [], [], time.perf_counter()
    for path, label in load_labels(data_dir):
        txt = pytesseract.image_to_string(Image.open(path).convert("RGB"), lang="aze", config=f"--psm {psm}")
        hyps.append(normalize(" ".join(txt.split())))
        refs.append(label)
    return refs, hyps, dict(ms_per_image=1000 * (time.perf_counter() - t0) / max(len(refs), 1))


def report(refs, hyps, out, extra):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cer, wer, ops = cer_wer(refs, hyps)
    with open(out / "predictions.tsv", "w", encoding="utf-8") as f:
        f.write("ref\thyp\tdüz\n")
        for r, h in zip(refs, hyps):
            f.write(f"{r}\t{h}\t{int(r == h)}\n")

    subs = Counter((r, h) for t, r, h in ops if t == "sub")
    dels = Counter(r for t, r, h in ops if t == "del")
    ref_count = Counter(r for t, r, h in ops if t in ("ok", "sub", "del"))
    with open(out / "top_substitutions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ref", "hyp", "say"])
        for (r, h), c in subs.most_common(40):
            w.writerow([r, h, c])

    rows = []
    print(f"\nCER {cer*100:.2f}% | WER {wer*100:.2f}% | {len(refs)} nümunə")
    print("\nXüsusi hərf qarışıqlıqları (ref -> hyp):")
    print(f"{'cüt':8s}{'xüsusi->əsas':>14s}{'əsas->xüsusi':>14s}{'xüsusi səhv %':>16s}")
    for sp, base in CONFUSION_PAIRS:
        a, b = subs[(sp, base)], subs[(base, sp)]
        err_total = a + sum(c for (r, h), c in subs.items() if r == sp and h != base) + dels[sp]
        rate = 100 * err_total / max(ref_count[sp], 1)
        rows.append([sp, base, a, b, ref_count[sp], f"{rate:.2f}"])
        print(f"{sp}/{base:<6s}{a:>14d}{b:>14d}{rate:>15.2f}%")
    with open(out / "confusion_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["xüsusi", "əsas", "xüsusi->əsas", "əsas->xüsusi", "xüsusi_say", "xüsusi_səhv_%"])
        w.writerows(rows)

    res = dict(cer=cer, wer=wer, n=len(refs), **extra)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print("Nəticə:", out / "metrics.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--baseline", choices=["tesseract"])
    ap.add_argument("--mode", choices=["word", "line"], default="word")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.baseline == "tesseract":
        refs, hyps, extra = run_tesseract(a.data, a.mode)
    else:
        refs, hyps, extra = run_model(a.ckpt, a.data)
    report(refs, hyps, a.out, extra)


if __name__ == "__main__":
    main()
