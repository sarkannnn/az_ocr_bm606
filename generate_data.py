"""Addım 3: TRDG ilə sintetik söz/sətir şəkilləri.

Fontlar qovluqdan götürülür, Azərbaycan hərflərini (əıöğüşç) dəstəkləməyənlər atılır.
Fontların ~20%-i YALNIZ test üçün ayrılır (font ayrımı).

  python generate_data.py --mode word --corpus data/corpus_word --fonts fonts --out data/synth_word \
      --n-train 100000 --n-val 10000 --n-test 10000
"""
import argparse
import random
from pathlib import Path

from tqdm import tqdm

from common import SPECIAL_ALL


def font_ok(path):
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(path, fontNumber=0 if path.lower().endswith(".ttc") else -1, lazy=True)
        cmap = f.getBestCmap()
        return all(ord(c) in cmap for c in SPECIAL_ALL)
    except Exception:
        return False


def collect_fonts(folder, seed):
    paths = sorted(str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in {".ttf", ".otf", ".ttc"})
    good = [p for p in paths if font_ok(p)]
    print(f"Fontlar: {len(paths)} tapıldı, {len(good)} uyğun (ə ı ö ğ ü ş ç var)")
    if len(good) < 2:
        raise SystemExit("Ən azı 2 uyğun font lazımdır (məs. DejaVu, Noto Sans/Serif, Roboto, Open Sans).")
    random.Random(seed).shuffle(good)
    n_test = max(1, round(0.2 * len(good)))
    return good[n_test:], good[:n_test]  # train_fonts, test_fonts


def make_split(strings, fonts, out_dir, size, seed, bgs):
    from trdg.generators import GeneratorFromStrings
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    groups = {bg: [s for i, s in enumerate(strings) if i % len(bgs) == k] for k, bg in enumerate(bgs)}
    idx = 0
    with open(out_dir / "labels.tsv", "w", encoding="utf-8") as lab, tqdm(total=len(strings), desc=out_dir.name) as bar:
        for bg, strs in groups.items():
            if not strs:
                continue
            gen = GeneratorFromStrings(
                strs, count=len(strs), fonts=fonts, size=size,
                skewing_angle=3, random_skew=True, blur=1, random_blur=True,
                background_type=bg, text_color="#000000,#555555",
                margins=(3, 3, 3, 3), fit=True)
            for img, text in gen:
                if img is None:
                    continue
                fn = f"images/{idx:07d}.png"
                img.save(out_dir / fn)
                lab.write(f"{fn}\t{text}\n")
                idx += 1
                bar.update(1)


def sample(path, n, rng):
    items = [l for l in Path(path).read_text(encoding="utf-8").split("\n") if l]
    return [rng.choice(items) for _ in range(n)] if n > len(items) else rng.sample(items, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["word", "line"], required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-val", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--bg", type=int, nargs="+", default=[0, 1, 2],
                    help="TRDG fon: 0 gauss küyü, 1 ağ, 2 quasicrystal (yavaşdır)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    size = 64 if args.mode == "word" else 48
    tr_fonts, te_fonts = collect_fonts(args.fonts, args.seed)
    print("train/val fontları:", len(tr_fonts), "| test fontları:", len(te_fonts))
    corpus = Path(args.corpus)
    for split, n, fonts in [("train", args.n_train, tr_fonts), ("val", args.n_val, tr_fonts), ("test", args.n_test, te_fonts)]:
        strings = sample(corpus / f"{split}.txt", n, rng)
        make_split(strings, fonts, Path(args.out) / split, size, args.seed, args.bg)


if __name__ == "__main__":
    main()
