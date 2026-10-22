"""Addım 2: mətn korpusu -> təmiz söz/sətir siyahıları və 80/10/10 bölgü (şəkillərdən ƏVVƏL, sızma olmasın).

Nümunələr:
  python prepare_corpus.py --hf-wiki 20000 --mode word --out data/corpus_word
  python prepare_corpus.py --input az_text.txt --mode line --out data/corpus_line
"""
import argparse
import random
import re
from pathlib import Path

from common import AZ_LOWER, AZ_UPPER, CHARS, SPECIAL_ALL, az_lower, normalize

WORD_RE = re.compile("[" + AZ_LOWER + AZ_UPPER + "]+")
TRANS = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'",
                       "–": "-", "—": "-", "\u00a0": " "})
ALLOWED = set(CHARS) | {" "}


def clean(text):
    text = normalize(text).translate(TRANS)
    text = "".join(c if c in ALLOWED else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def load_texts(args):
    texts = []
    for p in args.input or []:
        texts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
    if args.hf_wiki:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.az", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= args.hf_wiki:
                break
            texts.append(row["text"])
    if not texts:
        raise SystemExit("--input və ya --hf-wiki verin")
    return texts


def split_list(items, rng, ratios=(0.8, 0.1, 0.1)):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def chunk_words(words, lo, hi):
    """Sözləri hi simvola qədər yığıb sətir kimi qaytarır."""
    out, cur = [], []
    for w in words:
        if cur and len(" ".join(cur + [w])) > hi:
            line = " ".join(cur)
            if len(line) >= lo:
                out.append(line)
            cur = []
        cur.append(w)
    line = " ".join(cur)
    if len(line) >= lo:
        out.append(line)
    return out


def balance_special(items, max_items, ratio, rng):
    """Xüsusi hərfli (əıöğüşç) nümunələrin payını ən azı `ratio` saxlayır."""
    if not max_items or len(items) <= max_items:
        return items
    sp = [x for x in items if any(c in SPECIAL_ALL for c in x)]
    ot = [x for x in items if not any(c in SPECIAL_ALL for c in x)]
    rng.shuffle(sp), rng.shuffle(ot)
    k = min(len(sp), int(max_items * ratio))
    res = sp[:k] + ot[: max_items - k]
    if len(res) < max_items:
        res += sp[k: k + (max_items - len(res))]
    rng.shuffle(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="*")
    ap.add_argument("--hf-wiki", type=int, default=0, help="HF Vikipediya (az) məqalə sayı")
    ap.add_argument("--mode", choices=["word", "line"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--max-items", type=int, default=0, help="hər bölgü üçün yox, cəmi limit")
    ap.add_argument("--special-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    lo = args.min_len or (2 if args.mode == "word" else 12)
    hi = args.max_len or (20 if args.mode == "word" else 35)
    texts = [clean(t) for t in load_texts(args)]

    if args.mode == "word":
        seen, words = set(), []
        for t in texts:
            for w in WORD_RE.findall(t):
                k = az_lower(w)
                if lo <= len(w) <= hi and k not in seen:
                    seen.add(k)
                    words.append(w)
        words = balance_special(words, args.max_items, args.special_ratio, rng)
        splits = split_list(words, rng)
    else:
        # bölgü cümlə səviyyəsində: eyni cümlənin parçaları fərqli bölgülərə düşməsin
        sents = []
        for t in texts:
            sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        sents = list(dict.fromkeys(sents))
        sp = split_list(sents, rng)
        splits = {}
        for k, ss in sp.items():
            lines = []
            for s in ss:
                lines += chunk_words(s.split(" "), lo, hi)
            lines = list(dict.fromkeys(lines))
            if k == "train" and args.max_items:
                lines = balance_special(lines, int(args.max_items * 0.8), args.special_ratio, rng)
            splits[k] = lines

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, items in splits.items():
        (out / f"{k}.txt").write_text("\n".join(items), encoding="utf-8")
        n_sp = sum(any(c in SPECIAL_ALL for c in x) for x in items)
        print(f"{k:5s}: {len(items):7d} nümunə, xüsusi hərfli: {n_sp} ({100*n_sp/max(len(items),1):.1f}%)")
    print("Yazıldı:", out)


if __name__ == "__main__":
    main()
