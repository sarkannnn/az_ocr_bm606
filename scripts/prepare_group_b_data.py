"""Grup B üçün Sintetik Sətir Veri Seti Hazırlama Betiyi.

Bu betik V1-in 8 standart təlim şrifti ilə (Arial, Calibri, Candara, Corbel, Courier New,
Segoe UI, Times New Roman, Verdana) 10,400 təlim və 1,000 doğrulama sətrini generasiya edir.
Tahoma və Consolas şriftləri sızmanın qarşısını almaq üçün təlimə QƏTİYYƏN DAXİL EDİLMİR.

Çıxış: data/synth_line_v1/train və data/synth_line_v1/val
"""
import os
import sys
import random
from pathlib import Path
from tqdm import tqdm
from trdg.generators import GeneratorFromStrings

BASE_DIR = Path(__file__).resolve().parent.parent

# V1-in dəqiq təlim şriftləri (Table 1-ə uyğun: 8 şrift)
V1_TRAIN_FONTS = [
    str(BASE_DIR / "fonts" / "arial.ttf"),
    str(BASE_DIR / "fonts" / "calibri.ttf"),
    str(BASE_DIR / "fonts" / "candara.ttf"),
    str(BASE_DIR / "fonts" / "corbel.ttf"),
    str(BASE_DIR / "fonts" / "cour.ttf"),
    str(BASE_DIR / "fonts" / "segoeui.ttf"),
    str(BASE_DIR / "fonts" / "times.ttf"),
    str(BASE_DIR / "fonts" / "verdana.ttf"),
]


def render_split(corpus_file: Path, out_dir: Path, n_samples: int, fonts: list, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_file = out_dir / "labels.tsv"

    lines = [l.strip() for l in corpus_file.read_text(encoding="utf-8").split("\n") if l.strip()]
    rng = random.Random(seed)
    sampled_lines = lines[:n_samples] if len(lines) >= n_samples else [rng.choice(lines) for _ in range(n_samples)]

    print(f"Generasiya: {out_dir.name} -> {len(sampled_lines)} sətir...")
    gen = GeneratorFromStrings(
        sampled_lines,
        count=len(sampled_lines),
        fonts=fonts,
        size=48,
        skewing_angle=3,
        random_skew=True,
        blur=1,
        random_blur=True,
        background_type=1,  # Ağ fon (fotometrik augmentasiya train_v2 zamanı tətbiq olunur)
        text_color="#000000,#333333",
        margins=(3, 3, 3, 3),
        fit=True
    )

    with open(labels_file, "w", encoding="utf-8") as lf:
        for idx, (img, text) in enumerate(tqdm(gen, total=len(sampled_lines), desc=out_dir.name)):
            if img is None:
                continue
            fn = f"images/{idx:07d}.png"
            img.save(out_dir / fn)
            lf.write(f"{fn}\t{text}\n")

    print(f"Tamamlandı: {out_dir} -> {labels_file}")


def main():
    print("=" * 70)
    print("GRUP B VERİ SETİ HAZIRLANMASI (V1 Standart Şriftləri)")
    print("=" * 70)
    for f in V1_TRAIN_FONTS:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Şrift tapılmadı: {f}")
    print(f"İstifadə olunan şrift sayı: {len(V1_TRAIN_FONTS)} (Tahoma və Consolas xaric edildi - sızma yoxdur).")

    corpus_train = BASE_DIR / "data" / "corpus_line" / "train.txt"
    corpus_val = BASE_DIR / "data" / "corpus_line" / "val.txt"
    out_base = BASE_DIR / "data" / "synth_line_v1"

    if (out_base / "train" / "labels.tsv").exists() and (out_base / "val" / "labels.tsv").exists():
        print(f"XƏBƏRDARLIQ: {out_base} artıq mövcuddur. Təkrar istehsal edilməyəcək.")
        return

    # Train: 10,400 sətir (V1 ilə dəqiq eyni həcm)
    render_split(corpus_train, out_base / "train", 10400, V1_TRAIN_FONTS, seed=42)
    # Val: 1,000 sətir
    render_split(corpus_val, out_base / "val", 1000, V1_TRAIN_FONTS, seed=42)
    print("Grup B üçün veri seti tam hazırdır!")


if __name__ == "__main__":
    main()
