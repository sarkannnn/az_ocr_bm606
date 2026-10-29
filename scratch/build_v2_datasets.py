import os
import sys
import shutil
import random
import re
from pathlib import Path

sys.path.insert(0, os.getcwd())
from generate_data import make_split

punct_set = set('.,;:!?"()\'`')
digits_re = re.compile(r'\d')

def main():
    rng = random.Random(42)

    print("=== Step 3: Preparing synth_word_v2 and synth_line_v2 ===")

    # 1. Identical copies of val and test splits
    for src_dir, dst_dir in [
        ("data/synth_word", "data/synth_word_v2"),
        ("data/synth_line", "data/synth_line_v2"),
    ]:
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        for split in ["val", "test"]:
            s = Path(src_dir) / split
            d = Path(dst_dir) / split
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
            print(f"Copied {s} -> {d} (identical)")

    # 2. Collect forbidden labels for leak check
    val_test_words = set()
    for p in [
        "data/synth_word/val/labels.tsv",
        "data/synth_word/test/labels.tsv",
        "data/corpus_word/val.txt",
        "data/corpus_word/test.txt",
    ]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                text = parts[1] if len(parts) > 1 else parts[0]
                val_test_words.add(text)

    # Also existing train labels
    existing_word_train_labels = []
    with open("data/synth_word/train/labels.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 1:
                existing_word_train_labels.append(parts[1])

    # 3. Build extra word corpus from train split text only
    # Extract tokens from data/corpus_line/train.txt and data/corpus_word/train.txt
    train_tokens_special = []
    train_tokens_plain = []

    def process_token(tok):
        tok = tok.strip()
        if len(tok) < 2 or tok in val_test_words:
            return
        # Must be valid chars
        is_spec = bool(digits_re.search(tok) or (tok[0] in punct_set or tok[-1] in punct_set))
        if is_spec:
            train_tokens_special.append(tok)
        else:
            train_tokens_plain.append(tok)

    with open("data/corpus_line/train.txt", encoding="utf-8") as f:
        for line in f:
            for tok in line.strip().split():
                process_token(tok)

    with open("data/corpus_word/train.txt", encoding="utf-8") as f:
        for line in f:
            process_token(line.strip())

    train_tokens_special = sorted(list(set(train_tokens_special)))
    train_tokens_plain = sorted(list(set(train_tokens_plain)))
    print(f"Candidate extra words from train split:")
    print(f"  Unique special tokens (digits/punct): {len(train_tokens_special)}")
    print(f"  Unique plain tokens: {len(train_tokens_plain)}")

    # Target: 15,000 extra words. ~20% special (3,000), ~80% plain (12,000)
    n_special = 3000
    n_plain = 12000
    sampled_specials = [rng.choice(train_tokens_special) for _ in range(n_special)]
    sampled_plains = [rng.choice(train_tokens_plain) for _ in range(n_plain)]
    extra_words = sampled_specials + sampled_plains
    rng.shuffle(extra_words)

    # Strict Leak Check
    leak_count = sum(1 for w in extra_words if w in val_test_words)
    print(f"Word Leak Check: {leak_count} overlapping tokens with val/test (Must be 0!)")
    assert leak_count == 0, "Data leakage detected!"

    # Count special tokens in extra_words
    actual_specials = sum(1 for w in extra_words if digits_re.search(w) or w[0] in punct_set or w[-1] in punct_set)
    print(f"Extra words special ratio: {actual_specials} / {len(extra_words)} ({actual_specials/len(extra_words)*100:.2f}%)")

    # 4. Render extra word images with fonts_new
    fonts_new_paths = [str(p) for p in Path("fonts_new").glob("*.ttf") if p.is_file()]
    print(f"Rendering 15000 extra words with fonts: {[Path(p).name for p in fonts_new_paths]}")

    temp_word_dir = Path("data/temp_extra_word")
    if temp_word_dir.exists():
        shutil.rmtree(temp_word_dir)

    make_split(extra_words, fonts_new_paths, temp_word_dir, size=64, seed=42, bgs=[0, 1])

    # Merge into data/synth_word_v2/train
    dst_word_train = Path("data/synth_word_v2/train")
    (dst_word_train / "images").mkdir(parents=True, exist_ok=True)

    # Copy existing train images and labels
    print("Copying existing 20000 synth_word train images...")
    merged_word_labels = []
    src_word_train = Path("data/synth_word/train")
    with open(src_word_train / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            fn, lbl = line.strip().split("\t", 1)
            shutil.copy2(src_word_train / fn, dst_word_train / fn)
            merged_word_labels.append((fn, lbl))

    # Append extra 15000 renumbered images
    print("Merging extra 15000 word images...")
    base_idx = len(merged_word_labels) # 20000
    with open(temp_word_dir / "labels.tsv", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            fn, lbl = line.strip().split("\t", 1)
            new_fn = f"images/{base_idx + idx:07d}.png"
            shutil.copy2(temp_word_dir / fn, dst_word_train / new_fn)
            merged_word_labels.append((new_fn, lbl))

    with open(dst_word_train / "labels.tsv", "w", encoding="utf-8") as f:
        for fn, lbl in merged_word_labels:
            f.write(f"{fn}\t{lbl}\n")
    print(f"synth_word_v2/train complete: {len(merged_word_labels)} images.")
    shutil.rmtree(temp_word_dir)

    # 5. Render extra line images (+6,000)
    print("\nPreparing extra 6000 lines from data/corpus_line/train.txt...")
    with open("data/corpus_line/train.txt", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]

    extra_lines = [rng.choice(train_lines) for _ in range(6000)]
    temp_line_dir = Path("data/temp_extra_line")
    if temp_line_dir.exists():
        shutil.rmtree(temp_line_dir)

    make_split(extra_lines, fonts_new_paths, temp_line_dir, size=48, seed=42, bgs=[0, 1])

    # Merge into data/synth_line_v2/train
    dst_line_train = Path("data/synth_line_v2/train")
    (dst_line_train / "images").mkdir(parents=True, exist_ok=True)

    print("Copying existing 10000 synth_line train images...")
    merged_line_labels = []
    src_line_train = Path("data/synth_line/train")
    with open(src_line_train / "labels.tsv", encoding="utf-8") as f:
        for line in f:
            fn, lbl = line.strip().split("\t", 1)
            shutil.copy2(src_line_train / fn, dst_line_train / fn)
            merged_line_labels.append((fn, lbl))

    print("Merging extra 6000 line images...")
    base_idx = len(merged_line_labels) # 10000
    with open(temp_line_dir / "labels.tsv", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            fn, lbl = line.strip().split("\t", 1)
            new_fn = f"images/{base_idx + idx:07d}.png"
            shutil.copy2(temp_line_dir / fn, dst_line_train / new_fn)
            merged_line_labels.append((new_fn, lbl))

    with open(dst_line_train / "labels.tsv", "w", encoding="utf-8") as f:
        for fn, lbl in merged_line_labels:
            f.write(f"{fn}\t{lbl}\n")
    print(f"synth_line_v2/train complete: {len(merged_line_labels)} images.")
    shutil.rmtree(temp_line_dir)

    print("\n=== All datasets created successfully ===")

if __name__ == "__main__":
    main()
