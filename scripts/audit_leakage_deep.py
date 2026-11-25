import os
import re
import sys
import json
import unicodedata
from collections import Counter, defaultdict

def nfc(s):
    return unicodedata.normalize('NFC', s.strip())

def read_lines_txt(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return [line.rstrip('\r\n') for line in f]

def read_labels_tsv(path):
    if not os.path.exists(path):
        return []
    res = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 2:
                img, text = parts[0], parts[1]
                res.append((img, text))
            elif len(parts) == 1 and parts[0]:
                res.append((f"line_{idx}", parts[0]))
    return res

def main():
    print("=== DATASET INSPECTION ===")
    
    # 1. Corpus files
    c_train = read_lines_txt("data/corpus_line/train.txt")
    c_val = read_lines_txt("data/corpus_line/val.txt")
    c_test = read_lines_txt("data/corpus_line/test.txt")
    
    print(f"corpus_line/train.txt: total={len(c_train)}, unique={len(set(c_train))}, unique_NFC={len(set(nfc(x) for x in c_train))}")
    print(f"corpus_line/val.txt: total={len(c_val)}, unique={len(set(c_val))}, unique_NFC={len(set(nfc(x) for x in c_val))}")
    print(f"corpus_line/test.txt: total={len(c_test)}, unique={len(set(c_test))}, unique_NFC={len(set(nfc(x) for x in c_test))}")
    
    # Check internal duplicates
    for name, lines in [("corpus_train", c_train), ("corpus_val", c_val), ("corpus_test", c_test)]:
        c = Counter(lines)
        dups = {k: v for k, v in c.items() if v > 1}
        print(f"  {name} internal exact duplicates: {len(dups)} distinct strings, {sum(dups.values()) - len(dups)} redundant occurrences")

    # Overlaps between corpus splits
    train_set = set(c_train)
    val_set = set(c_val)
    test_set = set(c_test)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    print(f"Corpus train & val overlap: {len(train_val_overlap)}")
    print(f"Corpus train & test overlap: {len(train_test_overlap)}")
    print(f"Corpus val & test overlap: {len(val_test_overlap)}")

    # Overlaps NFC
    train_set_nfc = set(nfc(x) for x in c_train)
    val_set_nfc = set(nfc(x) for x in c_val)
    test_set_nfc = set(nfc(x) for x in c_test)
    print(f"Corpus train & val overlap (NFC): {len(train_set_nfc & val_set_nfc)}")
    print(f"Corpus train & test overlap (NFC): {len(train_set_nfc & test_set_nfc)}")
    print(f"Corpus val & test overlap (NFC): {len(val_set_nfc & test_set_nfc)}")

    # 2. Synth V1 labels
    v1_train = read_labels_tsv("data/synth_line_v1/train/labels.tsv")
    v1_val = read_labels_tsv("data/synth_line_v1/val/labels.tsv")
    v1_train_texts = [t for _, t in v1_train]
    v1_val_texts = [t for _, t in v1_val]
    print(f"\nsynth_line_v1/train: total={len(v1_train)}, unique_text={len(set(v1_train_texts))}")
    print(f"synth_line_v1/val: total={len(v1_val)}, unique_text={len(set(v1_val_texts))}")
    print(f"synth_line_v1 train & val text overlap: {len(set(v1_train_texts) & set(v1_val_texts))}")

    # 3. Synth V2 labels
    v2_train = read_labels_tsv("data/synth_line_v2/train/labels.tsv")
    v2_val = read_labels_tsv("data/synth_line_v2/val/labels.tsv")
    v2_test = read_labels_tsv("data/synth_line_v2/test/labels.tsv")
    v2_train_texts = [t for _, t in v2_train]
    v2_val_texts = [t for _, t in v2_val]
    v2_test_texts = [t for _, t in v2_test]
    print(f"\nsynth_line_v2/train: total={len(v2_train)}, unique_text={len(set(v2_train_texts))}")
    print(f"synth_line_v2/val: total={len(v2_val)}, unique_text={len(set(v2_val_texts))}")
    print(f"synth_line_v2/test: total={len(v2_test)}, unique_text={len(set(v2_test_texts))}")
    
    print(f"synth_line_v2 train & val text overlap: {len(set(v2_train_texts) & set(v2_val_texts))}")
    print(f"synth_line_v2 train & test text overlap: {len(set(v2_train_texts) & set(v2_test_texts))}")
    print(f"synth_line_v2 val & test text overlap: {len(set(v2_val_texts) & set(v2_test_texts))}")

    # What about 100 and 84? Let's check val/test sizes or subsets!
    # Let's inspect where 100 and 84 could come from.
    # What was val/test size in some split?
    # Notice v1_val / v2_val size:
    print(f"v1_val size: {len(v1_val)}, v2_val size: {len(v2_val)}, v2_test size: {len(v2_test)}")

    # 4. Check benchmark lines (B1-B15)
    import generate_and_test_unseen as bm
    print("\nBenchmark docs loaded from generate_and_test_unseen.py:")
    bm_lines = []
    # Real photos P1, P2, P3
    for p_name, refs in [("P1", bm.P1_REFS), ("P2", bm.P2_REFS), ("P3", bm.P3_REFS)]:
        for idx, r in enumerate(refs):
            bm_lines.append((f"{p_name}_{idx+1}", r))
    # Synthetic docs B4-B15
    for doc_name, text_block in bm.GENERATED_PHONE_DOCS.items():
        doc_lines = [l.strip() for l in text_block.strip().split('\n') if l.strip()]
        for idx, l in enumerate(doc_lines):
            bm_lines.append((f"{doc_name}_{idx+1}", l))
            
    print(f"Total benchmark lines: {len(bm_lines)}")
    bm_texts = [t for _, t in bm_lines]
    bm_set = set(bm_texts)
    bm_set_nfc = set(nfc(t) for t in bm_texts)

    # Check benchmark leakage into training/val/test
    print("\n=== BENCHMARK LEAKAGE CHECK ===")
    # vs c_train
    bm_in_c_train = bm_set & train_set
    print(f"Benchmark in corpus_line/train.txt: {len(bm_in_c_train)}")
    bm_in_c_train_nfc = bm_set_nfc & train_set_nfc
    print(f"Benchmark in corpus_line/train.txt (NFC): {len(bm_in_c_train_nfc)}")

    # vs v1_train
    bm_in_v1_train = bm_set & set(v1_train_texts)
    print(f"Benchmark in synth_line_v1/train: {len(bm_in_v1_train)}")

    # vs v2_train
    bm_in_v2_train = bm_set & set(v2_train_texts)
    print(f"Benchmark in synth_line_v2/train: {len(bm_in_v2_train)}")

    # vs corpus val/test
    print(f"Benchmark in corpus_line/val.txt: {len(bm_set & val_set)}")
    print(f"Benchmark in corpus_line/test.txt: {len(bm_set & test_set)}")
    print(f"Benchmark in synth_line_v2/val: {len(bm_set & set(v2_val_texts))}")
    print(f"Benchmark in synth_line_v2/test: {len(bm_set & set(v2_test_texts))}")

if __name__ == "__main__":
    main()
