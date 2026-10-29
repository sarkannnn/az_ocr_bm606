import json, sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import common

# 1. Photo 2
with open('results/real_test2/results.json', encoding='utf-8') as f:
    d2 = json.load(f)

lines2 = d2['lines']['details']
words2 = d2['words']['details']

print("==================================================")
print("=== PHOTO 2: 21 LINES, 89 WORDS, 837 CHARS ===")
print("==================================================")

# Let's check exact match rules for Photo 2 lines:
print("\n--- Line exact matches for V2, Tesseract, V1 ---")
for model_key in ['v2', 'tes', 'v1']:
    exact_raw = 0
    exact_strip = 0
    exact_norm_strip = 0
    err_lines = 0
    for idx, x in enumerate(lines2):
        gt = x['gt']
        pred = x[model_key]
        if idx == 0 and model_key == 'tes': # if fixed
            pass
        if pred == gt:
            exact_raw += 1
        if pred.strip() == gt.strip():
            exact_strip += 1
        if common.normalize(pred).strip() == common.normalize(gt).strip():
            exact_norm_strip += 1
        else:
            err_lines += 1
            if model_key == 'v2':
                print(f"  V2 Error Line {idx:02d}:")
                print(f"    GT:   '{gt}'")
                print(f"    Pred: '{pred}'")
    print(f"{model_key.upper()}: exact_raw={exact_raw}, exact_strip={exact_strip}, exact_norm_strip={exact_norm_strip}, error_lines={err_lines}")

# Recompute CER and WER with common.cer_wer:
# Original Photo 2 lines
r_l2 = [x['gt'] for x in lines2]
p_l2_v1 = [x['v1'] for x in lines2]
p_l2_v2 = [x['v2'] for x in lines2]
p_l2_tes = [x['tes'] for x in lines2]

# Fixed Line 0 (TƏQDİMAT)
p_l2_v1_f = list(p_l2_v1)
p_l2_v1_f[0] = 'jirəggitifTijj'
p_l2_v2_f = list(p_l2_v2)
p_l2_v2_f[0] = ' TəqDiMAT ı'
p_l2_tes_f = list(p_l2_tes)
p_l2_tes_f[0] = 'TƏQDİMAT'

print("\n--- Photo 2 Lines (common.cer_wer, Fixed Crop) ---")
for name, hyps in [('V1', p_l2_v1_f), ('V2', p_l2_v2_f), ('Tesseract', p_l2_tes_f)]:
    c, w, _ = common.cer_wer(r_l2, hyps)
    # counts
    tot_c = sum(len(r) for r in r_l2)
    tot_w = sum(len(r.split()) for r in r_l2)
    err_c = sum(common.align(r, h)[0] for r, h in zip(r_l2, hyps))
    err_w = sum(common.align(r.split(), h.split())[0] for r, h in zip(r_l2, hyps))
    print(f"{name:10s} CER={c*100:6.2f}% ({err_c:3d}/{tot_c:3d}), WER={w*100:6.2f}% ({err_w:3d}/{tot_w:3d})")

print("\n==================================================")
print("=== PHOTO 2: 30 WORDS, 304 CHARS ===")
print("==================================================")
r_w2 = [x['gt'] for x in words2]
p_w2_v1 = [x['v1'] for x in words2]
p_w2_v2 = [x['v2'] for x in words2]
p_w2_tes = [x['tes'] for x in words2]

for name, hyps in [('V1', p_w2_v1), ('V2', p_w2_v2), ('Tesseract', p_w2_tes)]:
    c, w, _ = common.cer_wer(r_w2, hyps)
    tot_c = sum(len(r) for r in r_w2)
    tot_w = sum(len(r.split()) for r in r_w2)
    err_c = sum(common.align(r, h)[0] for r, h in zip(r_w2, hyps))
    err_w = sum(common.align(r.split(), h.split())[0] for r, h in zip(r_w2, hyps))
    exact_w = sum(1 for r, h in zip(r_w2, hyps) if r == h)
    print(f"{name:10s} CER={c*100:6.2f}% ({err_c:3d}/{tot_c:3d}), WER={w*100:6.2f}% ({err_w:3d}/{tot_w:3d}), exact={exact_w}/30")

print("\n==================================================")
print("=== PHOTO 1: 15 LINES (762 CHARS) & 30 WORDS (285 CHARS) ===")
print("==================================================")
# Parse Photo 1 from results/real_test/evaluation.md
with open('results/real_test/evaluation.md', encoding='utf-8') as f:
    content = f.read()

# Lines
p1_lines_blocks = content.split('## 2. Real Sətirlər Üzrə Təfərrüatlı Nəticələr (Line by Line)')[1].split('## 3. Real Sözlər Üzrə Nəticələr (Word by Word)')[0]
p1_words_block = content.split('## 3. Real Sözlər Üzrə Nəticələr (Word by Word)')[1]

import re
lines_gt = []
lines_v2 = []
lines_tes = []
lines_v1 = []

for block in p1_lines_blocks.split('### Sətir #')[1:]:
    m_gt = re.search(r'\*\*Əsl Mətn \(Düzgün\):\*\*\s*`([^`]*)`', block)
    m_v2 = re.search(r'\*\*V2 Modeli \(Bizim\):\*\*\s*`([^`]*)`', block)
    m_t = re.search(r'\*\*Tesseract:\*\*\s*`([^`]*)`', block)
    m_v1 = re.search(r'\*\*V1 Modeli \(Köhnə\):\*\*\s*`([^`]*)`', block)
    if m_gt and m_v2 and m_t and m_v1:
        lines_gt.append(m_gt.group(1))
        lines_v2.append(m_v2.group(1))
        lines_tes.append(m_t.group(1))
        lines_v1.append(m_v1.group(1))

print(f"Photo 1 parsed lines: {len(lines_gt)}")
for name, hyps in [('V1', lines_v1), ('V2', lines_v2), ('Tesseract', lines_tes)]:
    c, w, _ = common.cer_wer(lines_gt, hyps)
    tot_c = sum(len(r) for r in lines_gt)
    tot_w = sum(len(r.split()) for r in lines_gt)
    err_c = sum(common.align(r, h)[0] for r, h in zip(lines_gt, hyps))
    err_w = sum(common.align(r.split(), h.split())[0] for r, h in zip(lines_gt, hyps))
    print(f"Photo 1 Lines {name:10s} CER={c*100:6.2f}% ({err_c:3d}/{tot_c:3d}), WER={w*100:6.2f}% ({err_w:3d}/{tot_w:3d})")

# Words
words_gt = []
words_v2 = []
words_tes = []
words_v1 = []

for line in p1_words_block.splitlines():
    parts = [p.strip() for p in line.split('|')]
    # | # | Əsl Söz (Reference) | V2 Modeli (Bizim) | Tesseract | V1 Modeli | V2 Düzgündürmü? |
    if len(parts) >= 7 and parts[1].isdigit():
        def clean_val(s):
            m = re.search(r'`([^`]*)`', s)
            return m.group(1) if m else s.replace('*', '').strip()
        words_gt.append(clean_val(parts[2]))
        words_v2.append(clean_val(parts[3]))
        words_tes.append(clean_val(parts[4]))
        words_v1.append(clean_val(parts[5]))

print(f"\nPhoto 1 parsed words: {len(words_gt)}")
for name, hyps in [('V1', words_v1), ('V2', words_v2), ('Tesseract', words_tes)]:
    c, w, _ = common.cer_wer(words_gt, hyps)
    tot_c = sum(len(r) for r in words_gt)
    tot_w = sum(len(r.split()) for r in words_gt)
    err_c = sum(common.align(r, h)[0] for r, h in zip(words_gt, hyps))
    err_w = sum(common.align(r.split(), h.split())[0] for r, h in zip(words_gt, hyps))
    exact_w = sum(1 for r, h in zip(words_gt, hyps) if r == h)
    print(f"Photo 1 Words {name:10s} CER={c*100:6.2f}% ({err_c:3d}/{tot_c:3d}), WER={w*100:6.2f}% ({err_w:3d}/{tot_w:3d}), exact={exact_w}/30")
