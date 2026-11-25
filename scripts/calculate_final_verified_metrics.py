import pandas as pd
import numpy as np
import json
import unicodedata
import difflib

def nfc(text):
    if pd.isna(text):
        return ""
    return unicodedata.normalize('NFC', str(text).strip())

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cer_wer(ref, hyp):
    ref_norm = nfc(ref)
    hyp_norm = nfc(hyp)
    d_c = levenshtein(ref_norm, hyp_norm)
    cer = d_c / max(1, len(ref_norm))
    r_words = ref_norm.split()
    h_words = hyp_norm.split()
    if len(r_words) < len(h_words):
        d_w = levenshtein(h_words, r_words)
    elif len(h_words) == 0:
        d_w = len(r_words)
    else:
        prev = range(len(h_words) + 1)
        for i, w1 in enumerate(r_words):
            curr = [i + 1]
            for j, w2 in enumerate(h_words):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (w1 != w2)))
            prev = curr
        d_w = prev[-1]
    wer = d_w / max(1, len(r_words))
    return d_c, len(ref_norm), d_w, len(r_words), cer, wer

df = pd.read_csv('results/real30_evaluation/verified_alignment.csv')
gt_df = pd.read_csv('data/real30_evaluation/ground_truth_lines.csv')

models = ['crnn_v2_raw', 'tesseract5_psm7', 'easyocr', 'paddleocr']
model_display = {
    'crnn_v2_raw': 'CRNN-V2_Raw',
    'tesseract5_psm7': 'Tesseract5_PSM7',
    'easyocr': 'EasyOCR',
    'paddleocr': 'PaddleOCR'
}

# 1. Exact breakdown of CRNN 11,361 extra characters in the 285 extra crops
raw_df = pd.read_csv('results/real30_evaluation/raw_predictions.csv')
extra_crops = raw_df[raw_df['alignment_status'] == 'extra_crop'].copy()
extra_crops['crnn_len'] = extra_crops['crnn_v2_raw'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

# Merge with verified categories
extra_crops = extra_crops.merge(
    df[['page_id', 'crop_id', 'verified_crop_category']], 
    on=['page_id', 'crop_id'], 
    how='left'
)

# Categorize extra crops into exact subcategories
def get_subcat(r):
    cat = r['verified_crop_category']
    y0, y1 = r['bbox_y0'], r['bbox_y1']
    if cat == 'split_fragment':
        return 'split_text_fragment'
    elif cat == 'noise':
        if y0 <= 220 or y1 >= 1500:
            return 'border_desk_shadow_noise'
        else:
            return 'interline_paper_texture'
    elif cat == 'merged_multiline':
        return 'merged_multiline_block'
    else:
        return 'border_desk_shadow_noise'

extra_crops['subcat'] = extra_crops.apply(get_subcat, axis=1)

crnn_extra_breakdown = {}
total_extra_chars = int(extra_crops['crnn_len'].sum())
for sc, grp in extra_crops.groupby('subcat'):
    chars = int(grp['crnn_len'].sum())
    count = int(len(grp))
    pct = round(chars / max(1, total_extra_chars) * 100.0, 2)
    crnn_extra_breakdown[sc] = {
        'crop_count': count,
        'crnn_chars': chars,
        'percentage': pct
    }

print("CRNN EXTRA CHARACTERS EXACT BREAKDOWN:")
print(json.dumps(crnn_extra_breakdown, indent=2))

# 2. Compute metrics on 326 Genuine Single Lines
singles = df[df['verified_crop_category'] == 'genuine_single_line'].copy()
print(f"\nComputing metrics on {len(singles)} Genuine Single Lines...")

genuine_metrics = {}
for m in models:
    total_d_c = 0
    total_len_c = 0
    total_d_w = 0
    total_len_w = 0
    line_cers = []
    line_wers = []
    exact_matches = 0
    
    for _, r in singles.iterrows():
        gt = r['verified_ground_truth_text']
        pred = r[m]
        d_c, l_c, d_w, l_w, cer, wer = cer_wer(gt, pred)
        total_d_c += d_c
        total_len_c += l_c
        total_d_w += d_w
        total_len_w += l_w
        line_cers.append(cer)
        line_wers.append(wer)
        if nfc(gt) == nfc(pred):
            exact_matches += 1
            
    disp = model_display[m]
    genuine_metrics[disp] = {
        'matched_lines': len(singles),
        'line_cer_micro_pct': round(total_d_c / max(1, total_len_c) * 100.0, 2),
        'line_cer_macro_pct': round(float(np.mean(line_cers)) * 100.0, 2),
        'line_wer_micro_pct': round(total_d_w / max(1, total_len_w) * 100.0, 2),
        'line_wer_macro_pct': round(float(np.mean(line_wers)) * 100.0, 2),
        'exact_matches': exact_matches,
        'exact_match_rate_pct': round(exact_matches / len(singles) * 100.0, 2)
    }

print("\nGENUINE SINGLE LINES (326 lines) METRICS:")
print(json.dumps(genuine_metrics, indent=2))

# 3. Compute metrics on All Valid Text Lines (Genuine + Split Fragments = 326 + 42 = 368 lines)
valid_lines = df[df['verified_crop_category'].isin(['genuine_single_line', 'split_fragment'])].copy()
valid_metrics = {}
for m in models:
    total_d_c = 0
    total_len_c = 0
    total_d_w = 0
    total_len_w = 0
    line_cers = []
    line_wers = []
    exact_matches = 0
    
    for _, r in valid_lines.iterrows():
        gt = r['verified_ground_truth_text']
        pred = r[m]
        d_c, l_c, d_w, l_w, cer, wer = cer_wer(gt, pred)
        total_d_c += d_c
        total_len_c += l_c
        total_d_w += d_w
        total_len_w += l_w
        line_cers.append(cer)
        line_wers.append(wer)
        if nfc(gt) == nfc(pred):
            exact_matches += 1
            
    disp = model_display[m]
    valid_metrics[disp] = {
        'matched_lines': len(valid_lines),
        'line_cer_micro_pct': round(total_d_c / max(1, total_len_c) * 100.0, 2),
        'line_cer_macro_pct': round(float(np.mean(line_cers)) * 100.0, 2),
        'line_wer_micro_pct': round(total_d_w / max(1, total_len_w) * 100.0, 2),
        'line_wer_macro_pct': round(float(np.mean(line_wers)) * 100.0, 2),
        'exact_matches': exact_matches,
        'exact_match_rate_pct': round(exact_matches / len(valid_lines) * 100.0, 2)
    }

print("\nALL VALID TEXT LINES (368 lines) METRICS:")
print(json.dumps(valid_metrics, indent=2))

# 4. Page CER / WER (All 30 pages, all 421 lines, whole concatenated page text)
page_metrics = {}
for m in models:
    page_cers = []
    page_wers = []
    total_page_d_c = 0
    total_page_l_c = 0
    total_page_d_w = 0
    total_page_l_w = 0
    
    for pid in sorted(df['page_id'].unique()):
        # GT text for page
        p_gt = gt_df[gt_df['page_id'] == pid].sort_values('line_no')
        gt_full = " ".join([nfc(t) for t in p_gt['ground_truth_nfc']])
        
        # Predictions on page sorted by y0
        p_crops = df[df['page_id'] == pid].sort_values(['bbox_y0', 'bbox_x0'])
        pred_full = " ".join([nfc(t) for t in p_crops[m] if pd.notna(t) and len(nfc(t)) > 0])
        
        d_c, l_c, d_w, l_w, cer, wer = cer_wer(gt_full, pred_full)
        total_page_d_c += d_c
        total_page_l_c += l_c
        total_page_d_w += d_w
        total_page_l_w += l_w
        page_cers.append(cer)
        page_wers.append(wer)
        
    disp = model_display[m]
    page_metrics[disp] = {
        'page_cer_micro_pct': round(total_page_d_c / max(1, total_page_l_c) * 100.0, 2),
        'page_cer_macro_pct': round(float(np.mean(page_cers)) * 100.0, 2),
        'page_wer_micro_pct': round(total_page_d_w / max(1, total_page_l_w) * 100.0, 2),
        'page_wer_macro_pct': round(float(np.mean(page_wers)) * 100.0, 2)
    }

print("\nPAGE METRICS (30 pages):")
print(json.dumps(page_metrics, indent=2))

# 5. Character analysis on 326 Genuine Single Lines
target_chars = ['ə', 'Ə', 'ı', 'i', 'İ', 'I']
char_stats = {}

# Compute exact total occurrences in 326 genuine lines GT
char_counts_gt_326 = {ch: 0 for ch in target_chars}
for _, r in singles.iterrows():
    gt = nfc(r['verified_ground_truth_text'])
    for ch in target_chars:
        char_counts_gt_326[ch] += gt.count(ch)

for m in models:
    disp = model_display[m]
    char_stats[disp] = {}
    for ch in target_chars:
        denom = char_counts_gt_326[ch]
        if denom == 0:
            char_stats[disp][ch] = {
                'gt_count': 0,
                'correct': 0,
                'accuracy_pct': 'N/A'
            }
            continue
            
        correct = 0
        for _, r in singles.iterrows():
            gt = nfc(r['verified_ground_truth_text'])
            pred = nfc(r[m])
            # Count aligned matches
            sm = difflib.SequenceMatcher(None, gt, pred)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    correct += gt[i1:i2].count(ch)
                    
        acc = round(correct / denom * 100.0, 2)
        char_stats[disp][ch] = {
            'gt_count': denom,
            'correct': correct,
            'accuracy_pct': acc
        }

print("\nCHARACTER ACCURACY ON 326 GENUINE LINES:")
print(json.dumps(char_stats, indent=2))

# 6. Save final_verified_metrics.json
final_output = {
    'audit_metadata': {
        'status': 'verified_methodology_audit',
        'total_pages': 30,
        'total_gt_lines': 421,
        'total_crops': 690,
        'crop_breakdown': {
            'genuine_single_line': int((df['verified_crop_category'] == 'genuine_single_line').sum()),
            'split_fragment': int((df['verified_crop_category'] == 'split_fragment').sum()),
            'merged_multiline': int((df['verified_crop_category'] == 'merged_multiline').sum()),
            'noise': int((df['verified_crop_category'] == 'noise').sum())
        },
        'unmatched_missing_gt_lines': 421 - int((df['verified_crop_category'] == 'genuine_single_line').sum()),
        'crnn_extra_characters_exact_breakdown': crnn_extra_breakdown
    },
    'genuine_single_line_metrics_326': genuine_metrics,
    'all_valid_text_lines_metrics_368': valid_metrics,
    'page_full_text_metrics_30_pages': page_metrics,
    'character_analysis_326_lines': char_stats
}

with open('results/real30_evaluation/final_verified_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)
print("Saved final_verified_metrics.json.")
