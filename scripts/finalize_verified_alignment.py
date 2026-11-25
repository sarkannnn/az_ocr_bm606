import pandas as pd
import numpy as np
import difflib
import json
import unicodedata

# Load datasets
raw_df = pd.read_csv('results/real30_evaluation/raw_predictions.csv')
gt_df = pd.read_csv('data/real30_evaluation/ground_truth_lines.csv')

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

# Load prelim
prelim = pd.read_csv('results/real30_evaluation/verified_alignment_prelim.csv')

# Handle the 10 duplicate pairs: demote secondary to split_fragment
secondary_demotions = [
    ('P05', 16), # P05 GT 9 secondary
    ('P05', 19), # P05 GT 11 secondary
    ('P05', 21), # P05 GT 13 secondary
    ('P07', 18), # P07 GT 11 secondary
    ('P09', 25), # P09 GT 14 secondary
    ('P09', 28), # P09 GT 16 secondary
    ('P11', 27), # P11 GT 12 secondary
    ('P22', 19), # P22 GT 1 secondary
    ('P22', 32), # P22 GT 12 secondary
    ('P23', 18), # P23 GT 12 secondary
]

for pid, cid in secondary_demotions:
    mask = (prelim['page_id'] == pid) & (prelim['crop_id'] == cid)
    prelim.loc[mask, 'verified_crop_category'] = 'split_fragment'
    prelim.loc[mask, 'verification_notes'] += '; secondary overlap demoted from genuine_single_line'

print("Final Verified Crop Breakdown across all 690 crops:")
print(prelim['verified_crop_category'].value_counts())

# Save verified_alignment.csv
prelim.to_csv('results/real30_evaluation/verified_alignment.csv', index=False)
print("Saved verified_alignment.csv.")
