import os
import sys
import json
import unicodedata
import cv2
import numpy as np
import pandas as pd
import shutil
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

def main():
    print("=" * 80)
    print("AZOCR-V2: FINAL CONSISTENT REPRODUCIBILITY AUDIT SCRIPT")
    print("=" * 80)
    
    # Backup old verified_alignment.csv if not backed up
    if not os.path.exists('results/real30_evaluation/verified_alignment_v1_backup.csv'):
        shutil.copy('results/real30_evaluation/verified_alignment.csv', 'results/real30_evaluation/verified_alignment_v1_backup.csv')
        
    ver_df = pd.read_csv('results/real30_evaluation/verified_alignment.csv')
    gt_df = pd.read_csv('data/real30_evaluation/ground_truth_lines.csv')
    raw_df = pd.read_csv('results/real30_evaluation/raw_predictions.csv')
    
    # -------------------------------------------------------------
    # 1. UPDATE VERIFIED ALIGNMENT: P29 CROP 06 -> MERGED_MULTILINE
    # -------------------------------------------------------------
    # In P29 crop 06: it contains two text lines (Title Line 0 + Line 1).
    p29_mask = (ver_df['page_id'] == 'P29') & (ver_df['crop_id'] == 6)
    ver_df.loc[p29_mask, 'verified_crop_category'] = 'merged_multiline'
    ver_df.loc[p29_mask, 'verification_notes'] = (
        "Visually and textually confirmed merged multiline block (H=92px): contains both GT Line 0 (title) "
        "and GT Line 1 (body text 'Yaşayış məhəlləsində...')."
    )
    
    # Ensure P13 crop 12, 14 and P14 crop 8, 15 are documented with clear visual bleed notes
    p13_c12_mask = (ver_df['page_id'] == 'P13') & (ver_df['crop_id'] == 12)
    ver_df.loc[p13_c12_mask, 'verification_notes'] = "Merged multiline / bleed: contains title line plus visible text ascenders of next line at bottom."
    
    p13_c14_mask = (ver_df['page_id'] == 'P13') & (ver_df['crop_id'] == 14)
    ver_df.loc[p13_c14_mask, 'verification_notes'] = "Merged multiline block: contains text from both Line 1 and Line 2."
    
    p14_c08_mask = (ver_df['page_id'] == 'P14') & (ver_df['crop_id'] == 8)
    ver_df.loc[p14_c08_mask, 'verification_notes'] = "Merged multiline / bleed: contains title plus text fragments of next line."
    
    p14_c15_mask = (ver_df['page_id'] == 'P14') & (ver_df['crop_id'] == 15)
    ver_df.loc[p14_c15_mask, 'verification_notes'] = "Merged multiline block: multi-line paragraph block spanning multiple printed lines."

    ver_df.to_csv('results/real30_evaluation/verified_alignment.csv', index=False)
    print("\n[+] Updated verified_alignment.csv with unified visual determinations.")
    print("Verified Crop Category Counts:")
    print(ver_df['verified_crop_category'].value_counts())
    
    num_single = int((ver_df['verified_crop_category'] == 'genuine_single_line').sum())
    num_merged = int((ver_df['verified_crop_category'] == 'merged_multiline').sum())
    num_split = int((ver_df['verified_crop_category'] == 'split_fragment').sum())
    num_noise = int((ver_df['verified_crop_category'] == 'noise').sum())
    print(f"Total: {num_single} single, {num_merged} merged, {num_split} split, {num_noise} noise -> {num_single+num_merged+num_split+num_noise} crops")

    # -------------------------------------------------------------
    # 2. VISUAL AUDIT TARGETS DATA (Point 3: GT from ver_df)
    # -------------------------------------------------------------
    target_crops_info = [
        ('P13', 12, 'Title Line + Next Line Bleed'),
        ('P13', 14, 'Body Lines 1 & 2 Merged'),
        ('P14', 8, 'Title Line + Next Line Bleed'),
        ('P14', 15, 'Multiline Paragraph Block'),
        ('P15', 3, 'Genuine Single Title Line'),
        ('P15', 4, 'Paragraph 1 Merged Block (5 lines)'),
        ('P29', 6, 'Title Line 0 + Body Line 1 Merged')
    ]
    
    visual_audit_dict = {}
    for pid, cid, lbl in target_crops_info:
        r_ver = ver_df[(ver_df['page_id'] == pid) & (ver_df['crop_id'] == cid)].iloc[0]
        r_raw = raw_df[(raw_df['page_id'] == pid) & (raw_df['crop_id'] == cid)].iloc[0]
        
        visual_audit_dict[f"{pid}_crop{cid:02d}"] = {
            'page_id': pid,
            'crop_id': cid,
            'label': lbl,
            'bbox': [int(r_ver['bbox_x0']), int(r_ver['bbox_y0']), int(r_ver['bbox_x1']), int(r_ver['bbox_y1'])],
            'height': int(r_ver['crop_h']),
            'width': int(r_ver['crop_w']),
            'verified_crop_category': r_ver['verified_crop_category'],
            'verified_matched_gt_line_no': int(r_ver['verified_matched_gt_line_no']),
            'verified_ground_truth_text': r_ver['verified_ground_truth_text'], # POINT 3 FIXED!
            'paddleocr': str(r_raw['paddleocr']),
            'tesseract5_psm7': str(r_raw['tesseract5_psm7']),
            'crnn_v2_raw': str(r_raw['crnn_v2_raw']),
            'visual_determination': r_ver['verified_crop_category']
        }

    # -------------------------------------------------------------
    # 3. RECONCILE UNSEGMENTED GT LINES (Points 5 & 6)
    # -------------------------------------------------------------
    # In ver_df: genuine_single_line = 325.
    # Non-single lines = 421 - 325 = 96 lines.
    singles = ver_df[ver_df['verified_crop_category'] == 'genuine_single_line']
    matched_gt_set = set(zip(singles['page_id'], singles['verified_matched_gt_line_no']))
    
    merged_coverage = {
        ('P02', 10): 'P02_crop16',
        ('P02', 11): 'P02_crop16',
        ('P13', 0): 'P13_crop12',
        ('P13', 1): 'P13_crop14',
        ('P14', 0): 'P14_crop08',
        ('P14', 7): 'P14_crop15',
        ('P15', 1): 'P15_crop04',
        ('P15', 2): 'P15_crop04',
        ('P15', 3): 'P15_crop04',
        ('P15', 4): 'P15_crop04',
        ('P15', 5): 'P15_crop04',
        ('P15', 6): 'P15_crop05',
        ('P15', 7): 'P15_crop05',
        ('P15', 8): 'P15_crop05',
        ('P15', 9): 'P15_crop05',
        ('P15', 10): 'P15_crop05',
        ('P15', 11): 'P15_crop06',
        ('P15', 12): 'P15_crop06',
        ('P15', 13): 'P15_crop06',
        ('P15', 14): 'P15_crop06',
        ('P29', 0): 'P29_crop06' # Added!
    }
    
    split_fragments = ver_df[ver_df['verified_crop_category'] == 'split_fragment']
    split_gt_map = {}
    for _, r in split_fragments.iterrows():
        if r['verified_matched_gt_line_no'] != -1:
            split_gt_map[(r['page_id'], r['verified_matched_gt_line_no'])] = f"{r['page_id']}_crop{r['crop_id']:02d}"

    categorized_unsegmented = []
    for _, r in gt_df.iterrows():
        pair = (r['page_id'], r['line_no'])
        if pair in matched_gt_set:
            continue
            
        gt_id = f"{r['page_id']}_L{r['line_no']:02d}"
        gt_text = r['ground_truth_nfc']
        
        if pair in merged_coverage:
            cat = 'engulfed_in_merged_block'
            ref_crop = merged_coverage[pair]
            desc = f"Engulfed in merged multiline block ({ref_crop})"
        elif pair in split_gt_map:
            cat = 'captured_in_split_fragment'
            ref_crop = split_gt_map[pair]
            desc = f"Partially captured in oversegmented fragment ({ref_crop})"
        else:
            cat = 'completely_unsegmented_missed'
            ref_crop = 'None'
            desc = "Completely unsegmented / skipped by contour projection"
            
        categorized_unsegmented.append({
            'gt_id': gt_id,
            'page_id': r['page_id'],
            'line_no': int(r['line_no']),
            'category': cat,
            'reference_crop': ref_crop,
            'description': desc,
            'ground_truth_text': gt_text
        })
        
    df_unseg = pd.DataFrame(categorized_unsegmented)
    print(f"\n[+] Total Unsegmented / Non-Single GT Lines: {len(df_unseg)}")
    print(df_unseg['category'].value_counts())
    
    # Save both unsegmented_96_gt_lines.csv and update unsegmented_95_gt_lines.csv
    df_unseg.to_csv('results/real30_evaluation/unsegmented_gt_lines.csv', index=False)
    df_unseg.to_csv('results/real30_evaluation/unsegmented_95_gt_lines.csv', index=False)

    # -------------------------------------------------------------
    # 4. RECOMPUTE CONTINGENCY MATRIX (Point 3)
    # -------------------------------------------------------------
    raw_df['crnn_len'] = raw_df['crnn_v2_raw'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    merged_crops = raw_df.merge(
        ver_df[['page_id', 'crop_id', 'verified_crop_category', 'verified_matched_gt_line_no']], 
        on=['page_id', 'crop_id']
    )
    
    ct_counts = pd.crosstab(merged_crops['alignment_status'], merged_crops['verified_crop_category'], margins=True)
    ct_crnn = pd.crosstab(merged_crops['alignment_status'], merged_crops['verified_crop_category'], values=merged_crops['crnn_len'], aggfunc='sum', margins=True)
    
    ct_counts.to_csv('results/real30_evaluation/crop_status_contingency_matrix.csv')
    print("\n[+] Updated crop_status_contingency_matrix.csv:")
    print(ct_counts)
    print("\nCRNN Character Totals:")
    print(ct_crnn)

    # -------------------------------------------------------------
    # 5. RECOMPUTE METRICS ON 325 CLEAN SINGLE LINES (Point 5)
    # -------------------------------------------------------------
    models = ['crnn_v2_raw', 'tesseract5_psm7', 'easyocr', 'paddleocr']
    model_display = {
        'crnn_v2_raw': 'CRNN-V2_Raw',
        'tesseract5_psm7': 'Tesseract5_PSM7',
        'easyocr': 'EasyOCR',
        'paddleocr': 'PaddleOCR'
    }
    
    cohort_325_metrics = {}
    for m in models:
        total_d_c, total_l_c, total_d_w, total_l_w = 0, 0, 0, 0
        line_cers, line_wers = [], []
        exact_matches = 0
        
        for _, r in singles.iterrows():
            gt = r['verified_ground_truth_text']
            pred = r[m]
            d_c, l_c, d_w, l_w, cer, wer = cer_wer(gt, pred)
            total_d_c += d_c; total_l_c += l_c; total_d_w += d_w; total_l_w += l_w
            line_cers.append(cer); line_wers.append(wer)
            if nfc(gt) == nfc(pred):
                exact_matches += 1
                
        disp = model_display[m]
        cohort_325_metrics[disp] = {
            'evaluated_lines': len(singles),
            'cohort_type': 'selected_clean_subcohort_325',
            'line_cer_micro_pct': round(total_d_c / max(1, total_l_c) * 100.0, 2),
            'line_cer_macro_pct': round(float(np.mean(line_cers)) * 100.0, 2),
            'line_wer_micro_pct': round(total_d_w / max(1, total_l_w) * 100.0, 2),
            'line_wer_macro_pct': round(float(np.mean(line_wers)) * 100.0, 2),
            'exact_matches': exact_matches,
            'exact_match_rate_pct': round(exact_matches / len(singles) * 100.0, 2)
        }
    print("\n[+] Recomputed 325 Clean Lines Metrics:")
    print(json.dumps(cohort_325_metrics, indent=2))

    # -------------------------------------------------------------
    # 6. CHARACTER ACCURACY ON 325 CLEAN SINGLE LINES (Point 5)
    # -------------------------------------------------------------
    target_chars = ['ə', 'Ə', 'ı', 'i', 'İ', 'I']
    char_counts_gt_325 = {ch: 0 for ch in target_chars}
    for _, r in singles.iterrows():
        gt = nfc(r['verified_ground_truth_text'])
        for ch in target_chars:
            char_counts_gt_325[ch] += gt.count(ch)
            
    char_stats_325 = {}
    for m in models:
        disp = model_display[m]
        char_stats_325[disp] = {}
        for ch in target_chars:
            denom = char_counts_gt_325[ch]
            if denom == 0:
                char_stats_325[disp][ch] = {'gt_count': 0, 'correct': 0, 'accuracy_pct': 'N/A'}
                continue
            correct = 0
            for _, r in singles.iterrows():
                gt = nfc(r['verified_ground_truth_text'])
                pred = nfc(r[m])
                sm = difflib.SequenceMatcher(None, gt, pred)
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag == 'equal':
                        correct += gt[i1:i2].count(ch)
            char_stats_325[disp][ch] = {
                'gt_count': denom,
                'correct': correct,
                'accuracy_pct': round(correct / denom * 100.0, 2)
            }

    # -------------------------------------------------------------
    # 7. PAGE CER PRIMARY & SENSITIVITY PROTOCOLS (Point 7)
    # -------------------------------------------------------------
    page_primary_space = {}
    page_sensitivity_newline = {}
    
    for m in models:
        # Primary: Space separated (non-empty crops)
        d_c_sp_tot, l_c_sp_tot, d_w_sp_tot, l_w_sp_tot = 0, 0, 0, 0
        cers_sp, wers_sp = [], []
        # Sensitivity: Newline separated (all crops)
        d_c_nl_tot, l_c_nl_tot, d_w_nl_tot, l_w_nl_tot = 0, 0, 0, 0
        cers_nl, wers_nl = [], []
        
        for pid in sorted(raw_df['page_id'].unique()):
            p_gt = gt_df[gt_df['page_id'] == pid].sort_values('line_no')
            p_crops = raw_df[raw_df['page_id'] == pid].sort_values(['bbox_y0', 'bbox_x0'])
            
            # Primary (Space)
            gt_sp = " ".join([nfc(t) for t in p_gt['ground_truth_nfc']])
            pred_sp = " ".join([nfc(t) for t in p_crops[m] if pd.notna(t) and len(nfc(t)) > 0])
            d_c_sp, l_c_sp, d_w_sp, l_w_sp, cer_sp, wer_sp = cer_wer(gt_sp, pred_sp)
            d_c_sp_tot += d_c_sp; l_c_sp_tot += l_c_sp; d_w_sp_tot += d_w_sp; l_w_sp_tot += l_w_sp
            cers_sp.append(cer_sp); wers_sp.append(wer_sp)
            
            # Sensitivity (Newline)
            gt_nl = "\n".join([nfc(t) for t in p_gt['ground_truth_nfc']])
            pred_nl = "\n".join([nfc(t) if pd.notna(t) else "" for t in p_crops[m]])
            d_c_nl, l_c_nl, d_w_nl, l_w_nl, cer_nl, wer_nl = cer_wer(gt_nl, pred_nl)
            d_c_nl_tot += d_c_nl; l_c_nl_tot += l_c_nl; d_w_nl_tot += d_w_nl; l_w_nl_tot += l_w_nl
            cers_nl.append(cer_nl); wers_nl.append(wer_nl)
            
        disp = model_display[m]
        page_primary_space[disp] = {
            'page_cer_micro_pct': round(d_c_sp_tot / max(1, l_c_sp_tot) * 100.0, 2),
            'page_cer_macro_pct': round(float(np.mean(cers_sp)) * 100.0, 2),
            'page_wer_micro_pct': round(d_w_sp_tot / max(1, l_w_sp_tot) * 100.0, 2),
            'page_wer_macro_pct': round(float(np.mean(wers_sp)) * 100.0, 2),
            'total_gt_chars': l_c_sp_tot,
            'total_edit_distance': d_c_sp_tot
        }
        page_sensitivity_newline[disp] = {
            'page_cer_micro_pct': round(d_c_nl_tot / max(1, l_c_nl_tot) * 100.0, 2),
            'page_cer_macro_pct': round(float(np.mean(cers_nl)) * 100.0, 2),
            'page_wer_micro_pct': round(d_w_nl_tot / max(1, l_w_nl_tot) * 100.0, 2),
            'page_wer_macro_pct': round(float(np.mean(wers_nl)) * 100.0, 2),
            'total_gt_chars': l_c_nl_tot,
            'total_edit_distance': d_c_nl_tot
        }

    # Save final JSON
    output_audit = {
        'audit_metadata': {
            'status': 'final_consistent_reproducibility_audit',
            'dataset': 'Real30_P01_P30',
            'total_pages': 30,
            'total_ground_truth_lines': 421,
            'total_segmented_crops': 690,
            'crop_breakdown': {
                'genuine_single_line': num_single,
                'merged_multiline': num_merged,
                'split_fragment': num_split,
                'noise': num_noise
            },
            'selected_clean_lines_subcohort': num_single,
            'unsegmented_non_single_lines': len(df_unseg)
        },
        'visual_inspection_targets': visual_audit_dict,
        'contingency_matrix_crops_and_chars': {
            'crop_counts': ct_counts.to_dict(),
            'crnn_character_counts': ct_crnn.to_dict()
        },
        'selected_subcohort_325_metrics': cohort_325_metrics,
        'character_analysis_325_lines': char_stats_325,
        'full_page_cer_evaluation': {
            'primary_protocol_space_separated': page_primary_space,
            'sensitivity_protocol_newline_separated': page_sensitivity_newline,
            'protocol_definition': {
                'primary': 'Space separator (" ") with non-empty recognized text strips. Eliminates artificial consecutive newline insertion penalties from empty noise crops.',
                'sensitivity': 'Newline separator ("\\n") with all 690 crops included. Reflects strict line-break preservation including empty crop penalty.'
            }
        },
        'unsegmented_96_gt_lines_summary': {
            'total_lines': len(df_unseg),
            'breakdown': df_unseg['category'].value_counts().to_dict(),
            'unique_ids': df_unseg['gt_id'].tolist()
        }
    }
    
    with open('results/real30_evaluation/reproducibility_audit_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(output_audit, f, indent=2, ensure_ascii=False)
    print("\n[+] Successfully saved results/real30_evaluation/reproducibility_audit_metrics.json")

if __name__ == '__main__':
    main()
