import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"

sys.path.insert(0, str(Path.cwd()))
from common import align, normalize

OUT_DIR = Path("results/real_fix9")
CONTACT_DIR = OUT_DIR / "contact_sheets"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

df_details = pd.read_csv(OUT_DIR / "per_line_chunk_details.csv")
with open(OUT_DIR / "crop_geometry.json", "r", encoding="utf-8") as f:
    line_meta = json.load(f)
with open(OUT_DIR / "empty_outputs_log.json", "r", encoding="utf-8") as f:
    empty_outputs_log = json.load(f)

# Mutually exclusive error taxonomy function
PUNCT_SET = set(".,:;!?—-\"'()[]{}«»“”„’…–/\\")

def categorize_line_errors(ref, hyp):
    ref = str(ref) if pd.notnull(ref) else ""
    hyp = str(hyp) if pd.notnull(hyp) else ""
    ce, ops = align(ref, hyp)
    if ce == 0:
        return 0, 0, 0, 0, 0, 0, []
    
    first_match_idx = None
    last_match_idx = None
    for idx, (op, rc, hc) in enumerate(ops):
        if op in ("ok", "sub"):
            if first_match_idx is None:
                first_match_idx = idx
            last_match_idx = idx
            
    leading_ins = 0
    trailing_ins = 0
    internal_space = 0
    internal_o0 = 0
    internal_punct = 0
    internal_letter = 0
    details = []
    
    for idx, (op, rc, hc) in enumerate(ops):
        if op == "ok":
            continue
            
        if op == "ins":
            if first_match_idx is not None and idx < first_match_idx:
                leading_ins += 1
                details.append({"category": "Leading Edge Insertion", "op": op, "ref": rc, "hyp": hc})
            elif last_match_idx is not None and idx > last_match_idx:
                trailing_ins += 1
                details.append({"category": "Trailing Edge Insertion", "op": op, "ref": rc, "hyp": hc})
            elif hc == ' ':
                internal_space += 1
                details.append({"category": "Internal Spacing Error", "op": op, "ref": rc, "hyp": hc})
            elif hc in PUNCT_SET:
                internal_punct += 1
                details.append({"category": "Internal Punctuation Error", "op": op, "ref": rc, "hyp": hc})
            elif (rc, hc) in [('0', 'O'), ('O', '0'), ('o', '0'), ('0', 'o')]:
                internal_o0 += 1
                details.append({"category": "Internal 0/O Error", "op": op, "ref": rc, "hyp": hc})
            else:
                internal_letter += 1
                details.append({"category": "Internal Letter Error", "op": op, "ref": rc, "hyp": hc})
                
        elif op == "del":
            if rc == ' ':
                internal_space += 1
                details.append({"category": "Internal Spacing Error", "op": op, "ref": rc, "hyp": hc})
            elif rc in PUNCT_SET:
                internal_punct += 1
                details.append({"category": "Internal Punctuation Error", "op": op, "ref": rc, "hyp": hc})
            elif (rc, hc) in [('0', 'O'), ('O', '0'), ('o', '0'), ('0', 'o')]:
                internal_o0 += 1
                details.append({"category": "Internal 0/O Error", "op": op, "ref": rc, "hyp": hc})
            else:
                internal_letter += 1
                details.append({"category": "Internal Letter Error", "op": op, "ref": rc, "hyp": hc})
                
        elif op == "sub":
            if rc == ' ' or hc == ' ':
                internal_space += 1
                details.append({"category": "Internal Spacing Error", "op": op, "ref": rc, "hyp": hc})
            elif (rc, hc) in [('0', 'O'), ('O', '0'), ('o', '0'), ('0', 'o')]:
                internal_o0 += 1
                details.append({"category": "Internal 0/O Error", "op": op, "ref": rc, "hyp": hc})
            elif (rc in PUNCT_SET) or (hc in PUNCT_SET):
                internal_punct += 1
                details.append({"category": "Internal Punctuation Error", "op": op, "ref": rc, "hyp": hc})
            else:
                internal_letter += 1
                details.append({"category": "Internal Letter Error", "op": op, "ref": rc, "hyp": hc})
                
    return leading_ins, trailing_ins, internal_space, internal_o0, internal_punct, internal_letter, details

def compute_metrics(preds, refs):
    tot_cer_errs = 0
    tot_wer_errs = 0
    tot_chars = 0
    tot_words = 0
    exact = 0
    tot_space_errors = 0
    for r, p in zip(refs, preds):
        r = str(r) if pd.notnull(r) else ""
        p = str(p) if pd.notnull(p) else ""
        ce, ops = align(r, p)
        we, _ = align(r.split(), p.split())
        sp_c = sum(1 for op, rc, hc in ops if op != "ok" and (rc == ' ' or hc == ' '))
        tot_cer_errs += ce
        tot_wer_errs += we
        tot_chars += len(r)
        tot_words += len(r.split())
        tot_space_errors += sp_c
        if p == r:
            exact += 1
    return {
        "cer": (tot_cer_errs / max(tot_chars, 1)) * 100.0,
        "wer": (tot_wer_errs / max(tot_words, 1)) * 100.0,
        "exact": exact,
        "total": len(refs),
        "cer_errs": tot_cer_errs,
        "chars": tot_chars,
        "wer_errs": tot_wer_errs,
        "words": tot_words,
        "space_errors_char": tot_space_errors
    }

def precompute_line_errors(preds, refs):
    cer_errs, wer_errs, chars, words = [], [], [], []
    for r, p in zip(refs, preds):
        r = str(r) if pd.notnull(r) else ""
        p = str(p) if pd.notnull(p) else ""
        ce, _ = align(r, p)
        we, _ = align(r.split(), p.split())
        cer_errs.append(ce)
        wer_errs.append(we)
        chars.append(len(r))
        words.append(len(r.split()))
    return np.array(cer_errs), np.array(wer_errs), np.array(chars), np.array(words)

def bootstrap_paired_diff_fast(v2_preds, tess_preds, refs, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    v2_ce, v2_we, v2_c, v2_w = precompute_line_errors(v2_preds, refs)
    tess_ce, tess_we, tess_c, tess_w = precompute_line_errors(tess_preds, refs)
    n = len(refs)
    
    cer_diffs = []
    wer_diffs = []
    for _ in range(n_boot):
        idxs = rng.choice(n, size=n, replace=True)
        v2_cer = (v2_ce[idxs].sum() / max(v2_c[idxs].sum(), 1)) * 100.0
        tess_cer = (tess_ce[idxs].sum() / max(tess_c[idxs].sum(), 1)) * 100.0
        v2_wer = (v2_we[idxs].sum() / max(v2_w[idxs].sum(), 1)) * 100.0
        tess_wer = (tess_we[idxs].sum() / max(tess_w[idxs].sum(), 1)) * 100.0
        cer_diffs.append(v2_cer - tess_cer)
        wer_diffs.append(v2_wer - tess_wer)
        
    return [float(np.percentile(cer_diffs, 2.5)), float(np.percentile(cer_diffs, 97.5))], [float(np.percentile(wer_diffs, 2.5)), float(np.percentile(wer_diffs, 97.5))]

def evaluate_taxonomy(df_sub, v2_col, tess_col):
    res = {}
    for m_name, col_name in [("v2", v2_col), ("tesseract", tess_col)]:
        tot_lead, tot_trail, tot_sp, tot_o0, tot_punct, tot_let = 0, 0, 0, 0, 0, 0
        total_levenshtein = 0
        all_details = []
        for idx, r in df_sub.iterrows():
            ref = r["ref"]
            hyp = r[col_name]
            ce, _ = align(str(ref) if pd.notnull(ref) else "", str(hyp) if pd.notnull(hyp) else "")
            total_levenshtein += ce
            l_lead, l_trail, l_sp, l_o0, l_punct, l_let, d_list = categorize_line_errors(ref, hyp)
            tot_lead += l_lead
            tot_trail += l_trail
            tot_sp += l_sp
            tot_o0 += l_o0
            tot_punct += l_punct
            tot_let += l_let
            for d in d_list:
                d["line_idx"] = int(r["line_idx"])
                all_details.append(d)
        sum_cats = tot_lead + tot_trail + tot_sp + tot_o0 + tot_punct + tot_let
        res[m_name] = {
            "leading_edge_insertions": tot_lead,
            "trailing_edge_insertions": tot_trail,
            "internal_space_errors": tot_sp,
            "internal_o0_errors": tot_o0,
            "internal_punct_errors": tot_punct,
            "internal_letter_errors": tot_let,
            "sum_categories": sum_cats,
            "total_errors": total_levenshtein,
            "mutually_exclusive_check": (sum_cats == total_levenshtein),
            "details": all_details
        }
    return res

refs_all = df_details["ref"].tolist()

print("Calculating metrics for Condition A...")
m_cond_A = {
    "v1_full": compute_metrics(df_details["v1_full_A"].tolist(), refs_all),
    "v2_full": compute_metrics(df_details["v2_full_A"].tolist(), refs_all),
    "tess_full_psm7": compute_metrics(df_details["tess_full_A_psm7"].tolist(), refs_all),
    "tess_full_psm13": compute_metrics(df_details["tess_full_A_psm13"].tolist(), refs_all),
    "v1_rule_chunk": compute_metrics(df_details["v1_rule_joined_A"].tolist(), refs_all),
    "v2_rule_chunk": compute_metrics(df_details["v2_rule_joined_A"].tolist(), refs_all),
    "tess_rule_chunk_psm7": compute_metrics(df_details["tess_rule_joined_A"].tolist(), refs_all),
}
ci_cer_A, ci_wer_A = bootstrap_paired_diff_fast(df_details["v2_rule_joined_A"].tolist(), df_details["tess_rule_joined_A"].tolist(), refs_all)
m_cond_A["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_cond_A["v2_rule_chunk"]["cer"] - m_cond_A["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_cond_A["v2_rule_chunk"]["wer"] - m_cond_A["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_A,
    "wer_ci": ci_wer_A
}
tax_A = evaluate_taxonomy(df_details, "v2_rule_joined_A", "tess_rule_joined_A")

print("Calculating metrics for Condition B (padded)...")
m_cond_B = {
    "v1_full": compute_metrics(df_details["v1_full_B"].tolist(), refs_all),
    "v2_full": compute_metrics(df_details["v2_full_B"].tolist(), refs_all),
    "tess_full_psm7": compute_metrics(df_details["tess_full_B_psm7"].tolist(), refs_all),
    "tess_full_psm13": compute_metrics(df_details["tess_full_B_psm13"].tolist(), refs_all),
    "v1_rule_chunk": compute_metrics(df_details["v1_rule_joined_B"].tolist(), refs_all),
    "v2_rule_chunk": compute_metrics(df_details["v2_rule_joined_B"].tolist(), refs_all),
    "tess_rule_chunk_psm7": compute_metrics(df_details["tess_rule_joined_B"].tolist(), refs_all),
}
ci_cer_B, ci_wer_B = bootstrap_paired_diff_fast(df_details["v2_rule_joined_B"].tolist(), df_details["tess_rule_joined_B"].tolist(), refs_all)
m_cond_B["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_cond_B["v2_rule_chunk"]["cer"] - m_cond_B["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_cond_B["v2_rule_chunk"]["wer"] - m_cond_B["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_B,
    "wer_ci": ci_wer_B
}
tax_B = evaluate_taxonomy(df_details, "v2_rule_joined_B", "tess_rule_joined_B")

print("Calculating clean subset...")
lines_with_empty_A = set([item["line_idx"] for item in empty_outputs_log if "A" in item["condition"]])
clean_idxs_A = [i for i in range(33) if i not in lines_with_empty_A]
df_clean_A = df_details[df_details["line_idx"].isin(clean_idxs_A)].copy()
refs_clean = df_clean_A["ref"].tolist()

print(f"Clean lines in Condition A: {len(clean_idxs_A)}/33 lines (Excluded {len(lines_with_empty_A)} lines: {sorted(list(lines_with_empty_A))})")

m_clean = {
    "n_lines": len(clean_idxs_A),
    "clean_line_indices": clean_idxs_A,
    "excluded_line_indices": sorted(list(lines_with_empty_A)),
    "v1_full": compute_metrics(df_clean_A["v1_full_A"].tolist(), refs_clean),
    "v2_full": compute_metrics(df_clean_A["v2_full_A"].tolist(), refs_clean),
    "tess_full_psm7": compute_metrics(df_clean_A["tess_full_A_psm7"].tolist(), refs_clean),
    "tess_full_psm13": compute_metrics(df_clean_A["tess_full_A_psm13"].tolist(), refs_clean),
    "v1_rule_chunk": compute_metrics(df_clean_A["v1_rule_joined_A"].tolist(), refs_clean),
    "v2_rule_chunk": compute_metrics(df_clean_A["v2_rule_joined_A"].tolist(), refs_clean),
    "tess_rule_chunk_psm7": compute_metrics(df_clean_A["tess_rule_joined_A"].tolist(), refs_clean),
}
ci_cer_clean, ci_wer_clean = bootstrap_paired_diff_fast(df_clean_A["v2_rule_joined_A"].tolist(), df_clean_A["tess_rule_joined_A"].tolist(), refs_clean)
m_clean["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_clean["v2_rule_chunk"]["cer"] - m_clean["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_clean["v2_rule_chunk"]["wer"] - m_clean["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_clean,
    "wer_ci": ci_wer_clean
}
tax_clean = evaluate_taxonomy(df_clean_A, "v2_rule_joined_A", "tess_rule_joined_A")

print("Grouping V2 CER by ratio group...")
ratio_groups = ["<1.0", "1.0-1.1", "1.1-1.3", ">1.3"]
grouped_results = {}
for g in ratio_groups:
    sub = df_details[df_details["ratio_group"] == g]
    sub_refs = sub["ref"].tolist()
    m_full = compute_metrics(sub["v2_full_A"].tolist(), sub_refs)
    m_chunk = compute_metrics(sub["v2_rule_joined_A"].tolist(), sub_refs)
    grouped_results[g] = {
        "n_lines": len(sub),
        "lines": sub["line_idx"].tolist(),
        "v2_full_cer": m_full["cer"],
        "v2_full_wer": m_full["wer"],
        "v2_full_exact": f"{m_full['exact']}/{len(sub)}",
        "v2_chunk_cer": m_chunk["cer"],
        "v2_chunk_wer": m_chunk["wer"],
        "v2_chunk_exact": f"{m_chunk['exact']}/{len(sub)}",
        "cer_reduction_pp": m_full["cer"] - m_chunk["cer"]
    }

final_metrics = {
    "condition_A_all33": m_cond_A,
    "condition_B_padded_all33": m_cond_B,
    "clean_subset_A": m_clean,
    "v2_ratio_grouped_CER": grouped_results,
    "empty_outputs_count": {
        "condition_A": len([x for x in empty_outputs_log if "A" in x["condition"]]),
        "condition_B": len([x for x in empty_outputs_log if "B" in x["condition"]]),
        "by_model": {
            "v1": len([x for x in empty_outputs_log if x["model"] == "v1"]),
            "v2": len([x for x in empty_outputs_log if x["model"] == "v2"]),
            "tesseract": len([x for x in empty_outputs_log if x["model"] == "tesseract"]),
        }
    }
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "error_taxonomy.json", "w", encoding="utf-8") as f:
    json.dump({
        "condition_A": tax_A,
        "condition_B": tax_B,
        "clean_subset": tax_clean
    }, f, indent=2, ensure_ascii=False)

print("Metrics and taxonomy saved successfully.")

# Contact sheet
target_line_idxs = [17, 18, 20, 21]
target_rows = df_details[df_details["line_idx"].isin(target_line_idxs)].to_dict("records")

card_w = 1400
card_h = 320
cs_img = np.full((len(target_rows) * card_h + 80, card_w, 3), 245, dtype=np.uint8)

cv2.putText(cs_img, "Photo 5: Second Chunks for Lines 18, 19, 21, 22 (Condition A vs B)", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)

y_offset = 80
for r in target_rows:
    idx = r["line_idx"]
    line_num = idx + 1
    ch_A = cv2.imread(str(OUT_DIR / f"chunks/line_{idx:02d}_chunk_01_trimmed.png"))
    ch_B = cv2.imread(str(OUT_DIR / f"chunks_padded/line_{idx:02d}_chunk_01_padded.png"))
    
    cv2.rectangle(cs_img, (20, y_offset), (card_w - 20, y_offset + card_h - 20), (255, 255, 255), -1)
    cv2.rectangle(cs_img, (20, y_offset), (card_w - 20, y_offset + card_h - 20), (200, 200, 200), 1)
    
    ref_words = str(r['ref']).split()
    snippet = " ".join(ref_words[-4:]) if len(ref_words) >= 4 else str(r['ref'])
    header = f"Line {line_num:02d} (idx {idx:02d}) | Ref Chunk Snippet: \"{snippet}\" | Ratio: {r['ratio_w_h']:.3f} ({r['ratio_group']})"
    cv2.putText(cs_img, header, (40, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 150), 2, cv2.LINE_AA)
    
    h_a, w_a = ch_A.shape[:2]
    disp_h_a = 75
    disp_w_a = int(w_a * (disp_h_a / h_a))
    if disp_w_a > 500:
        disp_w_a = 500
        disp_h_a = int(h_a * (disp_w_a / w_a))
    res_A = cv2.resize(ch_A, (disp_w_a, disp_h_a))
    cs_img[y_offset + 60 : y_offset + 60 + disp_h_a, 40 : 40 + disp_w_a] = res_A
    cv2.rectangle(cs_img, (40, y_offset + 60), (40 + disp_w_a, y_offset + 60 + disp_h_a), (180, 0, 0), 1)
    cv2.putText(cs_img, f"Cond A (Trim+4px): {w_a}x{h_a}px", (40, y_offset + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    
    h_b, w_b = ch_B.shape[:2]
    disp_h_b = 75
    disp_w_b = int(w_b * (disp_h_b / h_b))
    if disp_w_b > 500:
        disp_w_b = 500
        disp_h_b = int(h_b * (disp_w_b / w_b))
    res_B = cv2.resize(ch_B, (disp_w_b, disp_h_b))
    cs_img[y_offset + 60 : y_offset + 60 + disp_h_b, 600 : 600 + disp_w_b] = res_B
    cv2.rectangle(cs_img, (600, y_offset + 60), (600 + disp_w_b, y_offset + 60 + disp_h_b), (0, 150, 0), 1)
    cv2.putText(cs_img, f"Cond B (Padded 20px): {w_b}x{h_b}px", (600, y_offset + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    
    t_A_7 = str(r["tess_raw_chunks_A_psm7"]).split(" | ")[-1] if " | " in str(r["tess_raw_chunks_A_psm7"]) else str(r["tess_raw_chunks_A_psm7"])
    if t_A_7 == "nan" or not t_A_7.strip():
        t_A_7 = ""
    t_A_13 = str(r["tess_raw_chunks_A_psm13"]).split(" | ")[-1] if " | " in str(r["tess_raw_chunks_A_psm13"]) else str(r["tess_raw_chunks_A_psm13"])
    v2_A = str(r["v2_raw_chunks_A"]).split(" | ")[-1] if " | " in str(r["v2_raw_chunks_A"]) else str(r["v2_raw_chunks_A"])
    
    txt_tess = f"Tesseract psm 7: \"{t_A_7}\" (EMPTY!) -> psm 13 fallback: \"{t_A_13[:35]}\""
    txt_v2 = f"V2 Line Output: \"{v2_A[:50]}\""
    cv2.putText(cs_img, txt_tess, (40, y_offset + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(cs_img, txt_v2, (40, y_offset + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 0), 2, cv2.LINE_AA)
    
    y_offset += card_h

cs_path = CONTACT_DIR / "contact_sheet_lines18_19_21_22.png"
cv2.imwrite(str(cs_path), cs_img)

brain_dir = Path("C:/Users/serxan.m/.gemini/antigravity-ide/brain/1ece9f72-797d-4105-b45d-b26d0dfbe9ac")
if brain_dir.exists():
    cv2.imwrite(str(brain_dir / "contact_sheet_p5_empty_chunks.png"), cs_img)

print(f"Contact sheet saved to {cs_path}")
print("ALL FINISHED SUCCESSFULLY!")
