import sys
import os
import math
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

sys.path.insert(0, str(Path.cwd()))
from common import load_model_checkpoint, infer_single_crop, normalize, cer_wer, align

OUT_DIR = Path("results/photo3_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = OUT_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = OUT_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth reference texts as printed on the physical document:
# Note: includes the exact printed misspellings from the printed paper:
# L09: "kimliyinir", L12: "tarixirə", L18: "hsduğu", L21: "ənənələnin"
REFS_PRINTED = [
    "Zamanın süzgəcindən keçmiş qədim sənət növləri",
    "hər bir xalqın tarixi yaddaşını və mədəniyyətini",
    "özündə yaşadır. Azərbaycanda xalçaçılıq,",
    "dulusçuluq, şəbəkə və misgərlik kimi sənət",
    "sahələri nəsildən-nəslə ötürülərək bu günümüzə",
    "qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə",
    "bir hekayə danışır, usta əllərinin zəhmətini və",
    "ruhunu əks etdirir.",
    "Bu sənət inciləri təkcə gözəllik deyil, həm də",
    "xalqımızın milli kimliyinir göstəricisidir. Onları",
    "qorumaq, yaşatmaq və gələcək nəsillərə ötürmək",
    "hər birimizin mənəvi borcudur. Çünki tarixinə",
    "tarixirə sahib çıxan millətlər öz gələcəyini daha",
    "möhkəm təməllər üzərində qura bilirlər. Hər bir",
    "qədim əşya keçmişimizlə bu günümüz arasında",
    "görünməz bir körpüdür.",
    "Müasir dünyada texnologiyanın sürətli inkişafına",
    "baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri",
    "heç vaxt azalmır. Əksinə, insan ruhunun hsduğu bu",
    "sənət nümunələrinə olan maraq günü-gündən artır.",
    "Gənc nəslin bu sənətlərə maraq göstərməsi, qədim",
    "ənənələnin müasir dizaynlarla birləşdirilməsi bu",
    "irsin gələcəyini daha da parlaq edir."
]

# Standard corrected version for comparison
REFS_CORRECTED = [
    "Zamanın süzgəcindən keçmiş qədim sənət növləri",
    "hər bir xalqın tarixi yaddaşını və mədəniyyətini",
    "özündə yaşadır. Azərbaycanda xalçaçılıq,",
    "dulusçuluq, şəbəkə və misgərlik kimi sənət",
    "sahələri nəsildən-nəslə ötürülərək bu günümüzə",
    "qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə",
    "bir hekayə danışır, usta əllərinin zəhmətini və",
    "ruhunu əks etdirir.",
    "Bu sənət inciləri təkcə gözəllik deyil, həm də",
    "xalqımızın milli kimliyinin göstəricisidir. Onları",
    "qorumaq, yaşatmaq və gələcək nəsillərə ötürmək",
    "hər birimizin mənəvi borcudur. Çünki tarixinə",
    "tarixinə sahib çıxan millətlər öz gələcəyini daha",
    "möhkəm təməllər üzərində qura bilirlər. Hər bir",
    "qədim əşya keçmişimizlə bu günümüz arasında",
    "görünməz bir körpüdür.",
    "Müasir dünyada texnologiyanın sürətli inkişafına",
    "baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri",
    "heç vaxt azalmır. Əksinə, insan ruhunun duyduğu bu",
    "sənət nümunələrinə olan maraq günü-gündən artır.",
    "Gənc nəslin bu sənətlərə maraq göstərməsi, qədim",
    "ənənələrin müasir dizaynlarla birləşdirilməsi bu",
    "irsin gələcəyini daha da parlaq edir."
]

img_path = r'C:\Users\serxan.m\.gemini\antigravity-ide\brain\1ece9f72-797d-4105-b45d-b26d0dfbe9ac\.user_uploaded\media_1793351399184.jpg'
img = cv2.imread(img_path)
cv2.imwrite(str(OUT_DIR / "original_photo.jpg"), img)

# Edge ink calculation standard
def get_edge_ink_standard(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thresh = (gray < 160)
    top2_px = int(thresh[:2, :].sum())
    btm2_px = int(thresh[-2:, :].sum())
    top2_f = float(top2_px) / (2 * w) if w > 0 else 0.0
    btm2_f = float(btm2_px) / (2 * w) if w > 0 else 0.0
    return top2_f, btm2_f, top2_px, btm2_px

# Tesseract bounding boxes to detect lines
data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0})

assert len(lines) == 23, f"Expected 23 lines, got {len(lines)}"

# Common x-bounds per paragraph or global to prevent any side clipping
# Left margin is around x=65-110, right margin around x=620-660
# For full lines, use x0 = line.x0 - 8, x1 = line.x1 + 8 (or min/max per paragraph)
line_crops = []
line_meta = []
for idx, l in enumerate(lines):
    # Padding: vertical 3-4 px to ensure descenders and ascenders are intact
    y0 = max(0, l['y0'] - 4)
    y1 = min(img.shape[0], l['y1'] + 4)
    x0 = max(0, l['x0'] - 8)
    x1 = min(img.shape[1], l['x1'] + 8)
    
    crop = img[y0:y1, x0:x1]
    c_path = CROPS_DIR / f"line_{idx:02d}.png"
    cv2.imwrite(str(c_path), crop)
    line_crops.append(crop)
    
    top_f, btm_f, top_px, btm_px = get_edge_ink_standard(crop)
    h_c, w_c = crop.shape[:2]
    line_meta.append({
        "line_idx": idx, "h": h_c, "w": w_c, "y0": y0, "y1": y1,
        "x0": x0, "x1": x1,
        "top2_frac": top_f, "btm2_frac": btm_f,
        "top2_pixels": top_px, "btm2_pixels": btm_px,
        "ref": REFS_PRINTED[idx]
    })

with open(OUT_DIR / "crop_geometry.json", "w", encoding="utf-8") as f:
    json.dump(line_meta, f, indent=2, ensure_ascii=False)

# Frozen chunking utilities
def find_split_points(crop_bgr, n_chunks):
    if n_chunks <= 1:
        return []
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 160).astype(np.float32)
    proj = ink.sum(axis=0)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    smooth_proj = np.convolve(proj, kernel, mode='same')
    
    w = crop_bgr.shape[1]
    nominal_w = w / n_chunks
    search_radius = int(nominal_w * 0.35)
    splits = []
    for k in range(1, n_chunks):
        ideal_x = int(k * nominal_w)
        x0 = max(15, ideal_x - search_radius)
        x1 = min(w - 15, ideal_x + search_radius)
        window = smooth_proj[x0:x1]
        min_val = window.min()
        candidates = np.where(window == min_val)[0] + x0
        best_x = candidates[np.argmin(np.abs(candidates - ideal_x))]
        splits.append(int(best_x))
    return splits

def split_into_chunks(crop_bgr, n_chunks):
    splits = find_split_points(crop_bgr, n_chunks)
    chunks = []
    prev_x = 0
    for sx in splits:
        chunks.append(crop_bgr[:, prev_x:sx])
        prev_x = sx
    chunks.append(crop_bgr[:, prev_x:])
    return chunks

def trim_chunk_to_ink(chunk_bgr):
    """Horizontal trim of chunk to ink bounds + 4 px margin."""
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY)
    thresh = (gray < 160)
    col_counts = thresh.sum(axis=0)
    valid_cols = np.where(col_counts >= 1)[0]
    if len(valid_cols) == 0:
        return chunk_bgr
    x0 = max(0, int(valid_cols[0]) - 4)
    x1 = min(chunk_bgr.shape[1], int(valid_cols[-1]) + 5)
    return chunk_bgr[:, x0:x1]

def run_tess_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    txt = pytesseract.image_to_string(Image.fromarray(rgb), lang='aze', config='--psm 7').strip()
    return normalize(" ".join(txt.split()))

# Load Models
print("Loading V1 and V2 models...")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")

print("Running evaluation on 23 lines...")
detailed_chunk_rows = []

for idx in range(23):
    c = line_crops[idx]
    ref = normalize(REFS_PRINTED[idx])
    h_c, w_c = c.shape[:2]
    val_formula = (w_c * 32) / (h_c * 512)
    rule_n = max(1, math.ceil(val_formula))
    
    raw_chunks = split_into_chunks(c, rule_n)
    chunk_paths = []
    v1_chunk_outs = []
    v2_chunk_outs = []
    tess_chunk_outs = []
    
    for ch_i, ch in enumerate(raw_chunks):
        ch_trimmed = trim_chunk_to_ink(ch)
        ch_fname = f"line_{idx:02d}_chunk_{ch_i:02d}_trimmed.png"
        ch_p = CHUNKS_DIR / ch_fname
        cv2.imwrite(str(ch_p), ch_trimmed)
        chunk_paths.append(str(ch_p.relative_to(OUT_DIR)))
        
        p1_ch = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, ch_trimmed, dev).strip()
        p2_ch = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, ch_trimmed, dev).strip()
        pt_ch = run_tess_crop(ch_trimmed)
        
        v1_chunk_outs.append(p1_ch)
        v2_chunk_outs.append(p2_ch)
        tess_chunk_outs.append(pt_ch)
        
    v1_full = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, c, dev).strip()
    v2_full = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, c, dev).strip()
    tess_full = run_tess_crop(c)
    
    detailed_chunk_rows.append({
        "line_idx": idx,
        "ref": ref,
        "crop_h": h_c,
        "crop_w": w_c,
        "formula_val": val_formula,
        "rule_n_chunks": rule_n,
        "chunk_paths": "; ".join(chunk_paths),
        "v1_raw_chunks": " | ".join(v1_chunk_outs),
        "v2_raw_chunks": " | ".join(v2_chunk_outs),
        "tess_raw_chunks": " | ".join(tess_chunk_outs),
        "v1_joined": normalize(" ".join(v1_chunk_outs)),
        "v2_joined": normalize(" ".join(v2_chunk_outs)),
        "tess_joined": normalize(" ".join(tess_chunk_outs)),
        "v1_full": v1_full,
        "v2_full": v2_full,
        "tess_full": tess_full,
    })

# Point 2: Metrics computation
def get_line_edits(ref, hyp):
    ce, ops = align(ref, hyp)
    we, _ = align(ref.split(), hyp.split())
    sp_char = sum(1 for op, rc, hc in ops if op != "ok" and (rc == ' ' or hc == ' '))
    return ce, len(ref), we, len(ref.split()), sp_char, ops

def compute_metrics(preds, refs):
    tot_cer_errs = 0
    tot_wer_errs = 0
    tot_chars = 0
    tot_words = 0
    exact = 0
    per_line_cers = []
    per_line_wers = []
    
    for p, r in zip(preds, refs):
        ce, cl, we, wl, _, _ = get_line_edits(r, p)
        tot_cer_errs += ce
        tot_wer_errs += we
        tot_chars += cl
        tot_words += wl
        per_line_cers.append(ce / cl if cl > 0 else 0.0)
        per_line_wers.append(we / wl if wl > 0 else 0.0)
        if p == r:
            exact += 1
            
    cer = (tot_cer_errs / tot_chars * 100.0) if tot_chars > 0 else 0.0
    wer = (tot_wer_errs / tot_words * 100.0) if tot_words > 0 else 0.0
    return {
        "cer": cer, "wer": wer, "exact": exact, "total": len(refs),
        "cer_errs": tot_cer_errs, "chars": tot_chars,
        "wer_errs": tot_wer_errs, "words": tot_words,
        "per_line_cers": per_line_cers, "per_line_wers": per_line_wers
    }

refs_list = [r["ref"] for r in detailed_chunk_rows]
results = {}

# Evaluate V1, V2, Tess on Full Lines and Rule Chunks
for model_col, key in [
    ("v1_full", "v1_full"),
    ("v2_full", "v2_full"),
    ("tess_full", "tess_full"),
    ("v1_joined", "v1_rule_chunk"),
    ("v2_joined", "v2_rule_chunk"),
    ("tess_joined", "tess_rule_chunk")
]:
    preds = [r[model_col] for r in detailed_chunk_rows]
    m_raw = compute_metrics(preds, refs_list)
    results[key] = m_raw

# Bootstrap 95% CI for V2 vs Tesseract
def bootstrap_paired_diff(p1_list, p2_list, refs, n_boot=2000, seed=42):
    n = len(refs)
    diffs_cer = []
    diffs_wer = []
    c_v2, c_tess, c_lens = [], [], []
    w_v2, w_tess, w_lens = [], [], []
    for r, v2, t in zip(refs, p1_list, p2_list):
        ce2, cl, we2, wl, _, _ = get_line_edits(r, v2)
        cet, _, wet, _, _, _ = get_line_edits(r, t)
        c_v2.append(ce2)
        c_tess.append(cet)
        c_lens.append(cl)
        w_v2.append(we2)
        w_tess.append(wet)
        w_lens.append(wl)
    c_v2 = np.array(c_v2)
    c_tess = np.array(c_tess)
    c_lens = np.array(c_lens)
    w_v2 = np.array(w_v2)
    w_tess = np.array(w_tess)
    w_lens = np.array(w_lens)
    
    rng = np.random.default_rng(seed)
    s_idx = rng.choice(n, size=(n_boot, n), replace=True)
    v2_cer_boot = c_v2[s_idx].sum(axis=1) / c_lens[s_idx].sum(axis=1) * 100.0
    tess_cer_boot = c_tess[s_idx].sum(axis=1) / c_lens[s_idx].sum(axis=1) * 100.0
    cer_diffs = v2_cer_boot - tess_cer_boot
    v2_wer_boot = w_v2[s_idx].sum(axis=1) / w_lens[s_idx].sum(axis=1) * 100.0
    tess_wer_boot = w_tess[s_idx].sum(axis=1) / w_lens[s_idx].sum(axis=1) * 100.0
    wer_diffs = v2_wer_boot - tess_wer_boot
    return [float(np.percentile(cer_diffs, 2.5)), float(np.percentile(cer_diffs, 97.5))], [float(np.percentile(wer_diffs, 2.5)), float(np.percentile(wer_diffs, 97.5))]

p_v2_chunk = [r["v2_joined"] for r in detailed_chunk_rows]
p_tess_chunk = [r["tess_joined"] for r in detailed_chunk_rows]
p_v2_full = [r["v2_full"] for r in detailed_chunk_rows]
p_tess_full = [r["tess_full"] for r in detailed_chunk_rows]

ci_cer_chunk, ci_wer_chunk = bootstrap_paired_diff(p_v2_chunk, p_tess_chunk, refs_list)
ci_cer_full, ci_wer_full = bootstrap_paired_diff(p_v2_full, p_tess_full, refs_list)

results["bootstrap_ci"] = {
    "rule_chunk_v2_minus_tess": {"cer_ci": ci_cer_chunk, "wer_ci": ci_wer_chunk},
    "full_line_v2_minus_tess": {"cer_ci": ci_cer_full, "wer_ci": ci_wer_full}
}

# Error Taxonomy for V2
def categorize_v2_errors(pred, ref):
    ce, ops = align(ref, pred)
    if ce == 0:
        return []
    
    errors = []
    first_match_idx = None
    last_match_idx = None
    for idx_op, (op, r_c, p_c) in enumerate(ops):
        if op in ('ok', 'sub'):
            if first_match_idx is None:
                first_match_idx = idx_op
            last_match_idx = idx_op
            
    for idx_op, (op, r_c, p_c) in enumerate(ops):
        if op == 'ok':
            continue
        if first_match_idx is not None and idx_op < first_match_idx and op == 'ins':
            cat = "Leading Insertion"
        elif last_match_idx is not None and idx_op > last_match_idx and op == 'ins':
            cat = "Trailing Insertion"
        elif r_c == ' ' or p_c == ' ':
            cat = "Spacing"
        elif (r_c, p_c) in [('0', 'O'), ('O', '0'), ('o', '0'), ('0', 'o')]:
            cat = "0 / O Confusion"
        elif r_c in '.,:;!?—-"\'()[]{}' or p_c in '.,:;!?—-"\'()[]{}':
            cat = "Punctuation"
        else:
            cat = "Letters"
        errors.append({
            "op": op, "ref_char": r_c, "pred_char": p_c, "category": cat
        })
    return errors

taxonomy_full = []
for idx, r in enumerate(detailed_chunk_rows):
    errs = categorize_v2_errors(r["v2_full"], r["ref"])
    for e in errs:
        e["line_idx"] = idx
        taxonomy_full.append(e)

taxonomy_chunk = []
for idx, r in enumerate(detailed_chunk_rows):
    errs = categorize_v2_errors(r["v2_joined"], r["ref"])
    for e in errs:
        e["line_idx"] = idx
        taxonomy_chunk.append(e)

# Export CSV and JSON
df_details = pd.DataFrame(detailed_chunk_rows)
df_details.to_csv(OUT_DIR / "per_line_chunk_details.csv", index=False, encoding="utf-8")

with open(OUT_DIR / "photo3_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(OUT_DIR / "error_taxonomy.json", "w", encoding="utf-8") as f:
    json.dump({
        "full_line_v2": taxonomy_full,
        "rule_chunk_v2": taxonomy_chunk
    }, f, indent=2, ensure_ascii=False)

print("Evaluation completed successfully!")
print("Full Line V2:", results["v2_full"]["cer"], "% CER,", results["v2_full"]["wer"], "% WER, Exact:", results["v2_full"]["exact"], "/ 23")
print("Rule Chunk V2:", results["v2_rule_chunk"]["cer"], "% CER,", results["v2_rule_chunk"]["wer"], "% WER, Exact:", results["v2_rule_chunk"]["exact"], "/ 23")
print("Full Line Tess:", results["tess_full"]["cer"], "% CER,", results["tess_full"]["wer"], "% WER, Exact:", results["tess_full"]["exact"], "/ 23")
print("Rule Chunk Tess:", results["tess_rule_chunk"]["cer"], "% CER,", results["tess_rule_chunk"]["wer"], "% WER, Exact:", results["tess_rule_chunk"]["exact"], "/ 23")
print("Full Line V1:", results["v1_full"]["cer"], "% CER,", results["v1_full"]["wer"], "% WER, Exact:", results["v1_full"]["exact"], "/ 23")
print("Rule Chunk V1:", results["v1_rule_chunk"]["cer"], "% CER,", results["v1_rule_chunk"]["wer"], "% WER, Exact:", results["v1_rule_chunk"]["exact"], "/ 23")
