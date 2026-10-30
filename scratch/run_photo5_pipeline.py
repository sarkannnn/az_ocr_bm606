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

OUT_DIR = Path("results/photo5_heic_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR = OUT_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = OUT_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Ground truth reference text (33 lines)
REFS = [
    "Zamanın süzgəcindən keçmiş qədim sənət növləri hər bir xalqın tarixi",
    "yaddaşını, ruhunu və mədəniyyətini özündə yaşadır. Azərbaycanda xalçaçılıq,",
    "dulusçuluq, şəbəkə və misgərlik kimi sənət sahələri nəsildən-nəslə, ustadan",
    "şəyirdə ötürülərək bu günümüzə qədər gəlib çatmışdır. Hər bir naxış, hər bir",
    "ilmə bir hekayə danışır, usta əllərinin zəhmətini, alın tərini və qəlbinin",
    "hərarətini əks etdirir. Bu sənətlər təkcə keçmişin mirası deyil, həm də gələcəyə",
    "uzanan mənəvi bir körpüdür.",
    "Məsələn, dulusçuluq sənətini götürək: torpaqla suyun, insan əlinin odu və",
    "istedadı ilə birləşib sənət əsərinə çevrilməsi əsl möcüzədir. Gəncə çayının",
    "kənarında, qədim ağacların kölgəsinə sığınmış balaca bir emalatxanada",
    "ömrünü bu sənətə həsr etmiş qocaman usta Səməndər yaşayırdı. Onun palçığa",
    "toxunan qabarlı əlləri sanki zamanı dayandırırdı. Divarlardan asılmış gil",
    "kasalar, bəzəkli küplər və naxışlı boşqablar otağa xüsusi bir torpaq qoxusu və",
    "qədimlik gətirirdi. Səməndər usta hər bir gil qaba sadəcə bir əşya kimi deyil,",
    "sanki canı-qanı olan bir varlıq kimi yanaşırdı.",
    "Günlərin bir günü onun şəhərdə təhsil almış, gənc nəvəsi Taryel babasının",
    "yanına gəldi. Taryel babasının yorğun, lakin qətiyyətli baxışlarını görüb ona",
    'yaxınlaşdı: "Baba, yenə saatlarla bu təkərin arxasında oturmusan. Özünü',
    'yorursan, gəl bu gün biraz dincəl," — dedi. Qoca Səməndər başını qaldırıb',
    "nəvəsinə baxdı, dodaqlarında yüngül, lakin dərin mənalı bir təbəssüm",
    'yarandı. Əlindəki gili yavaşca masaya qoyub dedi: "Ay bala, insan o vaxt',
    "qocalır ki, torpağa olan inamı ölür. Bu gil sadəcə torpaq parçası deyil. Hər",
    "küpün, hər qabın içində bizim tariximiz, bu torpağın göz yaşları və sevincləri",
    "gizlənib. Biz getdikdən sonra bu əllər olmayacaq, amma bu küplər bizim",
    'yerimizə danışacaq. Əgər biz bu sənəti unutsaq, özümüzü də unudacağıq."',
    "Bu sözlər Taryelin qəlbində dərin bir iz buraxdı. O, babasının ömrünü həsr",
    "etdiyi bu sənətin nə qədər böyük bir dəyər olduğunu, tariximizin və milli",
    "kimliyimizin bu gil qablarda necə yaşatıldığını dərindən hiss etdi. Taryel",
    "qərara gəldi ki, şəhərə geri qayıtmağa tələsməyəcək, babasının yanında qalıb",
    "bu ağır, lakin şərəfli yükü daşımağa kömək edəcək. Beləcə, gənc nəsil qədim",
    "irsi qorumaq üçün böyük bir addım atdı.",
    "Müasir dünyada texnologiyanın sürətli inkişafına baxmayaraq, əl əməyinin,",
    "milli sənətkarlığın və ulu babalarımızdan qalan bu qiymətli mirasın dəyəri"
]

img_path = Path('data/real_test5/photo5.jpg')
img = cv2.imread(str(img_path))
h_img, w_img = img.shape[:2]

# Detect line boxes with Tesseract
data = pytesseract.image_to_data(Image.open(img_path), lang='aze', output_type=pytesseract.Output.DATAFRAME)
data = data[data.text.notnull() & (data.text.str.strip() != '')]

lines = []
for (b, p, l), grp in data.groupby(['block_num', 'par_num', 'line_num']):
    text = ' '.join(grp.text.astype(str))
    x0, y0 = int(grp.left.min()), int(grp.top.min())
    x1, y1 = int((grp.left + grp.width).max()), int((grp.top + grp.height).max())
    lines.append({'text': text, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'w': x1 - x0, 'h': y1 - y0})

assert len(lines) == 33, f"Expected 33 lines, got {len(lines)}"

# Common left bound is around x=420, but let's use each line's detected box with 8 px margin
line_crops = []
line_meta = []
for idx, l in enumerate(lines):
    # Padding: vertical 4 px, horizontal 12 px
    y0 = max(0, l['y0'] - 4)
    y1 = min(h_img, l['y1'] + 4)
    # In line 12, start x at 430 to bypass printer speck at x=395
    line_x0 = 430 if idx == 11 else max(0, l['x0'] - 12)
    x1 = min(w_img, l['x1'] + 12)
    
    crop = img[y0:y1, line_x0:x1]
    cv2.imwrite(str(CROPS_DIR / f"line_{idx:02d}.png"), crop)
    line_crops.append(crop)
    
    line_meta.append({
        "line_idx": idx, "h": y1 - y0, "w": x1 - line_x0,
        "y0": y0, "y1": y1, "x0": line_x0, "x1": x1,
        "ref": REFS[idx]
    })

with open(OUT_DIR / "crop_geometry.json", "w", encoding="utf-8") as f:
    json.dump(line_meta, f, indent=2, ensure_ascii=False)

# Robust Chunking & Trimming Functions (exact real_fix8 pipeline)
def find_split_points(crop_bgr, n_chunks):
    if n_chunks <= 1:
        return []
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    w = crop_bgr.shape[1]
    ink = (gray < np.percentile(gray, 35)).astype(np.float32)
    proj = ink.sum(axis=0)
    ksize = 25
    smooth_proj = np.convolve(proj, np.ones(ksize)/ksize, mode='same')
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

def robust_trim_chunk(chunk_bgr):
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY)
    h_chunk, w_chunk = gray.shape
    ink = (gray < 160).astype(np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    clean_ink = np.zeros_like(ink)
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_left_or_right = (x == 0) or (x + w >= w_chunk)
        is_tall_shadow = touches_left_or_right and (h >= 0.70 * h_chunk)
        is_small_speck = (area < 8)
        
        if not (is_small_speck or is_tall_shadow):
            clean_ink[labels == i] = 1
            
    col_counts = clean_ink.sum(axis=0)
    valid_cols = np.where(col_counts > 0)[0]
    if len(valid_cols) == 0:
        return chunk_bgr
    x0 = max(0, int(valid_cols[0]) - 4)
    x1 = min(w_chunk, int(valid_cols[-1]) + 5)
    return chunk_bgr[:, x0:x1]

def run_tess_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    txt = pytesseract.image_to_string(Image.fromarray(rgb), lang='aze', config='--psm 7').strip()
    return normalize(" ".join(txt.split()))

# Load OCR models
print("Loading OCR models...")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")

print("Running evaluation on all 33 lines...")
detailed_rows = []

for idx in range(33):
    c = line_crops[idx]
    ref = normalize(REFS[idx])
    h_c, w_c = c.shape[:2]
    val_formula = (w_c * 32) / (h_c * 512)
    rule_n = max(1, math.ceil(val_formula))
    
    raw_chunks = split_into_chunks(c, rule_n)
    chunk_paths = []
    v1_chunk_outs = []
    v2_chunk_outs = []
    tess_chunk_outs = []
    
    for ch_i, ch in enumerate(raw_chunks):
        ch_trimmed = robust_trim_chunk(ch)
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
    
    join_str = " "
    v1_joined = normalize(join_str.join([p for p in v1_chunk_outs if p]))
    v2_joined = normalize(join_str.join([p for p in v2_chunk_outs if p]))
    tess_joined = normalize(join_str.join([p for p in tess_chunk_outs if p]))
    
    detailed_rows.append({
        "line_idx": idx,
        "ref": ref,
        "crop_h": h_c,
        "crop_w": w_c,
        "rule_n_chunks": rule_n,
        "chunk_paths": "; ".join(chunk_paths),
        "v1_raw_chunks": " | ".join(v1_chunk_outs),
        "v2_raw_chunks": " | ".join(v2_chunk_outs),
        "tess_raw_chunks": " | ".join(tess_chunk_outs),
        "v1_rule_joined": v1_joined,
        "v2_rule_joined": v2_joined,
        "tess_rule_joined": tess_joined,
        "v1_full": normalize(v1_full),
        "v2_full": normalize(v2_full),
        "tess_full": normalize(tess_full),
    })

df_details = pd.DataFrame(detailed_rows)
df_details.to_csv(OUT_DIR / "per_line_chunk_details.csv", index=False, encoding="utf-8")

# Metrics computation
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
    tot_sp = 0
    
    for p, r in zip(preds, refs):
        ce, cl, we, wl, sp, _ = get_line_edits(r, p)
        tot_cer_errs += ce
        tot_wer_errs += we
        tot_chars += cl
        tot_words += wl
        tot_sp += sp
        if p == r:
            exact += 1
            
    cer = (tot_cer_errs / tot_chars * 100.0) if tot_chars > 0 else 0.0
    wer = (tot_wer_errs / tot_words * 100.0) if tot_words > 0 else 0.0
    return {
        "cer": cer, "wer": wer, "exact": exact, "total": len(refs),
        "cer_errs": tot_cer_errs, "chars": tot_chars,
        "wer_errs": tot_wer_errs, "words": tot_words,
        "space_errors_char": tot_sp
    }

refs_list = [r["ref"] for r in detailed_rows]
results = {}

for col_name, key in [
    ("v1_full", "v1_full"),
    ("v2_full", "v2_full"),
    ("tess_full", "tess_full"),
    ("v1_rule_joined", "v1_rule_chunk"),
    ("v2_rule_joined", "v2_rule_chunk"),
    ("tess_rule_joined", "tess_rule_chunk")
]:
    preds = [r[col_name] for r in detailed_rows]
    results[key] = compute_metrics(preds, refs_list)

# Bootstrap 95% CI (V2 - Tess)
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

p_v2_ch = [r["v2_rule_joined"] for r in detailed_rows]
p_tess_ch = [r["tess_rule_joined"] for r in detailed_rows]
ci_cer, ci_wer = bootstrap_paired_diff(p_v2_ch, p_tess_ch, refs_list)

results["bootstrap_ci_rule_chunk"] = {
    "delta_cer": results["v2_rule_chunk"]["cer"] - results["tess_rule_chunk"]["cer"],
    "delta_wer": results["v2_rule_chunk"]["wer"] - results["tess_rule_chunk"]["wer"],
    "cer_ci": ci_cer,
    "wer_ci": ci_wer
}

with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Error taxonomy for V2 and Tesseract
def categorize_line_errors(ref, hyp):
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
        if first_match_idx is not None and idx < first_match_idx and op == "ins":
            leading_ins += 1
            details.append({"category": "Leading Edge Insertion", "op": op, "ref": rc, "hyp": hc})
        elif last_match_idx is not None and idx > last_match_idx and op == "ins":
            trailing_ins += 1
            details.append({"category": "Trailing Edge Insertion", "op": op, "ref": rc, "hyp": hc})
        elif rc == ' ' or hc == ' ':
            internal_space += 1
            details.append({"category": "Internal Spacing Error", "op": op, "ref": rc, "hyp": hc})
        elif (rc, hc) in [('0', 'O'), ('O', '0'), ('o', '0'), ('0', 'o')]:
            internal_o0 += 1
            details.append({"category": "Internal 0/O Error", "op": op, "ref": rc, "hyp": hc})
        elif rc in '.,:;!?—-"\'()[]{}' or hc in '.,:;!?—-"\'()[]{}':
            internal_punct += 1
            details.append({"category": "Internal Punctuation Error", "op": op, "ref": rc, "hyp": hc})
        else:
            internal_letter += 1
            details.append({"category": "Internal Letter Error", "op": op, "ref": rc, "hyp": hc})
            
    return leading_ins, trailing_ins, internal_space, internal_o0, internal_punct, internal_letter, details

tax_summary = {}
for m_name, col_name in [("v2", "v2_rule_joined"), ("tesseract", "tess_rule_joined")]:
    tot_lead, tot_trail, tot_sp, tot_o0, tot_punct, tot_let = 0, 0, 0, 0, 0, 0
    total_levenshtein = 0
    all_details = []
    
    for idx, r in df_details.iterrows():
        ref = r["ref"]
        hyp = r[col_name]
        ce, _, _, _, _, _ = get_line_edits(ref, hyp)
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
    tax_summary[m_name] = {
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

with open(OUT_DIR / "error_taxonomy.json", "w", encoding="utf-8") as f:
    json.dump(tax_summary, f, indent=2, ensure_ascii=False)

print("Photo 5 evaluation completed successfully!")
print("V2 Rule Chunk CER:", results["v2_rule_chunk"]["cer"], "%, Exact:", results["v2_rule_chunk"]["exact"], "/ 33")
print("Tess Rule Chunk CER:", results["tess_rule_chunk"]["cer"], "%, Exact:", results["tess_rule_chunk"]["exact"], "/ 33")
print("V1 Rule Chunk CER:", results["v1_rule_chunk"]["cer"], "%, Exact:", results["v1_rule_chunk"]["exact"], "/ 33")
