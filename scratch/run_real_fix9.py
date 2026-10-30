import sys
import os
import math
import json
import csv
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

sys.path.insert(0, str(Path.cwd()))
from common import load_model_checkpoint, infer_single_crop, normalize, cer_wer, align

OUT_DIR = Path("results/real_fix9")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_A_DIR = OUT_DIR / "crops"
CROPS_A_DIR.mkdir(parents=True, exist_ok=True)
CROPS_B_DIR = OUT_DIR / "crops_padded"
CROPS_B_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_A_DIR = OUT_DIR / "chunks"
CHUNKS_A_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_B_DIR = OUT_DIR / "chunks_padded"
CHUNKS_B_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_DIR = OUT_DIR / "contact_sheets"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

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

line_crops_A = []
line_crops_B = []
line_meta = []

for idx, l in enumerate(lines):
    y0 = max(0, l['y0'] - 4)
    y1 = min(h_img, l['y1'] + 4)
    line_x0 = 430 if idx == 11 else max(0, l['x0'] - 12)
    x1 = min(w_img, l['x1'] + 12)
    
    crop_A = img[y0:y1, line_x0:x1]
    cv2.imwrite(str(CROPS_A_DIR / f"line_{idx:02d}.png"), crop_A)
    line_crops_A.append(crop_A)
    
    # Condition B: Padded with 20 px white border
    crop_B = cv2.copyMakeBorder(crop_A, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imwrite(str(CROPS_B_DIR / f"line_{idx:02d}_padded.png"), crop_B)
    line_crops_B.append(crop_B)
    
    w_curr = x1 - line_x0
    h_curr = y1 - y0
    r_curr = (w_curr * 32.0) / (h_curr * 512.0)
    
    if r_curr < 1.0:
        r_group = "<1.0"
    elif r_curr <= 1.1:
        r_group = "1.0-1.1"
    elif r_curr <= 1.3:
        r_group = "1.1-1.3"
    else:
        r_group = ">1.3"
        
    line_meta.append({
        "line_idx": idx, "line_num": idx + 1, "h": h_curr, "w": w_curr,
        "y0": y0, "y1": y1, "x0": line_x0, "x1": x1,
        "ratio_w_h": r_curr, "ratio_group": r_group,
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
    if n_chunks <= 1:
        return [crop_bgr]
    splits = find_split_points(crop_bgr, n_chunks)
    all_bounds = [0] + splits + [crop_bgr.shape[1]]
    chunks = []
    for i in range(len(all_bounds) - 1):
        chunks.append(crop_bgr[:, all_bounds[i]:all_bounds[i+1]])
    return chunks

def robust_trim_chunk(chunk_bgr):
    gray = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2GRAY)
    h_chunk, w_chunk = gray.shape
    thresh_val = 160
    bin_img = (gray < thresh_val).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    clean_ink = np.zeros_like(bin_img)
    
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

def run_tesseract_eval(crop_bgr, psm=7):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    txt = pytesseract.image_to_string(Image.fromarray(rgb), lang='aze', config=f'--psm {psm}').strip()
    return normalize(" ".join(txt.split()))

# Load OCR models
print("Loading OCR models...")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")

print("Processing all 33 lines in Condition A and Condition B...")

empty_outputs_log = []

detailed_rows = []

for idx in range(33):
    c_A = line_crops_A[idx]
    c_B = line_crops_B[idx]
    ref = normalize(REFS[idx])
    h_c, w_c = c_A.shape[:2]
    val_formula = (w_c * 32) / (h_c * 512)
    rule_n = max(1, math.ceil(val_formula))
    
    r_meta = line_meta[idx]
    
    raw_chunks_A = split_into_chunks(c_A, rule_n)
    
    # Store per-condition chunks
    chunks_A_trimmed = []
    chunks_B_padded = []
    chunk_paths_A = []
    chunk_paths_B = []
    
    v1_outs_A, v2_outs_A, tess_outs_A_psm7, tess_outs_A_psm13 = [], [], [], []
    v1_outs_B, v2_outs_B, tess_outs_B_psm7, tess_outs_B_psm13 = [], [], [], []
    
    for ch_i, ch in enumerate(raw_chunks_A):
        ch_trimmed = robust_trim_chunk(ch)
        fname_A = f"line_{idx:02d}_chunk_{ch_i:02d}_trimmed.png"
        p_A = CHUNKS_A_DIR / fname_A
        cv2.imwrite(str(p_A), ch_trimmed)
        chunks_A_trimmed.append(ch_trimmed)
        chunk_paths_A.append(str(p_A.relative_to(OUT_DIR)))
        
        # Padded condition B
        ch_padded = cv2.copyMakeBorder(ch_trimmed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        fname_B = f"line_{idx:02d}_chunk_{ch_i:02d}_padded.png"
        p_B = CHUNKS_B_DIR / fname_B
        cv2.imwrite(str(p_B), ch_padded)
        chunks_B_padded.append(ch_padded)
        chunk_paths_B.append(str(p_B.relative_to(OUT_DIR)))
        
        # Condition A inference
        p1_ch_A = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, ch_trimmed, dev).strip()
        p2_ch_A = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, ch_trimmed, dev).strip()
        pt_ch_A_7 = run_tesseract_eval(ch_trimmed, psm=7)
        pt_ch_A_13 = run_tesseract_eval(ch_trimmed, psm=13)
        
        v1_outs_A.append(p1_ch_A)
        v2_outs_A.append(p2_ch_A)
        tess_outs_A_psm7.append(pt_ch_A_7)
        tess_outs_A_psm13.append(pt_ch_A_13)
        
        # Log empty outputs in Condition A
        if not p1_ch_A:
            empty_outputs_log.append({
                "model": "v1", "condition": "A (trim+4px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_A.relative_to(OUT_DIR)), "output": p1_ch_A, "fallback_psm13": None
            })
        if not p2_ch_A:
            empty_outputs_log.append({
                "model": "v2", "condition": "A (trim+4px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_A.relative_to(OUT_DIR)), "output": p2_ch_A, "fallback_psm13": None
            })
        if not pt_ch_A_7:
            empty_outputs_log.append({
                "model": "tesseract", "condition": "A (trim+4px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_A.relative_to(OUT_DIR)), "output": pt_ch_A_7, "fallback_psm13": pt_ch_A_13
            })
            
        # Condition B inference
        p1_ch_B = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, ch_padded, dev).strip()
        p2_ch_B = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, ch_padded, dev).strip()
        pt_ch_B_7 = run_tesseract_eval(ch_padded, psm=7)
        pt_ch_B_13 = run_tesseract_eval(ch_padded, psm=13)
        
        v1_outs_B.append(p1_ch_B)
        v2_outs_B.append(p2_ch_B)
        tess_outs_B_psm7.append(pt_ch_B_7)
        tess_outs_B_psm13.append(pt_ch_B_13)
        
        # Log empty outputs in Condition B
        if not p1_ch_B:
            empty_outputs_log.append({
                "model": "v1", "condition": "B (padded 20px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_B.relative_to(OUT_DIR)), "output": p1_ch_B, "fallback_psm13": None
            })
        if not p2_ch_B:
            empty_outputs_log.append({
                "model": "v2", "condition": "B (padded 20px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_B.relative_to(OUT_DIR)), "output": p2_ch_B, "fallback_psm13": None
            })
        if not pt_ch_B_7:
            empty_outputs_log.append({
                "model": "tesseract", "condition": "B (padded 20px)", "mode": "chunk", "line_idx": idx, "line_num": idx+1,
                "chunk_idx": ch_i, "path": str(p_B.relative_to(OUT_DIR)), "output": pt_ch_B_7, "fallback_psm13": pt_ch_B_13
            })

    # Full lines inference Condition A
    v1_full_A = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, c_A, dev).strip()
    v2_full_A = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, c_A, dev).strip()
    tess_full_A_7 = run_tesseract_eval(c_A, psm=7)
    tess_full_A_13 = run_tesseract_eval(c_A, psm=13)
    
    if not v1_full_A:
        empty_outputs_log.append({"model": "v1", "condition": "A (trim+4px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops/line_{idx:02d}.png", "output": v1_full_A, "fallback_psm13": None})
    if not v2_full_A:
        empty_outputs_log.append({"model": "v2", "condition": "A (trim+4px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops/line_{idx:02d}.png", "output": v2_full_A, "fallback_psm13": None})
    if not tess_full_A_7:
        empty_outputs_log.append({"model": "tesseract", "condition": "A (trim+4px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops/line_{idx:02d}.png", "output": tess_full_A_7, "fallback_psm13": tess_full_A_13})

    # Full lines inference Condition B
    v1_full_B = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, c_B, dev).strip()
    v2_full_B = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, c_B, dev).strip()
    tess_full_B_7 = run_tesseract_eval(c_B, psm=7)
    tess_full_B_13 = run_tesseract_eval(c_B, psm=13)
    
    if not v1_full_B:
        empty_outputs_log.append({"model": "v1", "condition": "B (padded 20px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops_padded/line_{idx:02d}_padded.png", "output": v1_full_B, "fallback_psm13": None})
    if not v2_full_B:
        empty_outputs_log.append({"model": "v2", "condition": "B (padded 20px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops_padded/line_{idx:02d}_padded.png", "output": v2_full_B, "fallback_psm13": None})
    if not tess_full_B_7:
        empty_outputs_log.append({"model": "tesseract", "condition": "B (padded 20px)", "mode": "full", "line_idx": idx, "line_num": idx+1, "chunk_idx": None, "path": f"crops_padded/line_{idx:02d}_padded.png", "output": tess_full_B_7, "fallback_psm13": tess_full_B_13})

    # Join strings
    join_str = " "
    v1_rule_joined_A = normalize(join_str.join([p for p in v1_outs_A if p]))
    v2_rule_joined_A = normalize(join_str.join([p for p in v2_outs_A if p]))
    tess_rule_joined_A = normalize(join_str.join([p for p in tess_outs_A_psm7 if p]))
    
    v1_rule_joined_B = normalize(join_str.join([p for p in v1_outs_B if p]))
    v2_rule_joined_B = normalize(join_str.join([p for p in v2_outs_B if p]))
    tess_rule_joined_B = normalize(join_str.join([p for p in tess_outs_B_psm7 if p]))
    
    detailed_rows.append({
        "line_idx": idx,
        "line_num": idx + 1,
        "ref": ref,
        "crop_h": h_c,
        "crop_w": w_c,
        "ratio_w_h": r_meta["ratio_w_h"],
        "ratio_group": r_meta["ratio_group"],
        "rule_n_chunks": rule_n,
        "chunk_paths_A": "; ".join(chunk_paths_A),
        "chunk_paths_B": "; ".join(chunk_paths_B),
        # Condition A
        "v1_raw_chunks_A": " | ".join(v1_outs_A),
        "v2_raw_chunks_A": " | ".join(v2_outs_A),
        "tess_raw_chunks_A_psm7": " | ".join(tess_outs_A_psm7),
        "tess_raw_chunks_A_psm13": " | ".join(tess_outs_A_psm13),
        "v1_rule_joined_A": v1_rule_joined_A,
        "v2_rule_joined_A": v2_rule_joined_A,
        "tess_rule_joined_A": tess_rule_joined_A,
        "v1_full_A": normalize(v1_full_A),
        "v2_full_A": normalize(v2_full_A),
        "tess_full_A_psm7": normalize(tess_full_A_7),
        "tess_full_A_psm13": normalize(tess_full_A_13),
        # Condition B
        "v1_raw_chunks_B": " | ".join(v1_outs_B),
        "v2_raw_chunks_B": " | ".join(v2_outs_B),
        "tess_raw_chunks_B_psm7": " | ".join(tess_outs_B_psm7),
        "tess_raw_chunks_B_psm13": " | ".join(tess_outs_B_psm13),
        "v1_rule_joined_B": v1_rule_joined_B,
        "v2_rule_joined_B": v2_rule_joined_B,
        "tess_rule_joined_B": tess_rule_joined_B,
        "v1_full_B": normalize(v1_full_B),
        "v2_full_B": normalize(v2_full_B),
        "tess_full_B_psm7": normalize(tess_full_B_7),
        "tess_full_B_psm13": normalize(tess_full_B_13),
    })

df_details = pd.DataFrame(detailed_rows)
df_details.to_csv(OUT_DIR / "per_line_chunk_details.csv", index=False, encoding="utf-8")

with open(OUT_DIR / "empty_outputs_log.json", "w", encoding="utf-8") as f:
    json.dump(empty_outputs_log, f, indent=2, ensure_ascii=False)

print(f"Total logged empty outputs: {len(empty_outputs_log)}")

# Mutually exclusive error taxonomy function
PUNCT_SET = set(".,:;!?—-\"'()[]{}«»“”„’…–/\\")

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
            # Deletion classified by the reference character being omitted!
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

def bootstrap_paired_diff(v2_preds, tess_preds, refs, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(refs)
    cer_diffs = []
    wer_diffs = []
    for _ in range(n_boot):
        idxs = rng.choice(n, size=n, replace=True)
        sub_refs = [refs[i] for i in idxs]
        sub_v2 = [v2_preds[i] for i in idxs]
        sub_tess = [tess_preds[i] for i in idxs]
        m_v2 = compute_metrics(sub_v2, sub_refs)
        m_tess = compute_metrics(sub_tess, sub_refs)
        cer_diffs.append(m_v2["cer"] - m_tess["cer"])
        wer_diffs.append(m_v2["wer"] - m_tess["wer"])
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
            ce, _ = align(ref, hyp)
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

# 1. Condition A metrics (all 33 lines)
m_cond_A = {
    "v1_full": compute_metrics(df_details["v1_full_A"].tolist(), refs_all),
    "v2_full": compute_metrics(df_details["v2_full_A"].tolist(), refs_all),
    "tess_full_psm7": compute_metrics(df_details["tess_full_A_psm7"].tolist(), refs_all),
    "tess_full_psm13": compute_metrics(df_details["tess_full_A_psm13"].tolist(), refs_all),
    "v1_rule_chunk": compute_metrics(df_details["v1_rule_joined_A"].tolist(), refs_all),
    "v2_rule_chunk": compute_metrics(df_details["v2_rule_joined_A"].tolist(), refs_all),
    "tess_rule_chunk_psm7": compute_metrics(df_details["tess_rule_joined_A"].tolist(), refs_all),
}
ci_cer_A, ci_wer_A = bootstrap_paired_diff(df_details["v2_rule_joined_A"].tolist(), df_details["tess_rule_joined_A"].tolist(), refs_all)
m_cond_A["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_cond_A["v2_rule_chunk"]["cer"] - m_cond_A["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_cond_A["v2_rule_chunk"]["wer"] - m_cond_A["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_A,
    "wer_ci": ci_wer_A
}
tax_A = evaluate_taxonomy(df_details, "v2_rule_joined_A", "tess_rule_joined_A")

# 2. Condition B metrics (all 33 lines)
m_cond_B = {
    "v1_full": compute_metrics(df_details["v1_full_B"].tolist(), refs_all),
    "v2_full": compute_metrics(df_details["v2_full_B"].tolist(), refs_all),
    "tess_full_psm7": compute_metrics(df_details["tess_full_B_psm7"].tolist(), refs_all),
    "tess_full_psm13": compute_metrics(df_details["tess_full_B_psm13"].tolist(), refs_all),
    "v1_rule_chunk": compute_metrics(df_details["v1_rule_joined_B"].tolist(), refs_all),
    "v2_rule_chunk": compute_metrics(df_details["v2_rule_joined_B"].tolist(), refs_all),
    "tess_rule_chunk_psm7": compute_metrics(df_details["tess_rule_joined_B"].tolist(), refs_all),
}
ci_cer_B, ci_wer_B = bootstrap_paired_diff(df_details["v2_rule_joined_B"].tolist(), df_details["tess_rule_joined_B"].tolist(), refs_all)
m_cond_B["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_cond_B["v2_rule_chunk"]["cer"] - m_cond_B["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_cond_B["v2_rule_chunk"]["wer"] - m_cond_B["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_B,
    "wer_ci": ci_wer_B
}
tax_B = evaluate_taxonomy(df_details, "v2_rule_joined_B", "tess_rule_joined_B")

# 3. Clean Subset: lines where no model produced empty output in Condition A
# Lines with empty chunk or full output in Condition A:
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
ci_cer_clean, ci_wer_clean = bootstrap_paired_diff(df_clean_A["v2_rule_joined_A"].tolist(), df_clean_A["tess_rule_joined_A"].tolist(), refs_clean)
m_clean["bootstrap_ci_rule_chunk_psm7"] = {
    "delta_cer": m_clean["v2_rule_chunk"]["cer"] - m_clean["tess_rule_chunk_psm7"]["cer"],
    "delta_wer": m_clean["v2_rule_chunk"]["wer"] - m_clean["tess_rule_chunk_psm7"]["wer"],
    "cer_ci": ci_cer_clean,
    "wer_ci": ci_wer_clean
}
tax_clean = evaluate_taxonomy(df_clean_A, "v2_rule_joined_A", "tess_rule_joined_A")

# 4. Grouped V2 CER by ratio group: (<1.0, 1.0-1.1, 1.1-1.3, >1.3)
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

# Save overall metrics
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

# 5. Generate Contact Sheet for lines 18, 19, 21, 22 (0-indexed 17, 18, 20, 21)
# Inspecting the second chunk (chunk_01) in Condition A vs Condition B
target_line_idxs = [17, 18, 20, 21]
target_rows = df_details[df_details["line_idx"].isin(target_line_idxs)].to_dict("records")

# Build a clean visual contact sheet image
card_w = 1400
card_h = 320
cs_img = np.full((len(target_rows) * card_h + 80, card_w, 3), 245, dtype=np.uint8)

# Title bar
cv2.putText(cs_img, "Photo 5: Second Chunks for Lines 18, 19, 21, 22 (Condition A vs B)", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)

y_offset = 80
for r in target_rows:
    idx = r["line_idx"]
    line_num = idx + 1
    ch_A = cv2.imread(str(OUT_DIR / f"chunks/line_{idx:02d}_chunk_01_trimmed.png"))
    ch_B = cv2.imread(str(OUT_DIR / f"chunks_padded/line_{idx:02d}_chunk_01_padded.png"))
    
    # Draw card background
    cv2.rectangle(cs_img, (20, y_offset), (card_w - 20, y_offset + card_h - 20), (255, 255, 255), -1)
    cv2.rectangle(cs_img, (20, y_offset), (card_w - 20, y_offset + card_h - 20), (200, 200, 200), 1)
    
    # Metadata text
    header = f"Line {line_num:02d} (idx {idx:02d}) | Ref Chunk: \"{r['ref'].split()[-4:]}\" | Ratio: {r['ratio_w_h']:.3f} ({r['ratio_group']})"
    cv2.putText(cs_img, header, (40, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 150), 2, cv2.LINE_AA)
    
    # Show Condition A chunk
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
    
    # Show Condition B chunk
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
    
    # Tesseract outputs
    t_A_7 = r["tess_raw_chunks_A_psm7"].split(" | ")[-1] if " | " in r["tess_raw_chunks_A_psm7"] else r["tess_raw_chunks_A_psm7"]
    t_A_13 = r["tess_raw_chunks_A_psm13"].split(" | ")[-1] if " | " in r["tess_raw_chunks_A_psm13"] else r["tess_raw_chunks_A_psm13"]
    v2_A = r["v2_raw_chunks_A"].split(" | ")[-1] if " | " in r["v2_raw_chunks_A"] else r["v2_raw_chunks_A"]
    
    txt_tess = f"Tesseract psm 7: \"{t_A_7}\" (EMPTY!) -> psm 13: \"{t_A_13[:35]}\""
    txt_v2 = f"V2 Line Output: \"{v2_A[:50]}\""
    cv2.putText(cs_img, txt_tess, (40, y_offset + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2, cv2.LINE_AA)
    cv2.putText(cs_img, txt_v2, (40, y_offset + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 0), 2, cv2.LINE_AA)
    
    y_offset += card_h

cs_path = CONTACT_DIR / "contact_sheet_lines18_19_21_22.png"
cv2.imwrite(str(cs_path), cs_img)

# Also save copy as artifact
brain_dir = Path("C:/Users/serxan.m/.gemini/antigravity-ide/brain/1ece9f72-797d-4105-b45d-b26d0dfbe9ac")
if brain_dir.exists():
    cv2.imwrite(str(brain_dir / "contact_sheet_p5_empty_chunks.png"), cs_img)

print(f"Contact sheet saved to {cs_path}")
