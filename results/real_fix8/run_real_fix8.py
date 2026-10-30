"""Master Script for Real Fix 8:
1. Fix joining & splitting:
   - Chunk split rule: ink projection with ksize=25 (from real_fix6) to ensure splits fall strictly in inter-word white gaps.
   - Robust CC trim rule: remove components with area < 8 px, remove components touching left/right border with h >= 0.70 * h_chunk, trim to remaining ink bounds + 4 px.
   - ONE unified joining rule for both photos and all models: " ".join([p for p in chunk_outs if p]).
2. Metrics computation:
   - All CER and WER stored and reported strictly in percent (%) (c * 100.0, w * 100.0).
   - Bootstrap CIs for V2 - Tess CER and WER in percent.
   - Mutually exclusive error taxonomy computed for BOTH V2 and Tesseract.
3. Photo 2 Diagnostics:
   - Report per line: n_chunks, crop widths before/after trimming, raw chunk predictions, join string, joined vs ref.
4. Photo 1 L12-L15 contact sheet and shadow stripe analysis.
"""
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import load_model_checkpoint, infer_single_crop, normalize, cer_wer, align

OUT_DIR = Path("results/real_fix8")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_P1_DIR = OUT_DIR / "crops_times"
CROPS_P1_DIR.mkdir(parents=True, exist_ok=True)
CROPS_P2_DIR = OUT_DIR / "crops_calibri"
CROPS_P2_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = OUT_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EDGE_DIR = OUT_DIR / "edge_inspection"
EDGE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Ground Truth References
P1_REFS = [
    "Çöldə isə nəvəsi Murad səhərin sərin havasında babasının köhnə velosipedini təmizləyirdi",
    "Murad şəhərdə oxusa da, hər tətil otağında kəndə, bu torpağın qoxusuna qayıdırdı. Amma bu",
    "dəfə qayıdışı başqa idi. O, hiss edirdi ki, babası təkdir və onun daşıdığı bu ağır yük —",
    "unudulmaqda olan xalçaçılıq sənəti hər keçən gün daha da zəifləyir. Murad əlindəki dəsmalla",
    "velosipedin sükanını silə-silə başını qaldırıb emalatxananın açıq qapısından içəri baxdı.",
    "Babasının əyilmiş belini, dəzgah arxasında təkbaşına necə mücadilə etdiyini görəndə qəlbində",
    "qəribə bir sızıltı oldu. O, ayaqqabılarını çıxarıb içəri daxil oldu, ehtiyatla babasının çiyinlərinə",
    "— Baba, səhər tezdən yenə başladın? Bir az dincəlsənə, — deyə Murad incə, lakin qayğılı səslə",
    "Usta Əli təəccüblə geri döndü, üzündəki qırışlar yumşaldı və dərin, mənalı gözləri ilə nəvəsinə",
    "baxdı. O, təbəssüm edərək dedi: — Ay bala, usta o qoca olar ki, əlləri işləməyəndə ruhu da",
    "dincəlir. Bu xalça sadəcə ipdən toxunmur. Görürsən bu naxışları? Hər ilmə bir zamandır, bir",
    "tariximizdir. Biz getsək də, bu naxışlar qalacaq. Amma naxışların dili olmasa, heç kim bizim kim",
    "Murad babasının dediklərinin mənasını tam dərk etməsə də, onun səsindəki həsrəti və qorxunu",
    "hiss edirdi. Babası qorxurdu ki, o öləndən sonra bu qədim sənət də onunla birlikdə torpağa",
    "basdırılacaq. Həmin an Murad daxilində qəti bir qərar verdi: bu xalçanın son ilməsinə qədər",
    "babasına kömək edəcək və bu sənətin yerə düşməsinə imkan verməyəcək.",
]

P2_REFS = [
    "Gəncə çayının kənarında, qədim ağacların kölgəsinə sığınmış balaca bir dulusçu emalatxanası var",
    "idi. Burada illərdir gil ilə nəfəs alan, gilə can verən Qoca Səməndər yaşayırdı. Səməndər usta",
    "əllərini palçığın tarixinə batıranda sanki zaman dayanırdı, onun hər bir hərəkətində əsrlərin",
    "sənəti, nəsildən-nəslə keçən gizli bir məharət yaşatdırdığı hiss olunurdu. Divarlardan asılmış gil",
    "kasalar, bəzəkli küplər, nəqşli boşqablar otağa xüsusi bir torpaq qoxusu və qədimlik gətirirdi.",
    "Səməndər usta çəliyini kənara qoyub ağır-ağar dolabın arxasına keçdi, qabarıq və çat-çat olmuş",
    "barmaqları ilə fırlanan dairəvi taxtaya toxundu. Bu taxta onun ömrünün yekunu, sonuncu böyük",
    "idi. Elə həmin an emalatxananın qapısı ehtiyatla açıldı və içəri şəhərdən yeni gəlmiş nəvəsi Taryel",
    "girdi. Taryel ali məktəbi bitirib şəhərdə böyük imkanlar qazanmış bir gənc idi, lakin hər dəfə bu",
    "kəndə gələndə özünü yenidən uşaqlıq illərindəki kimi azad və rahat hiss edirdi. O, babasının",
    "əyilmiş belini, yorğun, lakin qətiyyətli baxışlarını görüb asta addımlarla yaxınlaşdı, əlindəki",
    "çantanı bir kənara qoyub babasının çiyninə toxundu.",
    "— Baba, yenə saatlarla bu təkərin arxasında oturmusan. Özünü yorursan, gəl bu gün biraz dincəl,",
    "Qoca Səməndər başını qaldırıb nəvəsinə baxdı, dodaqlarında yüngül, lakin dərin mənalı bir",
    "təbəssüm yarandı. O, ovcundakı nəm gili yavaşca masaya qoyub dedi: — Ay bala, insan o vaxt",
    "qocalır ki, torpağa olan inamı ölür. Bu gil sadəcə torpaq parçasımı sanırsan? Xeyr! Hər küpün,",
]

# 2. Images & Geometry (Identical to real_fix6 and real_fix7)
print("Loading images and geometry...")
img1 = cv2.imread("data/real_docs/photo1_times.jpg")
img2_raw = cv2.imread("data/real_docs/photo2_calibri.jpg")

h2, w2 = img2_raw.shape[:2]
M2 = cv2.getRotationMatrix2D((w2 // 2, h2 // 2), -0.27, 1.0)
img2_deskewed = cv2.warpAffine(img2_raw, M2, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(str(OUT_DIR / "photo2_deskewed.jpg"), img2_deskewed)

def get_edge_ink_standard(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thresh = (gray < 160)
    top2_px = int(thresh[:2, :].sum())
    btm2_px = int(thresh[-2:, :].sum())
    top2_f = float(top2_px) / (2 * w)
    btm2_f = float(btm2_px) / (2 * w)
    return top2_f, btm2_f, top2_px, btm2_px

P1_Y_INTERVALS = [
    (1106, 1195), (1194, 1275), (1273, 1348), (1346, 1425),
    (1423, 1503), (1501, 1587), (1585, 1665), (1783, 1857),
    (1986, 2052), (2065, 2130), (2147, 2212), (2227, 2294),
    (2430, 2501), (2499, 2592), (2596, 2678), (2676, 2759)
]

P2_Y_INTERVALS = [
    (605, 675), (673, 739), (737, 802), (809, 878),
    (872, 942), (940, 1012), (1010, 1079), (1138, 1217),
    (1215, 1288), (1294, 1356), (1365, 1431), (1439, 1501),
    (1548, 1609), (1735, 1793), (1812, 1869), (1886, 1944)
]

P1_COMMON_X = (200, 2620)
P2_COMMON_X = (360, 2520)

p1_crops, p1_meta = [], []
for idx, (y0, y1) in enumerate(P1_Y_INTERVALS):
    r_x = 2070 if idx == 15 else P1_COMMON_X[1]
    c = img1[y0:y1, P1_COMMON_X[0]:r_x]
    c_path = CROPS_P1_DIR / f"line_{idx:02d}.png"
    cv2.imwrite(str(c_path), c)
    p1_crops.append(c)
    top_f, btm_f, top_px, btm_px = get_edge_ink_standard(c)
    h_c, w_c = c.shape[:2]
    p1_meta.append({
        "line_idx": idx, "h": h_c, "w": w_c, "y0": y0, "y1": y1,
        "x0": P1_COMMON_X[0], "x1": r_x,
        "top2_frac": top_f, "btm2_frac": btm_f,
        "top2_pixels": top_px, "btm2_pixels": btm_px,
        "ref": P1_REFS[idx]
    })

p2_crops, p2_meta = [], []
for idx, (y0, y1) in enumerate(P2_Y_INTERVALS):
    r_x = 1550 if idx == 11 else P2_COMMON_X[1]
    c = img2_deskewed[y0:y1, P2_COMMON_X[0]:r_x]
    c_path = CROPS_P2_DIR / f"line_{idx:02d}.png"
    cv2.imwrite(str(c_path), c)
    p2_crops.append(c)
    top_f, btm_f, top_px, btm_px = get_edge_ink_standard(c)
    h_c, w_c = c.shape[:2]
    p2_meta.append({
        "line_idx": idx, "h": h_c, "w": w_c, "y0": y0, "y1": y1,
        "x0": P2_COMMON_X[0], "x1": r_x,
        "top2_frac": top_f, "btm2_frac": btm_f,
        "top2_pixels": top_px, "btm2_pixels": btm_px,
        "ref": P2_REFS[idx]
    })

with open(OUT_DIR / "crop_geometry.json", "w", encoding="utf-8") as f:
    json.dump({"photo1": p1_meta, "photo2": p2_meta}, f, indent=2)

# 3. Interline Check zooms for Photo 1 L00-L01 and L01-L02
vis_01 = img1[1179:1211, 200:1600].copy()
cv2.line(vis_01, (0, 16), (vis_01.shape[1], 16), (0, 0, 255), 1)
cv2.line(vis_01, (0, 15), (vis_01.shape[1], 15), (0, 255, 0), 1)
cv2.imwrite(str(EDGE_DIR / "p1_boundary_L00_L01.png"), vis_01)

vis_12 = img1[1258:1291, 200:1600].copy()
cv2.line(vis_12, (0, 17), (vis_12.shape[1], 17), (0, 0, 255), 1)
cv2.line(vis_12, (0, 15), (vis_12.shape[1], 15), (0, 255, 0), 1)
cv2.imwrite(str(EDGE_DIR / "p1_boundary_L01_L02.png"), vis_12)

# 4. Splitting & Robust CC Trimming
def find_split_points(crop_bgr, n_chunks):
    """Splits at vertical white gaps using ksize=25 smoothing from real_fix6."""
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

removed_tall_components = []

def robust_trim_chunk(chunk_bgr, chunk_tag=""):
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
        
        if is_small_speck or is_tall_shadow:
            if h > 6:
                reason = "shadow_stripe" if is_tall_shadow else "area_lt_8"
                removed_tall_components.append({
                    "chunk": chunk_tag,
                    "component_id": int(i),
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "area": int(area),
                    "chunk_w": int(w_chunk), "chunk_h": int(h_chunk),
                    "reason": reason
                })
        else:
            clean_ink[labels == i] = 1
            
    col_counts = clean_ink.sum(axis=0)
    valid_cols = np.where(col_counts > 0)[0]
    if len(valid_cols) == 0:
        return chunk_bgr, w_chunk, w_chunk
    x0 = max(0, int(valid_cols[0]) - 4)
    x1 = min(w_chunk, int(valid_cols[-1]) + 5)
    trimmed = chunk_bgr[:, x0:x1]
    return trimmed, w_chunk, trimmed.shape[1]

def run_tess_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    txt = pytesseract.image_to_string(Image.fromarray(rgb), lang='aze', config='--psm 7').strip()
    return normalize(" ".join(txt.split()))

# 5. Load OCR Models
print("Loading V1 and V2 models...")
v2_line_m, v2_line_cfg, v2_line_voc, dev = load_model_checkpoint("runs/line_v2/best.pt")
v1_line_m, v1_line_cfg, v1_line_voc, dev = load_model_checkpoint("runs/line/best.pt")

# 6. Evaluation Execution
print("Executing evaluation on full lines and robustly trimmed chunks...")
detailed_chunk_rows = []
photo2_diagnostics = []

datasets = [
    ("Photo 1 (Times New Roman)", P1_REFS, p1_crops, "p1"),
    ("Photo 2 (Calibri)", P2_REFS, p2_crops, "p2"),
]

contact_data = {"p1": [], "p2": []}

for photo_name, refs, crops, p_tag in datasets:
    n_lines = len(refs)
    for idx in range(n_lines):
        c = crops[idx]
        ref = normalize(refs[idx])
        h_c, w_c = c.shape[:2]
        val_formula = (w_c * 32) / (h_c * 512)
        rule_n = max(1, math.ceil(val_formula))
        
        raw_chunks = split_into_chunks(c, rule_n)
        chunk_paths = []
        v1_chunk_outs = []
        v2_chunk_outs = []
        tess_chunk_outs = []
        trimmed_chunk_imgs = []
        widths_before = []
        widths_after = []
        
        for ch_i, ch in enumerate(raw_chunks):
            tag = f"{p_tag}_line_{idx:02d}_chunk_{ch_i:02d}"
            ch_trimmed, w_bef, w_aft = robust_trim_chunk(ch, tag)
            widths_before.append(w_bef)
            widths_after.append(w_aft)
            
            ch_fname = f"{tag}_trimmed.png"
            ch_p = CHUNKS_DIR / ch_fname
            cv2.imwrite(str(ch_p), ch_trimmed)
            chunk_paths.append(str(ch_p.relative_to(OUT_DIR)))
            trimmed_chunk_imgs.append(ch_trimmed)
            
            p1_ch = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, ch_trimmed, dev).strip()
            p2_ch = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, ch_trimmed, dev).strip()
            pt_ch = run_tess_crop(ch_trimmed)
            
            v1_chunk_outs.append(p1_ch)
            v2_chunk_outs.append(p2_ch)
            tess_chunk_outs.append(pt_ch)
            
        v1_full = infer_single_crop(v1_line_m, v1_line_cfg, v1_line_voc, c, dev).strip()
        v2_full = infer_single_crop(v2_line_m, v2_line_cfg, v2_line_voc, c, dev).strip()
        tess_full = run_tess_crop(c)
        
        # ONE unified join rule: " ".join([p for p in outs if p])
        join_str = " "
        v1_joined = normalize(join_str.join([p for p in v1_chunk_outs if p]))
        v2_joined = normalize(join_str.join([p for p in v2_chunk_outs if p]))
        tess_joined = normalize(join_str.join([p for p in tess_chunk_outs if p]))
        
        first_ch = trimmed_chunk_imgs[0]
        last_ch = trimmed_chunk_imgs[-1]
        contact_data[p_tag].append({
            "line_idx": idx,
            "first_ch": first_ch,
            "last_ch": last_ch,
            "n_chunks": rule_n
        })
        
        if p_tag == "p2":
            photo2_diagnostics.append({
                "line_idx": idx,
                "n_chunks": rule_n,
                "widths_before": widths_before,
                "widths_after": widths_after,
                "raw_v2_chunks": v2_chunk_outs,
                "raw_tess_chunks": tess_chunk_outs,
                "join_string": repr(join_str),
                "v2_joined": v2_joined,
                "tess_joined": tess_joined,
                "reference": ref,
                "v2_matches_ref": (v2_joined == ref),
                "tess_matches_ref": (tess_joined == ref),
            })
            
        detailed_chunk_rows.append({
            "photo": photo_name,
            "line_idx": idx,
            "ref": ref,
            "crop_h": h_c,
            "crop_w": w_c,
            "formula_val": val_formula,
            "rule_n_chunks": rule_n,
            "chunk_crop_paths": "; ".join(chunk_paths),
            "v1_chunks_raw": " | ".join(v1_chunk_outs),
            "v2_chunks_raw": " | ".join(v2_chunk_outs),
            "tess_chunks_raw": " | ".join(tess_chunk_outs),
            "v1_rule_joined": v1_joined,
            "v2_rule_joined": v2_joined,
            "tess_rule_joined": tess_joined,
            "v1_full": normalize(v1_full),
            "v2_full": normalize(v2_full),
            "tess_full": normalize(tess_full),
        })

df_chunks = pd.DataFrame(detailed_chunk_rows)
df_chunks.to_csv(OUT_DIR / "per_line_chunk_details.csv", index=False, encoding='utf-8-sig')

df_p2_diag = pd.DataFrame(photo2_diagnostics)
df_p2_diag.to_csv(OUT_DIR / "photo2_line_diagnostics.csv", index=False, encoding='utf-8-sig')

with open(OUT_DIR / "removed_tall_components.json", "w", encoding="utf-8") as f:
    json.dump(removed_tall_components, f, indent=2)

# 7. Generate Contact Sheets
print("Generating contact sheets...")
def build_contact_sheet(items, title, output_path):
    target_h = 40
    row_margin = 10
    total_w = 1600
    sheet_h = 60 + len(items) * (target_h + row_margin) + 20
    canvas = np.full((sheet_h, total_w, 3), 255, dtype=np.uint8)
    
    cv2.putText(canvas, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(canvas, "Format: [Line ID] | First Chunk (Left Margin) | Last Chunk (Right Margin)", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    
    y_off = 70
    for it in items:
        idx = it["line_idx"]
        lbl = f"L{idx:02d} (n={it['n_chunks']}):"
        cv2.putText(canvas, lbl, (20, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
        
        ch1 = it["first_ch"]
        h1, w1 = ch1.shape[:2]
        new_w1 = max(1, int(w1 * target_h / h1))
        ch1_res = cv2.resize(ch1, (new_w1, target_h), interpolation=cv2.INTER_AREA)
        
        ch2 = it["last_ch"]
        h2, w2 = ch2.shape[:2]
        new_w2 = max(1, int(w2 * target_h / h2))
        ch2_res = cv2.resize(ch2, (new_w2, target_h), interpolation=cv2.INTER_AREA)
        
        x_ch1 = 130
        x_ch1_end = min(total_w // 2 - 10, x_ch1 + new_w1)
        paste_w1 = x_ch1_end - x_ch1
        if paste_w1 > 0:
            canvas[y_off:y_off+target_h, x_ch1:x_ch1_end] = ch1_res[:, :paste_w1]
            cv2.rectangle(canvas, (x_ch1, y_off), (x_ch1_end, y_off+target_h), (0, 0, 0), 1)
            
        x_ch2 = total_w // 2 + 10
        x_ch2_end = min(total_w - 20, x_ch2 + new_w2)
        paste_w2 = x_ch2_end - x_ch2
        if paste_w2 > 0:
            canvas[y_off:y_off+target_h, x_ch2:x_ch2_end] = ch2_res[:, :paste_w2]
            cv2.rectangle(canvas, (x_ch2, y_off), (x_ch2_end, y_off+target_h), (0, 0, 0), 1)
            
        y_off += target_h + row_margin
        
    cv2.imwrite(str(output_path), canvas)

build_contact_sheet(contact_data["p1"], "Photo 1 (Times New Roman) - Trimmed Chunk Contact Sheet", OUT_DIR / "contact_sheet_photo1.png")
build_contact_sheet(contact_data["p2"], "Photo 2 (Calibri) - Trimmed Chunk Contact Sheet", OUT_DIR / "contact_sheet_photo2.png")

# Contact sheet specifically for Photo 1 Lines 12-15 (Task 3)
p1_12_15_items = [it for it in contact_data["p1"] if it["line_idx"] in [12, 13, 14, 15]]
build_contact_sheet(p1_12_15_items, "Photo 1 (Lines 12-15) - First & Last Chunks Shadow Inspection", OUT_DIR / "contact_sheet_p1_l12_l15.png")

# 8. Metrics Computation (Strictly in Percent %)
def norm_dashes(s):
    if not isinstance(s, str):
        return ""
    for d in ["—", "–", "−"]:
        s = s.replace(d, "-")
    return normalize(s)

def get_line_edits(ref, hyp):
    ce, ops = align(ref, hyp)
    we, _ = align(ref.split(), hyp.split())
    sp_char = sum(1 for op, rc, hc in ops if op != "ok" and (rc == ' ' or hc == ' '))
    return ce, len(ref), we, len(ref.split()), sp_char, ops

def fast_bootstrap(refs, hyps_v2, hyps_tess, n_resamples=1000, seed=42):
    n = len(refs)
    c_v2, c_tess, c_lens = [], [], []
    w_v2, w_tess, w_lens = [], [], []
    for r, v2, t in zip(refs, hyps_v2, hyps_tess):
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
    s_idx = rng.choice(n, size=(n_resamples, n), replace=True)
    v2_cer_boot = c_v2[s_idx].sum(axis=1) / c_lens[s_idx].sum(axis=1) * 100.0
    tess_cer_boot = c_tess[s_idx].sum(axis=1) / c_lens[s_idx].sum(axis=1) * 100.0
    cer_diffs = v2_cer_boot - tess_cer_boot
    v2_wer_boot = w_v2[s_idx].sum(axis=1) / w_lens[s_idx].sum(axis=1) * 100.0
    tess_wer_boot = w_tess[s_idx].sum(axis=1) / w_lens[s_idx].sum(axis=1) * 100.0
    wer_diffs = v2_wer_boot - tess_wer_boot
    return [float(np.percentile(cer_diffs, 2.5)), float(np.percentile(cer_diffs, 97.5))], [float(np.percentile(wer_diffs, 2.5)), float(np.percentile(wer_diffs, 97.5))]

metrics = {}
for mode in ["raw", "normalized_dashes"]:
    for photo_name in ["Photo 1 (Times New Roman)", "Photo 2 (Calibri)"]:
        sub_df = df_chunks[df_chunks.photo == photo_name].copy()
        n_lines = len(sub_df)
        refs = sub_df.ref.tolist()
        if mode == "normalized_dashes":
            refs = [norm_dashes(r) for r in refs]
        tot_chars = sum(len(r) for r in refs)
        
        for cond in ["full_line", "rule_based_trimmed"]:
            v1_col = "v1_full" if cond == "full_line" else "v1_rule_joined"
            v2_col = "v2_full" if cond == "full_line" else "v2_rule_joined"
            tess_col = "tess_full" if cond == "full_line" else "tess_rule_joined"
            
            p1_list = [str(x) if pd.notna(x) else "" for x in sub_df[v1_col]]
            p2_list = [str(x) if pd.notna(x) else "" for x in sub_df[v2_col]]
            pt_list = [str(x) if pd.notna(x) else "" for x in sub_df[tess_col]]
            
            if mode == "normalized_dashes":
                p1_list = [norm_dashes(p) for p in p1_list]
                p2_list = [norm_dashes(p) for p in p2_list]
                pt_list = [norm_dashes(p) for p in pt_list]
                
            c1, w1, _ = cer_wer(refs, p1_list)
            c2, w2, ops2 = cer_wer(refs, p2_list)
            ct, wt, opst = cer_wer(refs, pt_list)
            
            exact1 = sum(1 for r, p in zip(refs, p1_list) if r == p)
            exact2 = sum(1 for r, p in zip(refs, p2_list) if r == p)
            exactt = sum(1 for r, p in zip(refs, pt_list) if r == p)
            
            sp_char1 = sum(get_line_edits(r, p)[4] for r, p in zip(refs, p1_list))
            sp_char2 = sum(get_line_edits(r, p)[4] for r, p in zip(refs, p2_list))
            sp_chart = sum(get_line_edits(r, p)[4] for r, p in zip(refs, pt_list))
            
            boot_cer_ci, boot_wer_ci = fast_bootstrap(refs, p2_list, pt_list)
            
            # STORE STRICTLY IN PERCENT (%)
            c1_pct = c1 * 100.0
            w1_pct = w1 * 100.0
            c2_pct = c2 * 100.0
            w2_pct = w2 * 100.0
            ct_pct = ct * 100.0
            wt_pct = wt * 100.0
            delta_cer_pct = c2_pct - ct_pct
            delta_wer_pct = w2_pct - wt_pct
            
            key = f"{photo_name}::{cond}::{mode}"
            metrics[key] = {
                "photo": photo_name, "condition": cond, "mode": mode,
                "n_lines": n_lines, "n_chars": tot_chars,
                "v1": {"cer": c1_pct, "wer": w1_pct, "exact": exact1, "space_errors_char": sp_char1},
                "v2": {"cer": c2_pct, "wer": w2_pct, "exact": exact2, "space_errors_char": sp_char2},
                "tess": {"cer": ct_pct, "wer": wt_pct, "exact": exactt, "space_errors_char": sp_chart},
                "delta_cer": delta_cer_pct, "delta_wer": delta_wer_pct,
                "delta_cer_ci": boot_cer_ci, "delta_wer_ci": boot_wer_ci
            }

with open(OUT_DIR / "real_fix8_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# 9. Mutually Exclusive Error Taxonomy for BOTH V2 and Tesseract (Task 4)
print("Computing error taxonomy for V2 and Tesseract...")
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

tax_summary = {"raw": {}, "normalized_dashes": {}}
for mode in ["raw", "normalized_dashes"]:
    tax_summary[mode] = {"v2": {}, "tesseract": {}}
    for photo_name in ["Photo 1 (Times New Roman)", "Photo 2 (Calibri)"]:
        sub_df = df_chunks[df_chunks.photo == photo_name].copy()
        
        # Taxonomy for V2
        for model_name, col_name in [("v2", "v2_rule_joined"), ("tesseract", "tess_rule_joined")]:
            tot_lead, tot_trail, tot_sp, tot_o0, tot_punct, tot_let = 0, 0, 0, 0, 0, 0
            total_levenshtein = 0
            all_details = []
            
            for idx, r in sub_df.iterrows():
                ref = r["ref"]
                hyp = r[col_name]
                if mode == "normalized_dashes":
                    ref = norm_dashes(ref)
                    hyp = norm_dashes(hyp)
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
            tax_summary[mode][model_name][photo_name] = {
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

print("Real Fix 8 execution completed successfully!")
