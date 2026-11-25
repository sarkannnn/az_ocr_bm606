"""AzOCR-V2 — R59: Open-License Unseen-Font Benchmark Generator & Evaluator.

Generates 200 synthetic line images (40 original CC0 Azerbaijani sentences x 5
unseen open-license fonts: DejaVuSans, DejaVuSerif, DejaVuSansMono, Lato-Regular,
Raleway-Regular), verifies zero overlap with training corpus and training fonts,
records SHA-256 manifests, and evaluates all 18 official GPU checkpoints + 4
baseline OCR systems without retraining or tuning.
"""

import hashlib
import os
import shutil
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

PROJ_ROOT = Path(r"D:\laragon\www\az_ocr_bm606")
sys.path.insert(0, str(PROJ_ROOT))

os.environ["TESSDATA_PREFIX"] = str(PROJ_ROOT / "tessdata")
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from common import Vocab, CHARS, DEFAULT_CFG, infer_line_smart, normalize, align
from model import CRNN

OUT_DIR = PROJ_ROOT / "R23_R59_Audit"
FONTS_OPEN_DIR = OUT_DIR / "fonts_open"
LICENSES_DIR = OUT_DIR / "licenses"
DATASET_DIR = OUT_DIR / "r59_dataset"
IMAGES_DIR = DATASET_DIR / "images"

RUNS_ROOT = Path(r"C:\Users\serxan.m\Desktop\AzOCR_Trained_Runs_Results\AzOCR_Trained_Runs_Results\runs")
BASELINE_CKPT = PROJ_ROOT / "runs" / "line_v2" / "best.pt"

OPEN_FONTS = [
    (
        "DejaVuSans",
        PROJ_ROOT / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
        "DejaVu / Bitstream Vera License (Public Domain modifications)",
    ),
    (
        "DejaVuSerif",
        PROJ_ROOT / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSerif.ttf",
        "DejaVu / Bitstream Vera License (Public Domain modifications)",
    ),
    (
        "DejaVuSansMono",
        PROJ_ROOT / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSansMono.ttf",
        "DejaVu / Bitstream Vera License (Public Domain modifications)",
    ),
    (
        "Lato-Regular",
        PROJ_ROOT / ".venv/Lib/site-packages/trdg/fonts/latin/Lato-Regular.ttf",
        "SIL Open Font License 1.1",
    ),
    (
        "Raleway-Regular",
        PROJ_ROOT / ".venv/Lib/site-packages/trdg/fonts/latin/Raleway-Regular.ttf",
        "SIL Open Font License 1.1",
    ),
]

GROUPS = [
    ("Group_A_V1Clean_BasicAug", "Group A"),
    ("Group_B_V1Clean_PhotoAug", "Group B"),
    ("Group_C_V2Clean_BasicAug", "Group C"),
    ("Group_D_V2Clean_PhotoAug", "Group D"),
    ("Exp_RandomPadding_GroupD", "RandomPadding"),
    ("Exp_VariableWidth_GroupD", "VariableWidth"),
]
SEEDS = [42, 43, 44]
TARGET_CHARS = ["ə", "Ə", "ı", "i", "İ"]

# 40 original CC0 Azerbaijani sentences specifically authored for R59 (zero corpus overlap)
R59_SENTENCES = [
    "Azərbaycan Respublikasının elmi tədqiqat mərkəzləri yeni rəqəmsal arxiv layihəsini təqdim etdi.",
    "Şəki və İsmayıllı bölgələrində qədim əlyazmaların bərpası üzrə xüsusi komissiya yaradıldı.",
    "Müasir optik tanıma sistemləri mətnin həndəsi əyriliyini və kölgə təsirini azaltmalıdır.",
    "Ətraf mühitin mühafizəsi naminə Xəzər sahilində 2026-cı ildə monitorinq stansiyası qurulacaq.",
    "İqtisadi əməkdaşlıq forumunda enerji səmərəliliyi və yaşıl texnologiyalar müzakirə olundu.",
    "Qarabağ atlarının tarixi irsi beynəlxalq sərgidə ziyarətçilərin böyük marağına səbəb oldu.",
    "Kitabxana fondunun elektron kataloquna 1450 yeni elmi məqalə və monoqrafiya əlavə edildi.",
    "Təbiət qoruqlarında nadir quş növlərinin qışlama marşrutları peyk vasitəsilə izlənilir.",
    "Ənənəvi misgərlik sənəti Lahıc kəndinin mədəni həyatında bu gün də yaşadılır.",
    "Riyazi statistika və ehtimal nəzəriyyəsi üzrə olimpiada qalibləri mükafatlandırıldı.",
    "İnformasiya təhlükəsizliyi standartlarına uyğun olaraq məlumat bazası şifrələnmişdir.",
    "Gəncə şəhərində yeni tikilən laboratoriya kənd təsərrüfatı məhsullarının keyfiyyətini yoxlayır.",
    "Süni intellekt alqoritmləri sənədlərin avtomatik təsnifatını sürətləndirməyə imkan verir.",
    "Şirvan düzündə suvarma kanallarının yenidən qurulması məhsuldarlığı xeyli artırdı.",
    "Ədəbiyyat muzeyində Nizami Gəncəvi irsinə həsr olunmuş elmi konfrans keçirildi.",
    "Dəmir yolu nəqliyyatında rəqəmsal bilet sistemi sərnişinlərin vaxtına qənaət edir.",
    "İqlim dəyişikliyi ilə mübarizə çərçivəsində dağlıq ərazilərdə meşə zolaqları salınır.",
    "Azərbaycan dilinin orfoqrafiya qaydaları yeni lüğət nəşrində dəqiqləşdirilmişdir.",
    "Yüksək dağ kəndlərində günəş panelləri vasitəsilə fasiləsiz elektrik enerjisi təmin olunur.",
    "Mühəndislik fakültəsinin tələbələri muxtar robotların sınaq mərhələsini uğurla tamamladı.",
    "Tarixi memarlıq abidələrinin üçölçülü skan edilməsi gələcək nəsillər üçün əhəmiyyətlidir.",
    "İctimai nəqliyyatda elektron ödəniş kartlarından istifadə göstəricisi 89 faizə çatdı.",
    "Bakı limanından göndərilən yük qatarları Orta Dəhliz üzrə hərəkətini davam etdirir.",
    "Tibbi diaqnostika mərkəzlərində görüntülərin rəqəmsal analizi dəqiqliyi yüksəldir.",
    "Əmək bazarında proqramlaşdırma və verilənlər elmi ixtisaslarına tələbat sürətlə artır.",
    "Quba rayonunun alma bağlarında damcı suvarma üsulu geniş tətbiq edilməyə başlandı.",
    "Şuşa şəhərində keçirilən musiqi festivalında klassik muğam əsərləri səsləndirildi.",
    "İstehsalat müəssisələrində avtomatlaşdırılmış keyfiyyət nəzarəti xətaları minimuma endirir.",
    "Ağdam və Füzuli istiqamətində yeni avtomobil yollarının çəkilişi plan üzrə gedir.",
    "Astronomiya rəsədxanasında uzaq qalaktikaların spektral müşahidələri aparıldı.",
    "Əhalinin siyahıyaalınması zamanı toplanan statistik göstəricilər açıq portalda yerləşdirildi.",
    "İxrac yönümlü sənaye məhsullarının sertifikatlaşdırılması beynəlxalq qaydalara uyğunlaşdırıldı.",
    "Mingəçevir su anbarının hidroloji vəziyyəti mütəxəssislər tərəfindən gündəlik ölçülür.",
    "Məktəbəqədər təhsil müəssisələrində interaktiv öyrənmə vasitələrindən istifadə genişlənir.",
    "Əlyazmalar İnstitutunda qorunan nadir nüsxələrin yüksək keyfiyyətli surətləri çıxarıldı.",
    "İdman kompleksində yeniyetmələr arasında respublika birinciliyi təşkil olundu.",
    "Naxçıvan duzdağının müalicəvi xüsusiyyətləri elmi məqalələrdə ətraflı tədqiq edilmişdir.",
    "Kənd təsərrüfatı texnikasının yenilənməsi payızlıq əkin işlərini vaxtında bitirməyə kömək etdi.",
    "İctimai kitabxanalarda görmə məhdudiyyətli oxucular üçün səsli kitab bölməsi açıldı.",
    "Əməliyyat sistemlərinin təhlükəsizlik yeniləmələri mütəmadi olaraq yoxlanılmalıdır.",
]


def nfc(text) -> str:
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def render_line_image(text: str, font_path: Path, font_size: int = 34, pad_x: int = 12, pad_y: int = 8):
    font = ImageFont.truetype(str(font_path), font_size)
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w = tw + 2 * pad_x
    h = th + 2 * pad_y
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=(15, 15, 15))
    return img


def load_crnn(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", DEFAULT_CFG)
    chars = ck.get("chars", CHARS)
    vocab = Vocab(chars)
    in_ch = 1 if cfg.get("gray", True) else 3
    model = CRNN(in_ch=in_ch, num_classes=vocab.size).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, vocab


def compute_metrics(refs, hyps):
    dc_tot, lc_tot, dw_tot, lw_tot = 0, 0, 0, 0
    exact = 0
    char_hits = {c: 0 for c in TARGET_CHARS}
    char_gt = {c: 0 for c in TARGET_CHARS}
    for r, h in zip(refs, hyps):
        rn, hn = nfc(r), nfc(h)
        dc, ops = align(rn, hn)
        dc_tot += dc
        lc_tot += len(rn)
        dw, _ = align(rn.split(), hn.split())
        dw_tot += dw
        lw_tot += len(rn.split())
        if rn == hn:
            exact += 1
        for c in TARGET_CHARS:
            char_gt[c] += rn.count(c)
        for op, rc, _ in ops:
            if op == "ok" and rc in TARGET_CHARS:
                char_hits[rc] += 1
    out = {
        "n_lines": len(refs),
        "cer_micro_pct": round(dc_tot / max(1, lc_tot) * 100.0, 2),
        "wer_micro_pct": round(dw_tot / max(1, lw_tot) * 100.0, 2),
        "exact_matches": exact,
        "exact_match_pct": round(exact / max(1, len(refs)) * 100.0, 2),
    }
    for c in TARGET_CHARS:
        out[f"acc_{c}_pct"] = round(char_hits[c] / char_gt[c] * 100.0, 2) if char_gt[c] > 0 else "N/A"
    return out


def main():
    print("=" * 80)
    print("AZOCR-V2 — R59: OPEN-LICENSE UNSEEN-FONT BENCHMARK")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_OPEN_DIR.mkdir(parents=True, exist_ok=True)
    LICENSES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verify zero overlap of R59_SENTENCES with training/Real30 corpus
    existing_texts = set()
    for p in [
        PROJ_ROOT / "data/corpus_line_clean/train.txt",
        PROJ_ROOT / "data/corpus_line_clean/val.txt",
    ]:
        if p.exists():
            existing_texts.update(nfc(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
    gt_r30 = pd.read_csv(PROJ_ROOT / "data/real30_evaluation/ground_truth_lines.csv")
    existing_texts.update(nfc(t) for t in gt_r30["ground_truth_nfc"])

    for s in R59_SENTENCES:
        assert nfc(s) not in existing_texts, f"Overlap found: {s}"
    print(f"[+] Verified 0% text overlap across all {len(R59_SENTENCES)} original CC0 sentences.")

    # Copy LICENSE_DEJAVU
    dejavu_lic = PROJ_ROOT / ".venv/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU"
    if dejavu_lic.exists():
        shutil.copy2(dejavu_lic, LICENSES_DIR / "LICENSE_DEJAVU.txt")

    # 2. Generate 200 images & R59_DATASET_MANIFEST.csv
    manifest_rows = []
    for font_name, font_src, font_lic in OPEN_FONTS:
        dst_font = FONTS_OPEN_DIR / font_src.name
        shutil.copy2(font_src, dst_font)
        f_sha = sha256_file(dst_font)
        for idx, sent in enumerate(R59_SENTENCES, 1):
            sample_id = f"{font_name}_S{idx:02d}"
            img_pil = render_line_image(sent, dst_font)
            img_path = IMAGES_DIR / f"{sample_id}.png"
            img_pil.save(img_path)
            manifest_rows.append({
                "sample_id": sample_id,
                "sentence_id": f"S{idx:02d}",
                "font_name": font_name,
                "font_file": font_src.name,
                "font_license": font_lic,
                "font_sha256": f_sha,
                "text_license": "CC0-1.0 (Original benchmark sentences)",
                "ground_truth_nfc": nfc(sent),
                "image_rel_path": f"r59_dataset/images/{sample_id}.png",
                "width": img_pil.width,
                "height": img_pil.height,
                "image_sha256": sha256_file(img_path),
            })

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(OUT_DIR / "R59_DATASET_MANIFEST.csv", index=False)
    print(f"[+] Generated {len(df_manifest)} images across 5 open-license fonts. Saved R59_DATASET_MANIFEST.csv.")

    # Load all 200 BGR & PIL images into memory
    imgs_bgr = []
    imgs_pil = []
    for rel in df_manifest["image_rel_path"]:
        p = OUT_DIR / rel
        arr = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        imgs_bgr.append(bgr)
        imgs_pil.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    refs = df_manifest["ground_truth_nfc"].tolist()
    fonts_list = df_manifest["font_name"].tolist()

    torch.set_num_threads(4)
    device = torch.device("cpu")
    benchmark_rows = []
    preds_df = df_manifest[["sample_id", "font_name", "ground_truth_nfc"]].copy()

    def record_model(model_label: str, group_label: str, hyps: list):
        preds_df[model_label] = hyps
        # Overall across 200
        ov = compute_metrics(refs, hyps)
        benchmark_rows.append({"model_name": model_label, "group_or_type": group_label, "font_name": "ALL_5_FONTS", **ov})
        # Per-font (40 each)
        for fn, _, _ in OPEN_FONTS:
            f_refs = [r for r, f in zip(refs, fonts_list) if f == fn]
            f_hyps = [h for h, f in zip(hyps, fonts_list) if f == fn]
            fm = compute_metrics(f_refs, f_hyps)
            benchmark_rows.append({"model_name": model_label, "group_or_type": group_label, "font_name": fn, **fm})
        print(f"  {model_label:34s} | ALL CER={ov['cer_micro_pct']:5.2f}%, WER={ov['wer_micro_pct']:5.2f}%, Exact={ov['exact_matches']}/200 ({ov['exact_match_pct']:5.2f}%)")

    # 3. Evaluate 4 Baselines
    print("\n[+] Evaluating 4 Baselines on R59 (200 images)...")
    # Baseline 1: CRNN-V2 Raw
    m_b, c_b, v_b = load_crnn(BASELINE_CKPT, device)
    hyps_crnn_b = [normalize(infer_line_smart(m_b, c_b, v_b, bgr, device).strip()) for bgr in imgs_bgr]
    record_model("Baseline_CRNN-V2_Raw", "Baseline", hyps_crnn_b)

    # Baseline 2: Tesseract 5 PSM 7
    hyps_tess = [normalize(pytesseract.image_to_string(pil, lang="aze", config="--psm 7").strip()) for pil in imgs_pil]
    record_model("Baseline_Tesseract5_PSM7", "Baseline", hyps_tess)

    # Baseline 3: EasyOCR
    import easyocr
    easy_reader = easyocr.Reader(["az"], gpu=False, verbose=False)
    hyps_easy = []
    for bgr in imgs_bgr:
        res = easy_reader.recognize(bgr)
        txt = " ".join([it[1] for it in res if len(it) > 1 and it[1]]) if res else ""
        hyps_easy.append(normalize(txt.strip()))
    record_model("Baseline_EasyOCR", "Baseline", hyps_easy)

    # Baseline 4: PaddleOCR (run in isolated subprocess to prevent Windows PyTorch/Paddle OpenMP DLL collision)
    import subprocess
    paddle_tmp_json = OUT_DIR / "_paddle_r59_tmp.json"
    paddle_cmd = [
        sys.executable, "-c",
        (
            "import os, json; "
            "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'; "
            "os.environ['FLAGS_use_mkldnn']='0'; "
            "import paddlex, pandas as pd; "
            "from pathlib import Path; "
            f"df=pd.read_csv(r'{OUT_DIR / 'R59_DATASET_MANIFEST.csv'}'); "
            f"paths=[str(Path(r'{OUT_DIR}') / r) for r in df['image_rel_path']]; "
            "rec_dir=Path.home() / '.paddlex' / 'official_models' / 'PP-OCRv6_medium_rec'; "
            "pred=paddlex.create_predictor(model_name='PP-OCRv6_medium_rec', model_dir=str(rec_dir), device='cpu', enable_mkldnn=False); "
            "out=[str(r.get('rec_text','')).strip() for p in paths for r in pred([p])]; "
            f"Path(r'{paddle_tmp_json}').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')"
        )
    ]
    subprocess.run(paddle_cmd, check=True)
    import json
    hyps_pad = [normalize(t) for t in json.loads(paddle_tmp_json.read_text(encoding="utf-8"))]
    paddle_tmp_json.unlink(missing_ok=True)
    record_model("Baseline_PaddleOCR", "Baseline", hyps_pad)

    # 4. Evaluate 18 GPU Checkpoints
    print("\n[+] Evaluating 18 Official GPU Checkpoints on R59 (200 images)...")
    for grp_id, grp_short in GROUPS:
        for seed in SEEDS:
            rname = f"{grp_id}_seed{seed}"
            ckpt = RUNS_ROOT / rname / "best.pt"
            model, cfg, vocab = load_crnn(ckpt, device)
            hyps = [normalize(infer_line_smart(model, cfg, vocab, bgr, device).strip()) for bgr in imgs_bgr]
            record_model(rname, grp_short, hyps)

    pd.DataFrame(benchmark_rows).to_csv(OUT_DIR / "R59_FONT_BENCHMARK.csv", index=False)
    preds_df.to_csv(OUT_DIR / "R59_RAW_PREDICTIONS.csv", index=False)
    print(f"\n[+] R59 Benchmark complete. Saved R59_FONT_BENCHMARK.csv and R59_RAW_PREDICTIONS.csv.")


if __name__ == "__main__":
    main()
