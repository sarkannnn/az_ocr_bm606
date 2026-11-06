#!/usr/bin/env python3
"""AzOCR-V2: Azərbaycan Dili üçün Çoxdilli (AZ / TR / EN) Elmi-Tədqiqat və Sənaye OCR Platforması.

Rəsmi Akademik və İnstitusional 3 Dilli Arayüz:
- 🇦🇿 Azərbaycan dili
- 🇹🇷 Türkçe
- 🇬🇧 English

Bölmələr:
1. Sənəd və Təsvir Analizi (Normalizasiya zolağı, Xam CTC və Morfoloji korreksiya fərq analizi, Telemetriya).
2. Eksperimental Model Müqayisəsi (CRNN-V2, CRNN-V1 və Tesseract 5 üzrə CER, WER və İcra müddəti cədvəli).
3. Real-Vaxt Kamera Sınaq Modulu (Mobil və veb-kamera üzərindən canlı sınaq).
"""

import argparse
import base64
import html
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure working directory is always the repository root on any PC
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

import cv2
import gradio as gr
import numpy as np
import pytesseract
import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def setup_tesseract_env() -> None:
    """Automatically configure TESSDATA_PREFIX and locate the Tesseract binary across Windows/macOS/Linux."""
    local_tessdata = BASE_DIR / "tessdata"
    if local_tessdata.exists():
        os.environ["TESSDATA_PREFIX"] = str(local_tessdata)

    # Check if tesseract is already on PATH
    tess_bin = shutil.which("tesseract")
    if not tess_bin:
        candidate_dirs = [
            Path(r"C:\Program Files\Tesseract-OCR"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Tesseract-OCR",
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path("/usr/bin"),
        ]
        for d in candidate_dirs:
            exe = d / ("tesseract.exe" if os.name == "nt" else "tesseract")
            if exe.exists():
                tess_bin = str(exe)
                os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + str(d)
                break
    if tess_bin:
        pytesseract.pytesseract.tesseract_cmd = tess_bin


def ensure_gradio_heic_support() -> None:
    """Ensure the local Gradio frontend bundle accepts .heic/.heif drag-and-drop on any freshly installed PC."""
    try:
        assets_dir = Path(gr.__file__).resolve().parent / "templates" / "frontend" / "assets"
        if not assets_dir.exists():
            return
        needle = 'if(n.has("image/*"))'
        patch = 'if(n.has("image/*")){n.add(".heic"),n.add(".heif"),n.add("image/heic"),n.add("image/heif")}if(n.has("image/*"))'
        for js_file in assets_dir.glob("*.js"):
            txt = js_file.read_text(encoding="utf-8", errors="ignore")
            if needle in txt and 'n.add(".heic")' not in txt:
                js_file.write_text(txt.replace(needle, patch), encoding="utf-8")
    except Exception:
        pass


setup_tesseract_env()
ensure_gradio_heic_support()

from common import (
    cer_wer,
    infer_line_smart,
    load_model_checkpoint,
    normalize,
    robust_trim_chunk,
    segment_lines_opencv,
)
from postprocess import AzerbaijaniPostProcessor

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Model definitions (Formal nomenclature)
ACTIVE_LANG = "az"
DEFAULT_MODEL_KEY = "CRNN-V2 Sətir Modeli (32×512, Təklif olunan)"
WORD_V2_MODEL_KEY = "CRNN-V2 Söz Modeli (32×128)"
LINE_V1_MODEL_KEY = "CRNN-V1 Sətir Modeli (32×256, Baza)"
WORD_V1_MODEL_KEY = "CRNN-V1 Söz Modeli (32×128, Baza)"
TESSERACT_MODEL_KEY = "Tesseract 5 LSTM (aze, Sənaye Baseline)"

MODEL_CONFIGS: Dict[str, str] = {
    DEFAULT_MODEL_KEY: "runs/line_v2/best.pt",
    TESSERACT_MODEL_KEY: "tesseract",
    WORD_V2_MODEL_KEY: "runs/word_v2_20261027_180300/best.pt",
    LINE_V1_MODEL_KEY: "runs/line/best.pt",
    WORD_V1_MODEL_KEY: "runs/word/best.pt",
}
LOADED_MODELS = {}

print("[*] Lüğət bazası yüklənir...")
postprocessor = AzerbaijaniPostProcessor()
if postprocessor.loaded:
    print("[*] Azərbaycan dili tezlik lüğəti uğurla yükləndi (az_lexicon.txt).")
else:
    print("[!] Xəbərdarlıq: Lüğət faylı tapılmadı, yalnız xam model çıxışı göstəriləcək.")


def get_model(choice: str):
    """Lazily load and cache the requested PyTorch CRNN model."""
    if choice not in MODEL_CONFIGS or choice == TESSERACT_MODEL_KEY:
        choice = DEFAULT_MODEL_KEY
    if choice not in LOADED_MODELS:
        ckpt_path = MODEL_CONFIGS[choice]
        if not Path(ckpt_path).exists():
            ckpt_path = "runs/line_v2/best.pt" if "Sətir" in choice else "runs/word_v2_20261027_180300/best.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, vocab, device = load_model_checkpoint(ckpt_path, device=device)
        LOADED_MODELS[choice] = (model, cfg, vocab, device)
    return LOADED_MODELS[choice]


print("[*] Standart V2 Sətir modeli yüklənir...")
get_model(DEFAULT_MODEL_KEY)


def get_local_ip() -> str:
    """Get local LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ==============================================================================
# TRILINGUAL LOCALIZATION DICTIONARY (AZ / TR / EN)
# ==============================================================================
LANG_CHOICES = ["🇦🇿 Azərbaycan dili", "🇹🇷 Türkçe", "🇬🇧 English"]
LANG_CODE_MAP = {
    "🇦🇿 Azərbaycan dili": "az",
    "🇹🇷 Türkçe": "tr",
    "🇬🇧 English": "en",
}

I18N = {
    "az": {
        "title": "Azərbaycan Dili üçün Hibrid Optik Mətn Tanıma Sistemi (AzOCR-V2)",
        "sub": "Konvolyusiyalı Rekurrent Neyron Şəbəkəsi (CRNN-BiLSTM-CTC) və Morfoloji Leksikon Əsaslı Post-Prosesinq Mühiti",
        "badge": "Elmi-Tədqiqat Platforması (BM-606)",
        "spec_arch_label": "Arxitektura:",
        "spec_arch_val": "CNN + 2×BiLSTM + CTC (4.05M parametr)",
        "spec_tensor_label": "Giriş Tenzoru:",
        "spec_tensor_val": "32 × 512 piksel (Grayscale)",
        "spec_alpha_label": "Əlifba Həcmi:",
        "spec_alpha_val": "88 simvol (Ç, Ə, Ğ, İ, I, Ö, Ş, Ü daxil)",
        "spec_lex_label": "Leksikon:",
        "spec_lex_val": "116,143 söz + Morfoloji şəkilçi analizi",
        "upload_ph": "# Təsviri buraya sürüşdürün\n- və ya -\nYükləmək üçün klikləyin",
        "tab1": "Bölmə 1: Sənəd və Təsvir Analizi",
        "studio_img": "Giriş Təsviri (Sənəd və ya Sətir Kəsimi)",
        "studio_model": "Tanıma Mühərriki (Model Konfiqurasiyası)",
        "studio_pp": "Morfoloji Leksikon Korreksiyası (SymSpell)",
        "studio_multiline": "Avtomatik Çoxsətirli Seqmentasiya",
        "studio_btn": "Mətni Tanı və Analiz Et",
        "studio_clear": "Təmizlə",
        "studio_preview": "Normalizasiya Olunmuş Tenzor Zolağı (32×W px)",
        "studio_out": "Yekun Tanınmış Mətn (Post-Prosesinq Nəticəsi)",
        "studio_placeholder": "Tanınmış mətn burada əks olunacaq...",
        "diff_title": "MƏRHƏLƏLİ ÇIXIŞ ANALİZİ (CTC VƏ MORFOLOJİ KORREKSİYA)",
        "diff_empty": "Analiz üçün təsvir yükləyin və ya aşağıdakı etalon nümunələrdən birini seçin.",
        "diff_corr_note": "Morfoloji leksikon tərəfindən düzəldilmiş söz sayı: {}",
        "diff_clean_note": "Xam CTC çıxışı lüğət və morfoloji qaydalarla tam uzlaşır (0 dəyişiklik)",
        "diff_stage1": "1. Xam CTC Çıxışı:",
        "diff_stage2": "2. Leksikon Korreksiyası:",
        "tel_title": "HESABLAMA TELEMETRİYASI VƏ STATİSTİK GÖSTƏRİCİLƏR",
        "tel_time": "ÜMUMİ İCRA MÜDDƏTİ",
        "tel_time_sub": "Orta: {:.1f} ms / sətir",
        "tel_seg": "SEQMENTASİYA",
        "tel_seg_sub": "Üfüqi proyeksiya kəsimi",
        "tel_vol": "MƏTN HƏCMİ",
        "tel_vol_sub": "Cəmi: {} simvol",
        "tel_words": "{} söz",
        "tel_lines": "{} sətir",
        "tel_eng": "AKTİV KONFİQURASİYA",
        "tel_eng_sub": "Qurğu: {} | Post-prosesinq: {}",
        "active_on": "Aktiv",
        "active_off": "Deaktiv",
        "examples_label1": "Etalon Sınaq Nümunələri (Seçildikdə avtomatik icra olunur)",
        "tab2": "Bölmə 2: Eksperimental Model Müqayisəsi",
        "bench_desc": "Eyni təsviri paralel olaraq **4 fərqli konfiqurasiyada** (*CRNN-V2 + SymSpell*, *CRNN-V2 Xam CTC*, *CRNN-V1 Baza*, *Tesseract 5 aze*) icra edir. **Etalon Mətn (Ground Truth)** verildikdə **CER (%)**, **WER (%)** və **Simvol Dəqiqliyi (%)** cədvəli qurur.",
        "bench_img": "Müqayisə Ediləcək Təsvir",
        "bench_gt": "Etalon Mətn (Ground Truth — CER və WER hesablanması üçün)",
        "bench_gt_placeholder": "Etalon mətni daxil edin və ya aşağıdakı hazır nümunələrdən birini seçin...",
        "bench_btn": "Bütün Modelləri Paralel İcra Et və Müqayisə Cədvəlini Qur",
        "bench_empty": "Eksperimental müqayisə cədvəlini qurmaq üçün təsvir yükləyin və ya aşağıdakı etalon nümunələrdən birini seçin.",
        "bench_caption": "Cədvəl: Müxtəlif OCR arxitekturalarının seçilmiş təsvir üzrə kəmiyyət və keyfiyyət müqayisəsi",
        "bench_col_model": "Model / Konfiqurasiya",
        "bench_col_hyp": "Tanınmış Mətn (Hipotez)",
        "bench_col_lat": "Müddət (ms)",
        "bench_col_cer": "CER (%) ↓",
        "bench_col_wer": "WER (%) ↓",
        "bench_col_acc": "Dəqiqlik (%) ↑",
        "bench_m_v2_pp": "CRNN-V2 + Morfoloji SymSpell (Təklif olunan)",
        "bench_m_v2_raw": "CRNN-V2 (Xam CTC Dekodlama)",
        "bench_m_v1": "CRNN-V1 (Baza Arxitekturası)",
        "bench_m_tess": "Tesseract 5 LSTM (aze — Sənaye Baseline)",
        "bench_gt_banner": "Etalon Mətn (Ground Truth):",
        "examples_label2": "Etalon Mətnli Eksperimental Nümunələr (Seçildikdə CER/WER cədvəli qurulur)",
        "tab3": "Bölmə 3: Real-Vaxt Kamera Sınaq Modulu",
    },
    "tr": {
        "title": "Azerbaycan Türkçesi için Hibrit Optik Metin Tanıma Sistemi (AzOCR-V2)",
        "sub": "Evrişimli Tekrarlayan Sinir Ağı (CRNN-BiLSTM-CTC) ve Morfolojik Sözlük Tabanlı Son-İşleme Ortamı",
        "badge": "Bilimsel Araştırma Platformu (BM-606)",
        "spec_arch_label": "Mimari:",
        "spec_arch_val": "CNN + 2×BiLSTM + CTC (4.05M parametre)",
        "spec_tensor_label": "Girdi Tensörü:",
        "spec_tensor_val": "32 × 512 piksel (Gri tonlama)",
        "spec_alpha_label": "Alfabe Hacmi:",
        "spec_alpha_val": "88 karakter (Ç, Ə, Ğ, İ, I, Ö, Ş, Ü dahil)",
        "spec_lex_label": "Leksikon:",
        "spec_lex_val": "116,143 kelime + Morfolojik ek analizi",
        "upload_ph": "# Resmi buraya sürükle\n- veya -\nYüklemek için tıkla",
        "tab1": "Bölüm 1: Belge ve Görüntü Analizi",
        "studio_img": "Girdi Görüntüsü (Belge veya Satır Kırpıntısı)",
        "studio_model": "Tanıma Motoru (Model Yapılandırması)",
        "studio_pp": "Morfolojik Sözlük Düzeltmesi (SymSpell)",
        "studio_multiline": "Otomatik Çok Satırlı Segmentasyon",
        "studio_btn": "Metni Tanı ve Analiz Et",
        "studio_clear": "Temizle",
        "studio_preview": "Normalize Edilmiş Tensör Şeridi (32×W px)",
        "studio_out": "Nihai Tanınan Metin (Son-İşleme Sonucu)",
        "studio_placeholder": "Tanınan metin burada görüntülenecek...",
        "diff_title": "AŞAMALI ÇIKTI ANALİZİ (CTC VE MORFOLOJİK DÜZELTME)",
        "diff_empty": "Analiz için görüntü yükleyin veya aşağıdaki referans örneklerden birini seçin.",
        "diff_corr_note": "Morfolojik sözlük tarafından düzeltilen kelime sayısı: {}",
        "diff_clean_note": "Ham CTC çıktısı sözlük ve morfolojik kurallarla tam örtüşmektedir (0 değişiklik)",
        "diff_stage1": "1. Ham CTC Çıktısı:",
        "diff_stage2": "2. Sözlük Düzeltmesi:",
        "tel_title": "HESAPLAMA TELEMETRİSİ VE İSTATİSTİKSEL GÖSTERGELER",
        "tel_time": "TOPLAM İŞLEME SÜRESİ",
        "tel_time_sub": "Ort.: {:.1f} ms / satır",
        "tel_seg": "SEGMENTASYON",
        "tel_seg_sub": "Yatay projeksiyon kırpımı",
        "tel_vol": "METİN HACMİ",
        "tel_vol_sub": "Toplam: {} karakter",
        "tel_words": "{} kelime",
        "tel_lines": "{} satır",
        "tel_eng": "AKTİF YAPILANDIRMA",
        "tel_eng_sub": "Cihaz: {} | Son-işleme: {}",
        "active_on": "Aktif",
        "active_off": "Devre Dışı",
        "examples_label1": "Referans Test Örnekleri (Seçildiğinde otomatik çalıştırılır)",
        "tab2": "Bölüm 2: Deneysel Model Karşılaştırması",
        "bench_desc": "Aynı görüntüyü eşzamanlı olarak **4 farklı yapılandırmada** (*CRNN-V2 + SymSpell*, *CRNN-V2 Ham CTC*, *CRNN-V1 Temel*, *Tesseract 5 aze*) çalıştırır. **Referans Metin (Ground Truth)** girildiğinde **CER (%)**, **WER (%)** ve **Karakter Doğruluğu (%)** tablosu oluşturur.",
        "bench_img": "Karşılaştırılacak Görüntü",
        "bench_gt": "Referans Metin (Ground Truth — CER ve WER hesabı için)",
        "bench_gt_placeholder": "Referans metni girin veya aşağıdaki hazır örneklerden birini seçin...",
        "bench_btn": "Tüm Modelleri Paralel Çalıştır ve Karşılaştırma Tablosunu Oluştur",
        "bench_empty": "Deneysel karşılaştırma tablosunu oluşturmak için görüntü yükleyin veya aşağıdaki referans örneklerden birini seçin.",
        "bench_caption": "Tablo: Farklı OCR mimarilerinin seçilen görüntü üzerindeki nicel ve nitel karşılaştırması",
        "bench_col_model": "Model / Yapılandırma",
        "bench_col_hyp": "Tanınan Metin (Hipotez)",
        "bench_col_lat": "Süre (ms)",
        "bench_col_cer": "CER (%) ↓",
        "bench_col_wer": "WER (%) ↓",
        "bench_col_acc": "Doğruluk (%) ↑",
        "bench_m_v2_pp": "CRNN-V2 + Morfolojik SymSpell (Önerilen)",
        "bench_m_v2_raw": "CRNN-V2 (Ham CTC Çözümleme)",
        "bench_m_v1": "CRNN-V1 (Temel Mimari)",
        "bench_m_tess": "Tesseract 5 LSTM (aze — Endüstriyel Referans)",
        "bench_gt_banner": "Referans Metin (Ground Truth):",
        "examples_label2": "Referans Metinli Deneysel Örnekler (Seçildiğinde CER/WER tablosu oluşturulur)",
        "tab3": "Bölüm 3: Gerçek Zamanlı Kamera Test Modülü",
    },
    "en": {
        "title": "Hybrid Optical Character Recognition System for Azerbaijani (AzOCR-V2)",
        "sub": "Convolutional Recurrent Neural Network (CRNN-BiLSTM-CTC) with Morphological Lexicon-Driven Post-Processing",
        "badge": "Scientific Research Platform (BM-606)",
        "spec_arch_label": "Architecture:",
        "spec_arch_val": "CNN + 2×BiLSTM + CTC (4.05M parameters)",
        "spec_tensor_label": "Input Tensor:",
        "spec_tensor_val": "32 × 512 pixels (Grayscale)",
        "spec_alpha_label": "Alphabet Size:",
        "spec_alpha_val": "88 characters (incl. Ç, Ə, Ğ, İ, I, Ö, Ş, Ü)",
        "spec_lex_label": "Lexicon:",
        "spec_lex_val": "116,143 terms + Morphological suffix analysis",
        "upload_ph": "# Drop Image Here\n- or -\nClick to Upload",
        "tab1": "Section 1: Document & Image Analysis",
        "studio_img": "Input Image (Document Page or Line Crop)",
        "studio_model": "Recognition Engine (Model Configuration)",
        "studio_pp": "Morphological Lexicon Correction (SymSpell)",
        "studio_multiline": "Automatic Multi-line Segmentation",
        "studio_btn": "Recognize & Analyze Text",
        "studio_clear": "Clear",
        "studio_preview": "Normalized Input Tensor Strip (32×W px)",
        "studio_out": "Recognized Text Output (Post-Processed)",
        "studio_placeholder": "Recognized text will appear here...",
        "diff_title": "STAGED PIPELINE DECODING (CTC VS. MORPHOLOGICAL CORRECTION)",
        "diff_empty": "Upload an image or select a benchmark sample below to inspect staged decoding.",
        "diff_corr_note": "Tokens corrected by morphological lexicon: {}",
        "diff_clean_note": "Raw CTC hypothesis strictly conforms to morphological lexicon (0 edits)",
        "diff_stage1": "1. Raw CTC Hypothesis:",
        "diff_stage2": "2. Lexicon Correction:",
        "tel_title": "RUNTIME TELEMETRY & STATISTICAL METRICS",
        "tel_time": "TOTAL LATENCY",
        "tel_time_sub": "Avg: {:.1f} ms / line",
        "tel_seg": "SEGMENTATION",
        "tel_seg_sub": "Horizontal projection profile",
        "tel_vol": "TEXT VOLUME",
        "tel_vol_sub": "Total: {} characters",
        "tel_words": "{} words",
        "tel_lines": "{} lines",
        "tel_eng": "ACTIVE ENGINE",
        "tel_eng_sub": "Device: {} | Post-processing: {}",
        "active_on": "Active",
        "active_off": "Disabled",
        "examples_label1": "Benchmark Test Samples (Click to run evaluation)",
        "tab2": "Section 2: Experimental Model Benchmark",
        "bench_desc": "Evaluates the input image simultaneously across **4 configurations** (*CRNN-V2 + SymSpell*, *CRNN-V2 Raw CTC*, *CRNN-V1 Baseline*, *Tesseract 5 aze*). Generates a formal LaTeX booktabs comparison table with **CER (%)**, **WER (%)**, and **Character Accuracy (%)** when **Ground Truth** is supplied.",
        "bench_img": "Evaluation Image",
        "bench_gt": "Ground Truth Text (For CER and WER error calculation)",
        "bench_gt_placeholder": "Enter reference ground truth text or select from samples below...",
        "bench_btn": "Run Parallel Evaluation & Generate LaTeX Comparison Table",
        "bench_empty": "Upload an image or select a benchmark sample below to generate the comparative evaluation table.",
        "bench_caption": "Table: Quantitative and qualitative evaluation of OCR architectures on test input",
        "bench_col_model": "Model / Configuration",
        "bench_col_hyp": "Recognized Hypothesis",
        "bench_col_lat": "Latency (ms)",
        "bench_col_cer": "CER (%) ↓",
        "bench_col_wer": "WER (%) ↓",
        "bench_col_acc": "Accuracy (%) ↑",
        "bench_m_v2_pp": "CRNN-V2 + Morphological SymSpell (Proposed)",
        "bench_m_v2_raw": "CRNN-V2 (Raw CTC Decoding)",
        "bench_m_v1": "CRNN-V1 (Baseline Architecture)",
        "bench_m_tess": "Tesseract 5 LSTM (aze — Industrial Baseline)",
        "bench_gt_banner": "Ground Truth Reference:",
        "examples_label2": "Benchmark Samples with Ground Truth (Click to build evaluation table)",
        "tab3": "Section 3: Real-Time Live Camera Scanner",
    }
}


def build_top_bar_html(lang: str = "az") -> str:
    """Build the prominent academic journal language selector toolbar."""
    az_act = "active" if lang == "az" else ""
    tr_act = "active" if lang == "tr" else ""
    en_act = "active" if lang == "en" else ""

    return f"""
    <div class="az-academic-topbar" id="az-topbar-container">
        <div class="az-lang-bar-row">
            <span class="az-lang-heading">🌐 DİL / DİL / LANGUAGE:</span>
            <div class="az-lang-button-group">
                <button type="button" class="az-lang-btn {az_act}" id="btn-lang-az" onclick="window.azSwitchLang('az')">
                    <span class="az-flag">🇦🇿</span>
                    <span class="az-btn-text">Azərbaycan dili</span>
                </button>
                <button type="button" class="az-lang-btn {tr_act}" id="btn-lang-tr" onclick="window.azSwitchLang('tr')">
                    <span class="az-flag">🇹🇷</span>
                    <span class="az-btn-text">Türkçe</span>
                </button>
                <button type="button" class="az-lang-btn {en_act}" id="btn-lang-en" onclick="window.azSwitchLang('en')">
                    <span class="az-flag">🇬🇧</span>
                    <span class="az-btn-text">English</span>
                </button>
            </div>
        </div>
    </div>
    """


SHARE_URL = ""
LAN_URL = f"http://{get_local_ip()}:7860"


def build_qr_html() -> str:
    """Inline-SVG QR code (public Gradio link if available, else LAN link) for opening the app on a phone."""
    url = SHARE_URL or LAN_URL
    if not url:
        return ""
    try:
        import io
        import segno
        buf = io.BytesIO()
        segno.make(url, error="m").save(buf, kind="svg", scale=4, border=2, dark="#1b365d", xmldecl=False, svgns=True)
        svg = buf.getvalue().decode("utf-8")
    except Exception as exc:
        svg = f"<em>QR error: {html.escape(str(exc))}</em>"
    kind = "İctimai Gradio Linki (Public HTTPS)" if SHARE_URL else "Yerli Şəbəkə (Wi-Fi LAN)"
    return f"""
    <details class="az-qr-box" id="az-qr-box" open>
        <summary>📱 Telefonda Aç / Open on Phone (QR Kod)</summary>
        <div class="az-qr-inner">
            <div class="az-qr-svg">{svg}</div>
            <div class="az-qr-text">
                <div style="font-weight:700; color:#1b365d; margin-bottom:4px;">{kind}</div>
                <div style="font-size:12px; color:#475569; margin-bottom:6px;">Telefon kamerası ilə QR kodu oxudun və ya linkə toxunun:</div>
                <a href="{html.escape(url)}" target="_blank" rel="noopener" style="font-weight:700; color:#1b365d; text-decoration:underline;">{html.escape(url)}</a>
            </div>
        </div>
    </details>
    """


def build_hero_html(lang: str = "az") -> str:
    """Build the formal institutional header in the selected language."""
    T = I18N.get(lang, I18N["az"])
    return f"""
    <div class="az-official-header" id="az-hero-container">
        <div class="az-official-top">
            <div>
                <h1 class="az-official-title" id="az-hero-title">{html.escape(T['title'])}</h1>
                <p class="az-official-sub" id="az-hero-sub">{html.escape(T['sub'])}</p>
            </div>
        </div>
        <table class="az-spec-table">
            <tr>
                <td id="az-spec-arch"><strong>{html.escape(T['spec_arch_label'])}</strong> {html.escape(T['spec_arch_val'])}</td>
                <td id="az-spec-tensor"><strong>{html.escape(T['spec_tensor_label'])}</strong> {html.escape(T['spec_tensor_val'])}</td>
                <td id="az-spec-alpha"><strong>{html.escape(T['spec_alpha_label'])}</strong> {html.escape(T['spec_alpha_val'])}</td>
                <td id="az-spec-lex"><strong>{html.escape(T['spec_lex_label'])}</strong> {html.escape(T['spec_lex_val'])}</td>
            </tr>
        </table>
    </div>
    """


def extract_tight_text_lines(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Ağ boşluqları kənarlaşdırıb mürəkkəb olan sətirləri sıx kəsimlə çıxarır."""
    if img_bgr is None or img_bgr.size == 0:
        return []
    H, W = img_bgr.shape[:2]
    if H < 18 or W < 18:
        return [img_bgr]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    sigma = max(3.0, min(25.0, H / 4.0))
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    thresh = 14
    ink = diff > thresh
    row_counts = ink.sum(axis=1)

    thresh_pixels = max(5, int(W * 0.02))
    active_rows = row_counts >= thresh_pixels

    bands = []
    in_b = False
    start_y = 0
    for y, val in enumerate(active_rows):
        if val and not in_b:
            in_b = True
            start_y = y
        elif not val and in_b:
            in_b = False
            if (y - start_y) >= 8:
                bands.append((start_y, y))
    if in_b and (H - start_y) >= 8:
        bands.append((start_y, H))

    if not bands:
        return [img_bgr]

    crops = []
    for y0, y1 in bands:
        pad = max(2, int((y1 - y0) * 0.15))
        cy0 = max(0, y0 - pad)
        cy1 = min(H, y1 + pad)
        line_strip = img_bgr[cy0:cy1, :]

        strip_gray = gray[cy0:cy1, :]
        strip_bg = bg[cy0:cy1, :]
        strip_ink = (strip_bg.astype(np.int16) - strip_gray.astype(np.int16)) > thresh
        valid_cols = np.where(strip_ink.sum(axis=0) > 0)[0]
        if len(valid_cols) > 0:
            cx0 = max(0, int(valid_cols.min()) - 4)
            cx1 = min(W, int(valid_cols.max()) + 5)
            if cx1 > cx0 + 5:
                line_strip = line_strip[:, cx0:cx1]
        crops.append(line_strip)
    return crops


def run_tesseract_ocr(img_bgr: np.ndarray, multi_line: bool = True) -> str:
    """Tesseract baseline OCR execution on BGR image."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if multi_line and img_bgr.shape[0] > 90:
        try:
            txt = pytesseract.image_to_string(pil_img, lang="aze", config="--psm 6").strip()
            return normalize(txt)
        except Exception as e:
            return f"Tesseract xətası: {e}"

    lines = extract_tight_text_lines(img_bgr)
    out_lines = []
    for line in lines:
        r = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
        try:
            t = pytesseract.image_to_string(Image.fromarray(r), lang="aze", config="--psm 7").strip()
            if t:
                out_lines.append(normalize(t))
        except Exception as e:
            out_lines.append(f"Tesseract xətası: {e}")
    return "\n".join(out_lines)


def run_ocr_on_bgr(
    img_bgr: np.ndarray,
    model_choice: str = DEFAULT_MODEL_KEY,
    use_postprocess: bool = True,
    multi_line: bool = True,
) -> Tuple[str, str, float, int]:
    """Run smart line/multi-line OCR on a BGR numpy image."""
    start_time = time.perf_counter()

    # Böyük meqapikselli tam səhifə şəkilləri optimal 1800px ölçüyə sıxılır (yüksək sürət üçün)
    h_orig, w_orig = img_bgr.shape[:2]
    if max(h_orig, w_orig) > 1800 and min(h_orig, w_orig) > 400:
        scale = 1800.0 / max(h_orig, w_orig)
        img_bgr = cv2.resize(img_bgr, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # Hündür (səhifə formatlı) təsvirlər üçün seqmentasiya avtomatik aktivləşir
    h_cur, w_cur = img_bgr.shape[:2]
    if h_cur > 150 and (w_cur / max(1, h_cur)) < 4.0:
        multi_line = True

    if multi_line and img_bgr.shape[0] > 90:
        line_crops = segment_lines_opencv(img_bgr)
    else:
        line_crops = extract_tight_text_lines(img_bgr)

    if model_choice == TESSERACT_MODEL_KEY:
        tess_text = run_tesseract_ocr(img_bgr, multi_line=multi_line)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        n_lines = len([l for l in tess_text.splitlines() if l.strip()]) or 1
        final_text = tess_text
        if use_postprocess and postprocessor and postprocessor.loaded:
            lines = []
            for l in tess_text.splitlines():
                toks = [postprocessor.correct_word(t, max_edit_distance=1) for t in l.split()]
                lines.append(postprocessor.clean_punctuation(" ".join(toks)))
            final_text = "\n".join(lines)
        return final_text, tess_text, elapsed_ms, n_lines

    model, cfg, vocab, device = get_model(model_choice)

    raw_lines = []
    final_lines = []
    for lc in line_crops:
        rp = infer_line_smart(model, cfg, vocab, lc, device)
        if not rp:
            continue
        if use_postprocess and postprocessor and postprocessor.loaded:
            tokens = rp.split()
            # Çoxsətirli rejimdə lüğətə uyğun söz payı çox aşağı olan xəta/fon sətirlərini at
            if len(line_crops) > 3 and len(tokens) >= 4:
                ok = sum(1 for t in tokens if postprocessor.is_valid_morphologically(t.strip('.,;:!?()"-–').lower()))
                if ok / len(tokens) < 0.2:
                    continue
            raw_lines.append(rp)
            corrected_tokens = [postprocessor.correct_word(t, max_edit_distance=1) for t in tokens]
            final_lines.append(postprocessor.clean_punctuation(" ".join(corrected_tokens)))
        else:
            raw_lines.append(rp)
            final_lines.append(rp)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    raw_pred = "\n".join(raw_lines)
    final_pred = "\n".join(final_lines)
    return final_pred, raw_pred, elapsed_ms, len(line_crops)


def make_preprocessed_preview(img_bgr: np.ndarray, multi_line: bool = True) -> Optional[np.ndarray]:
    """Create a visual preview of the normalized grayscale tensor strip seen by the model."""
    try:
        crops = segment_lines_opencv(img_bgr) if (multi_line and img_bgr.shape[0] > 90) else extract_tight_text_lines(img_bgr)
        if not crops:
            crops = [img_bgr]
        first_crop = robust_trim_chunk(crops[0])
        gray = cv2.cvtColor(first_crop, cv2.COLOR_BGR2GRAY) if first_crop.ndim == 3 else first_crop
        h, w = gray.shape[:2]
        target_h = 48
        new_w = max(32, min(900, int(round(w * target_h / max(1, h)))))
        resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb
    except Exception:
        return None


def build_diff_html(raw_text: str, final_text: str, lang: str = "az") -> str:
    """Build a formal academic comparison panel between Raw CTC and Post-Processed output."""
    T = I18N.get(lang, I18N["az"])
    if not raw_text.strip() and not final_text.strip():
        return f"""
        <div class="az-formal-panel">
            <div class="az-panel-header">{html.escape(T['diff_title'])}</div>
            <div class="az-diff-empty" style="color:#555555; font-size:13px; font-style:italic; padding:6px 0;">
                {html.escape(T['diff_empty'])}
            </div>
        </div>
        """
    raw_words = raw_text.split()
    fin_words = final_text.split()
    corrections = []
    out_tokens = []
    if len(raw_words) == len(fin_words):
        for rw, fw in zip(raw_words, fin_words):
            if rw != fw:
                corrections.append((rw, fw))
                out_tokens.append(
                    f'<span class="az-diff-del">{html.escape(rw)}</span> '
                    f'<span class="az-diff-ins">{html.escape(fw)}</span>'
                )
            else:
                out_tokens.append(html.escape(fw))
        diff_body = " ".join(out_tokens)
    else:
        diff_body = html.escape(final_text)

    if corrections:
        status_note = T['diff_corr_note'].format(len(corrections))
    else:
        status_note = T['diff_clean_note']

    return f"""
    <div class="az-formal-panel">
        <div class="az-panel-header" style="display:flex; justify-content:space-between; align-items:center;">
            <span>{html.escape(T['diff_title'])}</span>
            <span style="font-weight:600; font-size:12px; color:#1b365d;">{html.escape(status_note)}</span>
        </div>
        <table class="az-stage-table">
            <tr>
                <td class="az-stage-key">{html.escape(T['diff_stage1'])}</td>
                <td class="az-stage-val" style="font-family:'Consolas', 'Courier New', monospace; font-size:13.5px;">{html.escape(raw_text)}</td>
            </tr>
            <tr>
                <td class="az-stage-key">{html.escape(T['diff_stage2'])}</td>
                <td class="az-stage-val" style="font-size:14px;">{diff_body}</td>
            </tr>
        </table>
    </div>
    """


def build_telemetry_html(elapsed_ms: float, n_lines: int, final_text: str, model_choice: str, use_postprocess: bool, lang: str = "az") -> str:
    """Build a formal tabular telemetry summary in the selected language."""
    T = I18N.get(lang, I18N["az"])
    device_name = "CPU"
    if model_choice != TESSERACT_MODEL_KEY:
        _, _, _, device = get_model(model_choice)
        device_name = device.type.upper()

    n_chars = len(final_text.strip())
    n_words = len(final_text.split())
    engine_short = "Tesseract 5 (aze)" if model_choice == TESSERACT_MODEL_KEY else model_choice.split("(")[0].strip()
    if lang == "en":
        engine_short = engine_short.replace("Sətir Modeli", "Line Model").replace("Söz Modeli", "Word Model")
    elif lang == "tr":
        engine_short = engine_short.replace("Sətir Modeli", "Satır Modeli").replace("Söz Modeli", "Kelime Modeli")
    ms_per_line = elapsed_ms / max(1, n_lines)
    pp_status = T['active_on'] if use_postprocess else T['active_off']

    return f"""
    <div class="az-formal-panel" style="margin-top:8px;">
        <div class="az-panel-header">{html.escape(T['tel_title'])}</div>
        <div class="az-metrics-grid">
            <div class="az-metric-box">
                <div class="az-metric-label">{html.escape(T['tel_time'])}</div>
                <div class="az-metric-val">{elapsed_ms:.1f} ms</div>
                <div class="az-metric-sub">{T['tel_time_sub'].format(ms_per_line)}</div>
            </div>
            <div class="az-metric-box">
                <div class="az-metric-label">{html.escape(T['tel_seg'])}</div>
                <div class="az-metric-val">{T['tel_lines'].format(n_lines)}</div>
                <div class="az-metric-sub">{html.escape(T['tel_seg_sub'])}</div>
            </div>
            <div class="az-metric-box">
                <div class="az-metric-label">{html.escape(T['tel_vol'])}</div>
                <div class="az-metric-val">{T['tel_words'].format(n_words)}</div>
                <div class="az-metric-sub">{T['tel_vol_sub'].format(n_chars)}</div>
            </div>
            <div class="az-metric-box">
                <div class="az-metric-label">{html.escape(T['tel_eng'])}</div>
                <div class="az-metric-val" style="font-size:14px;">{html.escape(engine_short)}</div>
                <div class="az-metric-sub">{T['tel_eng_sub'].format(device_name, pp_status)}</div>
            </div>
        </div>
    </div>
    """


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def run_studio_example(img: Optional[np.ndarray], txt: str = "") -> Tuple[str, str, str, Optional[np.ndarray]]:
    if img is None:
        return "", "", "", None
    return ocr_studio_recognize(
        img,
        model_choice=DEFAULT_MODEL_KEY,
        use_postprocess=True,
        multi_line=True,
        lang=ACTIVE_LANG,
    )


def ocr_studio_recognize(
    img: Optional[np.ndarray],
    model_choice: str,
    use_postprocess: bool,
    multi_line: bool,
    lang: str = "az",
) -> Tuple[str, str, str, Optional[np.ndarray]]:
    """Main callback for Section 1 (Document & Image Analysis)."""
    T = I18N.get(lang, I18N["az"])
    if img is None:
        empty_msg = T['diff_empty']
        return empty_msg, build_diff_html("", "", lang), build_telemetry_html(0.0, 0, "", model_choice, use_postprocess, lang), None

    img_bgr = to_bgr(img)
    final_pred, raw_pred, elapsed_ms, n_lines = run_ocr_on_bgr(
        img_bgr,
        model_choice=model_choice,
        use_postprocess=use_postprocess,
        multi_line=multi_line,
    )
    diff_html = build_diff_html(raw_pred, final_pred, lang)
    telemetry_html = build_telemetry_html(elapsed_ms, n_lines, final_pred, model_choice, use_postprocess, lang)
    preview_rgb = make_preprocessed_preview(img_bgr, multi_line=multi_line)
    return final_pred, diff_html, telemetry_html, preview_rgb


def run_benchmark_comparison(img: Optional[np.ndarray], ground_truth: str, lang: Optional[str] = None) -> str:
    """Callback for Section 2: Formal Booktabs-style Academic Model Comparison."""
    global ACTIVE_LANG
    if not lang or lang not in I18N:
        lang = ACTIVE_LANG if (ACTIVE_LANG and ACTIVE_LANG in I18N) else "az"
    T = I18N.get(lang, I18N["az"])
    if img is None:
        return f"""
        <div class="az-formal-panel az-bench-empty" style="padding:18px; text-align:center; color:#444444;">
            {html.escape(T['bench_empty'])}
        </div>
        """

    img_bgr = to_bgr(img)
    gt = normalize(ground_truth.strip()) if ground_truth and ground_truth.strip() else ""

    configs = [
        (T['bench_m_v2_pp'], DEFAULT_MODEL_KEY, True, True),
        (T['bench_m_v2_raw'], DEFAULT_MODEL_KEY, False, False),
        (T['bench_m_v1'], LINE_V1_MODEL_KEY, False, False),
        (T['bench_m_tess'], TESSERACT_MODEL_KEY, False, False),
    ]

    rows_html = []
    empty_note = "(Empty output)" if lang == "en" else ("(Boş çıktı)" if lang == "tr" else "(Boş çıxış)")
    for i, (label, m_key, use_pp, is_featured) in enumerate(configs):
        pred, raw_p, ms, n_l = run_ocr_on_bgr(img_bgr, model_choice=m_key, use_postprocess=use_pp, multi_line=True)
        pred_clean = " ".join(pred.split())

        if gt:
            cer_val, wer_val, _ = cer_wer([gt], [pred_clean])
            cer_pct = cer_val * 100.0
            wer_pct = wer_val * 100.0
            char_acc = max(0.0, 100.0 - cer_pct)
            cer_str = f"{cer_pct:.2f}%"
            wer_str = f"{wer_pct:.2f}%"
            acc_str = f"<strong>{char_acc:.2f}%</strong>" if is_featured else f"{char_acc:.2f}%"
        else:
            cer_str = "—"
            wer_str = "—"
            acc_str = "—"

        row_class = 'class="az-row-featured"' if is_featured else ""
        rows_html.append(f"""
        <tr {row_class} data-row-idx="{i}">
            <td class="az-model-cell" style="font-weight:{'700' if is_featured else '500'}; white-space:nowrap;">{html.escape(label)}</td>
            <td style="font-size:13.5px;">{html.escape(pred_clean) if pred_clean else f'<em>{empty_note}</em>'}</td>
            <td style="text-align:right; font-family:'Consolas', monospace;">{ms:.1f}</td>
            <td style="text-align:right; font-family:'Consolas', monospace;">{cer_str}</td>
            <td style="text-align:right; font-family:'Consolas', monospace;">{wer_str}</td>
            <td style="text-align:right; font-family:'Consolas', monospace;">{acc_str}</td>
        </tr>
        """)

    gt_banner = ""
    if gt:
        gt_banner = f"""
        <div class="az-gt-box">
            <strong class="az-gt-label">{html.escape(T['bench_gt_banner'])}</strong> {html.escape(gt)}
        </div>
        """

    return f"""
    <div class="az-bench-wrap">
        {gt_banner}
        <table class="az-booktabs-table">
            <caption>{html.escape(T['bench_caption'])}</caption>
            <thead>
                <tr>
                    <th>{html.escape(T['bench_col_model'])}</th>
                    <th>{html.escape(T['bench_col_hyp'])}</th>
                    <th style="text-align:right;">{html.escape(T['bench_col_lat'])}</th>
                    <th style="text-align:right;">{html.escape(T['bench_col_cer'])}</th>
                    <th style="text-align:right;">{html.escape(T['bench_col_wer'])}</th>
                    <th style="text-align:right;">{html.escape(T['bench_col_acc'])}</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows_html)}
            </tbody>
        </table>
    </div>
    """


# ==============================================================================
# HTML5 LIVE CAMERA SCANNER COMPONENT (SECTION 3)
# ==============================================================================
LIVE_CAMERA_HTML = """
<div id="az-cam-container" style="background: #ffffff; color: #111827; border: 1px solid #9ca3af; padding: 16px; max-width: 720px; margin: 0 auto;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #d1d5db; flex-wrap: wrap; gap: 8px;">
    <strong id="az-ui-cam-title" style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.4px; color: #111827;">
      Real-Vaxt Kamera Sınaq Modulu
    </strong>
    <div style="display: flex; gap: 4px; flex-wrap: wrap;">
      <button type="button" onclick="window.azSetBoxMode('line')" id="btn-mode-line" style="background: #1b365d; color: #fff; border: 1px solid #1b365d; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Tək Sətir</button>
      <button type="button" onclick="window.azSetBoxMode('para')" id="btn-mode-para" style="background: #f3f4f6; color: #1f2937; border: 1px solid #9ca3af; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Paraqraf</button>
      <button type="button" onclick="window.azSetBoxMode('full')" id="btn-mode-full" style="background: #f3f4f6; color: #1f2937; border: 1px solid #9ca3af; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Tam Səhifə</button>
    </div>
  </div>

  <!-- OCR Model Engine Switcher -->
  <div class="az-cam-engine-bar" style="display: flex; align-items: center; justify-content: space-between; background: #f9fafb; padding: 6px 10px; margin-bottom: 10px; border: 1px solid #d1d5db; flex-wrap: wrap; gap: 6px;">
    <span id="az-ui-cam-engine-label" style="font-size: 12px; color: #374151; font-weight: 600;">Aktiv Mühərrik:</span>
    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
      <button type="button" onclick="window.azSetEngine('crnn_v2')" id="btn-eng-crnn_v2" style="background: #1b365d; color: #fff; border: 1px solid #1b365d; padding: 4px 12px; font-size: 12px; font-weight: 600; cursor: pointer;">CRNN-V2 (Təklif olunan)</button>
      <button type="button" onclick="window.azSetEngine('tesseract')" id="btn-eng-tesseract" style="background: #f3f4f6; color: #1f2937; border: 1px solid #9ca3af; padding: 4px 12px; font-size: 12px; font-weight: 600; cursor: pointer;">Tesseract 5 (aze)</button>
    </div>
  </div>

  <!-- Video Viewport -->
  <div id="az-video-stage" style="position: relative; width: 100%; background: #000; overflow: hidden; min-height: 250px; max-height: 52vh; display: flex; align-items: center; justify-content: center; border: 1px solid #374151;">
    <video id="az-cam-video" autoplay playsinline muted style="width: 100%; height: 100%; max-height: 52vh; object-fit: cover; display: block;"></video>

    <div id="az-scan-box" style="position: absolute; left: 4%; right: 4%; top: 40%; height: 18%; border: 2px solid #22c55e; box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.50); pointer-events: none; transition: all 0.2s ease;">
      <div style="position: absolute; left: 0; right: 0; top: 50%; height: 1px; background: rgba(239, 68, 68, 0.85);"></div>
      <div id="az-ui-cam-guide" style="position: absolute; top: -22px; left: 0; background: #1b365d; color: #ffffff; font-size: 10.5px; font-weight: 600; padding: 2px 8px;">
        OXUMA ZONASI (MƏTNİ MƏRKƏZİ XƏTTƏ GƏTİRİN)
      </div>
    </div>

    <!-- Fallback / Permission Prompt -->
    <div id="az-cam-msg" style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(17, 24, 39, 0.90); padding: 20px; text-align: center; z-index: 5;">
      <p id="az-cam-msg-text" style="margin: 0 0 12px 0; font-size: 13.5px; color: #f3f4f6;">Kamera axını gözlənilir...</p>
      <button type="button" id="az-ui-cam-start-btn" onclick="window.azStartCamera()" style="background: #1b365d; color: #ffffff; border: 1px solid #ffffff; padding: 8px 18px; font-size: 13px; font-weight: 600; cursor: pointer;">
        Kameranı Aktivləşdir
      </button>
    </div>
  </div>

  <!-- Camera Action Buttons -->
  <div class="az-cam-actions-grid" style="display: grid; grid-template-columns: 2fr 1.4fr auto auto; gap: 8px; margin-top: 10px;">
    <button type="button" onclick="window.azSnapNow()" id="az-btn-snap" style="background: #1b365d; color: #ffffff; border: 1px solid #0f2342; padding: 11px 12px; font-size: 14px; font-weight: 700; cursor: pointer;">
      Kadrı Çək və Oxu
    </button>
    <button type="button" onclick="window.azToggleAuto()" id="az-btn-auto" style="background: #f3f4f6; color: #111827; border: 1px solid #9ca3af; padding: 11px 10px; font-size: 13px; font-weight: 600; cursor: pointer;">
      Avto-Oxu: Deaktiv
    </button>
    <button type="button" onclick="window.azSwitchCam()" id="az-btn-flip" title="Arxa/Ön Kamera" style="background: #f3f4f6; color: #111827; border: 1px solid #9ca3af; padding: 10px 12px; font-size: 12px; font-weight: 600; cursor: pointer;">
      Kamera Dəyiş
    </button>
    <button type="button" onclick="window.azToggleTorch()" id="az-btn-torch" title="Fənər" style="background: #f3f4f6; color: #111827; border: 1px solid #9ca3af; padding: 10px 12px; font-size: 12px; font-weight: 600; cursor: pointer;">
      İşıq
    </button>
  </div>

  <!-- Native Phone Camera Shutter -->
  <div style="margin-top: 8px;">
    <label id="az-ui-cam-native" style="display: block; background: #f9fafb; color: #111827; border: 1px solid #9ca3af; padding: 9px 14px; font-size: 13px; font-weight: 600; cursor: pointer; text-align: center;">
      Cihazın Yüksək Çözünürlüklü Kamerası ilə Şəkil Çək (Avtofokus / Makro)
      <input type="file" accept="image/*" capture="environment" onchange="window.azHandleNativeFile(this)" style="display: none;" />
    </label>
  </div>

  <!-- Live Result Display -->
  <div style="margin-top: 10px; background: #f9fafb; padding: 12px; border: 1px solid #d1d5db;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
      <span id="az-live-status" style="font-size: 12px; color: #4b5563; font-weight: 600;">Status: Hazır</span>
      <button type="button" onclick="window.azCopyResult()" id="az-btn-copy" style="background: #ffffff; color: #111827; border: 1px solid #9ca3af; padding: 3px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">
        Mətni Kopyala
      </button>
    </div>
    <textarea id="az-live-result" rows="3" placeholder="Tanınmış mətn burada əks olunacaq..." style="width: 100%; background: #ffffff; color: #111827; border: 1px solid #9ca3af; padding: 8px; font-size: 15px; font-family: 'Times New Roman', Georgia, serif; line-height: 1.45; resize: vertical; box-sizing: border-box;"></textarea>
    <div style="margin-top: 8px;">
      <div id="az-ui-cam-preview-label" style="font-size: 11px; color: #4b5563; margin-bottom: 3px;">Modelə ötürülən kadr kəsimi:</div>
      <canvas id="az-crop-preview" style="height: 44px; max-width: 100%; border: 1px solid #9ca3af; background: #f3f4f6; display: block;"></canvas>
    </div>
  </div>
</div>
"""

CUSTOM_HEAD_SCRIPT = """
<script>
window.azState = {
  stream: null,
  facingMode: "environment",
  boxMode: "line",
  engine: "crnn_v2",
  autoTimer: null,
  busy: false,
  torchOn: false,
  lang: "az"
};

window.azCamStrings = {
  "az": {
    title: "Real-Vaxt Kamera Sınaq Modulu",
    mode_line: "Tək Sətir",
    mode_para: "Paraqraf",
    mode_full: "Tam Səhifə",
    engine_label: "Aktiv Mühərrik:",
    engine_v2: "CRNN-V2 (Təklif olunan)",
    engine_tess: "Tesseract 5 (aze)",
    guide: "OXUMA ZONASI (MƏTNİ MƏRKƏZİ XƏTTƏ GƏTİRİN)",
    waiting: "Kamera axını gözlənilir...",
    start_btn: "Kameranı Aktivləşdir",
    snap: "Kadrı Çək və Oxu",
    auto_off: "Avto-Oxu: Deaktiv",
    auto_on: "Avto-Oxu: Aktiv",
    flip: "Kamera Dəyiş",
    torch: "İşıq",
    native: "Cihazın Yüksək Çözünürlüklü Kamerası ilə Şəkil Çək (Avtofokus / Makro)",
    status_ready: "Status: Hazır",
    status_proc: "Status: Emal edilir...",
    status_done: "Status: Tamamlandı — ",
    status_notfound: "(Mətn aşkar edilmədi)",
    copy: "Mətni Kopyala",
    copied: "Kopyalandı",
    preview_label: "Modelə ötürülən kadr kəsimi:",
    placeholder: "Tanınmış mətn burada əks olunacaq..."
  },
  "tr": {
    title: "Gerçek Zamanlı Kamera Test Modülü",
    mode_line: "Tek Satır",
    mode_para: "Paragraf",
    mode_full: "Tam Sayfa",
    engine_label: "Aktif Motor:",
    engine_v2: "CRNN-V2 (Önerilen)",
    engine_tess: "Tesseract 5 (aze)",
    guide: "OKUMA ALANI (METNİ MERKEZ ÇİZGİYE GETİRİN)",
    waiting: "Kamera akışı bekleniyor...",
    start_btn: "Kamerayı Etkinleştir",
    snap: "Kareyi Çek ve Oku",
    auto_off: "Oto-Oku: Kapalı",
    auto_on: "Oto-Oku: Açık",
    flip: "Kamera Değiştir",
    torch: "Flaş",
    native: "Cihazın Yüksek Çözünürlüklü Kamerası ile Fotoğraf Çek (Otomatik Odak / Makro)",
    status_ready: "Durum: Hazır",
    status_proc: "Durum: İşleniyor...",
    status_done: "Durum: Tamamlandı — ",
    status_notfound: "(Metin algılanamadı)",
    copy: "Metni Kopyala",
    copied: "Kopyalandı",
    preview_label: "Modele aktarılan kare kırpıntısı:",
    placeholder: "Tanınan metin burada görüntülenecek..."
  },
  "en": {
    title: "Real-Time Camera Scanner Module",
    mode_line: "Single Line",
    mode_para: "Paragraph",
    mode_full: "Full Page",
    engine_label: "Active Engine:",
    engine_v2: "CRNN-V2 (Proposed)",
    engine_tess: "Tesseract 5 (aze)",
    guide: "SCAN REGION (ALIGN TEXT WITH CENTER GUIDELINE)",
    waiting: "Waiting for camera video stream...",
    start_btn: "Enable Camera",
    snap: "Capture & Read",
    auto_off: "Auto-Scan: OFF",
    auto_on: "Auto-Scan: ON",
    flip: "Flip Camera",
    torch: "Torch",
    native: "Capture High-Resolution Document Photo via Native Camera (Autofocus / Macro)",
    status_ready: "Status: Ready",
    status_proc: "Status: Processing...",
    status_done: "Status: Complete — ",
    status_notfound: "(No text detected)",
    copy: "Copy Text",
    copied: "Copied",
    preview_label: "Frame crop passed to model:",
    placeholder: "Recognized text will appear here..."
  }
};

window.azSetUiLang = function(lang) {
  window.azState.lang = lang;
  var S = window.azCamStrings[lang] || window.azCamStrings.az;

  var elTitle = document.getElementById("az-ui-cam-title");
  if (elTitle) elTitle.innerText = S.title;

  var bLine = document.getElementById("btn-mode-line");
  if (bLine) bLine.innerText = S.mode_line;
  var bPara = document.getElementById("btn-mode-para");
  if (bPara) bPara.innerText = S.mode_para;
  var bFull = document.getElementById("btn-mode-full");
  if (bFull) bFull.innerText = S.mode_full;

  var elEngLbl = document.getElementById("az-ui-cam-engine-label");
  if (elEngLbl) elEngLbl.innerText = S.engine_label;

  var bV2 = document.getElementById("btn-eng-crnn_v2");
  if (bV2) bV2.innerText = S.engine_v2;
  var bTess = document.getElementById("btn-eng-tesseract");
  if (bTess) bTess.innerText = S.engine_tess;

  var elGuide = document.getElementById("az-ui-cam-guide");
  if (elGuide) elGuide.innerText = S.guide;

  var elStart = document.getElementById("az-ui-cam-start-btn");
  if (elStart) elStart.innerText = S.start_btn;

  var elSnap = document.getElementById("az-btn-snap");
  if (elSnap) elSnap.innerText = S.snap;

  var elAuto = document.getElementById("az-btn-auto");
  if (elAuto) {
    elAuto.innerText = window.azState.autoTimer ? S.auto_on : S.auto_off;
  }

  var elFlip = document.getElementById("az-btn-flip");
  if (elFlip) elFlip.innerText = S.flip;

  var elTorch = document.getElementById("az-btn-torch");
  if (elTorch) elTorch.innerText = S.torch;

  var elNative = document.getElementById("az-ui-cam-native");
  if (elNative) {
    elNative.childNodes[0].nodeValue = S.native + " ";
  }

  var elCopy = document.getElementById("az-btn-copy");
  if (elCopy) elCopy.innerText = S.copy;

  var elPreviewLbl = document.getElementById("az-ui-cam-preview-label");
  if (elPreviewLbl) elPreviewLbl.innerText = S.preview_label;

  var elRes = document.getElementById("az-live-result");
  if (elRes) elRes.placeholder = S.placeholder;
};

window.azSetBoxMode = function(mode) {
  window.azState.boxMode = mode;
  var box = document.getElementById("az-scan-box");
  if (!box) return;
  var modes = {
    line: { top: "40%", height: "18%" },
    para: { top: "24%", height: "50%" },
    full: { top: "6%",  height: "88%" }
  };
  var m = modes[mode] || modes.line;
  box.style.top = m.top;
  box.style.height = m.height;
  ["line", "para", "full"].forEach(function(k) {
    var b = document.getElementById("btn-mode-" + k);
    if (b) {
      b.style.background = (k === mode) ? "#1b365d" : "#f3f4f6";
      b.style.color = (k === mode) ? "#fff" : "#1f2937";
    }
  });
};

window.azSetEngine = function(eng) {
  window.azState.engine = eng;
  ["crnn_v2", "tesseract"].forEach(function(k) {
    var b = document.getElementById("btn-eng-" + k);
    if (b) {
      b.style.background = (k === eng) ? "#1b365d" : "#f3f4f6";
      b.style.color = (k === eng) ? "#fff" : "#1f2937";
    }
  });
};

window.azStartCamera = async function() {
  var video = document.getElementById("az-cam-video");
  var msgBox = document.getElementById("az-cam-msg");
  var msgText = document.getElementById("az-cam-msg-text");
  if (!video) return;

  if (window.azState.stream) {
    window.azState.stream.getTracks().forEach(function(t) { t.stop(); });
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (msgBox) msgBox.style.display = "flex";
    if (msgText) msgText.innerText = "Brauzer kameranı dəstəkləmir və ya HTTPS bağlantısı tələb olunur.";
    return;
  }

  try {
    var constraints = {
      audio: false,
      video: {
        facingMode: { ideal: window.azState.facingMode },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        advanced: [{ focusMode: "continuous" }]
      }
    };
    var stream = await navigator.mediaDevices.getUserMedia(constraints);
    window.azState.stream = stream;
    video.srcObject = stream;
    await video.play();
    if (msgBox) msgBox.style.display = "none";
  } catch (err) {
    console.warn("Camera start error:", err);
    if (msgBox) msgBox.style.display = "flex";
    if (msgText) msgText.innerText = "Kameranı aktivləşdirmək üçün aşağıdakı düyməni sıxın və icazə verin:";
  }
};

window.azSwitchCam = function() {
  window.azState.facingMode = (window.azState.facingMode === "environment") ? "user" : "environment";
  window.azStartCamera();
};

window.azToggleTorch = async function() {
  if (!window.azState.stream) return;
  var track = window.azState.stream.getVideoTracks()[0];
  if (!track) return;
  try {
    window.azState.torchOn = !window.azState.torchOn;
    await track.applyConstraints({ advanced: [{ torch: window.azState.torchOn }] });
    var btn = document.getElementById("az-btn-torch");
    if (btn) btn.style.background = window.azState.torchOn ? "#fde047" : "#f3f4f6";
  } catch (e) {
    alert("Bu cihazda fənər brauzerdən idarə oluna bilmir.");
  }
};

window.azCaptureAndSend = async function() {
  if (window.azState.busy) return;
  var video = document.getElementById("az-cam-video");
  var stage = document.getElementById("az-video-stage");
  var box = document.getElementById("az-scan-box");
  if (!video || !video.videoWidth || !video.videoHeight || !stage || !box) {
    await window.azStartCamera();
    return;
  }

  window.azState.busy = true;
  var S = window.azCamStrings[window.azState.lang] || window.azCamStrings.az;
  var statusEl = document.getElementById("az-live-status");
  if (statusEl) statusEl.innerText = S.status_proc;

  try {
    var vW = video.videoWidth;
    var vH = video.videoHeight;
    var sRect = stage.getBoundingClientRect();
    var bRect = box.getBoundingClientRect();

    var stageAspect = sRect.width / sRect.height;
    var videoAspect = vW / vH;

    var renderW, renderH, offsetX, offsetY;
    if (stageAspect > videoAspect) {
      renderW = sRect.width;
      renderH = sRect.width / videoAspect;
      offsetX = 0;
      offsetY = (sRect.height - renderH) / 2;
    } else {
      renderH = sRect.height;
      renderW = sRect.height * videoAspect;
      offsetY = 0;
      offsetX = (sRect.width - renderW) / 2;
    }

    var scale = vW / renderW;
    var sx = Math.max(0, (bRect.left - sRect.left - offsetX) * scale);
    var sy = Math.max(0, (bRect.top - sRect.top - offsetY) * scale);
    var sw = Math.min(vW - sx, bRect.width * scale);
    var sh = Math.min(vH - sy, bRect.height * scale);

    var canvas = document.getElementById("az-crop-preview");
    if (!canvas) canvas = document.createElement("canvas");
    canvas.width = Math.round(sw);
    canvas.height = Math.round(sh);
    canvas.style.display = "block";
    var ctx = canvas.getContext("2d");
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

    var b64 = canvas.toDataURL("image/jpeg", 0.90);
    var resp = await fetch("/api/ocr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: b64,
        engine: window.azState.engine || "crnn_v2",
        multi_line: (window.azState.boxMode !== "line"),
        postprocess: true
      })
    });
    var data = await resp.json();
    var resArea = document.getElementById("az-live-result");
    if (resArea && data.text !== undefined) {
      resArea.value = data.text || S.status_notfound;
    }
    if (statusEl) {
      var engLabel = data.engine || (window.azState.engine === "tesseract" ? "Tesseract 5" : "CRNN-V2");
      statusEl.innerText = S.status_done + engLabel + " (" + (data.ms || 0) + " ms | " + (data.lines || 1) + " sətir)";
    }
  } catch (err) {
    if (statusEl) statusEl.innerText = "Xəta: " + err.message;
  } finally {
    window.azState.busy = false;
  }
};

window.azSnapNow = function() {
  window.azCaptureAndSend();
};

window.azToggleAuto = function() {
  var btn = document.getElementById("az-btn-auto");
  var S = window.azCamStrings[window.azState.lang] || window.azCamStrings.az;
  if (window.azState.autoTimer) {
    clearInterval(window.azState.autoTimer);
    window.azState.autoTimer = null;
    if (btn) {
      btn.innerText = S.auto_off;
      btn.style.background = "#f3f4f6";
      btn.style.color = "#111827";
    }
  } else {
    window.azCaptureAndSend();
    window.azState.autoTimer = setInterval(window.azCaptureAndSend, 1100);
    if (btn) {
      btn.innerText = S.auto_on;
      btn.style.background = "#991b1b";
      btn.style.color = "#ffffff";
    }
  }
};

window.azCopyResult = function() {
  var resArea = document.getElementById("az-live-result");
  var btn = document.getElementById("az-btn-copy");
  var S = window.azCamStrings[window.azState.lang] || window.azCamStrings.az;
  if (!resArea || !resArea.value) return;
  navigator.clipboard.writeText(resArea.value);
  if (btn) {
    var old = btn.innerText;
    btn.innerText = S.copied;
    setTimeout(function() { btn.innerText = old; }, 1500);
  }
};

window.azHandleNativeFile = function(input) {
  if (!input.files || !input.files[0]) return;
  var file = input.files[0];
  var reader = new FileReader();
  var S = window.azCamStrings[window.azState.lang] || window.azCamStrings.az;
  var statusEl = document.getElementById("az-live-status");
  if (statusEl) statusEl.innerText = S.status_proc;
  reader.onload = async function(e) {
    var b64 = e.target.result;
    try {
      var resp = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: b64,
          engine: window.azState.engine || "crnn_v2",
          multi_line: true,
          postprocess: true
        })
      });
      var data = await resp.json();
      var resArea = document.getElementById("az-live-result");
      if (resArea && data.text !== undefined) {
        resArea.value = data.text || S.status_notfound;
      }
      if (statusEl) {
        var engLabel = data.engine || (window.azState.engine === "tesseract" ? "Tesseract 5" : "CRNN-V2");
        statusEl.innerText = S.status_done + engLabel + " (" + (data.ms || 0) + " ms | " + (data.lines || 1) + " sətir)";
      }
    } catch (err) {
      if (statusEl) statusEl.innerText = "Xəta: " + err.message;
    }
  };
  reader.readAsDataURL(file);
};

// Enforce Light Mode immediately and prevent browser dark-mode overrides
(function() {
  function enforceLight() {
    if (document.documentElement && document.documentElement.classList.contains("dark")) {
      document.documentElement.classList.remove("dark");
    }
    if (document.body && document.body.classList.contains("dark")) {
      document.body.classList.remove("dark");
    }
    if (document.documentElement) {
      document.documentElement.style.colorScheme = "light";
    }
  }
  enforceLight();
  if (typeof MutationObserver !== "undefined") {
    new MutationObserver(enforceLight).observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    window.addEventListener("DOMContentLoaded", function() {
      enforceLight();
      if (document.body) {
        new MutationObserver(enforceLight).observe(document.body, { attributes: true, attributeFilter: ["class"] });
      }
    });
  }
window.azI18N = {
  "az": {
    title: "Azərbaycan Dili üçün Hibrid Optik Mətn Tanıma Sistemi (AzOCR-V2)",
    sub: "Konvolyusiyalı Rekurrent Neyron Şəbəkəsi (CRNN-BiLSTM-CTC) və Morfoloji Leksikon Əsaslı Post-Prosesinq Mühiti",
    badge: "Elmi-Tədqiqat Platforması (BM-606)",
    spec_arch: "<strong>Arxitektura:</strong> CNN + 2×BiLSTM + CTC (4.05M parametr)",
    spec_tensor: "<strong>Giriş Tenzoru:</strong> 32 × 512 piksel (Grayscale)",
    spec_alpha: "<strong>Əlifba Həcmi:</strong> 88 simvol (Ç, Ə, Ğ, İ, I, Ö, Ş, Ü daxil)",
    spec_lex: "<strong>Leksikon:</strong> 116,143 söz + Morfoloji şəkilçi analizi",
        drop_heading: "Təsviri buraya sürüşdürün",
    drop_or: "- və ya -",
    drop_click: "Yükləmək üçün klikləyin",
    drop_sub: "- və ya -<br/>Yükləmək üçün klikləyin",
    tab1: "Bölmə 1: Sənəd və Təsvir Analizi",
    tab2: "Bölmə 2: Eksperimental Model Müqayisəsi",
    tab3: "Bölmə 3: Real-Vaxt Kamera Sınaq Modulu",
    studio_img: "Giriş Təsviri (Sənəd və ya Sətir Kəsimi)",
    studio_model: "Tanıma Mühərriki (Model Konfiqurasiyası)",
    studio_pp: "Morfoloji Leksikon Korreksiyası (SymSpell)",
    studio_multiline: "Avtomatik Çoxsətirli Seqmentasiya",
    studio_btn: "Mətni Tanı və Analiz Et",
    studio_clear: "Təmizlə",
    studio_preview: "Normalizasiya Olunmuş Tenzor Zolağı (32×W px)",
    studio_out: "Yekun Tanınmış Mətn (Post-Prosesinq Nəticəsi)",
    studio_placeholder: "Tanınmış mətn burada əks olunacaq...",
    diff_title: "MƏRHƏLƏLİ ÇIXIŞ ANALİZİ (CTC VƏ MORFOLOJİ KORREKSİYA)",
    diff_empty: "Analiz üçün təsvir yükləyin və ya aşağıdakı etalon nümunələrdən birini seçin.",
    diff_stage1: "1. Xam CTC Çıxışı:",
    diff_stage2: "2. Leksikon Korreksiyası:",
    tel_title: "HESABLAMA TELEMETRİYASI VƏ STATİSTİK GÖSTƏRİCİLƏR",
    tel_time: "ÜMUMİ İCRA MÜDDƏTİ",
    tel_seg: "SEQMENTASİYA",
    tel_vol: "MƏTN HƏCMİ",
    tel_eng: "AKTİV KONFİQURASİYA",
    examples_label1: "Etalon Sınaq Nümunələri (Seçildikdə avtomatik icra olunur)",
    bench_img: "Müqayisə Ediləcək Təsvir",
    bench_gt: "Etalon Mətn (Ground Truth — CER və WER hesablanması üçün)",
    bench_gt_placeholder: "Etalon mətni daxil edin və ya aşağıdakı hazır nümunələrdən birini seçin...",
    bench_btn: "Bütün Modelləri Paralel İcra Et və Müqayisə Cədvəlini Qur",
    bench_caption: "Cədvəl: Müxtəlif OCR arxitekturalarının seçilmiş təsvir üzrə kəmiyyət və keyfiyyət müqayisəsi",
    bench_empty: "Eksperimental müqayisə cədvəlini qurmaq üçün təsvir yükləyin və ya aşağıdakı etalon nümunələrdən birini seçin.",
    bench_col_model: "Model / Konfiqurasiya",
    bench_col_hyp: "Tanınmış Mətn (Hipotez)",
    bench_col_lat: "Müddət (ms)",
    bench_col_cer: "CER (%) ↓",
    bench_col_wer: "WER (%) ↓",
    bench_col_acc: "Dəqiqlik (%) ↑",
    bench_m_v2_pp: "CRNN-V2 + Morfoloji SymSpell (Təklif olunan)",
    bench_m_v2_raw: "CRNN-V2 (Xam CTC Dekodlama)",
    bench_m_v1: "CRNN-V1 (Baza Arxitekturası)",
    bench_m_tess: "Tesseract 5 LSTM (aze — Sənaye Baseline)",
    bench_gt_banner: "Etalon Mətn (Ground Truth):",
    examples_label2: "Etalon Mətnli Eksperimental Nümunələr (Seçildikdə CER/WER cədvəli qurulur)"
  },
  "tr": {
    title: "Azerbaycan Türkçesi için Hibrit Optik Metin Tanıma Sistemi (AzOCR-V2)",
    sub: "Evrişimli Tekrarlayan Sinir Ağı (CRNN-BiLSTM-CTC) ve Morfolojik Sözlük Tabanlı Son-İşleme Ortamı",
    badge: "Bilimsel Araştırma Platformu (BM-606)",
    spec_arch: "<strong>Mimari:</strong> CNN + 2×BiLSTM + CTC (4.05M parametre)",
    spec_tensor: "<strong>Girdi Tensörü:</strong> 32 × 512 piksel (Gri tonlama)",
    spec_alpha: "<strong>Alfabe Hacmi:</strong> 88 karakter (Ç, Ə, Ğ, İ, I, Ö, Ş, Ü dahil)",
    spec_lex: "<strong>Leksikon:</strong> 116,143 kelime + Morfolojik ek analizi",
        drop_heading: "Resmi buraya sürükle",
    drop_or: "- veya -",
    drop_click: "Yüklemek için tıkla",
    drop_sub: "- veya -<br/>Yüklemek için tıkla",
    tab1: "Bölüm 1: Belge ve Görüntü Analizi",
    tab2: "Bölüm 2: Deneysel Model Karşılaştırması",
    tab3: "Bölüm 3: Gerçek Zamanlı Kamera Test Modülü",
    studio_img: "Girdi Görüntüsü (Belge veya Satır Kırpıntısı)",
    studio_model: "Tanıma Motoru (Model Yapılandırması)",
    studio_pp: "Morfolojik Sözlük Düzeltmesi (SymSpell)",
    studio_multiline: "Otomatik Çok Satırlı Segmentasyon",
    studio_btn: "Metni Tanı ve Analiz Et",
    studio_clear: "Temizle",
    studio_preview: "Normalize Edilmiş Tensör Şeridi (32×W px)",
    studio_out: "Nihai Tanınan Metin (Son-İşleme Sonucu)",
    studio_placeholder: "Tanınan metin burada görüntülenecek...",
    diff_title: "AŞAMALI ÇIKTI ANALİZİ (CTC VE MORFOLOJİK DÜZELTME)",
    diff_empty: "Analiz için görüntü yükleyin veya aşağıdaki referans örneklerden birini seçin.",
    diff_stage1: "1. Ham CTC Çıktısı:",
    diff_stage2: "2. Sözlük Düzeltmesi:",
    tel_title: "HESAPLAMA TELEMETRİSİ VE İSTATİSTİKSEL GÖSTERGELER",
    tel_time: "TOPLAM İŞLEME SÜRESİ",
    tel_seg: "SEGMENTASYON",
    tel_vol: "METİN HACMİ",
    tel_eng: "AKTİF YAPILANDIRMA",
    examples_label1: "Referans Test Örnekleri (Seçildiğinde otomatik çalıştırılır)",
    bench_img: "Karşılaştırılacak Görüntü",
    bench_gt: "Referans Metin (Ground Truth — CER ve WER hesabı için)",
    bench_gt_placeholder: "Referans metni girin veya aşağıdaki hazır örneklerden birini seçin...",
    bench_btn: "Tüm Modelleri Paralel Çalıştır ve Karşılaştırma Tablosunu Oluştur",
    bench_caption: "Tablo: Farklı OCR mimarilerinin seçilen görüntü üzerindeki nicel ve nitel karşılaştırması",
    bench_empty: "Deneysel karşılaştırma tablosunu oluşturmak için görüntü yükleyin veya aşağıdaki referans örneklerden birini seçin.",
    bench_col_model: "Model / Yapılandırma",
    bench_col_hyp: "Tanınan Metin (Hipotez)",
    bench_col_lat: "Süre (ms)",
    bench_col_cer: "CER (%) ↓",
    bench_col_wer: "WER (%) ↓",
    bench_col_acc: "Doğruluk (%) ↑",
    bench_m_v2_pp: "CRNN-V2 + Morfolojik SymSpell (Önerilen)",
    bench_m_v2_raw: "CRNN-V2 (Ham CTC Çözümleme)",
    bench_m_v1: "CRNN-V1 (Temel Mimari)",
    bench_m_tess: "Tesseract 5 LSTM (aze — Endüstriyel Referans)",
    bench_gt_banner: "Referans Metin (Ground Truth):",
    examples_label2: "Referans Metinli Deneysel Örnekler (Seçildiğinde CER/WER tablosu oluşturulur)"
  },
  "en": {
    title: "Hybrid Optical Character Recognition System for Azerbaijani (AzOCR-V2)",
    sub: "Convolutional Recurrent Neural Network (CRNN-BiLSTM-CTC) with Morphological Lexicon-Driven Post-Processing",
    badge: "Scientific Research Platform (BM-606)",
    spec_arch: "<strong>Architecture:</strong> CNN + 2×BiLSTM + CTC (4.05M parameters)",
    spec_tensor: "<strong>Input Tensor:</strong> 32 × 512 pixels (Grayscale)",
    spec_alpha: "<strong>Alphabet Size:</strong> 88 characters (incl. Ç, Ə, Ğ, İ, I, Ö, Ş, Ü)",
    spec_lex: "<strong>Lexicon:</strong> 116,143 terms + Morphological suffix analysis",
        drop_heading: "Drop Image Here",
    drop_or: "- or -",
    drop_click: "Click to Upload",
    drop_sub: "- or -<br/>Click to Upload",
    tab1: "Section 1: Document & Image Analysis",
    tab2: "Section 2: Experimental Model Comparison",
    tab3: "Section 3: Real-Time Camera Testing Module",
    studio_img: "Input Image (Document Page or Line Crop)",
    studio_model: "Recognition Engine (Model Configuration)",
    studio_pp: "Morphological Lexicon Correction (SymSpell)",
    studio_multiline: "Automatic Multi-line Segmentation",
    studio_btn: "Recognize & Analyze Text",
    studio_clear: "Clear",
    studio_preview: "Normalized Input Tensor Strip (32×W px)",
    studio_out: "Final Recognized Text (Post-Processed Output)",
    studio_placeholder: "Recognized text will appear here...",
    diff_title: "STAGE-BY-STAGE OUTPUT ANALYSIS (CTC & MORPHOLOGY)",
    diff_empty: "Upload an image for analysis or select one of the benchmark samples below.",
    diff_stage1: "1. Raw CTC Output:",
    diff_stage2: "2. Lexicon Correction:",
    tel_title: "COMPUTATIONAL TELEMETRY & STATISTICAL METRICS",
    tel_time: "TOTAL LATENCY",
    tel_seg: "SEGMENTATION",
    tel_vol: "TEXT VOLUME",
    tel_eng: "ACTIVE ENGINE",
    examples_label1: "Benchmark Test Samples (Click to auto-execute)",
    bench_img: "Evaluation Image",
    bench_gt: "Ground Truth Text (For CER and WER error calculation)",
    bench_gt_placeholder: "Enter ground truth text or select from samples below...",
    bench_btn: "Execute All Models in Parallel & Generate Evaluation Table",
    bench_caption: "Table: Quantitative and qualitative evaluation of OCR architectures on the test image",
    bench_empty: "Upload an image or select a benchmark sample below to generate evaluation table.",
    bench_col_model: "Model / Architecture",
    bench_col_hyp: "Recognized Text (Hypothesis)",
    bench_col_lat: "Latency (ms)",
    bench_col_cer: "CER (%) ↓",
    bench_col_wer: "WER (%) ↓",
    bench_col_acc: "Accuracy (%) ↑",
    bench_m_v2_pp: "CRNN-V2 + Morphological SymSpell (Proposed)",
    bench_m_v2_raw: "CRNN-V2 (Raw CTC Decoding)",
    bench_m_v1: "CRNN-V1 (Baseline Architecture)",
    bench_m_tess: "Tesseract 5 LSTM (aze — Industrial Baseline)",
    bench_gt_banner: "Ground Truth Reference:",
    examples_label2: "Experimental Benchmark Samples with Ground Truth (Generates CER/WER Table)"
  }
};


  function updateUploadDropzone(code) {
    if (!code || !window.azI18N[code]) code = "az";
    var S = window.azI18N[code];

    var uploadZones = document.querySelectorAll(".image-container, .upload-container, [data-testid='image'], div[data-testid='dropzone'], [data-testid='upload-text']");
    uploadZones.forEach(function(zone) {
      // 1. If Gradio rendered placeholder h2/p:
      var h2 = zone.querySelector("h2");
      var p = zone.querySelector("p");
      if (h2) h2.innerText = S.drop_heading;
      if (p) p.innerHTML = S.drop_sub;

      // 2. If already custom-rendered by our script:
      var customBox = zone.querySelector(".az-dropzone-custom");
      if (customBox) {
        var h = customBox.querySelector(".az-drop-h");
        var o = customBox.querySelector(".az-drop-o");
        var c = customBox.querySelector(".az-drop-c");
        if (h && h.innerText !== S.drop_heading) h.innerText = S.drop_heading;
        if (o && o.innerText !== S.drop_or) o.innerText = S.drop_or;
        if (c && c.innerText !== S.drop_click) c.innerText = S.drop_click;
        return;
      }

      // 3. Fallback when placeholder is not yet rendered by Gradio:
      var uploadText = (zone.getAttribute && zone.getAttribute("data-testid") === "upload-text") ? zone : zone.querySelector("[data-testid='upload-text']");
      if (!uploadText) {
        var orSpan = zone.querySelector(".or, span.or");
        if (orSpan) uploadText = orSpan.parentElement;
      }
      if (uploadText && !h2) {
        var svg = uploadText.querySelector("svg");
        var svgHtml = svg ? svg.outerHTML : "";
        uploadText.innerHTML = (svgHtml ? svgHtml + "<br/>" : "") +
          "<div class='az-dropzone-custom' style='text-align:center; display:inline-block;'>" +
          "<span class='az-drop-h' style='font-size:14px; font-weight:600; display:block; margin:6px 0 3px; color:#0f172a;'>" + S.drop_heading + "</span>" +
          "<span class='az-drop-o' style='font-size:12px; color:#64748b; display:block;'>" + S.drop_or + "</span>" +
          "<span class='az-drop-c' style='font-size:13px; font-weight:600; color:#1b365d; display:block; margin-top:3px;'>" + S.drop_click + "</span>" +
          "</div>";
      }
    });
  }

  function updateExamplesDom(code) {
    if (!code || !window.azI18N[code]) code = "az";
    var S = window.azI18N[code];

    // 1. Table Headers in all dataset tables (excluding benchmark comparison table)
    var allThs = document.querySelectorAll("th[role='columnheader'], table:not(.az-booktabs-table) th");
    if (allThs.length >= 2) {
      allThs[0].innerText = S.studio_img;
      allThs[1].innerText = S.studio_out;
    }
    if (allThs.length >= 4) {
      allThs[2].innerText = S.bench_img;
      allThs[3].innerText = S.bench_gt;
    }

    // 2. Examples Labels above tables
    var allLabels = document.querySelectorAll("div.label, .label, [class*='label']");
    allLabels.forEach(function(lbl) {
      var txt = lbl.innerText.trim();
      if (txt.includes("Etalon") || txt.includes("Referans") || txt.includes("Benchmark") || txt.includes("Sınaq Nümunələri") || txt.includes("Test Samples")) {
        var svg = lbl.querySelector("svg");
        var svgHtml = svg ? svg.outerHTML + " " : "";
        if (txt.includes("Mətnli") || txt.includes("CER/WER") || txt.includes("Evaluation") || txt.includes("Ground Truth")) {
          lbl.innerHTML = svgHtml + S.examples_label2;
        } else {
          lbl.innerHTML = svgHtml + S.examples_label1;
        }
      }
    });
  }


  function azSetLabel(id, txt) {
    var root = document.getElementById(id);
    if (!root || !txt) return;
    var lbl = root.querySelector("[data-testid='block-label']") || root.querySelector("label > span") || root.querySelector("label span");
    if (!lbl) return;
    var nodes = Array.prototype.filter.call(lbl.childNodes, function(n){ return n.nodeType === 3 && n.textContent.trim(); });
    if (nodes.length) { if (nodes[0].textContent.trim() !== txt) nodes[0].textContent = " " + txt; }
    else if (lbl.innerText.trim() !== txt) { lbl.appendChild(document.createTextNode(" " + txt)); }
  }
  function azApplySection2(code) {
    var T = window.azI18N[code] || window.azI18N.az;
    azSetLabel("az-bench-img", T.bench_img);
    azSetLabel("az-bench-gt", T.bench_gt);
    azSetLabel("az-studio-img", T.studio_img);
    azSetLabel("az-studio-model", T.studio_model);
    azSetLabel("az-studio-pp", T.studio_pp);
    azSetLabel("az-studio-multiline", T.studio_multiline);
    azSetLabel("az-studio-preview", T.studio_preview);
    azSetLabel("az-studio-out", T.studio_out);
    var ta = document.querySelector("#az-bench-gt textarea");
    if (ta && ta.placeholder !== T.bench_gt_placeholder) ta.placeholder = T.bench_gt_placeholder;
    var tb = document.getElementById("az-bench-table");
    if (!tb) return;
    var cap = tb.querySelector("caption"); if (cap && cap.innerText !== T.bench_caption) cap.innerText = T.bench_caption;
    var ths = tb.querySelectorAll("th");
    var hk = ["bench_col_model","bench_col_hyp","bench_col_lat","bench_col_cer","bench_col_wer","bench_col_acc"];
    if (ths.length >= 6) hk.forEach(function(k,i){ if (ths[i].innerText !== T[k]) ths[i].innerText = T[k]; });
    var gl = tb.querySelector(".az-gt-box strong"); if (gl && gl.innerText !== T.bench_gt_banner) gl.innerText = T.bench_gt_banner;
    var mk = ["bench_m_v2_pp","bench_m_v2_raw","bench_m_v1","bench_m_tess"];
    tb.querySelectorAll("tbody tr").forEach(function(tr,i){ var td = tr.querySelector("td"); if (td && mk[i] && td.innerText !== T[mk[i]]) td.innerText = T[mk[i]]; });
    var em = tb.querySelector(".az-bench-empty"); if (em && em.innerText.trim() !== T.bench_empty) em.innerText = T.bench_empty;
  }
window.azSwitchLang = function(code) {
  if (!code || !window.azI18N[code]) code = "az";
  window.azActiveLang = code;
  var T = window.azI18N[code];

  // 1. Top Bar Buttons
  ["az", "tr", "en"].forEach(function(k) {
    var b = document.getElementById("btn-lang-" + k);
    if (b) {
      if (k === code) b.classList.add("active");
      else b.classList.remove("active");
    }
  });

  // 2. Header & Specifications
  var elTitle = document.getElementById("az-hero-title");
  if (elTitle) elTitle.innerText = T.title;
  var elSub = document.getElementById("az-hero-sub");
  if (elSub) elSub.innerText = T.sub;

  var sArch = document.getElementById("az-spec-arch");
  if (sArch) sArch.innerHTML = T.spec_arch;
  var sTensor = document.getElementById("az-spec-tensor");
  if (sTensor) sTensor.innerHTML = T.spec_tensor;
  var sAlpha = document.getElementById("az-spec-alpha");
  if (sAlpha) sAlpha.innerHTML = T.spec_alpha;
  var sLex = document.getElementById("az-spec-lex");
  if (sLex) sLex.innerHTML = T.spec_lex;

  // 3. Tab Navigation (All 3 Sections)
  var tabBtns = document.querySelectorAll("button[role='tab']");
  if (tabBtns.length >= 3) {
    tabBtns[0].innerText = T.tab1;
    tabBtns[1].innerText = T.tab2;
    tabBtns[2].innerText = T.tab3;
  }

  // 4. Section 1 (Document & Image Analysis)
  var bStudio = document.querySelector("#az-studio-btn button") || document.getElementById("az-studio-btn");
  if (bStudio) bStudio.innerText = T.studio_btn;
  var bClear = document.querySelector("#az-studio-clear button") || document.getElementById("az-studio-clear");
  if (bClear) bClear.innerText = T.studio_clear;

  // Section 1 Diff Panel
  var diffHeader = document.querySelector("#az-diff-container .az-panel-header span, #az-diff-container .az-panel-header");
  if (diffHeader) diffHeader.innerText = T.diff_title;
  var diffEmpty = document.querySelector("#az-diff-container .az-diff-empty");
  if (diffEmpty) diffEmpty.innerText = T.diff_empty;
  var stageKeys = document.querySelectorAll("#az-diff-container .az-stage-key");
  if (stageKeys.length >= 2) {
    stageKeys[0].innerText = T.diff_stage1;
    stageKeys[1].innerText = T.diff_stage2;
  }

  // Section 1 Telemetry Panel
  var telHeader = document.querySelector("#az-telemetry-container .az-panel-header");
  if (telHeader) telHeader.innerText = T.tel_title;
  var telLabels = document.querySelectorAll("#az-telemetry-container .az-metric-label");
  if (telLabels.length >= 4) {
    telLabels[0].innerText = T.tel_time;
    telLabels[1].innerText = T.tel_seg;
    telLabels[2].innerText = T.tel_vol;
    telLabels[3].innerText = T.tel_eng;
  }

  // 5. Section 2 (Experimental Model Comparison)
  var bBench = document.querySelector("#az-bench-btn button") || document.getElementById("az-bench-btn");
  if (bBench) bBench.innerText = T.bench_btn;

  var benchCaption = document.querySelector("#az-bench-table caption");
  if (benchCaption) benchCaption.innerText = T.bench_caption;
  var benchThs = document.querySelectorAll("#az-bench-table th");
  if (benchThs.length >= 6) {
    benchThs[0].innerText = T.bench_col_model;
    benchThs[1].innerText = T.bench_col_hyp;
    benchThs[2].innerText = T.bench_col_lat;
    benchThs[3].innerText = T.bench_col_cer;
    benchThs[4].innerText = T.bench_col_wer;
    benchThs[5].innerText = T.bench_col_acc;
  }
  var benchEmpty = document.querySelector("#az-bench-table .az-bench-empty");
  if (benchEmpty) benchEmpty.innerText = T.bench_empty;

  azApplySection2(code);

  // 6. Section 3 (Real-Time Camera Module)
  if (window.azSetUiLang) {
    window.azSetUiLang(code);
  }

  // 7. Update Examples dataset table and labels
  updateExamplesDom(code);
  updateUploadDropzone(code);

  // 8. Trigger Python server directly via hidden native button
  var trigBtn = document.getElementById("az-trig-" + code);
  if (trigBtn) {
    trigBtn.click();
  }
};

document.addEventListener("change", function(e) {
  if (e.target && e.target.name && e.target.closest && (e.target.closest(".az-lang-bar") || e.target.closest(".az-hidden-radio"))) {
    var v = e.target.value || (e.target.parentElement ? e.target.parentElement.innerText : "");
    var code = v.includes("Türk") ? "tr" : (v.includes("Eng") ? "en" : "az");
    if (window.azActiveLang !== code) {
      window.azSwitchLang(code);
    }
  }
});

setInterval(function() {
  if (!window._azQrPublicLoaded) {
    var qrBox = document.getElementById("az-qr-box");
    if (qrBox) {
      fetch("/api/qr").then(function(r){ return r.json(); }).then(function(d){
        if (d && d.html) {
          var cur = document.getElementById("az-qr-box");
          if (cur && (!cur.innerHTML.includes(d.url))) {
            var wasOpen = cur.hasAttribute("open");
            cur.outerHTML = d.html;
            var next = document.getElementById("az-qr-box");
            if (next && !wasOpen) next.removeAttribute("open");
          }
          if (d.is_public) window._azQrPublicLoaded = true;
        }
      }).catch(function(){});
    }
  }
  if (!window.azActiveLang) return;
  var T = window.azI18N[window.azActiveLang];
  if (!T) return;
  var tabBtns = document.querySelectorAll("button[role='tab']");
  if (tabBtns.length >= 3) {
    if (tabBtns[0].innerText !== T.tab1) tabBtns[0].innerText = T.tab1;
    if (tabBtns[1].innerText !== T.tab2) tabBtns[1].innerText = T.tab2;
    if (tabBtns[2].innerText !== T.tab3) tabBtns[2].innerText = T.tab3;
  }
  azApplySection2(window.azActiveLang);
  if (typeof updateExamplesDom === "function") updateExamplesDom(window.azActiveLang);
  updateUploadDropzone(window.azActiveLang);

  // Ensure file inputs accept HEIC/HEIF photos
  document.querySelectorAll('input[type="file"]').forEach(function(inp) {
    if (inp.accept && inp.accept.includes("image/*") && !inp.accept.includes(".heic")) {
      inp.accept += ",.heic,.heif,.HEIC,.HEIF";
    }
  });
}, 600);

})();
</script>
"""

# ==============================================================================
# FORMAL INSTITUTIONAL & ACADEMIC JOURNAL CSS (100% HIGH-CONTRAST LIGHT MODE)
# ==============================================================================
custom_css = """
:root, .dark, html, body, .gradio-container {
    color-scheme: light !important;
    background-color: #ffffff !important;
    color: #0f172a !important;
    --body-background-fill: #ffffff !important;
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #334155 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-label-background-fill: #f1f5f9 !important;
    --block-label-text-color: #0f172a !important;
    --block-title-text-color: #0f172a !important;
    --block-border-color: #94a3b8 !important;
    --border-color-primary: #94a3b8 !important;
    --border-color-accent: #1b365d !important;
    --input-background-fill: #ffffff !important;
    --input-text-color: #0f172a !important;
    --input-border-color: #94a3b8 !important;
    --table-even-background-fill: #f8fafc !important;
    --table-odd-background-fill: #ffffff !important;
    --table-text-color: #0f172a !important;
    --table-border-color: #cbd5e1 !important;
    --panel-background-fill: #f8fafc !important;
    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-selected: #1b365d !important;
    --checkbox-background-color-hover: #f1f5f9 !important;
    --checkbox-border-color: #64748b !important;
    --checkbox-border-color-hover: #1b365d !important;
    --checkbox-border-color-focus: #1b365d !important;
    --checkbox-border-color-selected: #1b365d !important;
    --checkbox-label-text-color: #0f172a !important;
    --checkbox-label-background-fill: #ffffff !important;
}
.gradio-container {
    width: 100% !important;
    min-width: 0 !important;
    max-width: min(100%, 1220px) !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
}
/* Academic Top Toolbar for Journal Publication */
.az-academic-topbar {
    background: #ffffff !important;
    border: 1.5px solid #1b365d !important;
    border-top: 3px solid #1b365d !important;
    border-radius: 4px !important;
    padding: 8px 16px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
}
.az-lang-bar-row {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    flex-wrap: wrap !important;
}
.az-lang-heading {
    font-size: 14px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: 0.6px !important;
    white-space: nowrap !important;
}
.az-lang-button-group {
    display: flex !important;
    gap: 12px !important;
    flex: 1 !important;
    max-width: 680px !important;
}
.az-lang-btn {
    flex: 1 !important;
    min-width: 130px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    padding: 9px 16px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #1e293b !important;
    background: #ffffff !important;
    border: 1.5px solid #94a3b8 !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    user-select: none !important;
}
.az-lang-btn, .az-lang-btn span, .az-lang-btn * {
    color: #1e293b !important;
    -webkit-text-fill-color: #1e293b !important;
}
.az-lang-btn:hover {
    background: #f1f5f9 !important;
    border-color: #1b365d !important;
}
.az-lang-btn:hover, .az-lang-btn:hover span, .az-lang-btn:hover * {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
}
/* ACTIVE LANGUAGE BUTTON: Crisp High-Contrast Oxford Navy with Pure White Text */
.az-lang-btn.active {
    background: #1b365d !important;
    background-color: #1b365d !important;
    border: 2px solid #0f172a !important;
    box-shadow: 0 2px 6px rgba(27, 54, 93, 0.35) !important;
    transform: translateY(-1px) !important;
}
.az-lang-btn.active,
.az-lang-btn.active *,
.az-lang-btn.active span,
.az-lang-btn.active .az-btn-text,
.az-lang-btn.active .az-flag {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 14.5px !important;
}

.az-hidden-triggers, .az-hidden-radio {
    display: none !important;
}
.az-hidden-radio {
    display: none !important;
}
/* Tabs Styling (High Contrast & Equal Width) */
.tab-wrapper, .tab-nav, [role="tablist"] {
    border-bottom: 2px solid #1b365d !important;
    background: #f8fafc !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 4px !important;
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow: visible !important;
}
.tab-nav button, [role="tab"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 42px !important;
    max-height: none !important;
    overflow: visible !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    color: #1e293b !important;
    background: #f1f5f9 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    line-height: 1.25 !important;
    border: 1px solid #cbd5e1 !important;
    border-bottom: none !important;
    padding: 9px 14px !important;
    margin-right: 0 !important;
    white-space: normal !important;
    word-break: break-word !important;
}
.az-qr-box {
    margin: 8px 0 10px 0;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    padding: 6px 12px;
}
.az-qr-box summary {
    cursor: pointer;
    font-weight: 700;
    font-size: 13px;
    color: #1b365d;
}
.az-qr-inner {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    padding: 10px 0 4px 0;
}
.az-qr-svg svg {
    width: 170px;
    height: 170px;
    display: block;
}
.az-qr-text {
    font-size: 13px;
    color: #0f172a;
    word-break: break-all;
    max-width: 100%;
}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
    background: #1b365d !important;
    color: #ffffff !important;
    border-color: #1b365d !important;
}
/* Block labels, inputs, dropdowns, checkboxes, tables */
label, .block label, span[data-testid="block-info"], .gr-box, .gr-form, .gr-panel {
    color: #0f172a !important;
}
input:not([type="checkbox"]):not([type="radio"]), textarea, select, .wrap-inner, .single-select {
    color: #0f172a !important;
    background-color: #ffffff !important;
}

/* Explicit Checkbox Styling - High Contrast, Visible Checkmark, and Fully Clickable */
input[type="checkbox"], input[data-testid="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    min-width: 18px !important;
    min-height: 18px !important;
    cursor: pointer !important;
    accent-color: #1b365d !important;
    background-color: #ffffff !important;
    border: 1.5px solid #64748b !important;
    border-radius: 3px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    margin: 0 8px 0 0 !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}
input[type="checkbox"]:checked, input[data-testid="checkbox"]:checked {
    background-color: #1b365d !important;
    accent-color: #1b365d !important;
}
.checkbox-container, label.svelte-1q8xtp9, label:has(input[type="checkbox"]), #az-checkbox-row label {
    cursor: pointer !important;
    user-select: none !important;
    display: inline-flex !important;
    align-items: center !important;
    pointer-events: auto !important;
}
.checkbox-container .label-text, label:has(input[type="checkbox"]) span, #az-checkbox-row span {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #0f172a !important;
    cursor: pointer !important;
    pointer-events: auto !important;
}
#az-checkbox-row {
    padding: 6px 2px !important;
    gap: 16px !important;
}
table, th, td, tr, .table-wrap, .examples-holder * {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}
/* Formal Institutional Header */
.az-official-header {
    background-color: #f8fafc !important;
    border: 1px solid #94a3b8 !important;
    border-top: 4px solid #1b365d !important;
    border-bottom: 2px solid #1b365d !important;
    padding: 16px 18px 14px 18px !important;
    margin-bottom: 16px !important;
}
.az-official-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 12px;
}
.az-official-title {
    margin: 0 !important;
    font-family: 'Times New Roman', 'Georgia', 'Palatino Linotype', serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2px !important;
    color: #0f172a !important;
    text-transform: uppercase !important;
}
.az-official-sub {
    margin: 5px 0 0 0 !important;
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 14.5px !important;
    font-style: italic !important;
    color: #1e293b !important;
}
.az-official-meta {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    border: 1px solid #1b365d !important;
    padding: 4px 12px !important;
    background: #1b365d !important;
}
.az-lang-pills {
    display: inline-flex !important;
    gap: 6px !important;
    align-items: center !important;
}
.az-lang-pill {
    background: #ffffff !important;
    color: #1b365d !important;
    border: 1.5px solid #1b365d !important;
    padding: 3px 10px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    border-radius: 3px !important;
    transition: all 0.15s ease !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
.az-lang-pill:hover {
    background: #e2e8f0 !important;
    color: #0f172a !important;
}
.az-lang-pill.active {
    background: #1b365d !important;
    color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
}
.az-spec-table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin-top: 12px !important;
    border-top: 1px solid #94a3b8 !important;
    font-size: 12.5px !important;
    color: #0f172a !important;
    background: transparent !important;
}
.az-spec-table td, .az-spec-table strong {
    padding: 8px 12px 2px 0 !important;
    color: #0f172a !important;
    background: transparent !important;
    border: none !important;
}
/* Formal Panels */
.az-formal-panel {
    background: #ffffff !important;
    border: 1px solid #64748b !important;
    padding: 12px 14px !important;
    color: #0f172a !important;
}
.az-panel-header, .az-panel-header span {
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #1b365d !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
    border-bottom: 1.5px solid #94a3b8 !important;
    padding-bottom: 6px !important;
    margin-bottom: 8px !important;
}
.az-stage-table {
    width: 100% !important;
    border-collapse: collapse !important;
}
.az-stage-table td {
    padding: 7px 6px !important;
    border-bottom: 1px solid #cbd5e1 !important;
    vertical-align: top !important;
    color: #0f172a !important;
}
.az-stage-table tr:last-child td {
    border-bottom: none !important;
}
.az-stage-key {
    width: 170px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    white-space: nowrap !important;
}
.az-stage-val {
    color: #0f172a !important;
    line-height: 1.5 !important;
    font-weight: 500 !important;
}
.az-output-box textarea {
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 1.55 !important;
    color: #0f172a !important;
    background: #ffffff !important;
    border: 1.5px solid #475569 !important;
    border-radius: 2px !important;
}
.az-diff-del {
    color: #991b1b !important;
    background: #fee2e2 !important;
    text-decoration: line-through !important;
    padding: 1px 4px !important;
    font-weight: 700 !important;
}
.az-diff-ins {
    color: #065f46 !important;
    background: #dcfce7 !important;
    border-bottom: 1.5px solid #065f46 !important;
    padding: 1px 4px !important;
    font-weight: 700 !important;
}
/* Telemetry Grid - Default 2x2 inside half-width column */
.az-metrics-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 10px !important;
}
.az-metric-box {
    background: #f8fafc !important;
    border: 1px solid #94a3b8 !important;
    padding: 9px 11px !important;
}
.az-metric-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #334155 !important;
    letter-spacing: 0.3px !important;
}
.az-metric-val {
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 19px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 3px 0 !important;
}
.az-metric-sub {
    font-size: 11.5px !important;
    color: #334155 !important;
    font-weight: 500 !important;
}
/* LaTeX Booktabs Style Benchmark Table */
.az-gt-box, .az-gt-box strong {
    background: #f1f5f9 !important;
    border: 1px solid #64748b !important;
    border-left: 4px solid #1b365d !important;
    padding: 9px 12px !important;
    margin-bottom: 12px !important;
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 15px !important;
    color: #0f172a !important;
}
.az-booktabs-table {
    width: 100% !important;
    border-collapse: collapse !important;
    background: #ffffff !important;
    border-top: 2px solid #0f172a !important;
    border-bottom: 2px solid #0f172a !important;
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 14.5px !important;
    color: #0f172a !important;
}
.az-booktabs-table caption {
    caption-side: top !important;
    text-align: left !important;
    font-family: 'Times New Roman', 'Georgia', serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin-bottom: 8px !important;
}
.az-booktabs-table thead th {
    border-bottom: 1.5px solid #0f172a !important;
    padding: 9px 10px !important;
    text-align: left !important;
    font-weight: 700 !important;
    font-size: 13.5px !important;
    color: #0f172a !important;
    background: #f1f5f9 !important;
}
.az-booktabs-table tbody td {
    padding: 9px 10px !important;
    border-bottom: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
}
.az-row-featured td {
    background-color: #f0fdf4 !important;
    color: #0f172a !important;
}
/* Override Gradio buttons for formal rectangular look */
button.primary {
    background: #1b365d !important;
    color: #ffffff !important;
    border: 1px solid #0f2342 !important;
    border-radius: 2px !important;
    font-weight: 700 !important;
}

/* Normalized Tensor Strip Preview - Clean Academic Box */
#az-studio-preview {
    min-height: 110px !important;
    background: #ffffff !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 4px !important;
}
#az-studio-preview > label {
    margin: 0 !important;
    padding: 0 !important;
}
#az-studio-preview span[data-testid="block-info"] {
    display: block !important;
    background: #f1f5f9 !important;
    padding: 5px 10px !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    border-bottom: 1px solid #cbd5e1 !important;
    margin: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
#az-studio-preview .image-container,
#az-studio-preview [data-testid="image"] {
    padding: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 52px !important;
    background: #ffffff !important;
}
#az-studio-preview img {
    max-height: 46px !important;
    object-fit: contain !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    padding: 2px !important;
}

/* Examples Table Clean Formal Styling */
.dataset, div[data-testid="dataset"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
    margin-top: 10px !important;
    overflow-x: auto !important;
}
.dataset table, div[data-testid="dataset"] table {
    width: 100% !important;
    border-collapse: collapse !important;
}
.dataset th, div[data-testid="dataset"] th {
    background: #f1f5f9 !important;
    color: #0f172a !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 9px 12px !important;
    border-bottom: 2px solid #cbd5e1 !important;
    text-align: left !important;
}
.dataset td, div[data-testid="dataset"] td {
    padding: 8px 12px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    color: #0f172a !important;
    font-size: 13.5px !important;
    vertical-align: middle !important;
}
.dataset td img, div[data-testid="dataset"] td img {
    max-height: 44px !important;
    width: auto !important;
    object-fit: contain !important;
    display: block !important;
}
.dataset tr:hover, div[data-testid="dataset"] tr:hover {
    background: #f8fafc !important;
    cursor: pointer !important;
}

button.secondary {
    background: #e2e8f0 !important;
    color: #0f172a !important;
    border: 1px solid #64748b !important;
    border-radius: 2px !important;
    font-weight: 700 !important;
}

/* Upload Dropzone Typography & Layout */
div[data-testid="dropzone"], .upload-container, .image-container {
    background: #ffffff !important;
}
div[data-testid="upload-text"], .prompt {
    text-align: center !important;
}
div[data-testid="upload-text"] h2, .prompt h2, .az-dropzone-custom .az-drop-h {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    margin: 8px 0 2px 0 !important;
    font-family: inherit !important;
}
div[data-testid="upload-text"] p, .prompt p, .az-dropzone-custom .az-drop-o, .az-dropzone-custom .az-drop-c {
    font-size: 12px !important;
    color: #475569 !important;
    line-height: 1.4 !important;
    font-family: inherit !important;
}

/* Model Dropdown - Pure Select Behavior, Non-editable */
#az-studio-model input {
    cursor: pointer !important;
    user-select: none !important;
    caret-color: transparent !important;
}
#az-studio-model .wrap {
    cursor: pointer !important;
}

/* ==============================================================================
   COMPREHENSIVE RESPONSIVE ADAPTATION (TABLETS, PHONES & RESIZED WINDOWS)
   ============================================================================== */
html, body {
    overflow-x: hidden !important;
    max-width: 100vw !important;
    box-sizing: border-box !important;
}
*, *:before, *:after {
    box-sizing: border-box !important;
}
/* Prevent flex children from being sized by their content (root cause of horizontal overflow on phones) */
gradio-app, .main, .wrap, .contain, .column, .tabs, .tabitem, .tab-wrapper, .tab-container, .block, .form, .row {
    min-width: 0 !important;
    max-width: 100% !important;
}
.gradio-container {
    min-width: 0 !important;
    width: 100% !important;
    max-width: min(100%, 1220px) !important;
}

/* Two-column rows stack vertically at <= 1040px */
@media (max-width: 1040px) {
    .az-responsive-row, #az-studio-row, #az-bench-row {
        flex-direction: column !important;
        display: flex !important;
        gap: 16px !important;
    }
    .az-responsive-row > div,
    #az-studio-row > div,
    #az-bench-row > div {
        width: 100% !important;
        max-width: 100% !important;
        flex: 0 0 auto !important;
    }
    /* When stacked at tablet width (768px - 1040px), metrics grid has full width, so 4 columns fits easily (240px each!) */
    .az-metrics-grid {
        grid-template-columns: repeat(4, 1fr) !important;
        gap: 8px !important;
    }
    .gradio-container {
        padding: 8px 10px !important;
        width: 100% !important;
    }
    .az-academic-topbar {
        padding: 10px 14px !important;
        margin-bottom: 12px !important;
    }
    .az-lang-bar-row {
        flex-direction: column !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .az-lang-heading {
        text-align: center !important;
        font-size: 13.5px !important;
        width: 100% !important;
    }
    .az-lang-button-group {
        width: 100% !important;
        max-width: 650px !important;
        display: flex !important;
        justify-content: center !important;
        gap: 8px !important;
    }
    .az-lang-btn {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 9px 8px !important;
        font-size: 13px !important;
        white-space: nowrap !important;
    }
    .az-lang-btn .az-flag {
        font-size: 15px !important;
    }
    .az-lang-btn .az-btn-text {
        font-size: 13px !important;
        white-space: nowrap !important;
    }
    .az-official-header {
        padding: 12px 14px 10px 14px !important;
        margin-bottom: 12px !important;
    }
    .az-official-title {
        font-size: 18px !important;
        line-height: 1.35 !important;
        text-align: center !important;
    }
    .az-official-sub {
        font-size: 13px !important;
        line-height: 1.35 !important;
        text-align: center !important;
        margin-top: 4px !important;
    }
    .az-spec-table {
        margin-top: 10px !important;
    }
    .az-spec-table tr {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: 6px 14px !important;
    }
    .az-spec-table td {
        display: block !important;
        padding: 2px 0 !important;
        font-size: 12px !important;
        line-height: 1.35 !important;
    }
    /* Tabs Equal Width & Wrap Text Cleanly */
    .tab-nav, [role="tablist"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 4px !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .tab-nav button, [role="tab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        text-align: center !important;
        justify-content: center !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        padding: 8px 10px !important;
        font-size: 13px !important;
        margin-right: 0 !important;
    }
}

@media (max-width: 768px) {
    /* Below 768px (phones), metrics grid is 2x2 */
    .az-metrics-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 8px !important;
    }
    .az-spec-table tr {
        grid-template-columns: 1fr 1fr !important;
    }
    #az-checkbox-row {
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 12px !important;
    }
}

@media (max-width: 640px) {
    .az-academic-topbar {
        padding: 8px 8px !important;
    }
    .az-lang-button-group {
        gap: 4px !important;
    }
    .az-lang-btn, .az-lang-btn.active {
        padding: 7px 4px !important;
        gap: 3px !important;
        min-width: 0 !important;
    }
    .az-lang-btn, .az-lang-btn *, .az-lang-btn.active, .az-lang-btn.active *, .az-lang-btn .az-btn-text, .az-lang-btn.active .az-btn-text {
        font-size: 11px !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        text-align: center !important;
    }
    .az-lang-btn .az-flag, .az-lang-btn.active .az-flag {
        font-size: 11px !important;
    }
    .az-qr-svg svg {
        width: 130px !important;
        height: 130px !important;
    }
    .az-spec-table tr {
        grid-template-columns: 1fr !important;
    }
    /* Equal-width 3-column tabs on phones with wrapped labels */
    .tab-wrapper, .tab-nav, [role="tablist"] {
        display: flex !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        gap: 3px !important;
        width: 100% !important;
        max-width: 100% !important;
        border-bottom: 2px solid #1b365d !important;
        margin-bottom: 10px !important;
    }
    .tab-nav button, [role="tab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        height: auto !important;
        min-height: 44px !important;
        max-height: none !important;
        overflow: visible !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.2 !important;
        padding: 6px 4px !important;
        font-size: 11px !important;
        margin-right: 0 !important;
    }
    /* Checkboxes stacked cleanly */
    #az-checkbox-row {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
        padding: 4px 0 !important;
    }
    /* Studio Action Buttons Touch-friendly */
    #az-studio-btn, #az-studio-clear {
        min-height: 44px !important;
    }
    /* Normalized Tensor Strip Preview */
    #az-studio-preview {
        width: 100% !important;
        box-sizing: border-box !important;
    }
    #az-studio-preview img {
        max-width: 100% !important;
        height: auto !important;
        max-height: 52px !important;
        object-fit: contain !important;
    }
    /* Stage-by-Stage Diff Table Stacking */
    .az-formal-panel {
        padding: 10px !important;
    }
    .az-stage-table tr {
        display: flex !important;
        flex-direction: column !important;
        padding: 6px 0 !important;
        border-bottom: 1px solid #cbd5e1 !important;
    }
    .az-stage-table tr:last-child {
        border-bottom: none !important;
    }
    .az-stage-key {
        width: 100% !important;
        margin-bottom: 3px !important;
        font-size: 12px !important;
    }
    .az-stage-val {
        width: 100% !important;
        font-size: 13px !important;
        word-break: break-word !important;
    }
    /* Telemetry Grid 2x2 */
    .az-metrics-grid {
        grid-template-columns: 1fr 1fr !important;
        gap: 6px !important;
    }
    .az-metric-box {
        padding: 7px 9px !important;
    }
    .az-metric-val {
        font-size: 15.5px !important;
    }
    .az-metric-label {
        font-size: 10px !important;
    }
    .az-metric-sub {
        font-size: 10px !important;
    }
    /* Benchmark Table Horizontal Scrolling */
    .az-bench-wrap, div:has(> .az-booktabs-table) {
        width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }
    .az-booktabs-table {
        min-width: 620px !important;
    }
    /* Examples Table Fits Mobile Screen Cleanly */
    .dataset, div[data-testid="dataset"], .table-wrap {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    .table-wrap table, .dataset table, div[data-testid="dataset"] table {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        table-layout: fixed !important;
    }
    .table-wrap th:first-child, .table-wrap td:first-child,
    .dataset th:first-child, .dataset td:first-child,
    div[data-testid="dataset"] th:first-child, div[data-testid="dataset"] td:first-child {
        width: 88px !important;
    }
    .table-wrap td, .table-wrap th, .table-wrap td *, .table-wrap th *,
    .dataset td, .dataset th, .dataset td *, .dataset th *,
    div[data-testid="dataset"] td, div[data-testid="dataset"] th,
    div[data-testid="dataset"] td *, div[data-testid="dataset"] th * {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        min-width: 0 !important;
        max-width: 100% !important;
        font-size: 11.5px !important;
    }
    .table-wrap td, .table-wrap th, .dataset td, .dataset th, div[data-testid="dataset"] td, div[data-testid="dataset"] th {
        padding: 6px !important;
    }
    /* Camera Module Responsive Controls */
    #az-cam-container {
        padding: 10px !important;
    }
    .az-cam-actions-grid {
        grid-template-columns: 1fr 1fr !important;
        gap: 6px !important;
    }
    .az-cam-engine-bar {
        flex-direction: column !important;
        align-items: stretch !important;
    }
    .az-cam-engine-bar > div {
        display: flex !important;
        width: 100% !important;
    }
    .az-cam-engine-bar > div button {
        flex: 1 1 0 !important;
        text-align: center !important;
        padding: 6px 4px !important;
        font-size: 11.5px !important;
    }
}

@media (max-width: 420px) {
    .az-lang-btn {
        padding: 7px 3px !important;
        font-size: 11px !important;
    }
    .az-lang-btn .az-flag {
        font-size: 13px !important;
    }
    .az-lang-btn .az-btn-text {
        font-size: 10.5px !important;
    }
    .az-official-title {
        font-size: 15px !important;
    }
    .az-official-sub {
        font-size: 11px !important;
    }
}
"""

BENCHMARK_SAMPLES = [
    [
        "results/real_fix8/crops_calibri/line_00.png",
        "Gəncə çayının kənarında, qədim ağacların kölgəsinə sığınmış balaca bir dulusçu emalatxanası var",
    ],
    [
        "results/real_fix8/crops_times/line_01.png",
        "Murad şəhərdə oxusa da, hər tətil otağında kəndə, bu torpağın qoxusuna qayıdırdı. Amma bu",
    ],
    [
        "results/real_fix8/crops_times/line_02.png",
        "dəfə qayıdışı başqa idi. O, hiss edirdi ki, babası təkdir və onun daşıdığı bu ağır yük -",
    ],
    [
        "results/real_fix8/crops_times/line_00.png",
        "Çöldə isə nəvəsi Murad səhərin sərin havasında babasının köhnə velosipedini təmizləyirdi.",
    ],
    [
        "results/real_fix8/crops_calibri/line_03.png",
        "sənəti, nəsildən-nəslə keçən gizli bir məharət yaşatdırdığı hiss olunurdu. Divarlardan asılmış gil",
    ],
]

with gr.Blocks(title="AzOCR-V2 — Multilingual Academic OCR Studio") as demo:
    # State tracking the active language code ("az", "tr", "en")
    current_lang_state = gr.State(value="az")

    # Top Prominent Academic Language Selector Toolbar
    top_bar_html = gr.HTML(value=build_top_bar_html("az"), elem_id="az-topbar-container")

    # Native hidden trigger buttons for 100% reliable Python execution
    with gr.Row(elem_classes=["az-hidden-triggers"]):
        btn_trig_az = gr.Button("az", elem_id="az-trig-az")
        btn_trig_tr = gr.Button("tr", elem_id="az-trig-tr")
        btn_trig_en = gr.Button("en", elem_id="az-trig-en")

    # Formal Institutional Header (Reactive to Language)
    hero_html = gr.HTML(value=build_hero_html("az"), elem_id="az-hero-container")

    # QR code to open the app on a phone (initial LAN QR, auto-upgraded to Public Gradio URL via /api/qr)
    qr_html = gr.HTML(value=build_qr_html(), elem_id="az-qr-container")

    with gr.Tabs(overflow_behavior="wrap") as main_tabs:
        # ======================================================================
        # SECTION 1: DOCUMENT & IMAGE ANALYSIS
        # ======================================================================
        with gr.TabItem(I18N["az"]["tab1"], id="tab_studio") as tab_studio:
            with gr.Row(elem_id="az-studio-row", elem_classes=["az-responsive-row"]):
                with gr.Column(scale=5, elem_id="az-studio-col-left"):
                    studio_img = gr.Image(
                        label=I18N["az"]["studio_img"],
                        sources=["upload", "clipboard"],
                        type="numpy",
                        height=230,
                        placeholder=I18N["az"]["upload_ph"],
                        elem_id="az-studio-img",
                    )
                    studio_model = gr.Dropdown(
                        label=I18N["az"]["studio_model"],
                        choices=list(MODEL_CONFIGS.keys()),
                        value=DEFAULT_MODEL_KEY,
                        filterable=False,
                        elem_id="az-studio-model",
                    )
                    with gr.Row(elem_id="az-checkbox-row"):
                        studio_pp = gr.Checkbox(
                            label=I18N["az"]["studio_pp"],
                            value=True,
                            container=False,
                            elem_id="az-studio-pp",
                        )
                        studio_multiline = gr.Checkbox(
                            label=I18N["az"]["studio_multiline"],
                            value=True,
                            container=False,
                            elem_id="az-studio-multiline",
                        )
                    with gr.Row():
                        studio_btn = gr.Button(I18N["az"]["studio_btn"], variant="primary", size="lg", scale=3, elem_id="az-studio-btn")
                        studio_clear = gr.Button(I18N["az"]["studio_clear"], variant="secondary", size="lg", scale=1, elem_id="az-studio-clear")

                    studio_norm_preview = gr.Image(
                        label=I18N["az"]["studio_preview"],
                        type="numpy",
                        interactive=False,
                        height=115,
                        elem_id="az-studio-preview",
                    )

                with gr.Column(scale=6, elem_id="az-studio-col-right"):
                    studio_out_text = gr.Textbox(
                        label=I18N["az"]["studio_out"],
                        placeholder=I18N["az"]["studio_placeholder"],
                        lines=4,
                        elem_classes=["az-output-box"],
                        elem_id="az-studio-out",
                    )
                    studio_diff_html = gr.HTML(
                        value=build_diff_html("", "", "az"),
                        label="Mərhələli Çıxış Analizi",
                        elem_id="az-diff-container",
                    )
                    studio_metrics_html = gr.HTML(
                        value=build_telemetry_html(0.0, 0, "", DEFAULT_MODEL_KEY, True, "az"),
                        label="Telemetriya",
                        elem_id="az-telemetry-container",
                    )

            valid_studio_examples = [
                [ex[0], ex[1]]
                for ex in BENCHMARK_SAMPLES
                if Path(ex[0]).exists()
            ]
            if valid_studio_examples:
                gr.Examples(
                    examples=valid_studio_examples,
                    inputs=[studio_img, studio_out_text],
                    outputs=[studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
                    fn=run_studio_example,
                    run_on_click=True,
                    cache_examples=False,
                    label=I18N["az"]["examples_label1"],
                )

            studio_btn.click(
                fn=ocr_studio_recognize,
                inputs=[studio_img, studio_model, studio_pp, studio_multiline, current_lang_state],
                outputs=[studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
            )
            studio_clear.click(
                fn=lambda lang: (
                    None,
                    "",
                    build_diff_html("", "", lang),
                    build_telemetry_html(0.0, 0, "", DEFAULT_MODEL_KEY, True, lang),
                    None
                ),
                inputs=[current_lang_state],
                outputs=[studio_img, studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
            )

        # ======================================================================
        # SECTION 2: EXPERIMENTAL MODEL COMPARISON (BOOKTABS TABLE)
        # ======================================================================
        with gr.TabItem(I18N["az"]["tab2"], id="tab_bench") as tab_bench:
            bench_desc_md = gr.Markdown(value=I18N["az"]["bench_desc"], elem_id="az-bench-desc")
            with gr.Row(elem_id="az-bench-row", elem_classes=["az-responsive-row"]):
                with gr.Column(scale=5, elem_id="az-bench-col-left"):
                    bench_img = gr.Image(
                        label=I18N["az"]["bench_img"],
                        sources=["upload", "clipboard"],
                        type="numpy",
                        height=180,
                        placeholder=I18N["az"]["upload_ph"],
                        elem_id="az-bench-img",
                    )
                with gr.Column(scale=6, elem_id="az-bench-col-right"):
                    bench_gt = gr.Textbox(
                        label=I18N["az"]["bench_gt"],
                        placeholder=I18N["az"]["bench_gt_placeholder"],
                        lines=3,
                        elem_id="az-bench-gt",
                    )
                    bench_btn = gr.Button(
                        I18N["az"]["bench_btn"],
                        variant="primary",
                        size="lg",
                        elem_id="az-bench-btn",
                    )

            bench_table_html = gr.HTML(
                value=run_benchmark_comparison(None, "", "az"),
                label="Eksperimental Müqayisə Cədvəli",
                elem_id="az-bench-table",
            )

            valid_bench_examples = [
                [ex[0], ex[1]] for ex in BENCHMARK_SAMPLES if Path(ex[0]).exists()
            ]
            if valid_bench_examples:
                gr.Examples(
                    examples=valid_bench_examples,
                    inputs=[bench_img, bench_gt],
                    outputs=[bench_table_html],
                    fn=run_benchmark_comparison,
                    run_on_click=True,
                    cache_examples=False,
                    label="Etalon Mətnli Eksperimental Nümunələr (Seçildikdə CER/WER cədvəli qurulur)",
                )

            bench_btn.click(
                fn=run_benchmark_comparison,
                inputs=[bench_img, bench_gt, current_lang_state],
                outputs=[bench_table_html],
            )

        # ======================================================================
        # SECTION 3: REAL-TIME CAMERA MODULE
        # ======================================================================
        with gr.TabItem(I18N["az"]["tab3"], id="tab_camera") as tab_camera:
            gr.HTML(LIVE_CAMERA_HTML)

    # Reactive Language Switcher Event Handler
    def on_language_trigger(target_lang: str):
        global ACTIVE_LANG
        ACTIVE_LANG = target_lang
        T = I18N.get(target_lang, I18N["az"])
        return (
            build_top_bar_html(target_lang),
            build_hero_html(target_lang),
            build_diff_html("", "", target_lang),
            build_telemetry_html(0.0, 0, "", DEFAULT_MODEL_KEY, True, target_lang),
            T["bench_desc"],
            run_benchmark_comparison(None, "", target_lang),
            target_lang,
            gr.update(label=T["tab1"]),
            gr.update(label=T["tab2"]),
            gr.update(label=T["tab3"]),
            gr.update(label=T["studio_img"], placeholder=T["upload_ph"]),
            gr.update(label=T["studio_model"]),
            gr.update(label=T["studio_pp"]),
            gr.update(label=T["studio_multiline"]),
            gr.update(value=T["studio_btn"]),
            gr.update(value=T["studio_clear"]),
            gr.update(label=T["studio_preview"]),
            gr.update(label=T["studio_out"], placeholder=T["studio_placeholder"]),
            gr.update(label=T["bench_img"], placeholder=T["upload_ph"]),
            gr.update(label=T["bench_gt"], placeholder=T["bench_gt_placeholder"]),
            gr.update(value=T["bench_btn"]),
        )

    common_lang_outputs = [
        top_bar_html,
        hero_html,
        studio_diff_html,
        studio_metrics_html,
        bench_desc_md,
        bench_table_html,
        current_lang_state,
        tab_studio,
        tab_bench,
        tab_camera,
        studio_img,
        studio_model,
        studio_pp,
        studio_multiline,
        studio_btn,
        studio_clear,
        studio_norm_preview,
        studio_out_text,
        bench_img,
        bench_gt,
        bench_btn,
    ]

    btn_trig_az.click(fn=lambda: on_language_trigger("az"), outputs=common_lang_outputs, api_name="set_lang_az")
    btn_trig_tr.click(fn=lambda: on_language_trigger("tr"), outputs=common_lang_outputs, api_name="set_lang_tr")
    btn_trig_en.click(fn=lambda: on_language_trigger("en"), outputs=common_lang_outputs, api_name="set_lang_en")


def register_api_routes(fastapi_app):
    """Attach the /api/ocr and /api/qr JSON endpoints to the underlying FastAPI server."""
    @fastapi_app.get("/api/qr")
    async def api_qr_endpoint():
        return JSONResponse({
            "url": SHARE_URL or LAN_URL,
            "is_public": bool(SHARE_URL),
            "html": build_qr_html(),
        })
    fastapi_app.router.routes.insert(0, fastapi_app.router.routes.pop())

    @fastapi_app.post("/api/ocr")
    async def api_ocr_endpoint(req: Request):
        try:
            body = await req.json()
            b64_str = body.get("image", "")
            engine_req = body.get("engine", "crnn_v2")
            multi_line = bool(body.get("multi_line", False))
            use_post = bool(body.get("postprocess", True))
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return JSONResponse({"error": "Invalid image"}, status_code=400)

            Path("scratch").mkdir(exist_ok=True)
            cv2.imwrite("scratch/last_mobile_request.jpg", img_bgr)
            h_in, w_in = img_bgr.shape[:2]
            print(f"[*] /api/ocr gələn şəkil: {w_in}x{h_in}, model: {engine_req}")

            model_choice = TESSERACT_MODEL_KEY if engine_req == "tesseract" else DEFAULT_MODEL_KEY
            final_pred, raw_pred, elapsed_ms, n_lines = run_ocr_on_bgr(
                img_bgr,
                model_choice=model_choice,
                use_postprocess=use_post,
                multi_line=multi_line,
            )
            return JSONResponse({
                "text": final_pred,
                "raw": raw_pred,
                "engine": "Tesseract 5 (aze)" if engine_req == "tesseract" else "CRNN-V2",
                "ms": round(elapsed_ms, 1),
                "lines": n_lines,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    fastapi_app.router.routes.insert(0, fastapi_app.router.routes.pop())


def main():
    parser = argparse.ArgumentParser(description="AzOCR-V2 Çoxdilli Elmi-Tədqiqat Veb Tətbiqi")
    parser.add_argument("--share", action="store_true", help="Telefonda istifadə üçün ictimai Gradio linki yaradın")
    parser.add_argument("--port", type=int, default=7860, help="Server portu (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dinləniləcək IP (default: 0.0.0.0)")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 65)
    print("AzOCR-V2 ELMI-TEDQIQAT VE KAMERA SERVERI ISE SALINIR")
    print("=" * 65)
    print(f"Lokal unvan (Komputer):  http://localhost:{args.port}")
    print(f"Lokal sebeke (Wi-Fi):    http://{local_ip}:{args.port}")
    if args.share:
        print("Ictimai HTTPS linki hazirlanir...")
    print("=" * 65)

    # Enforce light mode institutional theme
    formal_theme = gr.themes.Default(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="gray",
        radius_size=gr.themes.sizes.radius_none,
        font=[gr.themes.GoogleFont("Source Serif 4"), "Times New Roman", "Georgia", "serif"],
        font_mono=["Consolas", "Courier New", "monospace"],
    )

    app_instance, local_url, share_url = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
        theme=formal_theme,
        css=custom_css,
        head=CUSTOM_HEAD_SCRIPT + "<script>document.documentElement.classList.remove('dark');</script>",
        pwa=True,
        prevent_thread_lock=True,
    )
    register_api_routes(app_instance)

    if args.share and not share_url:
        try:
            from gradio.networking import setup_tunnel
            import secrets
            share_token = secrets.token_hex(16)
            share_url = setup_tunnel("127.0.0.1", args.port, share_token, None, None)
        except Exception as te:
            print(f"[!] Tunnel fallback xətası: {te}")

    global SHARE_URL, LAN_URL
    SHARE_URL = (share_url or "").strip()
    LAN_URL = f"http://{local_ip}:{args.port}"

    if share_url:
        try:
            with open("public_url.txt", "w", encoding="utf-8") as f:
                f.write(share_url.strip() + "\n")
        except Exception:
            pass
        print("\n" + "=" * 65)
        print("📱 SİZİN İCTİMAİ TELEFON VƏ CANLI KAMERA LİNKİNİZ:")
        print(f"👉  {share_url}")
        print("💡 Qeyd: Bu link 'public_url.txt' faylında da saxlanıldı.")
        print("=" * 65 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Server dayandırıldı.")


if __name__ == "__main__":
    main()
