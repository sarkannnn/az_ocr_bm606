#!/usr/bin/env python3
"""AzOCR-V2: Azərbaycan Dili üçün Elmi-Tədqiqat və Sənaye OCR Platforması (Gradio 6 + FastAPI).

Xüsusiyyətlər:
1. 📄 Sənəd və Şəkil Studiyası:
   - Yüksək dəqiqlikli sətir və tam səhifə oxuma, vizual preprosesinq zolağı (32x512 tenzor görünüşü),
     Xam CTC çıxışı ilə Morfoloji SymSpell korreksiyasının fərq analizi (Diff) və telemetriya kartları.
2. ⚖️ Elmi Müqayisə Stendi (Məqalə və Eksperimentlər üçün):
   - Eyni təsviri paralel olaraq CRNN-V2 (SymSpell), CRNN-V2 (Xam CTC), CRNN-V1 (Baza) və
     Tesseract 5 (aze) modellərində sınaqdan keçirir; Etalon mətn (Ground Truth) verildikdə
     avtomatik CER (%), WER (%), Dəqiqlik (%) və Sürət (ms) cədvəli qurur.
3. 📱 Canlı Mobil Kamera Skaneri:
   - Smartfonlarda birbaşa arxa kameranı açır, lazer bələdçi xətti və doğma makro kamera (HD)
     dəstəyi ilə real-vaxt rejimində mətn oxuyur.
"""

import argparse
import base64
import html
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import pytesseract
import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from PIL import Image

# Setup Tesseract Environment
os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"D:\laragon\www\az_ocr_bm606\tessdata"

from common import (
    cer_wer,
    infer_line_smart,
    load_model_checkpoint,
    normalize,
    preprocess,
    robust_trim_chunk,
    segment_lines_opencv,
)
from postprocess import AzerbaijaniPostProcessor

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Model definitions
DEFAULT_MODEL_KEY = "CRNN Sətir Modeli (V2 - 32×512) [Təklif olunan]"
WORD_V2_MODEL_KEY = "CRNN Söz Modeli (V2 - 32×128)"
LINE_V1_MODEL_KEY = "CRNN Sətir Modeli (V1 - Baza)"
WORD_V1_MODEL_KEY = "CRNN Söz Modeli (V1 - Baza)"
TESSERACT_MODEL_KEY = "Tesseract 5 OCR (aze - Sənaye Baseline)"

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
    """Get local LAN IP address so user can connect from mobile phone."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


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
            return f"Tess xətası: {e}"

    lines = extract_tight_text_lines(img_bgr)
    out_lines = []
    for line in lines:
        r = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
        try:
            t = pytesseract.image_to_string(Image.fromarray(r), lang="aze", config="--psm 7").strip()
            if t:
                out_lines.append(normalize(t))
        except Exception as e:
            out_lines.append(f"Tess xətası: {e}")
    return "\n".join(out_lines)


def run_ocr_on_bgr(
    img_bgr: np.ndarray,
    model_choice: str = DEFAULT_MODEL_KEY,
    use_postprocess: bool = True,
    multi_line: bool = True,
) -> Tuple[str, str, float, int]:
    """Run smart line/multi-line OCR on a BGR numpy image."""
    start_time = time.perf_counter()

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
        raw_lines.append(rp)
        if use_postprocess and postprocessor and postprocessor.loaded:
            tokens = rp.split()
            corrected_tokens = [postprocessor.correct_word(t, max_edit_distance=1) for t in tokens]
            final_lines.append(postprocessor.clean_punctuation(" ".join(corrected_tokens)))
        else:
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
        _, cfg, _, _ = get_model(DEFAULT_MODEL_KEY)
        # Render normalized 32px height strip
        gray = cv2.cvtColor(first_crop, cv2.COLOR_BGR2GRAY) if first_crop.ndim == 3 else first_crop
        h, w = gray.shape[:2]
        target_h = 48
        new_w = max(32, min(900, int(round(w * target_h / max(1, h)))))
        resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return rgb
    except Exception:
        return None


def build_diff_html(raw_text: str, final_text: str) -> str:
    """Build an academic visual diff card comparing Raw CTC vs. SymSpell Post-Processed text."""
    if not raw_text.strip() and not final_text.strip():
        return """
        <div class="az-diff-card">
            <span style="color:#64748b;">Emal gözlənilir — təsvir seçib "Mətni Tanı" düyməsini sıxın.</span>
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
        badge = f'<span class="az-badge-corr">✨ Lüğət Korreksiyası: {len(corrections)} söz düzəldildi</span>'
    else:
        badge = '<span class="az-badge-clean">✓ Xam CTC çıxışı morfoloji cəhətdən tam dəqiqdir (0 düzəliş)</span>'

    return f"""
    <div class="az-diff-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; flex-wrap:wrap; gap:6px;">
            <strong style="font-size:12.5px; color:#334155; text-transform:uppercase; letter-spacing:0.4px;">
                🔬 Xam CTC Çıxışı və Morfoloji Post-Prosesinq Müqayisəsi
            </strong>
            {badge}
        </div>
        <div style="font-family:'JetBrains Mono', 'Consolas', monospace; font-size:13.5px; color:#1e293b; background:#f8fafc; padding:8px 10px; border-radius:6px; border:1px solid #e2e8f0; margin-bottom:6px;">
            <span style="color:#64748b; font-size:11.5px;">XAM CTC:</span> {html.escape(raw_text)}
        </div>
        <div style="font-size:14px; color:#0f172a; line-height:1.5;">
            <span style="color:#64748b; font-size:11.5px; font-family:monospace;">KORREKSİYA:</span> {diff_body}
        </div>
    </div>
    """


def build_telemetry_html(elapsed_ms: float, n_lines: int, final_text: str, model_choice: str, use_postprocess: bool) -> str:
    """Build 4 clean academic metric cards."""
    device_name = "CPU"
    if model_choice != TESSERACT_MODEL_KEY:
        _, _, _, device = get_model(model_choice)
        device_name = device.type.upper()

    n_chars = len(final_text.strip())
    n_words = len(final_text.split())
    engine_short = "Tesseract 5 (aze)" if model_choice == TESSERACT_MODEL_KEY else model_choice.split("[")[0].strip()
    ms_per_line = elapsed_ms / max(1, n_lines)

    return f"""
    <div class="az-metrics-grid">
        <div class="az-metric-box">
            <div class="az-metric-label">⏱️ İCRA MÜDDƏTİ</div>
            <div class="az-metric-val">{elapsed_ms:.1f} <span style="font-size:13px;font-weight:500;">ms</span></div>
            <div class="az-metric-sub">{ms_per_line:.1f} ms / sətir</div>
        </div>
        <div class="az-metric-box">
            <div class="az-metric-label">📏 SEQMENTASİYA</div>
            <div class="az-metric-val">{n_lines} <span style="font-size:13px;font-weight:500;">sətir</span></div>
            <div class="az-metric-sub">Avto-proyeksiya kəsimi</div>
        </div>
        <div class="az-metric-box">
            <div class="az-metric-label">🔤 MƏTN HƏCMİ</div>
            <div class="az-metric-val">{n_words} <span style="font-size:13px;font-weight:500;">söz</span></div>
            <div class="az-metric-sub">{n_chars} simvol tanındı</div>
        </div>
        <div class="az-metric-box">
            <div class="az-metric-label">⚙️ AKTİV MÜHƏRRİK</div>
            <div class="az-metric-val" style="font-size:14px;">{html.escape(engine_short)}</div>
            <div class="az-metric-sub">Cihaz: {device_name} | Lüğət: {'Aktiv' if use_postprocess else 'Deaktiv'}</div>
        </div>
    </div>
    """


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def ocr_studio_recognize(
    img: Optional[np.ndarray],
    model_choice: str,
    use_postprocess: bool,
    multi_line: bool,
) -> Tuple[str, str, str, Optional[np.ndarray]]:
    """Main callback for Tab 1 (Document & Image Studio)."""
    if img is None:
        empty_msg = "Zəhmət olmasa təsvir yükləyin və ya aşağıdakı hazır etalon nümunələrdən birini seçin."
        return empty_msg, build_diff_html("", ""), "", None

    img_bgr = to_bgr(img)
    final_pred, raw_pred, elapsed_ms, n_lines = run_ocr_on_bgr(
        img_bgr,
        model_choice=model_choice,
        use_postprocess=use_postprocess,
        multi_line=multi_line,
    )
    diff_html = build_diff_html(raw_pred, final_pred)
    telemetry_html = build_telemetry_html(elapsed_ms, n_lines, final_pred, model_choice, use_postprocess)
    preview_rgb = make_preprocessed_preview(img_bgr, multi_line=multi_line)
    return final_pred, diff_html, telemetry_html, preview_rgb


def run_benchmark_comparison(img: Optional[np.ndarray], ground_truth: str) -> str:
    """Callback for Tab 2: Side-by-Side Academic Model Comparison."""
    if img is None:
        return """
        <div class="az-diff-card" style="text-align:center; padding:24px;">
            ⚠️ Müqayisə üçün zəhmət olmasa şəkil yükləyin və ya aşağıdakı <b>Etalon Test Nümunələrindən</b> birinə klikləyin.
        </div>
        """

    img_bgr = to_bgr(img)
    gt = normalize(ground_truth.strip()) if ground_truth and ground_truth.strip() else ""

    # Run all 4 configurations
    configs = [
        ("🏆 Təklif olunan CRNN-V2 + SymSpell", DEFAULT_MODEL_KEY, True, "#16a34a", True),
        ("🧠 Təklif olunan CRNN-V2 (Xam CTC)", DEFAULT_MODEL_KEY, False, "#2563eb", False),
        ("📦 Baza Modeli CRNN-V1 (Əvvəlki nəsil)", LINE_V1_MODEL_KEY, False, "#64748b", False),
        ("🏢 Tesseract 5 LSTM (aze - Sənaye)", TESSERACT_MODEL_KEY, False, "#d97706", False),
    ]

    rows_html = []
    for label, m_key, use_pp, accent_color, is_featured in configs:
        pred, raw_p, ms, n_l = run_ocr_on_bgr(img_bgr, model_choice=m_key, use_postprocess=use_pp, multi_line=True)
        pred_clean = " ".join(pred.split())

        if gt:
            cer_val, wer_val, _ = cer_wer([gt], [pred_clean])
            cer_pct = cer_val * 100.0
            wer_pct = wer_val * 100.0
            char_acc = max(0.0, 100.0 - cer_pct)
            cer_badge = f'<span style="font-weight:700; color:{"#15803d" if cer_pct < 2 else "#b45309" if cer_pct < 10 else "#dc2626"};">{cer_pct:.2f}%</span>'
            wer_badge = f'{wer_pct:.2f}%'
            acc_badge = f'<span class="az-acc-pill" style="background:{"#dcfce7" if char_acc >= 98 else "#fef3c7" if char_acc >= 90 else "#fee2e2"}; color:{"#166534" if char_acc >= 98 else "#92400e" if char_acc >= 90 else "#991b1b"};">{char_acc:.2f}%</span>'
        else:
            cer_badge = '<span style="color:#94a3b8;">—</span>'
            wer_badge = '<span style="color:#94a3b8;">—</span>'
            acc_badge = '<span style="color:#94a3b8;">Etalon yoxdur</span>'

        row_style = "background:#f0fdf4; border-left:4px solid #16a34a;" if is_featured else ""
        rows_html.append(f"""
        <tr style="{row_style}">
            <td style="font-weight:700; color:#0f172a; white-space:nowrap;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{accent_color}; margin-right:6px;"></span>
                {html.escape(label)}
            </td>
            <td style="font-family:'Inter', sans-serif; font-size:14.5px; color:#1e293b; font-weight:{'600' if is_featured else '400'};">
                {html.escape(pred_clean) if pred_clean else '<em style="color:#94a3b8;">(Boş çıxış)</em>'}
            </td>
            <td style="text-align:center; font-family:monospace; font-weight:600;">{ms:.1f} ms</td>
            <td style="text-align:center;">{cer_badge}</td>
            <td style="text-align:center;">{wer_badge}</td>
            <td style="text-align:center;">{acc_badge}</td>
        </tr>
        """)

    gt_banner = ""
    if gt:
        gt_banner = f"""
        <div style="background:#f8fafc; border:1px solid #cbd5e1; border-left:4px solid #0f172a; border-radius:8px; padding:10px 14px; margin-bottom:12px;">
            <div style="font-size:11.5px; font-weight:700; color:#475569; text-transform:uppercase;">📌 Etalon Mətn (Ground Truth Reference):</div>
            <div style="font-size:15px; font-weight:600; color:#0f172a; margin-top:2px;">{html.escape(gt)}</div>
        </div>
        """

    return f"""
    <div class="az-bench-wrap">
        {gt_banner}
        <div style="overflow-x:auto;">
            <table class="az-bench-table">
                <thead>
                    <tr>
                        <th>OCR Mühərriki / Konfiqurasiya</th>
                        <th>Tanınmış Mətn Çıxışı (Hypothesis)</th>
                        <th style="text-align:center;">Müddət</th>
                        <th style="text-align:center;">CER ↓</th>
                        <th style="text-align:center;">WER ↓</th>
                        <th style="text-align:center;">Simvol Dəqiqliyi ↑</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
    </div>
    """


# ==============================================================================
# HTML5 LIVE CAMERA SCANNER COMPONENT (TAB 3)
# ==============================================================================
LIVE_CAMERA_HTML = """
<div id="az-cam-container" style="background: #0f172a; color: #f8fafc; border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.22); border: 1px solid #1e293b; max-width: 720px; margin: 0 auto;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
      <span style="display: inline-block; width: 10px; height: 10px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e;"></span>
      <strong style="font-size: 15px; letter-spacing: 0.3px;">📱 Canlı Mobil Kamera Skaneri</strong>
    </div>
    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
      <button type="button" onclick="window.azSetBoxMode('line')" id="btn-mode-line" style="background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Tək Sətir</button>
      <button type="button" onclick="window.azSetBoxMode('para')" id="btn-mode-para" style="background: #334155; color: #cbd5e1; border: none; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Paraqraf</button>
      <button type="button" onclick="window.azSetBoxMode('full')" id="btn-mode-full" style="background: #334155; color: #cbd5e1; border: none; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">Tam Ekran</button>
    </div>
  </div>

  <!-- OCR Model Engine Switcher -->
  <div style="display: flex; align-items: center; justify-content: space-between; background: #1e293b; border-radius: 8px; padding: 6px 10px; margin-bottom: 10px; border: 1px solid #334155;">
    <span style="font-size: 12px; color: #94a3b8; font-weight: 600;">🧠 Aktiv Mühərrik:</span>
    <div style="display: flex; gap: 6px;">
      <button type="button" onclick="window.azSetEngine('crnn_v2')" id="btn-eng-crnn_v2" style="background: #16a34a; color: #fff; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 700; cursor: pointer;">🚀 CRNN-V2 (Bizim Model)</button>
      <button type="button" onclick="window.azSetEngine('tesseract')" id="btn-eng-tesseract" style="background: #334155; color: #cbd5e1; border: none; border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 700; cursor: pointer;">🏢 Tesseract 5 (aze)</button>
    </div>
  </div>

  <!-- Video Viewport with Green OCR Target Guide Overlay -->
  <div id="az-video-stage" style="position: relative; width: 100%; background: #000; border-radius: 12px; overflow: hidden; min-height: 250px; max-height: 52vh; display: flex; align-items: center; justify-content: center; border: 1px solid #334155;">
    <video id="az-cam-video" autoplay playsinline muted style="width: 100%; height: 100%; max-height: 52vh; object-fit: cover; display: block;"></video>

    <!-- Darkened mask outside target box & glowing green guide box -->
    <div id="az-scan-box" style="position: absolute; left: 4%; right: 4%; top: 40%; height: 18%; border: 2.5px solid #22c55e; border-radius: 10px; box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.52), inset 0 0 15px rgba(34, 197, 94, 0.25); pointer-events: none; transition: all 0.2s ease;">
      <div style="position: absolute; left: 0; right: 0; top: 50%; height: 1.5px; background: rgba(239, 68, 68, 0.85); box-shadow: 0 0 8px #ef4444;"></div>
      <div style="position: absolute; top: -24px; left: 6px; background: #22c55e; color: #022c22; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">
        🎯 MƏTNİ QIRMIZI XƏTTƏ GƏTİRİN
      </div>
    </div>

    <!-- Fallback / Permission Prompt -->
    <div id="az-cam-msg" style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(15, 23, 42, 0.88); padding: 20px; text-align: center; z-index: 5;">
      <p id="az-cam-msg-text" style="margin: 0 0 12px 0; font-size: 14px; color: #e2e8f0;">Kamera başlatılır...</p>
      <button type="button" onclick="window.azStartCamera()" style="background: #22c55e; color: #022c22; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 700; cursor: pointer;">
        📷 Kameranı Aktivləşdir
      </button>
    </div>
  </div>

  <!-- Camera Action Buttons -->
  <div style="display: grid; grid-template-columns: 2fr 1.4fr auto auto; gap: 8px; margin-top: 12px;">
    <button type="button" onclick="window.azSnapNow()" id="az-btn-snap" style="background: linear-gradient(135deg, #22c55e, #16a34a); color: #022c22; border: none; border-radius: 10px; padding: 13px 12px; font-size: 16px; font-weight: 800; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 6px; box-shadow: 0 4px 12px rgba(34,197,94,0.35);">
      📸 Çək və Oxu
    </button>
    <button type="button" onclick="window.azToggleAuto()" id="az-btn-auto" style="background: #1e293b; color: #f8fafc; border: 1.5px solid #475569; border-radius: 10px; padding: 13px 10px; font-size: 14px; font-weight: 700; cursor: pointer;">
      🔄 Canlı Oxu: OFF
    </button>
    <button type="button" onclick="window.azSwitchCam()" title="Arxa/Ön Kamera" style="background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 10px; padding: 12px; font-size: 16px; cursor: pointer;">
      🔁
    </button>
    <button type="button" onclick="window.azToggleTorch()" id="az-btn-torch" title="Fənər (Flash)" style="background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 10px; padding: 12px; font-size: 16px; cursor: pointer;">
      🔦
    </button>
  </div>

  <!-- Native Phone Camera Shutter for Ultra-sharp Focus -->
  <div style="margin-top: 8px;">
    <label style="display: flex; align-items: center; justify-content: center; gap: 8px; background: #334155; color: #f8fafc; border: 1.5px dashed #64748b; border-radius: 10px; padding: 10px 14px; font-size: 14px; font-weight: 700; cursor: pointer; text-align: center;">
      🎯 <span>Telefonun Doğma Kamerası ilə Çək (Tam Aydın / Makro Fokus)</span>
      <input type="file" accept="image/*" capture="environment" onchange="window.azHandleNativeFile(this)" style="display: none;" />
    </label>
  </div>

  <!-- Live Result Display right below camera -->
  <div style="margin-top: 12px; background: #1e293b; border-radius: 12px; padding: 12px; border: 1px solid #334155;">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
      <span id="az-live-status" style="font-size: 12px; color: #94a3b8;">Hazırdır — "📸 Çək və Oxu" və ya "🎯 Doğma Kamera" seçin</span>
      <button type="button" onclick="window.azCopyResult()" id="az-btn-copy" style="background: #3b82f6; color: #fff; border: none; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">
        📋 Kopyala
      </button>
    </div>
    <textarea id="az-live-result" rows="3" placeholder="Kameradan oxunan Azərbaycan dilindəki mətn dərhal burada görünəcək..." style="width: 100%; background: #0f172a; color: #f8fafc; border: 1px solid #475569; border-radius: 8px; padding: 10px; font-size: 18px; font-weight: 600; line-height: 1.4; resize: vertical; box-sizing: border-box;"></textarea>
    <div style="margin-top: 8px;">
      <div style="font-size: 11px; color: #64748b; margin-bottom: 3px;">Modelin gördüyü son kəsim:</div>
      <canvas id="az-crop-preview" style="height: 48px; max-width: 100%; border-radius: 6px; border: 1px solid #475569; background: #000; display: block;"></canvas>
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
  torchOn: false
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
      b.style.background = (k === mode) ? "#2563eb" : "#334155";
      b.style.color = (k === mode) ? "#fff" : "#cbd5e1";
    }
  });
};

window.azSetEngine = function(eng) {
  window.azState.engine = eng;
  ["crnn_v2", "tesseract"].forEach(function(k) {
    var b = document.getElementById("btn-eng-" + k);
    if (b) {
      b.style.background = (k === eng) ? "#16a34a" : "#334155";
      b.style.color = (k === eng) ? "#fff" : "#cbd5e1";
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
    if (msgText) msgText.innerText = "⚠️ Brauzer kameranı dəstəkləmir və ya HTTPS linki lazımdır (gradio.live linkindən istifadə edin).";
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
    if (msgText) msgText.innerText = "📷 Kameranı açmaq üçün aşağıdakı düyməyə toxunun və icazə (Allow) verin:";
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
    if (btn) btn.style.background = window.azState.torchOn ? "#eab308" : "#1e293b";
  } catch (e) {
    alert("Bu cihazda fənər (flash) brauzerdən idarə oluna bilmir.");
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
  var statusEl = document.getElementById("az-live-status");
  if (statusEl) statusEl.innerText = "⏳ Oxunur...";

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
      resArea.value = data.text || "(Çərçivədə mətn tapılmadı — kameranı yazıya yaxınlaşdırın)";
    }
    if (statusEl) {
      var engLabel = data.engine || (window.azState.engine === "tesseract" ? "Tesseract" : "CRNN V2");
      statusEl.innerText = "✅ " + engLabel + " (" + (data.ms || 0) + " ms | " + (data.lines || 1) + " sətir)";
    }
  } catch (err) {
    if (statusEl) statusEl.innerText = "⚠️ Xəta: " + err.message;
  } finally {
    window.azState.busy = false;
  }
};

window.azSnapNow = function() {
  window.azCaptureAndSend();
};

window.azToggleAuto = function() {
  var btn = document.getElementById("az-btn-auto");
  if (window.azState.autoTimer) {
    clearInterval(window.azState.autoTimer);
    window.azState.autoTimer = null;
    if (btn) {
      btn.innerText = "🔄 Canlı Oxu: OFF";
      btn.style.background = "#1e293b";
      btn.style.borderColor = "#475569";
    }
  } else {
    window.azCaptureAndSend();
    window.azState.autoTimer = setInterval(window.azCaptureAndSend, 1100);
    if (btn) {
      btn.innerText = "🔄 Canlı Oxu: ON";
      btn.style.background = "#dc2626";
      btn.style.borderColor = "#f87171";
    }
  }
};

window.azCopyResult = function() {
  var resArea = document.getElementById("az-live-result");
  var btn = document.getElementById("az-btn-copy");
  if (!resArea || !resArea.value) return;
  navigator.clipboard.writeText(resArea.value);
  if (btn) {
    var old = btn.innerText;
    btn.innerText = "✓ Kopyalandı!";
    setTimeout(function() { btn.innerText = old; }, 1500);
  }
};

window.azHandleNativeFile = function(input) {
  if (!input.files || !input.files[0]) return;
  var file = input.files[0];
  var reader = new FileReader();
  var statusEl = document.getElementById("az-live-status");
  if (statusEl) statusEl.innerText = "⏳ HD Şəkil oxunur...";
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
        resArea.value = data.text || "(Mətn tapılmadı)";
      }
      if (statusEl) {
        var engLabel = data.engine || (window.azState.engine === "tesseract" ? "Tesseract" : "CRNN V2");
        statusEl.innerText = "✅ " + engLabel + " (" + (data.ms || 0) + " ms | " + (data.lines || 1) + " sətir)";
      }
    } catch (err) {
      if (statusEl) statusEl.innerText = "⚠️ Xəta: " + err.message;
    }
  };
  reader.readAsDataURL(file);
};
</script>
"""

# ==============================================================================
# ACADEMIC PUBLICATION-GRADE CSS STYLING
# ==============================================================================
custom_css = """
.gradio-container {
    max-width: 1260px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.az-hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    color: #f8fafc;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    border: 1px solid #334155;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
}
.az-hero-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
}
.az-hero-title {
    margin: 0;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #ffffff;
}
.az-hero-sub {
    margin: 4px 0 0 0;
    font-size: 13.5px;
    color: #94a3b8;
    font-weight: 400;
}
.az-spec-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
}
.az-spec-pill {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid #475569;
    color: #e2e8f0;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.az-output-box textarea {
    font-size: 17px !important;
    font-weight: 600 !important;
    line-height: 1.55 !important;
    color: #0f172a !important;
    background: #f8fafc !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 10px !important;
}
.az-diff-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.az-diff-del {
    background: #fee2e2;
    color: #991b1b;
    text-decoration: line-through;
    padding: 1px 5px;
    border-radius: 4px;
    font-weight: 600;
}
.az-diff-ins {
    background: #dcfce7;
    color: #166534;
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 700;
    border: 1px solid #86efac;
}
.az-badge-corr {
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fcd34d;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 700;
}
.az-badge-clean {
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 700;
}
.az-metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 4px;
}
@media (max-width: 768px) {
    .az-metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
.az-metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: left;
}
.az-metric-label {
    font-size: 10.5px;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.5px;
}
.az-metric-val {
    font-size: 19px;
    font-weight: 800;
    color: #0f172a;
    margin: 2px 0;
}
.az-metric-sub {
    font-size: 11px;
    color: #64748b;
}
.az-bench-table {
    width: 100%;
    border-collapse: collapse;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #cbd5e1;
    font-size: 13.5px;
}
.az-bench-table th {
    background: #0f172a;
    color: #f8fafc;
    padding: 10px 12px;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.az-bench-table td {
    padding: 11px 12px;
    border-bottom: 1px solid #e2e8f0;
}
.az-acc-pill {
    padding: 3px 9px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12.5px;
}
"""

# Curated benchmark examples with Ground Truth for Tab 1 & Tab 2
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

with gr.Blocks(title="AzOCR-V2 — Azərbaycan Dili üçün Optik Mətn Tanıma Sistemi") as demo:
    # Academic Hero Banner
    gr.HTML("""
    <div class="az-hero-header">
        <div class="az-hero-top">
            <div>
                <h1 class="az-hero-title">🇦🇿 AzOCR-V2: Azərbaycan Dili üçün Hibrid Optik Mətn Tanıma Sistemi</h1>
                <p class="az-hero-sub">
                    Derin Konvolyusiya + İkiistiqamətli LSTM (BiLSTM) + CTC Dekodlama və Morfoloji SymSpell Lüğət Post-Prosesinqi
                </p>
            </div>
            <div style="background:#16a34a; color:#ffffff; font-size:12px; font-weight:700; padding:5px 12px; border-radius:8px; white-space:nowrap;">
                ✓ Elmi Tədqiqat Stendi (BM-606)
            </div>
        </div>
        <div class="az-spec-bar">
            <span class="az-spec-pill">🧠 <b>Arxitektura:</b> CRNN + 2×BiLSTM + CTC (4.05M Parametr)</span>
            <span class="az-spec-pill">📐 <b>Giriş Ölçüsü:</b> 32 × 512 Grayscale</span>
            <span class="az-spec-pill">🔤 <b>Əlifba:</b> 88 Simvol (Ç, Ə, Ğ, İ, I, Ö, Ş, Ü)</span>
            <span class="az-spec-pill">📚 <b>Leksikon:</b> 116,143 Söz + Morfoloji Şəkilçi Analizi</span>
            <span class="az-spec-pill">⚡ <b>Sürət:</b> ~14.2 ms / sətir</span>
        </div>
    </div>
    """)

    with gr.Tabs() as main_tabs:
        # ======================================================================
        # TAB 1: DOCUMENT & IMAGE STUDIO (DEFAULT FOR DESKTOP & PAPER FIGURES)
        # ======================================================================
        with gr.TabItem("📄 1. Sənəd və Şəkil Studiyası (Əsas Analiz)", id="tab_studio"):
            with gr.Row():
                # Left Column: Input & Controls
                with gr.Column(scale=5):
                    studio_img = gr.Image(
                        label="🖼️ Sənəd və ya Sətir Təsviri (Yüklə / Mübadilə Buferi)",
                        sources=["upload", "clipboard"],
                        type="numpy",
                        height=240,
                    )
                    with gr.Row():
                        studio_model = gr.Dropdown(
                            label="🧠 OCR Modeli / Mühərriki",
                            choices=list(MODEL_CONFIGS.keys()),
                            value=DEFAULT_MODEL_KEY,
                            scale=3,
                        )
                    with gr.Row():
                        studio_pp = gr.Checkbox(
                            label="📚 Morfoloji SymSpell Lüğət Korreksiyası",
                            value=True,
                        )
                        studio_multiline = gr.Checkbox(
                            label="📏 Avtomatik Çoxsətirli Seqmentasiya",
                            value=True,
                        )
                    with gr.Row():
                        studio_btn = gr.Button("🔍 Mətni Tanı və Analiz Et", variant="primary", size="lg", scale=3)
                        studio_clear = gr.Button("🗑️ Təmizlə", variant="secondary", size="lg", scale=1)

                    studio_norm_preview = gr.Image(
                        label="🔬 Modelin Gördüyü Normalizasiya Olunmuş Zolaq (32px Tenzo-Giriş)",
                        type="numpy",
                        interactive=False,
                        height=95,
                    )

                # Right Column: Outputs & Academic Diagnostics
                with gr.Column(scale=6):
                    studio_out_text = gr.Textbox(
                        label="✅ Yekun Tanınmış Mətn (Post-Processed Output)",
                        placeholder="Tanınmış mətn burada əks olunacaq...",
                        lines=4,
                        elem_classes=["az-output-box"],
                    )
                    studio_diff_html = gr.HTML(
                        value=build_diff_html("", ""),
                        label="Xam CTC və Lüğət Fərqi",
                    )
                    studio_metrics_html = gr.HTML(
                        value=build_telemetry_html(0.0, 0, "", DEFAULT_MODEL_KEY, True),
                        label="Telemetriya",
                    )

            # Curated Benchmark Examples for Tab 1
            valid_studio_examples = [
                [ex[0], DEFAULT_MODEL_KEY, True, True]
                for ex in BENCHMARK_SAMPLES
                if Path(ex[0]).exists()
            ]
            if valid_studio_examples:
                gr.Examples(
                    examples=valid_studio_examples,
                    inputs=[studio_img, studio_model, studio_pp, studio_multiline],
                    outputs=[studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
                    fn=ocr_studio_recognize,
                    run_on_click=True,
                    cache_examples=False,
                    label="💡 Hazır Kamera və Sənəd Test Nümunələri (Kliklədikdə dərhal icra olunur)",
                )

            studio_btn.click(
                fn=ocr_studio_recognize,
                inputs=[studio_img, studio_model, studio_pp, studio_multiline],
                outputs=[studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
            )
            studio_clear.click(
                fn=lambda: (None, "", build_diff_html("", ""), build_telemetry_html(0.0, 0, "", DEFAULT_MODEL_KEY, True), None),
                inputs=[],
                outputs=[studio_img, studio_out_text, studio_diff_html, studio_metrics_html, studio_norm_preview],
            )

        # ======================================================================
        # TAB 2: SIDE-BY-SIDE MODEL COMPARISON (BENCHMARK TABLE FOR PAPER)
        # ======================================================================
        with gr.TabItem("⚖️ 2. Elmi Müqayisə Stendi (CRNN-V2 vs. Tesseract 5)", id="tab_bench"):
            gr.Markdown("""
            ### 📊 Modellərarası Eksperimental Müqayisə (Ablation & Baseline Benchmark)
            Bu bölmə eyni təsviri **4 fərqli konfiqurasiyada** (*CRNN-V2 + SymSpell*, *CRNN-V2 Xam CTC*, *CRNN-V1 Baza*, *Tesseract 5 `aze`*) paralel icra edir.
            Əgər **Etalon Mətn (Ground Truth)** daxil edilərsə, hər bir model üçün **CER (%)**, **WER (%)** və **Simvol Dəqiqliyi (%)** avtomatik hesablanır.
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    bench_img = gr.Image(
                        label="🖼️ Müqayisə Ediləcək Təsvir",
                        sources=["upload", "clipboard"],
                        type="numpy",
                        height=190,
                    )
                with gr.Column(scale=6):
                    bench_gt = gr.Textbox(
                        label="📌 Etalon Mətn (Ground Truth — CER/WER hesablanması üçün, istəyə bağlı)",
                        placeholder="Məsələn: Gəncə çayının kənarında, qədim ağacların kölgəsinə sığınmış...",
                        lines=3,
                    )
                    bench_btn = gr.Button(
                        "🚀 Bütün Modelləri Eyni Anda Müqayisə Et (CRNN-V2 / V1 / Tesseract 5)",
                        variant="primary",
                        size="lg",
                    )

            bench_table_html = gr.HTML(
                value=run_benchmark_comparison(None, ""),
                label="Müqayisə Cədvəli",
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
                    label="💡 Etalon Mətnli Hazır Müqayisə Nümunələri (Bir kliklə CER/WER cədvəlini qurur)",
                )

            bench_btn.click(
                fn=run_benchmark_comparison,
                inputs=[bench_img, bench_gt],
                outputs=[bench_table_html],
            )

        # ======================================================================
        # TAB 3: LIVE MOBILE CAMERA SCANNER
        # ======================================================================
        with gr.TabItem("📱 3. Canlı Mobil Kamera Skaneri (Real-Time)", id="tab_camera"):
            gr.HTML(LIVE_CAMERA_HTML)


def register_api_routes(fastapi_app):
    """Attach the fast /api/ocr JSON endpoint to the underlying FastAPI server."""
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
    parser = argparse.ArgumentParser(description="AzOCR-V2 Elmi-Tədqiqat və Mobil Kamera Veb Tətbiqi")
    parser.add_argument("--share", action="store_true", help="Telefonda istifadə üçün ictimai Gradio linki yaradın")
    parser.add_argument("--port", type=int, default=7860, help="Server portu (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dinləniləcək IP (default: 0.0.0.0)")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 65)
    print("🚀 AzOCR-V2 ELMİ-TƏDQİQAT VƏ CANLI KAMERA SERVERİ İŞƏ SALINIR")
    print("=" * 65)
    print(f"💻 Kompüterdə açmaq üçün:     http://localhost:{args.port}")
    print(f"📱 Eyni Wi-Fi ilə telefonda:   http://{local_ip}:{args.port}")
    if args.share:
        print("🌐 İctimai HTTPS linki hazırlanır (Canlı Kamera üçün)...")
    print("=" * 65)

    app_instance, local_url, share_url = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
        css=custom_css,
        head=CUSTOM_HEAD_SCRIPT,
        pwa=True,
        prevent_thread_lock=True,
    )
    register_api_routes(app_instance)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Server dayandırıldı.")


if __name__ == "__main__":
    main()
