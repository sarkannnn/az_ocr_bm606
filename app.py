#!/usr/bin/env python3
"""Azərbaycan Dili üçün Canlı Kamera və Veb OCR Arayüzü (Gradio + HTML5 Live Camera).

Xüsusiyyətlər:
- Telefonda açıldıqda avtomatik Arxa Kameranı (canlı video) işə salır.
- Kamera üzərində hədəf oxuma çərçivəsi (Tək Sətir / Paraqraf / Tam Ekran).
- Həm "📸 Çək və Oxu", həm də "🔄 Canlı Avto-Oxu" (fasiləsiz real-vaxt OCR) rejimi.
- Uzun sətirlər üçün avtomatik sözarası bölmə (smart chunking) və çoxsətirli seqmentasiya.
- SymSpell (116,000+ söz) ilə Azərbaycan dili lüğət korreksiyası.
"""

import argparse
import base64
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

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
    infer_line_smart,
    load_model_checkpoint,
    normalize,
    segment_lines_opencv,
)
from postprocess import AzerbaijaniPostProcessor

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Model definitions
TESSERACT_MODEL_KEY = "Tesseract OCR (aze - Baseline)"
DEFAULT_MODEL_KEY = "CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]"

MODEL_CONFIGS = {
    DEFAULT_MODEL_KEY: "runs/line_v2/best.pt",
    TESSERACT_MODEL_KEY: "tesseract",
    "CRNN Söz Modeli (V2 - 32x128)": "runs/word_v2_20261027_180300/best.pt",
    "CRNN Sətir Modeli (V1 - Baza)": "runs/line/best.pt",
    "CRNN Söz Modeli (V1 - Baza)": "runs/word/best.pt",
}
LOADED_MODELS = {}

print("[*] Lüğət bazası yüklənir...")
postprocessor = AzerbaijaniPostProcessor()
if postprocessor.loaded:
    print("[*] Azərbaycan dili tezlik lüğəti uğurla yükləndi (az_lexicon.txt).")
else:
    print("[!] Xəbərdarlıq: az_lexicon.txt tapılmadı.")


def get_model(choice: str):
    """Lazily load and cache the requested model."""
    if choice not in MODEL_CONFIGS:
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


def get_local_ip():
    """Find the local Wi-Fi IP to show users how to connect from their phones."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def extract_tight_text_lines(img_bgr: np.ndarray) -> list:
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


def ocr_recognize(img: Optional[np.ndarray], model_choice: str, use_postprocess: bool) -> Tuple[str, str, str]:
    """Gradio callback for uploaded/example images."""
    if img is None:
        return "Zəhmət olmasa bir şəkil yükləyin və ya yuxarıdakı canlı kameradan istifadə edin.", "", ""

    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = img

    final_pred, raw_pred, elapsed_ms, n_lines = run_ocr_on_bgr(
        img_bgr, model_choice=model_choice, use_postprocess=use_postprocess, multi_line=True
    )

    device_name = "CPU"
    if model_choice != TESSERACT_MODEL_KEY:
        _, _, _, device = get_model(model_choice)
        device_name = device.type.upper()

    info_lines = [
        f"⚡ **Sürət:** {elapsed_ms:.1f} ms",
        f"📏 **Aşkarlanan sətir:** {n_lines}",
        f"🖥️ **Mühərrik:** {model_choice.split()[0]} ({device_name})",
        f"📚 **Lüğət Düzəlişi:** {'Aktiv' if use_postprocess else 'Söndürülüb'}",
    ]
    return final_pred, raw_pred, " | ".join(info_lines)


def ocr_from_base64_bridge(b64_payload: str, model_choice: str, use_postprocess: bool) -> Tuple[str, str, str]:
    """Fallback bridge if triggered via Gradio state."""
    if not b64_payload or "," not in b64_payload:
        return "", "", ""
    try:
        header, encoded = b64_payload.split(",", 1)
        multi = "mode=multi" in header
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return "Şəkil oxunmadı.", "", ""
        final_pred, raw_pred, elapsed_ms, n_lines = run_ocr_on_bgr(
            img_bgr, model_choice=model_choice, use_postprocess=use_postprocess, multi_line=multi
        )
        info = f"⚡ **Canlı Kamera Sürəti:** {elapsed_ms:.0f} ms | 📏 **Sətir:** {n_lines}"
        return final_pred, raw_pred, info
    except Exception as e:
        return f"Xəta: {e}", "", ""


LIVE_CAMERA_HTML = """
<div id="az-cam-wrapper" style="background: #0f172a; color: #f8fafc; border-radius: 16px; padding: 14px; margin-bottom: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.25);">
  <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
    <div style="display: flex; align-items: center; gap: 8px;">
      <span id="az-cam-dot" style="width: 12px; height: 12px; border-radius: 50%; background: #22c55e; display: inline-block; box-shadow: 0 0 8px #22c55e;"></span>
      <strong style="font-size: 16px;">📹 Canlı Mobil Kamera Skaneri</strong>
    </div>
    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
      <button type="button" onclick="window.azSetBoxMode('line')" id="btn-mode-line" style="background: #2563eb; color: #fff; border: none; border-radius: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">📏 Tək Sətir</button>
      <button type="button" onclick="window.azSetBoxMode('para')" id="btn-mode-para" style="background: #334155; color: #cbd5e1; border: none; border-radius: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">📄 Paraqraf</button>
      <button type="button" onclick="window.azSetBoxMode('full')" id="btn-mode-full" style="background: #334155; color: #cbd5e1; border: none; border-radius: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;">🖼️ Tam Ekran</button>
      <select id="az-cam-engine-select" onchange="window.azState.engine = this.value" style="background: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; border-radius: 8px; padding: 4px 8px; font-size: 12px; font-weight: 700; cursor: pointer;">
        <option value="crnn_v2" selected>⚡ Bizim CRNN Modelimiz (V2)</option>
        <option value="tesseract">🔍 Tesseract OCR (aze)</option>
      </select>
    </div>
  </div>

  <!-- Live Video Container -->
  <div id="az-video-stage" style="position: relative; width: 100%; max-height: 52vh; background: #020617; border-radius: 12px; overflow: hidden; display: flex; justify-content: center; align-items: center; min-height: 220px;">
    <video id="az-cam-video" autoplay playsinline muted style="width: 100%; height: 100%; max-height: 52vh; object-fit: cover; display: block;"></video>

    <!-- Darkened mask outside target box & glowing green guide box -->
    <div id="az-scan-box" style="position: absolute; left: 4%; right: 4%; top: 40%; height: 18%; border: 2.5px solid #22c55e; border-radius: 10px; box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.52), inset 0 0 15px rgba(34, 197, 94, 0.25); pointer-events: none; transition: all 0.2s ease;">
      <!-- Laser horizontal guide line -->
      <div style="position: absolute; left: 0; right: 0; top: 50%; height: 1.5px; background: rgba(239, 68, 68, 0.85); box-shadow: 0 0 8px #ef4444;"></div>
      <div style="position: absolute; top: -24px; left: 6px; background: #22c55e; color: #022c22; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px;">
        🎯 MƏTNİ QIRMIZI XƏTTƏ GƏTİRİN
      </div>
    </div>

    <div id="az-cam-msg" style="position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(2,6,23,0.85); color: #fff; padding: 16px; text-align: center; z-index: 5;">
      <div style="font-size: 15px; margin-bottom: 10px;" id="az-cam-msg-text">📷 Kamera icazəsi gözlənilir...</div>
      <button type="button" onclick="window.azStartCamera()" style="background: #22c55e; color: #022c22; font-weight: 700; border: none; padding: 10px 20px; border-radius: 10px; font-size: 15px; cursor: pointer;">
        📹 Kameranı İndi Aç
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
    var status = document.getElementById("az-live-status");
    if (status) status.innerText = "Bu cihazda fənər brauzerdən idarə olunmur.";
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

// Auto-launch camera as soon as the video element mounts in DOM
(function() {
  var booted = false;
  var checkTimer = setInterval(function() {
    var v = document.getElementById("az-cam-video");
    if (v && !booted) {
      booted = true;
      clearInterval(checkTimer);
      window.azStartCamera();
    }
  }, 300);
})();
</script>
"""

custom_css = """
.main-title { text-align: center; margin-bottom: 4px; }
.sub-title { text-align: center; color: #64748b; margin-bottom: 12px; font-size: 14px; }
.output-box textarea { font-size: 1.2rem !important; font-weight: 500 !important; }
"""

with gr.Blocks(title="Azərbaycan Dili OCR (BM 606)") as demo:
    gr.HTML("""
        <div class="main-title">
            <h2 style="margin: 0;">🇦🇿 Azərbaycan Dili üçün Optik Mətn Tanıma (CRNN + CTC)</h2>
        </div>
        <div class="sub-title">
            Canlı Mobil Kamera Skaneri və Sənəd Oxuma Sistemi (SymSpell Lüğət Dəstəyi ilə)
        </div>
    """)

    # 1. Top Live Mobile Camera Scanner (Auto-opens on phone!)
    gr.HTML(LIVE_CAMERA_HTML)

    # 2. File Upload & Examples Section
    with gr.Accordion("📁 Qalereyadan Şəkil Yüklə və ya Hazır Nümunələri Sına", open=False):
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="🖼️ Qalereyadan Şəkil Seçin",
                    sources=["upload", "clipboard"],
                    type="numpy"
                )
                model_dropdown = gr.Dropdown(
                    label="OCR Modeli",
                    choices=list(MODEL_CONFIGS.keys()),
                    value=DEFAULT_MODEL_KEY
                )
                postprocess_chk = gr.Checkbox(
                    label="Azərbaycan dili lüğət korreksiyasını tətbiq et (SymSpell)",
                    value=True
                )
                submit_btn = gr.Button("🔍 Yüklənmiş Şəkli Oxu", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="✅ Tanınmış Mətn (Düzəldilmiş)",
                    placeholder="Oxunmuş mətn burada görünəcək...",
                    lines=4,
                    elem_classes=["output-box"]
                )
                raw_text = gr.Textbox(
                    label="Xam Model Çıxışı (Lüğət düzəlişsiz)",
                    lines=2,
                    interactive=False
                )
                info_box = gr.Markdown("Emal üçün şəkil seçin.")

        example_paths = [
            ["results/real_fix8/crops_calibri/line_00.png", DEFAULT_MODEL_KEY, True],
            ["results/real_fix8/crops_times/line_00.png", DEFAULT_MODEL_KEY, True],
            ["results/real_fix9/crops/line_00.png", DEFAULT_MODEL_KEY, True],
        ]
        valid_examples = [ex for ex in example_paths if Path(ex[0]).exists()]
        if valid_examples:
            gr.Examples(
                examples=valid_examples,
                inputs=[image_input, model_dropdown, postprocess_chk],
                outputs=[output_text, raw_text, info_box],
                fn=ocr_recognize,
                cache_examples=False,
                label="💡 Hazır Test Nümunələri"
            )

        submit_btn.click(
            fn=ocr_recognize,
            inputs=[image_input, model_dropdown, postprocess_chk],
            outputs=[output_text, raw_text, info_box]
        )


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

            # Debug save to inspect exact user mobile input
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
                "engine": "Tesseract (aze)" if engine_req == "tesseract" else "CRNN V2",
                "ms": round(elapsed_ms, 1),
                "lines": n_lines,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Put /api/ocr at the beginning of router routes so catch-all routes don't shadow it
    fastapi_app.router.routes.insert(0, fastapi_app.router.routes.pop())


def main():
    parser = argparse.ArgumentParser(description="Azərbaycan Dili OCR Canlı Kamera və Veb Tətbiqi")
    parser.add_argument("--share", action="store_true", help="Telefonda istifadə üçün ictimai Gradio linki yaradın")
    parser.add_argument("--port", type=int, default=7860, help="Server portu (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dinləniləcək IP (default: 0.0.0.0)")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 65)
    print("🚀 AZƏRBAYCAN DİLİ CANLI KAMERA OCR SERVERİ İŞƏ SALINIR")
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
