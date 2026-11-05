#!/usr/bin/env python3
"""Azərbaycan Dili üçün OCR Veb və Mobil İstifadəçi Arayüzü (Gradio).

Bu tətbiq həm masaüstü brauzerlərdən, həm də smartfonlardan (kamera ilə sənəd çəkərək)
birbaşa istifadə oluna bilir.

İstifadə qaydası:
    python app.py                # Yalnız lokal şəbəkə və kompüter üçün (http://localhost:7860)
    python app.py --share        # Telefonda istifadə üçün 72 saatlıq pulsuz ictimai link yaradır
    python app.py --port 8080    # Xüsusi port təyin etmək üçün
"""

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch

from common import infer_single_crop, load_model_checkpoint
from postprocess import AzerbaijaniPostProcessor

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

# Model definitions
MODEL_CONFIGS = {
    "CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]": "runs/line_v2/best.pt",
    "CRNN Söz Modeli (V2 - 32x128)": "runs/word_v2_20261027_180300/best.pt",
    "CRNN Sətir Modeli (V1 - Baza)": "runs/line/best.pt",
    "CRNN Söz Modeli (V1 - Baza)": "runs/word/best.pt",
}

# Cache for loaded models
LOADED_MODELS = {}

# Initialize post-processor
print("[*] Lüğət bazası yüklənir...")
postprocessor = AzerbaijaniPostProcessor()
if postprocessor.loaded:
    print("[*] Azərbaycan dili tezlik lüğəti uğurla yükləndi (az_lexicon.txt).")
else:
    print("[!] Xəbərdarlıq: az_lexicon.txt tapılmadı.")


def get_model(choice: str):
    """Lazily load and cache the requested model."""
    if choice not in LOADED_MODELS:
        ckpt_path = MODEL_CONFIGS[choice]
        if not Path(ckpt_path).exists():
            # Fallback
            ckpt_path = "runs/line_v2/best.pt" if "Sətir" in choice else "runs/word_v2_20261027_180300/best.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, vocab, device = load_model_checkpoint(ckpt_path, device=device)
        LOADED_MODELS[choice] = (model, cfg, vocab, device)
    return LOADED_MODELS[choice]


# Preload the default model for fast first inference
print("[*] Standart V2 Sətir modeli yüklənir...")
get_model("CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]")


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


def ocr_recognize(img: Optional[np.ndarray], model_choice: str, use_postprocess: bool) -> Tuple[str, str, str]:
    """Main OCR pipeline function for Gradio."""
    if img is None:
        return "Zəhmət olmasa bir şəkil yükləyin və ya kamera ilə çəkin.", "", ""

    # Convert RGB/RGBA from Gradio to BGR for OpenCV processing
    if img.ndim == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = img

    # Load requested model
    model, cfg, vocab, device = get_model(model_choice)

    # Perform inference
    start_time = time.perf_counter()
    raw_pred = infer_single_crop(model, cfg, vocab, img_bgr, device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Post-process with Azerbaijani dictionary if requested
    final_pred = raw_pred
    postprocess_applied = False
    if use_postprocess and postprocessor and postprocessor.loaded:
        tokens = raw_pred.split()
        corrected_tokens = [postprocessor.correct_word(t, max_edit_distance=1) for t in tokens]
        final_pred = postprocessor.clean_punctuation(" ".join(corrected_tokens))
        postprocess_applied = True

    # Info summary
    info_lines = [
        f"⚡ **İnferens Sürəti:** {elapsed_ms:.1f} ms",
        f"🖥️ **Hesablama Cihazı:** {device.type.upper()}",
        f"📐 **Giriş Ölçüsü:** {cfg.get('height', 32)}×{cfg.get('width', 512)} px",
        f"📚 **Lüğət Düzəlişi:** {'Tətbiq edildi' if postprocess_applied else 'Söndürülüb'}"
    ]
    info_md = " | ".join(info_lines)

    return final_pred, raw_pred, info_md


# Build Custom Gradio Interface
custom_css = """
.main-title { text-align: center; margin-bottom: 5px; }
.sub-title { text-align: center; color: #555; margin-bottom: 20px; font-size: 15px; }
.output-box textarea { font-size: 1.25rem !important; font-weight: 500 !important; }
"""

with gr.Blocks(title="Azərbaycan Dili OCR (BM 606)") as demo:
    gr.HTML("""
        <div class="main-title">
            <h2>🇦🇿 Azərbaycan Dili üçün Optik Mətn Tanıma (CRNN + CTC)</h2>
        </div>
        <div class="sub-title">
            BM 606 Magistr Tədqiqatı: Telefon kamerası şəkilləri, çap sənədləri və SymSpell lüğət korreksiyası
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Sənəd / Sətir Şəkli (Kamera və ya Qalereya)",
                sources=["upload", "webcam", "clipboard"],
                type="numpy"
            )
            
            with gr.Accordion("⚙️ Model və Emal Parametrləri", open=True):
                model_dropdown = gr.Dropdown(
                    label="OCR Modeli",
                    choices=list(MODEL_CONFIGS.keys()),
                    value="CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]"
                )
                postprocess_chk = gr.Checkbox(
                    label="Azərbaycan dili lüğət korreksiyasını tətbiq et (SymSpell)",
                    value=True
                )

            submit_btn = gr.Button("🔍 Mətni Tanı (OCR)", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="✅ Tanınmış Mətn (Düzəldilmiş)",
                placeholder="Oxunmuş mətn burada görünəcək...",
                lines=4,
                elem_classes=["output-box"]
            )
            
            with gr.Accordion("📝 Ətraflı Məlumat və Xam Çıxış", open=False):
                raw_text = gr.Textbox(
                    label="Xam Model Çıxışı (Lüğət düzəlişsiz)",
                    lines=3,
                    interactive=False
                )
            
            info_box = gr.Markdown("Emal üçün sənəd şəkli təqdim edin.")

    # Examples gallery
    example_paths = [
        ["results/real_fix8/crops_calibri/line_00.png", "CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]", True],
        ["results/real_fix8/crops_times/line_00.png", "CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]", True],
        ["results/real_fix9/crops/line_00.png", "CRNN Sətir Modeli (V2 - 32x512) [Tövsiyə olunan]", True]
    ]
    
    # Filter examples that actually exist
    valid_examples = [ex for ex in example_paths if Path(ex[0]).exists()]
    if valid_examples:
        gr.Examples(
            examples=valid_examples,
            inputs=[image_input, model_dropdown, postprocess_chk],
            outputs=[output_text, raw_text, info_box],
            fn=ocr_recognize,
            cache_examples=False,
            label="💡 Hazır Test Nümunələri (Bir kliklə sınaqdan keçirin)"
        )

    submit_btn.click(
        fn=ocr_recognize,
        inputs=[image_input, model_dropdown, postprocess_chk],
        outputs=[output_text, raw_text, info_box]
    )

    gr.HTML("""
        <div style="text-align: center; margin-top: 25px; font-size: 13px; color: #777;">
            📱 <b>Telefonda istifadə qaydası:</b> Kompüterlə eyni Wi-Fi şəbəkəsinə qoşulun və telefonun brauzerində yuxarıda göstərilən yerli IP ünvanını açın.<br>
            Həmçinin <code>python app.py --share</code> komandası ilə hər yerdən açıla bilən ictimai link əldə edə bilərsiniz.
        </div>
    """)


def main():
    parser = argparse.ArgumentParser(description="Azərbaycan Dili OCR Gradio Tətbiqi")
    parser.add_argument("--share", action="store_true", help="Telefonda istifadə üçün ictimai Gradio linki yaradın")
    parser.add_argument("--port", type=int, default=7860, help="Server portu (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Dinləniləcək IP (default: 0.0.0.0)")

    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 65)
    print("🚀 AZƏRBAYCAN DİLİ OCR VEB VƏ MOBİL ARAYÜZÜ İŞƏ SALINIR")
    print("=" * 65)
    print(f"💻 Kompüterdə açmaq üçün:     http://localhost:{args.port}")
    print(f"📱 Eyni Wi-Fi ilə telefonda:   http://{local_ip}:{args.port}")
    if args.share:
        print("🌐 İctimai internet linki hazırlanır (Gradio Share)...")
    else:
        print("💡 Qeyd: Əgər hər yerdən telefonda açmaq istəsəniz: python app.py --share")
    print("=" * 65)

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=custom_css,
        pwa=True
    )


if __name__ == "__main__":
    main()
