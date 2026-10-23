"""Simulated phone photo test set generator for Azerbaijani OCR.

Generates realistic phone-camera photos of document pages using unseen fonts
and applies homography, illumination gradients, soft shadows, blur, noise,
downscaling, and JPEG compression. Then extracts line and word crops with exact
mapped bounding boxes.
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_data import font_ok

# Configuration parameters for rendering and photo distortions
CONFIG = {
    "page_width": 1240,
    "page_height": 1754,
    "dpi": 150,
    "font_size_min": 28,
    "font_size_max": 34,
    "left_margin": 120,
    "top_margin": 75,
    "lines_per_page_min": 22,
    "lines_per_page_max": 24,
    "paper_bg_rgb": (248, 246, 240),
    "paper_texture_std": 1.5,
    "text_color_rgb": (26, 26, 30),
    # Transform: perspective jitter and small rotation
    "perspective_jitter_pct": 0.035,
    "max_rotation_deg": 1.8,
    # Illumination and shadow
    "illumination_strength": 0.18,
    "shadow_depth": 0.22,
    "shadow_width_pct": 0.28,
    # Blur and noise
    "blur_sigma_min": 0.8,
    "blur_sigma_max": 1.5,
    "motion_blur_prob": 0.30,
    "sensor_noise_sigma_min": 3.0,
    "sensor_noise_sigma_max": 7.0,
    # Resolution and compression
    "downscale_factor_min": 0.65,
    "downscale_factor_max": 0.80,
    "jpeg_quality_min": 60,
    "jpeg_quality_max": 80,
    "warm_tint_r": 1.03,
    "warm_tint_b": 0.97,
    "crop_margin_px": 4,
}


def find_candidate_fonts():
    """Discover unseen fonts in Windows that pass font_ok for Azerbaijani."""
    existing_fonts = {p.name.lower() for p in Path("fonts").iterdir() if p.is_file()}
    win_fonts_dir = Path("C:/Windows/Fonts")

    # Ordered list of high-quality candidates across styles
    candidates = [
        ("serif", "pala.ttf"),
        ("serif", "cambria.ttc"),
        ("sans", "micross.ttf"),
        ("sans", "bahnschrift.ttf"),
        ("sans", "l_10646.ttf"),
        ("mono", "CascadiaMono.ttf"),
        ("any", "ebrima.ttf"),
    ]

    passed = []
    for style, font_name in candidates:
        font_path = win_fonts_dir / font_name
        if not font_path.exists():
            continue
        if font_name.lower() in existing_fonts:
            continue
        try:
            if font_ok(str(font_path)):
                passed.append((style, font_name, str(font_path)))
        except Exception:
            pass

    return passed


def load_corpus_lines(corpus_path, min_len=12, max_len=35):
    """Load matching text lines from corpus file."""
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if min_len <= len(t) <= max_len:
                lines.append(t)
    return lines


def render_page(lines, font_path, font_size, rng):
    """Render text lines onto an unwarped paper canvas and record layout boxes."""
    w = CONFIG["page_width"]
    h = CONFIG["page_height"]

    # Create paper texture background
    bg_base = np.full((h, w, 3), CONFIG["paper_bg_rgb"], dtype=np.float32)
    noise = rng.normal(0, CONFIG["paper_texture_std"], (h, w, 3))
    paper_img = np.clip(bg_base + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(paper_img)
    draw = ImageDraw.Draw(pil_img)

    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()

    n_lines = len(lines)
    avail_h = h - CONFIG["top_margin"] - 80
    line_step = int(avail_h / max(n_lines, 1))

    line_records = []
    word_records = []

    for i, line_text in enumerate(lines):
        y_line = CONFIG["top_margin"] + i * line_step
        x_line = CONFIG["left_margin"] + int(rng.integers(-4, 5))

        # Draw line
        draw.text((x_line, y_line), line_text, font=font, fill=CONFIG["text_color_rgb"], anchor="lt")

        # Line bounding box (tightly contains glyphs including descenders and diacritics)
        l_bbox = font.getbbox(line_text, anchor="lt")
        l_x0 = x_line + l_bbox[0]
        l_y0 = y_line + l_bbox[1]
        l_x1 = x_line + l_bbox[2]
        l_y1 = y_line + l_bbox[3]
        line_records.append((line_text, [l_x0, l_y0, l_x1, l_y1]))

        # Word bounding boxes
        for m in re.finditer(r"\S+", line_text):
            word_text = m.group()
            if len(word_text) < 2:
                continue
            # Word bounding boxes (x from prefix length, y from font metrics, tightly containing glyphs)
            prefix = line_text[:m.start()]
            prefix_w = font.getlength(prefix)
            w_bbox = font.getbbox(word_text, anchor="lt")
            w_x0 = x_line + prefix_w + w_bbox[0]
            w_y0 = y_line + min(0, w_bbox[1])
            w_x1 = x_line + prefix_w + w_bbox[2]
            w_y1 = y_line + max(ascent + descent, w_bbox[3])
            word_records.append((word_text, [w_x0, w_y0, w_x1, w_y1]))

    return np.array(pil_img), line_records, word_records


def apply_photo_simulation(page_rgb, rng):
    """Apply unified 3x3 homography and realistic phone camera distortions."""
    h, w = page_rgb.shape[:2]

    # 1. 3x3 Homography (perspective + small rotation)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-CONFIG["perspective_jitter_pct"], CONFIG["perspective_jitter_pct"], (4, 2)).astype(np.float32) * w
    dst_pts = (src_pts + jitter).astype(np.float32)

    # Small rotation around center
    angle = rng.uniform(-CONFIG["max_rotation_deg"], CONFIG["max_rotation_deg"])
    rad = math.radians(angle)
    cx, cy = w / 2.0, h / 2.0
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx = dst_pts[:, 0] - cx
    dy = dst_pts[:, 1] - cy
    dst_x = cx + dx * cos_a - dy * sin_a
    dst_y = cy + dx * sin_a + dy * cos_a
    dst_pts = np.column_stack((dst_x, dst_y)).astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(page_rgb, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Convert to BGR float for OpenCV operations
    img = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR).astype(np.float32)

    # 2. Uneven illumination gradient
    gx = np.linspace(-1, 1, w, dtype=np.float32)
    gy = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(gx, gy)
    s = CONFIG["illumination_strength"]
    a, b, c = rng.uniform(-s, s), rng.uniform(-s, s), rng.uniform(-s * 0.5, s * 0.5)
    illum = 1.0 + a * X + b * Y + c * X * Y

    # 3. Soft shadow band
    phi = rng.uniform(0, math.pi)
    nx, ny = math.cos(phi), math.sin(phi)
    d0 = rng.uniform(-0.4, 0.4)
    dist = X * nx + Y * ny - d0
    bw = CONFIG["shadow_width_pct"]
    shadow = 1.0 - CONFIG["shadow_depth"] * np.exp(-(dist / bw) ** 2)

    img = img * illum[:, :, None] * shadow[:, :, None]

    # 4. Gaussian blur
    sigma = rng.uniform(CONFIG["blur_sigma_min"], CONFIG["blur_sigma_max"])
    img = cv2.GaussianBlur(img, (0, 0), sigma)

    # 5. Motion blur (optional)
    if rng.random() < CONFIG["motion_blur_prob"]:
        ksize = 3
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        m_angle = rng.uniform(0, math.pi)
        dx_k = int(round(math.cos(m_angle)))
        dy_k = int(round(math.sin(m_angle)))
        kernel[1, 1] = 1.0
        kernel[1 + dy_k, 1 + dx_k] = 1.0
        kernel /= kernel.sum()
        img = cv2.filter2D(img, -1, kernel)

    # 6. Sensor noise
    n_sigma = rng.uniform(CONFIG["sensor_noise_sigma_min"], CONFIG["sensor_noise_sigma_max"])
    img += rng.normal(0, n_sigma, img.shape)

    # 7. Downscale then upscale
    scale = rng.uniform(CONFIG["downscale_factor_min"], CONFIG["downscale_factor_max"])
    sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 8. Warm color tint (slight yellow/warm phone look)
    img[:, :, 2] *= CONFIG["warm_tint_r"]  # Red
    img[:, :, 0] *= CONFIG["warm_tint_b"]  # Blue

    # 9. JPEG compression
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.uniform(CONFIG["jpeg_quality_min"], CONFIG["jpeg_quality_max"]))
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    final_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return final_bgr, M


def crop_boxes(image_bgr, boxes, M, margin):
    """Map unwarped bounding boxes through M and crop from distorted page."""
    h, w = image_bgr.shape[:2]
    crops = []
    for text, box in boxes:
        x0, y0, x1, y1 = box
        pts = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
        mapped = cv2.perspectiveTransform(pts, M)
        min_x = np.min(mapped[:, 0, 0])
        max_x = np.max(mapped[:, 0, 0])
        min_y = np.min(mapped[:, 0, 1])
        max_y = np.max(mapped[:, 0, 1])

        cx0 = max(0, int(math.floor(min_x - margin)))
        cy0 = max(0, int(math.floor(min_y - margin)))
        cx1 = min(w, int(math.ceil(max_x + margin)))
        cy1 = min(h, int(math.ceil(max_y + margin)))

        crop = image_bgr[cy0:cy1, cx0:cx1]
        crops.append((text, crop))
    return crops


def run_automatic_checks(crops):
    """Check for empty crops and crops where ink touches border."""
    n = len(crops)
    if n == 0:
        return 0.0, 0.0

    empty_count = 0
    border_touch_count = 0

    for _, img in crops:
        if img.size == 0:
            empty_count += 1
            border_touch_count += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink_mask = gray < 140
        total_px = gray.size
        ink_ratio = np.sum(ink_mask) / max(total_px, 1)

        if ink_ratio < 0.01:
            empty_count += 1

        top_edge = np.any(ink_mask[0, :])
        bot_edge = np.any(ink_mask[-1, :])
        l_edge = np.any(ink_mask[:, 0])
        r_edge = np.any(ink_mask[:, -1])
        if top_edge or bot_edge or l_edge or r_edge:
            border_touch_count += 1

    pct_empty = (empty_count / n) * 100.0
    pct_border = (border_touch_count / n) * 100.0
    return pct_empty, pct_border


def build_contact_sheet(line_crops, word_crops, out_path):
    """Render a clean contact sheet of 12 lines and 24 words with labels."""
    special_chars = set("əıöğüşçƏİÖĞÜŞÇ")

    def sort_key(item):
        text = item[0]
        has_spec = sum(1 for c in text if c in special_chars)
        return -has_spec

    # Pick 12 lines rich in special characters
    sel_lines = sorted(line_crops, key=sort_key)[:12]
    # Pick 24 words rich in special characters
    sel_words = sorted(word_crops, key=sort_key)[:24]

    # Base canvas
    sheet_w = 1400
    line_card_h = 75
    word_card_w = 320
    word_card_h = 75

    lines_total_h = 60 + len(sel_lines) * line_card_h
    words_total_h = 60 + 6 * word_card_h + 40
    total_h = lines_total_h + words_total_h + 60

    canvas = Image.new("RGB", (sheet_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    label_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)

    # 1. Section Header: Line crops
    y_cur = 20
    draw.text((30, y_cur), "Sample Line Crops (12 lines with labels)", fill=(0, 50, 150), font=header_font)
    y_cur += 40

    for text, crop_bgr in sel_lines:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        # Scale height if needed to fit card
        if pil_crop.height > 40:
            scale = 40.0 / pil_crop.height
            new_w = max(10, int(pil_crop.width * scale))
            pil_crop = pil_crop.resize((new_w, 40), Image.Resampling.LANCZOS)

        draw.text((30, y_cur), f"Label: {text}", fill=(60, 60, 60), font=label_font)
        canvas.paste(pil_crop, (30, y_cur + 20))
        y_cur += line_card_h

    # 2. Section Header: Word crops
    y_cur += 20
    draw.text((30, y_cur), "Sample Word Crops (24 words with labels, grid 4x6)", fill=(0, 50, 150), font=header_font)
    y_cur += 40

    cols = 4
    for idx, (text, crop_bgr) in enumerate(sel_words):
        r = idx // cols
        c = idx % cols
        x_cell = 30 + c * (word_card_w + 20)
        y_cell = y_cur + r * word_card_h

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)
        if pil_crop.height > 36:
            scale = 36.0 / pil_crop.height
            new_w = max(5, int(pil_crop.width * scale))
            pil_crop = pil_crop.resize((new_w, 36), Image.Resampling.LANCZOS)

        draw.text((x_cell, y_cell), text, fill=(60, 60, 60), font=label_font)
        canvas.paste(pil_crop, (x_cell, y_cell + 20))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Simulate phone photos and generate test crops.")
    parser.add_argument("--pages", type=int, default=2, help="Number of pages to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--out", type=str, default="data/sim_preview", help="Output directory.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1. Fonts discovery
    print("[1/5] Discovering unseen Windows fonts for Azerbaijani...")
    passed_fonts = find_candidate_fonts()
    print(f"Passed candidate fonts: {[f[1] for f in passed_fonts]}")

    if not passed_fonts:
        sys.exit("Error: No candidate fonts passed font_ok.")

    # Select distinct fonts across styles for the requested pages
    selected_fonts = []
    for i in range(args.pages):
        f = passed_fonts[i % len(passed_fonts)]
        selected_fonts.append(f)
    print(f"Fonts selected for pages: {[f[1] for f in selected_fonts]}")

    # 2. Load text corpus lines
    print("[2/5] Loading lines from data/corpus_line/test.txt...")
    corpus_lines = load_corpus_lines("data/corpus_line/test.txt")
    print(f"Available corpus lines: {len(corpus_lines)}")

    out_base = Path(args.out)
    line_out_dir = out_base / "line" / "test"
    word_out_dir = out_base / "word" / "test"
    (line_out_dir / "images").mkdir(parents=True, exist_ok=True)
    (word_out_dir / "images").mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/sim_preview")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_line_crops = []
    all_word_crops = []

    # 3. Generate pages
    print(f"[3/5] Generating {args.pages} simulated phone photo pages...")
    for page_idx in range(args.pages):
        style, font_name, font_path = selected_fonts[page_idx]
        n_lines = int(rng.integers(CONFIG["lines_per_page_min"], CONFIG["lines_per_page_max"] + 1))
        page_lines = [rng.choice(corpus_lines) for _ in range(n_lines)]
        font_size = int(rng.integers(CONFIG["font_size_min"], CONFIG["font_size_max"] + 1))

        # Render unwarped page
        clean_page_rgb, line_boxes, word_boxes = render_page(page_lines, font_path, font_size, rng)

        # Apply realistic phone photo distortion
        distorted_bgr, M = apply_photo_simulation(clean_page_rgb, rng)

        # Save full distorted page
        page_save_path = results_dir / f"page_{page_idx + 1}.jpg"
        cv2.imwrite(str(page_save_path), distorted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"  Saved full page {page_idx + 1}: {page_save_path} (Font: {font_name}, Lines: {n_lines})")

        # Crop lines and words using exact mapped boxes
        page_line_crops = crop_boxes(distorted_bgr, line_boxes, M, CONFIG["crop_margin_px"])
        page_word_crops = crop_boxes(distorted_bgr, word_boxes, M, CONFIG["crop_margin_px"])

        all_line_crops.extend(page_line_crops)
        all_word_crops.extend(page_word_crops)

    # 4. Save crops and labels.tsv in standard dataset format
    print(f"[4/5] Saving {len(all_line_crops)} line crops and {len(all_word_crops)} word crops...")
    with open(line_out_dir / "labels.tsv", "w", encoding="utf-8") as f_line:
        for idx, (text, crop_img) in enumerate(all_line_crops):
            fn = f"images/{idx:07d}.png"
            cv2.imwrite(str(line_out_dir / fn), crop_img)
            f_line.write(f"{fn}\t{text}\n")

    with open(word_out_dir / "labels.tsv", "w", encoding="utf-8") as f_word:
        for idx, (text, crop_img) in enumerate(all_word_crops):
            fn = f"images/{idx:07d}.png"
            cv2.imwrite(str(word_out_dir / fn), crop_img)
            f_word.write(f"{fn}\t{text}\n")

    # 5. Verification: Contact sheet & Automatic checks
    print("[5/5] Running verification checks...")
    contact_sheet_path = results_dir / "crops_check.png"
    build_contact_sheet(all_line_crops, all_word_crops, contact_sheet_path)
    print(f"  Contact sheet saved: {contact_sheet_path}")

    empty_lines, border_lines = run_automatic_checks(all_line_crops)
    empty_words, border_words = run_automatic_checks(all_word_crops)

    print("\n=== AUTOMATIC QUALITY CHECKS ===")
    print(f"Line crops ({len(all_line_crops)}):")
    print(f"  Empty crops (<1% ink): {empty_lines:.2f}%")
    print(f"  Ink touches border:    {border_lines:.2f}%")
    print(f"Word crops ({len(all_word_crops)}):")
    print(f"  Empty crops (<1% ink): {empty_words:.2f}%")
    print(f"  Ink touches border:    {border_words:.2f}%")


if __name__ == "__main__":
    main()
