"""Generates a distortion ladder for Azerbaijani OCR diagnosis.

Builds 7 controlled distortion levels using the same text lines and layout across 4 pages:
  L0:  Clean render (pure white, black text, identity transform)
  L0c: Clean render with direct axis-aligned crop (baseline check vs L0)
  L1:  L0 + off-white paper + texture + illumination gradient + soft shadow
  L2:  L1 + Gaussian blur + sensor noise
  L3:  L2 + downscale/upscale + JPEG compression + warm tint
  L4:  L3 + perspective + rotation (full simulated phone photo)
  L4n: L4 with per-crop min-max contrast stretching
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_data import font_ok

CONFIG = {
    "page_width": 1240,
    "page_height": 1754,
    "font_size_min": 28,
    "font_size_max": 34,
    "left_margin": 120,
    "top_margin": 75,
    "bottom_margin": 80,
    "lines_per_page_min": 22,
    "lines_per_page_max": 24,
    "paper_bg_rgb": (248, 246, 240),
    "paper_texture_std": 1.5,
    "text_color_rgb": (26, 26, 30),
    "perspective_jitter_pct": 0.035,
    "max_rotation_deg": 1.8,
    "illumination_strength": 0.18,
    "shadow_depth": 0.22,
    "shadow_width_pct": 0.28,
    "blur_sigma_min": 0.8,
    "blur_sigma_max": 1.5,
    "motion_blur_prob": 0.30,
    "sensor_noise_sigma_min": 3.0,
    "sensor_noise_sigma_max": 7.0,
    "downscale_factor_min": 0.65,
    "downscale_factor_max": 0.80,
    "jpeg_quality_min": 60,
    "jpeg_quality_max": 80,
    "warm_tint_r": 1.03,
    "warm_tint_b": 0.97,
    "crop_margin_px": 4,
}


def load_corpus_lines(corpus_path, min_len=12, max_len=35):
    """Load text lines of matching length."""
    lines = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if min_len <= len(t) <= max_len:
                lines.append(t)
    return lines


def setup_pages_plan(fonts_info, corpus_lines, seed=7):
    """Pre-compute identical text, font sizes, and layout boxes for 4 pages."""
    rng = np.random.default_rng(seed)
    pages = []
    w = CONFIG["page_width"]
    h = CONFIG["page_height"]
    avail_h = h - CONFIG["top_margin"] - CONFIG["bottom_margin"]

    for page_idx, (style, font_name, font_path) in enumerate(fonts_info):
        font_size = int(rng.integers(CONFIG["font_size_min"], CONFIG["font_size_max"] + 1))
        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()

        n_lines = int(rng.integers(CONFIG["lines_per_page_min"], CONFIG["lines_per_page_max"] + 1))
        page_lines = [rng.choice(corpus_lines) for _ in range(n_lines)]
        line_step = int(avail_h / max(n_lines, 1))

        line_boxes = []
        word_boxes = []

        for i, line_text in enumerate(page_lines):
            y_line = CONFIG["top_margin"] + i * line_step
            x_line = CONFIG["left_margin"] + int(rng.integers(-4, 5))

            l_bbox = font.getbbox(line_text, anchor="lt")
            l_x0 = x_line + l_bbox[0]
            l_y0 = y_line + l_bbox[1]
            l_x1 = x_line + l_bbox[2]
            l_y1 = y_line + l_bbox[3]
            line_boxes.append((line_text, [l_x0, l_y0, l_x1, l_y1], font_name))

            for m in re.finditer(r"\S+", line_text):
                w_text = m.group()
                if len(w_text) < 2:
                    continue
                prefix = line_text[:m.start()]
                prefix_w = font.getlength(prefix)
                w_bbox = font.getbbox(w_text, anchor="lt")
                w_x0 = x_line + prefix_w + w_bbox[0]
                w_y0 = y_line + min(0, w_bbox[1])
                w_x1 = x_line + prefix_w + w_bbox[2]
                w_y1 = y_line + max(ascent + descent, w_bbox[3])
                word_boxes.append((w_text, [w_x0, w_y0, w_x1, w_y1], font_name))

        # Pre-generate distortion parameters for reproducibility
        # 1. Homography
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = rng.uniform(-CONFIG["perspective_jitter_pct"], CONFIG["perspective_jitter_pct"], (4, 2)).astype(np.float32) * w
        dst_pts = (src_pts + jitter).astype(np.float32)
        angle = rng.uniform(-CONFIG["max_rotation_deg"], CONFIG["max_rotation_deg"])
        rad = math.radians(angle)
        cx, cy = w / 2.0, h / 2.0
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        dx = dst_pts[:, 0] - cx
        dy = dst_pts[:, 1] - cy
        dst_pts = np.column_stack((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)).astype(np.float32)
        M_full = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 2. Lighting & Shadow
        s = CONFIG["illumination_strength"]
        a = rng.uniform(-s, s)
        b = rng.uniform(-s, s)
        c = rng.uniform(-s * 0.5, s * 0.5)
        phi = rng.uniform(0, math.pi)
        d0 = rng.uniform(-0.4, 0.4)
        bw = CONFIG["shadow_width_pct"]
        shadow_depth = CONFIG["shadow_depth"]

        # 3. Blur & Noise
        blur_sigma = rng.uniform(CONFIG["blur_sigma_min"], CONFIG["blur_sigma_max"])
        do_motion = (rng.random() < CONFIG["motion_blur_prob"])
        motion_angle = rng.uniform(0, math.pi)
        noise_sigma = rng.uniform(CONFIG["sensor_noise_sigma_min"], CONFIG["sensor_noise_sigma_max"])
        noise_arr = rng.normal(0, 1.0, (h, w, 3)).astype(np.float32)
        paper_noise = rng.normal(0, CONFIG["paper_texture_std"], (h, w, 3)).astype(np.float32)

        # 4. Downscale & JPEG
        scale = rng.uniform(CONFIG["downscale_factor_min"], CONFIG["downscale_factor_max"])
        jpeg_q = int(rng.uniform(CONFIG["jpeg_quality_min"], CONFIG["jpeg_quality_max"]))

        pages.append({
            "page_idx": page_idx,
            "style": style,
            "font_name": font_name,
            "font_path": font_path,
            "font_size": font_size,
            "lines": page_lines,
            "line_boxes": line_boxes,
            "word_boxes": word_boxes,
            "M_full": M_full,
            "illum_params": (a, b, c),
            "shadow_params": (phi, d0, bw, shadow_depth),
            "blur_sigma": blur_sigma,
            "do_motion": do_motion,
            "motion_angle": motion_angle,
            "noise_sigma": noise_sigma,
            "noise_arr": noise_arr,
            "paper_noise": paper_noise,
            "scale": scale,
            "jpeg_q": jpeg_q,
        })
    return pages


def render_page_image(page_plan, level):
    """Render page at the specified distortion level."""
    w = CONFIG["page_width"]
    h = CONFIG["page_height"]
    font = ImageFont.truetype(page_plan["font_path"], page_plan["font_size"])

    # 1. Base Paper & Text Color
    if level in ("L0", "L0c"):
        bg = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)
        text_color = (0, 0, 0)
    else:
        bg_base = np.full((h, w, 3), CONFIG["paper_bg_rgb"], dtype=np.float32)
        bg = np.clip(bg_base + page_plan["paper_noise"], 0, 255).astype(np.uint8)
        text_color = CONFIG["text_color_rgb"]

    pil_img = Image.fromarray(bg)
    draw = ImageDraw.Draw(pil_img)

    for text, box, _ in page_plan["line_boxes"]:
        draw.text((box[0], box[1]), text, font=font, fill=text_color, anchor="lt")

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

    # Transform matrix M (identity by default; warp only in L4/L4n)
    if level in ("L4", "L4n"):
        M = page_plan["M_full"]
        warped = cv2.warpPerspective(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        img = warped.astype(np.float32)
    else:
        M = np.eye(3, dtype=np.float32)
        img = img_bgr

    # If L0 or L0c, no photometric distortions
    if level in ("L0", "L0c"):
        return np.clip(img, 0, 255).astype(np.uint8), M

    # 2. Uneven Illumination Gradient & Shadow (L1, L2, L3, L4, L4n)
    gx = np.linspace(-1, 1, w, dtype=np.float32)
    gy = np.linspace(-1, 1, h, dtype=np.float32)
    X, Y = np.meshgrid(gx, gy)
    a, b, c = page_plan["illum_params"]
    illum = 1.0 + a * X + b * Y + c * X * Y

    phi, d0, bw, sdepth = page_plan["shadow_params"]
    nx, ny = math.cos(phi), math.sin(phi)
    dist = X * nx + Y * ny - d0
    shadow = 1.0 - sdepth * np.exp(-(dist / bw) ** 2)

    img = img * illum[:, :, None] * shadow[:, :, None]

    if level == "L1":
        return np.clip(img, 0, 255).astype(np.uint8), M

    # 3. Gaussian Blur & Sensor Noise (L2, L3, L4, L4n)
    sigma = page_plan["blur_sigma"]
    img = cv2.GaussianBlur(img, (0, 0), sigma)

    if page_plan["do_motion"]:
        ksize = 3
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        m_angle = page_plan["motion_angle"]
        dx_k = int(round(math.cos(m_angle)))
        dy_k = int(round(math.sin(m_angle)))
        kernel[1, 1] = 1.0
        kernel[1 + dy_k, 1 + dx_k] = 1.0
        kernel /= kernel.sum()
        img = cv2.filter2D(img, -1, kernel)

    img += page_plan["noise_arr"] * page_plan["noise_sigma"]

    if level == "L2":
        return np.clip(img, 0, 255).astype(np.uint8), M

    # 4. Downscale/Upscale, Tint & JPEG (L3, L4, L4n)
    scale = page_plan["scale"]
    sw, sh = max(10, int(w * scale)), max(10, int(h * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    img[:, :, 2] *= CONFIG["warm_tint_r"]  # Red
    img[:, :, 0] *= CONFIG["warm_tint_b"]  # Blue
    img = np.clip(img, 0, 255).astype(np.uint8)

    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), page_plan["jpeg_q"]])
    final_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return final_bgr, M


def crop_items(image_bgr, boxes_info, M, margin, is_direct_slice=False, normalize_contrast=False):
    """Crop bounding boxes from image using homography mapping or direct slice."""
    h, w = image_bgr.shape[:2]
    crops = []
    for text, box, font_name in boxes_info:
        x0, y0, x1, y1 = box

        if is_direct_slice:
            cx0 = max(0, int(math.floor(x0 - margin)))
            cy0 = max(0, int(math.floor(y0 - margin)))
            cx1 = min(w, int(math.ceil(x1 + margin)))
            cy1 = min(h, int(math.ceil(y1 + margin)))
        else:
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

        if normalize_contrast:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            norm_gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            crop = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)

        crops.append((text, crop, font_name))
    return crops


def build_ladder_lines_sheet(sample_crops_per_level, out_path):
    """Render 3 lines across all levels with level names as labels."""
    levels = list(sample_crops_per_level.keys())
    sheet_w = 1500
    row_h = 65
    header_h = 50
    total_h = header_h + len(levels) * row_h + 30

    canvas = Image.new("RGB", (sheet_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    hdr_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
    lbl_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)

    draw.text((20, 15), "Distortion Ladder: Comparison of the Same 3 Line Crops Across Levels", fill=(0, 40, 120), font=hdr_font)

    y_cur = header_h
    for lvl in levels:
        draw.text((20, y_cur + 20), f"{lvl:4s}", fill=(40, 40, 40), font=lbl_font)
        x_cur = 90
        crops = sample_crops_per_level[lvl]
        for text, crop_bgr in crops:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            if pil_crop.height > 45:
                s = 45.0 / pil_crop.height
                pil_crop = pil_crop.resize((max(10, int(pil_crop.width * s)), 45), Image.Resampling.LANCZOS)
            canvas.paste(pil_crop, (x_cur, y_cur + 10))
            x_cur += pil_crop.width + 30
        y_cur += row_h

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Build distortion ladder datasets.")
    parser.add_argument("--seed", type=int, default=7, help="Fixed random seed.")
    parser.add_argument("--out", type=str, default="data/sim_ladder", help="Base output directory.")
    args = parser.parse_args()

    fonts_to_use = [
        ("serif", "pala.ttf", "C:/Windows/Fonts/pala.ttf"),
        ("serif", "cambria.ttc", "C:/Windows/Fonts/cambria.ttc"),
        ("sans", "micross.ttf", "C:/Windows/Fonts/micross.ttf"),
        ("mono", "CascadiaMono.ttf", "C:/Windows/Fonts/CascadiaMono.ttf"),
    ]

    for style, name, path in fonts_to_use:
        assert Path(path).exists(), f"Font not found: {path}"
        assert font_ok(path), f"Font failed font_ok: {path}"
    print(f"Verified 4 ladder fonts: {[f[1] for f in fonts_to_use]}")

    corpus_lines = load_corpus_lines("data/corpus_line/test.txt")
    print(f"Loaded {len(corpus_lines)} corpus lines.")

    pages_plan = setup_pages_plan(fonts_to_use, corpus_lines, seed=args.seed)
    total_lines = sum(len(p["line_boxes"]) for p in pages_plan)
    total_words = sum(len(p["word_boxes"]) for p in pages_plan)
    print(f"Planned 4 pages: {total_lines} line crops, {total_words} word crops per level.")

    levels = ["L0", "L0c", "L1", "L2", "L3", "L4", "L4n"]
    out_base = Path(args.out)
    results_dir = Path("results/sim_ladder")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Pick 3 line indices rich in special characters for visual ladder
    special_chars = set("əıöğüşçƏİÖĞÜŞÇ")
    flat_lines_meta = []
    for p in pages_plan:
        flat_lines_meta.extend(p["line_boxes"])

    ranked_lines = sorted(enumerate(flat_lines_meta), key=lambda x: sum(1 for c in x[1][0] if c in special_chars), reverse=True)
    sample_indices = [idx for idx, _ in ranked_lines[:3]]
    print(f"Selected sample line indices for ladder_lines.png: {sample_indices}")

    sample_crops_per_level = {}

    for lvl in levels:
        print(f"Processing Level: {lvl}...")
        lvl_line_dir = out_base / lvl / "line" / "test"
        lvl_word_dir = out_base / lvl / "word" / "test"
        (lvl_line_dir / "images").mkdir(parents=True, exist_ok=True)
        (lvl_word_dir / "images").mkdir(parents=True, exist_ok=True)

        all_lines = []
        all_words = []

        for p in pages_plan:
            page_bgr, M = render_page_image(p, lvl)

            # Save full pages for L0, L2, L4
            if p["page_idx"] == 0 and lvl in ("L0", "L2", "L4"):
                save_p = results_dir / f"page_{lvl}.jpg"
                cv2.imwrite(str(save_p), page_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                print(f"  Saved full page: {save_p}")

            is_direct = (lvl == "L0c")
            is_norm = (lvl == "L4n")

            line_crops = crop_items(page_bgr, p["line_boxes"], M, CONFIG["crop_margin_px"], is_direct_slice=is_direct, normalize_contrast=is_norm)
            word_crops = crop_items(page_bgr, p["word_boxes"], M, CONFIG["crop_margin_px"], is_direct_slice=is_direct, normalize_contrast=is_norm)

            all_lines.extend(line_crops)
            all_words.extend(word_crops)

        # Save crops and labels
        with open(lvl_line_dir / "labels.tsv", "w", encoding="utf-8") as f_l:
            for idx, (text, img, font_name) in enumerate(all_lines):
                fn = f"images/{idx:07d}.png"
                cv2.imwrite(str(lvl_line_dir / fn), img)
                f_l.write(f"{fn}\t{text}\n")

        with open(lvl_word_dir / "labels.tsv", "w", encoding="utf-8") as f_w:
            for idx, (text, img, font_name) in enumerate(all_words):
                fn = f"images/{idx:07d}.png"
                cv2.imwrite(str(lvl_word_dir / fn), img)
                f_w.write(f"{fn}\t{text}\n")

        # Collect sample lines for contact sheet
        sample_crops_per_level[lvl] = [(all_lines[i][0], all_lines[i][1]) for i in sample_indices]

    # Render ladder lines sheet
    ladder_sheet_path = results_dir / "ladder_lines.png"
    build_ladder_lines_sheet(sample_crops_per_level, ladder_sheet_path)
    print(f"Saved ladder lines sheet: {ladder_sheet_path}")

    # Save font metadata for per-font evaluation
    meta = {
        "line_fonts": [item[2] for item in all_lines],
        "word_fonts": [item[2] for item in all_words],
    }
    with open(out_base / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved font metadata to {out_base / 'metadata.json'}")


if __name__ == "__main__":
    main()
