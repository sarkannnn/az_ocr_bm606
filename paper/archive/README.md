# Paper Archive Directory

This directory contains superseded historical versions, draft figure generation scripts, intermediate scratch tools, and legacy figures from earlier iterations of Project BM 606 Azerbaijani OCR.

---

### Archive Contents

1. **`versions/`**
   - `paper.md` (Version 1 initial draft)
   - `paper_v3.md` / `paper_v3.docx` (Version 3 draft before EasyOCR benchmark integration)
   - `CHANGES_v3.md`, `MISSING.md`, `MISSING_v3.md` (Historical version diff logs)

2. **`scripts/`**
   - `generate_figures.py` (Version 1 figure generation script)
   - `generate_figures_v2.py` (Version 2 audit figure generation script)
   - `generate_figures_v5.py` (Version 5 experimental draft script)
   - *Current Master Script:* `../generate_figures_master.py`

3. **`figures_v5/`**
   - Duplicate / experimental draft figures rendered during Version 5 development.

4. **`superseded_figures/`**
   - `f8_real_photos_detection.png` (replaced by `_v2`)
   - `f9_line_annotations.png` (replaced by `_v2`)
   - `f11_padding_sensitivity.png` (replaced by `_v2`)
   - `f13_tess_psm_comparison.png` (replaced by `_v2`)
   - `_panel_b_photo5_detected.png`, `_panel_b_photo5_only.png` (temporary visual crops)

5. **`scratch/`**
   - `build_monograph_docx.py` (v3 Word builder)
   - `build_changes_v3.py`
   - `add_easyocr_to_v4.py`, `update_and_build_v4.py`, `update_master_tables_v4.py` (one-off data migration scripts)
   - `dump_data.py`, `dump_logs.py`, `dump_metrics.py` (diagnostic inspection tools)

---

### Active Manuscript Files (Do NOT Move)
- **Manuscript (Word):** `../paper_v4.docx`
- **Manuscript (Markdown):** `../paper_v4.md`
- **Active Figures:** `../figures/` (15 canonical figures: `f1` through `f15`)
- **Active Tables:** `../tables/` (15 data tables: Table 1 through Table 9, 3b, 4b, 4c, 6b, 8b)
- **Master Figure Generator:** `../generate_figures_master.py`
- **Active Word Builder:** `../scratch/build_monograph_docx_v4.py`

