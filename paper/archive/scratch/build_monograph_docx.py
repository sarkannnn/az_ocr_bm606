"""
Script to build a complete, publication-grade Word Document (paper/paper_v3.docx)
from paper/paper_v3.md, embedding all figures (f1-f14) and all tables (Table 1 - Table 9, Table 4a, Appendix Table).
"""

import sys
import os
import re
from pathlib import Path

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parent.parent.parent
PAPER_MD = WORKSPACE / "paper" / "paper_v3.md"
FIGURES_DIR = WORKSPACE / "paper" / "figures"
TABLES_DIR = WORKSPACE / "paper" / "tables"
OUTPUT_DOCX = WORKSPACE / "paper" / "paper_v3.docx"

# Color constants
C_NAVY = RGBColor(27, 54, 93)       # #1B365D Primary headings
C_SLATE = RGBColor(44, 62, 80)      # #2C3E50 Secondary headings
C_CHARCOAL = RGBColor(34, 34, 34)   # #222222 Body text
C_MUTED = RGBColor(100, 100, 100)   # #646464 Captions/Notes
C_CODE = RGBColor(160, 40, 40)      # #A02828 Inline code
C_MATH = RGBColor(20, 60, 120)      # #143C78 Math equations

HEX_NAVY = "1B365D"
HEX_SLATE = "2C3E50"
HEX_ZEBRA = "F8F9FA"
HEX_BORDER = "CCCCCC"
HEX_HDR_BORDER = "1B365D"

def parse_markdown_table(text_or_lines):
    """Parse a markdown pipe table into title, headers, alignments, rows, notes."""
    if isinstance(text_or_lines, str):
        lines = text_or_lines.strip().splitlines()
    else:
        lines = text_or_lines

    title = ""
    table_lines = []
    notes = []
    in_table = False

    for line in lines:
        s = line.strip()
        if s.startswith("# Table") or s.startswith("Table "):
            title = s.lstrip("#").strip()
        elif s.startswith("|"):
            in_table = True
            table_lines.append(s)
        elif in_table and not s.startswith("|"):
            if s and not s.startswith("---"):
                notes.append(s)
        elif not in_table and not s.startswith("#") and not s.startswith("---"):
            if s:
                notes.append(s)

    headers = []
    alignments = []
    rows = []

    if table_lines:
        headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
        data_start = 1
        if len(table_lines) > 1 and re.match(r"^[\|\s:\-]+$", table_lines[1]):
            data_start = 2
            raw_aligns = [c.strip() for c in table_lines[1].strip("|").split("|")]
            for a in raw_aligns:
                if a.startswith(":") and a.endswith(":"):
                    alignments.append("center")
                elif a.endswith(":"):
                    alignments.append("right")
                else:
                    alignments.append("left")
        else:
            alignments = ["left"] * len(headers)

        while len(alignments) < len(headers):
            alignments.append("left")

        for line in table_lines[data_start:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            while len(cells) < len(headers):
                cells.append("")
            rows.append(cells[: len(headers)])

    return {
        "title": title,
        "headers": headers,
        "alignments": alignments,
        "rows": rows,
        "notes": notes,
    }

def clean_latex(math_str):
    """Simplify LaTeX markup into clean Unicode math text for Word."""
    s = math_str.strip()
    s = s.replace(r"\text{", "").replace(r"\mathbf{", "").replace(r"\mathcal{", "")
    s = s.replace(r"\Sigma", "Σ").replace(r"\Delta", "Δ").replace(r"\pi", "π")
    s = s.replace(r"\times", "×").replace(r"\cdot", "·").replace(r"\pm", "±")
    s = s.replace(r"\le", "≤").replace(r"\ge", "≥").replace(r"\approx", "≈")
    s = s.replace(r"\rightarrow", "→").replace(r"\cup", "∪").replace(r"\in", "∈")
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\_", "_").replace(r"\%", "%")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()

def add_inline(paragraph, text, base_font="Calibri", base_size=Pt(10.5), base_color=C_CHARCOAL, bold=False, italic=False):
    """Tokenize markdown inline formatting and append runs with proper styles."""
    pattern = re.compile(r"(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$|[^\*\`\$]+)")
    tokens = pattern.findall(text)

    for token in tokens:
        if not token:
            continue
        run = paragraph.add_run()
        run.font.name = base_font
        run.font.size = base_size
        run.font.color.rgb = base_color
        run.bold = bold
        run.italic = italic

        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            run.text = token[2:-2]
            run.bold = True
        elif token.startswith("*") and token.endswith("*") and len(token) >= 2:
            run.text = token[1:-1]
            run.italic = True
        elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
            run.text = token[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(base_size.pt * 0.92)
            run.font.color.rgb = C_CODE
        elif token.startswith("$") and token.endswith("$") and len(token) >= 2:
            run.text = clean_latex(token[1:-1])
            run.font.name = "Cambria Math"
            run.font.color.rgb = C_MATH
            run.italic = True
        else:
            run.text = token

def set_cell_margins(cell, top=70, bottom=70, left=100, right=100):
    """Set inner cell padding in twips (1 pt = 20 twips)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcBorders {nsdecls("w")}/>'
    )
    # python-docx cell margin via tcMar
    margin_xml = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(margin_xml)

def render_table_to_docx(doc, table_data, font_size=Pt(8.0)):
    """Render a parsed markdown table as a native, booktabs-style Word table."""
    headers = table_data["headers"]
    rows = table_data["rows"]
    alignments = table_data["alignments"]
    title = table_data.get("title", "")
    notes = table_data.get("notes", [])

    if title:
        p_title = doc.add_paragraph()
        p_title.paragraph_format.space_before = Pt(12)
        p_title.paragraph_format.space_after = Pt(4)
        p_title.paragraph_format.keep_with_next = True
        run_t = p_title.add_run(title)
        run_t.font.name = "Calibri"
        run_t.font.size = Pt(10)
        run_t.font.bold = True
        run_t.font.color.rgb = C_NAVY

    if not headers or not rows:
        return

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Table borders: academic booktabs style (top rule, header divider, bottom rule)
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="8" w:space="0" w:color="{HEX_NAVY}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="{HEX_NAVY}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    table._tbl.tblPr.append(tblBorders)

    # Header row
    hdr_row = table.rows[0]
    hdr_trPr = hdr_row._tr.get_or_add_trPr()
    hdr_trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    hdr_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

    for c_idx, h_text in enumerate(headers):
        cell = hdr_row.cells[c_idx]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_NAVY}"/>')
        cell._tc.get_or_add_tcPr().append(shd)
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        if alignments[c_idx] == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif alignments[c_idx] == "right":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_inline(p, h_text, base_font="Calibri", base_size=Pt(font_size.pt + 0.5), base_color=RGBColor(255, 255, 255), bold=True)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        row_trPr = row._tr.get_or_add_trPr()
        row_trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        is_even = (r_idx % 2 == 1)
        for c_idx, cell_value in enumerate(row_data):
            cell = row.cells[c_idx]
            if is_even:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{HEX_ZEBRA}"/>')
                cell._tc.get_or_add_tcPr().append(shd)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            if alignments[c_idx] == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif alignments[c_idx] == "right":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_inline(p, cell_value, base_font="Calibri", base_size=font_size, base_color=C_CHARCOAL)

    # Footnotes / Notes
    if notes:
        for note in notes:
            p_note = doc.add_paragraph()
            p_note.paragraph_format.space_before = Pt(2)
            p_note.paragraph_format.space_after = Pt(6)
            add_inline(p_note, note, base_font="Calibri", base_size=Pt(8.0), base_color=C_MUTED, italic=True)

def main():
    print(f"Reading markdown from {PAPER_MD}...")
    md_text = PAPER_MD.read_text(encoding="utf-8")
    lines = md_text.splitlines()

    doc = docx.Document()

    # Page setup: Standard Letter, 0.75 in margins
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.75)
        sec.right_margin = Inches(0.75)
        sec.header_distance = Inches(0.4)
        sec.footer_distance = Inches(0.4)

        # Header
        hdr_p = sec.header.paragraphs[0]
        hdr_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hdr_run = hdr_p.add_run("Project BM 606: A CPU-Trained CRNN for Azerbaijani Printed Text")
        hdr_run.font.name = "Calibri"
        hdr_run.font.size = Pt(8.5)
        hdr_run.font.italic = True
        hdr_run.font.color.rgb = C_MUTED

    # Track embedded tables
    embedded_tables = set()

    line_idx = 0
    total_lines = len(lines)

    in_abstract = False
    in_code_block = False

    while line_idx < total_lines:
        line = lines[line_idx].strip()

        # Empty line
        if not line:
            line_idx += 1
            continue

        # Horizontal rule
        if line == "---":
            line_idx += 1
            continue

        # Code block marker
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
            else:
                in_code_block = True
            line_idx += 1
            continue

        # Title (# ...)
        if line.startswith("# ") and not line.startswith("## "):
            title_text = line[2:].strip()
            p_title = doc.add_paragraph()
            p_title.paragraph_format.space_before = Pt(18)
            p_title.paragraph_format.space_after = Pt(12)
            p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_title.add_run(title_text)
            run.font.name = "Calibri"
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = C_NAVY
            line_idx += 1
            continue

        # Authors / Affiliations / Date metadata
        if line.startswith("**Authors:**") or line.startswith("**Affiliations:**") or line.startswith("**Date:**") or line.startswith("**Document Status:**"):
            p_meta = doc.add_paragraph()
            p_meta.paragraph_format.space_before = Pt(1)
            p_meta.paragraph_format.space_after = Pt(2)
            p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p_meta, line, base_font="Calibri", base_size=Pt(9.5), base_color=C_SLATE)
            line_idx += 1
            continue

        # Abstract header
        if line == "## Abstract":
            in_abstract = True
            p_abs_h = doc.add_paragraph()
            p_abs_h.paragraph_format.space_before = Pt(14)
            p_abs_h.paragraph_format.space_after = Pt(4)
            p_abs_h.paragraph_format.left_indent = Inches(0.4)
            p_abs_h.paragraph_format.right_indent = Inches(0.4)
            p_abs_h.paragraph_format.keep_with_next = True
            run = p_abs_h.add_run("ABSTRACT")
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = C_NAVY
            line_idx += 1
            continue

        # Abstract text & Keywords
        if in_abstract:
            if line.startswith("## "):
                in_abstract = False
                # fall through to header processing below
            else:
                p_abs = doc.add_paragraph()
                p_abs.paragraph_format.space_before = Pt(2)
                p_abs.paragraph_format.space_after = Pt(4)
                p_abs.paragraph_format.left_indent = Inches(0.4)
                p_abs.paragraph_format.right_indent = Inches(0.4)
                p_abs.paragraph_format.line_spacing = 1.15
                if line.startswith("**Keywords:**"):
                    add_inline(p_abs, line, base_font="Calibri", base_size=Pt(9.0), base_color=C_CHARCOAL, italic=False)
                    in_abstract = False
                else:
                    add_inline(p_abs, line, base_font="Calibri", base_size=Pt(9.5), base_color=C_CHARCOAL)
                line_idx += 1
                continue

        # Headings: H2 (## ...)
        if line.startswith("## ") and not line.startswith("### "):
            sec_title = line[3:].strip()
            p_h = doc.add_paragraph()
            p_h.paragraph_format.space_before = Pt(16)
            p_h.paragraph_format.space_after = Pt(5)
            p_h.paragraph_format.keep_with_next = True
            run = p_h.add_run(sec_title)
            run.font.name = "Calibri"
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = C_NAVY
            line_idx += 1
            continue

        # Headings: H3 (### ...)
        if line.startswith("### "):
            sec_title = line[4:].strip()
            p_h = doc.add_paragraph()
            p_h.paragraph_format.space_before = Pt(11)
            p_h.paragraph_format.space_after = Pt(3)
            p_h.paragraph_format.keep_with_next = True
            run = p_h.add_run(sec_title)
            run.font.name = "Calibri"
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = C_SLATE
            line_idx += 1
            continue

        # Display math ($$...$$)
        if line.startswith("$$") and line.endswith("$$") and len(line) >= 4:
            math_content = line[2:-2].strip()
            p_math = doc.add_paragraph()
            p_math.paragraph_format.space_before = Pt(6)
            p_math.paragraph_format.space_after = Pt(6)
            p_math.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_math.add_run(clean_latex(math_content))
            run.font.name = "Cambria Math"
            run.font.size = Pt(11)
            run.font.color.rgb = C_MATH
            run.italic = True
            line_idx += 1
            continue

        # Figure detection: ![Figure X](figures/...)
        fig_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", line)
        if fig_match:
            fig_caption_label = fig_match.group(1)
            fig_path_rel = fig_match.group(2)
            img_path = WORKSPACE / "paper" / fig_path_rel

            # Collect caption and source on following lines
            caption_line = ""
            source_line = ""
            if line_idx + 1 < total_lines and lines[line_idx + 1].strip().startswith("*Figure"):
                caption_line = lines[line_idx + 1].strip().strip("*")
                line_idx += 1
                if line_idx + 1 < total_lines and lines[line_idx + 1].strip().startswith("*Source:"):
                    source_line = lines[line_idx + 1].strip().strip("*")
                    line_idx += 1

            if img_path.exists():
                p_fig = doc.add_paragraph()
                p_fig.paragraph_format.space_before = Pt(12)
                p_fig.paragraph_format.space_after = Pt(4)
                p_fig.paragraph_format.keep_with_next = True
                p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_picture(str(img_path), width=Inches(6.2))

                # Caption
                if caption_line:
                    p_cap = doc.add_paragraph()
                    p_cap.paragraph_format.space_before = Pt(3)
                    p_cap.paragraph_format.space_after = Pt(2)
                    p_cap.paragraph_format.left_indent = Inches(0.2)
                    p_cap.paragraph_format.right_indent = Inches(0.2)
                    p_cap.paragraph_format.keep_with_next = True
                    add_inline(p_cap, caption_line, base_font="Calibri", base_size=Pt(9.0), base_color=C_CHARCOAL, italic=True)

                # Source
                if source_line:
                    p_src = doc.add_paragraph()
                    p_src.paragraph_format.space_before = Pt(1)
                    p_src.paragraph_format.space_after = Pt(10)
                    p_src.paragraph_format.left_indent = Inches(0.2)
                    p_src.paragraph_format.right_indent = Inches(0.2)
                    add_inline(p_src, source_line, base_font="Calibri", base_size=Pt(8.0), base_color=C_MUTED, italic=True)
            else:
                print(f"Warning: Image file not found: {img_path}")

            line_idx += 1
            continue

        # In-markdown tables (e.g. Table 4a or Appendix Table)
        if line.startswith("|") and line.endswith("|"):
            tbl_lines = [line]
            peek_idx = line_idx + 1
            while peek_idx < total_lines and lines[peek_idx].strip().startswith("|"):
                tbl_lines.append(lines[peek_idx].strip())
                peek_idx += 1

            tbl_data = parse_markdown_table(tbl_lines)
            render_table_to_docx(doc, tbl_data, font_size=Pt(7.5))
            line_idx = peek_idx
            continue

        # Regular paragraph or list item
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15

        if re.match(r"^\d+\.\s+", line):
            # Numbered list
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Inches(0.25)
            add_inline(p, line, base_font="Calibri", base_size=Pt(10.0), base_color=C_CHARCOAL)
        elif line.startswith("- ") or line.startswith("* "):
            # Bullet list
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Inches(0.25)
            # Replace bullet with clean dash or run
            bullet_text = line[2:].strip()
            add_inline(p, "•  " + bullet_text, base_font="Calibri", base_size=Pt(10.0), base_color=C_CHARCOAL)
        else:
            # Standard body paragraph
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(5)
            add_inline(p, line, base_font="Calibri", base_size=Pt(10.5), base_color=C_CHARCOAL)

        # Context-aware table insertions:
        # Check if we just processed a paragraph that references Table 1 - 9
        # 1. Table 1: Section 3.1
        if "In the synthetic line test set" in line and "table1" not in embedded_tables:
            t1_path = TABLES_DIR / "table1_dataset_sizes.md"
            if t1_path.exists():
                print("Embedding Table 1 (Dataset Sizes)...")
                t_data = parse_markdown_table(t1_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table1")

        # 2. Table 2: Section 3.2
        if "As shown in Table 2, multiple system fonts" in line and "table2" not in embedded_tables:
            t2_path = TABLES_DIR / "table2_fonts.md"
            if t2_path.exists():
                print("Embedding Table 2 (Fonts)...")
                t_data = parse_markdown_table(t2_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.0))
                embedded_tables.add("table2")

        # 3. Table 3: Section 4.2
        if "As documented in Table 3 and Figure 3:" in line and "table3" not in embedded_tables:
            t3_path = TABLES_DIR / "table3_hyperparameters.md"
            if t3_path.exists():
                print("Embedding Table 3 (Hyperparameters)...")
                t_data = parse_markdown_table(t3_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(8.0))
                embedded_tables.add("table3")

        # 4. Table 5 & Table 6: Section 6.3
        if "As detailed in Table 5 and Figure 6:" in line and "table5" not in embedded_tables:
            t5_path = TABLES_DIR / "table5_confidence_intervals.md"
            if t5_path.exists():
                print("Embedding Table 5 (Confidence Intervals)...")
                t_data = parse_markdown_table(t5_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table5")

            t6_path = TABLES_DIR / "table6_special_letters.md"
            if t6_path.exists() and "table6" not in embedded_tables:
                print("Embedding Table 6 (Special Letters)...")
                t_data = parse_markdown_table(t6_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table6")

        # 5. Table 4: Section 6.6
        if "Results (Table 4 and Figure 11) demonstrate acute model fragility:" in line and "table4" not in embedded_tables:
            t4_path = TABLES_DIR / "table4_main_results.md"
            if t4_path.exists():
                print("Embedding Table 4 (Main Results Across Benchmarks)...")
                t_data = parse_markdown_table(t4_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.0))
                embedded_tables.add("table4")

        # 6. Table 8: Section 6.7
        if "Clean Subset Evaluation (23 lines with zero empty outputs;" in line and "table8" not in embedded_tables:
            t8_path = TABLES_DIR / "table8_tesseract_psm.md"
            if t8_path.exists():
                print("Embedding Table 8 (Tesseract PSM)...")
                t_data = parse_markdown_table(t8_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table8")

        # 7. Table 7: Section 7.1
        if "Character errors on Photo 5 were mapped to six mutually exclusive categories" in line and "table7" not in embedded_tables:
            t7_path = TABLES_DIR / "table7_error_taxonomy.md"
            if t7_path.exists():
                print("Embedding Table 7 (Error Taxonomy)...")
                t_data = parse_markdown_table(t7_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table7")

        # 8. Table 9: Section 8
        if "A central contribution of this study is the documentation and quantification of five project pitfalls" in line and "table9" not in embedded_tables:
            t9_path = TABLES_DIR / "table9_bugs_and_lessons.md"
            if t9_path.exists():
                print("Embedding Table 9 (Pitfalls and Lessons)...")
                t_data = parse_markdown_table(t9_path.read_text(encoding="utf-8"))
                render_table_to_docx(doc, t_data, font_size=Pt(7.5))
                embedded_tables.add("table9")

        line_idx += 1

    print(f"Saving final monograph docx to {OUTPUT_DOCX}...")
    doc.save(str(OUTPUT_DOCX))
    print(f"Successfully generated {OUTPUT_DOCX} ({os.path.getsize(OUTPUT_DOCX):,} bytes).")

if __name__ == "__main__":
    main()
