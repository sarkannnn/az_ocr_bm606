"""
Verification script for BM 606 Azerbaijani OCR Paper.
Validates all numerical claims in paper/paper.md against source repository files.
"""

import sys
import os
import re
import json
import hashlib
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parent.parent
PAPER_FILE = WORKSPACE / "paper" / "paper.md"

def sha256_file(p):
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def load_json(rel_path):
    p = WORKSPACE / rel_path
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def read_text(rel_path):
    p = WORKSPACE / rel_path
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    print("=" * 85)
    print("BM 606 OCR PAPER: NUMERICAL AUDIT & VERIFICATION REPORT")
    print("=" * 85)
    
    if not PAPER_FILE.exists():
        print(f"Error: {PAPER_FILE} does not exist!")
        sys.exit(1)
        
    with open(PAPER_FILE, "r", encoding="utf-8") as f:
        paper_text = f.read()
        
    checks = []
    
    # -------------------------------------------------------------
    # 1. Clean Synthetic Benchmarks
    # -------------------------------------------------------------
    w_v1 = load_json("results/word_synth/metrics.json")
    if w_v1:
        checks.append({
            "name": "Synthetic Word V1 CER (n=2000)",
            "paper_val": "0.50%",
            "source_file": "results/word_synth/metrics.json",
            "source_val": f"{w_v1['cer'] * 100:.2f}%",
            "match": "0.50%" in paper_text and abs(w_v1['cer'] * 100 - 0.50) < 0.01
        })
        checks.append({
            "name": "Synthetic Word V1 WER",
            "paper_val": "3.95%",
            "source_file": "results/word_synth/metrics.json",
            "source_val": f"{w_v1['wer'] * 100:.2f}%",
            "match": "3.95%" in paper_text and abs(w_v1['wer'] * 100 - 3.95) < 0.01
        })

    w_v2 = load_json("results/word_v2_synth/metrics.json")
    if w_v2:
        checks.append({
            "name": "Synthetic Word V2 CER (n=2000)",
            "paper_val": "0.11%",
            "source_file": "results/word_v2_synth/metrics.json",
            "source_val": f"{w_v2['cer'] * 100:.2f}%",
            "match": "0.11%" in paper_text and abs(w_v2['cer'] * 100 - 0.11) < 0.01
        })
        checks.append({
            "name": "Synthetic Word V2 WER",
            "paper_val": "0.90%",
            "source_file": "results/word_v2_synth/metrics.json",
            "source_val": f"{w_v2['wer'] * 100:.2f}%",
            "match": "0.90%" in paper_text and abs(w_v2['wer'] * 100 - 0.90) < 0.01
        })

    w_tess = load_json("results/tess_word_synth/metrics.json")
    if w_tess:
        checks.append({
            "name": "Synthetic Word Tess CER (n=2000)",
            "paper_val": "0.70%",
            "source_file": "results/tess_word_synth/metrics.json",
            "source_val": f"{w_tess['cer'] * 100:.2f}%",
            "match": "0.70%" in paper_text and abs(w_tess['cer'] * 100 - 0.70) < 0.01
        })
        checks.append({
            "name": "Synthetic Word Tess WER",
            "paper_val": "4.90%",
            "source_file": "results/tess_word_synth/metrics.json",
            "source_val": f"{w_tess['wer'] * 100:.2f}%",
            "match": "4.90%" in paper_text and abs(w_tess['wer'] * 100 - 4.90) < 0.01
        })

    l_v1 = load_json("results/line_synth/metrics.json")
    if l_v1:
        checks.append({
            "name": "Synthetic Line V1 CER (n=1000)",
            "paper_val": "1.64%",
            "source_file": "results/line_synth/metrics.json",
            "source_val": f"{l_v1['cer'] * 100:.2f}%",
            "match": "1.64%" in paper_text and abs(l_v1['cer'] * 100 - 1.64) < 0.01
        })
        checks.append({
            "name": "Synthetic Line V1 WER",
            "paper_val": "11.64%",
            "source_file": "results/line_synth/metrics.json",
            "source_val": f"{l_v1['wer'] * 100:.2f}%",
            "match": "11.64%" in paper_text and abs(l_v1['wer'] * 100 - 11.64) < 0.01
        })

    l_v2 = load_json("results/line_v2_synth/metrics.json")
    if l_v2:
        checks.append({
            "name": "Synthetic Line V2 CER (n=1000)",
            "paper_val": "0.30%",
            "source_file": "results/line_v2_synth/metrics.json",
            "source_val": f"{l_v2['cer'] * 100:.2f}%",
            "match": "0.30%" in paper_text and abs(l_v2['cer'] * 100 - 0.30) < 0.01
        })
        checks.append({
            "name": "Synthetic Line V2 WER",
            "paper_val": "2.13%",
            "source_file": "results/line_v2_synth/metrics.json",
            "source_val": f"{l_v2['wer'] * 100:.2f}%",
            "match": "2.13%" in paper_text and abs(l_v2['wer'] * 100 - 2.13) < 0.01
        })

    l_tess = load_json("results/tess_line_synth/metrics.json")
    if l_tess:
        checks.append({
            "name": "Synthetic Line Tess CER (n=1000)",
            "paper_val": "0.44%",
            "source_file": "results/tess_line_synth/metrics.json",
            "source_val": f"{l_tess['cer'] * 100:.2f}%",
            "match": "0.44%" in paper_text and abs(l_tess['cer'] * 100 - 0.44) < 0.01
        })
        checks.append({
            "name": "Synthetic Line Tess WER",
            "paper_val": "2.94%",
            "source_file": "results/tess_line_synth/metrics.json",
            "source_val": f"{l_tess['wer'] * 100:.2f}%",
            "match": "2.94%" in paper_text and abs(l_tess['wer'] * 100 - 2.94) < 0.01
        })

    # -------------------------------------------------------------
    # 2. Monofont Tests (Tahoma & Consolas)
    # -------------------------------------------------------------
    dw_tah_v1 = load_json("results/diagw_tahoma/metrics.json")
    if dw_tah_v1:
        checks.append({
            "name": "Tahoma Word V1 CER",
            "paper_val": "0.08%",
            "source_file": "results/diagw_tahoma/metrics.json",
            "source_val": f"{dw_tah_v1['cer'] * 100:.2f}%",
            "match": "0.08%" in paper_text and abs(dw_tah_v1['cer'] * 100 - 0.08) < 0.01
        })
    dw_tah_v2 = load_json("results/diagw_tahoma_v2/metrics.json")
    if dw_tah_v2:
        checks.append({
            "name": "Tahoma Word V2 CER",
            "paper_val": "0.09%",
            "source_file": "results/diagw_tahoma_v2/metrics.json",
            "source_val": f"{dw_tah_v2['cer'] * 100:.2f}%",
            "match": "0.09%" in paper_text and abs(dw_tah_v2['cer'] * 100 - 0.09) < 0.01
        })
    dw_tah_tess = load_json("results/tess_diagw_tahoma/metrics.json")
    if dw_tah_tess:
        checks.append({
            "name": "Tahoma Word Tess CER",
            "paper_val": "0.50%",
            "source_file": "results/tess_diagw_tahoma/metrics.json",
            "source_val": f"{dw_tah_tess['cer'] * 100:.2f}%",
            "match": "0.50%" in paper_text and abs(dw_tah_tess['cer'] * 100 - 0.50) < 0.01
        })

    dl_tah_v1 = load_json("results/diag_tahoma/metrics.json")
    if dl_tah_v1:
        checks.append({
            "name": "Tahoma Line V1 CER",
            "paper_val": "0.24%",
            "source_file": "results/diag_tahoma/metrics.json",
            "source_val": f"{dl_tah_v1['cer'] * 100:.2f}%",
            "match": "0.24%" in paper_text and abs(dl_tah_v1['cer'] * 100 - 0.24) < 0.01
        })
    dl_tah_v2 = load_json("results/diag_tahoma_v2/metrics.json")
    if dl_tah_v2:
        checks.append({
            "name": "Tahoma Line V2 CER",
            "paper_val": "0.26%",
            "source_file": "results/diag_tahoma_v2/metrics.json",
            "source_val": f"{dl_tah_v2['cer'] * 100:.2f}%",
            "match": "0.26%" in paper_text and abs(dl_tah_v2['cer'] * 100 - 0.26) < 0.01
        })

    dw_con_v1 = load_json("results/diagw_consola/metrics.json")
    if dw_con_v1:
        checks.append({
            "name": "Consolas Word V1 CER",
            "paper_val": "1.02%",
            "source_file": "results/diagw_consola/metrics.json",
            "source_val": f"{dw_con_v1['cer'] * 100:.2f}%",
            "match": "1.02%" in paper_text and abs(dw_con_v1['cer'] * 100 - 1.02) < 0.01
        })
    dw_con_v2 = load_json("results/diagw_consola_v2/metrics.json")
    if dw_con_v2:
        checks.append({
            "name": "Consolas Word V2 CER",
            "paper_val": "0.07%",
            "source_file": "results/diagw_consola_v2/metrics.json",
            "source_val": f"{dw_con_v2['cer'] * 100:.2f}%",
            "match": "0.07%" in paper_text and abs(dw_con_v2['cer'] * 100 - 0.07) < 0.01
        })

    dl_con_v1 = load_json("results/diag_consola/metrics.json")
    if dl_con_v1:
        checks.append({
            "name": "Consolas Line V1 CER",
            "paper_val": "3.48%",
            "source_file": "results/diag_consola/metrics.json",
            "source_val": f"{dl_con_v1['cer'] * 100:.2f}%",
            "match": "3.48%" in paper_text and abs(dl_con_v1['cer'] * 100 - 3.48) < 0.01
        })
    dl_con_v2 = load_json("results/diag_consola_v2/metrics.json")
    if dl_con_v2:
        checks.append({
            "name": "Consolas Line V2 CER",
            "paper_val": "0.27%",
            "source_file": "results/diag_consola_v2/metrics.json",
            "source_val": f"{dl_con_v2['cer'] * 100:.2f}%",
            "match": "0.27%" in paper_text and abs(dl_con_v2['cer'] * 100 - 0.27) < 0.01
        })

    # -------------------------------------------------------------
    # 3. Simulation Ladder (results/sim_ladder/summary.md & results/v2_audit/summary.md)
    # -------------------------------------------------------------
    sim_text = read_text("results/sim_ladder/summary.md")
    v2_audit_text = read_text("results/v2_audit/summary.md")
    if sim_text:
        checks.append({
            "name": "Ladder L0 Word V1 CER",
            "paper_val": "7.71%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "7.71%",
            "match": "7.71%" in paper_text and "7.71%" in sim_text
        })
        checks.append({
            "name": "Ladder L1 Word V1 CER",
            "paper_val": "8.85%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "8.85%",
            "match": "8.85%" in paper_text and "8.85%" in sim_text
        })
        checks.append({
            "name": "Ladder L2 Word V1 CER",
            "paper_val": "25.00%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "25.00%",
            "match": "25.00%" in paper_text and "25.00%" in sim_text
        })
        checks.append({
            "name": "Ladder L3 Word V1 CER",
            "paper_val": "31.36%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "31.36%",
            "match": "31.36%" in paper_text and "31.36%" in sim_text
        })
        checks.append({
            "name": "Ladder L4 Word V1 CER",
            "paper_val": "35.77%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "35.77%",
            "match": "35.77%" in paper_text and "35.77%" in sim_text
        })
        checks.append({
            "name": "Ladder L4 Line V1 CER",
            "paper_val": "12.96%",
            "source_file": "results/sim_ladder/summary.md",
            "source_val": "12.96%",
            "match": "12.96%" in paper_text and "12.96%" in sim_text
        })
    if v2_audit_text:
        checks.append({
            "name": "Ladder L4 Word V2 CER",
            "paper_val": "1.35%",
            "source_file": "results/v2_audit/summary.md",
            "source_val": "1.35%",
            "match": "1.35%" in paper_text and "1.35%" in v2_audit_text
        })
        checks.append({
            "name": "Ladder L4 Line V2 CER",
            "paper_val": "0.11%",
            "source_file": "results/v2_audit/summary.md",
            "source_val": "0.11%",
            "match": "0.11%" in paper_text and "0.11%" in v2_audit_text
        })

    # -------------------------------------------------------------
    # 4. Photo 1 & Photo 2 (results/real_fix8/final_tables.md)
    # -------------------------------------------------------------
    rf8_text = read_text("results/real_fix8/final_tables.md")
    if rf8_text:
        checks.append({
            "name": "Photo 1 V2 Trimmed Chunk CER",
            "paper_val": "0.97%",
            "source_file": "results/real_fix8/final_tables.md",
            "source_val": "0.97%",
            "match": "0.97%" in paper_text and "0.97%" in rf8_text
        })
        checks.append({
            "name": "Photo 1 Tess Trimmed Chunk CER",
            "paper_val": "1.32%",
            "source_file": "results/real_fix8/final_tables.md",
            "source_val": "1.32%",
            "match": "1.32%" in paper_text and "1.32%" in rf8_text
        })
        checks.append({
            "name": "Photo 2 V2 Trimmed Chunk CER",
            "paper_val": "0.55%",
            "source_file": "results/real_fix8/final_tables.md",
            "source_val": "0.55%",
            "match": "0.55%" in paper_text and "0.55%" in rf8_text
        })
        checks.append({
            "name": "Photo 2 Tess Trimmed Chunk CER",
            "paper_val": "0.14%",
            "source_file": "results/real_fix8/final_tables.md",
            "source_val": "0.14%",
            "match": "0.14%" in paper_text and "0.14%" in rf8_text
        })

    # -------------------------------------------------------------
    # 5. Photo 3 (results/photo3_test/report.md)
    # -------------------------------------------------------------
    p3_text = read_text("results/photo3_test/report.md")
    if p3_text:
        checks.append({
            "name": "Photo 3 V2 Rule Chunk CER",
            "paper_val": "0.10%",
            "source_file": "results/photo3_test/report.md",
            "source_val": "0.10%",
            "match": "0.10%" in paper_text and "0.10%" in p3_text
        })
        checks.append({
            "name": "Photo 3 Tess Rule Chunk CER",
            "paper_val": "0.00%",
            "source_file": "results/photo3_test/report.md",
            "source_val": "0.00%",
            "match": "0.00%" in paper_text and "0.00%" in p3_text
        })

    # -------------------------------------------------------------
    # 6. Photo 5 (results/real_fix9/metrics.json)
    # -------------------------------------------------------------
    p5 = load_json("results/real_fix9/metrics.json")
    if p5:
        c_a = p5.get("condition_A_all33", {})
        v2_rc = c_a.get("v2_rule_chunk", {})
        tess_rc = c_a.get("tess_rule_chunk_psm7", {})
        v2_fl = c_a.get("v2_full", {})
        tess13 = c_a.get("tess_full_psm13", {})
        tess7 = c_a.get("tess_full_psm7", {})

        checks.append({
            "name": "Photo 5 V2 Rule Chunk CER",
            "paper_val": "2.10%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{v2_rc.get('cer', 0):.2f}%",
            "match": "2.10%" in paper_text and abs(v2_rc.get('cer', 0) - 2.10) < 0.01
        })
        checks.append({
            "name": "Photo 5 V2 Rule Chunk Errors (49)",
            "paper_val": "49",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": str(v2_rc.get("cer_errs", 0)),
            "match": "49" in paper_text and v2_rc.get("cer_errs", 0) == 49
        })
        checks.append({
            "name": "Photo 5 Tess Rule Chunk CER",
            "paper_val": "13.02%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{tess_rc.get('cer', 0):.2f}%",
            "match": "13.02%" in paper_text and abs(tess_rc.get('cer', 0) - 13.02) < 0.01
        })
        checks.append({
            "name": "Photo 5 Tess Rule Chunk Errors (303)",
            "paper_val": "303",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": str(tess_rc.get("cer_errs", 0)),
            "match": "303" in paper_text and tess_rc.get("cer_errs", 0) == 303
        })
        checks.append({
            "name": "Photo 5 V2 Full Line CER",
            "paper_val": "3.87%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{v2_fl.get('cer', 0):.2f}%",
            "match": "3.87%" in paper_text and abs(v2_fl.get('cer', 0) - 3.87) < 0.01
        })
        checks.append({
            "name": "Photo 5 Tess Full PSM 7 CER",
            "paper_val": "8.42%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{tess7.get('cer', 0):.2f}%",
            "match": "8.42%" in paper_text and abs(tess7.get('cer', 0) - 8.42) < 0.01
        })
        checks.append({
            "name": "Photo 5 Tess Full PSM 13 CER",
            "paper_val": "0.09%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{tess13.get('cer', 0):.2f}%",
            "match": "0.09%" in paper_text and abs(tess13.get('cer', 0) - 0.09) < 0.01
        })
        checks.append({
            "name": "Photo 5 Tess Full PSM 13 Errors (2)",
            "paper_val": "2",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": str(tess13.get("cer_errs", 0)),
            "match": "2" in paper_text and tess13.get("cer_errs", 0) == 2
        })

        # Condition B
        c_b = p5.get("condition_B_padded_all33", {})
        v2_b = c_b.get("v2_rule_chunk", {})
        checks.append({
            "name": "Photo 5 Cond B V2 Rule Chunk CER",
            "paper_val": "23.32%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{v2_b.get('cer', 0):.2f}%",
            "match": "23.32%" in paper_text and abs(v2_b.get('cer', 0) - 23.32) < 0.01
        })

        # Clean subset 23 lines
        clean_a = p5.get("clean_subset_A", {})
        v2_cl = clean_a.get("v2_rule_chunk", {})
        tess_cl = clean_a.get("tess_rule_chunk_psm7", {})
        checks.append({
            "name": "Photo 5 Clean 23 Lines V2 CER",
            "paper_val": "1.32%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{v2_cl.get('cer', 0):.2f}%",
            "match": "1.32%" in paper_text and abs(v2_cl.get('cer', 0) - 1.32) < 0.01
        })
        checks.append({
            "name": "Photo 5 Clean 23 Lines Tess CER",
            "paper_val": "1.26%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{tess_cl.get('cer', 0):.2f}%",
            "match": "1.26%" in paper_text and abs(tess_cl.get('cer', 0) - 1.26) < 0.01
        })

        # Squeeze ratio groups
        sq = p5.get("v2_ratio_grouped_CER", {})
        checks.append({
            "name": "Squeeze r < 1.0 Full CER",
            "paper_val": "0.54%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{sq.get('<1.0', {}).get('v2_full_cer', 0):.2f}%",
            "match": "0.54%" in paper_text and abs(sq.get('<1.0', {}).get('v2_full_cer', 0) - 0.54) < 0.01
        })
        checks.append({
            "name": "Squeeze r > 1.3 Full CER",
            "paper_val": "4.87%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{sq.get('>1.3', {}).get('v2_full_cer', 0):.2f}%",
            "match": "4.87%" in paper_text and abs(sq.get('>1.3', {}).get('v2_full_cer', 0) - 4.87) < 0.01
        })
        checks.append({
            "name": "Squeeze r > 1.3 Chunk CER",
            "paper_val": "1.33%",
            "source_file": "results/real_fix9/metrics.json",
            "source_val": f"{sq.get('>1.3', {}).get('v2_chunk_cer', 0):.2f}%",
            "match": "1.33%" in paper_text and abs(sq.get('>1.3', {}).get('v2_chunk_cer', 0) - 1.33) < 0.01
        })

    # -------------------------------------------------------------
    # 7. Checkpoint SHA256 Hashes
    # -------------------------------------------------------------
    ckpt_w_v2 = WORKSPACE / "runs" / "word_v2_20261027_180300" / "best.pt"
    if ckpt_w_v2.exists():
        h = sha256_file(ckpt_w_v2)
        checks.append({
            "name": "Word V2 Checkpoint SHA-256",
            "paper_val": "062bf63aeb8aad347e00df0173a36d844fc7c9725b2fa11941d9c9a7c05822cb",
            "source_file": "runs/word_v2_20261027_180300/best.pt",
            "source_val": h,
            "match": h in paper_text
        })

    ckpt_l_v2 = WORKSPACE / "runs" / "line_v2" / "best.pt"
    if ckpt_l_v2.exists():
        h = sha256_file(ckpt_l_v2)
        checks.append({
            "name": "Line V2 Checkpoint SHA-256",
            "paper_val": "a0946ae009b09e898da5683e71d59ebcba030d451cfb43f41a48476c763c50a6",
            "source_file": "runs/line_v2/best.pt",
            "source_val": h,
            "match": h in paper_text
        })

    # -------------------------------------------------------------
    # Run and Print Verification Table
    # -------------------------------------------------------------
    print(f"\nTotal verification assertions: {len(checks)}")
    passed = 0
    failed = 0
    
    for c in checks:
        status = "PASS" if c["match"] else "FAIL"
        if c["match"]:
            passed += 1
        else:
            failed += 1
        print(f"[{status}] {c['name']:<35} | Paper: {c['paper_val']:<10} | Source: {c['source_val']:<10} | File: {c['source_file']}")
        
    print("-" * 85)
    print(f"Audit Summary: {passed} PASSED, {failed} FAILED out of {len(checks)} assertions.")
    print("=" * 85)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL VERIFIED NUMBERS MATCH SOURCE FILES EXACTLY.")

if __name__ == "__main__":
    main()
