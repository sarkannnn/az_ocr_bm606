# AzOCR-V2 — Photo 3 (D3 / B3) CER Discrepancy Reconciliation: 0.10% vs. 0.89% (Task T07)

## 1. Root Cause Summary

The two reported CER values for **Photo 3 (`IMG_5123.jpg`, Times New Roman, 23 lines, 1,015 NFC chars)** are **both real** and originate from **different checkpoints/decoding/post-processing configurations** across two separate evaluations:

| Metric Value | Source File | Model / Key | Decoding / Post-Processing | Edit Dist / Denom | Protocol |
|---:|---|---|---|---|---|
| **0.10%** (`0.0985%`) | `results/photo3_test/photo3_metrics.json` | `v2_rule_chunk` (early CRNN-V2 diagnostic run) | Rule-based chunking, **raw CTC greedy (no SymSpell)** | `1 / 1,015` chars (`22/23` exact) | Diagnostic D3 test |
| **0.00%** | `results/unseen_benchmark/unseen_metrics.json` | `v2_raw` (`runs/line_v2/best.pt`) | `infer_line_smart`, **raw CTC greedy (no SymSpell)** | `0 / 1,015` chars (`23/23` exact) | B1–B15 benchmark (`B3`) |
| **0.89%** | `results/unseen_benchmark/unseen_metrics.json` | `v2_pp` (`runs/line_v2/best.pt`) | `infer_line_smart` + **SymSpell (`max_edit_distance=1`)** | `9 / 1,015` chars (`20/23` exact, WER `2.17%`) | B1–B15 benchmark (`B3`) |

## 2. Why SymSpell Turned 0.00% into 0.89% on Photo 3

As documented in `PHOTO_MAPPING_AUDIT.md` (Section 2), the physical A4 printout of D3/B3 contains **4 real typographical errors in the printed text itself** (`kimliyinir`, `tarixirə`, `hsduğu`, `ənənələnin`).
- **`v2_raw`** transcribed the printed page verbatim (`0 / 1,015` = **0.00% CER**, `23/23` exact).
- **`v2_pp`** applied SymSpell unigram dictionary correction (`data/az_lexicon.txt`) on top of `v2_raw`, altering those printed tokens toward dictionary words and thereby **introducing 9 character edits** relative to the verbatim printed ground truth (`9 / 1,015` = **0.89% CER**, `20/23` exact).
- **`0.10%`** is the earlier `v2_rule_chunk` raw result (`1 / 1,015` = `0.0985%`) from `results/photo3_test/photo3_metrics.json` before SymSpell post-processing was applied.

**Correct Interpretation:** Never mix `v2_rule_chunk` (`0.10%`), `v2_raw` (`0.00%`), and `v2_pp` (`0.89%`); always label whether SymSpell post-processing was enabled.
