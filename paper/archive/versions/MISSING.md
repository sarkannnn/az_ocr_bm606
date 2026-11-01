# Missing Numbers and Data Inventory (paper/MISSING.md)

In strict accordance with the project's scientific verification rules, no numbers have been invented, estimated, or extrapolated. Any metric, parameter, or sample count not explicitly found in existing repository artifacts (`metrics.json`, `.csv`, `.log`, `.md`) is recorded here as **NOT AVAILABLE**.

---

### 1. Evaluation Metrics Not Recorded in Result Files

1. **Simulation Ladder (L0–L4n) Tesseract and V2 Word/Line WER:**
   - *Status:* **NOT AVAILABLE**
   - *Context:* In `results/sim_ladder/summary.md:5-14`, the table records `Word CRNN CER`, `Word CRNN WER`, `Line CRNN CER`, `Line CRNN WER`, `Tess Word CER`, and `Tess Line CER`. Tesseract Word/Line WER and V2 Word/Line WER across all individual ladder levels were not logged in the file.
   - *Handling in Paper:* Marked as "NOT AVAILABLE" in Table 4 and Figure 5 text.

2. **Per-Group Cluster Bootstrap WER Intervals:**
   - *Status:* **NOT AVAILABLE**
   - *Context:* In `results/v2_audit4/summary.md:118-132`, cluster-level bootstrap confidence intervals were computed and logged for character error rates (CER) across Groups A, B, and C. Cluster bootstrap intervals for Word Error Rate (WER) per individual group were not calculated or saved in `audit4_metrics.json` (WER decomposition was performed in Section 6 as a pooled set of 1,076 words).
   - *Handling in Paper:* Marked as "NOT AVAILABLE" in Table 5.

3. **GPU Memory Footprint (`gpu_mem_mb`):**
   - *Status:* **NOT AVAILABLE (`null`)**
   - *Context:* In all `metrics.json` files (`results/word_synth/metrics.json`, `results/line_synth/metrics.json`, etc.), the field `"gpu_mem_mb"` is explicitly set to `null` because training and inference were conducted entirely on CPU (Intel Core i7 Windows environment).
   - *Handling in Paper:* Disclosed as CPU-only execution; GPU memory is omitted.

4. **Intermediate Border Padding Values (0 px, 8 px, 12 px):**
   - *Status:* **NOT AVAILABLE (Never Executed)**
   - *Context:* In `results/real_fix9/metrics.json`, the padding sensitivity experiment was conducted under two preregistered conditions: Condition A (4 px margin) and Condition B (20 px padded white border). No runs with 0 px, 8 px, or 12 px padding exist on disk.
   - *Handling in Paper:* Disclosed in Section 6.7 and Figure 11; no new runs were launched.

---

### 2. Code Provenance and Artifact Limitations

1. **Original Dataset Generation Script for V2 (`scratch/build_v2_datasets.py`):**
   - *Status:* **PARTIALLY PRESERVED / REPRODUCIBILITY LIMITATION**
   - *Context:* As noted in `results/v2_audit/summary.md:10` and `results/archive/Az_OCR_Akademik_Hesabat_FIXES_v2.md:19`, while `scratch/build_v2_datasets.py` was restored in scratch, the original render loop parameters for V2 additions (15,000 words and 6,000 lines) were derived from `generate_data.py` calls. Exact seeds and command line history for the original run on 2026-10-23 were not committed to git.
   - *Handling in Paper:* Disclosed as a reproducibility limitation in Section 3 and the Appendix.

2. **Per-Crop Inference Latency Breakdown:**
   - *Status:* **VARIABLE BY HARDWARE LOAD**
   - *Context:* As documented in `results/archive/final_report.md:102-105`, `evaluate.py` records pure PyTorch tensor inference time (5.21 ms/image in `results/word_synth/metrics.json` vs 16.41 ms in `logs/run_all.log`), whereas Tesseract includes external subprocess invocation, disk I/O, and image conversion (96.85 ms/image). A standardized end-to-end latency measurement under identical process conditions is not available.
   - *Handling in Paper:* Disclosed in Section 5.3; latency comparison is labeled non-comparable.
