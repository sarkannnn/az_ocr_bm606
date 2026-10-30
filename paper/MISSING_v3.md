# Inventory of Unverified Items and Missing Data (paper/MISSING_v3.md)

In accordance with strict scientific integrity rules, any parameter, affiliation, or metric that cannot be directly verified in an existing repository artifact (`metrics.json`, `.csv`, `.log`, `.py`) is marked as `[to verify]` and inventoried below.

---

### 1. Text Placeholders Marked as `[to verify]`

1. **`[AFFILIATION]`**
   - *Location in Paper:* Authors header, line 4 of `paper/paper_v3.md`.
   - *Prior Text:* "Baku / Nakhchivan State University Academic Collaborative Project".
   - *Status:* **UNVERIFIED IN REPO FILES** (Replaced by `[AFFILIATION]`).
   - *Context:* No institutional affiliation string appears in any source code, dataset metadata, or execution log within the repository. Left as a clean placeholder for the user to specify.

2. **`[CPU MODEL]`**
   - *Location in Paper:* Abstract, Section 1.1, Section 4.2, Section 9.
   - *Prior Text:* "Intel Core i7 CPU hardware".
   - *Status:* **UNVERIFIED EXACT HARDWARE SPECIFICATION** (Replaced by `[CPU MODEL]`).
   - *Context:* All evaluation metrics explicitly record `"gpu_mem_mb": null` (confirming CPU execution), and project documentation notes a generic Intel Core i7 Windows environment, but the exact CPU model (e.g., i7-12700K, i7-10700) is nowhere logged in system metadata. Left as a clean placeholder for the user to specify.

---

### 2. Empirical Metrics Not Recorded in Repository Files

1. **Tesseract and V2 Word/Line Word Error Rate (WER) across Simulation Ladder Levels ($L_0$–$L_4n$):**
   - *Status:* **NOT AVAILABLE**
   - *Context:* `results/sim_ladder/summary.md` records `Word CRNN CER`, `Word CRNN WER`, `Line CRNN CER`, `Line CRNN WER`, `Tess Word CER`, and `Tess Line CER` for the baseline V1 model. Intermediate WERs for Tesseract and V2 across individual ladder rungs were not saved.
   - *Handling in Paper:* Omitted or marked as "NOT AVAILABLE" in Table 4a.

2. **Tesseract PSM 13 Standalone Evaluation across all 33 Rule Chunks:**
   - *Status:* **NOT MEASURED**
   - *Context:* In `results/real_fix9/metrics.json`, PSM 13 was evaluated comprehensively on Full Lines (`tess_full_psm13`), achieving 0.09% on all 33 lines and 0.00% on the 23-line clean subset. On rule chunks, PSM 13 was evaluated as a targeted fallback on the 9 chunks where PSM 7 returned empty strings (`results/real_fix9/empty_outputs_log.json`), but a standalone evaluation across all 33 chunks was not run.
   - *Handling in Paper:* Omitted from the chunk column and explicitly labeled as "not measured" in Figure 13 and Section 6.7.

3. **Intermediate Canvas Border Padding Values (0 px, 8 px, 12 px):**
   - *Status:* **NOT MEASURED**
   - *Context:* In `results/real_fix9/metrics.json`, the padding sensitivity experiment was executed exclusively under two preregistered conditions: Condition A (4 px margin) and Condition B (20 px white canvas border). No intermediate runs exist on disk.
   - *Handling in Paper:* Explicitly disclosed in Section 6.6, Table 4a, and Figure 11 as "two padding values only (4 px, 20 px)".

4. **Cluster Bootstrap Word Error Rate (WER) Intervals for Group C:**
   - *Status:* **NOT AVAILABLE**
   - *Context:* In `results/v2_audit4/summary.md:118-132`, cluster-level bootstrap confidence intervals were computed and logged for character error rates (CER). Per-group cluster bootstrap intervals for WER were not computed or logged in `audit4_metrics.json`.
   - *Handling in Paper:* Marked as "NOT AVAILABLE" in Table 5.

---

### 3. Code Provenance and Artifact Limitations

1. **Exact Random Seeds for Initial V2 Synthetic Data Generation:**
   - *Status:* **REPRODUCIBILITY LIMITATION**
   - *Context:* While `scratch/build_v2_datasets.py` documents the data augmentation recipe, the exact command-line seeds used during the initial dataset generation run on 2026-10-23 were not committed to git (`results/v2_audit/summary.md:10`).
   - *Handling in Paper:* Disclosed as an empirical reproducibility caveat in Section 9.
