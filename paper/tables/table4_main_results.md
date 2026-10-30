# Table 4: Comprehensive Empirical Recognition Performance Across All Benchmarks

| Benchmark Category | Dataset / Test Setting | Modality / Condition | Sample ($n$) | V1 CER (%) | V1 WER (%) | V2 CER (%) | V2 WER (%) | Tess 5 CER (%) | Tess 5 WER (%) | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Synthetic Mixed** | `word_synth` | Word (Clean) | 2,000 | 0.50% | 3.95% | **0.11%** | **0.90%** | 0.70% | 4.90% | `results/word_v2_synth/metrics.json` |
| **Synthetic Mixed** | `line_synth` | Line (Clean) | 1,000 | 1.64% | 11.64% | **0.30%** | **2.13%** | 0.44% | 2.94% | `results/line_v2_synth/metrics.json` |
| **Synthetic Monofont** | `diagw_tahoma` (New Fam) | Word (Clean) | 1,000 | **0.08%** | 0.70% | 0.09% | 0.70% | 0.50% | 3.20% | `results/diagw_tahoma_v2/metrics.json` |
| **Synthetic Monofont** | `diag_tahoma` (New Fam) | Line (Clean) | 500 | **0.24%** | 1.74% | 0.26% | 1.84% | 0.22% | 1.59% | `results/diag_tahoma_v2/metrics.json` |
| **Synthetic Monofont** | `diagw_consola` (Leakage)| Word (Clean) | 1,000 | 1.02% | 8.10% | **0.07%** | **0.60%** | 0.86% | 6.10% | `results/diagw_consola_v2/metrics.json` |
| **Synthetic Monofont** | `diag_consola` (Leakage) | Line (Clean) | 500 | 3.48% | 23.76% | **0.27%** | **1.93%** | 0.47% | 3.12% | `results/diag_consola_v2/metrics.json` |
| **Distortion Ladder** | L0 (Zero Distortion) | Word | 346 | 7.71% | 27.75% | **0.20%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:7` |
| **Distortion Ladder** | L0 (Zero Distortion) | Line | 92 | 1.87% | 11.14% | **0.11%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:7` |
| **Distortion Ladder** | L1 (+ Illumination) | Word | 346 | 8.85% | 32.08% | **0.20%** | NOT AVAIL | 0.45% | NOT AVAIL | `results/sim_ladder/summary.md:9` |
| **Distortion Ladder** | L1 (+ Illumination) | Line | 92 | 4.10% | 22.56% | **0.11%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:9` |
| **Distortion Ladder** | L2 (+ Blur & Noise) | Word | 346 | 25.00% | 57.80% | **0.41%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:10` |
| **Distortion Ladder** | L2 (+ Blur & Noise) | Line | 92 | 13.84% | 37.60% | **0.07%** | NOT AVAIL | 0.29% | NOT AVAIL | `results/sim_ladder/summary.md:10` |
| **Distortion Ladder** | L3 (+ Downscale/JPEG)| Word | 346 | 31.36% | 67.92% | **0.53%** | NOT AVAIL | 0.41% | NOT AVAIL | `results/sim_ladder/summary.md:11` |
| **Distortion Ladder** | L3 (+ Downscale/JPEG)| Line | 92 | 19.14% | 42.62% | **0.18%** | NOT AVAIL | 0.22% | NOT AVAIL | `results/sim_ladder/summary.md:11` |
| **Distortion Ladder** | L4 (Full Phone Sim) | Word | 346 | 35.77% | 75.72% | **1.35%** | NOT AVAIL | 0.37% | NOT AVAIL | `results/sim_ladder/summary.md:12` |
| **Distortion Ladder** | L4 (Full Phone Sim) | Line | 92 | 12.96% | 36.21% | **0.11%** | NOT AVAIL | 0.33% | NOT AVAIL | `results/sim_ladder/summary.md:12` |
| **Distortion Ladder** | L4n (+ Contrast Stretch)| Word | 346 | 24.43% | 67.92% | **1.14%** | NOT AVAIL | 0.37% | NOT AVAIL | `results/sim_ladder/summary.md:13` |
| **Distortion Ladder** | L4n (+ Contrast Stretch)| Line | 92 | 7.43% | 25.91% | **0.07%** | NOT AVAIL | 0.33% | NOT AVAIL | `results/sim_ladder/summary.md:13` |
| **Clean Unseen Aggregate**| L0 Clean Unseen | Word | 699 | 10.66% | 38.63% | 1.62% | 8.01% | **1.28%** | **6.29%** | `results/v2_audit3/summary.md:110` |
| **Clean Unseen Aggregate**| L0 Clean Unseen | Line | 180 | 5.84% | 31.38% | 0.75% | 7.67% | **0.58%** | **3.21%** | `results/v2_audit3/summary.md:111` |
| **Clean Unseen Aggregate**| L4 Full Sim | Word | 699 | 44.34% | 77.83% | 1.85% | 9.87% | **1.22%** | **5.72%** | `results/v2_audit3/summary.md:112` |
| **Clean Unseen Aggregate**| L4 Full Sim | Line | 180 | 21.98% | 56.90% | 1.39% | 13.25% | **1.22%** | **3.49%** | `results/v2_audit3/summary.md:113` |
| **Real Photo 1 (TNR)** | Full Line | Line | 16 | 77.05% | 100.00%| 53.61% | 92.63% | **1.32%** | **5.07%** | `results/real_fix8/final_tables.md:9` |
| **Real Photo 1 (TNR)** | Rule Chunks (Trim+4px)| Line | 16 | 18.38% | 73.27% | **0.97%** | 5.53% | 1.32% | **5.07%** | `results/real_fix8/final_tables.md:13` |
| **Real Photo 2 (Calibri)** | Full Line | Line | 16 | 67.81% | 99.05% | 36.43% | 96.21% | **0.14%** | **1.42%** | `results/real_fix8/final_tables.md:15` |
| **Real Photo 2 (Calibri)** | Rule Chunks (Trim+4px)| Line | 16 | 9.77% | 55.45% | 0.55% | 3.79% | **0.14%** | **1.42%** | `results/real_fix8/final_tables.md:19` |
| **Real Photo 3 (TNR 23L)**| Full Line | Line | 23 | 7.39% | 39.86% | 0.20% | 1.45% | **0.00%** | **0.00%** | `results/photo3_test/report.md:12` |
| **Real Photo 3 (TNR 23L)**| Rule Chunks (Trim+4px)| Line | 23 | 4.63% | 32.61% | 0.10% | 0.72% | **0.00%** | **0.00%** | `results/photo3_test/report.md:11` |
| **Real Photo 4 (Arial 23L)**| Full Line | Line | 23 | 7.06% | 42.34% | 0.10% | 0.73% | **0.00%** | **0.00%** | `results/photo4_sans/report.md:12` |
| **Real Photo 4 (Arial 23L)**| Rule Chunks (Trim+4px)| Line | 23 | 5.67% | 38.69% | 0.60% | 3.65% | 0.60% | 3.65% | `results/photo4_sans/report.md:11` |
| **Real Photo 5 (IMG_5130)**| Full Line (Cond A) | Line | 33 | 19.80% | 69.35% | 3.87% | 21.13% | 8.42% (psm7)| 15.18% | `results/real_fix9/metrics.json:4-35` |
| **Real Photo 5 (IMG_5130)**| Full Line psm 13 (Cond A)| Line | 33 | — | — | — | — | **0.09%** | **0.60%** | `results/real_fix9/metrics.json:37` |
| **Real Photo 5 (IMG_5130)**| Rule Chunks (Cond A) | Line | 33 | 8.76% | 40.48% | **2.10%** | **12.80%** | 13.02% (psm7)| 19.35% | `results/real_fix9/metrics.json:48-78` |
| **Real Photo 5 (IMG_5130)**| Clean Subset (Cond A) | Line | 23 | 6.73% | 32.16% | 1.32% | 9.25% | **1.26%** (psm7)| **7.05%** | `results/real_fix9/metrics.json:268-299` |
| **Real Photo 5 (IMG_5130)**| Padded 20px (Cond B) | Line | 33 | 38.57% | 86.90% | 23.32% | 66.07% | **18.86%** (psm7)| **27.38%**| `results/real_fix9/metrics.json:149-169` |
| *Heuristic Pilot (Snoop)*| V2 + Heuristic Lexicon | Line | 33 | — | — | *1.80%* | NOT AVAIL | *0.09%* (psm13)| *0.60%* | `scratch/test_safe_corrector.py` |

Source: Compiled directly from metrics.json files across results directories (results/word_v2_synth, results/line_v2_synth, results/diag_tahoma_v2, results/diag_consola_v2, results/sim_ladder, results/v2_audit3, results/real_fix8, results/real_fix9).
