# Table 7: Mutually Exclusive Error Taxonomy Breakdown (Photo 5, IMG_5130)

| Error Category | Scope / Definition | Condition A: V2 Line | Condition A: Tess psm 7 | Clean 23: V2 Line | Clean 23: Tess psm 7 | Impact & Root Cause Analysis | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1. Leading Edge Insertions** | Insertions strictly before 1st aligned char | **0** | **3** | **0** | **3** | Margin artifacts; V2 has 0 leading edge noise | `results/real_fix9/report.md:75` |
| **2. Trailing Edge Insertions**| Insertions strictly after last aligned char | **0** | **0** | **0** | **0** | Clean baseline termination | `results/real_fix9/report.md:76` |
| **3. Internal Spacing Errors** | Space insertions/deletions inside body | **3** | **44** | **2** | **1** | Tess has 44 space errors from 9 empty chunk runs | `results/real_fix9/report.md:77` |
| **4. Internal 0 / O Errors** | Digit `0` vs letter `O / o` confusion | **0** | **0** | **0** | **0** | Zero confusion in literary text | `results/real_fix9/report.md:78` |
| **5. Internal Punctuation** | Punctuation edits (e.g. `.` vs `:`) | **12** | **20** | **6** | **7** | Colon/period ambiguities on faint print dots | `results/real_fix9/report.md:79` |
| **6. Internal Letter Errors** | True glyph substitutions and deletions | **34** | **236** | **13** | **9** | Tess 236 includes complete line deletions | `results/real_fix9/report.md:80` |
| **MUTUALLY EXCLUSIVE SUM** | $\sum \text{Categories}$ | **49** | **303** | **21** | **20** | **100% Equal to Levenshtein Sum** | `results/real_fix9/report.md:81` |
| **Levenshtein Char Distance** | Independent edit distance check | **49** | **303** | **21** | **20** | Sum verification passed | `results/real_fix9/metrics.json:19, 75, 240, 294` |
| **Total Ground Truth Chars** | Total character pool | 2,328 | 2,328 | 1,591 | 1,591 | 33 lines (all) vs 23 lines (clean subset) | `results/real_fix9/metrics.json:20, 241` |
| **Final Evaluated CER (%)** | Total Errors / Total Chars | **2.10%** | **13.02%** | **1.32%** | **1.26%** | V2 ~= Tess when empty outputs excluded | `results/real_fix9/metrics.json:59, 70, 279, 290` |

Source: results/real_fix9/error_taxonomy.json, results/real_fix9/report.md:69-82, results/real_fix9/metrics.json
