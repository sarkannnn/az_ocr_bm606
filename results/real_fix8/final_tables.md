# Real Fix 8: Comprehensive Evaluation and Regression Resolution Report

All results, statistics, and tables in this report were generated directly from `real_fix8_metrics.json`, `crop_geometry.json`, `error_taxonomy.json`, `photo2_line_diagnostics.csv`, and `removed_tall_components.json`.

### Evaluation Table: RAW (Direct Character Levenshtein)

| Photo | Condition | Model | CER (%) | WER (%) | Exact Lines | SpErr (char) | $N$ Lines | $N$ Chars | $\Delta$ (V2 - Tess) CER [95% CI] | $\Delta$ (V2 - Tess) WER [95% CI] |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| Photo 1 (TNR) | Full Line | V1 Line | 77.05% | 100.00% | 0 / 16 | — | 16 | 1442 | — | — |
| Photo 1 (TNR) | Full Line | **V2 Line** | **53.61%** | **92.63%** | **1 / 16** | **61** | 16 | 1442 | **+52.29% [+46.11%, +57.89%]** | **+87.56% [+76.77%, +94.23%]** |
| Photo 1 (TNR) | Full Line | Tesseract | 1.32% | 5.07% | 9 / 16 | 0 | 16 | 1442 | baseline | baseline |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | V1 Line | 18.38% | 73.27% | 0 / 16 | — | 16 | 1442 | — | — |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | **V2 Line** | **0.97%** | **5.53%** | **7 / 16** | **1** | 16 | 1442 | **-0.35% [-1.52%, +0.50%]** | **+0.46% [-3.69%, +4.02%]** |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | Tesseract | 1.32% | 5.07% | 8 / 16 | 0 | 16 | 1442 | baseline | baseline |
| Photo 2 (Cal) | Full Line | V1 Line | 67.81% | 99.05% | 0 / 16 | — | 16 | 1463 | — | — |
| Photo 2 (Cal) | Full Line | **V2 Line** | **36.43%** | **96.21%** | **1 / 16** | **70** | 16 | 1463 | **+36.30% [+32.29%, +39.57%]** | **+94.79% [+86.50%, +100.48%]** |
| Photo 2 (Cal) | Full Line | Tesseract | 0.14% | 1.42% | 14 / 16 | 1 | 16 | 1463 | baseline | baseline |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | V1 Line | 9.77% | 55.45% | 0 / 16 | — | 16 | 1463 | — | — |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | **V2 Line** | **0.55%** | **3.79%** | **10 / 16** | **1** | 16 | 1463 | **+0.41% [+0.07%, +0.94%]** | **+2.37% [+0.48%, +5.05%]** |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | Tesseract | 0.14% | 1.42% | 14 / 16 | 1 | 16 | 1463 | baseline | baseline |

### Evaluation Table: NORMALIZED DASHES (—, –, − mapped to -)

| Photo | Condition | Model | CER (%) | WER (%) | Exact Lines | SpErr (char) | $N$ Lines | $N$ Chars | $\Delta$ (V2 - Tess) CER [95% CI] | $\Delta$ (V2 - Tess) WER [95% CI] |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| Photo 1 (TNR) | Full Line | V1 Line | 77.05% | 100.00% | 0 / 16 | — | 16 | 1442 | — | — |
| Photo 1 (TNR) | Full Line | **V2 Line** | **53.33%** | **92.63%** | **1 / 16** | **59** | 16 | 1442 | **+52.01% [+45.87%, +57.52%]** | **+87.56% [+76.77%, +94.23%]** |
| Photo 1 (TNR) | Full Line | Tesseract | 1.32% | 5.07% | 9 / 16 | 0 | 16 | 1442 | baseline | baseline |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | V1 Line | 18.38% | 73.27% | 0 / 16 | — | 16 | 1442 | — | — |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | **V2 Line** | **0.69%** | **4.15%** | **9 / 16** | **1** | 16 | 1442 | **-0.62% [-1.78%, +0.27%]** | **-0.92% [-4.96%, +2.60%]** |
| Photo 1 (TNR) | **Rule Chunks (Trimmed)** | Tesseract | 1.32% | 5.07% | 8 / 16 | 0 | 16 | 1442 | baseline | baseline |
| Photo 2 (Cal) | Full Line | V1 Line | 67.74% | 99.05% | 0 / 16 | — | 16 | 1463 | — | — |
| Photo 2 (Cal) | Full Line | **V2 Line** | **36.30%** | **96.21%** | **1 / 16** | **70** | 16 | 1463 | **+36.16% [+32.22%, +39.31%]** | **+94.79% [+86.50%, +100.48%]** |
| Photo 2 (Cal) | Full Line | Tesseract | 0.14% | 1.42% | 14 / 16 | 1 | 16 | 1463 | baseline | baseline |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | V1 Line | 9.77% | 55.45% | 0 / 16 | — | 16 | 1463 | — | — |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | **V2 Line** | **0.41%** | **2.84%** | **12 / 16** | **1** | 16 | 1463 | **+0.27% [+0.00%, +0.80%]** | **+1.42% [+0.00%, +3.89%]** |
| Photo 2 (Cal) | **Rule Chunks (Trimmed)** | Tesseract | 0.14% | 1.42% | 14 / 16 | 1 | 16 | 1463 | baseline | baseline |

### Photo 2 Line-by-Line Diagnostics (Chunk Widths, Raw Outputs, Joining & Match)

| Line | $n$ | Widths Before Trim | Widths After Trim | Raw Chunk Outputs (V2) | Join Str | Joined V2 vs Reference | V2 Match | Tess Match |
|:---:|:---:|:---|:---|:---|:---:|:---|:---:|:---:|
| L00 | 2 | [1167, 993] | [1085, 913] | Gəncə çayının kənarında, qədim ağacların kölgəsinə `|` sığınmış balaca bir dulusçu emalatxanası var | ' ' | `Gəncə çayının kənarında, qədim ağacların kölgəsinə sığınmış balaca bir dulusçu emalatxanası var` | **EXACT** | **EXACT** |
| L01 | 3 | [516, 1153, 491] | [438, 1140, 313] | idi. Burada illərdir gil `|` ilə nəfəs alan, gilə can verən Qoca Səməndər yaşayırdı. `|` Səməndər usta | ' ' | `idi. Burada illərdir gil ilə nəfəs alan, gilə can verən Qoca Səməndər yaşayırdı. Səməndər usta` | **EXACT** | **EXACT** |
| L02 | 3 | [573, 904, 683] | [500, 895, 491] | əllərini palçığın tarixinə `|` batıranda sanki zaman dayanırdı; onun hər `|` bir hərəkətində əsrlərin | ' ' | V2: `əllərini palçığın tarixinə batıranda sanki zaman dayanırdı; onun hər bir hərəkətində əsrlərin`<br>Ref: `əllərini palçığın tarixinə batıranda sanki zaman dayanırdı, onun hər bir hərəkətində əsrlərin` | DIFF | **EXACT** |
| L03 | 2 | [1378, 782] | [1305, 671] | sənəti, nəsildən-nəslə keçən gizli bir məharət yaşatdırdığı hiss `|` olunurdu. Divarlardan asılmış gil | ' ' | `sənəti, nəsildən-nəslə keçən gizli bir məharət yaşatdırdığı hiss olunurdu. Divarlardan asılmış gil` | **EXACT** | **EXACT** |
| L04 | 2 | [1412, 748] | [1337, 617] | kasalar, bəzəkli küplər, nəqşli boşqablar otağa xüsusi bir torpaq `|` qoxusu və qədimlik gətirirdi. | ' ' | `kasalar, bəzəkli küplər, nəqşli boşqablar otağa xüsusi bir torpaq qoxusu və qədimlik gətirirdi.` | **EXACT** | **EXACT** |
| L05 | 2 | [1413, 747] | [1354, 649] | Səməndər usta çəliyini kənara qoyub ağır-ağar dolabın arxasına `|` keçdi, qabarıq və çat-çat olmuş | ' ' | `Səməndər usta çəliyini kənara qoyub ağır-ağar dolabın arxasına keçdi, qabarıq və çat-çat olmuş` | **EXACT** | **EXACT** |
| L06 | 2 | [1381, 779] | [1319, 711] | barmaqları ilə hrlanan dairəvi taxtaya toxundu: Bu taxta onun `|` ömrünün yekunu, sonuncu böyük | ' ' | V2: `barmaqları ilə hrlanan dairəvi taxtaya toxundu: Bu taxta onun ömrünün yekunu, sonuncu böyük`<br>Ref: `barmaqları ilə fırlanan dairəvi taxtaya toxundu. Bu taxta onun ömrünün yekunu, sonuncu böyük` | DIFF | **EXACT** |
| L07 | 2 | [1372, 788] | [1315, 737] | idi.Elə həmin an emalatxananın qapısı ehtiyatla açıldı və içəri `|` şəhərdən yeni gəlmiş nəvəsi Taryel | ' ' | V2: `idi.Elə həmin an emalatxananın qapısı ehtiyatla açıldı və içəri şəhərdən yeni gəlmiş nəvəsi Taryel`<br>Ref: `idi. Elə həmin an emalatxananın qapısı ehtiyatla açıldı və içəri şəhərdən yeni gəlmiş nəvəsi Taryel` | DIFF | DIFF |
| L08 | 2 | [1446, 714] | [1403, 622] | girdi. Taryel ali məktəbi bitirib şəhərdə böyük imkanlar qazanmış `|` bir gənc idi, lakin hər dəfə bu | ' ' | `girdi. Taryel ali məktəbi bitirib şəhərdə böyük imkanlar qazanmış bir gənc idi, lakin hər dəfə bu` | **EXACT** | **EXACT** |
| L09 | 3 | [522, 1078, 560] | [477, 1072, 419] | kəndə gələndə özünü `|` yenidən uşaqlıq illərindəki kimi azad və rahat hiss `|` edirdi. O, babasının | ' ' | `kəndə gələndə özünü yenidən uşaqlıq illərindəki kimi azad və rahat hiss edirdi. O, babasının` | **EXACT** | **EXACT** |
| L10 | 3 | [867, 711, 582] | [824, 703, 411] | əyilmiş belini, yorğun, lakin qətiyyətli `|` baxışlarını görüb asta addımlarla `|` yaxınlaşdı, əlindəki | ' ' | `əyilmiş belini, yorğun, lakin qətiyyətli baxışlarını görüb asta addımlarla yaxınlaşdı, əlindəki` | **EXACT** | **EXACT** |
| L11 | 2 | [590, 600] | [560, 588] | çantanı bir kənara qoyub `|` babasının çiyninə toxundu. | ' ' | `çantanı bir kənara qoyub babasının çiyninə toxundu.` | **EXACT** | **EXACT** |
| L12 | 3 | [770, 853, 537] | [741, 849, 507] | - Baba, yenə saatlarla bu təkərin `|` arxasında oturmusan. Özünü yorursan, `|` gəl bu gün biraz dincəl, | ' ' | V2: `- Baba, yenə saatlarla bu təkərin arxasında oturmusan. Özünü yorursan, gəl bu gün biraz dincəl,`<br>Ref: `— Baba, yenə saatlarla bu təkərin arxasında oturmusan. Özünü yorursan, gəl bu gün biraz dincəl,` | DIFF | **EXACT** |
| L13 | 3 | [927, 622, 611] | [915, 613, 476] | Qoca Səməndər başını qaldırıb nəvəsinə `|` baxdı, dodaqlarında yüngül, `|` lakin dərin mənalı bir | ' ' | `Qoca Səməndər başını qaldırıb nəvəsinə baxdı, dodaqlarında yüngül, lakin dərin mənalı bir` | **EXACT** | **EXACT** |
| L14 | 3 | [498, 1077, 585] | [491, 1061, 502] | təbəssüm yarandı. O, `|` ovcundakı nəm gili yavaşca masaya qoyub dedi: `|` - Ay bala, insan o vaxt | ' ' | V2: `təbəssüm yarandı. O, ovcundakı nəm gili yavaşca masaya qoyub dedi: - Ay bala, insan o vaxt`<br>Ref: `təbəssüm yarandı. O, ovcundakı nəm gili yavaşca masaya qoyub dedi: — Ay bala, insan o vaxt` | DIFF | **EXACT** |
| L15 | 3 | [933, 552, 675] | [922, 543, 608] | qocalır ki, torpağa olan inamı ölür. Bu gil `|` sadəcə torpaq parçasımı `|` sanırsan? Xeyrl Hər küpün, | ' ' | V2: `qocalır ki, torpağa olan inamı ölür. Bu gil sadəcə torpaq parçasımı sanırsan? Xeyrl Hər küpün,`<br>Ref: `qocalır ki, torpağa olan inamı ölür. Bu gil sadəcə torpaq parçasımı sanırsan? Xeyr! Hər küpün,` | DIFF | DIFF |

### Photo 1 Lines 12-15 Shadow Stripe Analysis & Component Audit

- **Contact Sheet for Photo 1 Lines 12-15:** [contact_sheet_p1_l12_l15.png](file:///d:/laragon/www/az_ocr_bm606/results/real_fix8/contact_sheet_p1_l12_l15.png)
- **Status of Left Shadow Stripe:**
  - The faint gray shadow stripe (width 15–18 px) on the left edge has an average grayscale brightness of **163.0 to 167.0** (with column minimums mostly 148–156).
  - Because the established ink binarization threshold is $\text{gray} < 160$, **over 80% of the pixels in this stripe are lighter than the ink threshold**.
  - Therefore, the stripe does **not** form a solid, continuous connected component that spans $\ge 70\%$ of the chunk height.
  - Instead, the few pixels dipping below 160 fragment into small, isolated sub-pixel speck clusters.
  - All clusters with $\text{area} < 8$ px were completely removed by the robust trim rule.
  - In `removed_tall_components.json`, **exactly 0 components taller than 6 px were removed**, because the shadow stripe was lighter than ink rather than an opaque glyph, so no tall component was ever formed.

### Mutually Exclusive Error Taxonomy (V2 Line vs Tesseract on Trimmed Chunks)

| Model | Error Category | Photo 1 (RAW) | Photo 1 (Norm) | Photo 2 (RAW) | Photo 2 (Norm) | Definition & Scope |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **V2 Line** | **1. Leading Edge Insertions** | **0** | **0** | **0** | **0** | Insertions strictly before first aligned ref char |
| **V2 Line** | **2. Trailing Edge Insertions** | **1** | **1** | **0** | **0** | Insertions strictly after last aligned ref char |
| **V2 Line** | **3. Internal Spacing Errors** | **1** | **1** | **1** | **1** | Spacing edit operations strictly inside line body |
| **V2 Line** | **4. Internal 0 / O Errors** | **0** | **0** | **0** | **0** | Confusions between digit `0` and letter `O/o` |
| **V2 Line** | **5. Internal Punctuation** | **7** | **3** | **6** | **4** | Punctuation substitutions/edits inside line body |
| **V2 Line** | **6. Internal Letter Errors** | **5** | **5** | **1** | **1** | True glyph recognition errors |
| **V2 Line** | **SUM OF CATEGORIES** | **14** | **10** | **8** | **6** | $\sum \text{Categories}$ |
| **V2 Line** | **TOTAL CHAR ERRORS** | **14** | **10** | **8** | **6** | Total Levenshtein distance against reference |
| **V2 Line** | **SUM CHECK STATUS** | **PASSED** | **PASSED** | **PASSED** | **PASSED** | $100\%$ mutually exclusive |
|---|---|---|---|---|---|---|
| Tesseract 5 | **1. Leading Edge Insertions** | **1** | **1** | **0** | **0** | Insertions strictly before first aligned ref char |
| Tesseract 5 | **2. Trailing Edge Insertions** | **1** | **1** | **0** | **0** | Insertions strictly after last aligned ref char |
| Tesseract 5 | **3. Internal Spacing Errors** | **0** | **0** | **1** | **1** | Spacing edit operations strictly inside line body |
| Tesseract 5 | **4. Internal 0 / O Errors** | **0** | **0** | **0** | **0** | Confusions between digit `0` and letter `O/o` |
| Tesseract 5 | **5. Internal Punctuation** | **4** | **4** | **1** | **1** | Punctuation substitutions/edits inside line body |
| Tesseract 5 | **6. Internal Letter Errors** | **13** | **13** | **0** | **0** | True glyph recognition errors |
| Tesseract 5 | **SUM OF CATEGORIES** | **19** | **19** | **2** | **2** | $\sum \text{Categories}$ |
| Tesseract 5 | **TOTAL CHAR ERRORS** | **19** | **19** | **2** | **2** | Total Levenshtein distance against reference |
| Tesseract 5 | **SUM CHECK STATUS** | **PASSED** | **PASSED** | **PASSED** | **PASSED** | $100\%$ mutually exclusive |
|---|---|---|---|---|---|---|

### Pipeline Unification & Development Data Note

1. **Unified Joining and Splitting Rule:**
   - Both photos use the exact same inter-word projection splitting rule ($k_{\text{size}} = 25$ with 35th percentile ink threshold) so splits occur strictly at whitespace gaps.
   - Both photos use the exact same joining string (`' '`) across all models (V1, V2, Tesseract).
   - Both photos use the exact same robust CC trimming rule (specks $< 8$ px and border shadows $\ge 70\%$ height removed, trimmed to active ink $+ 4$ px).
2. **Methodological Note:**
   - Both Photo 1 and Photo 2 serve as development/pilot data. All generalizability conclusions will be tested on the upcoming benchmark of 20–30 new unseen document photos.