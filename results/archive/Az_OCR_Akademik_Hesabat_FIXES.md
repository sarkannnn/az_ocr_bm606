# Düzəlişlər və Fakt Yoxlaması Hesabatı (Verification & Audit Fixes)

Bu sənəd `results/Az_OCR_Akademik_Hesabat.docx` və `results/academic_paper_comprehensive.md` sənədlərindəki bütün rəqəmlərin, cədvəllərin və iddiaların ilkin mənbə faylları (`results/v2_audit3/summary.md`, `results/v2_audit4/summary.md`, `results/v2_audit/summary.md`, `results/sim_ladder/summary.md`, `results/real_test/evaluation.md`, `results/real_test2/evaluation.md`, `train_v2.py`, `train.py`, `generate_data.py`, `prepare_corpus.py`) ilə tutuşdurulması nəticəsində tərtib edilmişdir.

---

## 1. Təsdiqlənməmiş / Uydurma İddiaların Siyahısı (Items Marked "UNVERIFIED")

Aşağıdakı bəndlər mənbə fayllarında heç bir ölçmə və ya loqa əsaslanmadığı üçün **UNVERIFIED (Təsdiqlənməmiş / Silinməli)** kimi işarələnmişdir:

1. **Sec 6.2:** `"ı -> i V1 14.2% -> V2 0.8%"` və `"ə -> e V1 8.7% -> V2 0.4%"`
   - *Status:* **UNVERIFIED (Mənbəsi yoxdur)**. Audit 3 və Audit 4-də V2-nin `ə/e` xətası 0.00% (0/394), `ı/i` söz xətası isə 6.52%–7.25% (9–10 / 138) olaraq ölçülmüşdür. V1 üçün xüsusi hərf cədvəlləri heç vaxt bu faizləri verməmişdir.
2. **Table 3:** `"Tahoma V1 CER 0.32%"` və `"Consolas Tesseract CER 1.15%"`
   - *Status:* **UNVERIFIED (Səhv köçürülüb)**. `results/v2_evaluation_report.md` və `results/*/metrics.json` fayllarında Tahoma V1 CER söz üçün **0.08%**, sətir üçün **0.24%**-dir; Consolas Tesseract CER söz üçün **0.86%**, sətir üçün **0.47%**-dir.
3. **Reference 5:** `"Memmedov, S., et al. (2026). Azerbaijani Document Digitization Benchmark and CRNN Optical Character Recognition. BM 606 Research Series, 1-45."`
   - *Status:* **UNVERIFIED (Uydurma biblioqrafik mənbə)**. Belə bir rəsmi məqalə və ya jurnal mövcud deyil, dərhal silinməlidir.
4. **Abstract & Sec 5:** `"V2 outperforms Tesseract 5 on real document phone photos"`
   - *Status:* **UNVERIFIED / YANILDIRICI**. Tesseract hər iki real foto sınağında sətir səviyyəsində V2-dən irəlidədir (Foto 1: 21.92% vs 28.87%; Foto 2: 0.72% vs 5.02%). Söz səviyyəsindəki 47.37% vs 52.63% fərqi cəmi 30 sözlük və kəsim koordinatları qismən sürüşmüş (qonşu sözlərə toxunan) kiçik bir sınaqdan qaynaqlanır. WER üzrə Foto 2-də Tesseract (86.67%) V2-dən (90.00%) üstündür.
5. **Sec 7:** `"LM inteqrasiyası sətir CER-ni 5%-dən 1%-in altına endirəcəkdir"`
   - *Status:* **UNVERIFIED (Ölçülməmiş fərziyyə)**. Mətndə "endirə bilər (may reduce)" ehtimal formasına çevrilməlidir.

---

## 2. Detallı Dəyişikliklər Siyahısı (Change List)

### Qrup A: Audit Nəticələrinə Zidd Olan İddialar

| № | Bölmə / Yer | Əvvəlki İddia (Old Value) | Düzəliş / Faktiki Dəyər (New Value) | İlkin Mənbə Faylı |
|---|---|---|---|---|
| **1** | **Sec 6.2** | `ı->i V1 14.2% -> V2 0.8%`, `ə->e V1 8.7% -> V2 0.4%` | **Söz səviyyəsində:** `ə/e` n=394: V2 L0=0.00%, L4=0.00% (Tess L0=2.03%, L4=1.78%). `ı/i` n=138: V2 L0=6.52% (9/138), L4=7.25% (10/138) (Tess L0=0.72%, L4=0.00%). **Sətir səviyyəsində:** `ı/i` V2 L0=0.72%, L4=2.17%. Dotless-i xətalarının 91.5%-i 3 dar şriftdə (`Moba_Fanta`: 53, `Moba_Font`: 30, `SansSerifCollection`: 14) cəmləşib. | `results/v2_audit3/summary.md:121-143`<br>`results/v2_audit4/summary.md:139-174`<br>`results/v2_audit4/dotless_i_errors.csv` |
| **2** | **Sec 6.1** | "True generalization proven using Cambria, Microsoft Sans Serif, Moba_Fira only" | Həqiqi ümumiləşdirmə Audit 4-ün tam A/B/C qrupları ilə təhlil edilməlidir. **Qrup C (Yeni 7 font ailəsi):** L4 Söz səviyyəsində V2 CER 2.36% [1.23, 3.95], Tesseract CER 0.81% [0.37, 1.31], fərq: **+1.55% [+0.59, +2.94]** (Tesseract V2-dən əhəmiyyətli dərəcədə yaxşıdır!). `Moba_Font` (3.52%) və `Moba_Fanta` (6.04%) V2 üçün çətin şriftlərdir. `Moba_Fira` isə monospacedir və təlimdə `CascadiaCode`/`consolab` monospace şriftləri olmuşdur. | `results/v2_audit4/summary.md:53-135` |
| **3** | **Sec 2.1** | "Text leakage fully prevented, split at sentence and word level" | Cümlə/sətir səviyyəsində sızma yoxdur: **0 / 294** test sətri təlimdə iştirak edib (0.00%). Lakin lüğət səviyyəsində test sözlərinin **22.40%-i (276 / 1232 söz)** təbii olaraq təlim korpusunun lüğətində mövcuddur. Korpus mənbələri: `wikimedia/wikipedia` (az) 20231101.az + xəbərlər; `prepare_corpus.py` daxilində `balance_special` ilə diakritikli sözlərin payı $\ge 50\%$ saxlanılıb. | `results/v2_audit3/summary.md:147-151`<br>`prepare_corpus.py:30-78` |
| **4** | **Table 1** | V2 rotation: $\pm 3^\circ$, bəsit qradiyent | `train.py` (V1): Fırlatma $\pm 5^\circ$ ($p=0.5$), Gaussian blur $k \in \{3, 5\}$ ($p=0.3$), Duz-İstiot 0.5% ($p=0.3$), Perspektiv ($p=0.3$).<br>`train_v2.py` (V2): Fırlatma $\pm 5^\circ$ ($p=0.5$), Perspektiv ($p=0.3$), İşıq qradiyenti ($p=0.5, 0.7-1.0$), Soft dark band ($p=0.5$), Parlaqlıq/Kontrast ($p=0.6, \alpha: 0.6-1.1, \beta: -40..20$), Gaussian blur ($p=0.6, \sigma: 0.5-2.0$), Motion blur ($p=0.15, k: 3-7$), Sensor küyü ($p=0.6, \sigma: 3-12$), Duz-istiot ($p=0.2$), Downscale ($p=0.4, 0.5-0.9\times$), JPEG sıxılması ($p=0.4, Q: 40-90$), Kontrast dartılması ($p=0.5$). | `train.py:21-40`<br>`train_v2.py:26-80`<br>`results/audit.md:20` |
| **5** | **Table 3** | "Independent synthetic tests", Tahoma V1 0.32%, Consolas Tess 1.15% | Sintetik test dəstləri V1-in 8 təlim şriftindən kopyalandığı üçün **SEEN (Görünmüş)** şriftlərdir. Faktiki göstəricilər:<br>• `word_synth`: n=2000, V1 0.50%/3.95%, V2 0.11%/0.90%, Tess 0.70%/4.90%<br>• `line_synth`: n=1000, V1 1.64%/11.64%, V2 0.30%/2.13%, Tess 0.44%/2.94%<br>• `diagw_tahoma`: n=1000, V1 **0.08%**/0.70%, V2 **0.09%**/0.70%, Tess **0.50%**/3.20%<br>• `diagw_consola`: n=1000, V1 1.02%/8.10%, V2 0.07%/0.60%, Tess **0.86%**/6.10%<br>• `diag_tahoma`: n=500, V1 0.24%/1.74%, V2 0.26%/1.84%, Tess 0.22%/1.59%<br>• `diag_consola`: n=500, V1 3.48%/23.76%, V2 0.27%/1.93%, Tess 0.47%/3.12% | `results/v2_evaluation_report.md:26-34`<br>`results/audit.md:19` |
| **6** | **Table 2** | Val CER 0.24% / 0.28% | Bu val CER göstəriciləri **yalnız 8 görülmüş şrift** üzərində ölçülmüşdür (Best epoch bu 8 şriftin itkisinə görə seçilmişdir). Checkpoint ölçüsü: diskdə 98.7 MB (model çəkisi + Adam vəziyyətləri daxil), xalis model çəkisi 32.9 MB. Təlim vaxtları: Word V2: 212.3 dəq (Epox 13), Line V2: 280.4 dəq (Epox 10). | `results/v2_audit4/summary.md:288-293`<br>`results/v2_prep.md:54-60`<br>`results/v2_evaluation_report.md:13-16` |
| **7** | **Table 4** | "7-Pilləli Nərdivan", Line V1 CER 0.00% / 0.00% / 0.44% | **6 pillədir (L0, L1, L2, L3, L4, L4n)**. Şriftlər: `cambria, micross, pala, CascadiaMono` (2 seen/near, 2 new).<br>Faktiki rəqəmlər (`sim_ladder/summary.md` və `v2_audit/summary.md`):<br>• L0: Word V1 7.71%, V2 0.20%, Tess 0.41%; Line V1 **1.87%**, V2 0.11%, Tess 0.22%<br>• L1: Word V1 8.85%, V2 0.20%, Tess 0.45%; Line V1 **4.10%**, V2 0.11%, Tess 0.22%<br>• L2: Word V1 25.00%, V2 0.41%, Tess 0.41%; Line V1 **13.84%**, V2 0.07%, Tess 0.29%<br>• L3: Word V1 31.36%, V2 0.53%, Tess 0.41%; Line V1 **19.14%**, V2 0.18%, Tess 0.22%<br>• L4: Word V1 35.77%, V2 1.35%, Tess 0.37%; Line V1 **12.96%**, V2 0.11%, Tess 0.33%<br>• L4n: Word V1 24.43%, V2 1.14%, Tess 0.37%; Line V1 **7.43%**, V2 0.07%, Tess 0.33% | `results/v2_audit/summary.md:64-72`<br>`results/sim_ladder/summary.md:9-14`<br>`results/audit.md:19` |

---

### Qrup B: Real Sənəd Sınaqları və Tesseract Müqayisəsi

| № | Bölmə / Yer | Əvvəlki İddia (Old Value) | Düzəliş / Faktiki Dəyər (New Value) | İlkin Mənbə Faylı |
|---|---|---|---|---|
| **8** | **Abstract & Sec 5** | "V2 outperforms Tesseract 5 on phone photos" | Sətir səviyyəsində **Tesseract hər iki sınaqda V2-dən irəlidədir**: Foto 1-də Tesseract CER 21.92% (V2: 28.87%), Foto 2-də Tesseract CER **0.72%** (V2: 5.02%). Word səviyyəsində Foto 2 WER göstəricisi: V2 90.00% vs Tesseract **86.67%**. V2-nin söz CER üzrə Foto 1-də 47.37% vs 52.63% və Foto 2-də 46.71% vs 50.99% nəticəsi cəmi 30 sözlük kiçik kəsimlərə əsaslanır və **ilkin (preliminary)** xarakter daşıyır. | `results/real_test/evaluation.md:10-14`<br>`results/real_test2/evaluation.md:8-16` |
| **9** | **Abstract** | Tesseract sətir göstəricisi (0.72%) qeyd edilməyib | Xülasəyə Tesseract-ın sətir CER göstəricisi (**0.72%**) əlavə edilməlidir: "V1 48.15% $\rightarrow$ V2 5.02% (Tesseract 5: 0.72%)". | `results/real_test2/evaluation.md:12` |
| **10** | **Sec 5.2 / Table 5** | Nümunə sayı təfərrüatları çatışmır | Foto 2: 1 tək sənəd fotosu (`media_1793194879421.jpg`, 768x1024), 21 sətir kəsimi (837 simvol, 89 söz) və 30 söz kəsimi (296 simvol, 30 söz). Bütün kəsimlər əllə təsdiqlənmişdir. | `results/real_test2/results.json:1-12` |
| **11** | **Table 6** | Yalnız V2-nin 100% dəqiq çıxardığı uğurlu nümunələr göstərilir | Table 5-də 21 sətirdən yalnız 7-sinin tam qüsursuz olduğu göstərildiyi üçün, Table 6-ya **V2-nin xarakterik xətaları da əlavə olunmalıdır**: sonluq `-ı` və ya `..` hallusinasiyaları (məs. `elektron mühitdə idarə olunması. -ı`, `hazırlanması..`), kənar simvol itkisi (`TƏQDİMAT` $\rightarrow$ `İMAT ı`), daxili hərf səhvləri (`xldmət`, `təkmlləşdirilməsi`). | `results/real_test/evaluation.md:24,60`<br>`results/real_test2/results.json:18,40` |
| **12** | **Sec 1.3 / Sec 5** | "İki fərqli sənəd sınağı" ifadəsi | Hər iki foto **eyni bir rəsmi təqdimat sənədinə** aiddir (biri bucaq altında və kölgəli, digəri düz çəkiliş). Bu fakt dürüst şəkildə qeyd olunmalı, "iki fərqli sənəd" ifadəsi silinməli və 20-30 fərqli sənəddən ibarət 100+ sətirlik geniş test bazasının hələ formalaşdırılmadığı məhdudiyyət kimi göstərilməlidir. | `media_1793193977057.jpg`<br>`media_1793194879421.jpg` |
| **13** | **Sec 1 & Sec 1.2** | "Tesseract mobil fotolarda tamamilə iflasa uğrayır" və "Tesseract hərflərin fərdi kəsilməsinə söykənir" | 1. Tesseract 5 düz sənəd fotosunda 0.72% sətir CER göstərdiyi üçün onun "tam iflasa uğraması" iddiası təkzib olunur və "kontekstsiz kəsilmiş sözlərdə zəifləyir" kimi dəqiqləşdirilməlidir.<br>2. Tesseract 4 və 5 ənənəvi hərfi kəsmə yox, **LSTM əsaslı sətir tanıma arxitekturasından** istifadə edir. | Smith (2007)<br>`results/real_test2/evaluation.md:12` |

---

### Qrup C: Əlavə Edilməli Olan Vacib Məzmunlar

| № | Əlavə Edilən Mövzu | Faktiki Dəyərlər və Dəlillər | Mənbə Faylı |
|---|---|---|---|
| **14** | **Bootstrap İnam İntervalları (95% CI)** | Əsas cədvəllərə Audit 3/4-ün klaster bootstrap intervalları əlavə edilməlidir. Qrup C L4 Line V2 CER: **1.28% [0.16%, 2.75%]**, Tesseract: **1.46% [0.37%, 3.72%]**, Fərq: -0.17% [-2.60%, +1.53%] (0 intervala daxildir, yəni fərq statistik cəhətdən əhəmiyyətsizdir). | `results/v2_audit4/summary.md:118-135` |
| **15** | **Boşluq (Space) Xətalarının Təhlili** | Sətir səviyyəsində V2-nin söz xətalarının (WER) **65.3%–79.3%-i xalis boşluq xətalarından** ibarətdir (məs. `t icarət`, `v irus`, `d in`, `ist isna`). Boşluqlar nəzərə alınmadıqda V2-nin L4 sətir WER-i **12.87%-dən 4.55%-ə** enir (Tesseract: 3.81%). | `results/v2_audit4/summary.md:200-213` |
| **16** | **Metodoloji Məhdudiyyətlər (Limitations)** | 1. **Əlyazma/Cursive (`Inkfree`):** V2 L0-da 30.69% CER göstərir, Tesseract isə 14.20% (V2 əlyazmada Tesseract-dan 2 dəfədən çox zəifdir).<br>2. **Dotless-i Konsentrasiyası:** Dotless-i xətalarının 91.5%-i serifləri olmayan 3 fontda (`Moba_Fanta`, `Moba_Font`, `SansSerifCollection`) cəmləşib.<br>3. **Kəsim Kəsilməsi (Clipping):** Kəsilmiş sözlərin 2.3%-də (24 / 1049 crop) mürəkkəb itkisi var (`SansSerifCollection` 'və' sözü 11.79% ink loss).<br>4. **Kiçik Hərf Nümunəsi:** Nadir simvolların test sayı azdır (`ç/c`: n=23, `ş/s`: n=56, `ğ/g`: n=20, `ö/o`: n=32). | `results/v2_audit4/summary.md:75-76, 163-174, 263-286`<br>`results/v2_audit3/summary.md:121-131` |

---

### Qrup D: Ədəbiyyat, Müəlliflik və Dəqiqləşdirmələr

| № | Maddə | Əvvəlki Mətn | Düzəldilmiş Mətn |
|---|---|---|---|
| **17** | **Ref 5** | `Memmedov, S., et al. (2026). Azerbaijani Document Digitization Benchmark...` | **TAMAMİLƏ SİLİNDİ** (Uydurma biblioqrafik qeyd). |
| **18** | **Sec 2.2** | "TRDG mühərrikindən istifadə edilmişdir" | V1 generasiyasında TRDG (`generate_data.py:40`) istifadə edilmişdir. V2-yə əlavə olunan 15,000 söz və 6,000 sətir isə `fonts_new/` şriftləri ilə birbaşa PIL (`ImageDraw`) vasitəsilə generasiya edilmişdir. |
| **19** | **Titul** | "Antigravity AI & Tədqiqat Qrupu" / "Elmi Nəşrə Hazır / IEEE Standard Format" | **"[Müəllif Adı] / BM 606 Tədqiqat Qrupu"** və **"İlkin Tədqiqat və Texniki Audit Hesabatı (Internal Technical & Audit Report)"**. |
| **20** | **Sec 7** | "LM inteqrasiyası CER-i 5%-dən 1%-in altına endirəcəkdir" | "LM inteqrasiyası CER-i 5%-dən 1%-in altına **endirə bilər (may reduce)**" olaraq dəyişdirildi. |
| **21** | **Sec 2.3** | "SansSerifCollection üzrə xəta 84.32%-dən 0.76%-ə enmişdir" (model uğuru kimi) | Bu nəticə modelin deyil, `make_sim_ladder.py` skriptindəki **bounding-box həndəsi ölçmə qüsurunun düzəldilməsinin** nəticəsidir kimi qeyd edildi. |

---

Bu dəyişikliklər siyahısı tam faktiki fayllara əsaslanır və heç bir təxmini rəqəm ehtiva etmir.
