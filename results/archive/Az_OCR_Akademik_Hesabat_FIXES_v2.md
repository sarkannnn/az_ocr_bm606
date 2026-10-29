# Akademik Hesabat üçün Tam Fakt-Yoxlama və Dəqiqləşdirilmiş Düzəlişlər Sənədi (FIXES v3)

**Sənəd Kodu:** `BM606-DOCX-AUDIT-FIXES-V3`  
**Tarix:** 29 Oktyabr 2026  
**Status:** Yoxlanılmış və Mənbə Sətirləri ilə Təsdiqlənmiş Düzəlişlər Siyahısı (Orijinal `.docx` faylı dəyişdirilməmişdir).

---

## 1. Qalan və Təsdiqlənməmiş İddialar (Remaining UNVERIFIED Items)

1. **Sec 6.2 — Diakritik Faizlər:** `"ı -> i V1 14.2% -> V2 0.8%"` və `"ə -> e V1 8.7% -> V2 0.4%"`
   - *Status:* **UNVERIFIED (Uydurma rəqəm)**. Audit 3 və Audit 4-də V2-nin `ə/e` xətası **0.00%** ($n=394$), `ı/i` söz xətası isə **6.52%–7.25%** ($n=138$) ölçülmüşdür (`results/v2_audit3/summary.md:124-125`). V1 üçün bu xüsusi hərf matrisi heç vaxt ölçülməmişdir.
2. **Table 3 — Tahoma və Consolas Göstəriciləri:** `"Tahoma V1 CER 0.32%"` və `"Consolas Tesseract CER 1.15%"`
   - *Status:* **UNVERIFIED (Səhv köçürülüb)**. `results/v2_evaluation_report.md:30-33` faylında Tahoma V1 CER söz üçün **0.08%**, sətir üçün **0.24%**; Consolas Tesseract CER söz üçün **0.86%**, sətir üçün **0.47%**-dir. Həmçinin Tahoma üzrə V2 (0.09% söz / 0.26% sətir) V1-dən (0.08% / 0.24%) yaxşı deyil; "Tahoma 4x yaxşılaşdı" iddiası əsassızdır.
3. **Reference 5 — Biblioqrafiya:** `"Memmedov, S., et al. (2026). Azerbaijani Document Digitization Benchmark... BM 606 Research Series, 1-45."`
   - *Status:* **UNVERIFIED (Uydurma sitat)**. Repozitoriyada və ya elmi bazada belə bir nəşr yoxdur; tamamilə silinməlidir.
4. **Abstract & Sec 5 — Real Foto Üstünlüyü İddiası:** `"V2 outperforms Tesseract 5 on real document phone photos"`
   - *Status:* **UNVERIFIED / YANILDIRICI**. Sətir səviyyəsində Tesseract hər iki sınaqda V2-dən irəlidədir: Foto 1-də 21.92% vs 28.87% (`real_test/evaluation.md:12`); Foto 2-də **0.72% vs 5.02%** (düzəldilmiş `TƏQDİMAT` ilə: **0.24% vs 5.02%**). WER üzrə Foto 2-də Tesseract (86.67%) V2-dən (90.00%) üstündür. V2-nin söz CER üzrə 46.71% vs 50.99% nəticəsi cəmi 30 sözlük kiçik sınaqdır və ilkin (preliminary) xarakter daşıyır.
5. **D.18 — V2 Əlavələrinin Render Skripti:** `scratch/build_v2_datasets.py`
   - *Status:* **UNVERIFIED (Script not preserved, reproducibility limitation)**. `run_v2_audit.py:319` loqunda göstərilən `scratch/build_v2_datasets.py` skripti diskdə saxlanılmadığı üçün onun render çağırışını birbaşa sitat gətirmək mümkün deyil. Bu məqam tədqiqatın təkrar-istehsal məhdudiyyəti (reproducibility limitation) kimi qeyd edilməlidir.

---

## 2. 9 Əsas Bənd üzrə Dəqiqləşdirilmiş Dəyişikliklər (Change List)

### Bənd 1: Cədvəl 3 Şrift Paylanması və Dəqiq Saylar

* **Köhnə Mətn:** Bütün sətirlər "Seen" adlandırılmış və Tahoma üçün "4 dəfə azalma" iddia edilmişdir.
* **Faktiki Vəziyyət və Dəqiq Şrift Sayları:**
  - `generate_data.py:28-36, 88` kodunda `collect_fonts('fonts', 42)` funksiyası 10 fontdan 8-ni təlim və val üçün (`tr_fonts`), 2-sini test üçün (`te_fonts`: `tahoma.ttf` və `consola.ttf`) ayırır.
  - TRDG generatoru (`GeneratorFromStrings.next`, `self.fonts[(generated_count-1) % len(fonts)]`) fontları 2 test şrifti arasında növbəli (round-robin) bölmüşdür:
    * `data/synth_word_v2/test` ($n=2000$): **`tahoma.ttf`: dəqiq 1,001 şəkil**, **`consola.ttf`: dəqiq 999 şəkil**.
    * `data/synth_line_v2/test` ($n=1000$): **`tahoma.ttf`: dəqiq 501 şəkil**, **`consola.ttf`: dəqiq 499 şəkil**.
    * `val` dəstləri (`synth_word_v2/val` $n=2000$, `synth_line_v2/val` $n=1000$): V1-in 8 təlim şriftini ehtiva edir (`candara`: 252/126, `times`: 252/126, `cour`: 251/126, `segoeui`: 249/126, `verdana`: 249/126, `corbel`: 249/124, `arial`: 249/123, `calibri`: 249/123).
  - **Təsnifat:**
    * `tahoma.ttf` V2 təlimində yoxdur $\rightarrow$ **NEW FAMILY (Görünməmiş ailə)**. Nəticə: Word V1 **0.08%** $\rightarrow$ V2 **0.09%**; Line V1 **0.24%** $\rightarrow$ V2 **0.26%**. V2 yaxşılaşmayıb; "4x yaxşılaşma" iddiası yalandır və silinir.
    * `consola.ttf` üçün V2 təlimində `consolab.ttf` var $\rightarrow$ **FAMILY LEAKAGE (Ailə sızması)**. Nəticə: Word 1.02% $\rightarrow$ 0.07%; Line 3.48% $\rightarrow$ 0.27% (Ailə sızması hesabına kəskin düşüb).
    * `word_synth` və `line_synth` $\rightarrow$ **Qarışıq (1001/501 New Family + 999/499 Family Leakage)**.
* **Mənbə:** `generate_data.py:28-36, 88`, `results/v2_evaluation_report.md:26-34`, `results/v2_audit/summary.md:51-58`, `results/v2_prep.md:54-60`.

*Düzəldilmiş Cədvəl 3 (Azərbaycan dilində):*

| Test Dəsti | Dəqiq Şrift Tərkibi və Təsnifatı | Nümunə ($n$) | V1 CER (%) | V2 CER (%) | Tesseract 5 CER (%) | V1 $\rightarrow$ V2 Dəyişimi |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Söz, qarışıq test** (`word_synth`) | 1,001 `tahoma` (New) + 999 `consola` (Leakage) | 2000 | 0.50% | **0.11%** | 0.70% | CER 4.55 dəfə azaldı |
| **Sətir, qarışıq test** (`line_synth`) | 501 `tahoma` (New) + 499 `consola` (Leakage) | 1000 | 1.64% | **0.30%** | 0.44% | CER 5.47 dəfə azaldı |
| **Söz, Tahoma** (`diagw_tahoma`) | 1,000 `tahoma.ttf` — **NEW FAMILY** | 1000 | **0.08%** | 0.09% | 0.50% | Dəyişmədi (+0.01% dalğalanma) |
| **Sətir, Tahoma** (`diag_tahoma`) | 500 `tahoma.ttf` — **NEW FAMILY** | 500 | **0.24%** | 0.26% | 0.22% | Dəyişmədi (+0.02% dalğalanma) |
| **Söz, Consolas** (`diagw_consola`) | 1,000 `consola.ttf` — **FAMILY LEAKAGE** (`consolab`) | 1000 | 1.02% | **0.07%** | 0.86% | CER 14.57 dəfə azaldı (Ailə sızması) |
| **Sətir, Consolas** (`diag_consola`) | 500 `consola.ttf` — **FAMILY LEAKAGE** (`consolab`) | 500 | 3.48% | **0.27%** | 0.47% | CER 12.89 dəfə azaldı (Ailə sızması) |

---

### Bənd 2: WER Boşluq Təhlili (Retokenized WER Sütunu Tamamilə Çıxarıldı)

* **Düzəliş:** Retokenized WER sütunu dilimləmə artefaktı yaratdığı üçün tamamilə çıxarıldı. Eyni 1076 söz (Qrup B və C sətirləri) üzərində həm V2, həm Tesseract üçün xalis (Raw) WER və xəta növləri (xalis boşluq, tək hərf, çoxhərfli) təqdim olunur:

*Cədvəl (Groups B & C Sətirləri, eyni $n=1076$ söz):*

| Səviyyə və Model | Ümumi Söz ($n$) | Xalis WER (Raw) | Xalis Boşluq Xətası Sayı | Tək Hərf Xətası Sayı | Çoxhərfli Xəta Sayı | Boşluq Xətalarının Payı (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **L0 V2 Modeli** | 1076 | **5.39%** | 46 söz (4.3%) | 8 (0.7%) | 4 (0.4%) | **79.3%** |
| **L0 Tesseract 5** | 1076 | **2.70%** | 0 söz (0.0%) | 22 (2.0%) | 7 (0.7%) | **0.0%** |
| **L4 V2 Modeli** | 1076 | **9.11%** | 64 söz (5.9%) | 26 (2.4%) | 8 (0.7%) | **65.3%** |
| **L4 Tesseract 5** | 1076 | **3.16%** | 2 söz (0.2%) | 23 (2.1%) | 9 (0.8%) | **5.9%** |

*Elmi İzah:* V2 modelinin sətir söz xətalarının böyük əksəriyyəti (**L0-da 79.3%, L4-də 65.3%**) sözün daxilinə səhvən boşluq daxil edilməsindən (`t icarət inə`, `v irus`, `d in`) qaynaqlanır. Tesseract-da isə boşluq xətası demək olar ki, yoxdur (0.0% - 5.9%), lakin hərfi əvəzləmələr üstünlük təşkil edir.
*Mənbə:* `run_v2_audit4.py:521-556`, `results/v2_audit4/summary.md:205-212`.

---

### Bənd 3: Real Test 2 — `TƏQDİMAT` Düzəlişi, Kənar Kəsilmə və Qutuların Mənşəyi

1. **`TƏQDİMAT` Kəsiminin Düzəldilməsi (`x=300..485`):**
   - İlkin kəsim `test_real_doc2.py:48` daxilində `y=23..48, x=365..485` olaraq təyin edilmişdi. Bu kəsimdə sözün əsl başlanğıcı (`x=320`) kənarda qaldığı üçün həm V2 (`İMAT ı`), həm Tesseract (`İMAT`) yalnız görünən hissəni oxumuşdu.
   - Kəsim qutusu `x=300..485` olaraq yenidən kəsilib test edildikdə:
     * **Tesseract 5:** `TƏQDİMAT` $\rightarrow$ CER: **0.0%** (əvvəl 50.0% idi).
     * **V2 Modeli:** ` TəqDiMAT ı` $\rightarrow$ CER: **75.0%** (əvvəl 75.0% idi, 6 əməliyyat məsafəsi: 1 boşluq, 3 registr xətası `ə, q, i`, 1 boşluq, 1 `ı`).
     * **V1 Modeli:** `jirəggitifTijj` $\rightarrow$ CER: **162.5%** (əvvəl 125.0% idi).
   - **Bütün Foto 2 üzrə Sətir Göstəricilərinin Yenidən Hesablanması (Old vs New):**
     * **Tesseract 5 Sətir CER:** 0.72%-dən **0.24%**-ə endi! (Bütün sənəddə cəmi 2 hərf xətası: `Vöb-saytın` və `lurnal`).
     * **Tesseract 5 Sətir WER:** 3.37%-dən **2.25%**-ə endi! (21 sətirdən 19-u 100% tam qüsursuz).
     * **V2 Sətir CER:** **5.02%** olaraq qaldı (xəta sayı 42 / 837).
     * **V2 Sətir WER:** **13.48%** olaraq qaldı.
     * **V1 Sətir CER:** 48.15%-dən **48.51%**-ə dəyişdi.
     * **Sətir WER Dəqiqləşdirilməsi və Uyğunsuzluğun İzahı (V1 WER: 23.60% vs 98.88%):**
       - `results/real_test2/results.json:9` faylındakı `v1_wer: 23.60%` metriki standart WER deyildir! `test_real_doc2.py:167-178` kodunda sətirdə xəta olduqda `v1_word_err += 1` (sətir xətası) artırılmış, lakin cəmi söz sayına ($n=89$) bölünmüşdür: $21 / 89 = 23.5955\%$ (21 xətalı sətir / 89 söz).
       - Standart `common.cer_wer` formulası üzrə:
         $$\text{WER} = \frac{\sum_{i=1}^{N_{\text{lines}}} \text{Levenshtein}(\text{ref}_{i,\text{words}}, \text{hyp}_{i,\text{words}})}{\sum_{i=1}^{N_{\text{lines}}} \text{len}(\text{ref}_{i,\text{words}})}$$
       - V1 modeli 89 etalon sözdən 88-də söz xətası törətmişdir (yalnız sətir 17-dəki `daxil` sözü düzgün tanınmışdır). Yeganə düzgün standart WER: **88 / 89 = 98.88%**.
       - Eyni qayda ilə `results.json`-da V2 üçün göstərilən 13.48% (12 xətalı sətir / 89 söz) əvəzinə standart WER = **22.47%** (20 söz xətası / 89 söz); Tesseract üçün düzəldilmiş standart WER = **2.25%** (2 söz xətası / 89 söz)-dir.

2. **Kənar Kəsilmə və Qonşu Müdaxiləsi (Clipped-Ink & Neighbour Intrusion):**
   - **21 Sətir Kəsimi — Üfüqi Kəsilmə (Horizontal Clipping):** Yalnız **1 ədəd sətir (4.8%)** kəsilmişdir: Sətir #0 (`TƏQDİMAT`, sol kənar itkisi: 94 piksel / 19.0% itki). Qalan 20 sətirdə üfüqi itki **0 pikseldir**.
   - **21 Sətir Kəsimi — Şaquli Kəsilmə Yoxlanışı (Vertical Border Continuing Ink):** Kəsim sərhədi ($y_{top}$ və $y_{bot}$) boyunca birbaşa mürəkkəb kəsilməsi yoxlandıqda:
     * İlkin qutularda (`y0..y1`): 21 sətirdən **10-da (47.6%)** kəsim xəttini kəsən mürəkkəb mövcuddur (9 yuxarı, 5 aşağı).
     * Faktiki götürülmüş kəsimlərdə (`y0-2..y1+2`): 21 sətirdən **17-də (81.0%)** şaquli sərhəddə mürəkkəb davamiyyəti mövcuddur (14 yuxarı, 13 aşağı), çünki sətirlərarası məsafə kiçikdir və qonşu sətirlərin hərfləri kəsim sərhədinə toxunur. Yalnız 4 sətir (L00, L02, L04, L05) tam təcrid olunmuş təmiz sətirlərdir.
   - **30 Söz Kəsimi — Qonşu Müdaxiləsi (Neighbour Intrusion):** 30 söz kəsiminin **18-də (60.0%)** 10px kənar zonasında >15 piksel qonşu söz mürəkkəbi müdaxiləsi mövcuddur. 30 kəsimin **30-da da (100.0%)** 10px daxilində qonşu mürəkkəb aşkarlanmışdır.

3. **Qutuların və Mətnlərin Mənşəyi:**
   - Kəsim qutuları dinamik `image_to_data` çağırışı deyil, Tesseract `image_to_data` çıxarışının ilkin analizi əsasında əllə təyin edilmiş və `test_real_doc2.py:46-135` faylına daxil edilmiş sabit koordinat cütləridir:
     ```python
     # test_real_doc2.py:46-50
     REAL_LINES = [
         ("TƏQDİMAT", 23, 48, 365, 485),
         ("Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi", 149, 173, 2, 495),
         ("hazırlanması.", 194, 214, 50, 165),
     ```
   - Etalon mətnlər (Ground Truth) Tesseract çıxışı deyil, fotoşəkildə gözlə aydın oxunan sənəd mətni əsasında əllə transkripsiya edilmişdir.
* **Mənbə:** `test_real_doc2.py:46-135`, `results/real_test2/results.json:1-45`.

---

### Bənd 4: Foto 1 və Foto 2 Simvol Sayları

* **Foto 1 (`media_1793193977057.jpg`):**
  - Sətir səviyyəsi: $n=15$ sətir, **762 simvol**.
  - Söz səviyyəsi: $n=30$ söz, **285 simvol** (dəyişdirilmir, qorunur).
  *Mənbə:* `results/real_test/evaluation.md:12-13`.
* **Foto 2 (`media_1793194879421.jpg`):**
  - Sətir səviyyəsi: $n=21$ sətir, **837 simvol**, 89 söz.
  - Söz səviyyəsi: $n=30$ söz, **304 simvol**, 30 söz (əvvəlki 296 və 285 səhvləri 304 olaraq düzəldildi).
  *Mənbə:* `results/real_test2/results.json:4, 248`.

---

### Bənd 5: Nərdivan Qeyri-Monotonluğunun İzahı (A.7)

* **Köhnə Cümlə:** "Binarizasiyaya L4-dəki bucaq və işıq dreyfindən daha dağıdıcı təsir göstərməsidir."
* **Düzəliş:** "Binarizasiya" fərziyyəsi silindi (V1 CRNN arxitekturası grayscale təsvirlərlə işləyir). Dəqiq ifadə:
  *"V1 sətir CER göstəricisi pillələr üzrə monoton artmır; belə ki, L3 səviyyəsindəki xəta (19.14%) L4 səviyyəsindəki xətadan (12.96%) daha yüksəkdir. Bu fərqin dəqiq səbəbi araşdırılmamışdır (cause not investigated)."*
* **Mənbə:** `results/v2_audit/summary.md:64-73`.

---

### Bənd 6: V2 Əlavələrinin Render Provenansı (D.18)

* **Köhnə Mətn:** "V2-yə əlavə olunan 15,000 söz və 6,000 sətir fonts_new/ şriftləri ilə birbaşa PIL (ImageDraw) vasitəsilə generasiya edilmişdir."
* **Düzəliş:** V1 generasiyasında TRDG (`generate_data.py:40`, `from trdg.generators import GeneratorFromStrings`) istifadəsi təsdiqlənmişdir. V2 əlavələrini generasiya edən `scratch/build_v2_datasets.py` skripti isə diskdə saxlanılmadığı üçün onun render çağırışını birbaşa sitat gətirmək mümkün deyil və bu, **təkrar-istehsal məhdudiyyəti (script not preserved, reproducibility limitation)** kimi qeyd olunmalıdır.
* **Mənbə:** `generate_data.py:40-55`, `run_v2_audit.py:319`, `results/v2_audit/summary.md:10`.

---

### Bənd 7: Əsas Cədvələ Bütün Qrupların (A, B, C) Bootstrap İntervalları ilə Əlavəsi (C.14)

*Audit 4-ün tam klaster bootstrap təhlili (`results/v2_audit4/summary.md:118-132` və `run_v2_audit4.py:718-732`):*

*Cədvəl: Qrup A (Seen), Qrup B (Near-Relative) və Qrup C (New Family) üzrə Klaster Bootstrap (95% CI) Nəticələri*

| Qrup | Səviyyə və Rejim | Nümunə ($n$) | Simvol Sayı | V1 CER (%) | V2 CER (%) [95% CI] | Tesseract 5 CER (%) [95% CI] | Fərq $\Delta$ (V2 - Tess) [95% CI] | Sıfır CI-yə Daxildir? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qrup A** | Söz L0 | 378 | 2472 | 7.36% | **0.81%** [0.32%, 1.57%] | 2.39% [1.39%, 3.83%] | **-1.58%** [-2.68%, -0.76%] | **XEYR (V2 öndədir)** |
| **Qrup A** | Sətir L0 | 95 | 2777 | 0.29% | **0.11%** [0.00%, 0.30%] | 0.29% [0.00%, 0.65%] | **-0.18%** [-0.57%, +0.11%] | BƏLİ (Bərabər) |
| **Qrup A** | Söz L4 | 378 | 2472 | 45.27% | **1.33%** [0.58%, 2.20%] | 1.70% [0.87%, 2.84%] | **-0.36%** [-1.25%, +0.50%] | BƏLİ (Bərabər) |
| **Qrup A** | Sətir L4 | 95 | 2777 | 13.83% | **1.15%** [0.18%, 3.00%] | 1.62% [0.11%, 4.09%] | **-0.47%** [-1.91%, +0.44%] | BƏLİ (Bərabər) |
| **Qrup B** | Söz L0 | 436 | 3012 | 9.56% | **0.46%** [0.07%, 0.97%] | 0.96% [0.30%, 1.89%] | **-0.50%** [-1.28%, +0.09%] | BƏLİ (Bərabər) |
| **Qrup B** | Sətir L0 | 114 | 3368 | 2.35% | **0.30%** [0.00%, 0.92%] | 0.12% [0.00%, 0.36%] | **+0.18%** [-0.31%, +0.88%] | BƏLİ (Bərabər) |
| **Qrup B** | Söz L4 | 436 | 3012 | 41.33% | **0.73%** [0.27%, 1.31%] | 1.10% [0.39%, 2.08%] | **-0.37%** [-1.26%, +0.50%] | BƏLİ (Bərabər) |
| **Qrup B** | Sətir L4 | 114 | 3368 | 16.12% | **0.50%** [0.06%, 1.17%] | 0.18% [0.00%, 0.44%] | **+0.33%** [-0.18%, +1.06%] | BƏLİ (Bərabər) |
| **Qrup C** | Söz L0 | 609 | 4194 | 9.73% | **1.62%** [0.22%, 3.44%] | 1.00% [0.36%, 1.78%] | **+0.62%** [-0.35%, +2.08%] | BƏLİ (Bərabər) |
| **Qrup C** | Sətir L0 | 158 | 4673 | 6.03% | **0.71%** [0.08%, 1.68%] | 0.71% [0.19%, 1.46%] | **+0.00%** [-0.74%, +0.76%] | BƏLİ (Tam bərabər) |
| **Qrup C** | Söz L4 | 609 | 4194 | 41.49% | **2.36%** [1.23%, 3.95%] | 0.81% [0.37%, 1.31%] | **+1.55%** [+0.59%, +2.94%] | **XEYR (Tess öndədir)** |
| **Qrup C** | Sətir L4 | 158 | 4673 | 20.93% | **1.28%** [0.16%, 2.75%] | 1.46% [0.37%, 3.72%] | **-0.17%** [-2.60%, +1.53%] | BƏLİ (Bərabər) |

*Metodoloji Qeyd:* Qrup A-da cəmi 4 şrift (`arial, calibri, segoeui, times`), Qrup B-də isə cəmi 5 şrift (`pala, CascadiaMono, verdanab, seguisb, calibrib`) iştirak etdiyi üçün bu qruplarda klaster sayı azdır və klaster bootstrap intervalları şriftlərarası yüksək dispersiya səbəbindən həssasdır (fragile).
*Mənbə:* `run_v2_audit4.py:718-732`, `results/v2_audit4/summary.md:118-132`.

---

### Bənd 8: Korpus Bölgüsü — Word Types vs Tokens (A.3)

* **Kodun Dəqiq Təhlili (`prepare_corpus.py:99-125`):**
  - `--mode word` rejimində kod mətndəki bütün sözləri yığır, kiçik hərfə çevirir və `seen` çoxluğu ilə **unikal söz tiplərini (unique word forms)** seçir (`prepare_corpus.py:100-106`). Sonra bu unikal söz siyahısı `split_list(words)` ilə 80% train / 10% val / 10% test olaraq bölünür.
  - `--mode line` rejimində isə mətnlər cümlələrə bölünür, cümlələr deduplikasiya edilir (`prepare_corpus.py:114`) və **cümlə səviyyəsində** `split_list(sents)` ilə bölünərək sətirlərə yığılır.
* **22.40% Overlap İzahı:**
  `run_v2_audit3.py:490-510` kodunda `test_words_all` test dəstlərindəki **bütün söz instansiyalarını (word tokens)** ehtiva edir ($n=1232$ token). Onlardan 276 ədədi təlim lüğətində mövcuddur. Deməli, 22.40% (276 / 1232) söz tiplərinin deyil, **test söz tokenlərinin təlim lüğəti ilə təbii kəsişməsidir**.
* **Mənbə:* `prepare_corpus.py:99-125`, `run_v2_audit3.py:490-511`, `results/v2_audit3/summary.md:147-151`.

---

### Bənd 9: Cədvəl 4 Şrift Tərkibi (A.7)

* **Başlıq:** *Cədvəl 4. 4-Şriftli Simulyasiya Nərdivanında (`cambria.ttc`, `micross.ttf` — 2 yeni ailə; `pala.ttf`, `CascadiaMono.ttf` — 2 yaxın qohum) Söz və Sətir CER (%) dinamikası*.
* **Şrift Tərkibi:** `cambria.ttc` ($n=81$ söz, 22 sətir), `micross.ttf` ($n=98$ söz, 24 sətir), `pala.ttf` ($n=84$ söz, 23 sətir), `CascadiaMono.ttf` ($n=83$ söz, 23 sətir). Cəmi: 346 söz, 92 sətir.
* **Mənbə:** `results/v2_audit3/summary.md:68-77`, `results/sim_ladder/summary.md:9-14`.

---

Bu dəqiqləşdirilmiş düzəlişlər toplusu [`results/Az_OCR_Akademik_Hesabat_FIXES_v2.md`](file:///d:/laragon/www/az_ocr_bm606/results/Az_OCR_Akademik_Hesabat_FIXES_v2.md) faylında tam saxlanılmışdır. Orijinal `.docx` faylı dəyişdirilməmişdir.
