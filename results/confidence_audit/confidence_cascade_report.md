# AzOCR-V2 — Aşama 2.9: CTC Güvən Skoru, AUROC və Hibrid Kaskad Denetimi Hesabatı

**Tarix:** 24 Noyabr 2026  
**Qiymətləndirmə Mühiti:** Intel CPU, 4 sabit `torch` / OpenMP worker axını, PyTorch 2.6.0+cpu, Tesseract 5.4.1  
**Məlumat Bazası:** Rəsmi 295 sətirlik B1–B15 Benchmarkı (55 real fiziki foto [B1–B3] + 240 simulyasiya edilmiş smartfon çəkilişi [B4–B15])  
**Əlaqəli Fayllar:**
- Xam ölçmə qeydləri: [confidence_295_lines.csv](file:///d:/laragon/www/az_ocr_bm606/results/confidence_audit/confidence_295_lines.csv)
- Ətraflı metrik JSON bazası: [confidence_cascade_audit_results.json](file:///d:/laragon/www/az_ocr_bm606/results/confidence_audit/confidence_cascade_audit_results.json)
- Əlaqəli sürət hesabatı: [cpu_speed_benchmark_report.md](file:///d:/laragon/www/az_ocr_bm606/results/speed_benchmark/cpu_speed_benchmark_report.md)

---

## 1. Giriş və Tədqiqatın Məqsədi

Doktorluq dissertasiyası və məqalə reviziyasında qaldırılan Aşama 2.9 tələblərinə uyğun olaraq:
1. CRNN-V2 Raw modelinin CTC Greedy dekodlamasında yaranan simvol inam skorlarının xətaları aşkarlama (error detection) qabiliyyəti AUROC (Area Under ROC Curve) metriki ilə rəsmi 295 sətirlik benchmark üzərində audit edilmişdir.
2. Məqalənin 6.8.1 bölməsində və Cədvəl 8b-də təqdim olunan **"%20 fallback eşiyi"nin mənşəyi və metodoloji etibarlılığı** araşdırılmışdır.
3. CRNN-V2 və Tesseract 5 (PSM 7) arasında inam skoru əsasında qurulan **hibrid kaskad (confidence-gated cascade)** sistemi 295 benchmark sətirinin tamamında yoxlanılmış; CER, WER, Exact Match və effektiv CPU gecikməsi hesablanmışdır.
4. Təxminlər üzərində **həqiqi 2x2 eşləşdirilmiş McNemar cədvəli** qurulmuş, əvvəlki Cədvəl 6b üçün çatışmayan xam məlumatlar bəyan edilmişdir.
5. Aşama 2.8 CPU sürət hesabatındakı model inisializasiyası vahidləri və Tesseract API sürət təxminləri metodoloji baxımdan düzəldilmişdir.

---

## 2. CTC Güvən Skoru və AUROC Xəta Aşkarlama Analizi

CRNN-V2 modelində CTC zaman addımlarında hər emit edilmiş simvol üçün $P(c_t \mid x)$ posterior ehtimalı hesablanmış, Greedy ardıcıllıqdan blank və təkrar simvollar çıxarıldıqdan sonra iki inam metriki formalaşdırılmışdır:
- **Min Simvol İnamı ($\min_{t} P(c_t)$):** Sətir daxilində ən zəif tanınan simvolun inam dərəcəsi.
- **Orta Simvol İnamı ($\frac{1}{L}\sum_{t} P(c_t)$):** Sətirdəki bütün simvolların orta inamı.

Sətirlərin düzgünlüyü rəsmi **Exact Match** meyarına görə etiketlənmişdir ($1 = \text{proqnoz} \equiv \text{ground truth}$, $0 = \text{xətalı sətir}$).

### Cədvəl 1: B1–B15 Benchmarkı Üzrə AUROC Xəta Aşkarlama Nəticələri

| Bölmə | Sətir Sayı ($N$) | Tam Dəqiq ($Exact$) | Xətalı Sətir | Dəqiqlik (%) | Exact Orta Min İnam | Xətalı Orta Min İnam | AUROC (Min İnam) | AUROC (Orta İnam) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bütün Benchmark (B1–B15)** | **295** | **253** | **42** | **85.76%** | **0.6844** | **0.6086** | **0.6344** | **0.6973** |
| Real Fiziki Sənədlər (B1–B3) | 55 | 44 | 11 | 80.00% | 0.7435 | 0.6610 | **0.6198** | **0.5723** |
| Simulyasiya Edilmiş (B4–B15) | 240 | 209 | 31 | 87.08% | 0.6719 | 0.5899 | **0.6592** | **0.7469** |

### Elmi Təhlil və Nəticə:
1. **Məqalədəki İddianın Təkzibi:** Əlyazmada (Bölmə 6.8.1) CTC inam skorunun xəta aşkarlama qabiliyyətinin $\text{AUROC} = 0.848$ olduğu iddia edilmişdi. Lakin həmin analiz yalnız Fotos 1–5 (111 sətir) üzərində və seqmentlərə ayrılmış natamam sətir fraqmentlərində aparılmışdı. Rəsmi 295 sətirlik tam benchmarkda real xəta aşkarlama qabiliyyəti **0.6344 (Min İnam)** və **0.6973 (Orta İnam)** səviyyəsindədir.
2. **Real vs Simulyasiya Fərqi:** Real fiziki fotoşəkillərdə (B1–B3) orta inam üzrə AUROC cəmi 0.5723-ə düşür (təsadüfi təxmin olan 0.50-yə yaxın). Bunun səbəbi, modelin bəzi fiziki təhriflərdə (perspektiv, kölgə) yüksək inamla səhv simvol seçməsidir (overconfident errors). Simulyasiya edilmiş sətirlərdə isə model xətaları nisbətən daha aydın aşağı inamla büruzə verir ($\text{AUROC} = 0.7469$).

---

## 3. Məqalədəki "%20 Kaskad Eşiyi"nin Mənşəyi və Metodoloji Tənqidi

Layihənin kod bazasında (`scratch/run_strengthening_analyses.py`, sətir 327–353) aparılan audit aşağıdakı faktı aşkar etmişdir:
- **Eşik Mənbəyi:** Əlyazmada istifadə olunan %20 fallback eşiyi müstəqil bir validasiya (tüninq) dəstində deyil, birbaşa Fotos 1–5 test sənədlərindəki 111 sətir üzərində post-hoc şəbəkə axtarışı (`sweep = [0, 5, 10, 15, 20, 25, 30%]`) ilə seçilmişdir.
- **Data Snooping (Məlumat Sızması/Uyğunlaşdırma):** Test məlumatlarının nəticələrinə baxaraq hiperparametr (kaskad kəsim həddi) seçmək klassik metodoloji səhvdir (data snooping / test set overfitting).
- **Audit Qərarı:** Bu auditdə test dəstinə təkrar uyğunlaşdırma (optimallaşdırma) aparılmamış; orijinal %20 eşiyi və digər fallback səviyyələri 295 benchmark sətirində olduğu kimi dondurularaq yoxlanılmışdır. Bu hal reviziyada açıq elmi məhdudiyyət kimi göstərilməlidir.

---

## 4. 295 Sətirlik Benchmark Üzərində Hibrid Kaskad Qiymətləndirməsi

Kaskad məntiqi: Ən aşağı inam skoruna malik sətirlər Tesseract 5 (PSM 7) mühərrikinə yönləndirilir, qalan sətirlər CRNN-V2 Raw modelində saxlanılır.  
Effektiv CPU gecikməsi düsturu:
$$\text{Gecikmə}_{\text{eff}} = T_{\text{CRNN}} + (\text{Fallback } \%) \times T_{\text{Tess}}$$
(burada $T_{\text{CRNN}} = 85.71\text{ ms}$, $T_{\text{Tess}} = 128.40\text{ ms}$).

### Cədvəl 2: Fallback Nisbətlərinə Görə Kaskad Nəticələri (295 Sətir)

| Fallback (%) | Yönləndirilən Sətir | Kəsim Həddi ($\min \text{conf} \le$) | CER (%) | WER (%) | Exact Match | Exact (%) | Effektiv CPU Gecikməsi (ms/sətir) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Təmiz CRNN Raw)** | **0** | **0.0000** | **0.298%** | **1.64%** | **253 / 295** | **85.76%** | **85.7 ms** |
| 5% | 15 | 0.5096 | 0.269% | 1.40% | 257 / 295 | 87.12% | 92.1 ms |
| 10% | 30 | 0.5233 | 0.287% | 1.49% | 256 / 295 | 86.78% | 98.5 ms |
| 15% | 44 | 0.5331 | 0.284% | 1.43% | 257 / 295 | 87.12% | 105.0 ms |
| **20% (Məqalə Eşiyi)** | **59** | **0.5435** | **0.291%** | **1.49%** | **255 / 295** | **86.44%** | **111.4 ms** |
| 25% | 74 | 0.5508 | 0.280% | 1.40% | 257 / 295 | 87.12% | 117.8 ms |
| 30% | 88 | 0.5647 | 0.269% | 1.31% | 258 / 295 | 87.46% | 124.2 ms |
| 50% | 148 | 0.6428 | 0.272% | 1.34% | 257 / 295 | 87.12% | 149.9 ms |
| **100% (Təmiz Tesseract)** | **295** | **0.9970** | **0.310%** | **1.55%** | **250 / 295** | **84.75%** | **214.1 ms\*** |

*\*Qeyd: 100% fallback rejimində ikiqat ardıcıl emal (CRNN inferensi 85.7 ms + Tesseract 128.4 ms = 214.1 ms) baş verir.*

---

## 5. Baş-başa Müqayisə: CRNN Raw vs Tesseract vs %20 Kaskad

### Cədvəl 3: Modellərin Baş-başa Müqayisəsi

| Model / Konfiqurasiya | CER (%) | WER (%) | Exact Match | Exact (%) | Fallback Nisbəti | CPU Gecikməsi (ms/sətir) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2 Raw** | **0.298%** | 1.64% | 253 / 295 | 85.76% | 0% | **85.71 ms** |
| **Tesseract 5 (PSM 7)** | 0.310% | 1.55% | 250 / 295 | 84.75% | 100% | 128.40 ms (birbaşa) |
| **Kaskad (%20 Fallback)** | 0.291% | **1.49%** | **255 / 295** | **86.44%** | 20% (59 sətir) | 111.39 ms |

### Kritik Elmi Müşahidələr:
1. **Dəqiqlik Qazancının Həddindən Artıq Kiçik Olması:** 295 sətirlik benchmarkda təmiz CRNN-V2 Raw artıq Tesseract-dan daha çox tam dəqiq sətir verir (253 vs 250). Ən şübhəli 59 sətiri Tesseract-a yönləndirdikdə tam dəqiq sətir sayı cəmi **2 sətir** (253-dən 255-ə) artır, CER isə 0.298%-dən cəmi 0.291%-ə enir.
2. **Gecikmə Cəzası:** Bu cüzi qazanc (+2 sətir) üçün orta gecikmə **85.7 ms-dən 111.4 ms-ə (+30.0% artım)** yüksəlir.
3. **Kaskadın Mənasızlaşması:** Məqalədəki "kaskad sistemi hər iki modeli əhəmiyyətli dərəcədə qabaqlayır" iddiası 295 sətirlik rəsmi benchmarkda təsdiqlənmir; kaskad sistemi həm hesablama mürəkkəbliyini artırır, həm də əhəmiyyətli dəqiqlik sıçrayışı təmin etmir.

---

## 6. Real Eşləşdirilmiş McNemar Testi və Çatışmayan İlkin Məlumatlar

### 6.1. 295 Sətir Üzrə Həqiqi Eşləşdirilmiş McNemar Cədvəli (Exact Match)

Aşama 2.7-də müstəqil Fisher testi ilə bağlı qaldırılan iradı aradan qaldırmaq üçün CRNN-V2 Raw və Tesseract 5 (PSM 7) proqnozları sətirbəsətir eşləşdirilmişdir ($N = 295$):

```
                       Tesseract 5 Düzgün     Tesseract 5 Səhv       Cəmi
CRNN-V2 Düzgün              232 (a)                 21 (b)            253
CRNN-V2 Səhv                 18 (c)                 24 (d)             42
Cəmi                        250                     45               295
```

- **Uyğunsuz Cütlər (Discordant Pairs):** $b + c = 21 + 18 = 39$
- **Dəqiq İkitərəfli Binomial McNemar Testi:**
  $$p = 2 \times \sum_{k=0}^{\min(21, 18)} \binom{39}{k} 0.5^{39} = 0.7493$$
- **Nəticə:** $p = 0.7493 \gg 0.05$. İki modelin 295 sətirlik rəsmi benchmark üzrə xəta paylanmaları arasında statistik cəhətdən əhəmiyyətli heç bir fərq yoxdur.

### 6.2. Tarixi Cədvəl 6b (244 Böyük Hərf Xüsusi Simvolu) Üzrə Məlumat Çatışmazlığı
- Əvvəlki hesabatlarda 244 xüsusi böyük hərf simvolu üzrə CRNN-V2-nin 6 xəta, Tesseract-ın isə 9 xəta etdiyi qeyd olunmuşdu.
- Layihənin arxiv qeydlərində həmin 244 simvolun **simvolbəsimvol binar (0/1) proqnoz cütləri saxlanılmamışdır**, yalnız ümumi cəmlər (6 və 9) mövcuddur.
- Şərikli xəta sayı $a \in [0, 6]$ aralığında dəyişə bildiyindən, uyğunsuz cütlər $(b, c) = (6 - a, 9 - a)$ təşkil edir. Mümkün bütün 7 ssenari üzrə McNemar $p$-dəyəri $0.2500 \le p \le 0.6072$ aralığında qalır. Həqiqi cütlər bərpa edilə bilməsə belə, statistik qeyri-əhəmiyyətlilik qəti şəkildə qorunur.

---

## 7. Aşama 2.8 Sürət Hesabatının Metodoloji Düzəlişləri

Aşama 2.8 hesabatında müəllif tərəfindən aşkar edilmiş və bu hesabatla rəsmən düzəldilən məqamlar:
1. **Model İnisializasiya Müddətinin Vahid Düzəlişi:**
   - Aşama 2.8 hesabatında PaddleOCR üçün göstərilən `2.210 ms` göstəricisi əslində **2,209.9 ms (yəni 2.21 saniyə)** olmuşdur. Hesabatdakı vergül və nöqtə fərqi aradan qaldırılmış və Cədvələ `2.21 s (2,210 ms)` kimi daxil edilmişdir.
2. **Standart Meylin (Std) Şərhi:**
   - CRNN ($\pm 23.58\text{ ms}$) və Tesseract ($\pm 15.32\text{ ms}$) üçün göstərilən standart meyllər təkrar testlər arası qeyri-sabitliyi deyil, **sətirdən-sətirə (line-to-line) uzunluq və simvol sayı dəyişkənliyini** ifadə edir.
   - 3 təkrarın ümumi icra müddətləri arasındakı meyl (repetition-to-repetition) olduqca aşağıdır: CRNN üçün $\pm 0.95\text{ s}$ ($\pm 3.2\text{ ms/sətir}$), Tesseract üçün $\pm 0.25\text{ s}$ ($\pm 0.8\text{ ms/sətir}$).
3. **Tesseract Doğrudan API Sürəti Təxmini:**
   - Aşama 2.8 hesabatında Tesseract üçün qeyd olunan "~60–65 ms/sətir birbaşa C++ API sürəti" ölçülmüş fakt deyil, Windows subprocess spawn gecikməsinin (64.8 ms) 128.4 ms-dən çıxılması ilə əldə edilmiş nəzəri aşağı həddir. Rəsmi elmi müqayisədə yalnız real ölçülmüş 128.40 ms göstəricisi əsas götürülməlidir.

---

## 8. Xülasə və Nəticə

| Meyar | CRNN-V2 Raw | Tesseract 5 (PSM 7) | Hibrid Kaskad (%20) |
| :--- | :---: | :---: | :---: |
| **CER (%)** | 0.298% | 0.310% | 0.291% |
| **WER (%)** | 1.64% | 1.55% | 1.49% |
| **Exact Match (295)** | 253 (85.76%) | 250 (84.75%) | 255 (86.44%) |
| **CPU Gecikməsi** | **85.71 ms** | 128.40 ms | 111.39 ms |
| **AUROC (Min İnam)** | 0.6344 | — | — |
| **McNemar $p$-dəyəri** | — | $p = 0.7493$ (Əhəmiyyətsiz) | — |

**Yekun Elmi Rəy:**
295 sətirlik tam benchmark üzərində aparılan dürüst analiz göstərir ki, CTC inam skoru əsasında qurulan hibrid kaskad sistemi **praktik və statistik cəhətdən əsaslandırılmamışdır**. Cəmi 2 sətirlik qazanc üçün gecikmə 30% artır. Dissertasiya və məqalədə kaskad sistemi əvəzinə təkbaşına yüngül CRNN-V2 Raw modelinin və ya post-processing üçün lüğət əsaslı SymSpell yanaşmasının tövsiyə edilməsi elmi və praktiki baxımdan ən optimal qərardır.
