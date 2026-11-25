# AzOCR-V2: CPU Speed Benchmark və Eşləşdirilmiş Statistik Yoxlama Hesabatı (Aşama 2.8)

**Tarih:** 2026-11-24  
**Test Mühiti:**  
- **Prosessor (CPU):** Intel(R) Core(TM) 7 240H (10 fiziki nüvə, 16 məntiqi prosessor)  
- **Sabitlənmiş Thread Sayı:** 4 thread (`torch.set_num_threads(4)`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`)  
- **Test Küməsi:** Rəsmi 295 sətirlik B1–B15 smartfon benchmarkı (55 real fiziki sətir + 240 simulyasiya sətiri)  
- **Protokol:** 10 sətir ilkin warm-up; hər model üçün ardıcıl 3 tam təkrar (repetition) dövrü  
- **Disk I/O Təcridi:** Bütün 295 sətir əvvəlcədən operativ yaddaşa (RAM) yüklənmiş və disk oxuma gecikməsi aradan qaldırılmışdır.

---

## 1. Yekun Nəticələr Cədvəli (CPU Speed Benchmark)

| Model / Konfiqurasiya | Model Başlatma (Init) | 3 Təkrar Müddəti (s) | **Orta Gecikmə (ms / sətir)** | **Standart Sapma ($\sigma$)** | **Real 55 Sətir (ms/sətir)** | **Simüle 240 Sətir (ms/sətir)** | **Buraxılış Qabiliyyəti (sətir/s)** | Çağırış Mexanizmi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CRNN-V2 Raw** | 2.21 s (2,210 ms) | 26.35 / 24.90 / 24.61 | **85.71 ms** | **$\pm$ 23.58 ms** | 79.55 ms | 87.12 ms | **11.67 sətir/s** | Birbaşa C++ libtorch tensor |
| **CRNN-V2 + SymSpell** | 2.21 s (2,210 ms) | 24.46 / 24.19 / 24.42 | **82.56 ms** | **$\pm$ 22.33 ms** | 75.07 ms | 84.28 ms | **12.11 sətir/s** | Birbaşa C++ in-memory çağırış |
| **Tesseract 5 PSM 7** | — (hər çağırışda 64.8 ms subprocess) | 38.06 / 37.97 / 37.60 | **128.40 ms** | **$\pm$ 15.32 ms** | 130.94 ms | 127.82 ms | 7.79 sətir/s | Windows CLI Subprocess (`pytesseract`) |
| **Tesseract 5 PSM 13** | — (hər çağırışda 64.8 ms subprocess) | 37.85 / 37.39 / 37.63 | **127.53 ms** | **$\pm$ 16.26 ms** | 133.61 ms | 126.14 ms | 7.84 sətir/s | Windows CLI Subprocess (`pytesseract`) |
| **EasyOCR (`az`)** | 1.37 s (1,365 ms) | 41.79 / 41.72 / 41.79 | **141.59 ms** | **$\pm$ 34.71 ms** | 103.33 ms | 150.35 ms | 7.06 sətir/s | Birbaşa PyTorch CPU (`recognize`) |
| **PaddleOCR (`PP-OCRv6`)** | 1.28 s (1,278 ms) | 563.93 / 600.82 / 581.20 | **1972.82 ms** | **$\pm$ 448.58 ms** | 1406.81 ms | 2102.53 ms | 0.51 sətir/s | Birbaşa PaddleX oneDNN CPU |

---

## 2. Tesseract Çağırış Mexanizmi və Subprocess Əlavə Yükü (Overhead)

Tesseract testlərində Python tərəfindən istifadə olunan kitabxana `pytesseract` paketidir.
- **Çağırış Növü:** Birbaşa C API (`libtesseract.dll`) deyil, **Windows CLI Subprocess** (`subprocess.Popen(["tesseract.exe", ...])`) vasitəsilə icra olunur.
- **Ölçülən Subprocess Əlavə Yükü:** Boş 1x1 ağ görüntü və versiya çağırışları üzərində aparılan 20 təkrar ölçüm nəticəsində hər çağırış üçün sırf OS prosesi yaratma (`CreateProcessW`), DLL yükləmə və `aze.traineddata` lüğətini diskdən yaddaşa çəkmə əlavə yükü tam **64.82 ms** təşkil etmişdir.
- **Təhlil:** Tesseract-ın ölçülən 127.5 ms gecikməsinin **50.8%-i** (64.8 ms) OCR mühərrikinin öz hesablamasından deyil, Windows-da hər sətirdə təkrar subprocess açılmasından qaynaqlanır. Əgər birbaşa in-process C API istifadə edilsəydi, Tesseract-ın faktiki xalis tanıma müddəti təxminən **60–65 ms/sətir** səviyyəsində olardı.

---

## 3. Makalədəki Fərqli Sürət Rəqəmlərinin İzahı (155 ms vs. 86 ms)

Makalə layihəsi ilə cari hesabatlar arasındakı gecikmə fərqləri tamamilə hardware (prosessor arxitekturası) və ölçüm konfiqurasiyası ilə bağlıdır:

1. **Makalədəki 155.35 ms / sətir:**  
   - **Hardware:** Köhnə noutbuk prosessoru: `Intel Core i5-7200U @ 2.50 GHz` (cəmi 2 fiziki nüvə, 4 məntiqi prosessor, köhnə Kaby Lake arxitekturası).
   - **Test Küməsi:** Yalnız Foto 5 ($n=33$ sətir, `batch=1` unbatched).
   - **Batched CRNN:** Eyni zəif prosessorda `batch=16` ilə 127.6 ms ölçülmüşdür.
2. **Cari Hesabatdakı 85.71 ms / sətir:**  
   - **Hardware:** Müasir yüksək tezlikli masaüstü prosessoru: `Intel Core 7 240H` (10 fiziki nüvə, 16 thread, müasir Raptor Lake IPC).
   - **Test Küməsi:** Rəsmi 295 sətirlik B1–B15 benchmarkı (`batch=1`, in-memory).
3. **Elmi Prinsip:** Müxtəlif donanımlarda (i5-7200U vs. Core 7 240H) ölçülən nəticələri birbaşa mütləq sürət üstünlüyü kimi təqdim etmək metodoloji xətadır. Bütün rəqəmlər yalnız eyni avadanlıq və eyni thread parametrində (`OMP_NUM_THREADS=4`) müqayisə edilməlidir.

---

## 4. PaddleOCR Model Başlatma vs. Steady-State İnference Ayrımı

- **Kitabxana və Model Başlatma (Init Time):** **1.28 saniyə (1,277.6 ms)**. Model çəkilərinin diskdən oxunması və oneDNN icra qrafının tərtibi cəmi 1.28 saniyə çəkir.
- **Steady-State Gecikmə:** Hər 295 sətirlik tam təkrar dövrü təxminən 563–600 saniyə (9.4 – 10 dəqiqə) çəkir; orta steady-state gecikmə **1,972.82 ms / sətir** ($\approx 1.97$ s/sətir) təşkil edir.
- **Səbəb:** PaddleOCR-ın `PP-OCRv6_medium_rec` modeli daha mürəkkəb Vision-Transformer / dərin konvolyusiya laylarına malikdir və CPU üzərində float32 ilə işlədikdə CRNN-ə nisbətən kəskin dərəcədə ağır hesablanır.

---

## 5. Eşləşdirilmiş Statistik Yoxlama (McNemar Testi vs. Fisher Testi)

### 5.1. Müstəqil Fisher Testinin Metodoloji Qüsuru
Aşama 2.7-də 244 böyük hərf xüsusi diakritikində V2 (6 xəta) ilə Tesseract (9 xəta) arasında müstəqil Fisher testi ($p = 0.6015$) aparılmışdı.  
**Tənqid:** Fisher testi iki *müstəqil* qrup fərz edir. Halbuki hər iki model eyni 244 xarakter üzərində sınaqdan keçirilib; yəni müşahidələr **tamamilə eşləşdirilmişdir (paired data)**.

### 5.2. Eşləşdirilmiş McNemar Dəqiq Binomial Testi
Eşləşdirilmiş cütlərdə qeyri-parametrik düzgün test **McNemar testidir**. Hər iki modelin ortaq xətalarının sayı $a \in [0, 6]$ olmaqla bütün mümkün uyğunsuz cütlər ($(b, c)$, burada $b = 6-a$, $c = 9-a$) üzrə hesablanan dəqiq iki-tərəfli binomial $p$-dəyərləri:

| Ortaq Xəta ($a$) | V2 Tək Xətası ($b$) | Tess Tək Xətası ($c$) | Uyğunsuz Cüt ($b+c$) | McNemar $p$-dəyəri | Statistik Əhəmiyyətli ($p < 0.05$)? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 6 | 9 | 15 | **0.6072** | Qətiyyən Yox |
| **1** | 5 | 8 | 13 | **0.5811** | Qətiyyən Yox |
| **2** | 4 | 7 | 11 | **0.5488** | Qətiyyən Yox |
| **3** | 3 | 6 | 9 | **0.5078** | Qətiyyən Yox |
| **4** | 2 | 5 | 7 | **0.4531** | Qətiyyən Yox |
| **5** | 1 | 4 | 5 | **0.3750** | Qətiyyən Yox |
| **6** | 0 | 3 | 3 | **0.2500** | Qətiyyən Yox |

> **Nəticə:** Ortaq xəta paylanmasının necə olmasından asılı olmayaraq, bütün ssenarilərdə $p \ge 0.2500 \gg 0.05$'dir. İki model arasında heç bir statistik üstünlük yoxdur.

---

## 6. "Ekvivalentlik Sübutu" Yanılgısı və TOST (Two One-Sided Tests) Analizi

Klassik fərziyyə yoxlamasında (NHST) $p > 0.05$ çıxması *"iki modelin ekvivalent olduğunu sübut etməz"*. Sadəcə sıfır fərziyyəni rədd etmək üçün sübutun kifayət etmədiyini göstərir (absence of evidence $\neq$ evidence of absence).

1. **Seçmənin Gücü (Statistical Power):** $n=244$ seçmə həcmində $1.2\%$ fərqi aşkar etmək üçün testin gücü $1 - \beta < 20\%$-dir (yəni test çox zəifdir).
2. **Ekvivalentlik Hüdudları (TOST):** İki modelin ekvivalentliyini elmi şəkildə iddia etmək üçün əvvəlcədən $\pm \Delta_{eq}$ marjası (məsələn $\pm 2.0\%$) təyin edilməli və fərqin 95% güvən aralığı bu marjanın daxilinə düşməlidir.
3. **Mövcud Vəziyyət:** Fərqin 95% güvən aralığı **$[-4.51\%, +2.05\%]$** təşkil edir. Bu interval $\pm 2.0\%$ marjasını aşır.
4. **Metodoloji Düzəliş:** Makalədə *"V2 Tesseract-ı geridə qoydu"* və ya *"V2 Tesseract ilə tam ekvivalentdir"* yazılmamalıdır. Düzgün elmi cümlə:  
   > *"V2 və Tesseract 5 modellərinin böyük hərf xüsusi diakritiklərindəki 3 xarakterlik fərqi statistik cəhətdən sıfırdan fərqlənmir ($p \ge 0.25$, McNemar testi); lakin $n=244$ kiçik seçmə həcmi ekvivalentliyi qəti şəkildə elan etmək üçün kifayət qədər gücə malik deyil."*

---

## 7. Çıxış Faylları

- **Detallı Nəticələr JSON:** `results/speed_benchmark/cpu_speed_benchmark_results.json`
- **Sürət və Statistik Hesabat:** `results/speed_benchmark/cpu_speed_benchmark_report.md`
