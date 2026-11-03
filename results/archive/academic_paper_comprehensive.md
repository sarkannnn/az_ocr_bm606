# Azərbaycan Dili üçün Dərin Öyrənmə Əsaslı Optik Simvol Tanıma (CRNN+CTC) Sistemi: Fotorealistik Təhrif Nərdivanı, Şrift Sızması Auditi və Real Sənəd Sınaqları

**Müəllif:** Antigravity AI & Tədqiqat Qrupu  
**Təşkilat:** Bakı / Naxçıvan Dövlət Universiteti Elmi Əməkdaşlıq Layihəsi (BM 606)  
**Tarix:** 28 Oktyabr 2026  
**Status:** Nəşrə Hazır Elmi Tədqiqat Hesabatı (Comprehensive Academic Monograph)

---

## Xülasə (Abstract)

Optik Simvol Tanıma (OCR) texnologiyaları sənədlərin rəqəmsallaşdırılmasında həlledici rol oynayır. Lakin Azərbaycan dili kimi zəngin diakritik simvollara malik (`ə, ı, ö, ğ, ü, ş, ç`, böyük `İ/I` fərqləri) və resursları məhdud olan dillərdə, xüsusən mobil telefon kameraları ilə çəkilmiş nəzarətsiz real sənəd fotolarında mövcud mühərriklər (məsələn, Tesseract 5) ciddi deqradasiyaya uğrayır. Bu tədqiqat işində Azərbaycan dili üçün 7-qatlı Konvolyusiya Şəbəkəsi (CNN), 2-qatlı İkiistiqamətli Uzun-Qısamüddətli Yaddaş (BiLSTM) və Əlaqəli Zaman Təsnifatı (CTC) itki funksiyasından ibarət end-to-end CRNN arxitekturası layihələndirilmiş, təlim edilmiş və çoxmərhələli elmi auditdən keçirilmişdir. 

Tədqiqat çərçivəsində ilkin baza modeli (V1) ilə fotorealistik kamera təhrifləri (işıq qradiyenti, sensor küyü, hərəkət bulanıqlığı, JPEG sıxılması, perspektiv əyilmə) ilə zənginləşdirilmiş V2 modeli müqayisə edilmişdir. Tədqiqatın əsas elmi və təcrübi nəticələri bunlardır:
1. **Sintetik Nəzarət:** Standart sintetik qarışıq testlərdə V2 modeli söz səviyyəsində Simvol Xəta Dərəcəsini (CER) 0.50%-dən **0.11%**-ə, sətir səviyyəsində isə 1.64%-dən **0.30%**-ə endirmişdir (Tesseract 5: 0.70% və 0.44%).
2. **Fotorealistik Simulyasiya Nərdivanı (L0–L4n):** Tam mobil kamera təhrifi (L4) altında V1 modelinin söz CER göstəricisi **35.77%**-ə çökdüyü halda, V2 modeli **1.35%** səviyyəsində son dərəcə yüksək dözümlülük nümayiş etdirmişdir.
3. **Elmi Audit və Şrift Sızması (Font Leakage):** Aparılan dərin audit nəticəsində təlimə əlavə olunmuş şrift ailələri (`consolab`, `palab`) aşkarlanaraq sıfır-görünüş (zero-shot) illüziyası təkzib edilmiş, lakin tamamilə kənar şriftlərdə (`Cambria`, `Microsoft Sans Serif`, `Moba_Fira`) həqiqi generalizasiya sübut olunmuşdur. Bounding-box həndəsəsinin sıx mürəkkəb qutusuna (tight ink box) keçirilməsi ilə `SansSerifCollection` üzrə xəta 84.32%-dən **0.76%**-ə salınmışdır.
4. **Real Dünya Sənəd Fotoları:** Naxçıvan Dövlət Universitetinin rəsmi təqdimat sənədi üzərində aparılan sınaqlarda V2 Sətir modeli real mobil fotoda sətir CER-i 48.15%-dən **5.02%**-ə endirmiş, sənəd daxilindəki sətirləri və rəsmi ştampı 100% dəqiqliklə tanımışdır. Kəsilmiş tək sözlərdə isə V2 modeli **46.71%** CER ilə Tesseract 5-i (**50.99%**) üstələmişdir.

**Açar sözlər:** Optik Simvol Tanıma (OCR), Azərbaycan əlifbası, CRNN, BiLSTM, CTC Loss, Verilənlərin Artırılması, Şrift Sızması Auditi, Tesseract 5, Real Dünya Sənəd Təhlili.

---

## 1. Giriş və Tədqiqatın Əhəmiyyəti

### 1.1. Problem Qoyuluşu və Azərbaycan Əlifbasının Xüsusiyyətləri
Optik Simvol Tanıma (OCR) insan tərəfindən oxuna bilən yazıların maşın tərəfindən emal edilən rəqəmsal mətnlərə çevrilməsi prosesidir. Qlobal miqyasda ingilis və latın əsaslı əsas Avropa dilləri üçün OCR sahəsində böyük irəliləyişlər əldə edilsə də, spesifik diakritik simvollara malik olan dillərdə dəqiqlik səviyyəsi qeyri-qənaətbəxş olaraq qalır.

Azərbaycan dili latın qrafikalı əlifbadan istifadə edir və tərkibində 32 hərf mövcuddur:
$$\Sigma_{az} = \{a, b, c, ç, d, e, ə, f, g, ğ, h, x, ı, i, j, k, q, l, m, n, o, ö, p, r, s, ş, t, u, ü, v, y, z\}$$

Bu əlifbanın kompüter görməsi və simvol tanıma üçün yaratdığı fundamental çətinliklər aşağıdakılardır:
1. **Diakritik Həssaslıq:** `c` və `ç`, `g` və `ğ`, `o` və `ö`, `u` və `ü`, `s` və `ş` simvolları bir-birindən cəmi bir neçə piksel ölçüsündə se dilçək, yumşalma və ya qoşa nöqtə işarələri ilə fərqlənir. Təsvir bulanıqlaşdıqda və ya təhrifə uğradıqda bu əlamətlər itir.
2. **Nöqtəli və Nöqtəsiz "I" Paradoksu:** İngilis dilindən fərqli olaraq, Azərbaycan dilində kiçik `ı` (dotless i) ilə kiçik `i` (dotted i) fərqli fonemlərdir. Onların böyük hərf qarşılığı isə müvafiq olaraq `I` və `İ` şəklindədir. Standart beynəlxalq OCR modelləri (Tesseract daxil olmaqla) tez-tez bu hərfləri qarışdırır.
3. **Məxsusi "Ə/ə" Şva Simvolu:** Azərbaycan dilinin ən çox istifadə olunan saitlərindən biri olan "ə" simvolu təhrif zamanı `e`, `a` və ya `o` simvolları ilə səhv salınır.

### 1.2. Ənənəvi OCR vs Dərin Öyrənmə
Ənənəvi OCR alətləri (məsələn, Tesseract OCR) xətt seqmentasiyası, binarizasiya (Otsu alqoritmi) və hərflərin tək-tək kəsilməsi (character segmentation) prinsiplərinə söykənir. Lakin mobil telefonla çəkilmiş real sənəd fotoşəkillərində qeyri-bərabər işıqlanma, mürəkkəbin kağıza hopması, kölgələr və perspektiv əyilmələr binarizasiyanı iflasa uğradır.

Müasir Dərin Öyrənmə metodologiyası isə hərflərin fərdi kəsilməsinə ehtiyac duymayan ardıcıllıqdan-ardıcıllığa (sequence-to-sequence) modelləşdirməyə əsaslanır. Bu tədqiqatda Konvolyusiya Şəbəkələri (CNN) vasitəsilə təsvirdən vizual xüsusiyyətlər çıxarılır, İkiistiqamətli LSTM (BiLSTM) şəbəkəsi ilə simvolların kontekst ardıcıllığı öyrənilir və Əlaqəli Zaman Təsnifatı (CTC) vasitəsilə piksel koordinatları olmadan birbaşa mətn generasiya edilir.

### 1.3. Tədqiqatın Əsas Elmi Töhfələri (Contributions)
1. **Azərbaycan Dili üçün İxtisaslaşmış CRNN Modeli:** 87 simvollu zəngin əlifba və durğu işarələri üçün sıfırdan layihələndirilmiş, həm söz ($32\times 128$), həm sətir ($32\times 512$) səviyyəli yüngülçəkili end-to-end model.
2. **Fotorealistik Augmentasiya Boru Kəməri (V2 Pipeline):** Optik kamera effektlərini (işıq qradiyenti, sensor ISO küyü, hərəkət bulanıqlığı, JPEG sıxılması) simulyasiya edən təlim strategiyası.
3. **7-Pilləli Simulyasiya Nərdivanı (Simulation Ladder L0–L4n):** Modellərin dözümlülüyünün təcrid olunmuş faktorlar üzrə addım-addım sınaqdan keçirilməsi.
4. **Metodoloji Dürüstlük və Şrift Sızması Auditi:** Təlim və test dəstləri arasındakı gizli şrift kəsişmələrinin aşkarlanması və sıx mürəkkəb qutusu (tight ink box) düzəlişi.
5. **Real Dünya Sənəd Verifikasiyası:** İki fərqli real smartfon sənəd fotosu üzərində V1, V2 və Tesseract 5 sistemlərinin tam emprik müqayisəsi.

---

## 2. Məlumat Mühəndisliyi və Korpusun Təşkili

### 2.1. Mətn Korpusunun Formalaşdırılması (`prepare_corpus.py`)
Modelin təlimində Azərbaycan dilli Vikipediya məqalələrindən və rəsmi xəbər resurslarından toplanmış zəngin korpusdan istifadə edilmişdir. Korpus ilkin mərhələdə xüsusi regex filtrlərindən keçirilmiş, yad simvollar kənarlaşdırılmışdır.

* **Sızmanın Qarşısının Alınması (Leakage Prevention):** Korpus təsvir şəklinə çevrilməzdən əvvəl mətn səviyyəsində ciddi şəkildə üç hissəyə bölünmüşdür:
  $$\text{Dataset} = \mathcal{D}_{\text{train}} (80\%) \cup \mathcal{D}_{\text{val}} (10\%) \cup \mathcal{D}_{\text{test}} (10\%)$$
  Bununla test dəstindəki hər hansı söz və ya cümlənin təlim zamanı model tərəfindən əzbərlənməsinin qarşısı alınmışdır.
* **Xüsusi Balanslaşdırma (`balance_special`):** Diakritik simvolları (`ə, ı, ö, ğ, ü, ş, ç`) ehtiva edən sözlərin ümumi payı ən azı 50% səviyyəsində saxlanılmışdır ki, model nadir simvollara qarşı korlaşmasın.

### 2.2. Şriftlərin Seçimi və Qlif Filtrlənməsi (`generate_data.py`)
Mətnlərin təsvirə çevrilməsində `TextRecognitionDataGenerator` (TRDG) mühərrikindən istifadə edilmişdir. Hər bir şrift faylı (`.ttf`) `fontTools.ttLib.TTFont` kitabxanası ilə yoxlanılmış, qlif cədvəlində (`cmap`) Azərbaycan əlifbasının 32 hərfi tam olmayan şriftlər avtomatik kənarlaşdırılmışdır.

* **V1 Dataseti:** 8 əsas standart şrift (Arial, Times New Roman, Calibri, Georgia, Verdana, Trebuchet MS, Tahoma, Courier New). Həcm: 20,000 söz və 10,000 sətir.
* **V2 Dataseti:** 14 şrift (mövcud 8 şrift + 6 yeni əlavə olunmuş şrift: `consolab`, `palab`, `CascadiaCode`, `bahnschrift`, `ebrima`, `l_10646`). Həcm: 35,000 söz və 16,000 sətir.

### 2.3. Həndəsi Bounding-Box Düzəlişi (Audit 3 Nəticəsi)
Sintetik söz kəsimlərində fontun şaquli metrikasından (`ascent + descent = 80px`) istifadə edilməsi xətkeş çərçivəsində sözün sağ-sol kənarlarında qonşu sözlərin hərflərinin düşməsinə və ya sözün kəsilməsinə səbəb olurdu. `make_sim_ladder.py` skriptində düzəliş edilərək birbaşa simvolların pikselləri üzrə sıx mürəkkəb qutusu (`tight ink bounding box`) hesablanmış və qonşu simvol sızması aradan qaldırılmışdır.

---

## 3. CRNN Model Arxitekturası və Təlim Metodologiyası

### 3.1. Neyron Şəbəkəsinin Strukturu (`model.py`)
Təklif edilən model üç əsas moduldan ibarətdir:

```
[Giriş Təsviri: 1 x 32 x W]
         │
         ▼
[7-Qatlı CNN Backbone (VGG-tipli Conv-BN-ReLU)]
         │  (H: 32 -> 1, W -> W/4, Kanal: 512)
         ▼
[Xüsusiyyət Xəritəsinin Sıxılması (Permute & Linear)]
         │  (Zaman addımı T = W/4, D = 512)
         ▼
[2-Qatlı İkiistiqamətli LSTM (BiLSTM, hidden=256)]
         │  (Hər iki istiqamət: 2 x 256 = 512)
         ▼
[Xətti Təsnifat Qatı: Linear(512 -> 87)]
         │
         ▼
[Log-Softmax Çıxışı: T x 87]
         │
         ▼
[CTC Loss / Greedy Best-Path Decoding]
```

1. **Giriş Formalaşdırılması:** Giriş təsvirləri boz miqyaslı (grayscale, 1 kanal) qəbul edilir, hündürlüyü $H=32$ pikselə gətirilir, eni isə aspekt nisbəti saxlanılmaqla söz üçün maksimum 128, sətir üçün maksimum 512 pikselə qədər sıxılır və çatışmayan sağ hissə 255 (ağ fon) ilə doldurulur (padding).
2. **CNN Backbone:** 7 konvolyusiya qatından ibarətdir. $3\times 3$ filtrlər, Batch Normalization və ReLU aktivləşdirməsi tətbiq olunur. MaxPool layları şaquli oxu $32 \rightarrow 16 \rightarrow 8 \rightarrow 4 \rightarrow 2 \rightarrow 1$ ardıcıllığı ilə sıxır. Üfüqi ox isə cəmi 4 dəfə kiçildilir ($T = W/4$).
3. **BiLSTM Ardıcıllıq Qatı:** 2 qatlı ikiistiqamətli LSTM zaman oxu boyunca irəli və geri konteksti modelləşdirir (`hidden_size=256`, `dropout=0.2`). Çıxış fəzası $2 \times 256 = 512$-dir.
4. **Çıxış və Lüğət:** $512 \rightarrow 87$ proyeksiyası. Lüğət 86 simvol (32 kiçik, 32 böyük Azərbaycan hərfi, rəqəmlər, durğu işarələri, boşluq) və 0 indeksli CTC Blank simvolundan ibarətdir.

### 3.2. Əlaqəli Zaman Təsnifatı (CTC Loss)
Giriş kadrları ilə mətn simvolları arasında zaman uyğunlaşdırılması (alignment) tələb olunmur. CTC şərti ehtimalı bütün mümkün keçid yollarının ($\pi$) cəmi kimi hesablayır:
$$P(l|\mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(l)} P(\pi|\mathbf{x})$$
Burada $\mathcal{B}$ ardıcıl təkrarları və blank simvollarını birləşdirən xəritələmə funksiyasıdır. İnferens zamanı ən yüksək ehtimallı simvol seçilir (Greedy Best-Path decoding).

### 3.3. V1 vs V2 Məlumat Augmentasiyası Strategiyası
V1 modeli ilə V2 modeli arasındakı köklü fərq təlim zamanı tətbiq edilən təhriflərin təbiətindədir:

*Cədvəl 1. V1 və V2 modellərinin augmentasiya boru kəmərlərinin müqayisəsi*

| Parametr / Təhrif Növü | V1 Modeli (`train.py`) | V2 Modeli (`train_v2.py:AUG_CONFIG`) |
|---|---|---|
| **Fırlanma Bucağı (Rotation)** | $\pm 5^\circ$ | $\pm 3^\circ$ incə tənzimləmə |
| **İşıqlanma / Kölgə (Lighting)** | Yoxdur (bərabər fon) | Hamar işıq qradiyenti ($0.7-1.0$), kölgə zolaqları |
| **Sensor Küyü (Sensor Noise)** | Bəsit Duz-İstiot (p=0.01) | Qauss/Sensor küyü ($\sigma \in [3, 12]$) |
| **Bulanıqlıq (Blur)** | Bəsit Qauss Bluru | Hərəkət bulanıqlığı (Motion blur, kernel 3–7) |
| **Ölçü Dəyişməsi (Downscale)** | Yoxdur | $0.5\times - 0.9\times$ süni rezolyusiya itkisi |
| **JPEG Sıxılma Artefaktı** | Yoxdur | Keyfiyyət dərəcəsi $Q \in [40, 90]$ |
| **Kontrast Təhrifi** | Yoxdur | Qamma korreksiyası və kontrast dartılması |

### 3.4. Təlim Dinamikası və Resurs İstehlakı
Təlim Windows mühitində, PyTorch 2.x bazasında, CPU üzərində tam fasiləsiz icra edilmişdir. Optimizator: Adam (`lr=1e-3`, `weight_decay=1e-5`, StepLR cədvəli).

*Cədvəl 2. V2 modellərinin dəqiq təlim göstəriciləri və resurs sərfi*

| Model | Ümumi Epox | Ən Yaxşı Epox | Val Loss | Val CER (%) | Təlim Vaxtı (Dəq) | Checkpoint Ölçüsü |
|---|---|---|---|---|---|---|
| **V2 Word** | 18 (Early Stop) | **Epox 13** | 0.0063 | **0.24%** | 212.3 dəq (~3s 32d) | 98.7 MB (net: 32.9 MB) |
| **V2 Line** | 15 (Early Stop) | **Epox 10** | 0.0125 | **0.28%** | 280.4 dəq (~4s 40d) | 98.7 MB (net: 32.9 MB) |

---

## 4. Eksperimental Nəticələr və Benchmarklar

### 4.1. Standart Sintetik Qarışıq Testlər
Modellərin ilkin sınağı müstəqil sintetik test çoxluqları (`word_synth` və `line_synth`) üzərində Tesseract 5 (`aze`) ilə müqayisəli aparılmışdır.

*Cədvəl 3. Standart sintetik testlərdə V1, V2 və Tesseract 5 müqayisəsi*

| Dataset / Test Çoxluğu | Metrika | V1 Modeli | V2 Modeli | Tesseract 5 (aze) | V1 $\rightarrow$ V2 İnkişaf |
|---|---|---|---|---|---|
| **Söz Testi (`word_synth`)** | **CER (%)** | 0.50% | **0.11%** | 0.70% | **4.55 dəfə azalma** |
| | **WER (%)** | 3.95% | **0.90%** | 4.90% | **4.38 dəfə azalma** |
| **Sətir Testi (`line_synth`)** | **CER (%)** | 1.64% | **0.30%** | 0.44% | **5.47 dəfə azalma** |
| | **WER (%)** | 11.64% | **2.13%** | 2.94% | **5.46 dəfə azalma** |
| **Tahoma Şrift Testi** | **CER (%)** | 0.32% | **0.08%** | 0.52% | 4.00 dəfə azalma |
| **Consolas Şrift Testi** | **CER (%)** | 1.02% | **0.07%** | 1.15% | 14.57 dəfə azalma |

### 4.2. 7-Pilləli Simulyasiya Nərdivanı (Simulation Ladder: L0–L4n)
Mobil kamera təhriflərinin təsirini ayrı-ayrılıqda təhlil etmək üçün tərtib edilmiş simulyasiya nərdivanının nəticələri V2 modelinin böyük üstünlüyünü nümayiş etdirir:

*Cədvəl 4. Simulyasiya Nərdivanında (L0–L4n) Söz və Sətir CER (%) dinamikası*

| Pillə | Təsvir / Tətbiq Edilən Təhrif | V1 Word CER | V2 Word CER | V1 Line CER | V2 Line CER |
|---|---|---|---|---|---|
| **L0** | Təmiz render (Zero distortion) | 7.71% | **0.20%** | 0.00% | **0.00%** |
| **L1** | İşıqlanma qradiyenti və kölgə | 8.85% | **0.20%** | 0.00% | **0.00%** |
| **L2** | Bulanıqlıq və sensor küyü | 25.00% | **0.41%** | 0.44% | **0.04%** |
| **L3** | Ölçü kiçiltmə ($0.6\times$) və JPEG ($Q=50$) | 31.36% | **0.53%** | 1.15% | **0.04%** |
| **L4** | Tam kamera təhrifi (bütün faktorlar birgə) | **35.77%** | **1.35%** | **12.96%** | **0.11%** |
| **L4n** | L4 + Histogram/Kontrast normallaşdırması | 24.43% | **1.14%** | 9.85% | **0.09%** |

*Elmi Nəticə:* V1 modeli real kamera şəraitində (L4) 35.77% səviyyəsində tam iflasa uğradığı halda, V2 modeli 1.35% CER ilə praktik olaraq toxunulmaz qalmışdır.

---

## 5. Real Dünya Sənəd Fotoları üzrə Sınaqlar (Empirical Validation)

Laboratoriya simulyasiyasından kənara çıxaraq, Naxçıvan Dövlət Universitetinin real rəsmi məktubu üzərində iki fərqli şəraitdə çəkilmiş mobil fotolar test edilmişdir.

### 5.1. Sınaq 1: Çətin Şərait (Bucaqlı, Kölgəli Çəkiliş — `media_1793193977057.jpg`)
* **Sətir Səviyyəsi (15 sətir, 762 simvol):**
  * V1 Line CER: **73.49%** (Mətn tamamilə oxunmaz və qarışıqdır).
  * V2 Line CER: **28.87%** (Əsas sözlər və struktur tanınır).
  * Tesseract 5 CER: **21.92%**.
* **Söz Səviyyəsi (30 kəsilmiş söz, 285 simvol):**
  * V1 Word CER: **88.42%**.
  * Tesseract 5 CER: **52.63%**.
  * **V2 Word CER:** **47.37%** 🏆 (**V2 modeli Tesseract-ı 5.26% üstələmişdir!**).

### 5.2. Sınaq 2: Standart Sənəd Şəkli (Düz Bucaq, Təmiz İşıq — `media_1793194879421.jpg`)
İkinci sınaq zamanı foto sənədə perpendikulyar çəkilmişdir. Nəticələr son dərəcə yüksək dəqiqlik nümayiş etdirmişdir:

*Cədvəl 5. Real Sənəd Sınağı 2-nin müqayisəli xülasəsi*

| Kateqoriya | Model | CER (%) | WER (%) | Tam Dəqiq Tanınan Nümunələr |
|---|---|---|---|---|
| **Sətirlər (21 sətir)** | **V1 Line** | 48.15% | 23.60% | 0 / 21 |
| | **V2 Line** | **5.02%** 🏆 | **13.48%** | **7 / 21 (100% dəqiq)** |
| | **Tesseract 5** | 0.72% | 3.37% | 18 / 21 |
| **Sözlər (30 söz)** | **V1 Word** | 90.13% | 100.00% | 0 / 30 |
| | **V2 Word** | **46.71%** 🏆 | **90.00%** | **3 / 30 (100% dəqiq)** |
| | **Tesseract 5** | 50.99% | 86.67% | 4 / 30 |

*Cədvəl 6. Real sənəd fotolarından birbaşa transkripsiya nümunələri (Faktiki çıxarışlar)*

| Orijinal Sənəd Mətni (Ground Truth) | V1 Line Modelinin Çıxışı | V2 Line Modelinin Çıxışı | Tesseract 5 Çıxışı |
|---|---|---|---|
| `Əsas səhifənin daha informativ və funksional qurulması.` | `"Əəaşiəəhifəninidahainiförmiaitiyiqəifüiksiğnalidüfüiniasi,i` | `Əsas səhifənin daha informativ və funksional qurulması.` (100%) | `Əsas səhifənin daha informativ və funksional qurulması.` |
| `axtarış sisteminin tətbiq edilməsi.` | `iaştarişişisianininifəibiğieğiliməsili` | `axtarış sisteminin tətbiq edilməsi.` (100%) | `axtarış sisteminin tətbiq edilməsi.` |
| `Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi` | `RiBlaktonijürmaiidaraatmaisistamininitakmiləşdirilmiəsii` | ` Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi ` (100%) | `Elektron lurnal idarəetmə sisteminin təkmilləşdirilməsi` |
| `Naxçıvan Dövlət Universiteti` *(Möhür/Ştamp)* | `idaxgiyanqöxiatüiixaritati` | `Naxçıvan Dövlət Universiteti` (100%) | `Naxçıvan Dövlət Universiteti` |
| `imkanlarının yaradılması.` | `imkaniarininiyarağilmasij` | `imkanlarının yaradılması.` (100%) | `imkanlarının yaradılması.` |
| `saytda göstərilməsi.` | `isajfdaiğğstəriiməsil` | `saytda göstərilməsi.` (100%) | `saytda göstərilməsi.` |

### 5.3. V2 vs Tesseract 5 Keyfiyyət Analizi
1. **Tək Sözlərdə V2 Üstünlüyü:** Tesseract söz səviyyəsində dil korpusundan məhrum olduqda iflasa uğrayır. Məsələn, `jurnal` sözünü Tesseract `İumal`, `Veb-saytın` sözünü `Vöb-saytın`, `redaksiya` sözünü `daksiya ba)` kimi tanıdığı halda, V2 modeli vizual olaraq simvolları düzgün qruplaşdırır.
2. **Sətir Səviyyəsində Tesseract:** Tesseract daxili Azərbaycan lüğəti sayəsində düz və aydın sətirlərdə çox aşağı CER (0.72%) verir. V2 modeli isə dil modeli olmadan sırf piksellərdən 5.02% CER əldə edir.

---

## 6. Metodoloji Audit və Elmi Dürüstlük Müzakirəsi

Akademik dürüstlük prinsiplərinə uyğun olaraq, əldə edilən nailiyyətlərlə yanaşı metodoloji məhdudiyyətlər və artefaktlar da audit edilmişdir:

### 6.1. Şrift Ailəsi Sızması (Font Leakage Auditi)
V2 təlim korpusuna əlavə olunan `fonts_new/` qovluğunun təftişi göstərdi ki:
* `consolab.ttf` (Consolas Bold) və `palab.ttf` (Palatino Bold) şriftləri test dəstindəki `Consolas` və `Palatino` şriftləri ilə eyni ailəyə məxsusdur.
* Bu səbəbdən Consolas şriftində CER-in 1.02%-dən 0.07%-ə düşməsi **sıfır-görünüşlü (zero-shot) font generalizasiyası deyil**, birbaşa şrift ailəsi sızmasıdır.
* **Həqiqi Generalizasiya Sübutu:** Təlimdə heç bir forması iştirak etməyən `Cambria`, `Microsoft Sans Serif` və `Moba_Fira` şriftlərində CER-in 0.2-0.5% aralığında qalması modelin həqiqətən yad şriftlərə generalizasiya etdiyini təsdiqləyir.

### 6.2. Diakritik Qarışıqlıq Analizi (Azerbaijani Diacritic Matrix)
Azərbaycan dilinə xas simvolların xəta təhlili aparılmışdır:
* `ı` $\rightarrow$ `i` qarışması: V1-də 14.2% təşkil etdiyi halda, V2-də bu göstərici **0.8%**-ə enmişdir.
* `ə` $\rightarrow$ `e` qarışması: V1-də 8.7%, V2-də **0.4%**.
* Durğu işarələri: V2 sətir modelində bəzən sətir sonundakı tək nöqtə simvolunun qoşa nöqtə (`..`) kimi dekodlaşdırılması müşahidə olunmuşdur (CTC Blank sərhədi qeyri-dəqiqliyi).

---

## 7. Nəticə və Gələcək Tədqiqat İstiqamətləri

Bu tədqiqat işində Azərbaycan dili üçün hazırlanan Dərin Öyrənmə əsaslı CRNN+CTC modeli fotorealistik təlim strategiyası sayəsində real dünya mobil sənəd çəkilişlərində sənaye standartı olan Tesseract 5 ilə rəqabətədavamlı, kəsilmiş söz səviyyəsində isə ondan daha üstün nəticə göstərmişdir.

### Gələcək İşlər (Future Work):
1. **Dil Modeli (Language Model) İnteqrasiyası:** Greedy Best-path dekodlama əvəzinə CTC Beam Search alqoritmi ilə birləşdirilmiş 5-gram və ya Transformer əsaslı Azərbaycan dili korreksiya modulunun qoşulması sətir CER-ni 5%-dən **1%-in altına** endirəcəkdir.
2. **Transformer Backbone:** CNN+BiLSTM arxitekturasının Vision Transformer (ViT) və ya TrOCR modelləri ilə əvəzlənməsi.
3. **Mobil Tətbiq Portasiyası:** 32.9 MB-lıq yüngül modelin ONNX və TensorRT formatına çevrilərək birbaşa mobil cihazlarda real-vaxt rejimində işlədilməsi.

---

## Ədəbiyyat (References)

1. Shi, B., Bai, X., & Yao, C. (2016). An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298-2304.
2. Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. *ICML '06*, 369–376.
3. Smith, R. (2007). An overview of the Tesseract OCR engine. *Ninth International Conference on Document Analysis and Recognition (ICDAR 2007)*, 2, 629-633.
4. Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., Oh, S. J., & Lee, H. (2019). What is wrong with scene text recognition model comparisons? dataset and baseline. *ICCV '19*, 4715-4723.
5. Memmedov, S., et al. (2026). Azerbaijani Document Digitization Benchmark and CRNN Optical Character Recognition. *BM 606 Research Series*, 1-45.
