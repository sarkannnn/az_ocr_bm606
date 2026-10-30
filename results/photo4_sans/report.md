# Photo 4 (Sans-serif / Arial) OCR Qiymətləndirmə Hesabatı

Bu hesabat istifadəçi tərəfindən təqdim edilmiş 4-cü sənəd fotosu (Sans-serif / Arial şrifti ilə çap olunmuş mətn) üzərində dondurulmuş boru xəttinin (`pipeline`) icra nəticələrini əks etdirir.

## 1. Ümumi Metriklər Cədvəli (23 Sətir, 1 006 Simvol)

| Model və Rejim | CER (%) | WER (%) | Dəqiq Sətirlər (Exact) | Simvol Xətası | Söz Xətası | Boşluq Xətası |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Tesseract 5 (Tam Sətir) | 0.00% | 0.00% | 23/23 | 0 | 0 | 0 |
| Tesseract 5 (Qayda Əsaslı Parçalama) | 0.60% | 3.65% | 20/23 | 6 | 5 | 2 |
| **V2 Line (Qayda Əsaslı Parçalama)** | 0.60% | 3.65% | **20/23** | 6 | 5 | 2 |
| V2 Line (Tam Sətir) | 0.10% | 0.73% | 22/23 | 1 | 1 | 0 |
| V1 Line (Qayda Əsaslı Parçalama) | 5.67% | 38.69% | 4/23 | 57 | 53 | 23 |
| V1 Line (Tam Sətir) | 7.06% | 42.34% | 2/23 | 71 | 58 | 20 |

## 2. Bootstrap 95% Etibarlılıq İntervalı (V2 - Tesseract)

- **$\Delta$CER:** +0.00% (95% CI: [-0.29%, 0.29%])
- **$\Delta$WER:** +0.00% (95% CI: [-2.16%, 2.14%])

## 3. Qarşılıqlı Müstəsna Xəta Taksonomiyası (V2 Line və Tesseract)

| Xəta Kateqoriyası | V2 Line Sayı | Tesseract Sayı | İzah |
|---|:---:|:---:|---|
| **1. Başlanğıc Kənar Əlavəsi (Leading)** | **0** | **0** | 0 ədəd (Artıq dırnaq `"` əlavələri tam yoxdur) |
| **2. Sonluq Kənar Əlavəsi (Trailing)** | **0** | **0** | Sətir sonu artıq simvol |
| **3. Daxili Boşluq Xətası** | **2** | **2** | Boşluq əməliyyatları |
| **4. Daxili 0 / O Qarışıqlığı** | **0** | **0** | Rəqəm `0` və hərf `O` |
| **5. Daxili Durğu Xətası** | **3** | **4** | Nöqtə, vergül, defis |
| **6. Daxili Hərf Xətası** | **1** | **0** | Qlif tanınma fərqi |
| **CƏM (Mutually Exclusive)** | **6** | **6** | $100\%$ Levenshtein cəminə bərabərdir |

## 4. Bütün 23 Sətir Üzrə Müfəssəl Nəticələr

### Sətir 01 (Ölçü: 528x31 px, Parça: 2)
- **Orijinal (Çap olunmuş ref):** `Zamanın süzgəcindən keçmiş qədim sənət növləri`
- **V2 Line (Parçalanmış):** `Zamanın süzgəcindər I keçmiş qədim sənət növləri` (FƏRQ VAR)
- **Tesseract 5:** `Zamanın süzgəcindər ) keçmiş qədim sənət növləri` (FƏRQ VAR)
- **V1 Line (Parçalanmış):** `Bamaninısüzgəcindər "keçmiş qədim sənət növləril`

### Sətir 02 (Ölçü: 499x31 px, Parça: 2)
- **Orijinal (Çap olunmuş ref):** `hər bir xalqın tarixi yaddaşını və mədəniyyətini`
- **V2 Line (Parçalanmış):** `hər bir xalqın tarixi yaddaşını və mədəniyyətini` (DƏQİQ DÜZ)
- **Tesseract 5:** `hər bir xalqın tarixi yaddaşın və mədəniyyətini` (FƏRQ VAR)
- **V1 Line (Parçalanmış):** `ihəribir"xalqın tarixi yaddaşını (və mədəniyyətini`

### Sətir 03 (Ölçü: 446x31 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `özündə yaşadır. Azərbaycanda xalçaçılıq,`
- **V2 Line (Parçalanmış):** `özündə yaşadır. Azərbaycanda xalçaçılıq,` (DƏQİQ DÜZ)
- **Tesseract 5:** `özündə yaşadır. Azərbaycanda xalçaçılıq,` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `özündəiyaşadır: Azərbaycanda xalçaçılıq,)`

### Sətir 04 (Ölçü: 464x31 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `dulusçuluq, şəbəkə və misgərlik kimi sənət`
- **V2 Line (Parçalanmış):** `dulusçuluq, şəbəkə və misgərlik kimi sənət` (DƏQİQ DÜZ)
- **Tesseract 5:** `dulusçuluq, şəbəkə və misgərlik kimi sənət` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `dülüsçülüq; işəbəkə və misgərlik kimi sənət`

### Sətir 05 (Ölçü: 520x30 px, Parça: 2)
- **Orijinal (Çap olunmuş ref):** `sahələri nəsildən-nəslə ötürülərək bu günümüzə`
- **V2 Line (Parçalanmış):** `sahələri nəsildən. nəslə ötürülərək bu günümüzə` (FƏRQ VAR)
- **Tesseract 5:** `sahələri nəsildən: nəslə ötürülərək bu günümüzə` (FƏRQ VAR)
- **V1 Line (Parçalanmış):** `sahələriinəsildəni inəslə ötürülərək bu günümüzə`

### Sətir 06 (Ölçü: 525x33 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə`
- **V2 Line (Parçalanmış):** `qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə` (DƏQİQ DÜZ)
- **Tesseract 5:** `qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `qədərigəlibiçatmışdır: Hər bir naxış, hər bir ilmə`

### Sətir 07 (Ölçü: 499x33 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `bir hekayə danışır, usta əllərinin zəhmətini və`
- **V2 Line (Parçalanmış):** `bir hekayə danışır, usta əllərinin zəhmətini və` (DƏQİQ DÜZ)
- **Tesseract 5:** `bir hekayə danışır, usta əllərinin zəhmətini və` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `birihekayə danışır, üsta əllərinin zəhmətini və`

### Sətir 08 (Ölçü: 211x27 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `ruhunu əks etdirir.`
- **V2 Line (Parçalanmış):** `ruhunu əks etdirir.` (DƏQİQ DÜZ)
- **Tesseract 5:** `ruhunu əks etdirir.` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `irühünüiəks etdirir.`

### Sətir 09 (Ölçü: 499x34 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `Bu sənət inciləri təkcə gözəllik deyil, həm də`
- **V2 Line (Parçalanmış):** `Bu sənət inciləri təkcə gözəllik deyil, həm də` (DƏQİQ DÜZ)
- **Tesseract 5:** `Bu sənət inciləri təkcə gözəllik deyil, həm də` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `Buisənəti inciləri təkcə gözəllik deyil, həm də`

### Sətir 10 (Ölçü: 520x35 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `xalqımızın milli kimliyinin göstəricisidir. Onları`
- **V2 Line (Parçalanmış):** `xalqımızın milli kimliyinin göstəricisidir. Onları` (DƏQİQ DÜZ)
- **Tesseract 5:** `xalqımızın milli kimliyinin göstəricisidir. Onları` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `xalqımızınımilli kimliyinin göstəricisidir. Onları`

### Sətir 11 (Ölçü: 551x36 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək`
- **V2 Line (Parçalanmış):** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək` (DƏQİQ DÜZ)
- **Tesseract 5:** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `qorümaq; yaşatmaq və gələcək nəsillərə ötürmək`

### Sətir 12 (Ölçü: 523x34 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `hər birimizin mənəvi borcudur. Çünki tarixinə`
- **V2 Line (Parçalanmış):** `hər birimizin mənəvi borcudur. Çünki tarixinə` (DƏQİQ DÜZ)
- **Tesseract 5:** `hər birimizin mənəvi borcudur. Çünki tarixinə` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `hər"birimizin mənəvi borcudur: Çünki tarixinə`

### Sətir 13 (Ölçü: 464x37 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `sahib çıxan millətlər öz gələcəyini daha`
- **V2 Line (Parçalanmış):** `sahib çıxan millətlər öz gələcəyini daha` (DƏQİQ DÜZ)
- **Tesseract 5:** `sahib çıxan millətlər öz gələcəyini daha` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `sahib"çıxan millətlər öz gələcəyini daha`

### Sətir 14 (Ölçü: 550x35 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `möhkəm təməllər üzərində qura bilirlər. Hər bir`
- **V2 Line (Parçalanmış):** `möhkəm təməllər üzərində qura bilirlər. Hər bir` (DƏQİQ DÜZ)
- **Tesseract 5:** `möhkəm təməllər üzərində qura bilirlər. Hər bir` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `möhkəm təməllər üzərində qura bilirlər. Hər bir`

### Sətir 15 (Ölçü: 562x39 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `qədim əşya keçmişimizlə bu günümüz arasında`
- **V2 Line (Parçalanmış):** `qədim əşya keçmişimizlə bu günümüz arasında` (DƏQİQ DÜZ)
- **Tesseract 5:** `qədim əşya keçmişimizlə bu günümüz arasında` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `qədim əşya keçmişimizlə bu günümüz arasında`

### Sətir 16 (Ölçü: 287x38 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `görünməz bir körpüdür.`
- **V2 Line (Parçalanmış):** `görünməz bir körpüdür.` (DƏQİQ DÜZ)
- **Tesseract 5:** `görünməz bir körpüdür.` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `görünməz bir körpüdür.`

### Sətir 17 (Ölçü: 556x39 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `Müasir dünyada texnologiyanın sürətli inkişafına`
- **V2 Line (Parçalanmış):** `Müasir dünyada texnologiyanın sürətli inkişafına` (DƏQİQ DÜZ)
- **Tesseract 5:** `Müasir dünyada texnologiyanın sürətli inkişafına` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `Müasir dünyada texnologiyanın sürətli inkişafınal`

### Sətir 18 (Ölçü: 587x40 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri`
- **V2 Line (Parçalanmış):** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri` (DƏQİQ DÜZ)
- **Tesseract 5:** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri`

### Sətir 19 (Ölçü: 595x42 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `heç vaxt azalmır. Əksinə, insan ruhunun hpduğu bu`
- **V2 Line (Parçalanmış):** `heç vaxt azalmır, Əksinə, insan ruhunun hpduğu bu` (FƏRQ VAR)
- **Tesseract 5:** `heç vaxt azalmır. Əksinə, insan ruhunun hpduğu bu` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `heçvaxt azalmır, Əksinə, insan ruhunun hpduğu.bu`

### Sətir 20 (Ölçü: 593x36 px, Parça: 2)
- **Orijinal (Çap olunmuş ref):** `sənət nümunələrinə olan maraq günü-gündən artır.`
- **V2 Line (Parçalanmış):** `sənət nümunələrinə olan maraq günü-gündən artır.` (DƏQİQ DÜZ)
- **Tesseract 5:** `sənət nümunələrinə olan maraq günü-gündən artır.` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `sənətinümunələrinə olan maraq günü-gündən artır.`

### Sətir 21 (Ölçü: 600x38 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim`
- **V2 Line (Parçalanmış):** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim` (DƏQİQ DÜZ)
- **Tesseract 5:** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `Gəncinəslin bu sənətlərə maraq göstərməsi, qədiml`

### Sətir 22 (Ölçü: 580x40 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `ənənələrin müasir dizaynlarla birləşdirilməsi bu`
- **V2 Line (Parçalanmış):** `ənənələrin müasir dizaynlarla birləşdirilməsi bu` (DƏQİQ DÜZ)
- **Tesseract 5:** `ənənələrin müasir dizaynlarla birləşdirilməsi bu` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `Ənənələrinimüasir dizaynlarla birləşdirilməsi bül`

### Sətir 23 (Ölçü: 445x43 px, Parça: 1)
- **Orijinal (Çap olunmuş ref):** `irsin gələcəyini daha da parlaq edir.`
- **V2 Line (Parçalanmış):** `irsin gələcəyini daha da parlaq edir.` (DƏQİQ DÜZ)
- **Tesseract 5:** `irsin gələcəyini daha da parlaq edir.` (DƏQİQ DÜZ)
- **V1 Line (Parçalanmış):** `irsinigələcəyini dahadaparlaq.edir:`
