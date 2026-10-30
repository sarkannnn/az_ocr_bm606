# Photo 3 (Yeni Sənəd) OCR Qiymətləndirmə Hesabatı

Bu hesabat istifadəçi tərəfindən təqdim edilən 3-cü real çap olunmuş sənəd fotosu üzərində dondurulmuş OCR boru xəttinin (`pipeline`) icra nəticələrini əks etdirir.

## 1. Ümumi Metriklər Cədvəli

| Model & Rejim | CER (%) | WER (%) | Dəqiq Sətirlər (Exact) | Səhv Hərf/Simvol Sayı | Səhv Söz Sayı |
|---|---|---|---|---|---|
| Tesseract 5 (Tam Sətir) | 0.00% | 0.00% | 23/23 | 0 | 0 |
| Tesseract 5 (Qayda Əsaslı Chunk) | 0.00% | 0.00% | 23/23 | 0 | 0 |
| **V2 Line (Qayda Əsaslı Chunk)** | 0.10% | 0.72% | **22/23** | 1 | 1 |
| V2 Line (Tam Sətir) | 0.20% | 1.45% | 21/23 | 2 | 2 |
| V1 Line (Qayda Əsaslı Chunk) | 4.63% | 32.61% | 4/23 | 47 | 45 |
| V1 Line (Tam Sətir) | 7.39% | 39.86% | 1/23 | 75 | 55 |

## 2. Bootstrap 95% Etibarlılıq İntervalları (V2 - Tesseract)

- **Qayda Əsaslı Bölünmə (Chunking):**
  - $\Delta$CER 95% CI: [0.00%, 0.30%]
  - $\Delta$WER 95% CI: [0.00%, 2.16%]
- **Tam Sətir (Full Line):**
  - $\Delta$CER 95% CI: [0.00%, 0.50%]
  - $\Delta$WER 95% CI: [0.00%, 3.62%]

## 3. V2 Xəta Taksonomiyası (Mutually Exclusive)

- **Qayda Əsaslı V2 Xətaları cəmi:** 1 ədəd
  - Sətir 6: Tip: `del`, Kateqoriya: `Punctuation`, Ref simvol: '.', Model simvol: ''

- **Tam Sətir V2 Xətaları cəmi:** 2 ədəd
  - Sətir 8: Tip: `ins`, Kateqoriya: `Punctuation`, Ref simvol: '', Model simvol: '.'
  - Sətir 19: Tip: `sub`, Kateqoriya: `Punctuation`, Ref simvol: '.', Model simvol: ':'

## 4. Sətirbəsətir Ətraflı Nəticələr (23 Sətir)

### Sətir 01 (Ölçü: 525x32 px, Parça sayı: 2)
- **Orijinal (Çap olunmuş mətndəki ref):** `Zamanın süzgəcindən keçmiş qədim sənət növləri`
- **V2 (Qayda əsaslı chunk):** `Zamanın süzgəcindən keçmiş qədim sənət növləri` (DƏQİQ DÜZ)
- **Tesseract 5:** `Zamanın süzgəcindən keçmiş qədim sənət növləri` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `Eamaninsüzgəcindən keçmiş qədim sənət növləri`

### Sətir 02 (Ölçü: 500x33 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `hər bir xalqın tarixi yaddaşını və mədəniyyətini`
- **V2 (Qayda əsaslı chunk):** `hər bir xalqın tarixi yaddaşını və mədəniyyətini` (DƏQİQ DÜZ)
- **Tesseract 5:** `hər bir xalqın tarixi yaddaşını və mədəniyyətini` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `hər bir"xalqın tarixi yaddaşını və mədəniyyətini`

### Sətir 03 (Ölçü: 442x33 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `özündə yaşadır. Azərbaycanda xalçaçılıq,`
- **V2 (Qayda əsaslı chunk):** `özündə yaşadır. Azərbaycanda xalçaçılıq,` (DƏQİQ DÜZ)
- **Tesseract 5:** `özündə yaşadır. Azərbaycanda xalçaçılıq,` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `özündəiyaşadır: Azərbaycanda xalçaçılıq,`

### Sətir 04 (Ölçü: 464x33 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `dulusçuluq, şəbəkə və misgərlik kimi sənət`
- **V2 (Qayda əsaslı chunk):** `dulusçuluq, şəbəkə və misgərlik kimi sənət` (DƏQİQ DÜZ)
- **Tesseract 5:** `dulusçuluq, şəbəkə və misgərlik kimi sənət` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `dülüsçülüq, şəbəkəivə misgərlik kimi sənət`

### Sətir 05 (Ölçü: 516x31 px, Parça sayı: 2)
- **Orijinal (Çap olunmuş mətndəki ref):** `sahələri nəsildən-nəslə ötürülərək bu günümüzə`
- **V2 (Qayda əsaslı chunk):** `sahələri nəsildən-nəslə ötürülərək bu günümüzə` (DƏQİQ DÜZ)
- **Tesseract 5:** `sahələri nəsildən-nəslə ötürülərək bu günümüzə` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `sahələrinəsildənsnəsləl ötürülərək bu günümüzə`

### Sətir 06 (Ölçü: 524x35 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə`
- **V2 (Qayda əsaslı chunk):** `qədər gəlib çatmışdır Hər bir naxış, hər bir ilmə` (FƏRQ VAR)
- **Tesseract 5:** `qədər gəlib çatmışdır. Hər bir naxış, hər bir ilmə` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `qədərigəlibçatmışdır: Hər bir naxış, hər bir ilmə`

### Sətir 07 (Ölçü: 498x34 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `bir hekayə danışır, usta əllərinin zəhmətini və`
- **V2 (Qayda əsaslı chunk):** `bir hekayə danışır, usta əllərinin zəhmətini və` (DƏQİQ DÜZ)
- **Tesseract 5:** `bir hekayə danışır, usta əllərinin zəhmətini və` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `birihekayə danışır; üsta əllərinin zəhmətini və`

### Sətir 08 (Ölçü: 212x28 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `ruhunu əks etdirir.`
- **V2 (Qayda əsaslı chunk):** `ruhunu əks etdirir.` (DƏQİQ DÜZ)
- **Tesseract 5:** `ruhunu əks etdirir.` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `irühünüəks etdirir.`

### Sətir 09 (Ölçü: 498x37 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `Bu sənət inciləri təkcə gözəllik deyil, həm də`
- **V2 (Qayda əsaslı chunk):** `Bu sənət inciləri təkcə gözəllik deyil, həm də` (DƏQİQ DÜZ)
- **Tesseract 5:** `Bu sənət inciləri təkcə gözəllik deyil, həm də` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `Busənət inciləri təkcə gözəllik deyil, həm də`

### Sətir 10 (Ölçü: 519x36 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `xalqımızın milli kimliyinir göstəricisidir. Onları`
- **V2 (Qayda əsaslı chunk):** `xalqımızın milli kimliyinir göstəricisidir. Onları` (DƏQİQ DÜZ)
- **Tesseract 5:** `xalqımızın milli kimliyinir göstəricisidir. Onları` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `xalqımızın milli kimliyinir göstəricisidir, Onları`

### Sətir 11 (Ölçü: 547x38 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək`
- **V2 (Qayda əsaslı chunk):** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək` (DƏQİQ DÜZ)
- **Tesseract 5:** `qorumaq, yaşatmaq və gələcək nəsillərə ötürmək` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `qörümaq, yaşatmaq və gələcək nəsillərə ötürmək`

### Sətir 12 (Ölçü: 527x35 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `hər birimizin mənəvi borcudur. Çünki tarixinə`
- **V2 (Qayda əsaslı chunk):** `hər birimizin mənəvi borcudur. Çünki tarixinə` (DƏQİQ DÜZ)
- **Tesseract 5:** `hər birimizin mənəvi borcudur. Çünki tarixinə` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `hər birimizin mənəvi borcudur: Çünki tarixinə`

### Sətir 13 (Ölçü: 546x37 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `tarixirə sahib çıxan millətlər öz gələcəyini daha`
- **V2 (Qayda əsaslı chunk):** `tarixirə sahib çıxan millətlər öz gələcəyini daha` (DƏQİQ DÜZ)
- **Tesseract 5:** `tarixirə sahib çıxan millətlər öz gələcəyini daha` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `tarixirə sahib çıxan millətlər öz gələcəyini daha`

### Sətir 14 (Ölçü: 547x36 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `möhkəm təməllər üzərində qura bilirlər. Hər bir`
- **V2 (Qayda əsaslı chunk):** `möhkəm təməllər üzərində qura bilirlər. Hər bir` (DƏQİQ DÜZ)
- **Tesseract 5:** `möhkəm təməllər üzərində qura bilirlər. Hər bir` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `möhkəmtəməllər üzərində qura bilirlər: Hər bir`

### Sətir 15 (Ölçü: 553x41 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `qədim əşya keçmişimizlə bu günümüz arasında`
- **V2 (Qayda əsaslı chunk):** `qədim əşya keçmişimizlə bu günümüz arasında` (DƏQİQ DÜZ)
- **Tesseract 5:** `qədim əşya keçmişimizlə bu günümüz arasında` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `qədim əşya keçmişimizlə bu günümüz arasında`

### Sətir 16 (Ölçü: 285x39 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `görünməz bir körpüdür.`
- **V2 (Qayda əsaslı chunk):** `görünməz bir körpüdür.` (DƏQİQ DÜZ)
- **Tesseract 5:** `görünməz bir körpüdür.` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `görünməz bir körpüdür.`

### Sətir 17 (Ölçü: 559x40 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `Müasir dünyada texnologiyanın sürətli inkişafına`
- **V2 (Qayda əsaslı chunk):** `Müasir dünyada texnologiyanın sürətli inkişafına` (DƏQİQ DÜZ)
- **Tesseract 5:** `Müasir dünyada texnologiyanın sürətli inkişafına` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `Müasir dünyada texnologiyanın sürətli inkişafına`

### Sətir 18 (Ölçü: 586x42 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri`
- **V2 (Qayda əsaslı chunk):** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri` (DƏQİQ DÜZ)
- **Tesseract 5:** `baxmayaraq, əl əməyinin, milli sənətkarlığın dəyəri` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `baxmayaraq, əl əməyinin, mili sənətkarlığın dəyəri`

### Sətir 19 (Ölçü: 595x41 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `heç vaxt azalmır. Əksinə, insan ruhunun hsduğu bu`
- **V2 (Qayda əsaslı chunk):** `heç vaxt azalmır. Əksinə, insan ruhunun hsduğu bu` (DƏQİQ DÜZ)
- **Tesseract 5:** `heç vaxt azalmır. Əksinə, insan ruhunun hsduğu bu` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `heçivaxt azalmır: Əksinə, insan ruhunun hsduğu bü`

### Sətir 20 (Ölçü: 584x38 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `sənət nümunələrinə olan maraq günü-gündən artır.`
- **V2 (Qayda əsaslı chunk):** `sənət nümunələrinə olan maraq günü-gündən artır.` (DƏQİQ DÜZ)
- **Tesseract 5:** `sənət nümunələrinə olan maraq günü-gündən artır.` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `sənətnümünələrinə olan maraq günü-gündən artır.`

### Sətir 21 (Ölçü: 600x40 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim`
- **V2 (Qayda əsaslı chunk):** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim` (DƏQİQ DÜZ)
- **Tesseract 5:** `Gənc nəslin bu sənətlərə maraq göstərməsi, qədim` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `Gənoinəslin bu sənətlərə maraq göstərməsi, qədiml`

### Sətir 22 (Ölçü: 587x41 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `ənənələnin müasir dizaynlarla birləşdirilməsi bu`
- **V2 (Qayda əsaslı chunk):** `ənənələnin müasir dizaynlarla birləşdirilməsi bu` (DƏQİQ DÜZ)
- **Tesseract 5:** `ənənələnin müasir dizaynlarla birləşdirilməsi bu` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `əmənələnin müasir dizaynlarla birləşdirilməsi bül`

### Sətir 23 (Ölçü: 446x44 px, Parça sayı: 1)
- **Orijinal (Çap olunmuş mətndəki ref):** `irsin gələcəyini daha da parlaq edir.`
- **V2 (Qayda əsaslı chunk):** `irsin gələcəyini daha da parlaq edir.` (DƏQİQ DÜZ)
- **Tesseract 5:** `irsin gələcəyini daha da parlaq edir.` (DƏQİQ DÜZ)
- **V1 (Qayda əsaslı chunk):** `irsinigələcəyini daha daparlaq ediri`
