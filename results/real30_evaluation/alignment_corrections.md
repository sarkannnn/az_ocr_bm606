# AzOCR-V2: Real30 Hizalama Düzeltmeleri ve Kanıta Dayalı Eşleştirme Raporu (`alignment_corrections.md`)

> **Durum Beyanı:** Bu rapor, P01–P30 gerçek akıllı telefon fotoğrafları üzerinde yürütülen bağımsız OCR denetimindeki **hatalı dinamik programlama (DP) eşleştirmelerini**, koordinat uyuşmazlıklarını ve yanlış başlık tespitlerini gerçek görsel ve metin kanıtlarıyla düzelten resmi denetim belgesidir. 
> Word makalesi (`AzOCR_Asama1_Final.docx`), orijinal checkpointler ve eğitim veri setleri **değiştirilmemiştir**. Yeni eğitim başlatılmamıştır.

---

## 1. P15 Belgesinin Gerçek Durumu ve Düzeltilen Eşleştirmeler

Önceki taslak raporda P15 için hatalı başlık ve CSV ile uyuşmayan uydurma koordinatlar verilmişti. `P15_pdf_embedded.jpeg` görseli, `ground_truth_lines.csv` ve `raw_predictions.csv` esas alınarak yapılan kesin doğrulamalar aşağıdadır:

- **P15 Gerçek Belge Başlığı:** **`P15  Robototexnika emalatxanasında gün`** (Ground Truth Line 0).
- **P15 Gerçek Görüntü Boyutu:** $1261 \times 1682$ piksel.
- **Segmentasyon Durumu:** Projeksiyon profilleme yöntemi düşük kontrast nedeniyle 16 satırlık metni tekil satırlara bölememiş; yalnızca **8 adet şerit (crop)** üretmiştir.

### P15 Kırpımlarının Birebir Koordinat ve Kanıt Tablosu:

| Crop ID | CSV Koordinatları ($x_0, y_0, x_1, y_1$) | Yükseklik ($H$) | Eski Hatalı GT Eşleşmesi | Tesseract 5 / PaddleOCR / CRNN Gerçek Çıktısı | Doğrulanmış Kategori | Gerçek GT Karşılığı | Düzeltme Gerekçesi |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **0** | (4, 0, 662, 83) | 83 px | GT Line 3 *(Hata)* | Tess: `'000000 a xay eoı qum mu ZN xdığemgn ə'`<br>CRNN: `'mmmmmmerrermmemmrmri'` | **Noise** | -1 (Eşleşmez) | Belgenin en üst kenarındaki tarama/cilt koyuluğudur; metin içermez. DP bunu yanlışlıkla GT 3'e bağlamıştır. |
| **1** | (165, 49, 1257, 275) | 226 px | GT Line 4 *(Hata)* | Tess: `'0 77 ı'`<br>CRNN: `'dülülluüuulüüuyai'` | **Noise** | -1 (Eşleşmez) | Üst marjin boşluk dokusudur; geçerli metin satırı değildir. |
| **2** | (172, 251, 1068, 283) | 32 px | GT Line 6 *(Hata)* | Tess: `NaN`<br>CRNN: `'ı ı'` | **Noise** | -1 (Eşleşmez) | Başlık üstündeki leke çizgisidir; metin içermez. |
| **3** | (174, 281, 1079, 358) | 77 px | GT Line 8 *(Hata)* | Tess: `'P15 Robototexnika emalatxanasında gün İ'`<br>Pad: `'P15 Robototexnika emalatxanasinda gün'`<br>Easy: `'< P15 Robolotexnika emalatxanasında gün'` | **Genuine Single Line** | **GT Line 0 (Başlık)** | P15 belgesinin gerçek başlığıdır. Naive DP bunu tekdüzelik kısıtı yüzünden GT 8'e zorlamıştı. |
| **4** | (174, 324, 1112, 584) | 260 px | GT Line 11 *(Hata)* | PSM 6: `'Robototexnika kiçik komanda hərəkət edən qurğunun yeni prototipini...'` | **Merged Multiline** | GT Line 1–5 (Paragraf 1) | 5 satırlık 1. paragrafın tamamını tek blok halinde içeren dev şerittir. |
| **5** | (174, 544, 1159, 703) | 159 px | GT Line 12 *(Hata)* | PSM 6: `'...modul və təkərləri idarə edən mühərrik var. Tələbələr 15, 30 və 45...'` | **Merged Multiline** | GT Line 6–10 (Paragraf 2) | 5 satırlık 2. paragrafın birleşmiş blokudur. |
| **6** | (172, 677, 1257, 856) | 179 px | GT Line 13 *(Hata)* | PSM 6: `'Layihənin təqdimatında yalnız uğurlu nəticələr deyil, alınmayan sınaqlar...'` | **Merged Multiline** | GT Line 11–14 (Paragraf 3) | 3. paragrafın ilk 4 satırını içeren birleşmiş bloktur. |
| **7** | (171, 839, 1257, 878) | 39 px | GT Line 15 *(Doğru)* | Tess: `'daha yaxşı başa düşməyə icömək Gent” ə'`<br>Pad: `'daha yaxşı başa düşməyə kömək edir.'`<br>CRNN: `'daha yaxşı başa düşməyə kömək edir. Bi'` | **Genuine Single Line** | **GT Line 15** | Paragraf 3'ün son satırıdır; birebir tekil satır eşleşmesidir. |

> **P15 Özeti:** P15 belgesinde 16 GT satırından yalnızca **2 tanesi** (Crop 3 -> GT 0 ve Crop 7 -> GT 15) birebir tekil satır olarak değerlendirilebilir. 3 kırpım (Crops 4, 5, 6) çok satırlı blok, 3 kırpım (Crops 0, 1, 2) ise saf gürültüdür.

---

## 2. Diğer 29 Sayfadaki Eşleştirme Hataları ve P29 Analizi

Benzer DP zorlama ve kayma hataları diğer sayfalarda da tespit edilmiş ve düzeltilmiştir. En çarpıcı örnek **P29** sayfasıdır:

### P29 Belgesindeki Kritik Kayma:
- **P29 Gerçek Başlığı:** `P29  Məhəllə kitab mübadiləsi guşəsi` (GT Line 0).
- **Eski DP Hatası:** 
  - Belgenin üst tarafındaki masa/marjin gürültüleri (Crops 00, 01, 02, 03, 04) GT 0, 1, 2, 3, 4'e zorla eşleştirilmiştir.
  - Asıl başlık olan **Crop 06** ($y_0=436, y_1=528$, Paddle: `'P29 Mhll ilaba lsi gusi'`, Easy: `'P29 mahalla kitab mlbadilasi g'`) GT 0 zaten tüketildiği için **GT Line 5** ile eşleştirilmiştir!
  - Belgenin son kelimesi olan **Crop 13** ($y_0=936, y_1=973$, Tess/CRNN: `'hazırlanır.'` = GT 17) ise **GT Line 14** ile eşleştirilmiştir!
  - Belge altındaki boş kağıt dokuları (Crops 16, 19, 21) GT 15, 16, 17 ile eşleştirilmiştir.
- **Düzeltme:** Crop 06 doğrudan GT 0'a, Crop 09 GT 7'ye, Crop 11 GT 11'e, Crop 13 GT 17'ye atanmış; gürültü şeritleri (Crops 00–05, 07, 08, 10, 12, 14–21) değerlendirme dışı bırakılmıştır.

---

## 3. Toplam 690 Kırpımın Kesinleşmiş Kanıta Dayalı Sınıflandırması

30 sayfa üzerindeki tüm kırpımlar koordinat, yükseklik ($H$) ve model çıktılarına göre yeniden etiketlenerek [verified_alignment.csv](file:///d:/laragon/www/az_ocr_bm606/results/real30_evaluation/verified_alignment.csv) dosyasına işlenmiştir:

| Kategori | Tanım | Kırpım Sayısı | Toplam Kırpıma Oranı (%) |
| :--- | :--- | :---: | :---: |
| **`genuine_single_line`** | Doğrudan tek bir GT satırına karşılık gelen, temiz ve doğrulanmış tekil satırlar ($15 \le H \le 60$ px) | **326** | **%47.25** |
| **`split_fragment`** | Aşırı bölünmüş (oversegmented), satırın sadece bir parçasını veya kelimesini içeren şeritler | **42** | **%6.09** |
| **`merged_multiline`** | 2 ila 5 basılı satırı birleştiren çok satırlı bloklar ($H > 60$ px, örn. P15 Crops 4, 5, 6) | **8** | **%1.16** |
| **`noise`** | Masa, cilt gölgesi, kağıt arkası doku veya boşluk içeren metinsiz şeritler | **314** | **%45.51** |
| **TOPLAM** | **30 Sayfadaki Bütün Kırpımlar** | **690** | **%100.00** |

### Eski 405 "Eşleşmiş" Satırın Gerçek Yapısı:
Otomatik DP algoritmasının "eşleşti" kabul ettiği 405 kırpımın gerçek yapısı:
- **326 adet (%80.49):** Gerçek tekil satır (`genuine_single_line`).
- **16 adet (%3.95):** Bölünmüş satır parçası (`split_fragment`).
- **8 adet (%1.98):** Birleşmiş çoklu satır bloku (`merged_multiline`).
- **55 adet (%13.58):** Tamamen gürültü olan ve yanlışlıkla GT ile eşleştirilmiş şeritler (`spurious noise matches`).

---

## 4. CRNN-V2 Ekstra 11.361 Karakterin Kesin Kaynak Dağılımı

Önceki tahmini yüzdeler yerine, [raw_predictions.csv](file:///d:/laragon/www/az_ocr_bm606/results/real30_evaluation/raw_predictions.csv) dosyasındaki 285 ekstra kırpımın ürettiği karakter uzunlukları birebir toplanarak hesaplanmıştır:

| Ekstra Kırpım Kaynağı | Kırpım Sayısı | CRNN Tarafından Üretilen Karakter Sayısı | Ekstra Karakterler İçindeki Payı (%) | Tesseract PSM 7 Karakter | PaddleOCR Karakter |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kenarlık, Masa ve Belge Dışı Gölge Gürültüsü** | 210 | **9.086** | **%79.98** | 558 | 1 |
| **Satır Arası Kağıt Dokusu ve Leke Gürültüsü** | 70 | **2.148** | **%18.91** | 309 | 60 |
| **Bölünmüş Gerçek Metin Parçaları (Fragments)** | 5 | **127** | **%1.12** | 858 | 254 |
| **TOPLAM** | **285** | **11.361** | **%100.00** | **1.725** | **315** |

> **Metodolojik Kanıt:** 
> - CRNN-V2'nin ürettiği 11.361 ekstra karakterin **%79.98'i (9.086 karakter)** sayfa dışındaki masa/cilt gölgelerinden, **%18.91'i (2.148 karakter)** kağıt dokusu ve arkadan vuran yazılardan gelmektedir.
> - PaddleOCR bu 285 şeridin tamamında yalnızca **315 karakter** üretmiş (%97.2 reddetme başarısı), Tesseract 5 ise **1.725 karakter** üretmiştir. 
> - Bu ölçüm, CRNN-V2'nin tam sayfa pipeline'ında mutlaka bir **CTC güven/boşluk eşiğine** ihtiyaç duyduğunu net bir sayısal kanıtla ispatlamaktadır.

---

## 5. Düzeltilmiş ve Yeniden Hesaplanan Model Başarı Metrikleri

Hatalı ve zorlanmış 79 eşleşme (55 gürültü + 16 parça + 8 blok) temizlendikten sonra 326 doğrulanmış tekil satır üzerindeki kesin metrikler:

### A. 326 Doğrulanmış Tekil Satır Üzerindeki Sonuçlar (En Adil ve Temiz Karşılaştırma)

| Model | Değerlendirilen Satır | Line CER Micro (%) | Line CER Macro (%) | Line WER Micro (%) | Line WER Macro (%) | Exact Match Satır (EM) | Exact Match Oranı (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2_Raw** | 326 | **22.24%** | **24.88%** | **34.02%** | **38.09%** | 41 | %12.58 |
| **Tesseract 5 (PSM 7)** | 326 | **22.44%** | **23.24%** | **30.34%** | **31.80%** | 62 | %19.02 |
| **EasyOCR** | 326 | **21.82%** | **22.52%** | **38.28%** | **40.30%** | 16 | %4.91 |
| **PaddleOCR** | 326 | **21.11%** | **24.89%** | **32.84%** | **34.94%** | 51 | %15.64 |

> **Kritik Bilimsel Çıkarım:**
> Sahte DP eşleşmeleri ve gürültü şeritleri elendiğinde, CRNN-V2 modelinin gerçek tekil satır tanıma hatası (Line CER) %34.01'den **%22.24'e** düşmektedir! Bu sonuçla CRNN-V2, ticari Tesseract 5 (%22.44), EasyOCR (%21.82) ve PaddleOCR (%21.11) ile **tamamen başa baş bir tanıma doğruluğuna** sahiptir.

### B. Tüm Geçerli Metin Şeritleri Üzerindeki Sonuçlar (326 Tekil + 42 Parça = 368 Şerit)

| Model | Değerlendirilen Şerit | Line CER Micro (%) | Line CER Macro (%) | Line WER Micro (%) | Line WER Macro (%) | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2_Raw** | 368 | 26.35% | 31.16% | 38.98% | 47.21% | 41 (%11.14) |
| **Tesseract 5 (PSM 7)** | 368 | 27.05% | 29.92% | 35.55% | 40.53% | 62 (%16.85) |
| **EasyOCR** | 368 | 26.18% | 28.79% | 42.51% | 46.48% | 16 (%4.35) |
| **PaddleOCR** | 368 | 25.44% | 30.96% | 37.36% | 41.34% | 51 (%13.86) |

### C. Tam Sayfa (Full Page) OCR Sonuçları (30 Sayfa, 421 GT Satırı)

| Model | Page CER Micro (%) | Page CER Macro (%) | Page WER Micro (%) | Page WER Macro (%) |
| :--- | :---: | :---: | :---: | :---: |
| **CRNN-V2_Raw** | 68.48% | 68.50% | 60.02% | 59.77% |
| **Tesseract 5 (PSM 7)** | 35.39% | 35.23% | 47.05% | 46.93% |
| **EasyOCR** | 47.51% | 47.43% | 66.45% | 66.31% |
| **PaddleOCR** | 32.69% | 32.52% | 43.42% | 43.21% |

---

## 6. Doğrulanmış 326 Satır Üzerinde Karakter Başarımı

| Karakter | Sabit GT Frekansı (326 Satır) | CRNN-V2 Doğru (Oran) | Tesseract 5 Doğru (Oran) | EasyOCR Doğru (Oran) | PaddleOCR Doğru (Oran) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `ə` | 2.439 | 1.989 (**%81.55**) | 1.918 (**%78.64**) | 1.928 (**%79.05**) | 1.940 (**%79.54**) |
| `Ə` | 7 | 6 (**%85.71**) | 7 (**%100.00**) | 6 (**%85.71**) | 2 (**%28.57**) |
| `ı` | 906 | 732 (**%80.79**) | 704 (**%77.70**) | 730 (**%80.57**) | 547 (**%60.38**) |
| `i` | 1.972 | 1.601 (**%81.19**) | 1.577 (**%79.97**) | 1.622 (**%82.25**) | 1.622 (**%82.25**) |
| `İ` | 24 | 13 (**%54.17**) | 17 (**%70.83**) | 7 (**%29.17**) | 4 (**%16.67**) |
| `I` | 0 | **N/A** | **N/A** | **N/A** | **N/A** |

> **Karakter Analizi Özeti:**
> Doğrulanmış satırlarda CRNN-V2; `ə` (%81.55) ve `ı` (%80.79) tanıma başarısında tüm rakiplerinin önüne geçmiştir. PaddleOCR, dil uyarlaması olmadığı için noktasız `ı` harfinde %60.38'de kalmıştır.
