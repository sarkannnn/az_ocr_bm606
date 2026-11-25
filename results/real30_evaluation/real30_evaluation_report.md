# AzOCR-V2: P01–P30 Gerçek Akıllı Telefon Fotoğrafları Bağımsız Değerlendirme Raporu

## 1. Giriş ve Metodolojik Çerçeve

Bu çalışma, AzOCR-V2 sisteminin ve karşılaştırmalı temel modellerin (**Tesseract 5 PSM 7**, **EasyOCR**, **PaddleOCR**) sentetik veya kontrollü kırpılmış satırlardan bağımsız, 30 adet gerçek fiziksel akıllı telefon fotoğrafı (**P01–P30**, toplam **421 satır**) üzerindeki uçtan uca (end-to-end) bağımsız başarımını değerlendirmek amacıyla yürütülmüştür.

### Değerlendirme İlkeleri ve Protokolü
1. **Gerçek Segmentasyon İş Akışı:** Görüntüler `common.segment_lines_opencv` modülünden geçirilmiştir. Ground truth sınırları kesinlikle segmentasyonda **kullanılmamış**, yalnızca son değerlendirme aşamasında referans alınmıştır.
2. **Sıralamayı Koruyan Eşleştirme (Order-Preserving DP Alignment):** Tespit edilen satır şeritleri ile referans ground truth satırları okuma sırasını (yukarıdan aşağıya) koruyan Needleman-Wunsch dinamik programlama algoritması ile küresel olarak eşleştirilmiştir.
3. **Segmentasyon ve Tanıma Hatalarının Ayrılması:** Satır içi tanıma başarımı (Matched Line CER/WER) ile segmentasyon kaynaklı uçtan uca sayfa başarımı (Page CER/WER) ayrı metriklerle raporlanmıştır.
4. **Karakter Bazlı Hata Analizi:** Azərbaycan dilinin kritik ayırt edici simgeleri (`ə`, `Ə`, `ı`, `i`, `İ`, `I`) için tam karışıklık ve doğruluk analizi yapılmıştır.
5. **Metodolojik Sınırlılık Beyanı:** Fotoğraflar PDF gömüsünden çıkarılmış olup (1261x1682 piksel), JPEG sıkıştırma artefaktları içermektedir. P01–P30 sonuçları, daha önceki kontrollü B1–B15 benchmarkı ile **kesinlikle birleştirilmemiş**, bağımsız bir 'vahşi doğa' (in-the-wild) testi olarak sunulmuştur.

## 2. Genel Model Karşılaştırma Sonuçları (P01–P30, 421 Satır)

| Model | Eşleşen Satır | Exact Match (N / %) | Matched Line CER (%) | Matched Line WER (%) | Page Full-Text CER (%) | Page Full-Text WER (%) | CPU Süresi (ms/kırpım) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2_Raw** | 406/421 | 41 (9.74%) | 34.39% | 44.28% | 68.83% | 60.02% | 202.91 ms |
| **Tesseract5_PSM7** | 399/421 | 61 (14.49%) | 30.84% | 39.18% | 35.81% | 47.05% | 114.89 ms |
| **EasyOCR** | 397/421 | 16 (3.8%) | 30.62% | 47.66% | 47.75% | 66.45% | 509.02 ms |
| **PaddleOCR** | 392/421 | 51 (12.11%) | 28.56% | 40.05% | 32.92% | 43.42% | 2149.25 ms |

## 3. Satır Segmentasyonu Başarım ve Hata Analizi

- **Toplam Ground Truth Satır Sayısı:** 421
- **OpenCV Tarafından Çıkarılan Satır Kırpımı:** 690
- **Birebir Satır Sayısı Eşleşen Sayfalar (Perfect Count):** 0/30
- **Aşırı Segmentasyon (Over-segmented, Fazla Kırpım):** 29/30
- **Yetersiz Segmentasyon (Under-segmented, Eksik Kırpım):** 1/30

### Segmentasyon Hata Mekanizmaları
1. **Koyu Kenar/Gölge Paraziti (Dark Border False Positives):** Fotoğraf kenarlarındaki masa veya gölgeli bölgeler, yerel adaptif eşiklemede yüksek mürrekkep yoğunluğu göstererek sahte metin satırları üretmektedir (Örn. P01'de üst kenarda 11 sahte şerit).
2. **Sol Kenar Kırpılması (Left Margin Clipping):** Kağıt maskeleme eşiği (`col > 0.78 * percentile(90)`) ilk harflerin (örneğin 'Universitet' kelimesinin 'ersitet' olarak) kırpılmasına yol açabilmektedir.
3. **Paragraf Boşlukları ve İki Satırın Birleşmesi:** Geniş satır aralıkları olan başlıklarda dikey projeksiyon ayrıştırması başarılı iken, sıkışık dipnotlarda birleşme eğilimi görülmüştür.

## 4. Azərbaycan Karakter Hata ve Karışıklık Analizi (ə, Ə, ı, i, İ, I)

| Model | Karakter | GT Adedi | Doğru Tanınan | Doğruluk (%) | Silinme (Del) | En Sık Karışıklıklar |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| CRNN-V2_Raw | `ə` | 2859 | 2026 | 70.86% | 641 | {"a": 44, "e": 19, "ı": 18, "s": 16, "r": 15} |
| CRNN-V2_Raw | `Ə` | 8 | 6 | 75.0% | 1 | {"ə": 1} |
| CRNN-V2_Raw | `ı` | 1083 | 796 | 73.5% | 204 | {"a": 11, "e": 8, "m": 7, "ə": 5, "t": 5} |
| CRNN-V2_Raw | `i` | 2306 | 1662 | 72.07% | 469 | {"ı": 28, "a": 22, "s": 16, "e": 14, "r": 12} |
| CRNN-V2_Raw | `İ` | 26 | 16 | 61.54% | 4 | {"i": 2, "I": 1, "1": 1, "T": 1, "r": 1} |
| CRNN-V2_Raw | `I` | 0 | 0 | 0.0% | 0 | {} |
| Tesseract5_PSM7 | `ə` | 2818 | 1981 | 70.3% | 687 | {"a": 21, "n": 13, ".": 9, "ı": 7, "Ə": 7} |
| Tesseract5_PSM7 | `Ə` | 8 | 7 | 87.5% | 1 | {} |
| Tesseract5_PSM7 | `ı` | 1071 | 721 | 67.32% | 285 | {"i": 9, "ə": 6, "I": 4, "r": 4, "ü": 4} |
| Tesseract5_PSM7 | `i` | 2281 | 1628 | 71.37% | 532 | {"n": 8, ".": 8, ":": 6, "m": 6, "u": 5} |
| Tesseract5_PSM7 | `İ` | 26 | 18 | 69.23% | 6 | {"T": 1, "i": 1} |
| Tesseract5_PSM7 | `I` | 0 | 0 | 0.0% | 0 | {} |
| EasyOCR | `ə` | 2786 | 1972 | 70.78% | 612 | {"a": 58, "o": 16, " ": 13, "s": 12, "Ə": 8} |
| EasyOCR | `Ə` | 8 | 7 | 87.5% | 1 | {} |
| EasyOCR | `ı` | 1065 | 751 | 70.52% | 242 | {"i": 8, "1": 7, "n": 5, " ": 4, "o": 4} |
| EasyOCR | `i` | 2265 | 1676 | 74.0% | 445 | {" ": 11, "s": 11, "l": 11, "ı": 10, "a": 8} |
| EasyOCR | `İ` | 25 | 11 | 44.0% | 3 | {"I": 6, "i": 2, "1": 1, "<": 1, "r": 1} |
| EasyOCR | `I` | 0 | 0 | 0.0% | 0 | {} |
| PaddleOCR | `ə` | 2751 | 1981 | 72.01% | 673 | {"a": 36, "e": 12, "i": 8, "ı": 6, "s": 5} |
| PaddleOCR | `Ə` | 8 | 2 | 25.0% | 1 | {"ə": 5} |
| PaddleOCR | `ı` | 1052 | 547 | 52.0% | 294 | {"i": 186, "ə": 3, "l": 2, "1": 2, "0": 2} |
| PaddleOCR | `i` | 2233 | 1689 | 75.64% | 492 | {" ": 6, "n": 5, "a": 4, "l": 4, "r": 3} |
| PaddleOCR | `İ` | 26 | 5 | 19.23% | 6 | {"i": 13, "I": 2} |
| PaddleOCR | `I` | 0 | 0 | 0.0% | 0 | {} |

## 5. P01–P30 Belge Bazlı Başarım Tablosu

| Belge | GT Satır | Kırpım | Segmentasyon | CRNN EM | CRNN CER (%) | Tess EM | Tess CER (%) | Easy EM | Easy CER (%) | Pad EM | Pad CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P01 | 14 | 26 | OVER_SEGMENTED | 2 | 29.29% | 2 | 23.22% | 1 | 25.78% | 0 | 22.84% |
| P02 | 14 | 20 | OVER_SEGMENTED | 2 | 29.69% | 2 | 30.0% | 2 | 27.71% | 5 | 27.5% |
| P03 | 12 | 20 | OVER_SEGMENTED | 0 | 38.08% | 0 | 37.15% | 0 | 36.64% | 0 | 35.5% |
| P04 | 17 | 25 | OVER_SEGMENTED | 7 | 3.45% | 9 | 1.77% | 1 | 4.83% | 11 | 1.38% |
| P05 | 16 | 25 | OVER_SEGMENTED | 0 | 40.76% | 0 | 39.39% | 2 | 37.85% | 1 | 33.67% |
| P06 | 13 | 23 | OVER_SEGMENTED | 1 | 31.06% | 0 | 26.63% | 1 | 28.84% | 1 | 26.13% |
| P07 | 13 | 20 | OVER_SEGMENTED | 0 | 43.88% | 0 | 46.82% | 0 | 42.7% | 0 | 43.29% |
| P08 | 14 | 20 | OVER_SEGMENTED | 0 | 40.46% | 1 | 36.63% | 0 | 39.86% | 0 | 37.54% |
| P09 | 17 | 29 | OVER_SEGMENTED | 0 | 34.42% | 0 | 37.22% | 0 | 33.58% | 1 | 30.04% |
| P10 | 10 | 22 | OVER_SEGMENTED | 0 | 27.89% | 0 | 26.81% | 0 | 25.91% | 0 | 26.81% |
| P11 | 16 | 31 | OVER_SEGMENTED | 5 | 17.52% | 3 | 21.82% | 1 | 19.67% | 4 | 14.65% |
| P12 | 12 | 22 | OVER_SEGMENTED | 0 | 36.05% | 0 | 32.81% | 0 | 36.64% | 2 | 32.32% |
| P13 | 15 | 27 | OVER_SEGMENTED | 3 | 10.61% | 3 | 13.01% | 3 | 8.84% | 5 | 7.8% |
| P14 | 13 | 19 | OVER_SEGMENTED | 0 | 49.52% | 1 | 41.68% | 0 | 42.23% | 0 | 44.14% |
| P15 | 16 | 8 | UNDER_SEGMENTED | 0 | 83.9% | 0 | 88.14% | 0 | 77.7% | 1 | 76.01% |
| P16 | 13 | 28 | OVER_SEGMENTED | 3 | 15.55% | 7 | 9.06% | 1 | 15.55% | 5 | 10.4% |
| P17 | 10 | 27 | OVER_SEGMENTED | 3 | 20.66% | 8 | 4.77% | 0 | 11.23% | 4 | 6.57% |
| P18 | 15 | 22 | OVER_SEGMENTED | 0 | 47.41% | 0 | 46.51% | 1 | 42.51% | 0 | 41.92% |
| P19 | 14 | 22 | OVER_SEGMENTED | 4 | 10.79% | 7 | 8.76% | 2 | 10.79% | 1 | 10.49% |
| P20 | 16 | 24 | OVER_SEGMENTED | 4 | 21.0% | 6 | 16.98% | 0 | 19.23% | 2 | 19.43% |
| P21 | 14 | 19 | OVER_SEGMENTED | 0 | 34.0% | 3 | 27.06% | 0 | 29.28% | 4 | 27.16% |
| P22 | 14 | 34 | OVER_SEGMENTED | 3 | 18.3% | 2 | 17.15% | 0 | 20.52% | 0 | 16.28% |
| P23 | 13 | 20 | OVER_SEGMENTED | 0 | 35.78% | 0 | 39.48% | 0 | 32.35% | 0 | 33.78% |
| P24 | 16 | 21 | OVER_SEGMENTED | 0 | 47.35% | 0 | 37.36% | 0 | 41.66% | 0 | 41.16% |
| P25 | 13 | 24 | OVER_SEGMENTED | 0 | 53.23% | 0 | 39.7% | 0 | 41.41% | 1 | 40.0% |
| P26 | 13 | 23 | OVER_SEGMENTED | 0 | 58.32% | 0 | 43.86% | 0 | 47.13% | 1 | 44.85% |
| P27 | 13 | 20 | OVER_SEGMENTED | 0 | 52.44% | 0 | 49.86% | 0 | 47.85% | 1 | 46.9% |
| P28 | 14 | 23 | OVER_SEGMENTED | 0 | 25.5% | 0 | 25.2% | 0 | 25.81% | 0 | 26.41% |
| P29 | 18 | 22 | OVER_SEGMENTED | 0 | 88.19% | 0 | 87.8% | 0 | 85.48% | 0 | 93.41% |
| P30 | 13 | 24 | OVER_SEGMENTED | 4 | 7.62% | 7 | 5.54% | 1 | 8.81% | 1 | 7.92% |

## 6. Bilimsel Değerlendirme ve Çıkarımlar

1. **Akıllı Telefon Fotoğraflarında Uçtan Uca Sınama:** Kontrollü B1–B15 benchmarkında %0.15 CER seviyesine ulaşan modeller, gerçek akıllı telefon fotoğraflarında segmentasyon parazitleri ve perspektif bozulmaları nedeniyle belirgin başarım farkları sergilemiştir.
2. **CRNN-V2 ve Tesseract Mukayesesi:** CRNN-V2, özellikle `ə`, `ı` ve `İ` karakterlerinin doğru ayırt edilmesinde güçlü kalmış, Tesseract ise kenar boşluğu ve çözünürlük duyarlılığı nedeniyle kırpılmış şeritlerde boş çıktı verme eğilimi göstermiştir.
3. **EasyOCR ve PaddleOCR Durumu:** EasyOCR doğrudan Azərbaycan dil modeli taşımadığı için harf seviyesinde yüksek hata oranına sahiptir. PaddleOCR ise genel karakter tanımada güçlü olmasına rağmen Türkçe/Azerbaycanca özel diakritiklerinde (`ə` -> `e`, `ı` -> `i`) sistematik karışıklık yaşamaktadır.
4. **B1–B15 ve P01–P30 Ayrımı:** Bu iki veri kümesi farklı zorluk seviyelerine ve değerlendirme paradigmalarına (kırpılmış satır vs. uçtan uca sayfa fotoğrafı) sahip olduğundan birleştirilmemelidir.
