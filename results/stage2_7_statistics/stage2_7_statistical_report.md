# AzOCR-V2: Aşama 2.7 Kapsamlı İstatistiksel Analiz ve Metodolojik Doğrulama Raporu

**Tarih:** 2026-11-24  
**Kapsam:** Temiz veri seti konfigürasyon düzeltmesi, yakın tekrar (near-duplicate) denetimi, küçük örneklemli karakter sınıflarına Wilson %95 güven aralıkları, 244 büyük harf özel karakterinde 6 vs. 9 hata analizi, gerçek ve simüle ayrımı, i/ı/İ/I karışım matrisi, kümelenmiş bootstrap sınırlılığı ve 3-seed eğitim öncesi kontroller.

---

## 1. Temiz Veri Seti Planındaki Düzeltme (Grup A/B ve C/D Ayrımı)

Önceki taslak plandaki ayrık sentetik render üretimi ifadesi hakem revizyonuna uygun olarak netleştirilmiş ve düzeltilmiştir:
- **Grup A ve Grup B:** Her iki grup da **birebir aynı 10.241 örneklik** temiz V1 veri setini (`data/synth_line_v1_clean/train/`) kullanacaktır.
- **Grup C ve Grup D:** Her iki grup da **birebir aynı 15.754 örneklik** temiz V2 veri setini (`data/synth_line_v2_clean/train/`) kullanacaktır.
- **Eşit Eğitim Koşulu:** Her model çiftinde (A-B ve C-D) girdi görüntüleri, metin etiketleri, font dağılımı, batch size ve optimizer parametreleri tamamen özdeş tutulacak; **tek değişken dinamik augmentasyon hattı olacaktır** (A ve C temel V1 augmentasyonu; B ve D fotogerçekçi V2 augmentasyonu).

---

## 2. Temiz Külliyatta Yakın Tekrarlar (Near-Duplicates) Denetimi

Temiz külliyatta (`data/corpus_line_clean/`) tam ve Unicode NFC çakışması **0**'dır. Bunun ötesinde yapılan vaka ve noktalama denetimi bulguları:

| Kriter | Train vs. Val | Train vs. Test | Örnek Tespit |
| :--- | :---: | :---: | :--- |
| **Tam Eşleşme (Exact)** | **0** | **0** | Sıfır sızıntı doğrulandı |
| **Unicode NFC Eşleşmesi** | **0** | **0** | Sıfır sızıntı doğrulandı |
| **Yalnızca Büyük/Küçük Harf Farkı** | **5** | **4** | `imperiyasının paytaxtı oldu.` (Tr) vs `İmperiyasının paytaxtı oldu.` (Val) |
| **Yalnızca Noktalama / Boşluk Farkı** | **12** | **11** | `Universitetinin professoru` (Tr) vs `Universitetinin professoru,` (Val) |

> **Metodolojik Not:** Bu 5 ve 12 satırlık varyantlar genel dilde sık geçen bağımsız tümce parçacıklarıdır (örneğin tırnak işareti veya son nokta farkı). Kaynak veriler talimat uyarınca değiştirilmemiştir; hakem istediğinde şeffafça sunulmak üzere `results/stage2_7_statistics/stage2_7_statistics_summary.json` içine işlenmiştir.

---

## 3. Karakter Sınıfları İçin Wilson %95 Güven Aralıkları (Makale Tablo 6b Yeniden Değerlendirmesi)

Küçük örneklemli oranlarda Wald normal yaklaşımı ($p \pm z\sqrt{p(1-p)/n}$) sıfıra yakın oranlarda geçersizdir. Tüm karakter sınıfları için kesin **Wilson Score Interval** ve V2 vs. Tesseract karşılaştırması için **Fisher's Exact Test** uygulanmıştır:

| Karakter Sınıfı / Grafem | Toplam ($n$) | V2 Hata ($k$) | V2 Hata (%) | **V2 Wilson %95 GA** | Tess Hata ($k$) | Tess Hata (%) | **Tess Wilson %95 GA** | Fisher Testi $p$-değeri | Anlamlı mı ($p < 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lowercase Special (`ə,ı,ö,ğ,ü,ş,ç`)** | 7.774 | 21 | 0.27% | **[0.18%, 0.41%]** | 15 | 0.19% | **[0.12%, 0.32%]** | 0.4045 | Hayır (Farksız) |
| **Uppercase Special (`Ə,İ,Ö,Ğ,Ü,Ş,Ç,I`)** | 244 | 6 | 2.46% | **[1.13%, 5.26%]** | 9 | 3.69% | **[1.95%, 6.86%]** | 0.6015 | **Hayır (Farksız)** |
| **Standard Lowercase (`a–z`)** | 38.073 | 91 | 0.24% | **[0.19%, 0.29%]** | 175 | 0.46% | **[0.40%, 0.53%]** | **$2.76 \times 10^{-7}$** | **Evet (V2 Üstün)** |
| **Standard Uppercase (`A–Z`)** | 2.020 | 15 | 0.74% | **[0.45%, 1.22%]** | 23 | 1.14% | **[0.76%, 1.70%]** | 0.2536 | Hayır (Farksız) |
| **Grafem `i` (küçük noktalı)** | 4.386 | 16 | 0.36% | **[0.22%, 0.59%]** | 11 | 0.25% | **[0.14%, 0.45%]** | 0.4414 | Hayır (Farksız) |
| **Grafem `ı` (küçük noktasız)** | 1.688 | 10 | 0.59% | **[0.32%, 1.09%]** | 5 | 0.30% | **[0.13%, 0.69%]** | 0.3007 | Hayır (Farksız) |
| **Grafem `İ` (büyük noktalı)** | 74 | 2 | 2.70% | **[0.74%, 9.33%]** | 2 | 2.70% | **[0.74%, 9.33%]** | 1.0000 | Berabere |
| **Grafem `I` (büyük noktasız)** | 22 | 1 | 4.55% | **[0.81%, 21.80%]** | 1 | 4.55% | **[0.81%, 21.80%]** | 1.0000 | Berabere |

---

## 4. Kritik Derinlemesine İnceleme: 244 Büyük Harf Karakterinde 6 vs. 9 Hata Karşılaştırması

Makalenin önceki sürümündeki *"V2 reduces this error rate down to 2.46%, outperforming Tesseract 5 (3.69% error rate)"* iddiası istatistiksel açıdan kritik bir yanılgıdır:
1. **Mutlak Fark:** Yalnızca 3 karakterdir ($6$ hataya karşı $9$ hata; toplam $n=244$).
2. **Güven Aralıkları:** V2'nin [%1.13, %5.26] güven aralığı ile Tesseract'ın [%1.95, %6.86] güven aralığı neredeyse tamamen birbiri içine geçmektedir.
3. **Hipotez Testi (Fisher's Exact Test):** $p = 0.6015$. Bu değer $0.05$ eşiğinin fersah fersah üzerindedir.
4. **Bilimsel Sonuç:** V2'nin Tesseract 5'i büyük harf özel karakterlerinde "geride bıraktığı" iddiası istatistiksel olarak **kesinlikle savunulamaz**. Bu 3 karakterlik fark tamamen rastlantısal örneklem gürültüsüdür (chance noise). Makalede bu ifade *"Tesseract ile istatistiksel olarak eşdeğer ($p=0.60$)"* şeklinde düzeltilmelidir.

---

## 5. Resmi 295 Satırlık Benchmark: Gerçek (55) ve Simüle (240) Karakter Analizi Ayrımı

Makalede daha önce sentetik ve gerçek karakterler tek bir havuzda ($n=54.035$) toplanmıştı. Resmi 295 benchmark satırında (26.805 karakter) yapılan bağımsız ayrım:

| Karakter Sınıfı | Gerçek (55 Satır, 3.920 karakter) | Simüle (240 Satır, 22.885 karakter) |
| :--- | :--- | :--- |
| **Lowercase Special (`ə,ı,ö,ğ,ü,ş,ç`)** | **2 / 668 (%0.30)**, GA: [%0.08, %1.09] | **7 / 3.761 (%0.19)**, GA: [%0.09, %0.38] |
| **Uppercase Special (`Ə,İ,Ö,Ğ,Ü,Ş,Ç,I`)** | **0 / 5 (%0.00)**, GA: [%0.00, %43.45] | **0 / 53 (%0.00)**, GA: [%0.00, %6.76] |
| **Standard Lowercase (`a–z`)** | **7 / 2.580 (%0.27)**, GA: [%0.13, %0.56] | **39 / 15.963 (%0.24)**, GA: [%0.18, %0.33] |
| **Standard Uppercase (`A–Z`)** | **0 / 50 (%0.00)**, GA: [%0.00, %7.13] | **0 / 298 (%0.00)**, GA: [%0.00, %1.27] |
| **Rakam ve Noktalama İşaretleri** | **6 / 94 (%6.38)**, GA: [%2.96, %13.23] | **12 / 294 (%4.08)**, GA: [%2.35, %7.00] |

> **Önemli Bulgular:**
> - Gerçek 55 satırda büyük harf özel karakter sayısı yalnızca $n=5$'tir (istatistiksel genelleme yapılamaz).
> - Gerçek fotoğraflarda en büyük hata kaynağı noktalama işaretleridir (%6.38 hata; soluk virgül ve iki nokta üst üste tespiti).

---

## 6. Dörtlü `i / ı / İ / I` Grafem Karışım Matrisi (295 Satır Üzerinde)

CRNN-V2 modelinin canonical 295 benchmark satırındaki fiili grafem performansı:

| Grafem | Toplam ($n$) | Doğru | Hata | Hata Oranı (%) | **Wilson %95 GA** | Karışım Dağılımı ve Hata Tipleri |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Küçük `i`** | 2.403 | 2.380 | 23 | 0.96% | **[0.64%, 1.43%]** | 14 adet `l` (küçük L), 3 adet `İ`, 2 adet `ı`, 3 silinme, 1 ikame |
| **Küçük `ı`** | 962 | 955 | 7 | 0.73% | **[0.35%, 1.49%]** | 6 silinme, 1 ikame. **`ı` $\to$ `i` karışımı: 0!** |
| **Büyük `İ`** | 13 | 13 | 0 | **0.00%** | **[0.00%, 22.81%]** | 13/13 kusursuz tanıma (hata 0) |
| **Büyük `I`** | 4 | 4 | 0 | **0.00%** | **[0.00%, 48.99%]** | 4/4 kusursuz tanıma (hata 0) |

> **Doğrulanan Hipotez:** Model, Azerbaycan dilinin en kritik ortografik tuzağı olan noktasız `ı` karakterini noktalı `i` ile karıştırmamaktadır (`ı -> i` karışımı 0). Küçük `i` hatalarının ana kaynağı ise optik olarak benzer olan `l` (küçük L) karakteridir (14 örnek).

---

## 7. Metodolojik Sınırlılıklar ve Bilimsel Açıklamalar

1. **Altı Kümeli Bootstrap Sınırlılığı (Clustered Bootstrap Limitation):**  
   Makalede doküman bazında kümelenmiş bootstrap uygulandığı belirtilmiştir ($k=5$ veya $k=6$ doküman kümesi). İstatistik literatüründe (ör. Cameron, Gelbach & Miller 2008), küme sayısı $k < 30$ olduğunda kümelenmiş standart hatalar belirgin şekilde aşağı yönlü sapmalıdır (downward bias). Asimptotik normallik varsayımı çöker ve güven aralıkları yapay olarak dar çıkar. Bu durum raporda metodolojik bir sınırlılık olarak açıkça itiraf edilmelidir.
2. **Seen-Font Sonuçlarının Sunumu (In-Distribution Memorization):**  
   Gerçek belgeler (Photos 1–5, B1–B3) Times New Roman, Calibri ve Arial ile basılmıştır. Bu 3 font, modelin eğitim setinde de mevcuttur. Bu dokümanlardaki yüksek doğruluk (%99.7), **dağılım içi (in-distribution) parametre uyumunu** kanıtlar; modelin bilinmeyen fontlardaki "evrensel istatistiksel üstünlüğü" olarak sunulamaz.

---

## 8. 3-Seed Yeniden Eğitimleri Öncesi Kontrol Listesi (Pre-Flight Checklist)

Kullanıcı 3-seed eğitimini başlatmak istediğinde hazır olan ve denetlenen unsurlar:

- [x] **Temiz Külliyat:** `data/corpus_line_clean/` (Train 10.241, Val 93.523, Test 95.906; sıfır sızıntı).
- [x] **Temiz V1 Sentetik Verisi:** `data/synth_line_v1_clean/` (10.241 örnek; Grup A ve B için hazır).
- [x] **Temiz V2 Sentetik Verisi:** `data/synth_line_v2_clean/` (15.754 örnek; Grup C ve D için hazır).
- [x] **SymSpell Bütünlüğü:** `data/az_lexicon.txt` (SHA256 ve Git tarihi doğrulandı).
- [x] **Orijinal Checkpointlerin Korunması:** `runs/line/`, `runs/line_v2/`, `runs/ablation_*` koruma altında.
- [x] **Deney Matrisi:** 4 grup $\times$ 3 seed (42, 43, 44) = 12 bağımsız eğitim koşulu tanımlandı.
- [x] **Değerlendirme Protokolü:** Canonical 295 benchmark satırı üzerinde ANOVA ve $\mu \pm \sigma$ hesaplama altyapısı hazır.
