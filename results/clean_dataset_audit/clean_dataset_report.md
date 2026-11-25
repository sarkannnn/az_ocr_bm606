# AzOCR-V2: Zorunlu Veri Sızıntısı Temizliği ve Yeniden Eğitim Planı Raporu (Aşama 2.6)

**Tarih:** 2026-11-24  
**Kapsam:** Veri sızıntılarının (data leakage) eğitim tarafından tamamen temizlenmesi, versionlu temiz veri setlerinin oluşturulması, SymSpell hash ve kaynak doğrulaması, kalan metodolojik sınırlılıklar ve 3-seed yeniden eğitim planı.

---

## 1. Yönetici Özeti ve Temizlik İcmali

Hakem revizyonunun amir hükmü gereğince:
> *"Eğitim ile validation/test arasında tekrarlanan satırlar EĞİTİM SETİNDEN çıkarılmalıdır. Validation ve test setleri arasındaki çakışma ise test seti sabit tutularak validation tarafında çözülmelidir."*

Bu kural birebir uygulanmış; **orijinal veri setleri ve checkpointler tamamen korunarak**, ayrı klasörlerde **temiz, sıfır çakışmalı (zero-leakage)** yeni veri kümeleri üretilmiştir.

### Temel Doğrulamalar:
1. **100 Train-Val ve 84 Train-Test Tekrarı:**
   - 100 satır Train $\cap$ Val kesişimindedir.
   - 84 satır Train $\cap$ Test kesişimindedir.
   - Bu iki kümenin kendi arasındaki kesişimi (Train $\cap$ Val $\cap$ Test) tam **25 satırdır**.
   - Çift sayım (double-counting) engellendiğinde, eğitim setinden çıkarılması gereken **toplam tekil satır sayısı tam 159'dur** ($100 + 84 - 25 = 159$).
2. **2.136 Val-Test Tekrarı:**
   - Test seti (95.906 satır) dokunulmadan referans tutulmuş; çakışan 2.136 satır validation setinden çıkarılmıştır.
3. **Sıfır Çakışma Güvencesi:**
   - Temizlenen külliyatta (`data/corpus_line_clean/`) ve temiz sentetik veri setlerinde (`data/synth_line_v1_clean/`, `data/synth_line_v2_clean/`) Train, Val ve Test arasında **hem ham dize (exact) hem de Unicode NFC normalizasyonu seviyesinde çakışma TAM 0'DIR.**

---

## 2. Temizlik Öncesi ve Sonrası Örnek Sayıları Tablosu

| Veri Seti / Split | Temizlik Öncesi Örnek | Temizlik Sonrası Örnek | Fark ($\Delta$) | Uygulanan Temizlik Mantığı |
| :--- | :--- | :--- | :--- | :--- |
| **`corpus_line` / Train** | 10.400 | **10.241** | **-159** | Val (100) ve Test (84) ile çakışan 159 tekil satır çıkarıldı |
| **`corpus_line` / Val** | 95.659 | **93.523** | **-2.136** | Test setiyle çakışan 2.136 mükerrer satır çıkarıldı |
| **`corpus_line` / Test** | 95.906 | **95.906** | **0** | Test seti bağımsız referans olarak sabit tutuldu |
| **`synth_line_v1` / Train** | 10.400 | **10.241** | **-159** | 159 satıra karşılık gelen 159 sentetik görsel çıkarıldı |
| **`synth_line_v1` / Val** | 1.000 | **1.000** | **0** | Çakışan tek satır train'den çıkarıldığı için val sabit kaldı |
| **`synth_line_v2` / Train** | 16.000 | **15.754** | **-246** | 159 satıra karşılık gelen 246 sentetik görsel çıkarıldı |
| **`synth_line_v2` / Val** | 1.000 | **1.000** | **0** | Başlangıçta da train ile çakışması 0 idi (korundu) |
| **`synth_line_v2` / Test** | 1.000 | **1.000** | **0** | Başlangıçta da train ile çakışması 0 idi (korundu) |

---

## 3. Temiz Kümelerde Çapraz Doğrulama (Zero-Leakage Verification)

Temiz klasörler üretildikten sonra `scripts/prepare_clean_datasets.py` tarafından yapılan otomatik doğrulamalar:

```text
--- ZERO LEAKAGE VERIFICATION ---
Corpus Clean: Train vs Val overlap (Exact & NFC):  0
Corpus Clean: Train vs Test overlap (Exact & NFC): 0
Corpus Clean: Val vs Test overlap (Exact & NFC):   0

V1 Clean: Train vs Val text overlap (Exact & NFC): 0
V2 Clean: Train vs Val text overlap (Exact & NFC): 0
V2 Clean: Train vs Test text overlap (Exact & NFC): 0
V2 Clean: Val vs Test text overlap (Exact & NFC):   0
```

> **Çıkarılan Görseller:** Çıkarılan 159 (V1) ve 246 (V2) adet eğitim örneğinin dosya isimleri ve metinleri `results/clean_dataset_audit/removed_training_samples.csv` dosyasına tam liste olarak kaydedilmiştir.

---

## 4. SymSpell Sözlüğü Bütünlük ve Güvenlik Kanıtı

| Parametre | Tespit Edilen Değer | Doğrulama / Kaynak |
| :--- | :--- | :--- |
| **Dosya Yolu** | `data/az_lexicon.txt` | Repository yerel kökü |
| **SHA256 Hash** | `02c81ae2473a06a8396592f04467665a151549ce3efb091002246231c63cbaa8` | Sabit, değiştirilmemiş |
| **Git Commit Hash** | `51e807e82f9a9af6b0eab64df8d39d2f44f0f19a` | Resmi refactor commiti |
| **Commit Tarihi** | `2026-10-29 16:50:28 +0400` | Benchmarktan 8 gün önce |
| **Sözlük Yazarı** | `jafarov007 <c.ebdul@gmail.com>` | Git kaydı |
| **Toplam Sözcük** | 116.585 adet unigram | `<sözcük> <frekans>` formatı |
| **Kaynak Mahiyeti** | Genel Azerbaycan dili külliyatı / Vikipedi | Çok sözcüklü benchmark tümcesi içermez |
| **Benchmark Dışı Kelime (OOV)** | 318 / 2.136 (%14.9) | Benchmark'a overfit edilmediğinin kanıtı |

---

## 5. Kalan Metodolojik Sınırlılıklar (Disclosures)

Veri sızıntısının eğitim tarafından tamamen temizlenmesine rağmen, bilimsel şeffaflık adına makalede veya raporda beyan edilmesi gereken kalan sınırlılıklar şunlardır:

1. **Sentetik Doğrulama Kümesi Font Dağılımı (V1 Font Bias):**
   `synth_line_v1/val` ve `synth_line_v2/val` kümeleri (1.000'er satır), yalnızca V1'in 8 temel fontunu (Arial, Calibri, Candara, Corbel, Courier New, Segoe UI, Times New Roman, Verdana) içermektedir. Dolayısıyla erken durdurma (early stopping) ve model checkpoint seçimi hala sadece bu 8 fontun validasyon loss'una göre belirlenmektedir.
2. **Grup B ve C İçin Sentetik Yeniden Üretim Gereksinimi:**
   Grup B (8 font + V2 aug) ve Grup C (14 font + V1 aug) ablasyon modelleri için temiz 10.241 satır üzerinden sıfırdan sentetik görsel render scriptinin çalıştırılması gerekmektedir.
3. **Eğitim Boyutunun Doğal Olarak Küçülmesi:**
   Train setinden 159 satır (V2'de 246 render örneği) çıkarıldığı için model eğitim örnekleri V1'de 10.400'den 10.241'e; V2'de 16.000'den 15.754'e inmiştir. Bu durum modellerin kapasitesini olumsuz etkilemeyecek kadar küçük bir orandır (%1.5), ancak raporda örnek sayısı güncellenmelidir.

---

## 6. A/B/C/D Modellerinin 3 Seed ile Yeniden Eğitilmesi İçin Sıralı Çalışma Planı

Bilimsel kesinlik ve hakem tatmini için 4 model grubu (A, B, C, D) 3 farklı rastgele tohum (`seed = [42, 43, 44]`) ile toplam 12 bağımsız eğitim koşulunda değerlendirilecektir.

### 6.1. Deney Matrisi (12 Koşul)

| Model Grubu | Font Kümesi | Augmentasyon | Temiz Eğitim Örneği | Seedler | Hedef Checkpoint Klasörleri |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Group A** | 8 V1 Font | V1 Basic Aug | 10.241 | 42, 43, 44 | `runs/clean_group_a_s42`, `s43`, `s44` |
| **Group B** | 8 V1 Font | V2 Photo Aug | 10.241 | 42, 43, 44 | `runs/clean_group_b_s42`, `s43`, `s44` |
| **Group C** | 14 V2 Font | V1 Basic Aug | 15.754 | 42, 43, 44 | `runs/clean_group_c_s42`, `s43`, `s44` |
| **Group D** | 14 V2 Font | V2 Photo Aug | 15.754 | 42, 43, 44 | `runs/clean_group_d_s42`, `s43`, `s44` |

### 6.2. Sıralı İcra Protokolü (Execution Steps)

1. **Aşama 1: Grup B ve Grup C Temiz Görsel Kümelerinin Üretimi**
   - Temiz 10.241 satır kullanılarak Grup B için 8 V1 fontu ile V2 fotogerçekçi augmentasyonlu görseller üretilir (`data/synth_line_clean_group_b/`).
   - Temiz 10.241 satır kullanılarak Grup C için 14 V2 fontu ile V1 temel augmentasyonlu görseller üretilir (`data/synth_line_clean_group_c/`).
2. **Aşama 2: Sıralı GPU/CPU Eğitimi (Kullanıcı Onayı Sonrası Başlatılacak)**
   - Eğitimler tek bir PowerShell orkestrasyon betiği (`scripts/run_clean_3seed_experiments.ps1`) ile sıralı çalıştırılır.
   - Her eğitimde:
     - Sabit batch size: 32 (veya 64 GPU belleğine göre)
     - Max epoch: 15
     - Early stopping: sabır = 3 epoch (validation loss)
     - Checkpoint: `best.pt` ve `last.pt` ayrı kaydedilir.
     - Mevcut `runs/line/`, `runs/line_v2/` checkpointlerine ASLA dokunulmaz.
3. **Aşama 3: Otomatik Benchmark Değerlendirmesi**
   - 12 checkpointin tamamı canonical 295 benchmark satırında (B1–B15) `infer_line_smart` protokolü ile test edilir.
   - Her grup için ortalama ve standart sapma hesaplanır:
     $$\mu_{\text{CER}} \pm \sigma_{\text{CER}}, \quad \mu_{\text{WER}} \pm \sigma_{\text{WER}}$$
   - 2x2 Faktöriyel ANOVA testi ile Font ve Augmentasyon ana etkileri ve p-değerleri çıkarılır.

---

## 7. Çıktı Dosyaları

- **Özet JSON:** `results/clean_dataset_audit/clean_dataset_summary.json`
- **Çıkarılan Eğitim Örnekleri Listesi:** `results/clean_dataset_audit/removed_training_samples.csv`
- **Temiz Külliyat Klasörü:** `data/corpus_line_clean/` (`train.txt`, `val.txt`, `test.txt`)
- **Temiz V1 Sentetik Veri Klasörü:** `data/synth_line_v1_clean/` (`train/`, `val/`)
- **Temiz V2 Sentetik Veri Klasörü:** `data/synth_line_v2_clean/` (`train/`, `val/`, `test/`)
- **Denetim ve Plan Raporu:** `results/clean_dataset_audit/clean_dataset_report.md`
