# 15 Müstəqil Smartfon Sənəd Fotosu (`IMG_5121` – `IMG_5135`) üzrə Eksperimental Audit Hesabatı

Bu hesabat məqalənin (**"Robust Deep Text Recognition for Azerbaijani Script Under Smartphone Distortions: Photometric Augmentations and Empirical Audit"**) metodologiyasına uyğun olaraq **15 tam fərqli mətnli telefon sənəd fotosu (`Photo 1 / IMG_5121` – `Photo 15 / IMG_5135`, cəmi 295 sətir)** üzərində aparılmış kompleks qiymətləndirməni əks etdirir.

## 1. Ümumi Yekun Müqayisə Cədvəli (15 Telefon Fotosu, n = 295 Sətir, 26,805 Simvol)

| Model / Konfiqurasiya | Simvol Doğruluğu (%) ↑ | Söz Doğruluğu (%) ↑ | CER (%) ↓ | WER (%) ↓ | Tam Düzgün Sətir ↑ | Orta Müddət (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2 + Morfoloji SymSpell (Təklif olunan)** | **99.71%** | **98.63%** | **0.29%** | 1.37% | 259/295 (87.8%) | 86.4 ms |
| CRNN-V2 (Xam CTC Dekodlama) | **99.70%** | **98.36%** | **0.30%** | 1.64% | 253/295 (85.8%) | 86.3 ms |
| CRNN-V1 (Baza Arxitekturası — Fotometrik Aug. olmadan) | **87.63%** | **57.51%** | **12.37%** | 42.49% | 37/295 (12.5%) | 85.2 ms |
| Tesseract 5 LSTM (`aze`, `--psm 7`) | **99.69%** | **98.45%** | **0.31%** | 1.55% | 250/295 (84.7%) | 182.2 ms |
| Tesseract 5 LSTM (`aze`, `--psm 13`) | **99.53%** | **97.60%** | **0.47%** | 2.40% | 230/295 (78.0%) | 141.4 ms |

## 2. Hər Bir Telefon Fotosu (`Photo 1` – `Photo 15`) üzrə Fərdi Nəticələr

| Telefon Fotosu | Mövzu (100% Fərqli Mətn) | Şrift Ailəsi | Təhrif | Sətir | CRNN-V2+SymSpell Doğruluq (CER) | CRNN-V1 Doğruluq (CER) | Tesseract 5 Doğruluq (CER) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Photo 1 (`IMG_5121.jpg`)** | Xalçaçı Usta Əli və Nəvəsi Murad (Real | Times New Roman (Real Print) | `Real Phone (L2)` | 16 | **99.38%** (0.62%, 11/16) | 81.83% (18.17%) | 98.68% (1.32%) |
| **Photo 2 (`IMG_5122.jpg`)** | Gəncə Çayı və Dulusçu Qoca Səməndər (R | Calibri (Real Print) | `Real Phone (L2)` | 16 | **99.59%** (0.41%, 12/16) | 90.64% (9.36%) | 99.86% (0.14%) |
| **Photo 3 (`IMG_5123.jpg`)** | Milli Sənətkarlıq İrsi və Tarixi Yadda | Times New Roman (Real Print) | `Real Phone (L3)` | 23 | **99.11%** (0.89%, 20/23) | 95.67% (4.33%) | 100.00% (0.00%) |
| **Photo 4 (`IMG_5124.jpg`)** | Qobustan Qayaüstü Rəsmləri və İbtidai  | Arial (Sans-Serif) | `L2` | 20 | **100.00%** (0.00%, 20/20) | 88.57% (11.43%) | 99.85% (0.15%) |
| **Photo 5 (`IMG_5125.jpg`)** | Rəqəmsal İdarəetmə və Süni İntellekt T | Tahoma (Unseen Group C) | `L1` | 20 | **99.95%** (0.05%, 19/20) | 99.57% (0.43%) | 99.89% (0.11%) |
| **Photo 6 (`IMG_5126.jpg`)** | Şəki Xan Sarayı və Memarlıq İrsi | Cambria (Unseen Group C) | `L2` | 20 | **100.00%** (0.00%, 20/20) | 85.19% (14.81%) | 99.90% (0.10%) |
| **Photo 7 (`IMG_5127.jpg`)** | Göygöl Milli Parkı və Biomüxtəliflik | MS Sans Serif (Unseen Group C) | `L3` | 20 | **100.00%** (0.00%, 20/20) | 91.77% (8.23%) | 99.78% (0.22%) |
| **Photo 8 (`IMG_5128.jpg`)** | Azərbaycan Əlifbası və Fonetik Sistem | Tahoma (Unseen Group C) | `L4` | 20 | **100.00%** (0.00%, 20/20) | 93.40% (6.60%) | 99.84% (0.16%) |
| **Photo 9 (`IMG_5129.jpg`)** | Kibertəhlükəsizlik və Kriptoqrafik Şif | Consolas (Held-Out Monospace) | `L1` | 20 | **99.27%** (0.73%, 17/20) | 92.55% (7.45%) | 99.22% (0.78%) |
| **Photo 10 (`IMG_5130.jpg`)** | İçərişəhər Memarlıq Kompleksi və Qız Q | Cambria (Unseen Group C) | `L2` | 20 | **100.00%** (0.00%, 20/20) | 86.83% (13.17%) | 100.00% (0.00%) |
| **Photo 11 (`IMG_5131.jpg`)** | Xəzər Dənizinin Ekoloji Problemləri və | Verdana (Sans-Serif) | `L3` | 20 | **99.42%** (0.58%, 12/20) | 81.82% (18.18%) | 99.58% (0.42%) |
| **Photo 12 (`IMG_5132.jpg`)** | Naxçıvan Muxtar Respublikasının Tarixi | Courier New (Monospace) | `L2` | 20 | **98.61%** (1.39%, 9/20) | 58.03% (41.97%) | 98.88% (1.12%) |
| **Photo 13 (`IMG_5133.jpg`)** | Azərbaycan Muğam Sənəti və Milli Musiq | Segoe UI (Sans-Serif) | `L3` | 20 | **100.00%** (0.00%, 20/20) | 87.80% (12.20%) | 99.90% (0.10%) |
| **Photo 14 (`IMG_5134.jpg`)** | Azərbaycan Kosmik Sənayesi və Peyk Müş | MS Sans Serif (Unseen Group C) | `L4` | 20 | **100.00%** (0.00%, 20/20) | 88.53% (11.47%) | 99.95% (0.05%) |
| **Photo 15 (`IMG_5135.jpg`)** | Qarabağ Xalçaçılıq Məktəbi və Ornament | Candara (Humanist Sans) | `L4` | 20 | **99.95%** (0.05%, 19/20) | 94.84% (5.16%) | 99.95% (0.05%) |

## 3. Azərbaycan Xüsusi Diakritik Hərfləri və `i / ı / İ / I` Auditi (15 Telefon Fotosu)

| Hərf Qrupu / Qrafem | Nümunə Sayı (n) | CRNN-V2 + SymSpell Doğruluq (Xəta %) | CRNN-V1 Doğruluq (Xəta %) | Tesseract 5 Doğruluq (Xəta %) |
| :--- | :---: | :---: | :---: | :---: |
| Kiçik Xüsusi Hərflər (ə,ı,ö,ğ,ü,ş,ç) | 4429 | **99.77%** (0.23%, 10 səhv) | 88.33% (11.67%, 517 səhv) | 99.82% (0.18%, 8 səhv) |
| Böyük Xüsusi Hərflər (Ə,İ,Ö,Ğ,Ü,Ş,Ç,I) | 58 | **100.00%** (0.00%, 0 səhv) | 81.03% (18.97%, 11 səhv) | 98.28% (1.72%, 1 səhv) |
| 4-Tərəfli i / ı / İ / I Ailəsi | 3382 | **99.35%** (0.65%, 22 səhv) | 88.68% (11.32%, 383 səhv) | 99.65% (0.35%, 12 səhv) |
| Ə hərfi (ə / Ə) | 2263 | **99.96%** (0.04%, 1 səhv) | 94.03% (5.97%, 135 səhv) | 100.00% (0.00%, 0 səhv) |
