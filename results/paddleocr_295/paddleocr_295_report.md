# PaddleOCR 15-Sənədlik (295 Sətir) Rəsmi Benchmark Qiymətləndirmə Hesabatı

Bu hesabat AzOCR layihəsinin rəsmi 15 telefon sənədi benchmarkı (295 sətir) üzərində **PaddleOCR 3.7.0 / PaddlePaddle 3.3.1 (PP-OCRv6_medium_rec)** modelinin qiymətləndirmə nəticələrini təqdim edir.

## 1. Model Məlumatı və Texniki İcra Təfərrüatları

- **PaddleOCR Versiyası:** `3.7.0`
- **PaddlePaddle Versiyası:** `3.3.1` (Python 3.13.9, Windows x64)
- **İstifadə Edilən Model:** `PP-OCRv6_medium_rec` (72.9 MB model çəkisi)
- **Lüğət Həcmi:** 18,708 simvol (Azərbaycan xüsusi hərflərinin tamamı: `ə, ı, ö, ğ, ü, ş, ç, Ə, İ, Ö, Ğ, Ü, Ş, Ç, I` daxildir)
- **Texniki Müşahidə və Həll:** PaddleOCR-in tam səhifə detektoru (`PP-OCRv6_medium_det`) Windows x64 CPU mühitində oneDNN PIR instruktsiya uyğunsuzluğu (`NotImplementedError in onednn_instruction.cc:118`) səbəbilə xəta verir. Lakin sətir kəsimləri birbaşa `PP-OCRv6_medium_rec` tanıyıcısına verildikdə model uğurla və sabit şəkildə icra olunur.

## 2. Ümumi Nəticələr və Alt-Qrup Bölgüsü

| Qrup | Sətir Sayı | Simvol Sayı | CER (%) ↓ | Simvol Dəqiqliyi (%) ↑ | WER (%) ↓ | Tam Düzgün Sətir (Exact Match) ↑ | Orta Latensiya (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bütün Benchmark (B1–B15)** | **295** | 26,805 | **0.91%** | 99.09% | **5.19%** | **205/295 (69.5%)** | 1888.1 ms |
| Real Telefon Fotoları (B1–B3) | 55 | 3,920 | **0.41%** | 99.59% | **2.65%** | **42/55 (76.4%)** | 1560.4 ms |
| Simulyasiya Fotoları (B4–B15) | 240 | 22,885 | **0.99%** | 99.01% | **5.72%** | **163/240 (67.9%)** | 1963.2 ms |

## 3. Bütün Modellərin Hərtərəfli Müqayisəsi (B1–B15, n = 295 Sətir)

| Model / Alqoritm | CER (%) ↓ | WER (%) ↓ | Exact Match ↑ | Orta Latensiya (ms) ↓ | Model Həcmi | Azərbaycan Lüğəti |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2 + SymSpell (Təklif olunan)** | **0.29%** | **1.37%** | **259/295 (87.8%)** | **86.4 ms** | 98.7 MB | Morfoloji SymSpell (116k) |
| CRNN-V2 (Xam CTC) | **0.30%** | **1.64%** | **253/295 (85.8%)** | **86.3 ms** | 98.7 MB | Xam CTC |
| Tesseract 5 LSTM (`aze`, `--psm 7`) | **0.31%** | **1.55%** | **250/295 (84.7%)** | 182.2 ms | 5.99 MB | `aze.traineddata` DAWG |
| Tesseract 5 LSTM (`aze`, `--psm 13`) | **0.47%** | **2.40%** | **230/295 (78.0%)** | 141.4 ms | 5.99 MB | `aze.traineddata` DAWG |
| **PaddleOCR PP-OCRv6 (`PP-OCRv6_rec`)** | **0.91%** | **5.19%** | **205/295 (69.5%)** | 1888.1 ms | 72.9 MB | 18k Simvol Lüğəti |
| EasyOCR 1.7.2 (`latin_g2`) | **2.08%** | **16.81%** | 80/295 (27.1%) | 172.0 ms | 94.0 MB | Yoxdur (Ümumi Latin) |
| CRNN-V1 (Baza — Augmentasiyasız) | **12.37%** | **42.49%** | 37/295 (12.5%) | 85.2 ms | 32.9 MB | Xam CTC |

## 4. B1–B15 Sənədləri Üzrə Təfərrüatlı Nəticələr

| Sənəd Kodu | Tip | Şrift | Təhrif Dərəcəsi | Sətir | CER (%) | WER (%) | Exact Match | Latensiya (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Real Telefon (IMG_5121) | Times New Roman | Real L2 | 16 | 0.62% | 3.23% | 10/16 (62.5%) | 1851.3 ms |
| **B2** | Real Telefon (IMG_5122) | Calibri | Real L2 | 16 | 0.41% | 3.32% | 10/16 (62.5%) | 1646.1 ms |
| **B3** | Real Telefon (IMG_5123) | Times New Roman | Real L3 | 23 | 0.1% | 0.72% | 22/23 (95.7%) | 1183.7 ms |
| **B4** | Simulyasiya (IMG_5124) | Arial | L2 | 20 | 0.0% | 0.0% | 20/20 (100.0%) | 1718.2 ms |
| **B5** | Simulyasiya (IMG_5125) | Tahoma | L1 | 20 | 0.0% | 0.0% | 20/20 (100.0%) | 1748.3 ms |
| **B6** | Simulyasiya (IMG_5126) | Cambria | L2 | 20 | 0.0% | 0.0% | 20/20 (100.0%) | 1780.3 ms |
| **B7** | Simulyasiya (IMG_5127) | MS Sans Serif | L3 | 20 | 0.7% | 4.78% | 14/20 (70.0%) | 1832.9 ms |
| **B8** | Simulyasiya (IMG_5128) | Tahoma | L4 | 20 | 0.27% | 2.24% | 15/20 (75.0%) | 1990.1 ms |
| **B9** | Simulyasiya (IMG_5129) | Consolas | L1 | 20 | 4.06% | 22.77% | 1/20 (5.0%) | 2205.4 ms |
| **B10** | Simulyasiya (IMG_5130) | Cambria | L2 | 20 | 0.11% | 0.84% | 18/20 (90.0%) | 1950.0 ms |
| **B11** | Simulyasiya (IMG_5131) | Verdana | L3 | 20 | 1.21% | 8.97% | 7/20 (35.0%) | 2131.2 ms |
| **B12** | Simulyasiya (IMG_5132) | Courier New | L2 | 20 | 4.01% | 20.69% | 2/20 (10.0%) | 2317.0 ms |
| **B13** | Simulyasiya (IMG_5133) | Segoe UI | L3 | 20 | 0.21% | 1.65% | 17/20 (85.0%) | 2040.3 ms |
| **B14** | Simulyasiya (IMG_5134) | MS Sans Serif | L4 | 20 | 0.7% | 5.99% | 10/20 (50.0%) | 2019.8 ms |
| **B15** | Simulyasiya (IMG_5135) | Candara | L4 | 20 | 0.66% | 2.85% | 19/20 (95.0%) | 1956.1 ms |

## 5. Xüsusi Qrafemlər və Diakritik Hərflər Auditi

| Qrafem / Simvol Kateqoriyası | Nümunə Sayı (n) | PaddleOCR Xəta Sayı | PaddleOCR Xəta (%) | PaddleOCR Dəqiqlik (%) |
| :--- | :---: | :---: | :---: | :---: |
| Kiçik Xüsusi Hərflər (ə,ı,ö,ğ,ü,ş,ç) | 4429 | 198 | 4.47% | 95.53% |
| Böyük Xüsusi Hərflər (Ə,İ,Ö,Ğ,Ü,Ş,Ç,I) | 58 | 1 | 1.72% | 98.28% |
| 4-Tərəfli i / ı / İ / I Ailəsi | 3382 | 176 | 5.2% | 94.8% |
| Ə hərfi (ə / Ə) | 2263 | 12 | 0.53% | 99.47% |
| Standart Kiçik Hərflər (a–z) | 18543 | 23 | 0.12% | 99.88% |
| Standart Böyük Hərflər (A–Z) | 352 | 0 | 0.0% | 100.0% |
| Rəqəmlər və Durğu İşarələri | 427 | 16 | 3.75% | 96.25% |
