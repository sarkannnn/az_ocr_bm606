# EasyOCR 15-Sənədlik (295 Sətir) Rəsmi Benchmark Qiymətləndirmə Hesabatı

Bu hesabat AzOCR layihəsinin rəsmi 15 telefon sənədi benchmarkı (295 sətir) üzərində **EasyOCR 1.7.2** modelinin birbaşa qiymətləndirmə nəticələrini təqdim edir.

## 1. Model Məlumatı və Azərbaycan Dili Dəstəyi

- **EasyOCR Versiyası:** `1.7.2`
- **Konfiqurasiya Edilmiş Dil:** `['az']`
- **İstifadə Edilən Tanıyıcı (Recognizer):** `latin_g2` (1,388,416 parametr, 14.69 MB)
- **İstifadə Edilən Detektor (Detector):** `craft_mlt_25k` (20,770,466 parametr, 79.3 MB)
- **Doğma Azərbaycan Modeli Mövcuddurmu?:** **XEYR.** EasyOCR kitabxanasında Azərbaycan dili üçün ayrıca ixtisaslaşmış təlim çəkisi yoxdur. `'az'` dil kodu birbaşa 341 simvolluq ümumi `latin_g2` Latın modelinə yönləndirilir. Bu səbəbdən modeldə Azərbaycan dili üçün lüğət, n-gram və ya morfoloji dil modeli iştirak etmir.

## 2. Ümumi Nəticələr və Alt-Qrup Bölgüsü

| Qrup | Sətir Sayı | Simvol Sayı | CER (%) ↓ | Simvol Dəqiqliyi (%) ↑ | WER (%) ↓ | Tam Düzgün Sətir (Exact Match) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bütün Benchmark (B1–B15)** | **295** | 26,805 | **2.08%** | 97.92% | **16.81%** | **80/295 (27.1%)** |
| Real Telefon Fotoları (B1–B3) | 55 | 3,920 | **1.84%** | 98.16% | **9.89%** | **21/55 (38.2%)** |
| Simulyasiya Fotoları (B4–B15) | 240 | 22,885 | **2.12%** | 97.88% | **18.25%** | **59/240 (24.6%)** |

## 3. Digər Modellərlə Müqayisə (B1–B15, n = 295 Sətir)

| Model / Alqoritm | CER (%) ↓ | WER (%) ↓ | Exact Match ↑ | Orta Latensiya (ms) ↓ | Model Həcmi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CRNN-V2 + SymSpell (Təklif olunan)** | **0.29%** | **1.37%** | **259/295 (87.8%)** | 86.4 ms | 98.7 MB |
| CRNN-V2 (Xam CTC) | **0.30%** | **1.64%** | **253/295 (85.8%)** | 86.3 ms | 98.7 MB |
| Tesseract 5 LSTM (`aze`, `--psm 7`) | **0.31%** | **1.55%** | **250/295 (84.7%)** | 182.2 ms | 5.99 MB |
| Tesseract 5 LSTM (`aze`, `--psm 13`) | **0.47%** | **2.40%** | **230/295 (78.0%)** | 141.4 ms | 5.99 MB |
| **EasyOCR 1.7.2 (`az` / `latin_g2`)** | **2.08%** | **16.81%** | **80/295 (27.1%)** | 172.0 ms | 94.0 MB |
| CRNN-V1 (Baza — Augmentasiyasız) | **12.37%** | **42.49%** | 37/295 (12.5%) | 85.2 ms | 32.9 MB |

## 4. B1–B15 Sənədləri Üzrə Təfərrüatlı Nəticələr

| Sənəd Kodu | Tip | Şrift | Təhrif Dərəcəsi | Sətir | CER (%) | WER (%) | Exact Match | Latensiya (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** | Real Telefon (IMG_5121) | Times New Roman | Real L2 | 16 | 2.15% | 11.06% | 3/16 (18.8%) | 173.3 ms |
| **B2** | Real Telefon (IMG_5122) | Calibri | Real L2 | 16 | 2.32% | 11.85% | 2/16 (12.5%) | 170.8 ms |
| **B3** | Real Telefon (IMG_5123) | Times New Roman | Real L3 | 23 | 0.69% | 5.07% | 16/23 (69.6%) | 77.1 ms |
| **B4** | Simulyasiya (IMG_5124) | Arial | L2 | 20 | 0.77% | 10.3% | 7/20 (35.0%) | 188.7 ms |
| **B5** | Simulyasiya (IMG_5125) | Tahoma | L1 | 20 | 0.97% | 17.14% | 2/20 (10.0%) | 165.6 ms |
| **B6** | Simulyasiya (IMG_5126) | Cambria | L2 | 20 | 0.42% | 3.42% | 13/20 (65.0%) | 160.0 ms |
| **B7** | Simulyasiya (IMG_5127) | MS Sans Serif | L3 | 20 | 1.68% | 20.43% | 2/20 (10.0%) | 163.9 ms |
| **B8** | Simulyasiya (IMG_5128) | Tahoma | L4 | 20 | 1.86% | 22.42% | 1/20 (5.0%) | 165.4 ms |
| **B9** | Simulyasiya (IMG_5129) | Consolas | L1 | 20 | 1.04% | 3.96% | 12/20 (60.0%) | 230.1 ms |
| **B10** | Simulyasiya (IMG_5130) | Cambria | L2 | 20 | 0.69% | 5.49% | 10/20 (50.0%) | 160.4 ms |
| **B11** | Simulyasiya (IMG_5131) | Verdana | L3 | 20 | 5.47% | 42.15% | 0/20 (0.0%) | 180.2 ms |
| **B12** | Simulyasiya (IMG_5132) | Courier New | L2 | 20 | 7.44% | 40.52% | 0/20 (0.0%) | 251.9 ms |
| **B13** | Simulyasiya (IMG_5133) | Segoe UI | L3 | 20 | 2.72% | 20.66% | 3/20 (15.0%) | 159.1 ms |
| **B14** | Simulyasiya (IMG_5134) | MS Sans Serif | L4 | 20 | 1.16% | 16.13% | 6/20 (30.0%) | 182.1 ms |
| **B15** | Simulyasiya (IMG_5135) | Candara | L4 | 20 | 1.43% | 15.85% | 3/20 (15.0%) | 165.8 ms |

## 5. Xüsusi Qrafemlər və Diakritik Hərflər Auditi

| Qrafem / Simvol Kateqoriyası | Nümunə Sayı (n) | EasyOCR Xəta Sayı | EasyOCR Xəta (%) | EasyOCR Dəqiqlik (%) |
| :--- | :---: | :---: | :---: | :---: |
| Kiçik Xüsusi Hərflər (ə,ı,ö,ğ,ü,ş,ç) | 4429 | 235 | 5.31% | 94.69% |
| Böyük Xüsusi Hərflər (Ə,İ,Ö,Ğ,Ü,Ş,Ç,I) | 58 | 4 | 6.9% | 93.1% |
| 4-Tərəfli i / ı / İ / I Ailəsi | 3382 | 97 | 2.87% | 97.13% |
| Ə hərfi (ə / Ə) | 2263 | 81 | 3.58% | 96.42% |
| Standart Kiçik Hərflər (a–z) | 18543 | 125 | 0.67% | 99.33% |
| Standart Böyük Hərflər (A–Z) | 352 | 1 | 0.28% | 99.72% |
| Rəqəmlər və Durğu İşarələri | 427 | 40 | 9.37% | 90.63% |
