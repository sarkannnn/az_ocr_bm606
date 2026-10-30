# Real Fix 9: Photo 5 (`IMG_5130.HEIC`) Həssaslıq, İnteqrasiya və Xəta Təhlili Hesabatı

Bu hesabat istifadəçi tərəfindən təqdim edilmiş `IMG_5130.HEIC` (3024x4032, 33 sətir, 5 paraqraf) fotosu üzərində Tesseract 5-in seqmentasiya uğursuzluğunu (boş sətir xətaları), ağ haşiyə həssaslığını (`sensitivity, not selected`), nisbət qruplaşdırmasını ($W\cdot 32 / (h\cdot 512)$) və təmiz alt-çoxluq (`clean subset`) üzrə müqayisəli dəqiqlik nəticələrini əks etdirir.

## 1. Metodoloji Açıqlama və Şəkillərin Təsnifatı

### 1.1 Sənəd Nömrələnməsi (Photos 1 – 5 Haradadır?):
- **Photo 1 (`results/real_test/`, `results/real_fix2-8/`):** İstifadəçinin ilk yüklədiyi 16 sətirlik sənəd fotosu (sol tərəfində kölgə zolağı və em-dash olan mətn).
- **Photo 2 (`results/real_fix2/` – `results/real_fix8/`):** İkinci sənəd fotosu (16 sətir, `kasalar, bəzəkli küplər...` mətni).
- **Photo 3 (`results/photo3_test/`):** Serif / Times New Roman şriftli 23 sətirlik çap sənədi (`media_1793351399184.jpg`).
- **Photo 4 (`results/photo4_sans/`):** Sans-serif / Arial şriftli 23 sətirlik çap sənədi (`media_1793353856829.jpg`).
- **Photo 5 (`results/photo5_heic_test/`, `results/real_fix9/`):** Təqdim edilmiş `C:\Users\serxan.m\Downloads\IMG_5130.HEIC` (bütün 33 sətirlik əsas hekayə mətnini ehtiva edən tam səhifə fotosu).

### 1.2 Sətir Qutuları və Sərhədlərin Əldə Edilməsi:
- **Avtomatik Aşkar:** Tesseract `image_to_data` vasitəsilə söz qutuları çıxarılmış, `(block_num, par_num, line_num)` üzrə qruplaşdırılaraq 33 sətir aşkar edilmişdir.
- **Şaquli Sərhədlər ($y_0, y_1$):** Hər sətir üçün mətndən yuxarı və aşağı 4 px marja (`top - 4`, `bottom + 4`) əlavə edilmişdir.
- **Üfüqi Sərhədlər ($x_0, x_1$):** Sol və sağ tərəfə 12 px marja tətbiq edilmişdir.
- **Fərdiləşdirilmiş Tənzimləmə:** Sənəddə yalnız Line 12-də (0-indeks 11) sol kənarda ($x=395$) olan printer ləkəsinin mətndən kənarda qalması üçün $x_0=430$ təyin edilmişdir. Qalan 32 sətir tam avtomatik kəsilmişdir.
- **Referansların Uyğunlaşdırılması:** Çap olunmuş 33 sətirlik referans mətn yuxarıdan aşağıya oxu ardıcıllığı ilə aşkarlanmış 33 sətirə 1-ə 1 dəqiq uyğunlaşdırılmışdır.

## 2. Boş Çıxışların Qeydiyyatı və İnteqrasiya Uğursuzluğu (PSM 7 vs PSM 13)

- **V1 Line:** Şərt A-da 0, Şərt B-də 0 boş çıxış.
- **V2 Line:** Şərt A-da 0, Şərt B-də 0 boş çıxış.
- **Tesseract 5:** Şərt A-da **11 boş çıxış** (9 ədəd 2-ci parça: Sətir 18, 19, 21, 22, 23, 24, 26, 28, 32; və 2 ədəd tam sətir: Sətir 24, 29). Şərt B-də **13 boş çıxış**.
- **Səbəb:** Tesseract `--psm 7` rejimində tək sətir aşkarlayıcısı ətraf səhifə konteksti olmadan ensiz/yüksək aspektli kəsimlərdə baza xəttini (`baseline`) itirir və inteqrasiya səviyyəsində boş sətir (`""`) qaytarır.
- **PSM 13 Fallback Testi:** Həmin boş çıxan kəsimlər `--psm 13` (raw line) ilə təkrar yoxlandıqda Tesseract mətni demək olar ki, qüsursuz oxuyur (məsələn, Sətir 18: `bu təkərin arxasında oturmusan. Özünü`, Sətir 24: `gizlənib. Biz getdikdən sonra bu əllər olmayacaq, amma bu küplər bizim`). Bu fakt problemin optik tanıma (OCR) dəqiqliyindən deyil, Tesseract daxili xətt seqmentatorunun kəsilmiş zolaqlardakı inteqrasiya uğursuzluğundan qaynaqlandığını qəti sübut edir.

## 3. Əsas Nəticələr Cədvəli: Şərt A, Şərt B və Təmiz Alt-Çoxluq (Clean Subset)

*(Qeyd: Əsas cədvəldə Tesseract yalnız `--psm 7` üzrə qiymətləndirilir; Şərt B 'sensitivity, not selected' kimi etiketlənir)*

### 3.1 Bütün 33 Sətir Üzrə Müqayisə (Şərt A və Şərt B)

| Model və Konfiqurasiya | CER (%) | WER (%) | Dəqiq Sətir | CER Xəta | WER Xəta | Boşluq Xətası | Boş Çıxış Sayı |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **V2 Rule Chunk (Şərt A - Trim+4px)** | 2.10% | 12.80% | **9/33** | 49 | 43 | 3 | 0 |
| V2 Full Line (Şərt A - Trim+4px) | 3.87% | 21.13% | 5/33 | 90 | 71 | 2 | 0 |
| Tesseract 5 Full Line (Şərt A, psm 7) | 8.42% | 15.18% | 14/33 | 196 | 51 | 24 | 2 |
| V1 Rule Chunk (Şərt A - Trim+4px) | 8.76% | 40.48% | 0/33 | 204 | 136 | 12 | 0 |
| Tesseract 5 Rule Chunk (Şərt A, psm 7) | 13.02% | 19.35% | 11/33 | 303 | 65 | 45 | 9 |
| V1 Full Line (Şərt A - Trim+4px) | 19.80% | 69.35% | 0/33 | 461 | 233 | 15 | 0 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| V2 Rule Chunk (Şərt B - sensitivity, not selected) | 23.32% | 66.07% | 0/33 | 543 | 222 | 21 | 0 |
| Tesseract 5 Rule Chunk (Şərt B, psm 7 - sensitivity) | 18.86% | 27.38% | 8/33 | 439 | 92 | 60 | 11 |
| V2 Full Line (Şərt B - sensitivity, not selected) | 22.72% | 68.45% | 0/33 | 529 | 230 | 10 | 0 |
| Tesseract 5 Full Line (Şərt B, psm 7 - sensitivity) | 10.91% | 22.92% | 8/33 | 254 | 77 | 31 | 2 |

### 3.2 Təmiz Alt-Çoxluq (Clean Subset: 23 Sətir, Heç Bir Modelin Boş Çıxışı Olmayan Sətirlər)

Tesseract-ın inteqrasiya uğursuzluğuna uğramadığı 23 sətir üzrə birbaşa optik tanıma dəqiqliyi:

| Model və Konfiqurasiya | CER (%) | WER (%) | Dəqiq Sətir | CER Xəta | WER Xəta | Boşluq Xətası |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tesseract 5 Rule Chunk (psm 7)** | 1.26% | 7.05% | 11/23 | 20 | 16 | 2 |
| **V2 Line Rule Chunk** | 1.32% | 9.25% | 9/23 | 21 | 21 | 2 |
| Tesseract 5 Full Line (psm 7) | 1.07% | 5.29% | 13/23 | 17 | 12 | 1 |
| V2 Line Full Line | 3.83% | 22.03% | 5/23 | 61 | 50 | 2 |
| V1 Line Rule Chunk | 6.73% | 32.16% | 0/23 | 107 | 73 | 1 |
| V1 Line Full Line | 20.62% | 70.93% | 0/23 | 328 | 161 | 7 |

## 4. Bootstrap 95% Etibarlılıq İntervalları (V2 − Tesseract psm 7)

- **Şərt A (Bütün 33 Sətir):** $\Delta$CER: -10.91% (95% CI: [-17.79%, -4.56%]), $\Delta$WER: -6.55% (95% CI: [-14.96%, 1.55%])
- **Şərt B (Padded 20px, Həssaslıq):** $\Delta$CER: +4.47% (95% CI: [-1.89%, 10.55%]), $\Delta$WER: +38.69% (95% CI: [30.47%, 46.95%])
- **Təmiz Alt-Çoxluq (23 Sətir):** $\Delta$CER: **+0.06%** (95% CI: **[-0.77%, 0.80%]**), $\Delta$WER: **+2.20%** (95% CI: **[-1.99%, 6.09%]**)
> **Həlledici Müşahidə:** Təmiz alt-çoxluqda $\Delta$CER 95% intervalı `[-0.77%, +0.80%]` aralığında sıfırı tam kəsir. Yəni Tesseract seqmentasiya səhvi etmədikdə, V2 Line modeli (1.32% CER) ilə Tesseract 5 (1.26% CER) arasında fərq statistik cəhətdən sıfırdan fərqlənmir!

## 5. Düzəldilmiş Qarşılıqlı Müstəsna Xəta Taksonomiyası (Levenshtein Cəmi Dəqiq Bərabərdir)

Əvvəlki skriptdə xəritələnmə xətası səbəbilə silinmiş simvollar durğu işarələrinə düşmüşdü. Bu dəfə hər bir silinmə öz simvol növünə görə (hərf, boşluq, durğu) təsnif edilmişdir:

| Xəta Kateqoriyası | Şərt A: V2 | Şərt A: Tess | Təmiz: V2 | Təmiz: Tess | İzah |
|---|:---:|:---:|:---:|:---:|---|
| **1. Başlanğıc Kənar Əlavəsi (Leading)** | **0** | **3** | **0** | **3** | V2-də 0 ədəd (sol kənar artefaktı yoxdur) |
| **2. Sonluq Kənar Əlavəsi (Trailing)** | **0** | **0** | **0** | **0** | Sətir sonu artıq simvol |
| **3. Daxili Boşluq Xətası** | **3** | **44** | **2** | **1** | Boşluq əməliyyatları |
| **4. Daxili 0 / O Qarışıqlığı** | **0** | **0** | **0** | **0** | 0 və O qarışıqlığı |
| **5. Daxili Durğu Xətası** | **12** | **20** | **6** | **7** | Nöqtə/iki nöqtə (`.` vs `:`) fərqləri |
| **6. Daxili Hərf Xətası** | **34** | **236** | **13** | **9** | Hərf səhvləri və silinmələri |
| **CƏM (Mutually Exclusive Sum)** | **49** | **303** | **21** | **20** | 100% Levenshtein cəminə bərabərdir |

## 6. V2 Line Modelinin Nisbət Qrupları Üzrə CER Təhlili ($W\cdot 32 / (h\cdot 512)$)

Sətirlər en/hündürlük nisbətinə görə 4 qrupa bölünərək tam sətir və parçalanmış rejimdə yoxlanmışdır:

| Nisbət Qrupu | Sətir Sayı | Sətir İndeksləri | V2 Tam Sətir CER | V2 Parçalanmış CER | CER Azalması ($\Delta$ pp) |
|---|:---:|---|:---:|:---:|:---:|
| **<1.0** | 4 | L07, L15, L27, L31 | 0.54% | 1.61% | **-1.08 pp** |
| **1.0-1.1** | 10 | L18, L22, L23, L24, L25, L28, L29, L30, L32, L33 | 4.74% | 3.93% | **+0.81 pp** |
| **1.1-1.3** | 13 | L01, L02, L03, L04, L05, L10, L12, L14, L17, L19, L20, L21, L26 | 3.36% | 1.16% | **+2.21 pp** |
| **>1.3** | 6 | L06, L08, L09, L11, L13, L16 | 4.87% | 1.33% | **+3.54 pp** |

> **Nəticə:** Nisbət böyüdükcə ($>1.3$) V2 Line modelinin tam sətirdəki CER-i 4.87%-ə qalxır və parçalama sayəsində xəta **3.54 faiz bəndi azalaraq 1.33%-ə** enir. Qısa sətirlərdə isə ($<1.0$) parçalamaya ehtiyac qalmır.

## 7. Əlaqə Cədvəli (Contact Sheet: Sətir 18, 19, 21, 22 İkinci Parçaları)

Tesseract-ın `--psm 7` rejimində boş qayıtmasının vizual səbəbini göstərən əlaqə vərəqi [contact_sheet_lines18_19_21_22.png](file:///d:/laragon/www/az_ocr_bm606/results/real_fix9/contact_sheets/contact_sheet_lines18_19_21_22.png) faylında saxlanılmışdır.

- Sətir 18 (2-ci parça): En 1149 px, Hündürlük 128 px, Aspekt nisbəti 8.98 -> Tesseract psm 7: `""` (boş), lakin psm 13: `bu təkərin arxasında oturmusan. Özünü`
- Sətir 19 (2-ci parça): En 725 px, Hündürlük 120 px, Aspekt nisbəti 6.04 -> Tesseract psm 7: `""` (boş), lakin psm 13: `Səməndər başını qaldırıb`
- Sətir 21 (2-ci parça): En 1239 px, Hündürlük 119 px, Aspekt nisbəti 10.41 -> Tesseract psm 7: `""` (boş), lakin psm 13: `masaya qoyub dedi: "Ay bala, insan o vaxt`
- Sətir 22 (2-ci parça): En 743 px, Hündürlük 125 px, Aspekt nisbəti 5.94 -> Tesseract psm 7: `""` (boş), lakin psm 13: `torpaq parçası deyil. Hər`
