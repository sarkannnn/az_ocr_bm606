# Azərbaycan OCR (CRNN), BM 606: yoxlanmış ətraflı hesabat

Tarix: 2026-10-23. Bütün rəqəmlər fayllardan yenidən oxunub. Hər sətirdə status var:
**TƏSDİQ** (fayldan yoxlandı), **FƏRQLİ** (real dəyər göstərilib), **YOXLANMADI** (səbəb yazılıb).
Hipotezlər ayrıca "HİPOTEZ" işarəsi ilə verilib.

---

## 1. Data (yoxlanıb: şəkil sayı və labels.tsv sətir sayı eyni)

| Data | Train | Val | Test |
|---|---|---|---|
| synth_word | 20 000 | 2 000 | 2 000 |
| synth_line | 10 000 | 1 000 | 1 000 |
| synth_word_v2 | 35 000 (20 000 köhnə + 15 000 əlavə) | 2 000 | 2 000 |
| synth_line_v2 | 16 000 (10 000 köhnə + 6 000 əlavə) | 1 000 | 1 000 |

- v2 val/test: `labels.tsv` SHA256 v1 ilə eynidir, şəkil nümunələri eynidir. TƏSDİQ.
- Val/test etiketlərinin train ilə kəsişməsi: söz v1 = 0, söz v2 = 0, sətir v1 = 0. TƏSDİQ.
- Əlavə 15 000 sözün 3 000-i (20,00%) rəqəm/durğulu (`results/v2_prep.md`). TƏSDİQ.
- sim_preview: 171 söz / 46 sətir. sim_ladder: 346 söz / 92 sətir. TƏSDİQ.
- Fontlar: `fonts/` içində 10 (arial, calibri, candara, consola, corbel, cour, segoeui, tahoma, times, verdana).
  `collect_fonts('fonts', 42)`: train 8 (candara, times, cour, segoeui, verdana, corbel, arial, calibri),
  test 2 (tahoma, consola). `fonts_new/`: CascadiaCode, bahnschrift, consolab, ebrima, l_10646, palab. TƏSDİQ.
- Sətir korpusunda xüsusi hərfli nümunələr: train 89,79% (9 338/10 400), val 96,24% (92 059/95 659),
  test 96,04% (92 104/95 906). Söz korpusunda: 49,55 / 48,36 / 50,44%. TƏSDİQ.

## 2. Model və təlim

- CRNN, 8 225 815 parametr (8,23M). Giriş: söz 32x128, sətir 32x512. TƏSDİQ.
- Adam lr=1e-3, weight_decay=1e-5, dropout 0,2, grad clip 5,0, patience 5, CPU. TƏSDİQ (`train.py:99-130`).
- Söz modeli: 17 epox, erkən dayandı, ən yaxşı epox 12 (val loss 0,0051, val CER 0,14%). TƏSDİQ.
- Sətir modeli: 15 epox (`--epochs 15`), ən yaxşı epox 12 (val loss 0,0131, val CER 0,31%). TƏSDİQ.
  Epox 14-15-də val loss 0,276 və 0,423-ə sıçrayıb (CER 3,5%). Təlim qeyri-sabitdir; sənəd bunu yazmır.
- Müddət: loglarda söz 301,3 dəq, sətir 184,6 dəq. **FƏRQLİ ölçü**: `best.pt` içində `train_seconds` söz 212,5 dəq,
  sətir 148,1 dəq. `best.pt` ən yaxşı epoxda yazılır, ona görə kiçik ola bilər (HİPOTEZ). Sənəddə "log
  müddəti" yazılmalıdır.
- İlk (boşluqsuz) sətir təlimi: 510,4 dəq, CER 11,19%, WER 99,75% (`logs/line_train_nospace.log`,
  `results/line_synth_nospace`). TƏSDİQ.

## 3. Nəticələr (yoxlanıb: `results/*/metrics.json`)

| Test | n | Bizim CER/WER | Tesseract CER/WER |
|---|---|---|---|
| Söz, qarışıq | 2000 | 0,50 / 3,95 | 0,70 / 4,90 |
| Söz, Tahoma | 1000 | 0,08 / 0,70 | 0,50 / 3,20 |
| Söz, Consolas | 1000 | 1,02 / 8,10 | 0,86 / 6,10 |
| Sətir, qarışıq | 1000 | 1,64 / 11,64 | 0,44 / 2,94 |
| Sətir, Tahoma | 500 | 0,24 / 1,74 | 0,22 / 1,59 |
| Sətir, Consolas | 500 | 3,48 / 23,76 | 0,47 / 3,12 |

Hamısı TƏSDİQ. Xüsusi hərf xətaları (`confusion_pairs.csv`), hamısı TƏSDİQ:
- Söz, qarışıq: ç→c 22,43% (24/107), ş→s 7,69% (17/221), ı→i 0,17%.
- Söz, Consolas: ç→c 39,62% (21/53), ş→s 9,17% (9/109). Söz, Tahoma: hamısı 0%.
- Sətir: ş→s 2,96% (9/338), ç→c 2,30% (3/174), ö→o 0,56%.
- Sətir ən çox əvəzləmə: r→n 131, m→n 125.

## 4. Simulyasiya pilləsi (sim_ladder, n = 346 söz / 92 sətir)

| Pillə | Söz CER | Sətir CER | Tess söz/sətir |
|---|---|---|---|
| L0 | 7,71 | 1,87 | 0,41 / 0,22 |
| L1 | 8,85 | 4,10 | 0,45 / 0,22 |
| L2 | 25,00 | 13,84 | 0,41 / 0,29 |
| L3 | 31,36 | 19,14 | 0,41 / 0,22 |
| L4 | 35,77 | 12,96 | 0,37 / 0,33 |
| L4n | 24,43 | 7,43 | 0,37 / 0,33 |

TƏSDİQ. Ən böyük sıçrayış L1→L2 (söz +16,15, sətir +9,74). `results/sim_ladder/summary.md` isə
"ən böyük nöqtə L0→L1" yazır; bu ölçülmüş rəqəmlərə uyğun deyil.
L0 söz xətaları: 96/346, bunun 26-sı (27,08%) yalnız kənar durğu fərqidir. Durğu çıxarılanda CER 7,71→6,54%.
Tesseract: 9/346 xəta, 2-si durğu, CER 0,41→0,34%. TƏSDİQ (`diag2.md`).
Font üzrə L0 (söz/sətir CER): CascadiaMono 11,94/6,71; cambria 5,24/0,31; micross 6,77/0,00; pala 6,92/0,59.

## 5. Augmentasiya

`train.py:21-40` (v1): fırlanma ±5° p=0,5; Gauss blur k∈{3,5} p=0,3; duz-istiot 0,5% p=0,3; perspektiv
(±2% en, ±8% hündürlük) p=0,3. Gauss küyü, JPEG, kiçiltmə, parlaqlıq/kontrast, gradient, kontrast uzatma: yoxdur.

`train_v2.py:26-148`: fırlanma 0,5; perspektiv 0,3; gradient+kölgə 0,5; parlaqlıq/kontrast (alpha 0,6–1,1,
beta -40..+20) 0,6; Gauss blur sigma 0,5–2,0 p=0,6; hərəkət blur (3–7) 0,15; Gauss küyü sigma 3–12 p=0,6;
duz-istiot 0,2; kiçiltmə 0,5–0,9 p=0,4; JPEG 40–90 p=0,4; kontrast uzatma 0,5. TƏSDİQ.
`--resume`, `last.pt`, `--save-aug-preview` var.

## 6. Tapılmış problemlər (ciddilik sırası ilə)

1. **Font ailəsi təmasları (v2 üçün YENİ TAPINTI).** `fonts_new/` içində 3 font test fontlarının ailəsidir:
   - `consolab.ttf` = Consolas Bold; test fontu `consola.ttf` = Consolas.
   - `palab.ttf` = Palatino Linotype Bold; sim test fontu `pala.ttf` = Palatino Linotype.
   - `CascadiaCode.ttf`; sim test fontu `CascadiaMono.ttf` (eyni ailə, yalnız ligatur fərqi, HİPOTEZ: forma çox yaxındır).
   
   Fakt: ad/ailə eynidir. HİPOTEZ: v2-nin Consolas, Pala və Cascadia nəticələri "görünməyən font" yox,
   "yaxın font" nəticəsi olacaq və v1-ə qarşı müqayisə şişirdiləcək. `v2_prep.md` "4 sim-test fontu istisna
   edildi" deyir, amma yalnız fayl adına görə.
2. **Söz təlim korpusunda rəqəm və durğu 0%** (`prepare_corpus.py:14`; `v2_prep.md`: söz train 0 rəqəm, 0 durğu).
   Fakt: L0-da durğu fərqi CER-in yalnız 1,17 bəndini izah edir; qalan ~6,5 bəndin rəqəm və font payı ayrıca ölçülməyib.
3. **Val yalnız təlim fontlarındadır** (`generate_data.py:88`). Erkən dayandırma font ümumiləşməsini
   ölçmür. Val CER 0,14% və test CER 0,50% fərqinin izahı ola bilər (HİPOTEZ). v2-də val v1 ilə eyni olduğu
   üçün bu məhdudiyyət qalır.
4. **Padding və normallaşdırma** (`common.py:84-86`): sağdan ağ (255) padding, yalnız /255. Fakt: L4n
   (kontrast uzatma) söz CER-i 35,77→24,43, sətir CER-i 12,96→7,43 azaldır, yenidən təlimsiz. Mexanizm ölçülməyib.
5. **Sürət ölçməsi müqayisə olunmur** (`evaluate.py:32-66`): bizdə yalnız `model(x)` (batch 128, I/O,
   preprocess və decode xaric); Tesseract-da hər şəkil üçün ayrı proses, disk oxuma və RGB çevirmə daxil.
   Eyni model (word_synth) `logs/run_all.log`-da 16,41 ms, indi 5,21 ms (3 dəfə fərq, TƏSDİQ). Sətir: 19,10–22,78 ms.
   PC-nin yüklənməsindən asılıdır (HİPOTEZ).
6. **`Vocab.encode` naməlum simvolları səssiz atır** (`common.py:40-41`). Boşluq xətasının kökü bu idi. Düzəliş:
   `common.py:13` (`+ " "`). Söz checkpoint-i öz `chars`-ını saxlayır, ona təsir etməyib.
7. **CTC input_lengths həmişə tam T** (`train.py:77`). Qısa sözlərdə padding üzərində blank buraxmağa məcbur edir.
   Təsiri ölçülməyib.
8. **Etiketlər NFC normallaşdırılmır** (`common.py:130-141`) və Tesseract çıxışı normallaşdırılır (`evaluate.py:64`).
   Cari dataset NFC mətndir, ona görə təsir ölçülməyib. Aşağı risk.
9. **Sətir korpusu xüsusi hərflərə ağırlıqlıdır** (val/test ~96%). Sətir nəticəsi "ümumi mətn" yox, bu paylanmaya aiddir.
10. **Git vəziyyəti**: `origin` = github.com/sarkannnn/az_ocr_bm606, yalnız 1 commit ("Initial commit"). `common.py`
    düzəlişi, yeni `runs/line/best.pt`, nəticələr və yeni skriptlər commit olunmayıb (`git status`).

## 7. Sənəddə düzəliş tələb edən yerlər (Azerbaycan_OCR_Hesabat.docx)

| Yer | Problem | Düzəliş |
|---|---|---|
| Giriş qeydi | "bütün rəqəmlər audit ilə yoxlanıb" | Bu sənəddəki yoxlamadan sonra doğrudur; əvvəlki audit yalnız 10 iddianı əhatə edirdi |
| Bölmə 5, nəticə | "təlimdə oxşarını gördüyü fontda (Tahoma)" | Tahoma test fontudur (təlimdə yoxdur). "Oxşar font görməsi" hipotezdir |
| Bölmə 4 | 301 / 185 dəq | Log müddətidir; `best.pt` içində 212,5 / 148,1 dəq. Mənbəni yazın |
| Bölmə 9 | "nümunələrin ~17%-də (4/24) alt kənar kəsilir" | `v2_prep.md` 24 nümunədə kəsilmə olmadığını yazır; 4/24 rəqəmi fayllarda tapılmadı: YOXLANMADI |
| Bölmə 9 | 6 yeni font "test fontlarından kənardır" | 3 font test fontlarının ailəsidir (bölmə 6, bənd 1) |
| Bölmə 10 | sürət "5,2 və 16,4 ms" | Doğrudur: 16,41 köhnə log (`run_all.log`), 5,21 indiki `metrics.json` |
| Bölmə 12 | ehtiyat nüsxə | 337,7 MB TƏSDİQ (`D:\laragon\www\az_ocr_backup`); layihə qovluğu 3 536,8 MB |

## 8. Hələ də yoxlanmayanlar

- TRDG yaması (.venv içində, kəsilmə düzəlişi): .venv dəyişdirilməyib, yamanın içi oxunmayıb.
- "Alt kənarın 4/24 kəsilməsi": mənbə tapılmadı.
- Təlim vaxtı təxminləri (söz 6–9 saat, sətir 4–8 saat): təxmindir. Ölçülmüş əsas: söz ~17,7 dəq/epox
  (20 000 şəkil), sətir ~12,3 dəq/epox (10 000 şəkil). Kobud miqyaslama (HİPOTEZ): söz v2 ~31 dəq/epox,
  sətir v2 ~20 dəq/epox, augmentasiya CPU-da əlavə yük yaradır.

## 9. Planda olub, edilməyənlər

TrOCR, beam search (yalnız greedy, `common.py:43-49`), sözə bölmə xətası, real foto testi (`run_all.ps1`
`data\real_*` yoxdursa keçir), ön emal ablation-ı (yalnız `smoke`), xərc metrikası (yalnız parametr sayı
və `gpu_mem_mb`; CPU-da `null`), etibarlı sürət ölçməsi, v2 təlimi.
