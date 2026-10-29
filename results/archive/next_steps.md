# Qalan işlər: addım-addım plan

Qaydalar: hər addımda "Nə", "Necə", "Hazır olduğunu necə bilirsən". Əmrlər PowerShell, layihə qovluğundan
(`d:\laragon\www\az_ocr_bm606`) icra olunur. Təlim 3 gün sonra başlayır; ondan əvvəl A və B bölmələri edilməlidir.

## A. Təlimdən əvvəl (indi–3 gün)

**A1. Ehtiyat nüsxəni tamamla.**
1. `D:\laragon\www\az_ocr_backup` (337,7 MB) mövcuddur. Birinci hissədir; layihənin özü 3 536,8 MB.
2. Əlavə et: `data\corpus_word`, `data\corpus_line`, `fonts`, `fonts_new`, `runs\word\best.pt`,
   `runs\line\best.pt`, `results`, `logs`, scratch skriptləri (`build_v2_datasets.py`, `prep_fonts.py`,
   `measure_1.py`, `run_diag2.py`; yerləri:
   `C:\Users\serxan.m\.gemini\antigravity-ide\brain\912dbc3a-1632-49b2-a686-a0d9534b585d\scratch\`).
3. `Compress-Archive` ilə zip yarat, Drive-a yüklə.
4. Hazırdır: Drive-da zip açılır, `best.pt` fayllarının ölçüsü 31,4 MB.

**A2. Git-ə commit/push et (kodlar və nəticələr).**
1. `git status` ilə bax. Böyük `data\` qovluqları və `.venv` commit olunmamalıdır (`.gitignore` yoxla).
2. `git add common.py train_v2.py make_sim_photos.py make_sim_ladder.py run_sim_ladder.py results logs`
3. `git commit -m "Fix space in CHARS, retrain line model, add v2 training, sim ladder, audit"` və `git push`.
4. Hazırdır: GitHub-da 2-ci commit görünür.

**A3. Font ailəsi problemini həll et (3 gündə qərar ver, təlimdən əvvəl).**
Problem: `consolab`, `palab`, `CascadiaCode` test fontlarının ailəsidir (`final_report.md`, bölmə 6/1).
Seçim 1 (tövsiyə): bu 3 fontu `fonts_new/`-dən çıxar, `results/v2_prep.md` siyahısındakı 59 fontdan başqa
ailələr seç (məs. Courier ailəsi artıq təlimdədir; `Moba_*` şübhəlidir, əvvəlcə şəkillərə bax), əlavə
15 000 söz və 6 000 sətir şəklinin həmin fontlarla olan hissəsini yenidən yarat. Bunun üçün `build_v2_datasets.py`
və `prep_fonts.py` scratch skriptlərindən istifadə et (yeni qovluğa yaz, köhnəni silmə).
Seçim 2: fontları saxla, amma v2 nəticələrini yalnız onlarla əlaqəsiz fontlarda (tahoma, cambria, micross) hesabla
və hesabatda məhdudiyyət kimi yaz.
Hazırdır: seçdiyin variant `results/v2_prep.md`-də yazılıb.

**A4. Sənədi düzəlt.** `final_report.md`, bölmə 7-dəki cədvəl üzrə düzəlişləri `.docx`-ə daxil et.
Sonra bölmə 12-yə A3 qərarını yaz.

**A5. Sürəti PC boşda yenidən ölç.**
1. Başqa proqramları bağla.
2. Eyni şəkillərdə 3 dəfə: `.\.venv\Scripts\python.exe evaluate.py --ckpt runs/word/best.pt --data data/synth_word/test --out results/timing_word_r1` (r2, r3 üçün də).
3. İkinci ölçmə kimi tək-tək (batch=1) vaxt əlavə et (yeni kiçik skript lazımdır, mövcud kodu dəyişmə).
4. Hazırdır: 3 ölçmə arasındakı fərq 10%-dən azdır. Hesabatda "bizim: forward only, batch 128; Tesseract: tam
   proses/şəkil" yaz, rəqəmləri bərabər müqayisə etmə.

## B. v2 təlimi (3 gün sonra)

**B1. Təlim günü ön yoxlama.**
1. Windows Update-i fasilə ver, Güc planı: "heç vaxt yatma".
2. `tasklist | findstr /i python`: boş olmalıdır.
3. `runs\word_v2` və `runs\line_v2` mövcud deyilsə davam et (köhnə `runs\word`, `runs\line` toxunulmaz qalır).

**B2. Sətir və söz təlimini ardıcıl başlat (CPU eyni anda ikisini çəkməz).**
```powershell
$env:PYTHONUTF8="1"
.\.venv\Scripts\python.exe -u train_v2.py --mode word --train-dir data/synth_word_v2/train --val-dir data/synth_word_v2/val --out runs/word_v2 --epochs 25 --workers 0 2>&1 | Tee-Object logs\word_v2_train.log
.\.venv\Scripts\python.exe -u train_v2.py --mode line --train-dir data/synth_line_v2/train --val-dir data/synth_line_v2/val --out runs/line_v2 --epochs 15 --workers 0 2>&1 | Tee-Object logs\line_v2_train.log
```
Fasilə olarsa `--resume runs/word_v2/last.pt` əlavə et (sətir üçün eyni).
Təxmin (ölçülməyib): söz ~8–13 saat, sətir ~5 saat. Gecə ərzində izləmə üçün `Get-Content logs\word_v2_train.log -Tail 5`.
Hazırdır: log sonunda "Təlim vaxtı" sətri, `runs\word_v2\best.pt` və `runs\line_v2\best.pt` mövcuddur.

## C. v2 qiymətləndirməsi

**C1. Eyni test dəstləri üzərində (v1 ilə birbaşa müqayisə).**
```powershell
.\.venv\Scripts\python.exe evaluate.py --ckpt runs/word_v2/best.pt --data data/synth_word/test --out results/word_v2_synth
.\.venv\Scripts\python.exe evaluate.py --ckpt runs/line_v2/best.pt --data data/synth_line/test --out results/line_v2_synth
```
Tahoma/Consolas üçün `data\diagw_*` və `data\diag_*` ilə təkrarla (nəticə: `results\diagw_*_v2`, `results\diag_*_v2`).

**C2. sim_ladder ilə müqayisə.** `run_sim_ladder.py`-ın yeni checkpoint qəbul edib-etmədiyini aç və bax
(argumentlər orada). Qəbul etmirsə, v2 üçün surəti yeni fayl kimi yarat (köhnəni dəyişmə). Nəticəni
`results/sim_ladder_v2/` qovluğuna yaz.

**C3. Cədvəl hazırla:** v1 və v2 (CER/WER) yan-yana: söz qarışıq, Tahoma, Consolas, sətir; L0..L4n.
Qayda: yalnız ölçülmüş fərqi yaz; A3-də seçdiyin variantı qeyd et. Səbəb iddiasını (augmentasiya, rəqəm/durğu)
yalnız ablation olmadan "hipotez" kimi yaz.

## D. Real foto testi (hədəf 300 söz, 100 sətir; minimum 20–30 foto)

1. Müxtəlif mətn mənbələri (kitab, qəzet, ekran çapı olmayan real kağız), müxtəlif işıq, 20–30 foto çək.
2. Hər fotodan sətirləri və sözləri əl ilə kəs (məsələn Paint və ya `cv2.selectROI` skripti; yeni fayl).
3. Etiketləri `data/real_word/test/labels.tsv` və `data/real_line/test/labels.tsv` formatında yaz
   (`images/000001.png<TAB>mətn`). `run_all.ps1` bu yolları avtomatik tanıyır.
4. Qiymətləndir: `evaluate.py` (bizim) və `--baseline tesseract` (aze).
5. Hazırdır: `results/word_real`, `results/line_real`, `results/tess_*_real` metrics.json var.

## E. İxtiyari (vaxt qalarsa), ciddiyyət sırası ilə

1. **Ön emal ablation-ı (kiçik):** gray/rgb, filter none/gaussian/median, thresh none/otsu/adaptive, interp
   bilinear/bicubic. `train.py` bu seçimləri `--color/--filter/--thresh/--interp` ilə verir. Tam təlim çox
   uzundur; kiçik alt-dəst (məs. 5 000 şəkil, 5 epox) təklif olunur (bunu bir azı bir test dəsti ilə et).
2. **Beam search:** `common.py`-ə toxunmadan yeni modulda CTC prefix beam search yaz, eyni `results` formatında ölç.
3. **Sözə bölmə xətası:** sətir modelinin boşluq çıxarma xətası kimi ölç (hipotez: WER-in böyük hissəsi budur;
   bunu `predictions.tsv`-də boşluq sayı ilə yoxla).
4. **Xərc metrikası:** parametr sayı, model faylı (31,4 MB), CPU ms/şəkil (A5), təlim vaxtı.
5. **TrOCR:** yalnız vaxt çox qalarsa; kiçik fine-tune və ya yalnız zero-shot müqayisə kimi göstər.

## F. Son hesabat

1. `final_report.md`-dəki problemlər və məhdudiyyətlər bölməsini hesabata köçür.
2. v2 və real foto nəticələrini əlavə et.
3. Son commit/push, ikinci ehtiyat nüsxə (zip + Drive).
