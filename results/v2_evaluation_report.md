# Azərbaycan OCR (CRNN), BM 606: V2 Təlim və Qiymətləndirmə Yekun Hesabatı

Tarix: 2026-10-28  
Mühit: CPU, PyTorch 2.x, Windows  
Repozitoriya: `d:\laragon\www\az_ocr_bm606`

---

## 1. Təlim Müddətləri və İrəliləyiş

Bütün addımlar `logs/v2_progress.log` jurnalına əsasən ardıcıl və fasiləsiz icra olunmuşdur. Heç bir kəsilmə və ya gözləmə fasiləsi baş verməmişdir.

| Model | Başlama Vaxtı | Bitmə Vaxtı | Ümumi Müddət | Epox Sayı | Ən Yaxşı Epox | Ən Yaxşı Val Loss | Ən Yaxşı Val CER | Çıxış Faylı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Söz (Word CRNN v2)** | 2026-10-27 18:03:24 | 2026-10-27 21:35:45 | **212.3 dəq** (~3s 32d) | 18 (erkən dayandı) | Epox 13 | 0.0063 | 0.24% | `runs/word_v2_20261027_180300/best.pt` |
| **Sətir (Line CRNN v2)** | 2026-10-27 21:36:03 | 2026-10-28 02:16:29 | **280.4 dəq** (~4s 40d) | 15 (tamamlandı) | Epox 10 | 0.0125 | 0.28% | `runs/line_v2/best.pt` |

*Fasilə / Kəsilmə müddəti: 0 dəqiqə (proses fasiləsiz tamamlandı).*

---

## 2. V1 və V2 Qiymətləndirmə Nəticələrinin Müqayisəsi

Bütün rəqəmlər `results/*/metrics.json` fayllarından birbaşa oxunmuşdur (uydurma rəqəm yoxdur):

| Test Dəsti | Nümunə Sayı (n) | V1 CER (%) | V1 WER (%) | V2 CER (%) | V2 WER (%) | Tesseract CER (%) | Tesseract WER (%) | V1 $\rightarrow$ V2 İnkişaf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Söz, qarışıq test** (`word_synth`) | 2000 | 0.50% | 3.95% | **0.11%** | **0.90%** | 0.70% | 4.90% | **CER 4.5 dəfə, WER 4.4 dəfə yaxşılaşdı** |
| **Sətir, qarışıq test** (`line_synth`) | 1000 | 1.64% | 11.64% | **0.30%** | **2.13%** | 0.44% | 2.94% | **CER 5.5 dəfə, WER 5.5 dəfə yaxşılaşdı** |
| **Söz, Tahoma** (`diagw_tahoma`) | 1000 | 0.08% | 0.70% | **0.09%** | **0.70%** | 0.50% | 3.20% | Sabit yüksək dəqiqlik (Tesseract-dan üstün) |
| **Söz, Consolas** (`diagw_consola`) | 1000 | 1.02% | 8.10% | **0.07%** | **0.60%** | 0.86% | 6.10% | **CER 1.02% $\rightarrow$ 0.07% (14.5 dəfə yaxşılaşdı)** |
| **Sətir, Tahoma** (`diag_tahoma`) | 500 | 0.24% | 1.74% | **0.26%** | **1.84%** | 0.22% | 1.59% | Sabit yüksək dəqiqlik |
| **Sətir, Consolas** (`diag_consola`) | 500 | 3.48% | 23.76% | **0.27%** | **1.93%** | 0.47% | 3.12% | **CER 3.48% $\rightarrow$ 0.27% (13 dəfə yaxşılaşdı)** |

---

## 3. Tesseract Baseline ilə Müqayisə Qeydləri

Layihənin mövcud `results/tess_*/metrics.json` nəticələri ilə müqayisə:
1. **Söz səviyyəsində:** 
   - Qarışıq testdə V2 CRNN modeli **0.11% CER** ilə Tesseract-ı (**0.70% CER**) əhəmiyyətli dərəcədə üstələyir (WER: 0.90% vs 4.90%).
   - Monospace şrifti olan Consolas-da V1 modeli Tesseract-dan geri qalırdı (1.02% vs 0.86%), lakin V2 modeli **0.07% CER** ilə Tesseract-ı böyük fərqlə üstələdi.
2. **Sətir səviyyəsində:**
   - Qarışıq sətir testində V1 modeli Tesseract-dan zəif idi (CRNN V1: 1.64% vs Tesseract: 0.44%).
   - V2 modeli ilə sətir CER göstəricisi **0.30%-ə**, WER isə **2.13%-ə** düşərək Tesseract-ın hər iki göstəricisini (**0.44% CER, 2.94% WER**) qabaqlamışdır.
   - Consolas sətir testində V1-dəki 3.48% CER göstəricisi V2-də **0.27%-ə** düşərək Tesseract-dan (0.47%) daha dəqiq nəticə vermişdir.

---

## 4. Qayda 5 üzrə Tətbiq Olunmuş Yollar

Addım 0-da ilkin yoxlama zamanı əvvəlki cəhddən mövcud olan yolların üzərinə yazmamaq üçün Qayda 5 tətbiq olunmuşdur:
- `runs/word_v2` mövcud olduğu üçün $\rightarrow$ **`runs/word_v2_20261027_180300`** adı ilə yaradıldı.
- `logs/word_v2_train.log` mövcud olduğu üçün $\rightarrow$ **`logs/word_v2_train_20261027_180300.log`** adı ilə yaradıldı.
- Sətir təlimi üçün `runs/line_v2` və `logs/line_v2_train.log` mövcud olmadığı üçün birbaşa standart adlarla yaradıldı.
- Bütün nəticə qovluqları (`results/*_v2`) ilk dəfə yaradıldığı üçün standart adlarla qorundu.

---

## 5. Qayda 9 üzrə Bərpa və İcra Tarixçəsi

- Bütün proses sıfırdan başlandı (`logs/v2_progress.log` STEP0_PRECHECK qeydi ilə).
- Heç bir addımda şəbəkə, sessiya və ya agent kəsilməsi baş vermədi.
- Təlim addımları (Addım 1 və Addım 2) tək cəhddə uğurla tamamlandı.

---

## 6. Xətalar və Uyğunlaşdırmalar (Qayda 7)

Addım 3 icra olunarkən:
- İlkin `STEP3_3` əmri birbaşa `data/diagw_tahoma` qovluğunu hədəflədikdə xəta baş verdi:  
  *Xəta mətni:* `FileNotFoundError: [Errno 2] No such file or directory: 'data\diagw_tahoma\labels.tsv'`  
  *Səbəb:* Diaqnostik verilənlər dəstində etiketlər alt qovluqda (`data/diagw_tahoma/test/labels.tsv`) yerləşir (V1-də olduğu kimi).
- Qayda 7-nin Addım 3 istisnasına uyğun olaraq, test dəstinin faktiki yolu olan `data/diagw_*/test` və `data/diag_*/test` istifadə edilərək qiymətləndirmələr uğurla başa çatdırıldı (`exit=0`) və jurnala qeyd edildi.
