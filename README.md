# Azərbaycan dili üçün OCR (BM 606)

Azərbaycan dili çap və telefon şəkli sənədləri üçün Söz və Sətir səviyyəli **CRNN + CTC** optik tanıma mühərriki, **SymSpell** əsaslı lüğət korreksiyası və veb/mobil tanıma platforması.

---

## 📱 İstənilən Kompüterdə 1 Kliklə Quraşdırma və İşə Salma (Veb və Mobil UI)

Layihənin bütün öyrədilmiş model çəkiləri (`runs/line_v2/best.pt`, `runs/line/best.pt`), 116,143 sözlük morfoloji lüğət (`data/az_lexicon.txt`) və Azərbaycan dili Tesseract dil faylı (`tessdata/aze.traineddata`) birbaşa repozitoriyaya daxildir.

### 🪟 Windows-da Quraşdırma və Başlatma:
1. Kompüterdə **Python 3.10+** quraşdırıldığından əmin olun ([python.org](https://www.python.org/downloads/) — quraşdırarkən *"Add Python to PATH"* seçimini işarələyin).
2. Layihəni endirin və **`run_ui.bat`** faylına iki dəfə klikləyin (və ya terminalda işə salın):
   ```bat
   git clone https://github.com/sarkannnn/az_ocr_bm606.git
   cd az_ocr_bm606
   run_ui.bat
   ```
   > `run_ui.bat` ilk dəfə işə düşdükdə avtomatik olaraq `.venv` virtual mühitini yaradır, `requirements.txt` paketlərini quraşdırır, `.HEIC` dəstəyini aktivləşdirir və həm lokal (`http://localhost:7860`), həm də mobil telefon üçün ictimai HTTPS linkini + QR kodu işə salır.

### 🍎 macOS / 🐧 Linux-da Quraşdırma və Başlatma:
```bash
git clone https://github.com/sarkannnn/az_ocr_bm606.git
cd az_ocr_bm606
chmod +x run_ui.sh
./run_ui.sh
```

### ⚙️ Əl ilə (Manual) Quraşdırma:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python app.py --share
```
* **Kompüterdə:** Brauzerdə `http://localhost:7860` ünvanını açın.
* **Telefonda:** Ekranda yaranan **QR kodu** telefon kamerası ilə oxudun və ya `https://*.gradio.live` linkinə toxunun.
* *(Opsional)* **Tesseract 5 müqayisəsi üçün:** Sistemdə Tesseract-OCR quraşdırılıbsa (`winget install UB-Mannheim.TesseractOCR` və ya `sudo apt install tesseract-ocr`), proqram onu və daxili `tessdata/aze.traineddata` faylını avtomatik tanıyır.

---

## 🚀 Sürətli Başlanğıc (CLI İnferens / Tanıma)

Quraşdırılmış ən yaxşı model (`runs/line_v2/best.pt`) və Azərbaycan dili lüğəti ilə istənilən sətir şəkli üzərində birbaşa mətn tanıma aparın:

```bash
# Tək sətir şəklinin oxunması (avtomatik lüğət korreksiyası ilə)
python predict.py --image path/to/line.png

# Xam (raw) model çıxışı (lüğət korreksiyası olmadan)
python predict.py --image path/to/line.png --no-postprocess

# Müəyyən qovluqdakı şəkillərin toplu oxunması
python predict.py --dir path/to/crops/ --limit 20 --out results/my_preds.tsv
```

---

## 📦 Quraşdırma

```bash
# Asılılıqların quraşdırılması
pip install -r requirements.txt

# Tesseract baseline sınaqları üçün:
# Windows: C:\Program Files\Tesseract-OCR (aze.traineddata ilə)
# Linux: sudo apt install tesseract-ocr tesseract-ocr-aze
```

---

## 📂 Layihə Strukturu

```
az_ocr_bm606/
├── common.py                # Əsas utilitlər (Vocab, CER/WER, align, preprocess)
├── model.py                 # CRNN şəbəkə arxitekturası (CNN + BiLSTM + CTC)
├── training_utils.py        # Dataset, collate və CTC təlim dövrəsi
├── train.py                 # V1 baza təlim skripti (sintetik)
├── train_v2.py              # V2 təkmil təlim skripti (telefon/işıq/kölgə augmentasiyaları)
├── postprocess.py           # SymSpell əsaslı Azərbaycan dili lüğət korreksiyası
├── predict.py               # Asan CLI inferens və tanıma aləti
├── evaluate.py              # CER/WER, xərc analizi və Tesseract baseline
├── make_sim_ladder.py       # Təhrif pilləkəni (L0-L4) generasiyası
├── run_sim_ladder.py        # Təhrif pilləkəni qiymətləndirməsi
├── data/
│   ├── az_lexicon.txt       # 116,000+ sözlük tezlik lüğəti
│   ├── synth_word_v2/       # 39,000 sintetik söz verilənləri (train/val/test)
│   ├── synth_line_v2/       # 18,000 sintetik sətir verilənləri (train/val/test)
│   └── real_docs/           # Real telefon şəkli sənədləri (Times, Calibri)
├── fonts/                   # Əsas təlim fontları (10 font)
├── fonts_new/               # V2 əlavə fontları (6 font)
├── runs/
│   ├── word_v2_*/best.pt   # Ən yaxşı V2 söz modeli
│   └── line_v2/best.pt      # Ən yaxşı V2 sətir modeli
├── archive/                 # Köhnə və arxivlənmiş audit/skriptlər
└── results/                 # Eksperiment nəticələri və benchmarklar
```

---

## 🛠️ Təlim və Qiymətləndirmə

### 1. Korpusun Hazırlanması
```bash
python prepare_corpus.py --hf-wiki 20000 --mode word --out data/corpus_word --max-items 150000
python prepare_corpus.py --hf-wiki 20000 --mode line --out data/corpus_line --max-items 100000
```

### 2. Sintetik Verilənlərin Törədilməsi (TRDG)
```bash
python generate_data.py --mode word --corpus data/corpus_word --fonts fonts --out data/synth_word
python generate_data.py --mode line --corpus data/corpus_line --fonts fonts --out data/synth_line
```

### 3. CRNN Təlimi (V2 Kamera və Optik Təhriflərlə)
```bash
# Sətir modeli (32×512)
python train_v2.py --mode line --train-dir data/synth_line_v2/train --val-dir data/synth_line_v2/val --out runs/line_v2

# Söz modeli (32×128)
python train_v2.py --mode word --train-dir data/synth_word_v2/train --val-dir data/synth_word_v2/val --out runs/word_v2
```

### 4. Qiymətləndirmə və Baseline Müqayisəsi
```bash
python evaluate.py --ckpt runs/line_v2/best.pt --data data/synth_line_v2/test --out results/line_v2_synth
python evaluate.py --baseline tesseract --mode line --data data/synth_line_v2/test --out results/tess_line_synth
```