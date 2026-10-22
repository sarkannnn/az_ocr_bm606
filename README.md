# Azərbaycan dili üçün OCR (BM 606)

Söz və sətir səviyyəli CRNN + CTC. Addımlar təqdimatdakı planla eyni sıradadır.

## Quraşdırma
```bash
pip install -r requirements.txt
# Tesseract baseline üçün: sudo apt install tesseract-ocr tesseract-ocr-aze
```
`fonts/` qovluğuna ə ı ö ğ ü ş ç dəstəkləyən fontlar qoyun (DejaVu, Noto Sans/Serif, Roboto, Open Sans...). Uyğun olmayanlar avtomatik atılır, fontların ~20%-i yalnız test üçün ayrılır.

## 1. Korpus (şəkillərdən əvvəl 80/10/10 bölgü)
```bash
python prepare_corpus.py --hf-wiki 20000 --mode word --out data/corpus_word --max-items 150000
python prepare_corpus.py --hf-wiki 20000 --mode line --out data/corpus_line --max-items 100000
```
(`--input fayl.txt` ilə öz mətniniz də olar.)

## 2. Sintetik şəkillər (TRDG)
```bash
python generate_data.py --mode word --corpus data/corpus_word --fonts fonts --out data/synth_word
python generate_data.py --mode line --corpus data/corpus_line --fonts fonts --out data/synth_line --n-train 60000
```

## 3. Real test seti
Kamera/skan şəkillərini kəsin və `data/real_word/test/` (və `real_line/test/`) altına qoyun:
```
images/0001.png
labels.tsv        # hər sətir:  images/0001.png<TAB>mətn
```

## 4. Ön emal ablation-ı (CER cədvəli)
```bash
python run_ablation.py --mode word --train-dir data/synth_word/train \
    --val-dir data/synth_word/val --test-dir data/synth_word/test --epochs 10
```
Nəticə: `results/ablation.csv`

## 5. Vize: söz CRNN (32×128)
```bash
python train.py --mode word --train-dir data/synth_word/train --val-dir data/synth_word/val --out runs/word
python evaluate.py --ckpt runs/word/best.pt --data data/synth_word/test --out results/word_synth
python evaluate.py --baseline tesseract --mode word --data data/synth_word/test --out results/tess_word
```
`runs/word/loss_curve.png` — train/val itki əyriləri (overfitting yoxlaması).

## 6. Final: sətir CRNN (32×512) və müqayisə
```bash
python train.py --mode line --train-dir data/synth_line/train --val-dir data/synth_line/val --out runs/line
python evaluate.py --ckpt runs/line/best.pt --data data/synth_line/test --out results/line_synth
python evaluate.py --ckpt runs/line/best.pt --data data/real_line/test  --out results/line_real   # domain gap
```

## Çıxışlar (hər `results/*/` qovluğunda)
- `metrics.json`: CER, WER, ms/şəkil, parametr sayı, GPU yaddaşı, təlim müddəti (xərc müqayisəsi)
- `confusion_pairs.csv`: ə/e, ı/i, ö/o, ğ/g, ü/u, ş/s, ç/c qarışıqlıqları
- `top_substitutions.csv`, `predictions.tsv`

## Qeydlər
- Sətir üçün default uzunluq 12–35 simvoldur (32×512 girişə sığsın). Daha uzun sətir üçün `--max-len` və `PRESETS["line"]` (common.py) birlikdə artırılmalıdır.
- "Sözə bölüb oxumaq" xətası (segmentasiya) ayrıca ölçülməlidir; bu paketdə yalnız hazır kəsilmiş söz/sətir şəkilləri qiymətləndirilir.
- TrOCR və beam search + n-qram ( əlavə deneylər) bu paketə daxil deyil.
- Kod bu mühitdə torch olmadan yalnız ön emal, metrik və korpus hissələri üçün sınaqdan keçirilib; təlim/qiymətləndirmə skriptlərini GPU-lu mühitdə (Colab) bir neçə yüz şəkillə ilk dəfə sınayın.
