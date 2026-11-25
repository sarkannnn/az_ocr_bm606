# AzOCR-V2 — Padding Robustness Eksperimental Hesabatı

Sabit Test Bazası: **15 Müstəqil Smartfon Sənədi (`B1`–`B15`, cəmi 295 sətir)**
Model: **CRNN-V2 Xam CTC Çıxışı (`runs/line_v2/best.pt`)**

## 1. Əsas Eksperiment: Əlavə Edilmiş Ağ Haşiyə (Canvas Border Padding + Rəsmi infer_line_smart())

Kəsimin kənarlarına birbaşa $+P\text{ px}$ ağ fon əlavə edilir və rəsmi borudan keçirilir.

| Əlavə Haşiyə (Padding) | Xam (Raw) CER (%) ↓ | Xam (Raw) WER (%) ↓ | Tam Düzgün Sətir (Exact) ↑ | Orta Blok (Chunk) Sayı | Gecikmə (ms) |
| :---: | :---: | :---: | :---: | :---: :---: :---: |
| **+0 px** | **0.30%** | 1.64% | 253/295 (85.8%) | 2.64 | 86.7 ms |
| **+4 px** | **0.61%** | 3.10% | 221/295 (74.9%) | 2.09 | 76.1 ms |
| **+8 px** | **1.56%** | 8.56% | 161/295 (54.6%) | 1.97 | 86.1 ms |
| **+12 px** | **3.71%** | 21.09% | 41/295 (13.9%) | 1.92 | 89.9 ms |
| **+20 px** | **26.78%** | 80.70% | 0/295 (0.0%) | 1.36 | 61.2 ms |

## 2. Ayrı Eksperiment: Mürəkkəb Kənarı Haşiyəsi (Trim-to-Ink Margin Padding)

Ön-emal zamanı mürəkkəb sərhədindən sonra saxlanılan təhlükəsizlik haşiyəsinin tənzimlənməsi.

| Saxlanılan Haşiyə (Margin) | Xam (Raw) CER (%) ↓ | Xam (Raw) WER (%) ↓ | Tam Düzgün Sətir (Exact) ↑ | Gecikmə (ms) |
| :---: | :---: | :---: | :---: | :---: |
| **0 px** | **0.31%** | 1.73% | 248/295 (84.1%) | 81.3 ms |
| **4 px** | **0.30%** | 1.64% | 253/295 (85.8%) | 80.7 ms |
| **8 px** | **0.31%** | 1.82% | 248/295 (84.1%) | 90.1 ms |
| **12 px** | **0.36%** | 2.15% | 244/295 (82.7%) | 89.5 ms |
| **20 px** | **0.37%** | 2.19% | 243/295 (82.4%) | 84.6 ms |
