#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "======================================================================"
echo "  AZERBAYCAN DILI OCR (CRNN-V2 + CTC) - AVTOMATIK QURASDIRMA VE BASLATMA"
echo "======================================================================"
echo ""

# 1. Create virtual environment if missing
if [ ! -x ".venv/bin/python" ]; then
    echo "[*] Virtual muhit (.venv) tapilmadi. Yeni muhit yaradilir..."
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv .venv
    elif command -v python >/dev/null 2>&1; then
        python -m venv .venv
    else
        echo "[!] XETA: Sistemde Python 3 tapilmadi! Zehmet olmasa Python 3.10+ qurasdirin."
        exit 1
    fi
fi

# 2. Install dependencies if any core package is missing
if ! .venv/bin/python -c "import torch, cv2, gradio, symspellpy, pillow_heif, segno, pytesseract" >/dev/null 2>&1; then
    echo "[*] Lazimi Python paketleri qurasdirilir (requirements.txt)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    echo "[+] Butun paketler ugurla qurasdirildi!"
    echo ""
fi

# 3. Launch server with public mobile HTTPS tunnel
echo "[*] Server ve Mobil HTTPS Canli Kamera linki ise salinir..."
echo ""
exec .venv/bin/python -u app.py --share "$@"
