@echo off
chcp 65001 >nul
cd /d "%~dp0"
title AzOCR-V2 - Elmi-Tedqiqat ve Mobil Kamera OCR Platformasi

echo ======================================================================
echo   AZERBAYCAN DILI OCR (CRNN-V2 + CTC) - AVTOMATIK QURASDIRMA VE BASLATMA
echo ======================================================================
echo.

REM 1. Virtual muhit (.venv) yoxlanilir, yoxdursa avtomatik yaradilir
if not exist ".venv\Scripts\python.exe" (
    echo [*] Virtual muhit ^(.venv^) tapilmadi. Yeni muhit yaradilir...
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        python -m venv .venv
    ) else (
        where py >nul 2>&1
        if %errorlevel% equ 0 (
            py -3 -m venv .venv
        ) else (
            echo [!] XETA: Sistemde Python tapilmadi!
            echo     Zehmet olmasa Python 3.10+ qurasdirin ^(https://www.python.org/downloads/^)
            echo     ve qurasdirarken "Add Python to PATH" secimini isareleyin.
            pause
            exit /b 1
        )
    )
)

REM 2. Lazimi kitabxanalarin (.venv daxilinde) movcudlugu yoxlanilir
".venv\Scripts\python.exe" -c "import torch, cv2, gradio, symspellpy, pillow_heif, segno, pytesseract" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Lazimi Python paketleri qurasdirilir ^(requirements.txt^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [!] XETA: Paketlerin qurasdirilmasi zamani xeta bas verdi.
        pause
        exit /b 1
    )
    echo [+] Butun paketler ugurla qurasdirildi!
    echo.
)

REM 3. Server ve Mobil HTTPS Canli Kamera / QR linki ise salinir
echo [*] Server ve Mobil HTTPS Canli Kamera linki ise salinir...
echo.
".venv\Scripts\python.exe" -u app.py --share %*
pause
