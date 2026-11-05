@echo off
title Azerbaycan Dili OCR - Canli Mobil Kamera ve Veb Arayuzu
echo ======================================================================
echo   AZERBAYCAN DILI OCR (CRNN + CTC) - CANLI KAMERA VE VEB ARAYUZU
echo ======================================================================
echo.
echo Server ve Mobil HTTPS Canli Kamera linki ise salinir...
echo.
.\.venv\Scripts\python.exe -u app.py --share %*
pause
