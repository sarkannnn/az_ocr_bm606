@echo off
title Azerbaycan Dili OCR - Arayuz
echo ======================================================================
echo   AZERBAYCAN DILI OCR (CRNN + CTC) - VEB VE MOBIL ARAYUZ
echo ======================================================================
echo.
echo Server ise salinir...
echo.
.\.venv\Scripts\python.exe app.py %*
pause
