@echo off
chcp 65001 > nul
echo ========================================================
echo   AzOCR-V2 Gece Egitim ve Ablasyon Sureci Baslatiliyor
echo ========================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_night_training.ps1"
pause
