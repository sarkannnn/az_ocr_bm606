# AzOCR-V2: Gece Ablasyon Modelleri Təlimi və Avtomatlaşdırma Skripti
# Bu skript kompüterin yuxu rejiminə keçməsinin qarşısını alır və təlimləri ardıcıl icra edir.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path logs, runs, results | Out-Null
$masterLog = "logs\night_training.log"

function Write-NightLog($msg) {
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] $msg"
    Add-Content -Path $masterLog -Value $line -Encoding UTF8
    Write-Host $line -ForegroundColor Cyan
}

# Windows yuxu rejimini blokla
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win.Power]::SetThreadExecutionState([uint32]"0x80000001")
Write-NightLog "GECE TƏLİM SKRİPTİ BAŞLADI. Yuxu rejimi bloklandı."

try {
    # -------------------------------------------------------------
    # ADIM 1: Grup B üçün Veri Setinin Hazırlanması (10 font, V1)
    # -------------------------------------------------------------
    if (-not (Test-Path "data\synth_line_v1\train\labels.tsv")) {
        Write-NightLog "ADIM 1: data\synth_line_v1 hazırlanır..."
        & $py -u scripts\prepare_group_b_data.py
        if ($LASTEXITCODE -ne 0) { throw "Grup B veri seti hazırlanarkən xəta baş verdi!" }
        Write-NightLog "ADIM 1 TAMAMLANDI: data\synth_line_v1 uğurla hazırlandı."
    } else {
        Write-NightLog "ADIM 1 KEÇİLDİ: data\synth_line_v1 artıq mövcuddur."
    }

    # -------------------------------------------------------------
    # ADIM 2: GRUP B TƏLİMİ (Standart Fontlar + V2 Augmentasiyası)
    # -------------------------------------------------------------
    if (-not (Test-Path "runs\ablation_group_b_aug\best.pt")) {
        Write-NightLog "ADIM 2: GRUP B Təlimi Başlayır (Standart 8 font, V2 Augmentasiya VAR)..."
        $swB = [Diagnostics.Stopwatch]::StartNew()
        & $py -u train_v2.py --mode line --train-dir data/synth_line_v1/train --val-dir data/synth_line_v1/val --out runs/ablation_group_b_aug --seed 42 --epochs 25 --workers 0 2>&1 | Out-File -FilePath logs\ablation_group_b_aug.log -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Grup B təlimində xəta baş verdi! Bax: logs\ablation_group_b_aug.log" }
        Write-NightLog ("ADIM 2 TAMAMLANDI: Grup B modeli runs\ablation_group_b_aug\best.pt ({0:N1} dəq)" -f $swB.Elapsed.TotalMinutes)
    } else {
        Write-NightLog "ADIM 2 KEÇİLDİ: runs\ablation_group_b_aug\best.pt artıq mövcuddur."
    }

    # -------------------------------------------------------------
    # ADIM 3: GRUP C TƏLİMİ (Genişletilmiş 18 Font + Augmentasiya YOK)
    # -------------------------------------------------------------
    if (-not (Test-Path "runs\ablation_group_c_noaug\best.pt")) {
        Write-NightLog "ADIM 3: GRUP C Təlimi Başlayır (Geniş 18 font, Augmentasiya YOK)..."
        $swC = [Diagnostics.Stopwatch]::StartNew()
        & $py -u train_v2.py --mode line --train-dir data/synth_line_v2/train --val-dir data/synth_line_v2/val --out runs/ablation_group_c_noaug --no-aug --seed 42 --epochs 25 --workers 0 2>&1 | Out-File -FilePath logs\ablation_group_c_noaug.log -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Grup C təlimində xəta baş verdi! Bax: logs\ablation_group_c_noaug.log" }
        Write-NightLog ("ADIM 3 TAMAMLANDI: Grup C modeli runs\ablation_group_c_noaug\best.pt ({0:N1} dəq)" -f $swC.Elapsed.TotalMinutes)
    } else {
        Write-NightLog "ADIM 3 KEÇİLDİ: runs\ablation_group_c_noaug\best.pt artıq mövcuddur."
    }

    # -------------------------------------------------------------
    # ADIM 4: 2x2 FAKTÖRİYEL ABLASYON DEĞERLENDİRMESİ (15 Belge Benchmark)
    # -------------------------------------------------------------
    Write-NightLog "ADIM 4: 15 Belgelik Benchmark üzərində 2x2 qiymətləndirmə başlayır..."
    & $py -u scripts\eval_ablation_models.py
    Write-NightLog "ADIM 4 TAMAMLANDI: results\ablation_2x2_results.json yaradıldı."

    Write-NightLog "BÜTÜN GECE İŞLƏRİ UĞURLA BAŞA ÇATDI!"
}
catch {
    $err = $_.Exception.Message
    Write-NightLog "KRİTİK XƏTA: $err"
    Add-Content -Path "logs\night_training_error.log" -Value "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $err" -Encoding UTF8
    Write-Host "Təlim dayandırıldı. Xəta logs\night_training_error.log faylına yazıldı." -ForegroundColor Red
}
finally {
    # Yuxu blokunu qaldır
    [void][Win.Power]::SetThreadExecutionState([uint32]"0x80000000")
    Write-NightLog "Yuxu bloku qaldırıldı. Sistem normal rejimə qayıtdı."
}
