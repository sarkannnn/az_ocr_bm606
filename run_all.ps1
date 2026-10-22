$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path logs, results | Out-Null
$log = "logs\run_all.log"

function Log($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
}

function Run-Step($name, $argsList, $logfile) {
    Log "START: $name"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $py -u @argsList 2>&1 | Out-File -FilePath $logfile -Append -Encoding utf8
    Log ("END:   {0} (exit code {1}, {2:N1} min)" -f $name, $LASTEXITCODE, $sw.Elapsed.TotalMinutes)
}

function Wait-Training($pattern) {
    while ($true) {
        $p = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
             Where-Object { $_.CommandLine -like $pattern }
        if (-not $p) { break }
        Start-Sleep -Seconds 60
    }
}

Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win.Power]::SetThreadExecutionState([uint32]"0x80000001")
Log "Pipeline started. Sleep is blocked while this script runs."

Log "Waiting for the running word training (if any)..."
Wait-Training "*train.py*--mode word*"
if (-not (Test-Path "runs\word\best.pt")) {
    Log "runs\word\best.pt not found -> starting word training myself"
    Run-Step "train word" @("train.py","--mode","word","--train-dir","data/synth_word/train","--val-dir","data/synth_word/val","--out","runs/word","--epochs","25","--workers","0") "logs\word_train_rerun.log"
} else {
    Log "Word model found: runs\word\best.pt"
}

if (Test-Path "runs\word\best.pt") {
    Run-Step "eval word synthetic" @("evaluate.py","--ckpt","runs/word/best.pt","--data","data/synth_word/test","--out","results/word_synth") "logs\eval_word_synth.log"
    if (Test-Path "data\real_word\test\labels.tsv") {
        Run-Step "eval word real" @("evaluate.py","--ckpt","runs/word/best.pt","--data","data/real_word/test","--out","results/word_real") "logs\eval_word_real.log"
    } else { Log "No real word test set (data\real_word\test) - skipped" }
}

$tess = "C:\Program Files\Tesseract-OCR"
if ((Test-Path "$tess\tesseract.exe") -and ($env:Path -notlike "*Tesseract-OCR*")) { $env:Path = "$env:Path;$tess" }
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Run-Step "tesseract word synthetic" @("evaluate.py","--baseline","tesseract","--mode","word","--data","data/synth_word/test","--out","results/tess_word_synth") "logs\tess_word.log"
    if (Test-Path "data\real_word\test\labels.tsv") {
        Run-Step "tesseract word real" @("evaluate.py","--baseline","tesseract","--mode","word","--data","data/real_word/test","--out","results/tess_word_real") "logs\tess_word_real.log"
    }
} else { Log "Tesseract not installed - baseline skipped" }

if (-not (Test-Path "runs\line\best.pt")) {
    Run-Step "train line" @("train.py","--mode","line","--train-dir","data/synth_line/train","--val-dir","data/synth_line/val","--out","runs/line","--epochs","15","--workers","0") "logs\line_train.log"
} else { Log "runs\line\best.pt already exists - line training skipped" }
if (Test-Path "runs\line\best.pt") {
    Run-Step "eval line synthetic" @("evaluate.py","--ckpt","runs/line/best.pt","--data","data/synth_line/test","--out","results/line_synth") "logs\eval_line_synth.log"
    if (Test-Path "data\real_line\test\labels.tsv") {
        Run-Step "eval line real" @("evaluate.py","--ckpt","runs/line/best.pt","--data","data/real_line/test","--out","results/line_real") "logs\eval_line_real.log"
    } else { Log "No real line test set (data\real_line\test) - skipped" }
}

$sum = @()
Get-ChildItem results -Directory | Where-Object { $_.Name -ne "smoke" } | ForEach-Object {
    $m = Join-Path $_.FullName "metrics.json"
    if (Test-Path $m) {
        $j = Get-Content $m -Raw | ConvertFrom-Json
        $sum += ("{0,-22} CER {1,7:P2}  WER {2,7:P2}  n={3,-6} ms/img={4:N2}" -f $_.Name, $j.cer, $j.wer, $j.n, $j.ms_per_image)
    }
}
$sum | Out-File "logs\SUMMARY.txt" -Encoding utf8
Log "Summary written to logs\SUMMARY.txt"
$sum | ForEach-Object { Log $_ }

[void][Win.Power]::SetThreadExecutionState([uint32]"0x80000000")
Log "ALL DONE."
