# ==============================================================================
# Windows 11 一鍵建置 GPG + Cooldown 實驗環境腳本
# 用法（於 GPG\open-r1 目錄執行）：
#   powershell -ExecutionPolicy Bypass -File scripts\setup_win11.ps1
# ==============================================================================
$ErrorActionPreference = "Stop"

$OpenR1Root = Split-Path -Parent $PSScriptRoot
Set-Location $OpenR1Root

Write-Host "[1/5] 檢查 Python 3.11..." -ForegroundColor Cyan
$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        $ver = & py -3.11 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $pyCmd = @("py", "-3.11")
            Write-Host "  找到 $ver"
        }
    } catch {}
}
if (-not $pyCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $ver = & python --version 2>&1
    if ($ver -match "3\.11\.") {
        $pyCmd = @("python")
        Write-Host "  找到 $ver"
    }
}
if (-not $pyCmd) {
    throw "找不到 Python 3.11，請先安裝 Python 3.11.x（建議 3.11.9）後再執行本腳本。"
}

$VenvDir = Join-Path $OpenR1Root "openr1_venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[2/5] 建立虛擬環境 openr1_venv..." -ForegroundColor Cyan
    if ($pyCmd.Length -eq 2) {
        & $pyCmd[0] $pyCmd[1] -m venv $VenvDir
    } else {
        & $pyCmd[0] -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0) { throw "建立虛擬環境失敗" }
} else {
    Write-Host "[2/5] 已存在虛擬環境 openr1_venv，略過建立" -ForegroundColor Cyan
}

Write-Host "[3/5] 安裝前置工具 (pip / pyyaml / packaging)..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip 升級失敗" }
& $VenvPython -m pip install pyyaml packaging
if ($LASTEXITCODE -ne 0) { throw "安裝 pyyaml / packaging 失敗" }

Write-Host "[4/5] 依 experiment_config.yaml 安裝鎖定版本套件（不執行 pip install -e .）..." -ForegroundColor Cyan
$env:PYTHONPATH = Join-Path $OpenR1Root "src"
$env:PYTHONUTF8 = "1"
& $VenvPython -m open_r1.experiment_config --config experiment_config.yaml --sync-env
if ($LASTEXITCODE -ne 0) { throw "套件安裝失敗" }

Write-Host "[5/5] 執行環境與 CUDA GPU 嚴格驗證..." -ForegroundColor Cyan
& $VenvPython -m open_r1.experiment_config --config experiment_config.yaml --check --strict
if ($LASTEXITCODE -ne 0) { throw "環境驗證未通過" }

Write-Host "`n[DONE] Windows 11 環境建置與驗證完成！" -ForegroundColor Green
Write-Host "下一步可執行 dry-run 測試設定：" -ForegroundColor Yellow
Write-Host "  .\openr1_venv\Scripts\python.exe scripts\compare_gpg_vs_cooldown.py --dry_run"
