param(
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_XET = '1'
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = Join-Path (Split-Path -Parent $projectRoot) '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
$runName = 'test_visible_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$logPath = Join-Path $projectRoot ('outputs\' + $runName + '.log')
Start-Transcript -Path $logPath
try {
    Write-Host "Orthodox Qwen UA | real 0.8B test | device=$Device" -ForegroundColor Cyan
    & $pythonExe -u (Join-Path $PSScriptRoot 'test_training.py') --device $Device --output ('outputs/' + $runName)
    $trainingExitCode = $LASTEXITCODE
    Write-Host "Training exit code: $trainingExitCode"
    Write-Host "Transcript: $logPath"
} finally {
    Stop-Transcript
}
