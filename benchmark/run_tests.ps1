# TTR-SUITE Benchmark Suite - Script di test progressivi
# Uso: powershell -ExecutionPolicy Bypass -File .\run_tests.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py  = "$PSScriptRoot\.venv\Scripts\python.exe"
$pip = "$PSScriptRoot\.venv\Scripts\pip.exe"

if (-not (Test-Path $py)) {
    Write-Host "Venv non trovato, lo creo..." -ForegroundColor Yellow
    python -m venv "$PSScriptRoot\.venv"
}

Write-Host "Installo/aggiorno dipendenze..." -ForegroundColor Yellow
& $pip install -q -r "$PSScriptRoot\requirements.txt"

function Banner($msg) {
    $line = "=" * 60
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

# FASE 0 - Dry run (infrastruttura, nessuna chiamata AI)
Banner "FASE 0 - Dry run"
& $py benchmark_runner.py --dry-run
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERRORE: dry-run fallito. Correggi prima di procedere." -ForegroundColor Red
    exit 1
}

# FASE 1 - LegalBench quick su qwen3-30b-a3b
Banner "FASE 1 - LegalBench quick [qwen3-30b-a3b]"
& $py benchmark_runner.py --benchmarks legalbench --models qwen3-30b-a3b --quick --no-pull

# FASE 2 - Tutti i benchmark quick su qwen3-30b-a3b
Banner "FASE 2 - Full quick [qwen3-30b-a3b]"
& $py benchmark_runner.py --models qwen3-30b-a3b --quick --no-pull

# FASE 3 - Confronto modelli locali quick
# Prerequisito: ollama pull qwen3:32b-q4_K_M e mistral-small:24b
Banner "FASE 3 - Confronto modelli locali quick"
& $py benchmark_runner.py --models qwen3-30b-a3b qwen3-32b mistral-small-24b --quick --no-pull

# FASE 4 - Claude Sonnet 4.6 quick (richiede ANTHROPIC_API_KEY)
Banner "FASE 4 - Claude Sonnet 4.6 quick [API]"
if ($env:ANTHROPIC_API_KEY) {
    & $py benchmark_runner.py --models claude-sonnet-4-6 --quick --no-pull
} else {
    Write-Host "SKIP: ANTHROPIC_API_KEY non impostata." -ForegroundColor Yellow
    Write-Host "Per abilitare:" -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "sk-ant-..."' -ForegroundColor Yellow
    Write-Host "  & $py benchmark_runner.py --models claude-sonnet-4-6 --quick --no-pull" -ForegroundColor Yellow
}

Banner "Completato - risultati in C:\TTR_Benchmark\results\"
Write-Host "Apri summary_*.xlsx per il report." -ForegroundColor Green
