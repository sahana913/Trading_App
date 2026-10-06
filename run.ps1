<#
run.ps1 - Start the whole platform, each part in its own PowerShell window:
    1. the market simulator (the heartbeat that advances a running market)
    2. the trader app   -> http://localhost:8501
    3. the admin app    -> http://localhost:8502

Usage, from the project folder:
    powershell -ExecutionPolicy Bypass -File .\run.ps1

To stop: close the three windows (or press Ctrl+C in each).
First time? Do the setup steps in README.md first (venv, data, database).
#>

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot                                   # the folder this script is in
$python = Join-Path $root ".venv\Scripts\python.exe"
$database = Join-Path $root "db\paper_trading.db"

# --- Checks: fail early with a helpful message -------------------------------
if (-not (Test-Path $python)) {
    Write-Host "No virtual environment found at .venv" -ForegroundColor Red
    Write-Host "Create it first:  python -m venv .venv ; .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $database)) {
    Write-Host "No database found at db\paper_trading.db" -ForegroundColor Red
    Write-Host "Build it first (see README.md): download -> pipeline -> seed."
    exit 1
}

# --- Start each part in a new window -------------------------------------------
function Start-Part([string]$title, [string]$command) {
    # -NoExit keeps the window open so you can read its log; the title tells the windows apart
    $full = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$root'; $command"
    Start-Process powershell -ArgumentList @("-NoExit", "-Command", $full) | Out-Null
    Write-Host "Started: $title"
}

Start-Part "Simulator" "& '$python' -m src.admin.ticker"
Start-Part "Trader app (8501)" "& '$python' -m streamlit run app/trader/app.py --server.port 8501"
# The simulator window already advances the market, so the admin app's own heartbeat is turned off
Start-Part "Admin app (8502)" "`$env:PAPER_TRADING_TICKER = 'off'; & '$python' -m streamlit run app/admin/admin_app.py --server.port 8502"

Write-Host ""
Write-Host "Trader app: http://localhost:8501" -ForegroundColor Green
Write-Host "Admin app:  http://localhost:8502" -ForegroundColor Green
Write-Host "Start or pause the market in the admin app under Market control."
