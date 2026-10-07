<#
run.ps1 - Start PaperDesk: one app, one address, for traders and admins.
    http://localhost:8501   log in or register; the pages you get depend on your role

The market heartbeat (which advances a running market) runs inside the app,
so this is the only window you need.

Usage, from the project folder:
    powershell -ExecutionPolicy Bypass -File .\run.ps1

To stop: close the window (or press Ctrl+C in it).
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
$busy = Get-NetTCPConnection -State Listen -LocalPort 8501 -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "Port 8501 is already in use: PaperDesk (or another app) is still running." -ForegroundColor Yellow
    Write-Host "Close that window first (or press Ctrl+C in it), then run this script again."
    exit 1
}

# --- Start the app in its own window -----------------------------------------
# --server.folderWatchList: Streamlit only reloads code inside app/; watching src/ too means
# an edit to shared code (src/ui.py, the engine...) is picked up without a restart.
$command = "`$Host.UI.RawUI.WindowTitle = 'PaperDesk (8501)'; Set-Location '$root'; " +
           "& '$python' -m streamlit run app/main.py --server.port 8501 --server.folderWatchList '$root\src'"
Start-Process powershell -ArgumentList @("-NoExit", "-Command", $command) | Out-Null

Write-Host "Started PaperDesk: http://localhost:8501" -ForegroundColor Green
Write-Host "Traders: log in or register. Admins: log in with an admin account to get the admin pages."
