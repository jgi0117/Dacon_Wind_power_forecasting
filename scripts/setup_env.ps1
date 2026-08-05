param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.uv-python'

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv was not found. Install uv and run this script again.'
}

if (-not (Test-Path -LiteralPath '.venv313\Scripts\python.exe')) {
    uv venv .venv313 --python 3.13
}

if (-not $SkipInstall) {
    uv pip install --python .venv313\Scripts\python.exe -r requirements-py313.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 dependency installation failed.' }
    uv pip install --python .venv313\Scripts\python.exe --editable .
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 project installation failed.' }
}

& .venv313\Scripts\python.exe --version
