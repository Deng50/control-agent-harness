param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$WorkerArguments
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Configure the project Python environment first.' }
& $pythonPath -m control_harness @WorkerArguments
exit $LASTEXITCODE
