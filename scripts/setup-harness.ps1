param([string]$HarnessSource='', [switch]$SkipHarnessBuild)
$ErrorActionPreference='Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$harnessRoot = Join-Path $projectRoot 'vendor/deepseek-harness'
$pin = 'f3803c96974ad4a94195fb97e5c637e9c2a3afe3'
if ([string]::IsNullOrWhiteSpace($HarnessSource)) {
    $siblingSource = Join-Path (Split-Path $projectRoot -Parent) 'deepseek-harness'
    $HarnessSource = if (Test-Path -LiteralPath (Join-Path $siblingSource '.git')) {
        $siblingSource
    } else {
        'https://github.com/deepseek-ai/deepseek-harness.git'
    }
}
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the Python environment first: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e .'
}
$env:CONTROL_PYTHON = $pythonPath
$env:CONTROL_CODE_ROOT = $projectRoot
if (-not (Test-Path -LiteralPath $harnessRoot)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'vendor') | Out-Null
    & git clone --no-hardlinks -- $HarnessSource $harnessRoot
    if ($LASTEXITCODE -ne 0) { throw 'Harness clone failed' }
    & git -C $harnessRoot checkout --detach $pin
    if ($LASTEXITCODE -ne 0) { throw 'Pinned Harness revision is unavailable in source' }
}
$revision = (& git -C $harnessRoot rev-parse HEAD).Trim()
if ($revision -ne $pin) { throw "Unexpected Harness revision: $revision. No automatic reset will be performed." }
$env:TEMP = Join-Path $projectRoot 'test-output/temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
Push-Location $harnessRoot
try {
    & pnpm install --frozen-lockfile --store-dir (Join-Path $projectRoot '.pnpm-store') --fetch-timeout 300000 --network-concurrency 4
    if ($LASTEXITCODE -ne 0) { throw 'Harness dependency install failed' }
    if (-not $SkipHarnessBuild) {
        & pnpm run build:artifacts
        if ($LASTEXITCODE -ne 0) { throw 'Harness artifact build failed' }
        & pnpm run build:check
        if ($LASTEXITCODE -ne 0) { throw 'Harness type checks failed' }
    }
} finally { Pop-Location }
Push-Location (Join-Path $projectRoot 'harness-plugin')
try {
    & pnpm install --offline --ignore-scripts
    if ($LASTEXITCODE -ne 0) { throw 'Local plugin links failed' }
    & node (Join-Path $harnessRoot 'node_modules/tsdown/dist/run.mjs') --config tsdown.config.ts
    if ($LASTEXITCODE -ne 0) { throw 'Plugin build failed' }
} finally { Pop-Location }
$env:DSH_HOME=Join-Path $projectRoot '.dsh'
$cliPath=Join-Path $harnessRoot 'apps/cli/lib/bin.js'
& node $cliPath --profile web --dump-config | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Profile initialization failed' }
$pluginSpec='link:' + (Join-Path $projectRoot 'harness-plugin').Replace('\','/')
& node $cliPath plugin --profile web add $pluginSpec --offline --ignore-scripts
if ($LASTEXITCODE -ne 0) { throw 'Harness bundle installation failed' }
Write-Output 'Harness and external control plugin are ready. Run scripts/start-web.ps1.'
