param([int]$Port=5288, [switch]$NoOpen)
$ErrorActionPreference='Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$harnessRoot = Join-Path $projectRoot 'vendor/deepseek-harness'
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$cliPath = Join-Path $harnessRoot 'apps/cli/lib/bin.js'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python environment is missing. See README.md setup.' }
if (-not (Test-Path -LiteralPath $cliPath)) { throw 'Harness is not built. Run setup-harness.ps1.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'harness-plugin/lib/client.js'))) { throw 'Control Web plugin is not built. Run setup-harness.ps1.' }
$env:DSH_HOME = Join-Path $projectRoot '.dsh'
$env:CONTROL_PYTHON = $pythonPath
$env:CONTROL_CODE_ROOT = $projectRoot
$overlay = Join-Path $projectRoot 'config/web.overlay.yml'
$launchArgs = @($cliPath, 'web', '--patch', $overlay, '--port', $Port)
if ($NoOpen) { $launchArgs += '--no-open' }
Push-Location $projectRoot
try { & node @launchArgs; exit $LASTEXITCODE } finally { Pop-Location }
