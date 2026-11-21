# Offline regression. All file changes are confined to a new test-output directory.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-project-runtime.ps1')
$checkout = Split-Path -Parent $PSScriptRoot
$project = Join-Path $checkout ('test-output/runtime check ' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $project | Out-Null

# Use a synthetic download to prove that damaged bytes are rejected before extraction.
& {
    function Invoke-WebRequest([switch]$UseBasicParsing,[string]$Uri,[string]$OutFile) {
        if ($Uri -ne 'https://nodejs.org/dist/v24.22.3/node-v24.22.3-win-x64.zip') { throw 'Unexpected release URL' }
        [IO.File]::WriteAllBytes($OutFile, [byte[]]@(1,2,3))
    }
    $nodeTarget = Join-Path $project 'portable/node'
    try { Get-ControlPortableNode $nodeTarget; throw 'Invalid portable package accepted' }
    catch { if ($_.Exception.Message -notmatch 'SHA-256 校验失败') { throw } }
    if (Test-Path -LiteralPath $nodeTarget) { throw 'An incomplete Node.js installation was published' }
}

# Supply a fake tool discovery result and make any unexpected network/installation call fail.
function Find-ControlNode([string]$PrivateBin) { return (Join-Path $project 'runtime/node.exe') }
function Get-ControlPortableNode { throw 'Unexpected Node download during preflight' }
function Install-ControlUserPackage { throw 'Unexpected winget installation during preflight' }
$status = Initialize-ControlProjectRuntime -ProjectRoot $project -CheckOnly
if ($status.Ready -or '网页配置中的项目插件注册' -notin $status.Missing -or $status.Missing.Count -ne 4) { throw 'First launch did not report all missing stages' }
if (@(Get-ChildItem -LiteralPath $project -Force | Where-Object Name -ne 'portable').Count) { throw 'CheckOnly modified the checkout' }

# A failed/copied Python environment must survive. Its user files may be useful for recovery.
$venv = New-Item -ItemType Directory -Path (Join-Path $project '.venv')
Set-Content -LiteralPath (Join-Path $venv.FullName 'keep-me') -Value 'keep'
try { Initialize-ControlProjectRuntime -ProjectRoot $project | Out-Null; throw 'Broken environment accepted' }
catch { if ($_.Exception.Message -notmatch '移到备份目录') { throw } }
if (-not (Test-Path -LiteralPath (Join-Path $venv.FullName 'keep-me'))) { throw 'Existing environment changed' }

# Ready state depends on the invoking profile and its actual project link.
$userHome = Join-Path $project 'other-home'
$profilePath = Join-Path $userHome 'profiles/web'
$bundlePath = Join-Path $project 'harness-plugin'
New-Item -ItemType Directory -Path (Join-Path $bundlePath 'src') -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $bundlePath 'src/index.mjs') | Out-Null
$scope = New-Item -ItemType Directory -Path (Join-Path $profilePath 'node_modules/@control-engineering') -Force
New-Item -ItemType Junction -Path (Join-Path $scope.FullName 'harness-bundle') -Target $bundlePath | Out-Null
$profile = @{dependencies=@{'@control-engineering/harness-bundle'='link:D:/previous-checkout/harness-plugin'};dsh=@{profile=@{bundles=@('@control-engineering/harness-bundle')}}}
$manifest = Join-Path $profilePath 'package.json'
$profile | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
if (Test-ControlProfile $userHome $project) { throw 'A profile linked to another checkout was accepted' }
$profile.dependencies.'@control-engineering/harness-bundle' = 'link:' + (Join-Path $project 'harness-plugin').Replace('\','/')
$profile | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
if (-not (Test-ControlProfile $userHome $project)) { throw 'Profile link to the current checkout rejected' }
if (Test-ControlProfile (Join-Path $project '.dsh') $project) { throw 'A second home incorrectly reused another profile' }
Write-Output 'PASS: official archive hash rejection, offline preflight, partial venv preservation, per-home plugin registration'
