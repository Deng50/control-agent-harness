param(
    [Parameter(Mandatory=$true)][string]$OfficialFixture,
    [Parameter(Mandatory=$true)][ValidateSet('busy','cancel','timeout','file-timeout','repair-native')][string]$Phase
)
$ErrorActionPreference='Stop'
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$testRoot = Join-Path $project 'test-output'
$original = [IO.Path]::GetFullPath($OfficialFixture)
if ([IO.Path]::GetDirectoryName($original) -ne $testRoot) { throw 'Expected the official fixture under test-output.' }
$source = 'session'
$active = Join-Path $original 'active-test-launch.txt'
if (Test-Path -LiteralPath $active) {
    $source = (Get-Content -Raw -LiteralPath $active).Trim()
    if ($source -notmatch '^session-[a-zA-Z0-9-]+$') { throw 'Invalid official launch name.' }
}
$old = Get-Content -Raw -LiteralPath (Join-Path $original "$source/appdata/control-harness-session.json") | ConvertFrom-Json
$program = (Get-Content -Raw -LiteralPath (Join-Path $original "$source/launch.json") | ConvertFrom-Json).matlab_executable
$fixture = Join-Path $testRoot ('host-fault-'+$Phase+'-'+[guid]::NewGuid().ToString('N').Substring(0,10))
$launch = Join-Path $fixture 'session'
New-Item -ItemType Directory -Path (Join-Path $fixture 'workspace') | Out-Null
Write-Output $fixture
& (Join-Path $PSScriptRoot 'start-matlab-session.ps1') -MatlabExecutable $program -MatlabMcpExecutable $old.mcp_executable -SimulinkToolkitRoot $old.toolkit_root -MatlabMcpToolboxRoot $old.toolbox_root -SessionDirectory $launch
$record = Get-Content -Raw -LiteralPath (Join-Path $launch 'launch.json') | ConvertFrom-Json
$readyFile = Join-Path $launch 'appdata/control-harness-session.json'
$deadline = (Get-Date).AddMinutes(3)
while (-not (Test-Path -LiteralPath $readyFile)) {
    if ((Get-Date) -gt $deadline) { throw 'Dedicated startup timed out. Inspect its launch record.' }
    Start-Sleep -Milliseconds 300
}
$ready = Get-Content -Raw -LiteralPath $readyFile | ConvertFrom-Json
if ($ready.session_id -ne $record.session_id -or $ready.private_appdata -ne (Join-Path $launch 'appdata') -or
    $ready.project_root -ne $project -or $ready.matlab_release -ne '2024b' -or $ready.status -ne 'ready') { throw 'Unexpected test process identity.' }
$owned = Get-Process -Id $ready.matlab_pid
$expected = $owned.StartTime.ToUniversalTime().ToFileTimeUtc()
$programRoot = [IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($program))
if ($owned.ProcessName -ne 'MATLAB' -or -not $owned.Path.StartsWith(($programRoot+'\'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected test executable.' }
try {
    if ($Phase -eq 'repair-native') {
        & node (Join-Path $PSScriptRoot 'accept-matlab-repair-native.mjs') $fixture
    } else {
        & node (Join-Path $PSScriptRoot 'accept-matlab-fault.mjs') $fixture $Phase
    }
    if ($LASTEXITCODE -ne 0) { throw "Dedicated acceptance failed. Inspect the evidence under $fixture." }
} finally {
    # Dedicated synthetic fixture. Match the bound process birth before cleanup.
    $now = Get-Process -Id $ready.matlab_pid -ErrorAction SilentlyContinue
    if ($now -and $now.StartTime.ToUniversalTime().ToFileTimeUtc() -eq $expected -and $now.Path -eq $owned.Path) {
        Stop-Process -InputObject $now -Force
        $now.WaitForExit(15000) | Out-Null
        if (-not $now.HasExited) { throw 'Dedicated test process did not exit.' }
        [ordered]@{pid=$ready.matlab_pid;session_id=$ready.session_id;creation_time=$expected.ToString();verified_exit=$true} |
            ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $fixture 'exit.json')
    }
}
