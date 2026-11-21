param(
  [int]$Port=5288,
  [switch]$NoOpen,
  [switch]$Matlab,
  [string]$MatlabMcpExecutable=$env:CONTROL_MATLAB_MCP_EXECUTABLE,
  [string]$SimulinkToolkitRoot=$env:CONTROL_SIMULINK_TOOLKIT_ROOT,
  [string]$MatlabSessionAppData=$env:CONTROL_MATLAB_SESSION_APPDATA,
  [string]$MatlabResourceDirectory=$env:CONTROL_MATLAB_RESOURCE_DIRECTORY,
  [string[]]$AdditionalPatch=@(),
  [switch]$ValidateOnly,
  [string]$Workspace
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$harnessRoot = Join-Path $projectRoot 'vendor/deepseek-harness'
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$cliPath = Join-Path $harnessRoot 'apps/cli/lib/bin.js'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python environment is missing. See README.md setup.' }
if ($Workspace) {
  # Optional read-only diagnosis of this path. Session cwd is selected separately in the Web UI.
  # The session admission path checks the real restricted child in all cases.
  try { $workspaceDiagnosis = & (Join-Path $PSScriptRoot 'prepare-workspace.ps1') -Workspace $Workspace }
  catch { if ($workspaceDiagnosis) { Write-Output $workspaceDiagnosis }; throw }
}
if (-not (Test-Path -LiteralPath $cliPath)) { throw 'Harness is not built. Run setup-harness.ps1.' }
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'harness-plugin/lib/client.js'))) { throw 'Control Web plugin is not built. Run setup-harness.ps1.' }
$env:DSH_HOME = Join-Path $projectRoot '.dsh'
$env:CONTROL_PYTHON = $pythonPath
$env:CONTROL_CODE_ROOT = $projectRoot
$overlay = Join-Path $projectRoot 'config/web.overlay.yml'
$patches = @($overlay)
$matlabSessionPid = $null
if ($Matlab) {
  foreach ($variable in @('MW_MCP_SERVER_MATLAB_SESSION_CONNECTION_DETAILS','MW_MCP_SERVER_MATLAB_ROOT','MW_MCP_SERVER_INITIAL_WORKING_FOLDER','MW_MCP_SERVER_MATLAB_DISPLAY_MODE')) {
    if (-not [string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($variable,'Process'))) { throw ($variable + ' conflicts with private existing-session discovery; clear it in this launch environment. Its value was not printed.') }
  }
  foreach ($entry in @(
    @{Name='MatlabMcpExecutable';Value=$MatlabMcpExecutable},
    @{Name='SimulinkToolkitRoot';Value=$SimulinkToolkitRoot},
    @{Name='MatlabSessionAppData';Value=$MatlabSessionAppData},
    @{Name='MatlabResourceDirectory';Value=$MatlabResourceDirectory}
  )) {
    if ([string]::IsNullOrWhiteSpace($entry.Value)) { throw ($entry.Name + ' must be supplied when -Matlab is enabled.') }
    if (-not (Test-ControlFullyQualifiedWindowsPath $entry.Value)) { throw ($entry.Name + ' must be an absolute path.') }
  }
  if (-not (Test-Path -LiteralPath $MatlabMcpExecutable -PathType Leaf)) { throw 'MATLAB MCP executable was not found.' }
  if (-not (Test-Path -LiteralPath (Join-Path $SimulinkToolkitRoot 'tools/tools.json') -PathType Leaf)) { throw 'Simulink Toolkit tools/tools.json was not found.' }
  $privateAppData = [IO.Path]::GetFullPath($MatlabSessionAppData).TrimEnd('\')
  if ($privateAppData -eq [IO.Path]::GetFullPath($env:APPDATA).TrimEnd('\')) { throw 'Use a dedicated private session APPDATA, not the user global APPDATA.' }
  $markerPath = Join-Path $privateAppData 'control-harness-session.json'
  $detailsPath = Join-Path $privateAppData 'MathWorks/MATLAB MCP Server/v1/sessionDetails.json'
  if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf) -or -not (Test-Path -LiteralPath $detailsPath -PathType Leaf)) { throw 'Dedicated MATLAB is not initialized. Run the explicit session bootstrap first.' }
  $marker = Get-Content -LiteralPath $markerPath -Raw -Encoding UTF8 | ConvertFrom-Json
  # This file contains connector credentials. Read only for PID agreement; never print it.
  try { $details = Get-Content -LiteralPath $detailsPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop }
  catch { throw 'The private MATLAB connection descriptor is unreadable or invalid; its contents were not printed.' }
  if ($marker.kind -ne 'control-harness-dedicated-matlab-v1' -or $marker.status -ne 'ready' -or [int]$marker.pid -ne [int]$details.pid) { throw 'Dedicated MATLAB marker and connector PID do not agree.' }
  if ([IO.Path]::GetFullPath($marker.project_root).TrimEnd('\') -ne $projectRoot.TrimEnd('\')) { throw 'Dedicated MATLAB was prepared for a different project checkout.' }
  $matlabProcess = Get-Process -Id ([int]$marker.pid) -ErrorAction Stop
  if ($matlabProcess.ProcessName -ne 'MATLAB') { throw 'The dedicated MATLAB PID no longer identifies MATLAB.' }
  if ([int]$marker.matlab_pid -ne [int]$marker.pid -or [string]::IsNullOrWhiteSpace($marker.session_id)) { throw 'Dedicated MATLAB identity metadata is incomplete.' }
  if ($marker.createdAt -is [DateTimeOffset]) {
    $markerCreatedUtc = $marker.createdAt.UtcDateTime
  } elseif ($marker.createdAt -is [DateTime]) {
    if ($marker.createdAt.Kind -eq [DateTimeKind]::Unspecified) { throw 'Dedicated identity creation time must carry an explicit UTC offset.' }
    $markerCreatedUtc = $marker.createdAt.ToUniversalTime()
  } else {
    $createdText = [string]$marker.createdAt
    if ($createdText -notmatch '(Z|[+-]\d{2}:\d{2})$') { throw 'Dedicated identity creation time must carry an explicit UTC offset.' }
    $markerCreatedUtc = [DateTimeOffset]::Parse($createdText,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::None).UtcDateTime
  }
  if ($matlabProcess.StartTime.ToUniversalTime() -gt $markerCreatedUtc.AddSeconds(1)) { throw 'The recorded MATLAB PID was reused after this session identity was written.' }
  $matlabSessionPid = [int]$marker.pid
  $env:CONTROL_MATLAB_BOUND_EXECUTABLE = $matlabProcess.Path
  $env:CONTROL_MATLAB_MCP_EXECUTABLE = [IO.Path]::GetFullPath($MatlabMcpExecutable)
  $env:CONTROL_SIMULINK_TOOLKIT_ROOT = [IO.Path]::GetFullPath($SimulinkToolkitRoot).Replace('\','/')
  $env:CONTROL_MATLAB_SESSION_APPDATA = $privateAppData
  $env:CONTROL_MATLAB_RESOURCE_DIRECTORY = [IO.Path]::GetFullPath($MatlabResourceDirectory)
  $env:CONTROL_MATLAB_IDENTITY_FILE = $markerPath
  $patches += Join-Path $projectRoot 'config/matlab.overlay.yml'
}
foreach ($patch in $AdditionalPatch) {
  $patchPath = [IO.Path]::GetFullPath($patch)
  if (-not (Test-Path -LiteralPath $patchPath -PathType Leaf)) { throw 'An additional patch file was not found.' }
  $patches += $patchPath
}
$launchArgs = @($cliPath, 'web')
foreach ($patch in $patches) { $launchArgs += @('--patch', $patch) }
$launchArgs += @('--port', $Port)
if ($NoOpen) { $launchArgs += '--no-open' }
if ($ValidateOnly) {
  [ordered]@{valid=$true;matlab_enabled=[bool]$Matlab;port=$Port;patches=$patches;matlab_session_pid=$matlabSessionPid;
    workspace_diagnosis=if ($Workspace) {$workspaceDiagnosis | ConvertFrom-Json} else {$null}} | ConvertTo-Json -Depth 6
  return
}
if ($Matlab) {
  $previousPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH','Process')
  try {
    $env:PYTHONPATH = Join-Path $projectRoot 'src'
    & $pythonPath -m control_harness.matlab_resource bind-session --directory $env:CONTROL_MATLAB_RESOURCE_DIRECTORY --pid ([string]$marker.matlab_pid) --session-id ([string]$marker.session_id) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Dedicated MATLAB resource binding was rejected; inspect the existing binding before starting another Host.' }
  } finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH',$previousPythonPath,'Process')
  }
}
Push-Location $projectRoot
try { & node @launchArgs; exit $LASTEXITCODE } finally { Pop-Location }
