param(
  [Parameter(Mandatory=$true)][string]$MatlabExecutable,
  [Parameter(Mandatory=$true)][string]$MatlabMcpExecutable,
  [Parameter(Mandatory=$true)][string]$SimulinkToolkitRoot,
  [Parameter(Mandatory=$true)][string]$MatlabMcpToolboxRoot,
  [Parameter(Mandatory=$true)][string]$SessionDirectory,
  [switch]$ValidateOnly
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
foreach ($path in @($MatlabExecutable,$MatlabMcpExecutable,$SimulinkToolkitRoot,$MatlabMcpToolboxRoot,$SessionDirectory)) {
  if (-not (Test-ControlFullyQualifiedWindowsPath $path)) { throw 'All dedicated-session paths must be absolute.' }
}
if (-not (Test-Path -LiteralPath $MatlabExecutable -PathType Leaf)) { throw 'MATLAB executable was not found.' }
if (-not (Test-Path -LiteralPath $MatlabMcpExecutable -PathType Leaf)) { throw 'MATLAB MCP executable was not found.' }
if (-not (Test-Path -LiteralPath (Join-Path $SimulinkToolkitRoot 'tools/tools.json') -PathType Leaf)) { throw 'Installed Simulink Toolkit was not found.' }
if (-not (Test-Path -LiteralPath (Join-Path $MatlabMcpToolboxRoot 'shareMATLABSession.m') -PathType Leaf)) { throw 'Installed MATLAB MCP Server Toolbox was not found.' }
$sessionRoot = [IO.Path]::GetFullPath($SessionDirectory)
if ((Test-Path -LiteralPath (Join-Path $sessionRoot 'launch.json')) -or (Test-Path -LiteralPath (Join-Path $sessionRoot 'appdata/control-harness-session.json'))) { throw 'Session directory has a prior launch. Inspect that PID and choose a fresh directory; do not start a duplicate.' }
$privateAppData = Join-Path $sessionRoot 'appdata'
if ($ValidateOnly) {
  [ordered]@{valid=$true;starts_matlab=$false;session_directory=$sessionRoot;private_appdata=$privateAppData;mode='Dedicated new process, then MCP existing binding'}|ConvertTo-Json
  return
}
New-Item -ItemType Directory -Force -Path $privateAppData | Out-Null
$taskEnvironment = @{
  CONTROL_MATLAB_PROJECT_ROOT=$projectRoot
  CONTROL_MATLAB_SESSION_APPDATA=$privateAppData
  CONTROL_MATLAB_DEDICATED_SESSION_ID=[guid]::NewGuid().ToString()
  CONTROL_SIMULINK_TOOLKIT_ROOT=[IO.Path]::GetFullPath($SimulinkToolkitRoot)
  CONTROL_MATLAB_MCP_TOOLBOX_ROOT=[IO.Path]::GetFullPath($MatlabMcpToolboxRoot)
  CONTROL_MATLAB_MCP_EXECUTABLE=[IO.Path]::GetFullPath($MatlabMcpExecutable)
}
$taskEnvironment.CONTROL_HARNESS_SESSION_ID = $taskEnvironment.CONTROL_MATLAB_DEDICATED_SESSION_ID
$previous = @{}
foreach ($key in $taskEnvironment.Keys) { $previous[$key]=[Environment]::GetEnvironmentVariable($key,'Process'); [Environment]::SetEnvironmentVariable($key,$taskEnvironment[$key],'Process') }
try {
  $bootstrapDirectory = (Join-Path $PSScriptRoot 'matlab').Replace("'","''")
  $matlabCommand = "addpath('$bootstrapDirectory'); bootstrap_control_matlab_session"
  # Start-Process preserves one quoted MATLAB -r expression; none contains secrets.
  $arguments = @('-nosplash','-nodesktop','-logfile',('"'+(Join-Path $sessionRoot 'matlab.log')+'"'),'-r',('"'+$matlabCommand+'"'))
  $process = Start-Process -FilePath $MatlabExecutable -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
} finally {
  foreach ($key in $taskEnvironment.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
$record = [ordered]@{kind='control-harness-matlab-launch-v1';pid=$process.Id;session_id=$taskEnvironment.CONTROL_MATLAB_DEDICATED_SESSION_ID;project_root=$projectRoot;private_appdata=$privateAppData;started_at_utc=[DateTime]::UtcNow.ToString('o');matlab_executable=[IO.Path]::GetFullPath($MatlabExecutable);log=(Join-Path $sessionRoot 'matlab.log');status='starting'}
$record|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $sessionRoot 'launch.json') -Encoding UTF8
Write-Output ('Dedicated MATLAB launch recorded. PID='+$process.Id+'; wait for appdata/control-harness-session.json before enabling the Web MATLAB overlay.')
