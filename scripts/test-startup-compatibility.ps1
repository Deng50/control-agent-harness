param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$cases = @(
  @{Value='C:\MATLAB\bin\matlab.exe';Expected=$true},
  @{Value='D:/projects/test';Expected=$true},
  @{Value='\\server\share\test';Expected=$true},
  @{Value='//server/share/test';Expected=$true},
  @{Value='\\?\C:\test';Expected=$true},
  @{Value='C:\path with spaces\test';Expected=$true},
  @{Value='C:relative';Expected=$false},
  @{Value='\root-relative';Expected=$false},
  @{Value='/root-relative';Expected=$false},
  @{Value='.\relative';Expected=$false},
  @{Value='relative';Expected=$false},
  @{Value='';Expected=$false},
  @{Value=$null;Expected=$false}
)
foreach ($case in $cases) {
  if ((Test-ControlFullyQualifiedWindowsPath $case.Value) -ne $case.Expected) {
    throw ('Path contract failed: ' + $case.Value)
  }
}
# Rejection must happen before any installation lookup or MATLAB process launch.
try {
  & (Join-Path $PSScriptRoot 'start-matlab-session.ps1') -ValidateOnly `
    -MatlabExecutable 'C:relative' -MatlabMcpExecutable 'C:\missing.exe' `
    -SimulinkToolkitRoot 'C:\missing' -MatlabMcpToolboxRoot 'C:\missing' `
    -SessionDirectory 'C:\missing'
  throw 'Drive-relative MATLAB executable was accepted.'
} catch {
  if ($_.Exception.Message -ne 'All dedicated-session paths must be absolute.') { throw }
}
try {
  & (Join-Path $PSScriptRoot 'start-web.ps1') -ValidateOnly -Matlab `
    -MatlabMcpExecutable '\root-relative.exe' -SimulinkToolkitRoot 'C:\missing' `
    -MatlabSessionAppData 'C:\missing' -MatlabResourceDirectory 'C:\missing'
  throw 'Root-relative MCP executable was accepted.'
} catch {
  if ($_.Exception.Message -ne 'MatlabMcpExecutable must be an absolute path.') { throw }
}
[ordered]@{status='PASS';powershell=$PSVersionTable.PSVersion.ToString();path_cases=$cases.Count;launcher_rejection_cases=2} | ConvertTo-Json
