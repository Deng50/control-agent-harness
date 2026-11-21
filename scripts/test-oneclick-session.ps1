# 用真实资源状态读取协议和模拟的 MATLAB 进程信息，验证重启、版本冲突与隔离边界。
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-session.ps1')
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$fixture = Join-Path $project ('test-output\oneclick-session-check-' + [guid]::NewGuid())
$sessions = Join-Path $fixture 'matlab'
$resource = Join-Path $sessions 'shared-resource'
$directory = Join-Path $sessions 'oneclick-session-example'
$appData = Join-Path $directory 'appdata'
[void][IO.Directory]::CreateDirectory($appData)
[void][IO.Directory]::CreateDirectory($resource)
$started = (Get-Process -Id $PID).StartTime
$sessionId = [guid]::NewGuid().ToString()
$components = @{ MatlabExecutable='C:\fixture\bin\matlab.exe'; SimulinkToolkitRoot='C:\fixture\toolkit' }
$launch = @{kind='control-harness-matlab-launch-v1';session_id=$sessionId;project_root=$project;private_appdata=$appData;matlab_executable=$components.MatlabExecutable}
$marker = @{kind='control-harness-dedicated-matlab-v1';status='ready';session_id=$sessionId;project_root=$project;private_appdata=$appData;matlab_pid=$PID;toolkit_root=$components.SimulinkToolkitRoot;createdAt=[datetime]::UtcNow.ToString('o')}
$binding = @{host=[Environment]::MachineName;pid=$PID;session_id=$sessionId;identity=('win32:{0:x16}' -f $started.ToUniversalTime().ToFileTimeUtc())}
$state = @{schema_version=1;directory=$resource;next_ticket=0;queue=@();owner=$null;quarantine=$null;session=$binding}
function Write-TestJson($path,$data) {
    [IO.File]::WriteAllText($path,($data | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
Write-TestJson (Join-Path $directory 'launch.json') $launch
Write-TestJson (Join-Path $appData 'control-harness-session.json') $marker
$stateFile = Join-Path $resource 'resource.json'
Write-TestJson $stateFile $state
function Get-ControlLiveProcess([int]$ProcessId) {
    if ($ProcessId -eq $PID) { return [pscustomobject]@{ProcessName='MATLAB';StartTime=$started} }
}
$request = @{ProjectRoot=$project;SessionRoot=$sessions;ResourceDirectory=$resource;Components=$components}
if ((Get-ControlSessionAppData @request) -ne $appData) { throw 'Dedicated session was not reused' }
$components.SimulinkToolkitRoot = 'C:\fixture\other'
try { Get-ControlSessionAppData @request; throw 'Version mismatch was accepted' }
catch { if ($_.Exception.Message -notmatch '其他 MATLAB 或 Toolkit') { throw } }
$components.SimulinkToolkitRoot = $marker.toolkit_root
# 磁盘标记属于其他会话时，不能接管该进程或新建一个竞争的 MATLAB。
$marker.session_id = [guid]::NewGuid().ToString()
Write-TestJson (Join-Path $appData 'control-harness-session.json') $marker
try { Get-ControlSessionAppData @request; throw 'Foreign session was accepted' }
catch { if ($_.Exception.Message -notmatch '未找到匹配的启动记录') { throw } }
$state.quarantine = @{reason='fixture'}
Write-TestJson $stateFile $state
try { Get-ControlSessionAppData @request; throw 'Quarantined resource was accepted' }
catch { if ($_.Exception.Message -notmatch '已隔离') { throw } }
Write-Output 'PASS: bound session reuse, mismatched Toolkit, foreign identity, quarantine'
