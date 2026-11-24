# 用真实资源状态读取协议和模拟的 MATLAB 进程信息，验证重启、版本冲突与隔离边界。
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-session.ps1')
. (Join-Path $PSScriptRoot 'oneclick-web.ps1')
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
$components = @{ MatlabExecutable='C:\fixture\bin\matlab.exe'; SimulinkToolkitRoot='C:\fixture\toolkit'
    MatlabMcpExecutable='C:\fixture\mcp.exe'; MatlabMcpToolboxRoot='C:\fixture\toolbox' }
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
$marker.session_id = $sessionId
Write-TestJson (Join-Path $appData 'control-harness-session.json') $marker

function Assert-QuarantinePreserved([string]$Diagnostic) {
    Write-TestJson $stateFile $state
    try { Get-ControlSessionAppData @request; throw 'Quarantined resource was accepted' }
    catch { if ($_.Exception.Message -notmatch $Diagnostic) { throw } }
    $after = Get-Content -LiteralPath $stateFile -Encoding UTF8 -Raw | ConvertFrom-Json
    if (-not $after.quarantine -or (Test-Path (Join-Path $resource 'recoveries'))) { throw 'Quarantine was cleared or recovery evidence was fabricated' }
}
$ticket = [guid]::NewGuid().ToString('N')
$state.quarantine = @{reason='fixture';token=$ticket;owner=@{session=$binding.Clone()}}
# 活实例可按原绑定进入网页核验，启动器本身不解除暂挂。
Write-TestJson $stateFile $state
if ((Get-ControlSessionAppData @request) -ne $appData) { throw 'Live pending instance was not reused for verification' }
$after = Get-Content -LiteralPath $stateFile -Encoding UTF8 -Raw | ConvertFrom-Json
if ($after.quarantine.token -ne $ticket -or $after.session.session_id -ne $sessionId) { throw 'Startup cleared pending verification' }
$runId = [guid]::NewGuid().ToString('N')
$ack = Join-Path $fixture "ready-$runId.json"
$ready = @{launch_id=$runId;project_root=$project;port=12345;resource_directory=$resource;web_pid=2147483600;session=$binding}
Write-TestJson $ack $ready
foreach ($excluded in @(0,1,75,76,130,143,-1073741510)) {
    if (Test-ControlHostCrashRecovery $excluded $ack $runId $project $resource 12345) { throw 'Normal stop, Ctrl+C or generic error triggered restart' }
}
$state.quarantine.owner.phase = 'running'
Write-TestJson $stateFile $state
if (-not (Test-ControlHostCrashRecovery -1073741819 $ack $runId $project $resource 12345)) { throw 'Matching fatal exit ignored' }
if (Test-ControlHostCrashRecovery -1073741819 $ack 'previous-launch' $project $resource 12345) { throw 'Stale startup receipt accepted' }
$state.queue = @($binding)
Write-TestJson $stateFile $state
if ((Get-ControlSessionAppData @request) -ne $appData) { throw 'A queued caller blocked verification startup' }
$state.queue = @()
$state.quarantine.owner.session.session_id = 'another-session'
Assert-QuarantinePreserved '绑定记录不一致'
$state.quarantine.owner.session.session_id = $sessionId
$state.quarantine.owner.session.host = 'other-machine'
Assert-QuarantinePreserved '原电脑'
$state.quarantine.owner.session.Remove('identity')
Assert-QuarantinePreserved '不完整'

# 临时 Python 子进程报告自己的 OS 创建标识后正常退出，用以验证真实 broker 的归档和锁释放。
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $project 'src'
    $python = Join-Path $project '.venv\Scripts\python.exe'
    $exited = & $python -c 'import os; from control_harness.matlab_resource import process_identity; print(os.getpid(), process_identity(os.getpid()))'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the exited process fixture' }
} finally { $env:PYTHONPATH = $previousPythonPath }
$childPid,$identity = $exited.Split(' ')
$oldBinding = @{host=[Environment]::MachineName;pid=[int]$childPid;session_id=[guid]::NewGuid().ToString();identity=$identity}
$state.session = $oldBinding
$state.quarantine.owner.session = $oldBinding
Write-TestJson $stateFile $state
if ((Get-ControlSessionAppData @request) -ne $appData) { throw 'Startup did not continue after verified exit' }
$after = Get-Content -LiteralPath $stateFile -Encoding UTF8 -Raw | ConvertFrom-Json
if ($after.quarantine -or $after.session -or $after.last_recovery -ne (Join-Path $resource "recoveries\$ticket.json")) { throw 'Recovery state incorrect' }
$proof = Get-Content -LiteralPath $after.last_recovery -Encoding UTF8 -Raw | ConvertFrom-Json
if ($proof.kind -ne 'os_verified_bound_session_exit' -or $proof.session.identity -ne $identity -or $proof.token -ne $ticket) { throw 'Incorrect recovery evidence' }
Write-Output 'PASS: bound session reuse, version and identity conflicts, live/queued/unknown quarantine, verified exit recovery'
