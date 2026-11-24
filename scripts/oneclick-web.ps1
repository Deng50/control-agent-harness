# Windows PowerShell 5.1 / PowerShell 7. Only local, same-user Harness processes qualify.
function Test-ControlWebReady([string]$Url) {
    $request = [Net.HttpWebRequest]::Create($Url)
    $request.AllowAutoRedirect = $false
    $request.Proxy = $null
    $request.Timeout = 2000
    $request.ReadWriteTimeout = 2000
    $response = $null
    try {
        try { $response = $request.GetResponse() }
        catch [Net.WebException] { $response = $_.Exception.Response }
        if (-not $response -or [int]$response.StatusCode -ne 401) { return $false }
        $reader = [IO.StreamReader]::new($response.GetResponseStream())
        try { return $reader.ReadLine() -ceq 'dsh web authentication required; reopen the URL printed by dsh web.' }
        finally { $reader.Dispose() }
    } catch [IO.IOException] { return $false }
    finally { if ($response) { $response.Dispose() } }
}

function Get-ControlExistingWeb([string]$ProjectRoot, [int]$Port, [switch]$FindOtherPort) {
    $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction Stop)
    $requested = @($listeners | Where-Object LocalPort -eq $Port)
    if (-not $requested.Count -and -not $FindOtherPort) { return }

    # Match the actual argv shape built by start-web.ps1, including paths with spaces.
    $cli = [regex]::Escape((Join-Path $ProjectRoot 'vendor\deepseek-harness\apps\cli\lib\bin.js'))
    $overlay = [regex]::Escape((Join-Path $ProjectRoot 'config\web.overlay.yml'))
    $pattern = '^\s*(?:"[^"]+"|[^\s"]+)\s+"?'+$cli+'"?\s+web\s+--patch\s+"?'+$overlay+'"?(?:\s+--patch\s+(?:"[^"]+"|[^\s"]+))*\s+--port\s+(?<port>\d{1,5})(?:\s+--no-open)?\s*$'
    $userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    $fallback = $null
    foreach ($process in @(Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction Stop)) {
        if (-not $process.CommandLine -or $process.CommandLine.Replace('/','\') -notmatch $pattern) { continue }
        $candidate = [int]$Matches.port
        if ($candidate -ne $Port) {
            if (-not $FindOtherPort) { continue }
            # Dedicated test Hosts can use other ports in the same checkout. Only
            # the one-click launcher can hold this checkout's startup mutex.
            $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ParentProcessId)" -ErrorAction Stop
            $launcher = [regex]::Escape((Join-Path $ProjectRoot 'scripts\start-control-harness.ps1'))
            if (-not $parent -or -not $parent.CommandLine -or $parent.Name -notin @('powershell.exe','pwsh.exe') -or
                $parent.CommandLine.Replace('/','\') -notmatch ('\s-File\s+"?'+$launcher+'"?(?:\s|$)')) { continue }
        }
        $socket = @($listeners | Where-Object {
            $_.OwningProcess -eq $process.ProcessId -and $_.LocalPort -eq $candidate -and $_.LocalAddress -eq '127.0.0.1'
        })
        if (-not $socket.Count) { continue }
        try { $owner = Invoke-CimMethod -InputObject $process -MethodName GetOwnerSid -ErrorAction Stop }
        catch { continue }
        if ($owner.ReturnValue -ne 0 -or $owner.Sid -ne $userSid) { continue }
        $url = "http://127.0.0.1:$candidate/"
        $state = if (Test-ControlWebReady $url) { 'Ready' } else { 'Starting' }
        $info = [pscustomobject]@{State=$state; Port=$candidate; Url=$url; ProcessId=$process.ProcessId}
        if ($candidate -eq $Port) { return $info }
        $fallback = $info
    }
    if ($fallback) { return $fallback }
    if ($requested.Count) {
        return [pscustomobject]@{State='Occupied'; Port=$Port; Url=$null; ProcessId=$requested[0].OwningProcess}
    }
}

# Only explicit fatal Windows exceptions qualify. An ordinary nonzero exit,
# Ctrl+C, deliberate stop or failed startup has no automatic crash policy.
function Test-ControlHostCrashRecovery([int]$ExitCode, [string]$ReadyFile, [string]$LaunchId,
                                      [string]$ProjectRoot, [string]$ResourceDirectory, [int]$Port) {
    $fatal = @('c0000005','c000001d','c0000094','c00000fd','c0000374','c0000409')
    if (('{0:x8}' -f $ExitCode) -notin $fatal -or -not (Test-Path -LiteralPath $ReadyFile -PathType Leaf)) { return $false }
    $ready = Get-Content -LiteralPath $ReadyFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($ready.launch_id -cne $LaunchId -or $ready.project_root -ne $ProjectRoot -or $ready.port -ne $Port -or
        $ready.resource_directory -ne $ResourceDirectory -or $ready.web_pid -le 0 -or (Get-ControlLiveProcess $ready.web_pid) -or
        $ready.session.host -ne [Environment]::MachineName -or $ready.session.pid -le 0 -or
        $ready.session.identity -notmatch '^win32:(?!0{16}$)[a-f0-9]{16}$' -or
        $ready.session.session_id -notmatch '^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$') { return $false }
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    $previousPath = $env:PYTHONPATH
    $observedDispatch = $false
    $wait = [Diagnostics.Stopwatch]::StartNew()
    try {
        $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
        do {
            $state = Read-ControlStartupResource $python $ResourceDirectory
            $binding = $state.session
            if (-not $binding -or $binding.pid -ne $ready.session.pid -or $binding.host -cne $ready.session.host -or
                $binding.identity -cne $ready.session.identity -or $binding.session_id -cne $ready.session.session_id) { return $false }
            $previous = if ($state.quarantine) { $state.quarantine.owner } else { $state.owner }
            if ($previous -and $previous.phase -eq 'running' -and $previous.session.pid -eq $binding.pid -and
                $previous.session.host -ceq $binding.host -and $previous.session.identity -ceq $binding.identity -and
                $previous.session.session_id -ceq $binding.session_id) { $observedDispatch = $true }
            if (-not $observedDispatch) { return $false }
            if (-not $state.owner) { return $true }
            Start-Sleep -Milliseconds 200
        } while ($wait.Elapsed.TotalSeconds -lt 8)
        return $false
    } finally { $env:PYTHONPATH = $previousPath }
}

function Request-ControlWebRebind($Web, [string]$ProjectRoot, [string]$HarnessHome, [switch]$NoOpen) {
    $directory = Join-Path $ProjectRoot '.dsh\oneclick'
    $marker = Join-Path $directory "web-$($Web.ProcessId).json"
    $verified = $null
    if (Test-Path -LiteralPath $marker -PathType Leaf) {
        # The marker is published only after the Web process completes its
        # startup handshake. An older file belongs to a previous PID owner.
        $live = Get-Process -Id $Web.ProcessId -ErrorAction Stop
        if ($live.StartTime.ToUniversalTime() -gt [IO.File]::GetLastWriteTimeUtc($marker)) {
            throw '磁盘上的网页启动记录属于已退出的进程，请核对正在运行的 Host。'
        }
        $verified = Get-Content -Raw -Encoding UTF8 -LiteralPath $marker | ConvertFrom-Json
        if ($verified.web_pid -ne $Web.ProcessId -or $verified.project_root -ne $ProjectRoot -or
            $verified.session_id -notmatch '^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$') { throw '网页启动记录与当前进程不符。' }
        $resource = $verified.resource_directory
    } else {
        $resource = Join-Path $HarnessHome 'matlab\shared-resource'
    }
    $stateFile = Join-Path $resource 'resource.json'
    if (-not (Test-Path -LiteralPath $stateFile -PathType Leaf)) {
        if ($verified) { throw '当前网页缺少已绑定的 MATLAB 资源记录。' }
        return $false
    }
    $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
        $state = Read-ControlStartupResource $python $resource
    } finally { $env:PYTHONPATH = $previousPythonPath }
    if (-not $state.session -or $null -eq $state.session_alive) { throw '网页正在运行，但无法核实专用 MATLAB 的进程身份。' }
    if ($verified -and ($verified.matlab_pid -ne $state.session.pid -or $verified.session_id -ne $state.session.session_id)) {
        throw '网页启动记录与 MATLAB 资源绑定不一致。'
    }
    if ($state.session_alive) {
        if ($state.quarantine) {
            Write-Host "网页仍可使用。专用 MATLAB（PID $($state.session.pid)）的上一次调用待核验，可在网页查看资源状态。"
        }
        return $false
    }
    if (-not $verified) { throw '原 MATLAB 已退出。此网页使用早期启动器，请在当前网页任务结束后关闭原启动窗口，再运行一次新版启动器。' }
    $request = Join-Path $directory "web-$($Web.ProcessId)-$($verified.session_id).rebind"
    $mode = if ($NoOpen) { 'no-open' } else { 'open' }
    [IO.File]::WriteAllText($request,$mode,[Text.Encoding]::ASCII)
    Write-Host "已向本项目网页提交重新绑定请求。Host 将等待对话和作业空闲，核实原 MATLAB（PID $($state.session.pid)）退出后恢复：$($Web.Url)"
    return $true
}

function Show-ControlExistingWeb($Web, [switch]$NoOpen, [string]$ProjectRoot, [string]$HarnessHome) {
    if ($Web.State -eq 'Occupied') { throw "端口 $($Web.Port) 已被其他服务占用（PID $($Web.ProcessId)）。请使用 -Port 指定空闲端口。" }
    if ($Web.State -eq 'Starting') {
        Write-Host "本项目网页正在启动（端口 $($Web.Port)）。请稍候，或在原启动窗口查看进度。"
        return
    }
    if ($Web.State -ne 'Ready') { throw '无法确认现有网页服务的状态。' }
    if ($ProjectRoot -and (Request-ControlWebRebind -Web $Web -ProjectRoot $ProjectRoot -HarnessHome $HarnessHome -NoOpen:$NoOpen)) { return }
    Write-Host "项目网页已运行：$($Web.Url)"
    if (-not $NoOpen) { Start-Process -FilePath $Web.Url -ErrorAction Stop }
}
