# 只复用本项目留有公开启动记录的专用 MATLAB。连接描述符的检查由 start-web 完成。
function Get-ControlLiveProcess([int]$ProcessId) {
    try { return Get-Process -Id $ProcessId -ErrorAction Stop }
    catch {
        if ($_.FullyQualifiedErrorId -like 'NoProcessFoundForGivenId*') { return $null }
        throw
    }
}

function Read-ControlStartupResource([string]$Python, [string]$ResourceDirectory) {
    $snapshot = & $Python -m control_harness.matlab_resource inspect --directory $ResourceDirectory
    if ($LASTEXITCODE -ne 0) { throw "无法检查 MATLAB 资源状态，请检查 $ResourceDirectory。" }
    return ($snapshot | ConvertFrom-Json)
}

function Get-ControlSessionAppData {
    param([string]$ProjectRoot, [string]$SessionRoot, [string]$ResourceDirectory,
          [hashtable]$Components, [int]$TimeoutSeconds = 180)

    $boundSession = $null
    $retiredSessionId = $null
    $resourceState = Join-Path $ResourceDirectory 'resource.json'
    if (Test-Path -LiteralPath $resourceState) {
        $python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
        $previousPythonPath = $env:PYTHONPATH
        try {
            $env:PYTHONPATH = Join-Path $ProjectRoot 'src'
            $state = Read-ControlStartupResource $python $ResourceDirectory
            if ($state.quarantine) {
                if ($state.owner) { throw '专用 MATLAB 仍有占用者，请等待现有任务结束后重试。' }
                $ticket = $state.quarantine.token
                $binding = $state.quarantine.owner.session
                if ($null -eq $binding) { $binding = $state.session }
                if ($ticket -notmatch '^[0-9a-f]{32}$' -or [string]::IsNullOrWhiteSpace($binding.session_id) -or
                    -not $binding.host -or ($binding.pid -isnot [long] -and $binding.pid -isnot [int]) -or
                    $binding.pid -le 0 -or $binding.pid -gt [int]::MaxValue -or
                    $binding.identity -notmatch '^win32:[a-f0-9]{16}$') {
                    throw '专用 MATLAB 正在等待核验，旧会话的进程绑定或调用记录不完整。请按 docs/matlab-existing-session.md 核对资源记录。'
                }
                if ($binding.host -ne [Environment]::MachineName) { throw '专用 MATLAB 的调用记录属于另一台电脑，请在原电脑核实进程与调用状态。' }
                if ($state.session -and ($state.session.host -cne $binding.host -or
                    $state.session.pid -ne $binding.pid -or $state.session.identity -cne $binding.identity -or
                    $state.session.session_id -cne $binding.session_id)) {
                    throw '专用 MATLAB 的两份会话绑定记录不一致。请先核对原会话的归属，再恢复派发。'
                }
                # 恢复权限始终由 broker 决定。它在锁内复核本机进程创建标识、隔离票据及执行互斥锁并归档证据。
                # PS 5.1 会将原生命令 stderr 包装成 ErrorRecord；此处只使用退出码，再读当前状态决定下一步。
                if ($state.session_alive -eq $true) {
                    # 仅复用相同绑定启动 Host；普通 MATLAB 请求继续由 broker 拦住，直到在线核验通过。
                    Write-Host "专用 MATLAB（PID $($binding.pid)）仍在运行。网页将核验上次调用的终态。"
                } else {
                    $recoveryExit = & {
                        $ErrorActionPreference = 'Continue'
                        $PSNativeCommandUseErrorActionPreference = $false
                        & $python -m control_harness.matlab_resource recover-after-session-exit --directory $ResourceDirectory --token $ticket 2>$null | Out-Null
                        return $LASTEXITCODE
                    }
                    $state = Read-ControlStartupResource $python $ResourceDirectory
                    if ($state.quarantine) {
                        if ($state.quarantine.token -ne $ticket) { throw '专用 MATLAB 的待核验记录已更新，请重新运行启动入口以核对新记录。' }
                        throw "系统尚不能确认 PID $($binding.pid) 对应的原进程已退出且执行锁已释放。请用原启动账户核实后重试，详见 docs/matlab-existing-session.md。"
                    }
                    if ($recoveryExit -ne 0) {
                        throw '恢复期间资源记录已发生变化。请重新运行一键启动，以当前绑定重新核验。'
                    }
                    $retiredSessionId = $binding.session_id
                    Write-Host "已核实原专用 MATLAB（PID $($binding.pid)）已退出，恢复证据已归档。"
                }
            }
        } finally { $env:PYTHONPATH = $previousPythonPath }
        if ($state.owner -or ($state.queue -and -not $state.quarantine)) { throw '专用 MATLAB 正在处理任务。请使用当前网页，或在队列清空后重试。' }
        if ($state.session) {
            if ($state.session.host -ne [Environment]::MachineName) { throw '资源记录属于另一台电脑。请使用本机的 Harness 数据目录。' }
            $process = Get-ControlLiveProcess $state.session.pid
            if ($process) {
                $creation = 'win32:{0:x16}' -f $process.StartTime.ToUniversalTime().ToFileTimeUtc()
                if ($creation -eq $state.session.identity) { $boundSession = $state.session }
            }
        }
    }

    # 未绑定时只找本入口创建的会话。手动启动的 MATLAB 必须先有匹配的 broker 绑定。
    $chosen = $null
    $waiting = $null
    if (Test-Path -LiteralPath $SessionRoot) {
        foreach ($directory in (Get-ChildItem -LiteralPath $SessionRoot -Directory | Sort-Object LastWriteTimeUtc -Descending)) {
            if (-not $boundSession -and $directory.Name -notlike 'oneclick-session-*') { continue }
            $launchFile = Join-Path $directory.FullName 'launch.json'
            if (-not (Test-Path -LiteralPath $launchFile -PathType Leaf)) { continue }
            $launch = Get-Content -LiteralPath $launchFile -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($retiredSessionId -and $launch.session_id -eq $retiredSessionId) { continue }
            if ($boundSession -and $launch.session_id -ne $boundSession.session_id) { continue }
            $appData = Join-Path $directory.FullName 'appdata'
            if ($launch.kind -ne 'control-harness-matlab-launch-v1' -or
                $launch.project_root -ne $ProjectRoot -or $launch.private_appdata -ne $appData) { continue }
            $markerFile = Join-Path $appData 'control-harness-session.json'
            if (Test-Path -LiteralPath $markerFile -PathType Leaf) {
                $marker = Get-Content -LiteralPath $markerFile -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($marker.session_id -ne $launch.session_id -or $marker.status -ne 'ready' -or
                    $marker.private_appdata -ne $appData -or $marker.project_root -ne $ProjectRoot) { continue }
                $process = Get-ControlLiveProcess $marker.matlab_pid
                if (-not $process -or $process.ProcessName -ne 'MATLAB' -or
                    $process.StartTime.ToUniversalTime() -gt ([datetimeoffset]$marker.createdAt).UtcDateTime.AddSeconds(1)) { continue }
                if ($boundSession -and $marker.matlab_pid -ne $boundSession.pid) { continue }
                if ($launch.matlab_executable -ne $Components.MatlabExecutable -or $marker.toolkit_root -ne $Components.SimulinkToolkitRoot -or
                    ($marker.mcp_executable -and $marker.mcp_executable -ne $Components.MatlabMcpExecutable) -or
                    ($marker.toolbox_root -and $marker.toolbox_root -ne $Components.MatlabMcpToolboxRoot)) {
                    throw "已有专用会话使用其他 MATLAB 或 Toolkit 版本。请先在该版本完成工作并退出，然后重试。目录：$appData"
                }
                $chosen = $directory.FullName
                break
            }
            # Windows -wait 使本入口的启动器在实际 MATLAB 退出前持续存在。
            $launcher = Get-ControlLiveProcess $launch.pid
            if ($launcher -and $launcher.Path -eq $launch.matlab_executable -and
                [Math]::Abs(($launcher.StartTime.ToUniversalTime() - ([datetimeoffset]$launch.started_at_utc).UtcDateTime).TotalSeconds) -lt 15) {
                $waiting = $directory.FullName
            }
        }
    }
    if ($chosen) {
        Write-Host "复用专用 MATLAB 会话：$chosen"
        return (Join-Path $chosen 'appdata')
    }
    if ($boundSession) { throw "资源已绑定至运行中的专用 MATLAB，未找到匹配的启动记录。请沿用该会话的手动连接参数。PID $($boundSession.pid)" }
    if ($waiting) {
        $chosen = $waiting
        Write-Host "继续等待已启动的 MATLAB：$chosen"
    } else {
        $chosen = Join-Path $SessionRoot ('oneclick-session-' + [Guid]::NewGuid().ToString('N'))
        & (Join-Path $PSScriptRoot 'start-matlab-session.ps1') @Components -SessionDirectory $chosen | Out-Host
    }
    $appData = Join-Path $chosen 'appdata'
    $launch = Get-Content -LiteralPath (Join-Path $chosen 'launch.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $elapsed = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath (Join-Path $appData 'control-harness-session.json') -PathType Leaf)) {
        if (-not (Get-ControlLiveProcess $launch.pid)) { throw "专用 MATLAB 在初始化完成前退出。请检查许可证及启动日志：$chosen\matlab.log" }
        if ($elapsed.Elapsed.TotalSeconds -ge $TimeoutSeconds) { throw "MATLAB 初始化仍未完成。请检查许可证及日志后重新运行入口：$chosen\matlab.log" }
        Start-Sleep -Milliseconds 500
    }
    return $appData
}
