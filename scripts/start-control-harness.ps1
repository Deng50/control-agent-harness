<#
在 Windows 上准备当前检出目录并启动启用 MATLAB 的网页。
仅检查时不联网、不安装。可在多版本机器上用 -MatlabExecutable 指定版本。
#>
[CmdletBinding()]
param(
    [ValidateRange(1,65535)][int]$Port = 5288,
    [switch]$NoOpen,
    [switch]$CheckOnly,
    [string]$MatlabExecutable,
    [string]$MatlabMcpExecutable,
    [string]$SimulinkToolkitRoot,
    [string]$MatlabMcpToolboxRoot,
    [string]$Workspace,
    [string]$HarnessHome
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-matlab-components.ps1')
. (Join-Path $PSScriptRoot 'oneclick-project-runtime.ps1')
. (Join-Path $PSScriptRoot 'oneclick-session.ps1')
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $HarnessHome) { $HarnessHome = Join-Path $projectRoot '.dsh' }
if (-not $Workspace) { $Workspace = Join-Path $projectRoot 'workspaces\project-01' }
$Workspace = Get-ControlLocalWorkspacePath $Workspace
$HarnessHome = Get-ControlLocalWorkspacePath $HarnessHome
$componentPaths = @{}
foreach ($name in @('MatlabExecutable','MatlabMcpExecutable','SimulinkToolkitRoot','MatlabMcpToolboxRoot')) {
    if ($PSBoundParameters.ContainsKey($name)) { $componentPaths[$name] = $PSBoundParameters[$name] }
}
if ($CheckOnly) {
    $components = Resolve-ControlMatlabComponents @componentPaths -CheckOnly
    $runtime = Initialize-ControlProjectRuntime -ProjectRoot $projectRoot -HarnessHome $HarnessHome -CheckOnly
    [ordered]@{
        matlab = $components.MatlabExecutable
        missing_matlab_components = @($components.Missing)
        missing_project_dependencies = @($runtime.Missing)
        ready_to_launch = ($runtime.Ready -and -not $components.Missing)
        connectivity = 'not_tested'
    } | ConvertTo-Json
    return
}

# 一个检出目录共享构建文件和环境。核验与网页运行期间重复双击不启动第二套服务。
$hasher = [Security.Cryptography.SHA256]::Create()
try { $hash = $hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant())) }
finally { $hasher.Dispose() }
$lockName = 'Local\ControlHarnessStartup-' + [BitConverter]::ToString($hash).Replace('-','')
$mutex = [Threading.Mutex]::new($false,$lockName)
$acquired = $false
try {
    try { $acquired = $mutex.WaitOne(0) } catch [Threading.AbandonedMutexException] { $acquired = $true }
    if (-not $acquired) { throw '这个项目的启动窗口已经打开。请回到原窗口查看安装进度或使用其中的网页地址。' }
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,$Port)
    try {
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
    } catch { throw "无法监听本机端口 $Port。若网页已启动，请回到原窗口；若端口被其他程序占用，可使用 -Port 指定空闲端口。" }
    finally { $listener.Stop() }

    Write-Host '[1/4] 正在定位本机 MATLAB/Simulink 并补齐缺少的连接组件…'
    $components = Resolve-ControlMatlabComponents @componentPaths
    Write-Host '[2/4] 正在准备 Python、Node.js 和项目依赖。首次构建可能需要几分钟…'
    $runtime = Initialize-ControlProjectRuntime -ProjectRoot $projectRoot -HarnessHome $HarnessHome
    if (-not $runtime.Ready) { throw ('项目运行环境尚未准备好：' + ($runtime.Missing -join ', ')) }

    # 只在项目专用叶目录调用既有 ACL 准备协议。外部工作区只检查写入条件。
    $prepare = $Workspace.StartsWith((Join-Path $projectRoot 'workspaces\'),[StringComparison]::OrdinalIgnoreCase) -or
               $Workspace.StartsWith((Join-Path $projectRoot 'test-output\'),[StringComparison]::OrdinalIgnoreCase)
    & (Join-Path $PSScriptRoot 'prepare-workspace.ps1') -Workspace $Workspace -Prepare:$prepare | Out-Null
    Write-Host "[3/4] 工作区已准备：$Workspace。正在启动或复用专用 MATLAB…"
    $launchPaths = @{}
    foreach ($name in @('MatlabExecutable','MatlabMcpExecutable','SimulinkToolkitRoot','MatlabMcpToolboxRoot')) {
        $launchPaths[$name] = $components.$name
    }
    $sessionRoot = Join-Path $HarnessHome 'matlab'
    $resource = Join-Path $sessionRoot 'shared-resource'
    $appData = Get-ControlSessionAppData -ProjectRoot $projectRoot -SessionRoot $sessionRoot -ResourceDirectory $resource -Components $launchPaths
    Write-Host '[4/4] 正在从网页服务调用 MATLAB。验证完成后即可使用网页；Ctrl+C 停止网页服务。'
    & (Join-Path $PSScriptRoot 'start-web.ps1') -Matlab -VerifyMatlab -HarnessHome $HarnessHome -Workspace $Workspace -Port $Port -NoOpen:$NoOpen `
        -MatlabSessionAppData $appData -MatlabResourceDirectory $resource `
        -MatlabMcpExecutable $components.MatlabMcpExecutable -SimulinkToolkitRoot $components.SimulinkToolkitRoot
    if ($LASTEXITCODE -ne 0) { throw "网页服务退出（代码 $LASTEXITCODE）。请查看上面的连接检查或启动错误。" }
} finally {
    if ($acquired) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
