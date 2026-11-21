# MATLAB 和 Simulink 为已安装的有许可证产品。此模块按需补齐连接组件，
# 只从 MathWorks 官方固定版本下载，校验发布资产的 SHA-256 后才使用。
. (Join-Path $PSScriptRoot 'windows-paths.ps1')

function Find-ControlMatlabExecutable {
    param([string]$Explicit)
    if ($Explicit) {
        if (-not (Test-ControlFullyQualifiedWindowsPath $Explicit)) { throw '-MatlabExecutable 请使用 bin\matlab.exe 的绝对路径。' }
        $candidates = @($Explicit)
    } else {
        # PATH 可用来选择希望使用的版本，其次按注册的版本号及标准安装目录从新到旧查找。
        $candidates = @((Get-Command matlab.exe -CommandType Application -All -ErrorAction SilentlyContinue).Source)
        foreach ($hive in @('HKCU:', 'HKLM:')) {
            $keys = Get-ChildItem -LiteralPath "$hive\SOFTWARE\MathWorks\MATLAB" -ErrorAction SilentlyContinue
            foreach ($key in ($keys | Where-Object { $_.PSChildName -match '^\d+(\.\d+){1,3}$' } | Sort-Object { [version]$_.PSChildName } -Descending)) {
                $root = $key.GetValue('MATLABROOT')
                if ($root) { $candidates += Join-Path $root 'bin\matlab.exe' }
            }
        }
        $dirs = Get-ChildItem -Directory -LiteralPath (Join-Path $env:ProgramFiles 'MATLAB') -ErrorAction SilentlyContinue
        foreach ($dir in ($dirs | Sort-Object Name -Descending)) { $candidates += Join-Path $dir.FullName 'bin\matlab.exe' }
    }
    foreach ($exe in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $exe -PathType Leaf) -or (Split-Path -Leaf $exe) -ne 'matlab.exe') { continue }
        $root = Split-Path -Parent (Split-Path -Parent $exe)
        if (-not (Test-Path -LiteralPath (Join-Path $root 'toolbox\simulink') -PathType Container)) { continue }
        $info = Join-Path $root 'VersionInfo.xml'
        if (Test-Path -LiteralPath $info -PathType Leaf) {
            $version = ([xml](Get-Content -LiteralPath $info -Raw)).MathWorks_version_info.release
            if ($version -match '^R\d{4}[ab]$' -and [string]::CompareOrdinal($version, 'R2023a') -lt 0) { continue }
        }
        return (Get-Item -LiteralPath $exe).FullName
    }
    throw '找不到配有 Simulink 的 MATLAB R2023a 或更新版本。请安装产品，或用 -MatlabExecutable 指定实际 bin\matlab.exe。'
}

function Find-ControlMatlabComponent {
    param([string]$Name, [string]$Explicit, [string[]]$Locations)
    if ($Explicit) {
        if (-not (Test-ControlFullyQualifiedWindowsPath $Explicit)) { throw "$Name 请使用绝对路径。" }
        $Locations = @($Explicit)
    }
    foreach ($location in $Locations) {
        if (-not $location) { continue }
        $valid = switch ($Name) {
            MatlabMcpExecutable { Test-Path -LiteralPath $location -PathType Leaf }
            MatlabMcpToolboxRoot { Test-Path -LiteralPath (Join-Path $location 'shareMATLABSession.m') -PathType Leaf }
            SimulinkToolkitRoot {
                (Test-Path -LiteralPath (Join-Path $location 'tools\tools.json') -PathType Leaf) -and
                    ((Test-Path -LiteralPath (Join-Path $location 'tools\model_edit\model_edit.p') -PathType Leaf) -or
                     (Test-Path -LiteralPath (Join-Path $location 'tools\model_edit\model_edit.m') -PathType Leaf))
            }
        }
        if ($valid) { return (Get-Item -LiteralPath $location).FullName }
    }
    if ($Explicit) { throw "$Name 的指定路径缺少所需文件：$Explicit。请更正路径，或去掉该参数以自动查找和安装。" }
}

function Get-ControlMatlabAsset {
    param([string]$Directory, [string]$Filename, [string]$Uri, [string]$Digest)
    $dest = Join-Path $Directory $Filename
    if ((Test-Path -LiteralPath $dest -PathType Leaf) -and (Get-ControlFileSha256 $dest) -eq $Digest) { return $dest }
    [void][IO.Directory]::CreateDirectory($Directory)
    $part = Join-Path $Directory ([guid]::NewGuid().ToString() + '.part')
    try {
        Write-Host "正在下载并核验 MathWorks 组件 $Filename …"
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $part -TimeoutSec 180 -ErrorAction Stop
        if ((Get-ControlFileSha256 $part) -ne $Digest) { throw "$Filename SHA-256 与官方固定版本不符，本次下载已丢弃。" }
        # 仅在完整校验后发布。断网、半包或并发启动均不能让半成品成为可执行文件。
        if ([IO.File]::Exists($dest)) { [IO.File]::Replace($part, $dest, [System.Management.Automation.Language.NullString]::Value) }
        else {
            try { [IO.File]::Move($part, $dest) }
            catch [IO.IOException] {
                if (-not ([IO.File]::Exists($dest) -and (Get-ControlFileSha256 $dest) -eq $Digest)) { throw }
            }
        }
        return $dest
    } finally {
        if ([IO.File]::Exists($part)) { [IO.File]::Delete($part) }
    }
}

function Invoke-ControlMatlabAddonSetup {
    param([string]$MatlabExecutable, [string]$McpPackage, [string]$SimulinkPackage)
    $helper = Join-Path $PSScriptRoot 'matlab'
    $receipt = Join-Path ([IO.Path]::GetTempPath()) ('control-harness-addons-' + [guid]::NewGuid() + '.json')
    # 所有传给 MATLAB 的路径均使用单引号字面值；含空格、中文或撇号的路径仍是同一参数。
    $matlabArgs = @($receipt, $McpPackage, $SimulinkPackage) | ForEach-Object { "'" + ([string]$_).Replace("'", "''") + "'" }
    $code = "addpath('" + $helper.Replace("'", "''") + "'); control_install_matlab_addons(" + ($matlabArgs -join ',') + '); exit(0);'
    try {
        & $MatlabExecutable -wait -batch $code | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not [IO.File]::Exists($receipt)) {
            throw '在所选 MATLAB 内查询或安装附加工具箱失败。请检查许可证及上面的 MATLAB 诊断后重新启动。'
        }
        return (Get-Content -LiteralPath $receipt -Encoding UTF8 -Raw | ConvertFrom-Json)
    } finally {
        if ([IO.File]::Exists($receipt)) { [IO.File]::Delete($receipt) }
    }
}

function Resolve-ControlMatlabComponents {
    [CmdletBinding()]
    param(
        [string]$MatlabExecutable, [string]$MatlabMcpExecutable = $env:CONTROL_MATLAB_MCP_EXECUTABLE,
        [string]$SimulinkToolkitRoot = $env:CONTROL_SIMULINK_TOOLKIT_ROOT,
        [string]$MatlabMcpToolboxRoot = $env:CONTROL_MATLAB_MCP_TOOLBOX_ROOT, [switch]$CheckOnly
    )
    $ErrorActionPreference = 'Stop'
    $matlab = Find-ControlMatlabExecutable $MatlabExecutable
    $legacy = Join-Path $env:USERPROFILE '.matlab\agentic-toolkits'
    $addons = Join-Path $env:APPDATA 'MathWorks\MATLAB Add-Ons\Toolboxes'
    $cache = Join-Path $env:LOCALAPPDATA 'control-agent-harness\matlab\mcp-0.16.0-satk-2026.11.a'
    $mcpRelease = 'https://github.com/matlab/matlab-mcp-server/releases/download/v0.16.0/'
    $satkRelease = 'https://github.com/matlab/simulink-agentic-toolkit/releases/download/SATK-2026.11.a/'
    $mcpFile = 'matlab-mcp-server-windows-x64.exe'
    $mcpHash = '7c70fdc69e7e7d4762732fede64f348b18ecbc89c3de705f9dde89a86d1883ea'
    $managedExe = Join-Path $cache $mcpFile
    $server = Find-ControlMatlabComponent MatlabMcpExecutable $MatlabMcpExecutable @((Join-Path $legacy 'bin\matlab-mcp-server.exe'))
    if (-not $server -and [IO.File]::Exists($managedExe) -and (Get-ControlFileSha256 $managedExe) -eq $mcpHash) { $server = $managedExe }
    $toolkit = Find-ControlMatlabComponent SimulinkToolkitRoot $SimulinkToolkitRoot @((Join-Path $legacy 'simulink'), (Join-Path $addons 'Simulink Agentic Toolkit'))
    $toolbox = Find-ControlMatlabComponent MatlabMcpToolboxRoot $MatlabMcpToolboxRoot @((Join-Path $addons 'MATLAB MCP Server Toolbox'))
    if (-not $CheckOnly) {
        if (-not $toolbox -or -not $toolkit) {
            # 自定义 Add-On 目录由目标 MATLAB 的已注册路径定位，防止重复下载或安装到错误版本。
            $actual = Invoke-ControlMatlabAddonSetup $matlab '' ''
            if (-not $toolbox) { $toolbox = Find-ControlMatlabComponent MatlabMcpToolboxRoot '' @($actual.mcpRoot) }
            if (-not $toolkit) { $toolkit = Find-ControlMatlabComponent SimulinkToolkitRoot '' @($actual.toolkitRoot) }
        }
        if (-not $server) { $server = Get-ControlMatlabAsset $cache $mcpFile ($mcpRelease + $mcpFile) $mcpHash }
        if (-not $toolbox -or -not $toolkit) {
            $mcpPackage = $null
            $simulinkPackage = $null
            if (-not $toolbox) {
                $mcpPackage = Get-ControlMatlabAsset $cache 'MATLABMCPServerToolbox.mltbx' ($mcpRelease + 'MATLABMCPServerToolbox.mltbx') '0db28ffa84cc16cfb9537c2603dca2bf2875225a5c8e11044751f1e62f8a35a8'
            }
            if (-not $toolkit) {
                $simulinkPackage = Get-ControlMatlabAsset $cache 'simulink-agentic-toolkit.mltbx' ($satkRelease + 'simulink-agentic-toolkit.mltbx') '7ffa191d33aec65922466a7518183abb3df2a03dfda7ad44d1aa774090c9888b'
            }
            $actual = Invoke-ControlMatlabAddonSetup $matlab $mcpPackage $simulinkPackage
            if (-not $toolbox) { $toolbox = Find-ControlMatlabComponent MatlabMcpToolboxRoot '' @($actual.mcpRoot) }
            if (-not $toolkit) { $toolkit = Find-ControlMatlabComponent SimulinkToolkitRoot '' @($actual.toolkitRoot) }
        }
    }
    $missing = @(
        if (-not $server) { 'MatlabMcpExecutable' }
        if (-not $toolkit) { 'SimulinkToolkitRoot' }
        if (-not $toolbox) { 'MatlabMcpToolboxRoot' }
    )
    if (-not $CheckOnly -and $missing.Count) { throw ('安装后仍未找到 ' + ($missing -join ', ') + '。请在目标 MATLAB 的 Add-On 管理器中确认已启用上述工具箱，或显式指定安装路径。') }
    return [pscustomobject]@{ MatlabExecutable=$matlab; MatlabMcpExecutable=$server; SimulinkToolkitRoot=$toolkit; MatlabMcpToolboxRoot=$toolbox; Missing=$missing }
}
