# MATLAB 和 Simulink 为已安装的有许可证产品。此模块按需补齐连接组件，
# 只从 MathWorks 官方固定版本下载，校验发布资产的 SHA-256 后才使用。
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
. (Join-Path $PSScriptRoot 'official-matlab-components.ps1')

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

function Resolve-ControlMatlabComponents {
    [CmdletBinding()]
    param(
        [string]$MatlabExecutable, [string]$MatlabMcpExecutable = $env:CONTROL_MATLAB_MCP_EXECUTABLE,
        [string]$SimulinkToolkitRoot = $env:CONTROL_SIMULINK_TOOLKIT_ROOT,
        [string]$MatlabMcpToolboxRoot = $env:CONTROL_MATLAB_MCP_TOOLBOX_ROOT,
        [string]$ComponentDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\.control-matlab')),
        [switch]$CheckOnly
    )
    $ErrorActionPreference = 'Stop'
    $matlab = Find-ControlMatlabExecutable $MatlabExecutable
    $defaults = @()
    $result = @{MatlabExecutable=$matlab}
    foreach ($name in @('MatlabMcpExecutable','MatlabMcpToolboxRoot','SimulinkToolkitRoot')) {
        $explicit = Get-Variable -ValueOnly -Name $name
        if ($explicit) { $result[$name] = Find-ControlMatlabComponent $name $explicit @() }
        else { $defaults += $name }
    }
    $managed = Resolve-ControlOfficialMatlabComponents $ComponentDirectory -Names $defaults -CheckOnly:$CheckOnly
    $result.Missing = @($defaults | Where-Object { -not $managed[$_] })
    foreach ($name in $defaults) { $result[$name] = $managed[$name] }
    return [pscustomobject]$result
}
