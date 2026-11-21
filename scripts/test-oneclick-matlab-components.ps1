# 离线回归。只在随机测试目录模拟 MATLAB 安装与补装，不执行 MATLAB 或网络安装。
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-matlab-components.ps1')
$fixtureParent = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\test-output'))
$fixture = Join-Path $fixtureParent ('component-discovery-' + [guid]::NewGuid())
[void][IO.Directory]::CreateDirectory($fixture)
function Assert-Test([bool]$Ok, [string]$Message) { if (-not $Ok) { throw $Message } }
function New-TestFile([string]$Name) {
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Name))
    [IO.File]::WriteAllText($Name, 'fixture')
}
$names = @('USERPROFILE','APPDATA','LOCALAPPDATA','CONTROL_MATLAB_MCP_EXECUTABLE','CONTROL_SIMULINK_TOOLKIT_ROOT','CONTROL_MATLAB_MCP_TOOLBOX_ROOT')
$previous = @{}
foreach ($name in $names) { $previous[$name]=[Environment]::GetEnvironmentVariable($name,'Process') }
try {
    $env:USERPROFILE = Join-Path $fixture 'profile'
    $env:APPDATA = Join-Path $fixture 'roaming'
    $env:LOCALAPPDATA = Join-Path $fixture 'local'
    foreach ($name in $names[3..5]) { [Environment]::SetEnvironmentVariable($name,$null,'Process') }
    $matlabRoot = Join-Path $fixture 'MATLAB 2024b'
    $matlab = Join-Path $matlabRoot 'bin\matlab.exe'
    New-TestFile $matlab
    [void][IO.Directory]::CreateDirectory((Join-Path $matlabRoot 'toolbox\simulink'))
    $versions=Join-Path $matlabRoot 'VersionInfo.xml'
    [IO.File]::WriteAllText($versions,'<MathWorks_version_info><release>R2022b</release></MathWorks_version_info>')
    $denied=$false
    try { Resolve-ControlMatlabComponents -MatlabExecutable $matlab -CheckOnly } catch { $denied=$_.Exception.Message -match 'R2023a' }
    Assert-Test $denied 'unsupported installation selected'
    [IO.File]::WriteAllText($versions,'<MathWorks_version_info><release>R2024b</release></MathWorks_version_info>')
    $result=Resolve-ControlMatlabComponents -MatlabExecutable $matlab -CheckOnly
    Assert-Test ($result.Missing.Count -eq 3) 'missing components were not detected'
    Assert-Test (-not (Test-Path -LiteralPath $env:LOCALAPPDATA)) 'check created a cache'
    # 使用替身测试完整散列检验与原子发布逻辑，包括断网、错误文件、已有缓存修复。
    function Invoke-WebRequest { param($Uri,$OutFile,[switch]$UseBasicParsing,$TimeoutSec,$ErrorAction)
        if ($Uri -eq 'offline') { throw 'network offline' }
        [IO.File]::WriteAllText($OutFile,'fixture')
    }
    $cache = Join-Path $fixture 'downloads'
    $digest=(Get-FileHash -LiteralPath $matlab).Hash
    foreach ($case in @('wrong-hash','offline')) {
        $rejected=$false
        try { Get-ControlMatlabAsset $cache 'candidate.bin' $case ('0' * 64) } catch { $rejected=$true }
        Assert-Test ($rejected -and -not (Get-ChildItem -LiteralPath $cache -File)) 'partial download published'
    }
    $cached=Get-ControlMatlabAsset $cache 'candidate.bin' 'ok' $digest
    [IO.File]::WriteAllText($cached,'corrupt')
    $cached=Get-ControlMatlabAsset $cache 'candidate.bin' 'ok' $digest
    Assert-Test ((Get-FileHash -LiteralPath $cached).Hash -eq $digest) 'damaged cache not repaired'
    $cached=Get-ControlMatlabAsset $cache 'candidate.bin' 'offline' $digest
    # 透明模拟自定义目录已经注册了 MCP Toolbox，只有额外的 SATK 需要安装。
    $server=Join-Path $fixture 'custom server.exe'
    $fakeToolbox=Join-Path $fixture 'custom addons\MCP'
    $fakeToolkit=Join-Path $fixture 'custom addons\SATK'
    New-TestFile $server
    New-TestFile (Join-Path $fakeToolbox 'shareMATLABSession.m')
    $script:requests=@()
    function Get-ControlMatlabAsset { param($Directory,$Filename,$Uri,$Digest)
        $script:requests+=$Filename
        return (Join-Path $fixture $Filename)
    }
    function Invoke-ControlMatlabAddonSetup { param($MatlabExecutable,$McpPackage,$SimulinkPackage)
        if ($McpPackage) {throw 'existing MCP Toolbox should not be reinstalled'}
        if ($SimulinkPackage) {
            New-TestFile (Join-Path $fakeToolkit 'tools\tools.json')
            New-TestFile (Join-Path $fakeToolkit 'tools\model_edit\model_edit.p')
        }
        return [pscustomobject]@{mcpRoot=$fakeToolbox;toolkitRoot=$fakeToolkit}
    }
    $result=Resolve-ControlMatlabComponents -MatlabExecutable $matlab -MatlabMcpExecutable $server
    Assert-Test ($result.Missing.Count -eq 0 -and $result.SimulinkToolkitRoot -eq $fakeToolkit -and $result.MatlabMcpToolboxRoot -eq $fakeToolbox) 'post-install discovery failed'
    Assert-Test ($script:requests.Count -eq 1 -and $script:requests[0] -eq 'simulink-agentic-toolkit.mltbx') 'unexpected component updated'
    $result=Resolve-ControlMatlabComponents -MatlabExecutable $matlab -MatlabMcpExecutable $server -SimulinkToolkitRoot $fakeToolkit -MatlabMcpToolboxRoot $fakeToolbox
    Assert-Test ($script:requests.Count -eq 1 -and $result.Missing.Count -eq 0) 'installed layout was downloaded twice'
    # 三项全部缺失时检查下载目标及传给 MATLAB 的包，防止布尔分支只处理第一项。
    $script:requests=@()
    function Get-ControlMatlabAsset { param($Directory,$Filename,$Uri,$Digest)
        Assert-Test ($Uri.StartsWith('https://github.com/matlab/') -and $Digest -match '^[a-f0-9]{64}$') 'unverified package requested'
        $script:requests+=$Filename
        $candidate=Join-Path $Directory $Filename
        New-TestFile $candidate
        return $candidate
    }
    function Invoke-ControlMatlabAddonSetup { param($MatlabExecutable,$McpPackage,$SimulinkPackage)
        if ($McpPackage -or $SimulinkPackage) {
            Assert-Test ((Split-Path -Leaf $McpPackage) -eq 'MATLABMCPServerToolbox.mltbx') 'missing MCP Toolbox package'
            Assert-Test ((Split-Path -Leaf $SimulinkPackage) -eq 'simulink-agentic-toolkit.mltbx') 'missing Simulink package'
            return [pscustomobject]@{mcpRoot=$fakeToolbox;toolkitRoot=$fakeToolkit}
        }
        return [pscustomobject]@{mcpRoot='';toolkitRoot=''}
    }
    $result=Resolve-ControlMatlabComponents -MatlabExecutable $matlab
    Assert-Test ($script:requests.Count -eq 3 -and $result.Missing.Count -eq 0 -and (Test-Path -LiteralPath $result.MatlabMcpExecutable)) 'incomplete fresh installation'
    Write-Host 'PASS: MATLAB discovery, read-only checks, download failures, SHA-256, cache repair, fresh/selective installation and reuse.'
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
    $target=[IO.Path]::GetFullPath($fixture)
    if ((Split-Path -Parent $target) -eq $fixtureParent -and (Split-Path -Leaf $target) -match '^component-discovery-[a-f0-9-]{36}$') { Remove-Item -LiteralPath $target -Recurse -Force }
}
