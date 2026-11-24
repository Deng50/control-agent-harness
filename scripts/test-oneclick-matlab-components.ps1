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
    try { Resolve-ControlMatlabComponents -MatlabExecutable $matlab -ComponentDirectory (Join-Path $fixture 'private') -CheckOnly } catch { $denied=$_.Exception.Message -match 'R2023a' }
    Assert-Test $denied 'unsupported installation selected'
    [IO.File]::WriteAllText($versions,'<MathWorks_version_info><release>R2024b</release></MathWorks_version_info>')
    $result=Resolve-ControlMatlabComponents -MatlabExecutable $matlab -ComponentDirectory (Join-Path $fixture 'private') -CheckOnly
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
    $server=Join-Path $fixture 'explicit server.exe'
    $toolbox=Join-Path $fixture 'explicit connector'
    $toolkit=Join-Path $fixture 'explicit toolkit'
    New-TestFile $server
    New-TestFile (Join-Path $toolbox 'shareMATLABSession.m')
    New-TestFile (Join-Path $toolkit 'tools/tools.json')
    New-TestFile (Join-Path $toolkit 'tools/model_edit/model_edit.p')
    $private=Join-Path $fixture 'private'

    # 验证独立解包、配置文件与显式覆盖，使用小型本地 mltbx 结构检验路径布局。
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $script:requests=@()
    function Get-ControlMatlabAsset { param($Directory,$Filename,$Uri,$Digest)
        Assert-Test ($Uri.StartsWith('https://github.com/matlab/') -and $Digest -match '^[a-f0-9]{64}$') 'unverified package requested'
        $script:requests += $Filename
        $candidate=Join-Path $Directory $Filename
        if (Test-Path -LiteralPath $candidate) { return $candidate }
        if ($Filename.EndsWith('.exe')) { New-TestFile $candidate; return $candidate }
        $source=Join-Path $fixture $Filename
        New-TestFile (Join-Path $source 'metadata/configuration.xml')
        if ($Filename -eq 'MATLABMCPServerToolbox.mltbx') {
            New-TestFile (Join-Path $source 'fsroot/shareMATLABSession.m')
        } else {
            New-TestFile (Join-Path $source 'fsroot/tools/tools.json')
            New-TestFile (Join-Path $source 'fsroot/tools/model_edit/model_edit.p')
        }
        [void][IO.Directory]::CreateDirectory($Directory)
        [IO.Compression.ZipFile]::CreateFromDirectory($source,$candidate)
        return $candidate
    }
    $options=@{MatlabExecutable=$matlab;ComponentDirectory=$private}
    $result=Resolve-ControlMatlabComponents @options -MatlabMcpExecutable $server -MatlabMcpToolboxRoot $toolbox
    Assert-Test ($result.Missing.Count -eq 0 -and $result.SimulinkToolkitRoot.StartsWith($private)) 'private install failed'
    Assert-Test ($script:requests.Count -eq 1 -and $script:requests[0] -eq 'simulink-agentic-toolkit.mltbx') 'unexpected component download'
    $result=Resolve-ControlMatlabComponents @options -MatlabMcpExecutable $server -MatlabMcpToolboxRoot $toolbox -SimulinkToolkitRoot $toolkit
    Assert-Test ($script:requests.Count -eq 1 -and $result.SimulinkToolkitRoot -eq $toolkit) 'explicit override not honored'
    $script:requests=@()
    $result=Resolve-ControlMatlabComponents @options
    Assert-Test ($script:requests.Count -eq 3 -and $result.Missing.Count -eq 0 -and $result.MatlabMcpExecutable.StartsWith($private)) 'incomplete fresh installation'
    Write-Host 'PASS: R2024b discovery, read-only checks, asset hashes, partial download, private installation and explicit overrides.'
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
    $target=[IO.Path]::GetFullPath($fixture)
    if ((Split-Path -Parent $target) -eq $fixtureParent -and (Split-Path -Leaf $target) -match '^component-discovery-[a-f0-9-]{36}$') { Remove-Item -LiteralPath $target -Recurse -Force }
}
