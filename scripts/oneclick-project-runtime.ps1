# Windows x64, PowerShell 5.1+. Dot-sourced by the interactive launcher.
. (Join-Path $PSScriptRoot 'windows-paths.ps1')

function Find-ControlNode([string]$PrivateBin) {
    $candidates = @((Join-Path $PrivateBin 'node.exe')) + @(Get-Command node.exe -All -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object Source)
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $version = & $candidate -p "[process.versions.node,process.arch].join(' ')"
            if ($LASTEXITCODE -eq 0 -and $version -match '^(\d+\.\d+\.\d+) x64$' -and [version]$Matches[1] -ge [version]'24.9.0') { return $candidate }
        }
    }
}

function Get-ControlPortableNode([string]$Destination) {
    # The Node MSI normally needs elevation. The official portable archive includes node and npm.
    $tag = 'v24.22.3'
    $name = "node-$tag-win-x64"
    $origin = "https://nodejs.org/dist/$tag"
    $parent = Split-Path -Parent $Destination
    $stage = Join-Path $parent ('download-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    $archive = Join-Path $stage "$name.zip"
    # https://nodejs.org/dist/v24.22.3/SHASUMS256.txt
    $expected = 'e612af99f626ab415ea225af6bce5c971feb448cb5cae250bb1c3fa50c5f7d9f'
    Invoke-WebRequest -UseBasicParsing -Uri "$origin/$name.zip" -OutFile $archive
    if ((Get-ControlFileSha256 $archive) -ne $expected) { throw 'Node.js 下载文件的 SHA-256 校验失败，请检查网络后重试。' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, $stage)
    # Publish only the completed distribution. No partial installation becomes active after a failed transfer.
    $source = (Resolve-Path -LiteralPath (Join-Path $stage $name)).Path
    $destinationPath = [IO.Path]::GetFullPath($Destination)
    if ((Split-Path -Parent $source) -ne $stage -or (Split-Path -Parent $destinationPath) -ne $parent) { throw 'Node.js 暂存位置与项目工具目录不符，已停止安装。' }
    [IO.Directory]::Move($source, $destinationPath)
    Remove-Item -LiteralPath $archive
    [IO.Directory]::Delete($stage)
}

function Install-ControlUserPackage([string]$Id) {
    $winget = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $winget) { throw "缺少 WinGet。请安装 Microsoft 应用安装程序，或为当前用户安装 $Id 后重新启动。" }
    & $winget.Source install --id $Id --exact --source winget --scope user --architecture x64 --silent --disable-interactivity --accept-source-agreements --accept-package-agreements | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "WinGet 安装 $Id 失败，退出码为 $LASTEXITCODE。请查看上方 WinGet 输出并重试。" }
    # Winget updates user PATH in the registry. Preserve the caller's existing process environment.
    $userPath = [Environment]::GetEnvironmentVariable('Path','User')
    if ($userPath) { $env:Path += ';' + [Environment]::ExpandEnvironmentVariables($userPath) }
}

function Test-ControlPython([string]$Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { return $false }
    # Python 3.14+ and 32-bit Python cannot use the pinned Windows PyTorch wheels.
    try {
        $null = & $Python -I -B -c "import sys, sysconfig; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13) and sys.maxsize > 2**32 and not sysconfig.get_config_var('Py_GIL_DISABLED')))" 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Find-ControlBasePython {
    $candidates = @(Get-Command python.exe -All -CommandType Application -ErrorAction SilentlyContinue | ForEach-Object Source | Where-Object { $_ -notmatch '\\Microsoft\\WindowsApps\\' })
    foreach ($hive in 'HKCU','HKLM') {
        foreach ($sub in Get-ChildItem -Path ("${hive}:\SOFTWARE\Python\PythonCore") -ErrorAction SilentlyContinue) {
            $key = Get-Item -LiteralPath (Join-Path $sub.PSPath 'InstallPath') -ErrorAction SilentlyContinue
            if ($key -and $key.GetValue('')) { $candidates += Join-Path $key.GetValue('') 'python.exe' }
        }
    }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-ControlPython $candidate) { return $candidate }
    }
}

function Test-ControlPythonModules([string]$Python, [string]$ProjectRoot) {
    try {
        $null = & $Python -I -B -c 'import sys; sys.path.insert(0,sys.argv[1]); import control_harness, numpy, scipy, torch, openpyxl, threadpoolctl' (Join-Path $ProjectRoot 'src') 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

function Test-ControlBuild([string]$ProjectRoot) {
    foreach ($file in 'vendor/deepseek-harness/apps/cli/lib/bin.js','vendor/deepseek-harness/apps/web/dist/index.html','vendor/deepseek-harness/packages/mcp/mcp-client/lib/index.js','harness-plugin/lib/client.js','harness-plugin/node_modules/@deepseek-ai/dsh-tools/package.json') {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $file) -PathType Leaf)) { return $false }
    }
    return $true
}

function Test-ControlProfile([string]$HarnessHome,[string]$ProjectRoot) {
    $profileDir = Join-Path $HarnessHome 'profiles/web'
    $bundle = '@control-engineering/harness-bundle'
    $manifestFile = Join-Path $profileDir 'package.json'
    if (-not (Test-Path -LiteralPath $manifestFile -PathType Leaf)) { return $false }
    $manifest = Get-Content -LiteralPath $manifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $link = [string]$manifest.dependencies.$bundle
    $expectedPath = Join-Path $ProjectRoot 'harness-plugin'
    $entry = Join-Path $profileDir "node_modules/$bundle"
    if ($bundle -notin @($manifest.dsh.profile.bundles) -or $link.Replace('\','/') -ne ('link:' + $expectedPath.Replace('\','/')) -or
        -not (Test-Path -LiteralPath (Join-Path $entry 'src/index.mjs') -PathType Leaf)) { return $false }
    $target = @( (Get-Item -LiteralPath $entry).Target )[0]
    if (-not $target) { return $false }
    if (-not [IO.Path]::IsPathRooted($target)) { $target = Join-Path (Split-Path -Parent $entry) $target }
    return [IO.Path]::GetFullPath($target) -eq $expectedPath
}

function Initialize-ControlProjectRuntime {
    param([Parameter(Mandatory=$true)][string]$ProjectRoot, [string]$HarnessHome, [switch]$CheckOnly)
    $ErrorActionPreference = 'Stop'
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
    if (-not $HarnessHome) { $HarnessHome = Join-Path $ProjectRoot '.dsh' }
    $runtimeDir = Join-Path $ProjectRoot '.dsh/oneclick/runtime'
    $privateNode = Join-Path $runtimeDir 'node-v24.22.3-win-x64'
    $node = Find-ControlNode $privateNode
    $venv = Join-Path $ProjectRoot '.venv'
    $python = Join-Path $venv 'Scripts/python.exe'
    $pythonOK = Test-ControlPython $python
    $modulesOK = $pythonOK -and (Test-ControlPythonModules $python $ProjectRoot)
    $built = Test-ControlBuild $ProjectRoot
    $registered = Test-ControlProfile $HarnessHome $ProjectRoot
    $missing = @(
        if (-not $node) { 'Node.js >= 24.9 x64' }
        if (-not $pythonOK) { '项目 .venv（CPython 3.10–3.13 x64）' }
        if (-not $modulesOK) { 'Python 数值依赖' }
        if (-not $built) { 'Harness 与网页插件构建' }
        if (-not $registered) { '网页配置中的项目插件注册' }
    )
    if ($CheckOnly) { return [pscustomobject]@{Ready=($missing.Count -eq 0); Missing=$missing; NodePath=$node; PythonPath=$python} }
    if (-not $node) {
        Write-Host '[准备项目] 正在从 nodejs.org 下载仅供本项目使用的 Node.js…'
        Get-ControlPortableNode $privateNode
        $node = Find-ControlNode $privateNode
        if (-not $node) { throw '下载的 Node.js 无法运行。请确认使用 Windows x64，并检查安装输出。' }
    }
    $env:Path = (Split-Path -Parent $node) + ';' + $env:Path
    $creatingVenv = Join-Path $venv '.control-venv-setup'
    if (-not $pythonOK) {
        if ((Test-Path -LiteralPath $venv) -and -not (Test-Path -LiteralPath $creatingVenv -PathType Leaf)) {
            throw "已有环境 $venv 不完整或版本不兼容。请先将其移到备份目录，然后重新启动。"
        }
        $base = Find-ControlBasePython
        if (-not $base) {
            Write-Host '[准备项目] 正在为当前 Windows 用户安装 64 位 Python 3.12…'
            Install-ControlUserPackage 'Python.Python.3.12'
            $base = Find-ControlBasePython
            if (-not $base) { throw '未找到可用的 64 位 python.exe。请核对 Python 3.12 的安装后重试。' }
        }
        if (-not (Test-Path -LiteralPath $venv)) {
            New-Item -ItemType Directory -Path $venv | Out-Null
            New-Item -ItemType File -Path $creatingVenv | Out-Null
        }
        & $base -m venv $venv | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-ControlPython $python)) { throw '项目的 .venv 创建失败，请查看上方 Python 输出。' }
    }
    if (-not $modulesOK) {
        Write-Host '[准备项目] 正在将数值计算依赖安装到项目的 .venv…'
        # A prior interrupted venv creation may have left a working interpreter without pip.
        & $python -m ensurepip --upgrade | Out-Host
        if ($LASTEXITCODE -ne 0) { throw '在 .venv 中初始化 pip 失败，请检查 Python 安装后重试。' }
        & $python -m pip --disable-pip-version-check install --no-input -c (Join-Path $ProjectRoot 'constraints-v2-validated.txt') -e $ProjectRoot | Out-Host
        if ($LASTEXITCODE -ne 0 -or -not (Test-ControlPythonModules $python $ProjectRoot)) { throw 'Python 依赖安装或加载失败，请核对上方 pip 输出后重试。' }
    }
    if (Test-Path -LiteralPath $creatingVenv -PathType Leaf) { Remove-Item -LiteralPath $creatingVenv }
    if (-not $built -or -not $registered) {
        if (-not $built -and -not (Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue)) {
            Write-Host '[准备项目] 正在为当前 Windows 用户安装 Git（MinGit）…'
            Install-ControlUserPackage 'Git.MinGit'
            if (-not (Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue)) { throw '安装后仍未找到 git.exe，请重新打开启动脚本以刷新 PATH。' }
        }
        # The upstream package manager is pnpm 11. Install the pinned version privately when needed.
        $pnpmDir = Join-Path $runtimeDir 'pnpm'
        if (Test-Path -LiteralPath (Join-Path $pnpmDir 'pnpm.cmd')) { $env:Path = "$pnpmDir;$env:Path" }
        $pnpm = Get-Command -Name pnpm.cmd,pnpm.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $version = if ($pnpm) { & $pnpm.Source --version }
        if (-not $pnpm -or $LASTEXITCODE -ne 0 -or $version -notmatch '^11\.') {
            $npm = Join-Path (Split-Path -Parent $node) 'npm.cmd'
            if (-not (Test-Path -LiteralPath $npm)) { throw "$node 所在目录缺少 npm.cmd，请安装含 npm 的 Node.js。" }
            & $npm install --global --prefix $pnpmDir --registry https://registry.npmjs.org --no-audit --no-fund pnpm@11.28.5 | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $pnpmDir 'pnpm.cmd'))) { throw '项目内 pnpm 安装失败，请检查上方 npm 输出。' }
            $env:Path = "$pnpmDir;$env:Path"
        }
        Write-Host '[准备项目] 正在准备固定版本的 Harness 并注册网页插件…'
        & (Join-Path $ProjectRoot 'scripts/setup-harness.ps1') -HarnessHome $HarnessHome -RegisterOnly:$built | Out-Host
        if (-not (Test-ControlBuild $ProjectRoot) -or -not (Test-ControlProfile $HarnessHome $ProjectRoot)) { throw 'Harness 构建或网页插件注册未完成。' }
    }
    return [pscustomobject]@{Ready=$true; Missing=@(); NodePath=$node; PythonPath=$python}
}
