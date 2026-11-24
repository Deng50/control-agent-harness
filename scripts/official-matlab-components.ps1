# Private copy of the signed release assets. Register toolbox paths only in
# the dedicated process; installing these GUIDs in Add-On Manager is global.
function Resolve-ControlOfficialMatlabComponents {
    param([string]$Directory, [string[]]$Names = @('MatlabMcpExecutable','MatlabMcpToolboxRoot','SimulinkToolkitRoot'), [switch]$CheckOnly)
    $releases = @(
        @{ Name='matlab-mcp-server-windows-x64.exe'; Repo='matlab-mcp-server'; Tag='v0.16.1'; Hash='7feca987345a2c85b683f06d6bad8da8396676cf93e138e24adea665dfbd9db6'; Key='MatlabMcpExecutable' },
        @{ Name='MATLABMCPServerToolbox.mltbx'; Repo='matlab-mcp-server'; Tag='v0.16.1'; Hash='0db28ffa84cc16cfb9537c2603dca2bf2875225a5c8e11044751f1e62f8a35a8'; Key='MatlabMcpToolboxRoot' },
        @{ Name='simulink-agentic-toolkit.mltbx'; Repo='simulink-agentic-toolkit'; Tag='SATK-2026.11.c'; Hash='42cfc4d71446a1189914ffc5107c35ce7affdc67524a8837ead3e8e61a497047'; Key='SimulinkToolkitRoot' }
    )
    $paths = @{}
    foreach ($asset in $releases) {
        if ($asset.Key -notin $Names) { continue }
        $downloads = Join-Path $Directory 'downloads'
        $source = Join-Path $downloads $asset.Name
        if ($CheckOnly) {
            if (-not [IO.File]::Exists($source) -or (Get-ControlFileSha256 $source) -ne $asset.Hash) { continue }
        } else {
            $uri = 'https://github.com/matlab/' + $asset.Repo + '/releases/download/' + $asset.Tag + '/' + $asset.Name
            $source = Get-ControlMatlabAsset $downloads $asset.Name $uri $asset.Hash
        }
        if ($asset.Key -eq 'MatlabMcpExecutable') { $paths[$asset.Key] = $source; continue }
        $target = Join-Path $Directory $asset.Hash
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            if ($CheckOnly) { continue }
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            $staging = Join-Path $Directory ([guid]::NewGuid().ToString() + '.installing')
            [IO.Compression.ZipFile]::ExtractToDirectory($source, $staging)
            # The pinned packages contain their MATLAB files under fsroot and
            # their own ordered search path list in metadata/configuration.xml.
            try { [IO.Directory]::Move($staging, $target) }
            catch [IO.IOException] { if (-not [IO.Directory]::Exists($target)) { throw } }
        }
        $candidate = Find-ControlMatlabComponent $asset.Key '' @((Join-Path $target 'fsroot'))
        if ($candidate -and [IO.File]::Exists((Join-Path $target 'metadata/configuration.xml'))) {
            $paths[$asset.Key] = $candidate
        } elseif (-not $CheckOnly) {
            throw "Incomplete private toolbox installation at $target."
        }
    }
    return $paths
}
