# Windows PowerShell 5.1 uses .NET Framework, which has no
# Path.IsPathFullyQualified. Keep drive-relative and root-relative paths rejected.
function Test-ControlFullyQualifiedWindowsPath {
  param([AllowNull()][AllowEmptyString()][string]$Path)
  return $null -ne $Path -and $Path -match '^(?:[A-Za-z]:[\\/]|[\\/]{2})'
}

function Get-ControlLocalWorkspacePath {
  param([string]$Path)
  if ($Path -notmatch '^[A-Za-z]:[\\/]' -or $Path -match '(^|[\\/])\.\.([\\/]|$)') {
    throw 'Use an absolute local drive path without parent traversal for the workspace.'
  }
  $full = [IO.Path]::GetFullPath($Path)
  if ($full.Length -gt 3) { $full = $full.TrimEnd('\', '/') }
  return $full
}

# Stream the release asset using .NET, including on Windows PowerShell hosts
# where the optional Get-FileHash module was not loaded in a fresh profile.
function Get-ControlFileSha256([string]$Path) {
  $stream = [IO.File]::OpenRead($Path)
  try {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
  } finally { $stream.Dispose() }
}
