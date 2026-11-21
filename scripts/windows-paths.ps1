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
