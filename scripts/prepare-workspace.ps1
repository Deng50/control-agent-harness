param([Parameter(Mandatory=$true)][string]$Workspace)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$workspacePath = [IO.Path]::GetFullPath($Workspace).TrimEnd('\')
if (-not $workspacePath.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'This helper only prepares workspaces inside this integration project.'
}
New-Item -ItemType Directory -Path $workspacePath -Force | Out-Null
$workspaceAcl = Get-Acl -LiteralPath $workspacePath
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$ownerSid = $workspaceAcl.GetOwner([Security.Principal.SecurityIdentifier])
if ($ownerSid.Value -ne $identity.User.Value) {
    throw 'Workspace must be owned by the current user. This helper never takes ownership.'
}
# Harness Windows ACL confinement needs WRITE_OWNER to install its low label.
# D: commonly inherits Modify only; grant FullControl only to the existing owner.
$rule = [Security.AccessControl.FileSystemAccessRule]::new(
    $identity.User, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
$workspaceAcl.SetAccessRule($rule)
Set-Acl -LiteralPath $workspacePath -AclObject $workspaceAcl
Write-Output "Prepared owner permissions for Harness confinement: $workspacePath"
