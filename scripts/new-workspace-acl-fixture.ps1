param([Parameter(Mandatory=$true)][string]$Name, [switch]$ManagedWorkspace)
$ErrorActionPreference = 'Stop'
if ($Name -notmatch '^(acl-plant|web-agent-zoh)-[a-zA-Z0-9-]+$') { throw 'Invalid test workspace name' }
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$folder = if ($ManagedWorkspace) { 'workspaces' } else { 'test-output' }
$testRoot = (Resolve-Path -LiteralPath (Join-Path $project $folder)).ProviderPath
$ancestor = [IO.DirectoryInfo]::new($testRoot)
while ($ancestor) {
    if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing a redirected test root' }
    $ancestor = $ancestor.Parent
}
$leaf = Join-Path $testRoot $Name
# New-Item without Force refuses an existing work directory.
$directory = New-Item -ItemType Directory -Path $leaf -ErrorAction Stop
$directory.Refresh()
if ($directory.Parent.FullName -ne $testRoot -or $directory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Refusing unexpected newly created directory'
}
$probe = Join-Path $project 'src/control_harness/workspace_probe.py'
$python = Join-Path $project '.venv/Scripts/python.exe'
function Inspect-Leaf { return (& $python -I -S -B $probe --inspect --windows-acl -- $leaf | ConvertFrom-Json) }
$before = Inspect-Leaf
if (-not $before.owner_is_current_user -or $before.filesystem -ne 'NTFS') { throw 'Fixture requires local NTFS and current owner' }
if ($before.effective_rights.WRITE_OWNER -or $ManagedWorkspace) {
    # Only the new, empty test leaf. An explicit deny would make normal
    # prepare-workspace -Prepare unable to supply the missing allow bit.
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().User
    $system = [Security.Principal.SecurityIdentifier]::new('S-1-5-18')
    $acl = if ($PSVersionTable.PSEdition -eq 'Core') { [IO.FileSystemAclExtensions]::GetAccessControl($directory) } else { $directory.GetAccessControl() }
    $acl.SetAccessRuleProtection($true,$false)
    foreach ($entry in @($acl.Access)) { $acl.RemoveAccessRuleSpecific($entry) }
    $rights = [Security.AccessControl.FileSystemRights]'Modify,ChangePermissions'
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($user,$rights,'ContainerInherit,ObjectInherit','None','Allow'))
    $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($system,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
    if ($PSVersionTable.PSEdition -eq 'Core') { [IO.FileSystemAclExtensions]::SetAccessControl($directory,$acl) }
    else { $directory.SetAccessControl($acl) }
}
$after = Inspect-Leaf
if (-not $after.owner_is_current_user -or -not $after.effective_rights.WRITE_DAC -or $after.effective_rights.WRITE_OWNER -or $after.right_errors.WRITE_OWNER -ne 5) {
    throw 'Test workspace does not exhibit the requested WRITE_OWNER failure'
}
[ordered]@{workspace=$leaf;owner_is_current_user=$true;effective_rights=$after.effective_rights;right_errors=$after.right_errors;
    fixture_changed_new_leaf_acl=($before.effective_rights.WRITE_OWNER -or $ManagedWorkspace)} | ConvertTo-Json -Depth 3
