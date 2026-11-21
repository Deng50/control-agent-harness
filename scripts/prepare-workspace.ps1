param([Parameter(Mandatory=$true)][string]$Workspace, [switch]$Prepare)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-paths.ps1')
$project = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$path = Get-ControlLocalWorkspacePath $Workspace
$python = Join-Path $project '.venv\Scripts\python.exe'
$probe = Join-Path $project 'src\control_harness\workspace_probe.py'

function Read-WorkspaceSecurity([string]$Directory) {
    $json = & $python -I -S -B $probe --inspect --windows-acl -- $Directory
    if (-not $json) { throw 'Workspace diagnosis did not return a result. Check the project Python environment.' }
    return ($json | ConvertFrom-Json)
}

if (-not $Prepare) {
    $state = Read-WorkspaceSecurity $path
    $state | ConvertTo-Json -Depth 6
    if (-not $state.ok) { throw ('Workspace preflight failed: ' + ($state.error | ConvertTo-Json -Compress -Depth 4)) }
    return
}

# Preparation is explicit. Lock the entire existing path against replacement
# while checking for reparse points and adjusting the leaf security descriptor.
$prefixes = @((Join-Path $project 'workspaces'), (Join-Path $project 'test-output'))
if (-not ($prefixes | Where-Object { $path.StartsWith(($_ + '\'), [StringComparison]::OrdinalIgnoreCase) })) {
    throw 'Explicit preparation requires a dedicated child of this project workspaces or test-output directory.'
}
if (-not ('ControlWorkspaceDirectoryPin' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class ControlWorkspaceDirectoryPin {
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
    private static extern SafeFileHandle CreateFileW(string path, uint access, uint share,
        IntPtr security, uint disposition, uint flags, IntPtr template);
    public static SafeFileHandle Pin(string path) {
        var handle = CreateFileW(path, 0x80, 3, IntPtr.Zero, 3, 0x02200000, IntPtr.Zero);
        if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error());
        return handle;
    }
}
'@
}
$parts = [Collections.Generic.Stack[string]]::new()
$entry = [IO.DirectoryInfo]::new($path)
while ($null -ne $entry) { $parts.Push($entry.FullName); $entry = $entry.Parent }
$pins = [Collections.Generic.Stack[IDisposable]]::new()
try {
    foreach ($entryPath in $parts) {
        if (-not [IO.Directory]::Exists($entryPath)) {
            # Each parent has been pinned and checked; only new work directories
            # get created here. We never change an ancestor's ACL.
            [void][IO.Directory]::CreateDirectory($entryPath)
        }
        $pins.Push([ControlWorkspaceDirectoryPin]::Pin($entryPath))
        if (([IO.File]::GetAttributes($entryPath) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Preparation refused a reparse point: $entryPath"
        }
    }
    $before = Read-WorkspaceSecurity $path
    if (-not $before.owner_is_current_user -or $before.filesystem -ne 'NTFS') {
        throw 'Preparation requires a local NTFS workspace owned by the current user.'
    }
    $added = @()
    if (-not $before.ok) {
        foreach ($right in @('WRITE_DAC', 'WRITE_OWNER')) {
            if (-not $before.effective_rights.$right) {
                if ($before.right_errors.$right -ne 5) { throw "OS error checking $right. No ACL change attempted." }
                $added += $right
            }
        }
        if (-not $added.Count) { throw 'The preflight failure cannot be repaired with a leaf ACE.' }
        $mask = 0
        if ('WRITE_DAC' -in $added) { $mask = $mask -bor 0x40000 }
        if ('WRITE_OWNER' -in $added) { $mask = $mask -bor 0x80000 }
        $leaf = [IO.DirectoryInfo]::new($path)
        $sections = [Security.AccessControl.AccessControlSections]'Access, Owner'
        if ($PSVersionTable.PSEdition -eq 'Core') { $acl = [IO.FileSystemAclExtensions]::GetAccessControl($leaf, $sections) }
        else { $acl = $leaf.GetAccessControl($sections) }
        $user = [Security.Principal.WindowsIdentity]::GetCurrent().User
        if ($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -ne $user.Value) { throw 'Workspace owner changed.' }
        $rule = [Security.AccessControl.FileSystemAccessRule]::new($user,
            [Security.AccessControl.FileSystemRights]$mask, 'None', 'None', 'Allow')
        $acl.AddAccessRule($rule)
        if ($PSVersionTable.PSEdition -eq 'Core') { [IO.FileSystemAclExtensions]::SetAccessControl($leaf, $acl) }
        else { $leaf.SetAccessControl($acl) }
    }
    $after = Read-WorkspaceSecurity $path
    [ordered]@{prepared=$after.ok; workspace=$path; added_leaf_rights=$added; before=$before; after=$after} | ConvertTo-Json -Depth 8
    if (-not $after.ok) { throw 'Leaf preparation did not satisfy the effective-rights check.' }
} finally {
    while ($pins.Count) { $pins.Pop().Dispose() }
}
