# Test the duplicate-launch path without a browser, MATLAB or a configured Harness profile.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'oneclick-web.ps1')
$project = Split-Path -Parent $PSScriptRoot
$fixture = Join-Path $project ('test-output\oneclick web ' + [guid]::NewGuid())
$cli = Join-Path $fixture 'vendor\deepseek-harness\apps\cli\lib\bin.js'
$overlay = Join-Path $fixture 'config\web.overlay.yml'
$scripts = Join-Path $fixture 'scripts'
$foreignRoot = Join-Path $fixture 'other checkout'
foreach ($directory in (Split-Path -Parent $cli),(Split-Path -Parent $overlay),$scripts,$foreignRoot) {
    [void][IO.Directory]::CreateDirectory($directory)
}
foreach ($name in 'oneclick-web.ps1','windows-paths.ps1','start-control-harness.ps1') {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination (Join-Path $scripts $name)
}
[IO.File]::WriteAllText((Join-Path $scripts 'oneclick-matlab-components.ps1'),@'
function Resolve-ControlMatlabComponents {
    $fixturePath = Split-Path -Parent $PSScriptRoot
    if (Test-Path (Join-Path $fixturePath 'preparing')) { throw 'Unexpected second launch.' }
    [IO.File]::WriteAllText((Join-Path $fixturePath 'preparing'),'1')
    $wait = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path (Join-Path $fixturePath 'continue'))) {
        if ($wait.Elapsed.TotalSeconds -gt 45) { throw 'Test timed out.' }
        Start-Sleep -Milliseconds 100
    }
    return @{}
}
'@)
[IO.File]::WriteAllText((Join-Path $scripts 'oneclick-project-runtime.ps1'),
    'function Initialize-ControlProjectRuntime { return @{Ready=$true} }')
[IO.File]::WriteAllText((Join-Path $scripts 'oneclick-session.ps1'),@'
function Get-ControlSessionAppData { return 'unused' }
function Get-ControlLiveProcess([int]$ProcessId) { return Get-Process -Id $ProcessId -ErrorAction SilentlyContinue }
function Read-ControlStartupResource($Python,$Directory) {
    return Get-Content -Raw -LiteralPath (Join-Path $Directory 'resource.json') | ConvertFrom-Json
}
'@)
[IO.File]::WriteAllText((Join-Path $scripts 'prepare-workspace.ps1'),'# Test fixture.')
[IO.File]::WriteAllText((Join-Path $scripts 'start-web.ps1'),@'
param([int]$Port)
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& node.exe (Join-Path $projectRoot 'vendor\deepseek-harness\apps\cli\lib\bin.js') web --patch (Join-Path $projectRoot 'config\web.overlay.yml') --port $Port
exit $LASTEXITCODE
'@)
[IO.File]::WriteAllText($overlay,'# Test fixture.')
[IO.File]::WriteAllText($cli,@'
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
const server = http.createServer((req, res) => {
  const ready = fs.existsSync('ready');
  res.writeHead(ready ? 401 : 503, {'Content-Type': 'text/plain'});
  res.end(ready ? 'dsh web authentication required; reopen the URL printed by dsh web.\n' : 'starting\n');
});
const original = !fs.existsSync('listening');
const scope = path.resolve('.dsh/oneclick');
const resource = path.resolve('.dsh/matlab/shared-resource');
const id = require('node:crypto').randomUUID();
for (const dir of [scope,resource]) fs.mkdirSync(dir,{recursive:true});
const flag = path.join(scope,`web-${process.pid}-${id}.rebind`);
const binding = {pid:process.ppid,session_id:id,host:require('node:os').hostname(),identity:'win32:0000000000000001'};
fs.writeFileSync(path.join(scope,`web-${process.pid}.json`),JSON.stringify({
  web_pid:process.pid, project_root:process.cwd(), resource_directory:resource, matlab_pid:binding.pid, session_id:id
}));
fs.writeFileSync(path.join(resource,'resource.json'),JSON.stringify({session_alive:true,session:binding}));
server.listen(port, '127.0.0.1', () => fs.writeFileSync(original ? 'listening' : fs.existsSync('crashed') ? 'restarted' : 'rebound', String(process.pid)));
const timer = setInterval(() => {
  if (fs.existsSync('stop')) { clearInterval(timer); server.close(); }
  else if (fs.existsSync('request-fixture-crash') && !fs.existsSync('crashed')) {
    const launch = process.env.CONTROL_ONECLICK_HOST_RUN;
    fs.writeFileSync(path.join(scope,'ready-'+launch+'.json'),JSON.stringify({
      launch_id:launch,web_pid:process.pid,project_root:process.cwd(),resource_directory:resource,port,session:binding
    }));
    fs.writeFileSync(path.join(resource,'resource.json'),JSON.stringify({
      session_alive:true,session:binding,owner:null,queue:[],quarantine:{owner:{phase:'running',session:binding}}
    }));
    fs.writeFileSync('crashed','1');
    process.exit(-1073741819); // Controlled fixture exit code, no actual access violation.
  }
  else if (fs.existsSync(flag) && fs.readFileSync(flag,'utf8') === 'no-open') {
    clearInterval(timer); process.exitCode = 76; server.close();
  }
}, 100);
'@)
function Get-TestPort {
    $socket = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0)
    try { $socket.Start(); return $socket.LocalEndpoint.Port }
    finally { $socket.Stop() }
}
function Wait-TestMarker($Name,$Worker) {
    $marker = Join-Path $fixture $Name
    $wait = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Path -LiteralPath $marker)) {
        if ($Worker.HasExited -or $wait.Elapsed.TotalSeconds -gt 20) { throw "Test launcher did not reach '$Name'. See $fixture." }
        Start-Sleep -Milliseconds 100
    }
}
$port = Get-TestPort
$freePort = Get-TestPort
while ($freePort -eq $port) { $freePort = Get-TestPort }
$launcher = Join-Path $scripts 'start-control-harness.ps1'
$first = $null
try {
    $launchArgs = @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"{0}"' -f $launcher),'-Port',[string]$port,'-NoOpen')
    $first = Start-Process -FilePath powershell.exe -ArgumentList $launchArgs -WorkingDirectory $fixture -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $fixture 'stdout.log') -RedirectStandardError (Join-Path $fixture 'stderr.log')
    Wait-TestMarker 'preparing' $first
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Port $port -NoOpen | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'A duplicate launch during preparation exited with an error.' }

    [IO.File]::WriteAllText((Join-Path $fixture 'continue'),'1')
    Wait-TestMarker 'listening' $first
    $serverId = [int][IO.File]::ReadAllText((Join-Path $fixture 'listening'))
    $starting = Get-ControlExistingWeb -ProjectRoot $fixture -Port $port
    if ($starting.State -ne 'Starting') { throw 'Unready project listener reported as ready.' }
    [IO.File]::WriteAllText((Join-Path $fixture 'ready'),'1')
    $ready = Get-ControlExistingWeb -ProjectRoot $fixture -Port $port
    if ($ready.State -ne 'Ready' -or $ready.ProcessId -ne $serverId -or $ready.Port -ne $port) { throw 'Ready listener identity mismatch.' }
    $found = Get-ControlExistingWeb -ProjectRoot $fixture -Port $freePort -FindOtherPort
    if ($found.State -ne 'Ready' -or $found.Port -ne $port) { throw 'Existing custom-port launch was not found.' }

    # The very same HTTP response from another checkout must not be opened.
    $foreign = Get-ControlExistingWeb -ProjectRoot $foreignRoot -Port $port
    if ($foreign.State -ne 'Occupied') { throw 'Another checkout was incorrectly reused.' }
    & {
        $opened = [Collections.Generic.List[string]]::new()
        function Start-Process { [CmdletBinding()] param([string]$FilePath) $opened.Add($FilePath) }
        Show-ControlExistingWeb -Web $starting
        Show-ControlExistingWeb -Web $ready -NoOpen
        if ($opened.Count) { throw 'Preparing or -NoOpen attempted a browser launch.' }
        $rejected = $false
        try { Show-ControlExistingWeb -Web $foreign } catch { $rejected = $true }
        if (-not $rejected -or $opened.Count) { throw 'A foreign listener was offered to the browser.' }
        Show-ControlExistingWeb -Web $ready
        if ($opened.Count -ne 1 -or $opened[0].TrimEnd('/') -ne "http://127.0.0.1:$port") { throw 'Browser target is not the verified local origin.' }
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Port $port -NoOpen | Out-Host
    if ($LASTEXITCODE -ne 0 -or $first.HasExited) { throw 'Duplicate launch did not reuse the original Host.' }
    $handoff = Join-Path $fixture ".dsh/oneclick/web-$serverId.json"
    $published = [IO.File]::GetLastWriteTimeUtc($handoff)
    $staleRejected = $false
    try {
        [IO.File]::SetLastWriteTimeUtc($handoff,(Get-Process -Id $serverId).StartTime.ToUniversalTime().AddMinutes(-1))
        try { Request-ControlWebRebind $ready $fixture (Join-Path $fixture '.dsh') -NoOpen | Out-Null }
        catch { if ($_.Exception.Message -like '*Host*') { $staleRejected = $true } else { throw } }
    } finally { [IO.File]::SetLastWriteTimeUtc($handoff,$published) }
    if (-not $staleRejected) { throw 'The launcher reused an earlier PID owner marker.' }
    $stateFile = Join-Path $fixture '.dsh/matlab/shared-resource/resource.json'
    $state = Get-Content -Raw -LiteralPath $stateFile | ConvertFrom-Json
    $state.session_alive = $false
    [IO.File]::WriteAllText($stateFile,($state | ConvertTo-Json))
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Port $port -NoOpen | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'The launcher failed to request rebinding.' }
    Wait-TestMarker 'rebound' $first
    $after = Get-ControlExistingWeb -ProjectRoot $fixture -Port $port
    if ($after.State -ne 'Ready' -or $after.ProcessId -eq $serverId -or $first.HasExited) { throw 'Original supervisor did not start a replacement Host.' }
    [IO.File]::WriteAllText((Join-Path $fixture 'request-fixture-crash'),'1')
    Wait-TestMarker 'restarted' $first
    $replacement = Get-ControlExistingWeb -ProjectRoot $fixture -Port $port
    if ($replacement.State -ne 'Ready' -or $replacement.ProcessId -eq $after.ProcessId -or $first.HasExited) { throw 'Fatal ready Host was not restarted.' }
    Write-Output 'PASS: preparing, ready, owner and port matching, custom port, foreign listener, duplicate launcher, -NoOpen, browser target, stale marker, rebind and verified fatal Host restart.'
} finally {
    if ($first) {
        [IO.File]::WriteAllText((Join-Path $fixture 'continue'),'1')
        [IO.File]::WriteAllText((Join-Path $fixture 'stop'),'1')
        if (-not $first.WaitForExit(10000)) { $first.Kill(); $first.WaitForExit() }
        $first.Dispose()
    }
}
