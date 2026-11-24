# MATLAB 与 Simulink 的可选本机连接

## 在新电脑上一键启动

在 Windows x64 安装并激活 MATLAB 和 Simulink，将仓库完整解压至当前用户可写的本地 NTFS 目录，双击根目录的 `Start-ControlHarness.cmd`。它使用 Windows 自带的 PowerShell。双击入口的脚本执行选项只作用于该进程，单位统一设置的组策略仍然生效。

关闭了浏览器标签时，再次双击同一启动器即可打开已运行的网页，并沿用该服务的 MATLAB 会话。重复启动时使用 `-NoOpen` 可以只查看状态。若首次启动还在安装或做连接检查，请保留第一个窗口，等待它给出就绪消息。重复打开使用浏览器已有的本地登录状态；换用其他浏览器或清除了站点数据时，请使用首次启动窗口提供的登录地址。

首次启动会按固定的 GitHub Release SHA-256 校验 Windows MCP 可执行文件和两个 `.mltbx` 包，并将工具箱展开到本检出的 `.control-matlab/<包哈希>/fsroot`。仅在专用 MATLAB 进程中按官方包的 `metadata/configuration.xml` 加载已声明的路径。可通过绝对路径显式选择其他版本。[MATLAB MCP Server v0.16.1](https://github.com/matlab/matlab-mcp-server/releases/tag/v0.16.1) · [Simulink Agentic Toolkit 2026.11.c](https://github.com/matlab/simulink-agentic-toolkit/releases/tag/SATK-2026.11.c)

网页服务启动时会通过自己的 MCP 执行器查询 MATLAB 的发行版、PID、会话标识并计算 `2+3`，同时检查 Simulink 工具是否已注册。收到本次调用的正确回执才会打印“MATLAB 连接验证通过”并提供网页地址。按 Ctrl+C 可以停止前台网页服务。已核验的专用 MATLAB 可在下次启动时复用；首次安装依赖或等待许可证超时，可查看窗口给出的错误、修复后重新运行同一入口。

在项目根目录的 PowerShell 可使用以下命令。第一条只做本机文件与依赖检查。第二条示例显式选择 MATLAB 并指定网页端口，适合安装了多个版本的机器。

```powershell
.\Start-ControlHarness.cmd -CheckOnly
.\Start-ControlHarness.cmd -MatlabExecutable 'C:\Program Files\MATLAB\R2024b\bin\matlab.exe' -Port 5290
```

默认工作区为项目内的 `workspaces\project-01`，可用 `-Workspace` 指定其他绝对路径。一键入口会检查专用实例和上次未决调用的记录，恢复流程见下文。若从其他路径预装了依赖，可用 `-MatlabMcpExecutable`、`-SimulinkToolkitRoot`、`-MatlabMcpToolboxRoot` 指定位置。项目内工作区由现有准备脚本处理，外部目录由其管理者准备。

## 分步配置与连接

本机实测环境为 Windows x64、MATLAB/Simulink R2024b、MATLAB MCP Server v0.16.1、MATLAB MCP Server Toolbox 0.5.0、Simulink Agentic Toolkit 2026.11.c。两个包的 `metadata/systemRequirements.xml` 均声明自 R2023a 开始兼容。

### 安装归属与固定版本

| 组件 | 切换前在本机选用的位置 | 项目当前默认位置与版本 |
|---|---|---|
| MATLAB/Simulink | `C:\APP\MATLAB2024B\R2024b`，系统级产品 | 同一安装，启动项目独占的 R2024b 进程 |
| MCP Server | `%USERPROFILE%\.matlab\agentic-toolkits\bin\matlab-mcp-server.exe`，散列与 v0.16.0 一致 | `.control-matlab/downloads/matlab-mcp-server-windows-x64.exe`，v0.16.1 |
| MATLAB MCP Server Toolbox | `%APPDATA%\MathWorks\MATLAB Add-Ons\Toolboxes\MATLAB MCP Server Toolbox`，Contents.m 标明 0.5.0 | `.control-matlab/0db28f…35a8/fsroot`，0.5.0 |
| Simulink Agentic Toolkit | `%USERPROFILE%\.matlab\agentic-toolkits\simulink`，VERSION 为 2026.11.a | `.control-matlab/42cfc4…7047/fsroot`，发行标记 2026.11.c，包内版本 1.0.35 |
| Web 配置 | 本检出的 `config/web.overlay.yml`、`config/matlab.overlay.yml` 和 `config/matlab-startup-check.overlay.yml` | 继续使用相同 overlay。启动器将所选 Server、Toolkit 和私有会话目录传给唯一的原生 stdio client |
| 会话发现和资源代理 | `<HarnessHome>/matlab/<启动目录>/appdata` 和 `<HarnessHome>/matlab/shared-resource` | 默认 HarnessHome 为本检出的 `.dsh`。测试使用 `test-output` 下独立的 Host、会话目录和资源代理 |

原有三项连接组件属于用户级安装，Codex 或其他客户端可能引用它们。项目的默认副本专属当前检出。需要回到以前的组合时，可在旧专用进程正常保存并退出后，向一键入口显式传入上表中的三个旧组件路径。三项原文件均可继续使用。

| 官方资产 | GitHub Release SHA-256 |
|---|---|
| matlab-mcp-server-windows-x64.exe | `7feca987345a2c85b683f06d6bad8da8396676cf93e138e24adea665dfbd9db6` |
| MATLABMCPServerToolbox.mltbx | `0db28ffa84cc16cfb9537c2603dca2bf2875225a5c8e11044751f1e62f8a35a8` |
| simulink-agentic-toolkit.mltbx | `42cfc4d71446a1189914ffc5107c35ce7affdc67524a8837ead3e8e61a497047` |

启动脚本兼容 Windows PowerShell 5.1 与 PowerShell 7，无需安装 `pwsh`。绝对路径检查使用兼容实现，同时拒绝 `C:relative` 和 `\root-relative`。可用 `powershell.exe -NoProfile -File .\scripts\test-startup-compatibility.ps1` 检查本机兼容性。复制命令时仅复制代码，不包含 `(base) PS ...>` 或 `>>` 提示符。

先显式启动一份本项目拥有的专用 MATLAB，再让 Harness 的唯一 MCP client 使用 `existing` 连接该进程。`start-web.ps1` 自身不启动 MATLAB；没有 `-Matlab` 时不加载 MATLAB overlay，锁配置保持未启用。

启动器创建新的私有会话目录。专用 MATLAB 在该进程内临时设置 `APPDATA` 并调用包中的 `shareMATLABSession`，MCP 子进程只使用同一发现目录。以下命令在项目根目录的 PowerShell 执行。

```powershell
. .\scripts\oneclick-matlab-components.ps1
$components = Resolve-ControlMatlabComponents
$launch = @{}
foreach ($name in 'MatlabExecutable','MatlabMcpExecutable','MatlabMcpToolboxRoot','SimulinkToolkitRoot') {
    $launch[$name] = $components.$name
}
$session = Join-Path $PWD ('.dsh\matlab\session-' + [guid]::NewGuid())
.\scripts\start-matlab-session.ps1 @launch -SessionDirectory $session
```

`-ValidateOnly` 只核对安装路径，不启动进程。实际启动后保存 PID 与日志，等待 `appdata/control-harness-session.json` 的 `ready` 标记；未完成时检查所记录进程，不重复启动。MATLAB 新进程的用户 startup 仍可能自行运行，应在当前受信环境核对；本项目不替用户禁用或改写它。

此公开身份文件不含连接凭据，字段为：`schema_version=1`、`kind="control-harness-dedicated-matlab-v1"`、`status="ready"`、随机 `session_id`、实际 `feature('getpid')` 的 `matlab_pid`（兼容别名 `pid`）、UTC ISO时间 `createdAt`、`project_root`、`private_appdata`、`matlab_release`、`toolkit_root`、`toolbox_root` 和 `mcp_executable`。启动器的 `launch.json` 记录启动句柄PID；绑定及回收以就绪身份文件中的实际 `matlab_pid` 为准。MATLAB同时保留 `CONTROL_HARNESS_SESSION_ID` 环境标记，并加载项目 `matlab/` 供原生身份及空闲探针使用。

实际启用 `start-web.ps1 -Matlab` 时，脚本会在启动 Host 前把身份绑定到共享资源目录；绑定冲突会阻断启动，`-ValidateOnly` 不绑定。需要单独检查绑定时，也可以手动执行（只读公开身份文件，不读取连接凭据）：

```powershell
$identity = Get-Content '<session-directory>\appdata\control-harness-session.json' -Raw | ConvertFrom-Json
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m control_harness.matlab_resource bind-session `
  --directory '<shared-resource-directory>' --pid $identity.matlab_pid --session-id $identity.session_id
```

```powershell
.\scripts\start-web.ps1 -NoOpen -Matlab `
  -MatlabMcpExecutable '<absolute-server.exe>' `
  -SimulinkToolkitRoot '<absolute-toolkit-root>' `
  -MatlabSessionAppData '<absolute-session-directory>\appdata' `
  -MatlabResourceDirectory '<one-shared-absolute-resource-directory>'
```

可把四个值分别放在 `CONTROL_MATLAB_MCP_EXECUTABLE`、`CONTROL_SIMULINK_TOOLKIT_ROOT`、`CONTROL_MATLAB_SESSION_APPDATA`、`CONTROL_MATLAB_RESOURCE_DIRECTORY` 的当前进程环境中。Web 启动前只检查私有连接文件是否存在，并核对公开身份文件和操作系统进程。启动核验通过 Harness 原生工具取得实际 PID、UUID、版本和算术结果，逐项比对绑定记录。`-AdditionalPatch` 供本机额外配置使用，`-ValidateOnly` 不启动 Web。

`existing` 的 args 仅包含 `--matlab-session-mode=existing`、`--extension-file=<toolkit>/tools/tools.json`、`--disable-telemetry=true`；不混用 `matlab-root`、`initial-working-folder`、`matlab-display-mode`。MCP 只在 Host 插入一份，三个角色继承访问；原 bundle 的 `control-engineering` 行通过 overlay 增加 `matlabResource`，与 batch 共用同一个目录，`identityFile` 自动指向该专用身份文件。冲突的全局 `MW_MCP_SERVER_*` 会话选择环境参数会在启动前被拒绝而不打印其值。

`sessionDetails.json` 含连接凭据。请将会话目录放在本机已被 Git 忽略的私有目录中，避免在报告、下载包或日志里包含该文件。取消、超时或断连后，Host 自动核对原调用的终态。核验期间仍可在原会话查看日志、源码和已有文件。进程归属通过项目专用实例的身份记录核对。

固定Harness的patch会替换该行的`config`对象。因此MATLAB overlay在同一个`control-engineering`行显式保留原有pythonPath/codeRoot/worker数量/超时/取消宽限字段，再加入matlabResource；不是再插入第二个bundle，也不假定深度合并。部署若调整这些公共字段，应同步核对overlay中的生效值。

每个 MATLAB 工具调用先经过原生权限检查，再进入共享 FIFO。持锁后，由 Host 固定代码通过同一个原生工具执行器调用 `matlab/control_harness_terminal_probe.m`，核对专用会话 UUID、实际 PID、辅助函数位置及 MATLAB 管理的忙状态；主调用结束后再次检查。内部探针同样经过原生权限检查，不能由模型参数或普通工具输出中的 `idle=true` 代替。正常同步调用自动释放资源。

探针检查已加载 Simulink 模型的运行状态、本地 background/parallel futures 和正在运行的 timers。脱离 MATLAB 进程的 OS 或远程计算需要由拥有作业句柄的适配器确认终态。

原生 `evaluate_matlab_code` 会在派发前识别自包含的同步数值计算，包括矩阵运算和常用控制系统函数。对于这些计算，MATLAB 在函数中捕获 `MException`，返回标识、原始消息和本次调用的回执。Host 在同一资源锁内核实前后两次会话身份及空闲状态，再把可重试错误交回对话。例如，对连续状态空间模型误用 `d2d` 后，可以在同一 MATLAB 进程中改用 `c2d`。已经执行的赋值仍可能存在。文件 I/O、后台作业及其他不能确认为同步数值计算的代码继续使用原生执行路径。

原生 MATLAB 代码在调用者的基础工作区执行。Host 为调用分配唯一标记，MATLAB 在代码返回或抛出异常时记录终态。若库块不存在、参数或维度错误，智能体会收到原始诊断。在同一执行锁内核实进程来源和空闲状态后，可以读回模型并在原对话中修复失败步骤。之前已创建的块、参数或文件可能仍然存在。[SLX 保存与仿真验收](matlab-model-save.md)

超时、取消或连接中断后，未取得终态的调用保存在共享记录中。对于带 MATLAB 完成标记的调用，Host 核实原标记、进程身份和空闲状态。对于通过同步 MCP 工具派发的调用，Host 也能使用原连接和 JSON-RPC 请求 ID 匹配到的工具回执，再独立核验专用实例。证据落盘后，即使替换了 Host 也能继续核验。核验通过即恢复 FIFO，并将证据归档。其他文件工具仍按原权限执行，已取消的请求保持取消。

旧记录可能缺少调用标记，彻底中断的连接也可能丢失唯一的工具回执。这时相同 PID 的一次空闲观测不足以确认上一条写操作已经终止。原实例可保留供现场核验，用户可继续查看工作区。保存未完成的工作并正常退出该实例后，启动器可依据操作系统的退出证明安全重绑。

已关联请求的同步 MCP 工具到达调用者等待时限时，Host 立即返回超时，并继续观察同一条已派发请求的回复。用户主动取消仍会通知 MCP 服务器。其他角色的 MATLAB 请求按 FIFO 等待，调用方仍可取消尚未派发的请求。收到旧请求的迟到回执只用于核对原执行状态。

双击 `Start-ControlHarness.cmd` 时，如果上一次的专用 MATLAB 已退出，启动器会用当前隔离记录请求资源管理器恢复。资源管理器核对本机进程创建身份并取得执行锁后，把退出证据写入 `recoveries/`。启动器随后重新读取状态，确认资源可用才继续。普通用户无需复制隔离标记。

如果网页还在运行，重复双击会同时检查网页绑定的 MATLAB。已更新的网页 Host 会等待当前对话轮次、待处理消息、后台作业和 MATLAB 资源队列空闲，再有序退出。原启动器随后在同一端口绑定新的专用会话，实际调用 MCP 核对 PID、会话 UUID 和 MATLAB 计算结果后才开放网页。仍使用早期版本的 Host 需要在空闲时完成一次切换。

对于无法关联原调用终态的旧记录，可在确认专用实例归属后保存模型并正常退出。如果进程身份无法确认、记录不完整或仍有任务，资源继续隔离。可以用原启动账户检查本机状态。需要手动诊断时使用以下命令：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m control_harness.matlab_resource inspect --directory '<shared-resource-directory>'
.\.venv\Scripts\python.exe -m control_harness.matlab_resource recover-after-session-exit `
  --directory '<shared-resource-directory>' --token '<current-quarantine-token>'
```

Windows 发出终止命令与内核确认退出可能有短暂间隔；恢复命令仍拒绝时，继续检查同一进程和隔离记录，不删除锁文件。成功后用新的会话目录启动专用 MATLAB，再由启动脚本绑定；历史恢复证明保存在共享目录的 `recoveries/` 中。

核查依据：官方 [v0.16.0 README](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/README.md)、[Windows discovery APPDATA 路径](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/internal/adaptors/matlabmanager/sessionselector/sessiondiscovery/appdatadir/appdatadir.go)、[会话选择](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/internal/adaptors/matlabmanager/sessionselector/sessionselector.go)；安装的 Toolbox `+mcpserver/+internal/shareMATLABSession.m` 与 `+appdata/DefaultAppDataLocator.m`。本轮 S02 已实测 Harness 独立连接、Toolkit 操作与共享资源锁；S13 又完成真实 SDK 超时、隔离、进程退出证明、替换专用会话和正常生产服务恢复。实际版本、句柄与边界以[验收记录](acceptance.md)及对应阶段原始证据为准。
