# MATLAB 与 Simulink 的可选本机连接

当前接法针对 Windows：MATLAB R2024b、官方 MCP Server 0.16.0、Simulink Agentic Toolkit 2026.11.a。复用已有安装，不安装 Engine，不升级，也不改用户 startup.m。

先显式启动一份本项目拥有的专用 MATLAB，再让 Harness 的唯一 MCP client 使用 `existing` 连接该进程。`start-web.ps1` 自身不启动 MATLAB；没有 `-Matlab` 时不加载 MATLAB overlay，锁配置保持未启用。

官方默认 `existing` 使用最近执行 `shareMATLABSession` 的用户共享文件，不能据此保证选中工程会话。本接法只在专用 MATLAB 运行 `satk_initialize` 时临时设置该进程的 `APPDATA`，把共享文件写入私有会话目录，然后恢复环境；MCP 子进程使用同一私有 `APPDATA`。不改用户全局发现文件，不按名称猜选会话，不连接或关闭其他 MATLAB。

本机路径示例（换机须替换为实际绝对路径，路径不写死在 overlay）：

```powershell
.\scripts\start-matlab-session.ps1 `
  -MatlabExecutable 'C:\APP\MATLAB2024B\R2024b\bin\matlab.exe' `
  -MatlabMcpExecutable 'C:\Users\sk3547969879\.matlab\agentic-toolkits\bin\matlab-mcp-server.exe' `
  -SimulinkToolkitRoot 'C:\Users\sk3547969879\.matlab\agentic-toolkits\simulink' `
  -MatlabMcpToolboxRoot 'C:\Users\sk3547969879\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\MATLAB MCP Server Toolbox' `
  -SessionDirectory 'D:\projects\control-agent-harness\.dsh\matlab\session-UNIQUE'
```

`-ValidateOnly` 只核对安装路径，不启动进程。实际启动后保存 PID 与日志，等待 `appdata/control-harness-session.json` 的 `ready` 标记；未完成时检查所记录进程，不重复启动。MATLAB 新进程的用户 startup 仍可能自行运行，应在当前受信环境核对；本项目不替用户禁用或改写它。

此公开身份文件不含连接凭据，字段为：`schema_version=1`、`kind="control-harness-dedicated-matlab-v1"`、`status="ready"`、随机 `session_id`、实际 `feature('getpid')` 的 `matlab_pid`（兼容别名 `pid`）、UTC ISO时间 `createdAt`、`project_root`、`private_appdata`、`matlab_release`、`toolkit_root`。启动器的 `launch.json` 记录启动句柄PID；绑定及回收以就绪身份文件中的实际 `matlab_pid` 为准。MATLAB同时保留 `CONTROL_HARNESS_SESSION_ID` 环境标记，并加载项目 `matlab/` 供原生身份及空闲探针使用。

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

可把四个值分别放在 `CONTROL_MATLAB_MCP_EXECUTABLE`、`CONTROL_SIMULINK_TOOLKIT_ROOT`、`CONTROL_MATLAB_SESSION_APPDATA`、`CONTROL_MATLAB_RESOURCE_DIRECTORY` 的当前进程环境中。Web 启动前核对工程 marker 与连接文件的 PID，并确认该 PID 仍为 MATLAB。PID、版本和算术结果还须由真实 Harness 工具调用复核；静态配置不算连通验收。`-AdditionalPatch` 供本机额外配置使用，`-ValidateOnly` 不启动 Web。

`existing` 的 args 仅包含 `--matlab-session-mode=existing`、`--extension-file=<toolkit>/tools/tools.json`、`--disable-telemetry=true`；不混用 `matlab-root`、`initial-working-folder`、`matlab-display-mode`。MCP 只在 Host 插入一份，三个角色继承访问；原 bundle 的 `control-engineering` 行通过 overlay 增加 `matlabResource`，与 batch 共用同一个目录，`identityFile` 自动指向该专用身份文件。冲突的全局 `MW_MCP_SERVER_*` 会话选择环境参数会在启动前被拒绝而不打印其值。

`sessionDetails.json` 含连接凭据，不输出、不提交、不打入报告或下载包。会话目录应放在本机 ignored 私有目录中。取消、超时、连接丢失不证明 MATLAB 已停止；资源锁的隔离状态及显式恢复由共享锁实现处理。只处理本项目记录且已核实归属的进程，不按进程名批量关闭 MATLAB。

固定Harness的patch会替换该行的`config`对象。因此MATLAB overlay在同一个`control-engineering`行显式保留原有pythonPath/codeRoot/worker数量/超时/取消宽限字段，再加入matlabResource；不是再插入第二个bundle，也不假定深度合并。部署若调整这些公共字段，应同步核对overlay中的生效值。

每个 MATLAB 工具调用先经过原生权限检查，再进入共享 FIFO。持锁后，由 Host 固定代码通过同一个原生工具执行器调用 `matlab/control_harness_terminal_probe.m`，核对专用会话 UUID、实际 PID、辅助函数位置及 MATLAB 管理的忙状态；主调用结束后再次检查。内部探针同样经过原生权限检查，不能由模型参数或普通工具输出中的 `idle=true` 代替。正常同步调用自动释放资源。

探针覆盖已加载 Simulink 模型的运行状态、本地 background/parallel futures 和正在运行的 timers；它不是任意脱离进程的 OS/远程计算的停止证明。当前适配器面向同步 MATLAB/Simulink 批任务；需要异步数值作业时，应由拥有该作业句柄的可信适配器管理终态，不能把启动成功作为完成。取消、超时、传输失败或无法取得可信终态时，资源持久隔离，后续请求会被拒绝。本版本对 MCP 错误也采取该保守策略。

恢复属于本机管理操作，没有提供给 LLM 的解锁工具。先检查共享目录的 `resource.json`，核实隔离记录及其专用 PID 的归属；只停止本项目拥有的专用 MATLAB，不按进程名批量终止。随后执行下列命令；它会核对绑定进程的 OS 创建身份和真实退出状态，活进程或未知状态都不能恢复：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m control_harness.matlab_resource inspect --directory '<shared-resource-directory>'
.\.venv\Scripts\python.exe -m control_harness.matlab_resource recover-after-session-exit `
  --directory '<shared-resource-directory>' --token '<current-quarantine-token>'
```

Windows 发出终止命令与内核确认退出可能有短暂间隔；恢复命令仍拒绝时，继续检查同一进程和隔离记录，不删除锁文件。成功后用新的会话目录启动专用 MATLAB，再由启动脚本绑定；历史恢复证明保存在共享目录的 `recoveries/` 中。

核查依据：官方 [v0.16.0 README](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/README.md)、[Windows discovery APPDATA 路径](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/internal/adaptors/matlabmanager/sessionselector/sessiondiscovery/appdatadir/appdatadir.go)、[会话选择](https://github.com/matlab/matlab-mcp-server/blob/v0.16.0/internal/adaptors/matlabmanager/sessionselector/sessionselector.go)；安装的 Toolbox `+mcpserver/+internal/shareMATLABSession.m` 与 `+appdata/DefaultAppDataLocator.m`。本轮 S02 已实测 Harness 独立连接、Toolkit 操作与共享资源锁；S13 又完成真实 SDK 超时、隔离、进程退出证明、替换专用会话和正常生产服务恢复。实际版本、句柄与边界以[验收记录](acceptance.md)及对应阶段原始证据为准。
