# 技能库快照与历史对照复评

插件内的 [业务资产清单](../harness-plugin/control-skill-library/snapshot.json) 记录源库 Git 基线和工作树文件的 SHA-256，包含该版本未提交的 MATLAB 场景。运行记录中的版本、耗时、源机路径和数值属于源库当时的实验。在维护端可从 harness 根目录校验或更新快照。

```powershell
python scripts/sync-control-library.py --check
python scripts/sync-control-library.py --source ../control-skill-library --check
python scripts/sync-control-library.py --source ../control-skill-library
```

第一条只用已安装的文件。第二条逐项比对给定源库的当前工作树。第三条更新已登记的内容，保留来源指纹，并在目标存在手工修改或额外文件时报告冲突。维护时改源库后重新同步。试跑或改写案例时使用独立的工作副本。外部来源、MATLAB 配套输入和本地研究文件分别见 [源材料目录](../harness-plugin/control-skill-library/sources.json)、[场景输入清单](../harness-plugin/control-skill-library/cases/matlab-scenes/inputs.json) 和 [来源说明](../THIRD_PARTY_NOTICES.md)。

历史评测保存在 [独立复评目录](../tests/fixtures/control-library-evaluation/evaluation/README.md)。其评分器、选手源码、试验用库快照和评分回执仅供维护端使用。安装插件时交付业务资产。实施新的盲测时，为选手提供单独的任务工作区，在维护端执行隐藏评分。

以下 PowerShell 命令从 harness 根目录运行。每次创建新的临时工作副本，使用 harness 的 Python 环境和原始评分入口复算。依赖版本见 [案例依赖文件](../harness-plugin/control-skill-library/requirements.txt)。这三项评分只需 Python、NumPy、SciPy 和 PyTorch。

```powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
$trial = Join-Path ([IO.Path]::GetTempPath()) ("control-library-review-" + [guid]::NewGuid().ToString('N'))
Copy-Item -Recurse -LiteralPath tests/fixtures/control-library-evaluation -Destination $trial
Push-Location -LiteralPath $trial
try {
    $env:PYTHONUTF8 = '1'
    & $py -B evaluation/paired/regrade.py
    if ($LASTEXITCODE) { throw '三题首交评分失败' }
    & $py -B evaluation/paired/test_reach.py
    if ($LASTEXITCODE) { throw '故障注入复评失败' }
    & $py -B evaluation/valve/trials.py score
    if ($LASTEXITCODE) { throw '阀门 MPC 复评失败' }
} finally { Pop-Location }
```

同一组建模、预测与轻量化题目中，无库组、可查库组、显式调用技能组均有 3 个独立会话，每组首次提交均通过 9/9 个子任务。在单个阀门 MPC 交付中，无库组和显式调用组各有两次独立会话，均通过 2/2 次。以上数值是冻结旧提交的复评分，当前两种 harness 的差异还需用新的任务和会话测量。
