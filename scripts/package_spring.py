"""Package an executed spring demo as a source-traceable, slide-ready local bundle."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import zipfile
from pathlib import Path


def build(workspace):
    workspace = workspace.resolve()
    root = Path(__file__).resolve().parents[1]
    pack = workspace / "presentation"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    metrics, run = read(pack / "metrics-summary.json"), read(workspace / "case-run.json")
    index = read(pack / "figures-index.json")
    figures = index["figures"] if isinstance(index, dict) else index
    configs = pack / "reproduce" / "case-config"
    configs.mkdir(parents=True, exist_ok=True)
    for source in (root / "examples" / "spring-damper").iterdir():
        if source.is_file():
            shutil.copy2(source, configs / source.name)
    for name in ("run-spring-dialogue.mjs", "harness-rpc.mjs", "summarize_spring.py", "plot_spring.py", "package_spring.py"):
        shutil.copy2(root / "scripts" / name, pack / "reproduce" / name)
    shutil.copy2(workspace / "case-run.json", pack / "execution-record.json")
    for name in ("01-model", "02-controller", "03-lightweight"):
        shutil.copy2(workspace / f"{name}-workflow.json", pack / "reproduce" / f"{name}-workflow.json")

    one = {(row["method"], row["state"]): row["rmse"] for row in metrics["model_one_step_test_rmse"]}
    states = sorted({row["state"] for row in metrics["model_one_step_test_rmse"]})
    rmse_lines = [f"| {state} | {one['nominal',state]:.6g} | {one['corrected',state]:.6g} | {(1-one['corrected',state]/one['nominal',state])*100:.2f}% |" for state in states]
    controller = metrics["controller"]
    base_score, candidate_score = controller["verification_baseline_score"], controller["verification_candidate_score"]
    verification = controller['verification_metrics']
    verification_lines = [f"| {row['method']} | {row['rmse']:.6f} | {row['input_total_variation']:.4f} | {row['control_effort']:.6f} |" for row in verification]
    final_test_lines = [f"| {row['method']} | {row['rmse']:.6f} | {row['input_total_variation']:.4f} | {row['max_state_violation']:.3g} / {row['max_input_violation']:.3g} / {row['max_output_violation']:.3g} |" for row in controller['final_test_metrics']]
    policy = metrics["policy"]["held_out_comparison"]
    with (pack/'source-data/imitation-training.csv').open(encoding='utf-8-sig',newline='') as handle:
        imitation_log=list(csv.DictReader(handle))
    best_imitation_epoch=int(min(imitation_log,key=lambda row:float(row['validation_loss']))['epoch'])
    policy_lines = []
    labels = {"expert": "MPC专家", "imitation": "模仿策略", "optimized": "物理优化策略"}
    for method in labels:
        row = policy[method]
        policy_lines.append(f"| {labels[method]} | {row['rmse']:.6f} | {row['mean_control_seconds']*1000:.4f} | {row['p95_control_seconds']*1000:.4f} | {row['max_state_violation']:.3g} / {row['max_input_violation']:.3g} / {row['max_output_violation']:.3g} |")
    outcome = "降低" if policy['optimized']['rmse'] < policy['imitation']['rmse'] else "未降低"
    policy_change = (1-policy['optimized']['rmse']/policy['imitation']['rmse'])*100
    per_scenario_lines = []
    policy_rows = metrics['policy']['held_out_per_scenario']
    for scenario in dict.fromkeys(row['scenario'] for row in policy_rows):
        paired={row['method']:row for row in policy_rows if row['scenario']==scenario}
        change=(1-paired['optimized']['rmse']/paired['imitation']['rmse'])*100
        per_scenario_lines.append(f"| {scenario} | {paired['expert']['rmse']:.6f} | {paired['imitation']['rmse']:.6f} | {paired['optimized']['rmse']:.6f} | {change:+.2f}% |")
    rows = []
    for phase in run['phases']:
        rows.append(f"| {phase['name']} | `{phase['preset']}` | `{phase['sessionId']}` | `{phase['jobId']}` | {phase['status']} |")
    chain = "\n".join(f"- {name}：`{artifact}`" for name,artifact in run['artifacts'].items())
    figure_lines=[]
    chinese_captions = {
        '01-model-free-decay': ('模型校正后的独立自由衰减', '初始位移0.6m、速度为零，零控制、零扰动，三个模型各自滚动预测80步；蓝线接近参考对象，灰线为名义模型。'),
        '02-model-heldout-rmse': ('留出测试集的单步预测误差', f"全部{metrics['transition_split']['test']['samples']}条留出转移样本，位移和速度分别用m与m/s评价；纵轴为对数轴，避免混合单位。"),
        '03-residual-learning': ('残差网络训练与验证曲线', '完整训练日志；训练目标含边界惩罚，验证为归一化预测误差，两条曲线的目标并不完全相同。'),
        '04-mpc-ga-search': ('MPC遗传调参搜索过程', f"{controller['trials']}个候选的调参得分及截至当前的最佳可行得分；这里只用调参工况，采用候选仍须通过独立复验。"),
        '05-mpc-verification': ('初始MPC与GA候选的独立复验', '在同一复验工况比较位移跟踪和控制力；候选改善了误差，也增加了控制动作。'),
        '06-mpc-final-test': ('冻结参数后的最终MPC测试', '保留工况中的初始与选定MPC对比，展示位移、速度和控制力；该工况没有用于调参。'),
        '08-policy-accuracy-and-time': ('轻量策略的精度与计算耗时', '两个相同留出工况上的配对比较；右图为本机平均控制调用耗时的对数轴，200ms虚线为采样周期参照，不能据此宣称硬实时保证。'),
        '09-policy-learning': ('从模仿专家到物理策略优化', '左为模仿训练/验证曲线，训练值为每轮最后一个小批次；右为物理优化的验证选择分数。测试集没有用于选检查点。'),
    }
    for figure in figures:
        filename=figure.get('files',{}).get('png') or figure.get('png') or figure.get('file') or f"figures/{figure.get('stem',figure.get('id',''))}.png"
        if not str(filename).startswith('figures/') and '/' not in str(filename): filename='figures/'+str(filename)
        title, caption = chinese_captions.get(figure.get('id'), (figure.get('title',Path(filename).stem),figure.get('caption','')))
        if str(figure.get('id','')).startswith('07-policy-heldout'):
            title='留出工况的专家／模仿／优化策略控制曲线'
            caption='每个留出工况各一图，全部保留；深蓝为MPC专家，橙色为模仿策略，青绿为物理优化策略。位移点含初值，控制力按零阶保持画阶梯线。'
        figure_lines.append(f"- [{title}]({filename})：{caption}\n")
    guide = f"""# 二阶弹簧阻尼系统：完整演示与 PPT 素材说明

这是一次真实执行的三角色 Harness 接力案例。数值来源是明确声明的参考仿真对象，曲线来自实际训练和闭环计算。`figures/` 提供高清 PNG、可编辑文字 SVG 和 PDF；`source-data/` 保留全部作图数据；`agent-outputs/` 保留三个真实 DeepSeek 会话的提示词、可见回复和数值作业记录。

## 1. 从被控对象开始

\\[m\\ddot q+c\\dot q+kq=u+d,\\quad x=[q,v]^T,\\quad y=q.\\]

\\[\\dot x=\\begin{{bmatrix}}0&1\\\\-k/m&-c/m\\end{{bmatrix}}x+\\begin{{bmatrix}}0\\\\1/m\\end{{bmatrix}}(u+d).\\]

| 参数 | 名义模型 | 参考对象 | 单位 |
|---|---:|---:|---|
| m | {metrics['physical_parameters']['nominal']['mass_kg']:g} | {metrics['physical_parameters']['reference']['mass_kg']:g} | kg |
| c | {metrics['physical_parameters']['nominal']['damping_N_s_per_m']:g} | {metrics['physical_parameters']['reference']['damping_N_s_per_m']:g} | N·s/m |
| k | {metrics['physical_parameters']['nominal']['stiffness_N_per_m']:g} | {metrics['physical_parameters']['reference']['stiffness_N_per_m']:g} | N/m |

采样周期 0.2 s，连续线性模型按零阶保持精确离散化。位移约束 ±1 m，速度 ±1.5 m/s，控制力 ±4 N，扰动力 ±0.2 N。没有声明额外的控制力变化率硬约束；MPC 中的 s 是控制变化软惩罚。

## 2. 三个智能体如何接力

1. **模型自动化训练**：登记两套连续模型 → 采集参考对象仿真数据 → 按轨迹分组切分 → 训练离散残差网络 → 交付校正模型。网络输入 `[q,v,u,d]`，输出下一状态残差 `[Δq,Δv]`，4 入 2 出。
2. **控制器设计与调参**：读取校正模型作预测，参考对象作独立评价 → 设计约束 MPC → GA 搜索权重 r、s → 独立复验决定是否采用候选 → 保留最终测试结果。horizon=6，预测窗口 1.2 s；动力学不参与调参。
3. **控制器轻量化**：读取选定 MPC → 采集 12 个专家工况 → 按工况切分 → 模仿学习 → 可微物理 rollout 策略优化 → 在相同留出工况比较专家与两种神经策略。策略输入 `[q,v,目标位移,d]`，输出控制力 u，4 入 1 出。

各角色使用独立原生会话，数值作业完成并校验后才向下一角色传递产物 ID。下面的会话和作业确实存在，没有模拟智能体回答。

| 阶段 | Harness角色 | 会话 | 原生作业 | 结果 |
|---|---|---|---|---|
{chr(10).join(rows)}

## 3. 本次实际结果

模型数据划分：{json.dumps(metrics['transition_split'],ensure_ascii=False)}。

独立测试单步预测 RMSE，位移与速度分别保留自己的单位（m、m/s）：

| 状态 | 名义模型 RMSE | 校正模型 RMSE | RMSE相对下降 |
|---|---:|---:|---:|
{chr(10).join(rmse_lines)}

GA 执行 {controller['trials']} 个候选；独立复验评分由 {base_score:.6f} 变为 {candidate_score:.6f}（越低越好，变化 {(1-candidate_score/base_score)*100:.2f}%）。候选被采用：**{controller['tuned_candidate_accepted']}**。初始参数 `{json.dumps(controller['baseline_parameters'])}`；最终参数 `{json.dumps(controller['chosen_parameters'])}`。

调参体现了跟踪与控制动作之间的权衡。下面保留同一复验工况的控制力变化代价，不能只展示跟踪改善；控制力总变化量为逐步绝对力增量之和，包含零初始控制，量纲为 N。control_effort 是按控制范围归一化后的平均平方力，不是物理能量。

| 复验方法 | 位移RMSE/m | 控制力总变化量/N | 归一化平方力 |
|---|---:|---:|---:|
{chr(10).join(verification_lines)}

最终测试工况的基线与选定 MPC 曲线，是参数冻结后的同工况复评，未参与调参或选择；不能把不同工况的分数相除宣称改善。

| 最终测试方法 | 位移RMSE/m | 控制力总变化量/N | 状态/输入/输出违反量 |
|---|---:|---:|---:|
{chr(10).join(final_test_lines)}

神经策略比较取全部 {policy['expert']['scenarios']} 个留出专家工况的平均 RMSE，各工况样本量一致；时间列为各工况测量值的平均，p95列是各工况p95的平均，不是把全部调用重新合并计算的p95。最后一列是全部留出工况的最大违反量。

| 方法 | 位移RMSE/m | 平均控制调用/ms | 工况p95平均/ms | 状态/输入/输出违反量 |
|---|---:|---:|---:|---|
{chr(10).join(policy_lines)}

物理策略优化在本次独立测试上**{outcome}**模仿策略的位移 RMSE，相对下降 {policy_change:.2f}%。神经策略参数量 {metrics['policy']['parameter_count']}，模仿训练 {metrics['policy']['imitation_epochs']} 轮，物理优化 {metrics['policy']['optimization_epochs']} 轮。速度比只是本机 CPU、本次执行、相同工况下的控制调用耗时比较，不是硬件无关保证。

模仿策略发布的是验证损失最小的第 **{best_imitation_epoch}** 轮权重，而非训练结束时的最后一轮权重。训练曲线后段的验证误差上升说明出现过拟合趋势；早停和最佳验证检查点选择保留了较早的权重，测试集不参与该选择。

逐工况结果如下。最后一列正值表示物理优化相对模仿策略的 RMSE 降低；负值表示该工况退化。平均改善不能解释为每个工况都改善，神经策略与MPC专家的精度差距也应保留在演示结论中。

| 留出工况 | MPC专家RMSE/m | 模仿策略RMSE/m | 优化策略RMSE/m | 优化相对模仿的RMSE下降 |
|---|---:|---:|---:|---:|
{chr(10).join(per_scenario_lines)}

## 4. 建议的 12 页 PPT 讲解顺序

| 页 | 主题 | 建议素材与讲解重点 |
|---:|---|---|
| 1 | 为什么需要三个智能体 | 从“模型不准、MPC要调、在线求解耗时”引出任务 |
| 2 | 二阶被控对象 | 本文的连续方程、参数对照表、状态/输入/输出单位 |
| 3 | 三角色接力 | 第2节；用校正模型ID→控制器ID→策略ID说明文件可追踪 |
| 4 | 模型采集与训练 | 实际样本数、按轨迹切分、残差训练曲线与单步误差 |
| 5 | 校正后的预测 | 三模型自由衰减曲线；这是独立多步滚动，不能与单步RMSE混淆 |
| 6 | MPC设计 | horizon、约束、归一化权重及固定参数范围 |
| 7 | GA调参与独立复验 | GA候选/累计最佳曲线、复验位移/力曲线，说明为何采用或保留 |
| 8 | 最终MPC控制效果 | reserved final test 的基线/选定MPC位移、速度和力 |
| 9 | 专家数据与模仿学习 | 12个工况、8/2/2划分、4入1出神经策略及训练曲线 |
| 10 | 神经策略控制效果 | 两个留出工况都展示；专家/模仿/物理优化同工况曲线 |
| 11 | 轻量化的收益与代价 | 配对RMSE、控制调用耗时；说明效果差距和本机计时范围 |
| 12 | 智能体执行证据与结论 | `agent-outputs/`中的真实交接回复、产物链、下一步实测接入 |

讲解时优先插入 PNG（约4000像素宽），需要编辑或放大时使用 SVG；PDF 用于保留矢量原稿。图中坐标为英文，单位均标明，中文图注可使用下列索引。

## 5. 曲线索引

{chr(10).join(figure_lines)}

## 6. 如何自己复现

在项目根目录启动 Harness 并配置模型服务后，换一个全新的工作目录执行：

```powershell
node scripts/run-spring-dialogue.mjs workspaces/spring-damper-my-run
$env:PYTHONPATH="$PWD\\src"
.\\.venv\\Scripts\\python.exe scripts/summarize_spring.py workspaces/spring-damper-my-run
.\\.venv\\Scripts\\python.exe scripts/plot_spring.py workspaces/spring-damper-my-run
.\\.venv\\Scripts\\python.exe scripts/package_spring.py workspaces/spring-damper-my-run
```

亦可在“控制工程”界面用 `examples/spring-damper/workflow.json` 预检后运行完整八阶段，或按 `reproduce/case-config/README.md` 逐角色使用表单执行。界面路径可复现数值流程；上述素材导出脚本则以真实三会话脚本生成的 `case-run.json` 为入口，不直接接受只有UI工作流记录的目录。对话脚本需要本机正在运行的 Harness 和有效模型服务；图形渲染需要 Matplotlib，PDF质量审计需要 PyMuPDF，使用 nature-figure 的本地审计工具。精确运行记录、版本和时间范围见 `metrics-summary.json`、`execution-record.json` 和 `qa/`。图形素材包不包含 API Key、本地认证令牌或训练权重；完整权重与检查点保留在原工作目录的 `artifacts/` 中。

## 7. 展示时准确说明的边界

- 本次是线性二阶参考仿真案例、固定种子演示；没有物理台架数据，也没有多随机种子统计结论。
- 物理信息体现在动力学骨架、下一状态残差及边界惩罚；当前没有训练连续 ODE/PDE 方程残差，不能称作完整方程残差 PINN。
- 策略优化是可微物理 rollout 策略梯度，没有学习 Critic，不能标为 PPO 或 SAC。
- 目标和扰动在每个核心工况内恒定；完整状态和扰动可观测；没有验证未观测扰动、状态估计误差或实物部署的鲁棒性。
- 测试结果未用于训练、调参或检查点选择；后续复评与绘图没有改动已选择参数。图中未添加平滑曲线、虚构置信区间或显著性标记。

## 8. 本次产物链

{chain}
"""
    (pack / "PPT素材与完整案例说明.md").write_text(guide, encoding="utf-8")
    review_notes = ''
    if run.get('review'):
        review_notes = (
            '本次控制器原始回复中的“控制能量”对应control_effort，实际是按控制范围归一化后的平均平方力，不能标成焦耳或物理能量。\n\n'
            '轻量化原始回复把专家耗时秒误写成毫秒：专家约0.88–0.91秒，即880–910毫秒；神经策略约0.3–0.5毫秒。原回复中p95<1.5毫秒也不能用于MPC专家。PPT请采用程序按秒转毫秒后的表格与曲线。单种子不能推断统计显著性或判断增益属于随机噪声。\n\n'
            '03-lightweight-corrected-summary.txt 是同一真实会话后续生成的耗时更正版。其中“速率检查不通过”仍须更正为“速率约束未声明、检查不适用”，并不表示约束违规；其精度差27.7%与16.0%对应模仿策略相对MPC，不能用于优化策略。量化演示以本包程序生成的指标表为准。\n\n')
    (pack / 'agent-outputs' / '阅读说明.md').write_text(
        '# 真实智能体输出的阅读说明\n\n'
        '每个角色的 prompt.txt 是实际提交的提示词，assistant.txt 是会话的可见回复，attempts.json 是该会话的数值工具记录。没有导出模型思考过程、凭据或原生认证令牌。\n\n'
        '原始回复保持原文，演示中的最终指标及量纲以 metrics-summary.json 和中文素材说明为准。\n\n'
        + review_notes +
        '通用 prepare_handoff 说明可能列出下一阶段还需要工况配置；本次案例已在独立流程文件中提供这些配置，随后角色按文件执行，因此不是等待用户补充的未完成任务。\n\n'
        'reproduce/ 下的三个阶段流程和 execution-record.json 记录了本次真实输入、输出ID与作业接力关系。\n', encoding='utf-8')
    cards=[]
    for figure in figures:
        filename=figure.get('files',{}).get('png') or figure.get('png') or figure.get('file') or f"figures/{figure.get('stem',figure.get('id',''))}.png"
        if not str(filename).startswith('figures/') and '/' not in str(filename): filename='figures/'+str(filename)
        cards.append(f'<section><h2>{html.escape(figure.get("title",Path(filename).stem))}</h2><img src="{html.escape(filename)}"><p>{html.escape(figure.get("caption",""))}</p></section>')
    (pack / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>弹簧阻尼三智能体演示素材</title><style>body{max-width:1250px;margin:40px auto;font:18px/1.7 "Microsoft YaHei",sans-serif;color:#243349;background:#f2f5fa}section{padding:24px;background:white;margin:28px 0;border-radius:16px}img{width:100%}h1{line-height:1.3}a{color:#1763ad}</style><h1>二阶弹簧阻尼系统 · 三智能体完整接力</h1><p>实际参考仿真、训练与闭环结果。<a href="PPT素材与完整案例说明.md">完整中文说明与12页PPT大纲</a> · <a href="metrics-summary.json">真实指标</a></p>'+''.join(cards)+'</html>',encoding='utf-8')
    hashes={str(p.relative_to(pack)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pack.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    (pack / 'SHA256SUMS.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=workspace/'spring-damper-PPT-materials.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
        for file in sorted(pack.rglob('*')):
            if file.is_file(): target.write(file,Path('spring-damper-PPT-materials')/file.relative_to(pack))
    print(json.dumps({'archive':str(archive),'files':len(hashes),'size_bytes':archive.stat().st_size},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    build(parser.parse_args().workspace)
