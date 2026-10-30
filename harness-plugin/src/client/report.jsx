import React, { useState } from 'react';

const colors = ['#259a8c', '#b47535', '#6977c9', '#bd6084'];
const number = value => Number.isFinite(value) ? (Math.abs(value) < .001 && value !== 0 ? value.toExponential(3) : value.toPrecision(5)) : '—';

function Chart({ title, series, xLabel }) {
  const valid = series.map(line => ({ ...line, points: line.points.filter(([x,y]) => Number.isFinite(x) && Number.isFinite(y)) })).filter(line => line.points.length);
  if (!valid.length) return null;
  const all = valid.flatMap(line => line.points), xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  return <figure className="control-chart"><figcaption>{title}</figcaption><svg viewBox="0 0 440 175" role="img" aria-label={title}>
    <path d="M55 12v125h370" fill="none" stroke="currentColor" opacity=".3"/>
    {valid.map((line,index) => <polyline key={line.label} points={line.points.map(([x,y]) => `${55+(x-xmin)/(xmax-xmin || 1)*370},${137-(y-ymin)/(ymax-ymin || 1)*125}`).join(' ')} fill="none" stroke={colors[index%colors.length]} strokeWidth="2" strokeDasharray={line.dashed ? '5 4' : undefined}/>)}
    <g fill="currentColor" fontSize="9"><text x="2" y="19">{number(ymax)}</text><text x="2" y="138">{number(ymin)}</text><text x="55" y="152">{number(xmin)}</text><text x="425" y="152" textAnchor="end">{number(xmax)}</text><text x="240" y="170" textAnchor="middle">{xLabel}</text></g>
  </svg><div className="control-legend">{valid.map((line,index) => <span key={line.label} style={{ color: colors[index%colors.length] }}>{line.label}</span>)}</div></figure>;
}

function Metrics({ runs }) {
  return <div className="control-table"><table><thead><tr><th>策略 / 工况</th><th>跟踪 RMSE</th><th>平均耗时 ms</th><th>控制总变差</th><th>最大约束越界</th></tr></thead><tbody>{runs.map(({ label, run },index) => {
    const m = run.metrics ?? {}, checks = run.constraint_checks ?? {};
    const violations = ['state','input','output','input_rate'].filter(key => checks[key] === true).map(key => m[key === 'input_rate' ? 'max_rate_violation' : `max_${key}_violation`]);
    return <tr key={index}><td>{label}</td><td>{number(m.rmse)}</td><td>{number(Number.isFinite(m.mean_control_seconds) ? m.mean_control_seconds * 1000 : undefined)}</td><td>{number(m.input_total_variation)}</td><td>{violations.length && violations.every(Number.isFinite) ? number(Math.max(...violations)) : '未报告'}<small>{Object.keys(checks).filter(key => checks[key]).join(' / ')}</small></td></tr>;
  })}</tbody></table></div>;
}

/** Only renders persisted evaluation values. No curves or metrics are synthesized. */
export function ArtifactReport({ report, modelInterface }) {
  const [selection, setSelection] = useState(0);
  const comparisons = report.comparison;
  const runs = report.test?.runs ?? report.closed_loop?.test ?? [];
  const conditions = comparisons ?? runs;
  const index = Math.min(selection, Math.max(0, conditions.length - 1));
  const current = conditions[index];
  const traces = current ? comparisons ? ['expert','imitation','optimized'].filter(key => current[key]).map(key => ({ label: {expert:'专家 MPC',imitation:'模仿策略',optimized:'优化策略'}[key], run: current[key] })) : [{ label: current.scenario?.name ?? `工况 ${index+1}`, run: current }] : [];
  const dt = modelInterface?.sample_time;
  const log = report.log ?? [];
  const logSeries = ['training_loss','validation_loss','validation_score'].filter(key => log.some(row => Number.isFinite(row[key]))).map(key => ({ label: {training_loss:'训练损失',validation_loss:'验证损失',validation_score:'验证评分'}[key], points: log.map(row => [row.epoch, row[key]]) }));
  return <section aria-label="验证结果图表"><h3>验证结果</h3><small>{report.algorithm}</small>
    {report.test?.nominal_mse !== undefined && <div className="control-table"><table><thead><tr><th>数据划分</th><th>名义模型 MSE</th><th>校正模型 MSE</th><th>样本数</th></tr></thead><tbody>{['train','validation','test'].filter(key => report[key]).map(key => <tr key={key}><td>{key}</td><td>{number(report[key].nominal_mse)}</td><td>{number(report[key].corrected_mse)}</td><td>{report[key].samples}</td></tr>)}</tbody></table><small>独立测试误差变化：{report.test.nominal_mse > 0 ? `${((1-report.test.corrected_mse/report.test.nominal_mse)*100).toFixed(2)}%` : '名义误差为零，无相对变化值'}</small></div>}
    <Chart title="训练与验证曲线" series={logSeries} xLabel="训练轮次"/>
    {report.applicability && <details><summary>训练覆盖与适用范围</summary><p>训练样本 {report.applicability.samples} 条，采样周期 {report.applicability.sample_time} 秒。逐变量范围描述训练暴露，不证明联合工况覆盖或外推有效性。</p><div className="control-table"><table><thead><tr><th>变量 / 单位</th><th>训练最小值</th><th>训练最大值</th><th>声明边界</th></tr></thead><tbody>{report.applicability.marginal_ranges.map(row => <tr key={`${row.group}:${row.name}`}><td>{row.group} / {row.name} · {row.unit}</td><td>{number(row.observed_min)}</td><td>{number(row.observed_max)}</td><td>[{number(row.declared_lower)}, {number(row.declared_upper)}]</td></tr>)}</tbody></table></div></details>}
    {report.trials?.length > 0 && <Chart title="调参候选实际评分（越低越好）" series={[{label:'候选评分',points:report.trials.map((row,i) => [i+1,row.score])}]} xLabel="候选序号"/>}
    {typeof report.tuned_candidate_accepted === 'boolean' && <p>{report.tuned_candidate_accepted ? '独立复验接受调参候选。' : '独立复验保留基线控制器。'}</p>}
    {conditions.length > 0 && <><label>独立测试工况<select value={index} onChange={event => setSelection(Number(event.target.value))}>{conditions.map((row,i) => <option key={i} value={i}>{row.scenario?.name ?? `工况 ${i+1}`}</option>)}</select></label><Metrics runs={traces}/>
      {(modelInterface?.outputs ?? []).map((variable,channel) => <Chart key={variable.name} title={`${variable.name} · ${variable.unit}`} xLabel={dt ? '时间 / 秒' : '离散步'} series={[
        ...traces.map(({label,run}) => ({label,points:(run.trajectory ?? []).map(row => [(row.step+1)*(dt ?? 1),row.output?.[channel]])})),
        {label:'目标',dashed:true,points:(traces[0]?.run.trajectory ?? []).map(row => [(row.step+1)*(dt ?? 1),row.reference?.[channel]])},
      ]}/>)}
      {(modelInterface?.controls ?? []).map((variable,channel) => <Chart key={variable.name} title={`控制 ${variable.name} · ${variable.unit}`} xLabel={dt ? '时间 / 秒' : '离散步'} series={traces.map(({label,run}) => ({label,points:(run.trajectory ?? []).map(row => [row.step*(dt ?? 1),row.u?.[channel]])}))}/>)}</>}
    {report.display_summary && <p className="control-preview-note">图表采用有界报告预览，长轨迹可能已抽样；指标使用原报告值，完整数据保存在 report.json。{report.display_summary.omitted}</p>}
  </section>;
}
