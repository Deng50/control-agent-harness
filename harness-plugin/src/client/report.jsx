import React, { useEffect, useMemo, useRef, useState } from 'react';
import { normalizeReport, arrayCharts, splitSegments, formatValue, finite } from './reportmodel.mjs';
import './report.css';

const colors=['#147d73','#b36b22','#606abe','#b4507b','#66872d'];
function Chart({title,series=[],xLabel}) {
  const lines=series.map(line=>({...line,segments:splitSegments(line.points??[])})).filter(line=>line.segments.length);
  if(!lines.length)return null;
  const points=lines.flatMap(line=>line.segments.flat());const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const position=([x,y])=>`${56+(x-xmin)/(xmax-xmin||1)*368},${138-(y-ymin)/(ymax-ymin||1)*124}`;
  return <figure className="report-chart"><figcaption>{title}</figcaption><svg viewBox="0 0 440 180" role="img" aria-label={title}>
    <path d="M56 12v126h368" fill="none" stroke="currentColor" opacity=".25"/>
    {lines.flatMap((line,index)=>line.segments.map((segment,k)=>segment.length===1?<circle key={`${index}:${k}`} cx={position(segment[0]).split(',')[0]} cy={position(segment[0]).split(',')[1]} r="2.5" fill={colors[index%colors.length]}/>:
      <polyline key={`${index}:${k}`} points={segment.map(position).join(' ')} fill="none" stroke={colors[index%colors.length]} strokeWidth="2" strokeDasharray={line.dashed?'5 4':undefined}/>))}
    <g fill="currentColor" fontSize="9"><text x="2" y="19">{formatValue(ymax)}</text><text x="2" y="140">{formatValue(ymin)}</text><text x="56" y="154">{formatValue(xmin)}</text><text x="424" y="154" textAnchor="end">{formatValue(xmax)}</text><text x="240" y="174" textAnchor="middle">{xLabel}</text></g>
  </svg><div className="report-legend">{lines.map((line,index)=><span key={index} style={{color:colors[index%colors.length]}}>{line.label}</span>)}</div></figure>;
}
function Values({rows}) {
  if(!rows?.length)return null;
  return <div className="report-table"><table><thead><tr><th>项目 / 实际来源</th><th>保存值</th></tr></thead><tbody>{rows.map((row,index)=><tr key={index}><th>{row.label}</th><td>{formatValue(row.value)}</td></tr>)}</tbody></table></div>;
}
function MetricDetails({items}) {
  if(!items.length)return null;
  const keys=['rmse','rmse_K','nrmse','overshoot_ratio','settling_time','oscillation_crossings','energy_proxy','tv','normalized_physics_rms','normalized_initial_abs_error','normalized_initial_max_abs_error'];
  const columns=keys.filter(key=>items.some(item=>Object.hasOwn(item.row,key)));if(!columns.length)return null;
  const names={rmse:'RMSE',rmse_K:'RMSE / K',nrmse:'NRMSE',overshoot_ratio:'超调比',settling_time:'调节时间 / s',oscillation_crossings:'振荡过零次数',energy_proxy:'动作平方积分',tv:'动作总变差',normalized_physics_rms:'物理残差 RMS',normalized_initial_abs_error:'初值误差',normalized_initial_max_abs_error:'初值最大误差'};
  return <details><summary>逐输出 / 动作 / 条件的完整报告指标</summary><div className="report-table"><table><thead><tr><th>划分 / 变量 / 单位</th>{columns.map(key=><th key={key}>{names[key]}</th>)}</tr></thead><tbody>{items.map(({label,row},index)=><tr key={index}><th>{label}</th>{columns.map(key=><td key={key}>{formatValue(row[key])}</td>)}</tr>)}</tbody></table></div><p className="report-note">未报告或不适用显示为空值；多输出指标不合并为一个假定同单位的量。</p></details>;
}
function ClosedLoopTraces({model}) {
  const [selected,setSelected]=useState(0);const conditions=model.conditions;if(!conditions.length)return null;
  const index=Math.min(selected,conditions.length-1),current=conditions[index],dt=model.interface?.sample_time;
  return <section><h4>已保存闭环轨迹</h4><label>划分 / 控制器版本 / 工况<select value={index} onChange={event=>setSelected(Number(event.target.value))}>{conditions.map((row,i)=><option key={i} value={i}>{row.label}</option>)}</select></label>
    {current.source&&<p className="report-note">保存来源：report.json / {current.source}。仅切换已记录轨迹，未重新评价；调参工况、独立复验与测试分别标注。</p>}
    <Values rows={current.traces.flatMap(({label,run})=>Object.entries(run.metrics??{}).filter(([,value])=>finite(value)).map(([key,value])=>({label:`${label} / ${key}`,value})))}/>
    <div className="report-charts">{(model.interface?.outputs??[]).map((variable,channel)=><Chart key={`y${channel}`} title={`${variable.name} / ${variable.unit}`} xLabel={finite(dt)?'输出区间终点 / s':'输出离散步（未声明周期）'} series={[
      ...current.traces.map(({label,run})=>({label,points:(run.trajectory??[]).map(row=>[(row.step+1)*(finite(dt)?dt:1),row.output?.[channel]])})),
      {label:'目标',dashed:true,points:(current.traces[0]?.run.trajectory??[]).map(row=>[(row.step+1)*(finite(dt)?dt:1),row.reference?.[channel]])},
    ]}/>)}{(model.interface?.controls??[]).map((variable,channel)=><Chart key={`u${channel}`} title={`动作 ${variable.name} / ${variable.unit}`} xLabel={finite(dt)?'控制更新起点 / s':'动作离散步（未声明周期）'} series={current.traces.map(({label,run})=>({label,points:(run.trajectory??[]).map(row=>[row.step*(finite(dt)?dt:1),row.u?.[channel]])}))}/>)}</div>
  </section>;
}
function RawCharts({model,onReadFile}) {
  const [name,setName]=useState(model.rawFiles[0]?.name??'');const [filter,setFilter]=useState(null);
  const [preview,setPreview]=useState(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);
  const reader=useRef(onReadFile);reader.current=onReadFile;
  useEffect(()=>{let live=true;if(!name||!onReadFile)return()=>{live=false;};setLoading(true);setError('');setPreview(null);
    reader.current(name,{max_rows:200,...(filter?{filter}:{})}).then(result=>{if(!live)return;const value=result?.file_preview??result;if(!value?.array_rows)throw new Error('报告读取没有返回已登记的数组切片');setPreview(value);}).catch(cause=>{if(live)setError(cause.message??String(cause));}).finally(()=>{if(live)setLoading(false);});
    return()=>{live=false;};
  },[name,filter?.key,filter?.value,Boolean(onReadFile)]);
  if(!model.rawFiles.length)return null;
  const slice=preview?.array_rows,groups=slice?.groups??{};const groupKey=['trajectory_id','condition_index','split_id'].find(key=>groups[key]?.values?.length);
  const charts=arrayCharts(model,preview);
  return <section className="report-raw"><h4>原始结果曲线</h4><p className="report-note">只读取此产物已登记的 NPZ 数组。不同轨迹、时间重置和非有限点均断开；单文件最多展示 16 个通道，每条曲线最多 200 个实际点。</p>
    <div className="report-controls"><label>结果文件<select value={name} onChange={event=>{setName(event.target.value);setFilter(null);}}>{model.rawFiles.map(file=><option key={file.name} value={file.name}>{file.label} · {file.name}</option>)}</select></label>
      {groupKey&&<label>{groupKey==='condition_index'?'条件':'轨迹 / 分组'}<select value={filter?JSON.stringify(filter.value):''} onChange={event=>setFilter(event.target.value===''?null:{key:groupKey,value:JSON.parse(event.target.value)})}>
        <option value="">全部（边界断开）</option>{groups[groupKey].values.map(value=><option key={JSON.stringify(value)} value={JSON.stringify(value)}>{String(value)}</option>)}
      </select></label>}
    </div>
    {!onReadFile&&<p className="report-note">当前连接没有原始数组读取能力；未生成替代曲线。</p>}
    {loading&&<p role="status">正在读取已登记的结果数组…</p>}{error&&<p role="alert" className="report-error">{error}</p>}
    {preview&&<><p className="report-note">实际返回 {slice.returned_rows} / 过滤后 {slice.matched_rows} 行（原文件 {slice.total_rows} 行）{slice.truncated?'，已均匀抽样并保留端点':''}。原始行号和分组均保留，指标没有重算。{groupKey&&groups[groupKey].truncated?'分组列表仅显示前 64 项。':''}</p><details><summary>原始结果校验与坐标</summary><code>{preview.name} · {preview.bytes} bytes<br/>SHA256 {preview.sha256}</code><p className="report-note">来源行号：{slice.source_rows.join(', ')}</p></details>
      <div className="report-charts">{charts.map((chart,index)=><Chart key={index} {...chart}/>)}</div>{!charts.length&&<p>该切片没有可展示的数值曲线；没有推定缺失变量。</p>}</>}
  </section>;
}

/** All reads/downloads use the caller's existing native Session. */
export function ArtifactReport({report,modelInterface,artifact,onReadFile,onDownloadBundle}) {
  const model=useMemo(()=>normalizeReport({report,modelInterface,artifact}),[report,modelInterface,artifact]);
  const [download,setDownload]=useState(null),[downloadBusy,setDownloadBusy]=useState(false),[downloadError,setDownloadError]=useState('');
  async function downloadBundle(){setDownloadBusy(true);setDownloadError('');try{setDownload(await onDownloadBundle()??{});}catch(error){setDownloadError(error.message??String(error));}finally{setDownloadBusy(false);}}
  return <section className="artifact-report" aria-label="真实结果报告"><header><div><h3>结果报告</h3><span className="report-algorithm">{model.algorithm}</span></div><span className="report-kind">{model.kind??'产物'}</span></header>
    <div className="report-notices">{model.notices.map((text,index)=><p key={index}>{text}</p>)}</div><Values rows={model.summary}/>
    {model.metricRows.length>0&&<section><h4>预测 / 离线误差</h4><Values rows={model.metricRows}/></section>}
    {model.logSeries.length>0&&<section><h4>训练与验证记录</h4><div className="report-charts">{model.logSeries.map((line,index)=><Chart key={index} title={line.label} series={[line]} xLabel="训练轮次（保存日志）"/>)}</div><p className="report-note">不同损失分开显示；PINN 数据 / 物理 / 初值项是各自保存的未加权量。</p></section>}
    {model.candidateRows.length>0&&<section><h4>基线、最好与最后候选</h4><div className="report-table"><table><thead><tr><th>候选范围</th><th>评分</th><th>硬门通过</th><th>真实参数 / 失败</th></tr></thead><tbody>{model.candidateRows.map(({label,value})=><tr key={label}><th>{label}</th><td>{formatValue(value.score)}</td><td>{formatValue(value.hard_constraints_passed)}</td><td><code>{JSON.stringify(value.parameters??value.failure??value.error??null)}</code></td></tr>)}</tbody></table></div></section>}
    {model.trials.length>0&&<><Chart title="真实 GA 候选评分（失败候选留空并断线）" xLabel="候选序号" series={[{label:'评分',points:model.trials.map((row,index)=>[index+1,row.score])}]}/><details><summary>全部 {model.trials.length} 个候选及失败记录</summary><div className="report-table"><table><thead><tr><th>序号</th><th>评分</th><th>硬门</th><th>错误 / 参数</th></tr></thead><tbody>{model.trials.map((row,index)=><tr key={index}><td>{index+1}</td><td>{formatValue(row.score)}</td><td>{formatValue(row.hard_constraints_passed)}</td><td><code>{JSON.stringify(row.failure??row.error??row.parameters??null)}</code></td></tr>)}</tbody></table></div></details></>}
    {model.closedMetrics.length>0&&<section><h4>原 Plant 自主闭环指标</h4><Values rows={model.closedMetrics}/></section>}<MetricDetails items={model.metricDetails}/><ClosedLoopTraces model={model}/>
    <RawCharts key={artifact?.artifact_id??artifact?.id??model.algorithm} model={model} onReadFile={onReadFile}/>
    {model.benchmark.length>0&&<section><h4>实际控制器计时</h4><div className="report-table"><table><thead><tr><th>策略</th><th>中位数 / ms</th><th>P95 / ms</th><th>记录状态</th></tr></thead><tbody>{model.benchmark.map(row=><tr key={row.label}><th>{row.label}</th><td>{formatValue(finite(row.median)?row.median*1000:undefined)}</td><td>{formatValue(finite(row.p95)?row.p95*1000:undefined)}</td><td>{row.status??model.report.controller_benchmark.status??'未报告'}</td></tr>)}</tbody></table></div><details><summary>计时范围与样本设置</summary><pre>{JSON.stringify({scope:model.report.controller_benchmark.scope,settings:model.report.controller_benchmark.settings},null,2)}</pre></details></section>}
    <details><summary>实际配置、约束与验收范围</summary><pre>{JSON.stringify(Object.fromEntries(['physical_equation','conditional_parameterization','loss_weights','normalization','network_config','resolved_config','feature_order','target_order','physical_binding','controller_config','weights','cost','constraints','solver_check','evaluation','evaluation_hash','chosen_parameters','domain','deployment_scope','capability','accuracy_claim','stability_proof','closed_loop_validation','validation_scope'].filter(key=>Object.hasOwn(model.report,key)).map(key=>[key,model.report[key]])),null,2)}</pre></details>
    {model.coverage&&<details><summary>训练覆盖 / 适用范围（非全域保证）</summary><p className="report-note">逐变量范围不证明联合覆盖或外推有效性。</p><pre>{JSON.stringify(model.coverage,null,2)}</pre></details>}
    <section className="report-download"><h4>完整独立包</h4>{model.bundleAvailable?<><p>下载已发布 bundle 的完整代码、权重、接口、尺度与使用说明；通过原生文件服务获取完整 ZIP。</p><button type="button" disabled={!onDownloadBundle||downloadBusy} onClick={downloadBundle}>{downloadBusy?'正在准备并核验完整包…':'下载完整独立包 ZIP'}</button>{!onDownloadBundle&&<p>当前连接没有下载回调。</p>}</>:<p>此产物没有已发布的独立 bundle。报告或数据文件不能代替可执行模型包。</p>}
      {downloadError&&<p role="alert" className="report-error">{downloadError}</p>}{download&&<p className="report-note">{download.hash_verified===true?'浏览器已核验下载 SHA256。':'下载请求已完成。'}{download.filename}<br/>{download.sha256&&`SHA256 ${download.sha256}`}</p>}
    </section>
  </section>;
}
