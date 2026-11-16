"""Package completed web-run artifacts, screenshots, data and scientific figures."""
from pathlib import Path
import json,shutil,hashlib,zipfile,re,html,sys,csv

def build(ws):
    out=ws/'presentation';root=Path(__file__).resolve().parents[1]
    summary=json.loads((out/'metrics-summary.json').read_text(encoding='utf8'))
    if summary['status']!='verified':raise ValueError('Full eight-stage export required')
    dep=out/'trained-artifacts'
    portable=json.loads((ws/'manifest.json').read_text(encoding='utf8'))
    for row in portable['artifacts'].values():row['path']=Path(row['path']).name
    (dep/'manifest.json').write_text(json.dumps(portable,ensure_ascii=False,indent=2),encoding='utf8')
    (dep/'说明.txt').write_text('此目录是可加载的八产物快照，含权重、数据和依赖；manifest仅将相对路径改为平铺的ID目录，产物ID、元数据和文件哈希保持原样。可把此目录作为 load_trained_policy.py 的 workspace 参数，需安装项目Python依赖。勿把 .pt 视作完整独立网络。',encoding='utf8')
    sys.path.insert(0,str(root/'src'))
    from control_harness.artifacts import Workspace
    snapshot=Workspace(dep)
    for artifact in summary['artifacts'].values():snapshot.resolve(artifact)
    shutil.copytree(root/'examples/dc-motor',out/'reproduce',dirs_exist_ok=True)
    for name in ['export_dc_motor.py','package_dc_motor.py','prepare_dc_motor_case.py','create_dc_motor_workflow.py','audit_dc_motor_figures.py']:
        shutil.copy2(root/'scripts'/name,out/'reproduce'/name)
    # Retain domain requests and event records only, with no Harness global settings.
    jobs=out/'execution-records';jobs.mkdir(exist_ok=True)
    history=[]
    for folder in sorted((ws/'.control-jobs').iterdir()):
        if not(folder/'request.json').is_file():continue
        req=json.loads((folder/'request.json').read_text(encoding='utf8'))
        events=[json.loads(p.read_text(encoding='utf8')) for p in sorted(folder.glob('0*.json'))]
        public=dict(attempt_id=req.get('attemptId'),created_at=req.get('createdAt'),request=req.get('request'),events=events)
        (jobs/f'{folder.name}.json').write_text(json.dumps(public,ensure_ascii=True,indent=2),encoding='utf8')
        history.append(dict(attempt_id=req.get('attemptId'),created_at=req.get('createdAt'),operation=req.get('request',{}).get('operation'),last_state=next((e['state'] for e in reversed(events) if 'state'in e),None)))
    history.sort(key=lambda r:r.get('created_at') or '')
    (out/'网页阶段执行记录.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf8')
    index=json.loads((out/'figures-index.json').read_text(encoding='utf8'))
    figures=''.join(f'<section><h2>{html.escape(x["title"])}</h2><img src="{html.escape(x["files"]["png"])}"><p>{html.escape(x["caption"])}</p><a href="{x["files"]["svg"]}">矢量 SVG</a> · <a href="{x["files"]["pdf"]}">PDF</a></section>' for x in index)
    shots=''.join(f'<section><h2>{html.escape(p.stem)}</h2><img src="screenshots/{html.escape(p.name)}"></section>' for p in sorted((out/'screenshots').glob('*.png')))
    tables=''
    for p in out.glob('*对比.csv'):
        with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
        tables+='<section><h2>'+html.escape(p.stem)+'</h2><table>'+''.join('<tr>'+''.join(f'<{"th" if n==0 else "td"}>{html.escape(v)}</{"th" if n==0 else "td"}>' for v in row)+'</tr>' for n,row in enumerate(rows))+'</table></section>'
    page='<!doctype html><meta charset="utf-8"><title>直流电机：三智能体演示素材</title><style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;color:#173553}h1{color:#083b89}section{margin:36px 0;padding-bottom:24px;border-bottom:1px solid #ddd}img{max-width:100%}a{color:#1764ab}p{line-height:1.7}</style><h1>直流电机：模型校正 → MPC → 控制器轻量化</h1><p>完整真实网页运行；数学参考仿真。曲线保留全部时刻，留出工况不参与训练或选点。</p><p><a href="reproduce/网页操作与确认清单.md">逐步点击、输入与确认清单</a> · <a href="案例结果与演示说明.md">实际结果说明</a> · <a href="metrics-summary.json">指标原始汇总</a></p><h2>结果曲线</h2>'+figures+'<h2>真实网页操作截图</h2>'+shots
    page=page.replace('<h2>结果曲线</h2>',tables+'<h2>结果曲线</h2>').replace('img{max-width:100%}','img{max-width:100%}table{border-collapse:collapse;width:100%}th{background:#396daf;color:white}td,th{padding:10px;border:1px solid #d9e1ed}tr:nth-child(even){background:#f0f4fa}')
    (out/'index.html').write_text(page,encoding='utf8')
    (out/'qa/审核说明.txt').write_text('面向PPT的13.3333×7.5英寸画布，300 dpi PNG与可编辑文字SVG/PDF；微软雅黑优先。期刊尺寸、TIFF和600dpi提示不适用本次PPT素材。所有时刻保留；同一工况比较；单种子，不推断统计显著性。调用耗时是本机CPU单次运行计时，不等于硬实时保证。PNG截图来自真实浏览器，未经重绘。执行记录保留中文来源说明编码错误、英文说明重试，以及专家采集超时和按原配置重试。预先固定的专家划分使本次两个留出目标均为正转速，不据此声称负转速留出泛化。最终图形另有面板对齐、PDF字形与碰撞审查JSON。',encoding='utf8')
    # Verify hashes of original domain artifacts by resolving with the repository API externally.
    hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf8')
    suspicious=[]
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in ['.txt','.md','.json','.py','.html','.csv']:
            if re.search(r'\bsk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9_-]{16,}|[?&]token=[A-Za-z0-9_-]{16,}',p.read_text(encoding='utf-8-sig')):suspicious.append(p.name)
    if suspicious:raise ValueError('Credential pattern detected in package inputs')
    archive=ws/'dc-motor-PPT-materials-and-networks.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,'dc-motor-materials/'+str(p.relative_to(out)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in hashes.items():assert hashlib.sha256(z.read('dc-motor-materials/'+name)).hexdigest()==h
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,figures=len(index),screenshots=len(list((out/'screenshots').glob('*.png'))),hashes_verified=len(hashes),jobs=len(history),credential_patterns=len(suspicious)),indent=2))

if __name__=='__main__':build(Path(sys.argv[1]).resolve())
