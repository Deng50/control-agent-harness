"""Export actual DC motor artifacts to Chinese PPT figures and tables; no training."""
from pathlib import Path
import argparse, csv, json, sys, shutil, hashlib, zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from control_harness.artifacts import Workspace, read_json
from control_harness.modeling import load_model, tensor
from control_harness.data import read_transitions

SKILL=Path('C:/Users/sk3547969879/.agents/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

COL={'nominal':'#87929f','corrected':'#1764ab','reference':'#242a33','baseline':'#87929f','candidate':'#1764ab','chosen':'#1764ab','expert':'#1764ab','imitation':'#dc8b28','optimized':'#2b8c69'}
LABEL={'nominal':'名义模型','corrected':'校正模型','reference':'参考对象','baseline':'初始 MPC','candidate':'调参 MPC','chosen':'选定 MPC','expert':'MPC 专家','imitation':'模仿策略','optimized':'优化策略'}

def csvwrite(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def export(ws,model_only=False):
    torch.set_num_threads(1)
    w=Workspace(ws); out=ws/'presentation'; sources=out/'source-data'; figures=out/'figures'; qa=out/'qa'
    for p in [out,sources,figures,qa]:p.mkdir(exist_ok=True,parents=True)
    manifest=read_json(ws/'manifest.json'); ids={a['kind']:i for i,a in manifest['artifacts'].items() if a['status']=='verified'}
    names=dict(nominal='nominal_model',reference='reference_model',data='transition_dataset',corrected='corrected_model',controller='controller',expert='expert_dataset',imitation='imitation_policy',optimized='optimized_policy')
    stages=list(names)[:4] if model_only else list(names)
    roots={};meta={}
    for s in stages:roots[s],meta[s]=w.resolve(ids[names[s]])
    record=dict(case='dc-motor',execution_surface='visible_web_forms',workspace=str(ws),status='model_verified' if model_only else 'verified',artifacts={s:ids[names[s]] for s in stages})
    (ws/'case-run.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    models={s:load_model(w,ids[names[s]]) for s in ['nominal','reference','corrected']}
    dt=models['nominal'].interface.sample_time
    reports={s:read_json(roots[s]/'report.json') for s in stages if s in ['corrected','controller','imitation','optimized']}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],'font.size':16,'axes.titlesize':19,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    index=[]
    def canvas(title,n=1):
        fig,axs=plt.subplots(n,1,figsize=(13.3333,7.5),squeeze=False)
        fig.subplots_adjust(left=.11,right=.97,bottom=.12,top=.80,hspace=.34)
        fig.suptitle(title,y=.96,fontweight='bold');return fig,axs[:,0]
    def save(fig,key,title,source,caption):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=qa/f'{key}.alignment.json',overlay_svg=qa/f'{key}.alignment.svg',strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
        fig.savefig(figures/f'{key}.png',dpi=300)
        fig.savefig(figures/f'{key}.svg')
        fig.savefig(figures/f'{key}.pdf')
        index.append(dict(id=key,title=title,source=source,caption=caption,files={e:f'figures/{key}.{e}' for e in ['png','svg','pdf']}))
        plt.close(fig)
    def finish(fig,axs):
        for ax in axs:ax.grid(True,alpha=.18);ax.margins(x=0)
        h,l=axs[0].get_legend_handles_labels()
        if h:fig.legend(h,l,loc='upper center',bbox_to_anchor=(.53,.91),ncol=4)
    data,masks,_=read_transitions(w,ids['transition_dataset'])
    split={k:dict(samples=int(mask.sum()),trajectories=np.unique(data['trajectory_id'][mask]).tolist()) for k,mask in masks.items()}
    test={k:v[masks['test']] for k,v in data.items()}
    predictions={}
    with torch.no_grad():
        for name in ['nominal','corrected']:predictions[name]=models[name](tensor(test['x']),tensor(test['u']),tensor(test['d'])).numpy()
    rmses=[]
    for name,p in predictions.items():
        for j,(signal,unit) in enumerate([('电流','A'),('转速','rad/s')]):rmses.append(dict(method=name,state=signal,unit=unit,rmse=float(np.sqrt(np.mean((p[:,j]-test['next_x'][:,j])**2))),samples=len(p)))
    csvwrite(sources/'model-test-rmse.csv',rmses)
    csvwrite(out/'模型校正结果对比.csv',[dict(方法=LABEL[r['method']],状态=r['state'],单位=r['unit'],RMSE=r['rmse'],测试样本数=r['samples']) for r in rmses])
    csvwrite(sources/'model-test-predictions.csv',[dict(row=n,trajectory_id=int(test['trajectory_id'][n]),current_A=float(test['next_x'][n,0]),speed_rad_s=float(test['next_x'][n,1]),**{f'{m}_{j}':float(v[n,j]) for m,v in predictions.items() for j in [0,1]}) for n in range(len(test['x']))])
    for part,mask in masks.items():
        csvwrite(sources/f'transitions-{part}.csv',[dict(trajectory_id=int(data['trajectory_id'][n]),current_A=float(data['x'][n,0]),speed_rad_s=float(data['x'][n,1]),voltage_V=float(data['u'][n,0]),load_Nm=float(data['d'][n,0]),next_current_A=float(data['next_x'][n,0]),next_speed_rad_s=float(data['next_x'][n,1])) for n in np.flatnonzero(mask)])
    roll={};rollrows=[]
    for m,model in models.items():
        xx=np.array([0.,0.]);values=[xx.copy()]
        for n in range(60):xx=model.step(xx,[8.],[.03]);values.append(xx.copy())
        roll[m]=np.array(values)
        rollrows.extend(dict(method=m,time_s=n*dt,current_A=float(v[0]),speed_rad_s=float(v[1]),voltage_V=8,load_Nm=.03) for n,v in enumerate(values))
    csvwrite(sources/'model-independent-rollout.csv',rollrows)
    title='模型校正：独立恒压响应预测'
    fig,axs=canvas(title,2)
    for m,values in roll.items():
        for j,ax in enumerate(axs):ax.plot(np.arange(61)*dt,values[:,j],label=LABEL[m],color=COL[m],linestyle='--' if m=='reference' else '-',linewidth=2)
    axs[0].set_ylabel('电流 / A');axs[1].set_ylabel('转速 / (rad/s)');axs[1].set_xlabel('时间 / s');finish(fig,axs)
    save(fig,'01-model-rollout',title,'model-independent-rollout.csv','初始状态 [0,0]，电压8 V、负载0.03 N·m，60步；各模型使用自身前一预测状态，未用于训练或检查点选择。')
    title='模型校正：留出测试集单步预测误差';fig,axs=canvas(title,2)
    for j,ax in enumerate(axs):
        vals=[r['rmse'] for r in rmses if r['state']==['电流','转速'][j]]
        assert all(v>0 for v in vals), 'Log scale requires positive RMSE'
        ax.bar(['名义模型','校正模型'],vals,color=[COL['nominal'],COL['corrected']],width=.45);ax.set_yscale('log');ax.set_ylabel(['电流 RMSE / A','转速 RMSE / (rad/s)'][j]);ax.grid(axis='y',alpha=.18)
    save(fig,'02-model-test-error',title,'model-test-rmse.csv',f'按整条轨迹划分，测试样本 {len(test["x"])}；对数纵轴；单种子，无重复实验区间。')
    def learning(stage,key,title):
        log=reports[stage]['log'];csvwrite(sources/f'{key}.csv',log);fig,axs=canvas(title)
        for field,label in [('training_loss','训练损失'),('validation_loss','验证损失')]:
            if field in log[0]:axs[0].plot([r['epoch'] for r in log],[r[field] for r in log],label=label,linewidth=2)
        axs[0].set_xlabel('训练轮数');axs[0].set_ylabel('归一化损失');axs[0].set_yscale('log');finish(fig,axs);save(fig,key,title,f'{key}.csv','完整保存各轮记录；训练与验证目标可能包含不同的约束项，数值不直接等价。')
    learning('corrected','03-residual-learning','残差网络训练与验证曲线')
    summary=dict(**record,source='mathematical_reference_simulator_not_measurements',sample_time_s=dt,transition_split=split,model_test_rmse=rmses,residual_epochs=reports['corrected']['trained_epochs'])
    if not model_only:
        from control_harness.policy import read_expert
        expert,expert_masks,collection,_=read_expert(w,ids['expert_dataset'])
        expert_split={k:dict(samples=int(mask.sum()),scenarios=[collection['scenarios'][int(j)]['scenario']['name'] for j in np.unique(expert['trajectory_id'][mask])]) for k,mask in expert_masks.items()}
        summary['expert_split']=expert_split
        expert_labels={int(j):part for part,mask in expert_masks.items() for j in np.unique(expert['trajectory_id'][mask])}
        csvwrite(sources/'expert-data.csv',[dict(trajectory_id=int(expert['trajectory_id'][n]),partition=expert_labels[int(expert['trajectory_id'][n])],current_A=float(expert['x'][n,0]),speed_rad_s=float(expert['x'][n,1]),target_rad_s=float(expert['reference'][n,0]),load_Nm=float(expert['d'][n,0]),voltage_V=float(expert['u'][n,0])) for n in range(len(expert['x']))])
        ctrl=reports['controller'];csvwrite(sources/'controller-ga-trials.csv',[dict(trial=r['index']+1,generation=r['generation'],score=r['score'],feasible=r['hard_constraints_passed'],**r['parameters']) for r in ctrl['trials']])
        fig,axs=canvas('遗传算法：MPC 参数搜索')
        scores=[np.nan if r['score'] is None else r['score'] for r in ctrl['trials']];axs[0].plot(np.arange(1,len(scores)+1),scores,'o-',label='候选代价');axs[0].plot(np.arange(1,len(scores)+1),np.minimum.accumulate([np.inf if np.isnan(s) else s for s in scores]),label='当前最优代价');axs[0].set_xlabel('候选序号');axs[0].set_ylabel('归一化评价代价');finish(fig,axs);save(fig,'04-ga-search','遗传算法：MPC 参数搜索','controller-ga-trials.csv','候选得分来自调参工况；独立复验后才决定是否采用，测试不参与调参。')
        def response(runs,key,title):
            fig,axs=canvas(title,2);rows=[]
            for method,run in runs.items():
                traj=run['trajectory'];tt=np.arange(len(traj)+1)*dt
                yy=[traj[0]['x'][1]]+[r['next_x'][1] for r in traj]
                axs[0].plot(tt,yy,label=LABEL[method],color=COL[method],linewidth=2)
                axs[1].step(tt,np.r_[[r['u'][0] for r in traj],traj[-1]['u'][0]],where='post',color=COL[method],label=LABEL[method],linewidth=2)
                rows.extend(dict(method=method,step=r['step'],time_s=r['step']*dt,next_time_s=(r['step']+1)*dt,current_A=r['x'][0],speed_rad_s=r['x'][1],next_current_A=r['next_x'][0],next_speed_rad_s=r['next_x'][1],voltage_V=r['u'][0],target_rad_s=r['reference'][0],load_Nm=r['d'][0]) for r in traj)
            reference=next(iter(runs.values()))['scenario']['reference'][0];axs[0].axhline(reference,color='#242a33',ls='--',label='目标转速');axs[0].set_ylabel('转速 / (rad/s)');axs[1].set_ylabel('电枢电压 / V');axs[1].set_xlabel('时间 / s');axs[1].axhline(24,color='#555',ls=':',lw=1);axs[1].axhline(-24,color='#555',ls=':',lw=1);finish(fig,axs);csvwrite(sources/f'{key}.csv',rows);save(fig,key,title,f'{key}.csv','完整闭环轨迹；转速含初始点，电压按零阶保持绘制；上下界 ±24 V。')
        response(dict(baseline=ctrl['verification_baseline']['runs'][0],candidate=ctrl['verification_candidate']['runs'][0]),'05-mpc-verification','MPC 调参：独立复验工况对比')
        response(dict(chosen=ctrl['test']['runs'][0]),'06-mpc-final-test','选定 MPC：保留测试工况')
        comparisons=reports['optimized']['comparison'];table=[];policyrows=[]
        for n,c in enumerate(comparisons):
            response({m:c[m] for m in ['expert','imitation','optimized']},f'07-policy-heldout-{n+1:02}',f'控制器轻量化：留出工况 {c["scenario"]["name"]}')
            for m in ['expert','imitation','optimized']:
                rr=c[m];metric=rr['metrics'];policyrows.append(dict(scenario=c['scenario']['name'],method=m,**metric))
        csvwrite(sources/'policy-heldout-metrics.csv',policyrows)
        policy_summary={}
        for m in ['expert','imitation','optimized']:
            mm=[c[m]['metrics'] for c in comparisons];tr=[c[m]['trajectory'] for c in comparisons]
            policy_summary[m]=dict(rmse=float(np.mean([v['rmse'] for v in mm])),mae=float(np.mean([np.mean([abs(r['output'][0]-r['reference'][0]) for r in t]) for t in tr])),mean_control_ms=1000*float(np.mean([v['mean_control_seconds'] for v in mm])),max_violation=max(v[k] for v in mm for k in ['max_state_violation','max_input_violation','max_output_violation']))
            table.append(dict(方法=LABEL[m],转速RMSE_rad_s=policy_summary[m]['rmse'],转速MAE_rad_s=policy_summary[m]['mae'],平均调用耗时_ms=policy_summary[m]['mean_control_ms'],最大约束违规=policy_summary[m]['max_violation']))
        csvwrite(out/'轻量化结果对比.csv',table)
        learning('imitation','08-imitation-learning','专家模仿网络训练与验证曲线')
        optlog=reports['optimized']['log'];csvwrite(sources/'policy-optimization.csv',optlog)
        fig,axs=canvas('物理策略优化：独立验证代价')
        initial_validation_score=float(np.mean([r['metrics']['normalized_rmse']+.01*r['metrics']['control_effort'] for r in reports['imitation']['closed_loop']['validation']]))
        val=np.array([np.nan if r['validation_score'] is None else r['validation_score'] for r in optlog])
        axs[0].plot([r['epoch'] for r in optlog],val,label='当前策略验证代价',linewidth=2)
        axs[0].plot([r['epoch'] for r in optlog],np.minimum.accumulate(np.r_[initial_validation_score,np.where(np.isfinite(val),val,np.inf)])[1:],label='含初始策略的最优代价',linewidth=2,ls='--')
        axs[0].axhline(initial_validation_score,label='初始模仿策略',color='#444444',ls=':',linewidth=1.5)
        axs[0].set_xlabel('优化轮数');axs[0].set_ylabel('归一化验证代价');finish(fig,axs)
        save(fig,'10-policy-optimization','物理策略优化：独立验证代价','policy-optimization.csv','按完整验证工况选择共享最佳检查点；测试不参与更新或选择。空验证分数以缺口保留，不伪造数值。')
        fig,axs=canvas('控制器轻量化：精度与计算耗时',2)
        methods=['expert','imitation','optimized'];labels=[LABEL[m] for m in methods]
        axs[0].bar(labels,[policy_summary[m]['rmse'] for m in methods],color=[COL[m] for m in methods],width=.5);axs[0].set_ylabel('转速 RMSE / (rad/s)')
        axs[1].bar(labels,[policy_summary[m]['mean_control_ms'] for m in methods],color=[COL[m] for m in methods],width=.5);axs[1].set_yscale('log');axs[1].set_ylabel('平均调用耗时 / ms');axs[1].axhline(dt*1000,color='#242a33',ls='--',label='采样周期')
        h,l=axs[1].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.53,.91))
        save(fig,'09-accuracy-time','控制器轻量化：精度与计算耗时','policy-heldout-metrics.csv','两个留出工况分别计算后取算术平均；本机单次 CPU 计时，不代表硬件无关加速或实时性保证。')
        ctrlrows=[]
        for label,key in [('初始 MPC','verification_baseline'),('调参 MPC','verification_candidate')]:
            mm=ctrl[key]['runs'][0]['metrics'];ctrlrows.append(dict(方法=label,转速RMSE_rad_s=mm['rmse'],电压总变化_V=mm['input_total_variation'],最大约束违规=max(mm[k] for k in ['max_state_violation','max_input_violation','max_output_violation'])))
        csvwrite(out/'MPC独立复验对比.csv',ctrlrows)
        summary.update(controller_parameters=read_json(roots['controller']/'controller.json')['parameters'],controller_verification=ctrlrows,controller_candidate_accepted=ctrl['tuned_candidate_accepted'],policy=policy_summary,imitation_epochs=reports['imitation']['trained_epochs'],optimization_epochs=reports['optimized']['trained_epochs'],parameter_count=reports['imitation']['parameter_count'])
        imitation_state=torch.load(roots['imitation']/'policy.pt',map_location='cpu',weights_only=True)['policy_state']
        optimized_state=torch.load(roots['optimized']/'policy.pt',map_location='cpu',weights_only=True)['policy_state']
        summary['optimization_retained_imitation_weights']=all(torch.equal(imitation_state[k],optimized_state[k]) for k in imitation_state)
        summary['imitation_best_validation_epoch']=min(reports['imitation']['log'],key=lambda r:r['validation_loss'])['epoch']
        summary['initial_imitation_validation_score']=initial_validation_score
        # Copy complete deployable artifact dependencies, including weights and preprocessing.
        dep=out/'trained-artifacts'
        for s in stages:
            shutil.copytree(roots[s],dep/ids[names[s]],dirs_exist_ok=True)
        portable=json.loads((ws/'manifest.json').read_text(encoding='utf8'))
        for row in portable['artifacts'].values():row['path']=Path(row['path']).name
        (dep/'manifest.json').write_text(json.dumps(portable,ensure_ascii=False,indent=2),encoding='utf8')
        (dep/'说明.txt').write_text('此目录是可加载的八产物快照，含权重、数据和依赖；manifest仅将相对路径改为此处平铺的ID目录，产物ID、元数据和文件哈希保持原样。可把此目录作为 load_trained_policy.py 的 workspace 参数，需安装项目Python依赖。勿把 .pt 视作完整独立网络。',encoding='utf8')
    (out/'metrics-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'figures-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    table=['|状态|单位|名义模型 RMSE|校正模型 RMSE|','|---|---|---:|---:|']
    for state in ['电流','转速']:
        a=next(r for r in rmses if r['method']=='nominal' and r['state']==state);b=next(r for r in rmses if r['method']=='corrected' and r['state']==state);table.append(f'|{state}|{a["unit"]}|{a["rmse"]:.6g}|{b["rmse"]:.6g}|')
    content='# 直流电机完整案例：网页操作与 PPT 素材\n\n'+ '\n'.join(table)+'\n\n'
    content+='输入与确认步骤见 reproduce/README.md。screenshots/ 保存网页输入、接口确认、接力及结果截图；原始作业记录在工作空间 .control-jobs/。本次由可见网页表单逐阶段执行，不将数值后台作业记录冒充 LLM 回复。\n\n'
    if not model_only:
        content+='## MPC 独立复验\n\n|方法|转速 RMSE (rad/s)|电压总变化 (V)|最大约束违规|\n|---|---:|---:|---:|\n'
        for r in ctrlrows:content+=f'|{r["方法"]}|{r["转速RMSE_rad_s"]:.4f}|{r["电压总变化_V"]:.4f}|{r["最大约束违规"]:.4g}|\n'
        content+='\n## 轻量化对比：两个留出工况平均\n\n|方法|转速 RMSE (rad/s)|转速 MAE (rad/s)|平均调用耗时 (ms)|最大约束违规|\n|---|---:|---:|---:|---:|\n'
        for m,r in policy_summary.items():content+=f'|{LABEL[m]}|{r["rmse"]:.4f}|{r["mae"]:.4f}|{r["mean_control_ms"]:.4f}|{r["max_violation"]:.4g}|\n'
        content+='\n预先固定种子产生的两个留出目标均为正转速；负转速工况存在于训练和验证集，不能据此声称负转速留出泛化。\n\n'
        if summary['optimization_retained_imitation_weights']:content+='本次40轮物理策略更新没有在验证集上超过初始模仿策略，按验收逻辑保留初始模仿权重。已逐张量核对两份发布策略相同，曲线重合；两次计时差异不能归因于结构或算法加速。\n\n'
        content+='物理优化采用可微模型 rollout 策略梯度，无学习型 Critic。按实际测试保留改善与退化；模型为教学参考仿真，非实测。耗时单位由秒换算为毫秒。\n'
    content+='\n## 图形索引\n\n'+'\n'.join(f'- {x["title"]}：{x["caption"]}' for x in index)
    (out/'案例结果与演示说明.md').write_text(content,encoding='utf8')
    shutil.copytree(Path(__file__).resolve().parents[1]/'examples/dc-motor',out/'reproduce',dirs_exist_ok=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);p.add_argument('--model-only',action='store_true');a=p.parse_args();export(a.workspace.resolve(),a.model_only)
