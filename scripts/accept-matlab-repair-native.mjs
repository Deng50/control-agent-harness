/** Keyless real-MCP comparison. This script never calls or impersonates an LLM. */
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {mkdir,readFile,stat,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {configureFixtureFileGeneration,openLayer,project,readFixture} from './matlab-layer-harness.mjs';
import {execFileSync} from 'node:child_process';

const {fixture,marker,resource}=await readFixture(resolve(process.argv[2]));
const root=join(fixture,'native-'+randomUUID().slice(0,8));
const native='mcp__matlab__evaluate_matlab_code';
const q=s=>"'"+s.replaceAll("'","''")+"'";
const report={source:'deterministic_real_mcp',matlab_pid:marker.matlab_pid,matlab_session:marker.session_id,stages:[]};
const snapshot=async()=>writeFile(join(root,'evidence.json'),JSON.stringify(report,null,2));
await mkdir(root);
const layers=[];
const create=async (name)=>{
  const workspace=join(root,name); await mkdir(workspace);
  execFileSync('powershell.exe',['-NoProfile','-File',join(project,'scripts/prepare-workspace.ps1'),'-Workspace',workspace,'-Prepare'],{windowsHide:true,stdio:'pipe'});
  const layer=await openLayer({fixture,workspace,managed:true});
  layers.push(layer); return layer;
};
const call=async(layer,stage,name,args,fails=false)=>{
  const r=await layer.call(name,args),state=JSON.parse(await readFile(join(resource,'resource.json'),'utf8'));
  const content=(r.content??[]).filter(c=>c.type==='text').map(c=>c.text).join('\n');
  report.stages.push({stage,agent:layer.agent.id,tool:name,error:r.isError,content,completion:state.last_completion,quarantine:state.quarantine?.reason??null});
  await snapshot(); assert.equal(r.isError,fails,stage+' '+content);
  assert.equal(state.quarantine,null); assert.equal(state.owner,null); assert.equal(state.session.pid,marker.matlab_pid);
  return content;
};
const evaluate=(layer,stage,code,fails=false)=>call(layer,stage,native,{code,project_path:layer.agent.session.header.cwd},fails);
const edge=(sourceBlock,destinationBlock,destinationPort=1)=>({sourceBlock,sourcePort:1,destinationBlock,destinationPort});
try {
  const a=await create('partial'), b=await create('controller');
  await configureFixtureFileGeneration(a,fixture);
  const id=randomUUID().replaceAll('-','').slice(0,10), other='Keep'+id, model='Mpc'+id;
  await evaluate(a,'create constant plant',`new_system('${other}'); add_block('simulink/Sources/Constant','${other}/Constant','Value','2.5'); add_block('simulink/Sinks/Out1','${other}/Y'); set_param('${other}','SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset');`);
  await call(a,'connect constant plant','control_wire_simulink',{model:other,connections:[edge('Constant','Y')]});
  const saved=JSON.parse(await call(a,'save constant plant','control_save_simulink_model',{model:other}));
  const initial=await stat(saved.path);
  await call(a,'solver error after save','control_simulate_simulink_model',{model:other,solver:'NotARealSolver',stopTime:1},true);
  await call(a,'correct solver and simulate','control_simulate_simulink_model',{model:other,solver:'ode45',stopTime:1});
  const constant=await evaluate(a,'read actual constant output',`in=Simulink.SimulationInput('${other}'); out=sim(in); vals=out.get('yout').getElement(1).Values.Data; fprintf('CHECK_Y=%.17g\\n',vals(end)); ws=get_param('${other}','ModelWorkspace'); ws.assignin('KeepOtherChatValue',1701);`);
  assert.match(constant,/CHECK_Y=2\.5\b/);
  assert.equal((await stat(saved.path)).mtimeMs,initial.mtimeMs);
  assert.equal(createHash('sha256').update(await readFile(saved.path)).digest('hex'),saved.sha256);
  await evaluate(b,'partial MPC then incorrect library',`new_system('${model}'); add_block('simulink/Discrete/Discrete State-Space','${model}/Plant','A','0.8','B','0.2','C','1','D','0','SampleTime','0.1'); mpctest=mpc(ss(0.8,0.2,1,0,0.1),0.1,12,2); assignin('base','RetryFixtureMPC',mpctest); add_block('mpc_lib/MPC Controller','${model}/Controller');`,true);
  const toolkitError=await call(b,'Toolkit domain error on partial MPC','mcp__matlab__model_edit',{
    model,scope:'root',layout_mode:'incremental',
    operations:JSON.stringify([{op:'add_block',type:'mpc_lib/MPC Controller',name:'Controller',ref:'badblock',position:[150,110,205,190]}]),
  },true);
  assert.match(toolkitError,/INVALID_PARAM/);
  const discovery=await evaluate(b,'discover library and inspect partial model',`fprintf('LIB=%s\\n',which('mpclib')); disp(find_system('${model}','SearchDepth',1,'Type','Block')); load_system('mpclib'); disp(find_system('mpclib','SearchDepth',1,'Name','MPC Controller'));`);
  assert.match(discovery,/mpclib[\\/]MPC Controller/); assert.ok(discovery.includes(model+'/Plant'));
  await evaluate(b,'correct failed step',`add_block('mpclib/MPC Controller','${model}/Controller','mpcobj','RetryFixtureMPC','md_inport','off'); w=get_param('${model}','ModelWorkspace'); w.assignin('RetryFixtureMPC',mpctest); add_block('simulink/Sources/Constant','${model}/Reference','Value','1'); add_block('simulink/Sinks/Out1','${model}/Y'); set_param('${model}','Solver','FixedStepDiscrete','FixedStep','0.1','StopTime','4','SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset');`);
  await call(b,'wire actual feedback','control_wire_simulink',{model,connections:[edge('Plant','Controller'),edge('Reference','Controller',2),edge('Controller','Plant'),edge('Plant','Y')]});
  const mpc=JSON.parse(await call(b,'save controller','control_save_simulink_model',{model}));
  const result=await evaluate(b,'reload simulate and validate sibling',`close_system('${model}',0); load_system(${q(mpc.path)}); t=Simulink.SimulationInput('${model}'); out=sim(t); vals=out.get('yout').getElement(1).Values.Data; fprintf('CHECK_Y=%.17g %.17g %d\\n',vals(1),vals(end),numel(vals)); assert(bdIsLoaded('${other}')); w=get_param('${other}','ModelWorkspace'); fprintf('OTHER_CHAT=%.0f\\n',w.getVariable('KeepOtherChatValue'));`);
  const match=result.match(/CHECK_Y=([\d.e+-]+) ([\d.e+-]+) (\d+)/i);
  assert.ok(match && Number(match[1])===0 && Math.abs(1-Number(match[2]))<0.1 && Number(match[3])>=40);
  assert.match(result,/OTHER_CHAT=1701/);
  report.verified={partial:saved.path,mpc:mpc.path,initial:Number(match[1]),final:Number(match[2]),samples:Number(match[3])};
} catch(error) {report.error=error.message; throw error;}
finally {await snapshot(); for(const layer of layers.reverse()) await layer.close(); console.log(join(root,'evidence.json'));}
