/** S09 L03: native CSV/XLSX import of actual L01 external expert commands. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {createObserver,hash,immutableFile,json,near,py,runRoot} from './smoke-external-expert-python.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},fixtures:{type:'string'},'source-evidence':{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},'prepare-only':{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-external-expert-data.mjs --workspace-root <isolated S09 parent> --evidence <json> --source-evidence <PASS L01 json> [--fixtures <frozen fixtures>] [--web-log <Host log>] [--resume] [--prepare-only]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence&&values['source-evidence']);
const fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures')),f05Bytes=await readFile(resolve(fixtures,'F05_expert_and_tradeoff_candidates.json'));
assert.equal(hash(f05Bytes),'66493849a0fa57eaba303e576077d40e8ad5779118617fa0a34d982231a73d2b');
const f05=JSON.parse(f05Bytes),sourceEvidencePath=resolve(values['source-evidence']),sourceEvidence=await json(sourceEvidencePath);
assert.equal(sourceEvidence.stage,'S09');assert.equal(sourceEvidence.group,'L01');assert.equal(sourceEvidence.status,'PASS');
const source=sourceEvidence.checks.find(row=>row.branch==='f05');assert.ok(source);assert.equal(source.rows,1280);assert.deepEqual(source.dimensions,[9,2]);
const sourceRoot=resolve(source.artifact_path),npzBytes=await readFile(resolve(sourceRoot,'expert.npz')),metaBytes=await readFile(resolve(sourceRoot,'artifact.json'));
assert.equal(hash(npzBytes),source.expert_npz_sha256);assert.equal(hash(metaBytes),source.metadata_sha256);
// This read-only independent process re-resolves every byte/parent through the
// actual source Workspace. It extracts records already made by the external QP;
// no NN fixture, replacement teacher, or new expert action is generated here.
const extractor=String.raw`import hashlib,json,sys,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from control_harness.artifacts import Workspace
root,metadata=Workspace(sys.argv[2]).resolve(sys.argv[3],kind='external_expert_dataset')
description=json.loads((root/'dataset.json').read_text());split=json.loads((root/'split.json').read_text())
with np.load(root/'expert.npz',allow_pickle=False) as a:
 rows=[[float(a['time'][k]),int(a['trajectory_id'][k]),*a['X'][k].tolist(),*a['y'][k].tolist()] for k in range(len(a['X']))]
 print(json.dumps({'rows':rows,'interface':description['interface'],'feature_order':description['feature_order'],'target_order':description['target_order'],'split':split,'source':metadata['source'],
  'metadata_hash':Workspace(sys.argv[2]).manifest['artifacts'][sys.argv[3]]['metadata_hash'],'npz_sha256':hashlib.sha256((root/'expert.npz').read_bytes()).hexdigest()},allow_nan=False))`;
const original=JSON.parse(py(extractor,[resolve(runRoot,'../../..','src'),source.workspace,source.artifact_id]));
assert.equal(original.rows.length,1280);assert.equal(original.npz_sha256,source.expert_npz_sha256);
assert.equal(original.interface.input_dim,9);assert.equal(original.interface.output_dim,2);assert.equal(original.source.kind,'declared_simulation');
const expectedGroups={train:[],validation:[],test:[]};f05.collection_scenarios.forEach((s,k)=>expectedGroups[s.split].push(k));assert.deepEqual(original.split.groups,expectedGroups);
const featureNames=original.feature_order.map(x=>x.signal),targetNames=original.target_order.map(x=>x.signal),headers=['time','trajectory_id',...featureNames,...targetNames];
assert.equal(new Set(headers).size,13);
const preRegistered={expert_source_evidence:sourceEvidencePath,source_artifact:source.artifact_id,source_metadata_hash:original.metadata_hash,source_npz_sha256:source.expert_npz_sha256,
  frozen_f05_sha256:hash(f05Bytes),source_dimensions:[9,2],rows:1280,headers,split_counts:{train:20,validation:6,test:6},
  dataset_origin:'Actual L01 external fixed-QP closed-loop actions; no labels generated for import',action_constraints:{bounds:[[-1.5,1.5],[-1.5,1.5]],rate_bounds:[[-.25,.25],[-.25,.25]],rate_semantics:'per_control_step'},
  source_mapping_semantics:'Each previous_action column already contains actual applied u[k-1]; lag=0, no additional shift',training:'not_run'};
const observer=await createObserver(values,'L03',preRegistered),{evidence,save}=observer;
const xlsx=String.raw`import json,sys,openpyxl
from pathlib import Path
fixture=json.loads(Path(sys.argv[1]).read_text());book=openpyxl.Workbook();sheet=book.active;sheet.title='expert'
sheet.append(fixture['headers'])
for row in fixture['rows']:sheet.append(row)
book.save(sys.argv[2]);book.close()
print(json.dumps({'generator':'openpyxl','version':openpyxl.__version__,'rows':len(fixture['rows'])}))`;
const npz=String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as a:print(json.dumps({key:a[key].tolist() for key in a.files},allow_nan=False))`;
const inspectImported=String.raw`import json,sys
sys.path.insert(0,sys.argv[1])
from control_harness.artifacts import Workspace
from control_harness.external_expert_data import read_external_expert_dataset
arrays,interface,dataset,scaler,metadata=read_external_expert_dataset(Workspace(sys.argv[2]),sys.argv[3])
print(json.dumps({'input_dim':interface.input_dim,'output_dim':interface.output_dim,'rows':len(arrays['X']),'scaler':scaler,'source':metadata['source'],'parents':metadata['parents']},allow_nan=False))`;
try{
  const sourceFiles=resolve(dirname(observer.evidencePath),'external-expert-recorded-sources'),rowsFile=resolve(sourceFiles,'actual-expert-records.json');
  await immutableFile(rowsFile,JSON.stringify({headers,rows:original.rows,source_artifact:source.artifact_id,source_npz_sha256:source.expert_npz_sha256}));
  const csvPath=resolve(sourceFiles,'actual-expert.csv'),xlsxPath=resolve(sourceFiles,'actual-expert.xlsx');await immutableFile(csvPath,[headers,...original.rows].map(row=>row.join(',')).join('\n')+'\n');
  try{await readFile(xlsxPath);}catch(error){if(error.code!=='ENOENT')throw error;const result=JSON.parse(py(xlsx,[rowsFile,xlsxPath]));assert.equal(result.rows,1280);evidence.xlsx_generator=result;await save();}
  const hashes={csv:hash(await readFile(csvPath)),xlsx:hash(await readFile(xlsxPath))};
  if(evidence.generated_source_hashes)assert.deepEqual(evidence.generated_source_hashes,hashes,'Generated recorded sources changed on resume');else evidence.generated_source_hashes=hashes;
  if(values['prepare-only']){evidence.status='PREPARED';await save();process.exitCode=0;}
  else{
    evidence.status='RUNNING';await save();
    for(const format of ['csv','xlsx']){
      const workspace=await observer.workspace(format),local=resolve(workspace,`source-inputs/actual-expert.${format}`);await immutableFile(local,await readFile(format==='csv'?csvPath:xlsxPath));
      const material=await observer.stage(format,'material','lightweight','register_material',{local_path:local},{source_metadata:{kind:'actual_L01_external_QP_recording',artifact:source.artifact_id,npz_sha256:source.expert_npz_sha256,confirmation_source:'test_fixture'}});
      const declaration=structuredClone(original.interface),byName=Object.fromEntries(declaration.signals.map(s=>[s.name,s])),worksheet=format==='xlsx'?'expert':null;
      declaration.task='dynamic_prediction';declaration.time_column={index:0,header:'time',worksheet};declaration.trip_column={index:1,header:'trajectory_id',worksheet};
      declaration.signals=[...featureNames,...targetNames].map((name,k)=>({...structuredClone(byName[name]),column:{index:k+2,header:name,worksheet}}));
      delete declaration.source_mapping;delete declaration.source_format;delete declaration.column_semantics;
      const mappings={signals:{}};for(const name of featureNames){if(byName[name].expert_role==='previous_action')mappings.signals[name]={time_semantics:'already_previous_applied_action',action_target:'expert_action::'+name.split('::')[1]};}
      const settings={interface:declaration,source_mapping:mappings,split:{mode:'trip',train:20/32,validation:6/32,test:6/32},scaler:'standard',
        source:{kind:'recorded_external_expert',description:'Actual recorded external NumPy fixed-QP expert actions from S09 L01 simulator collection; no new measurements or replacement labels',previous_action_semantics:'actual_applied_action_before_current_decision',
          original_source_kind:'declared_simulation',source_artifact:source.artifact_id,source_npz_sha256:source.expert_npz_sha256,measured:false},action_constraints:preRegistered.action_constraints,...(worksheet?{worksheet}:{})};
      const inputs={material:material.artifact_id};await observer.deny(format,`${format}:missing-signals-confirmation`,'lightweight','import_expert_data',inputs,settings);
      const refs=await observer.confirm(format,'import','lightweight','import_expert_data',inputs,settings);assert.deepEqual(Object.keys(refs),['signals']);
      const changed=structuredClone(settings);changed.interface.signals[0].unit='changed-unit';
      await observer.deny(format,`${format}:changed-interface-with-old-confirmation`,'lightweight','import_expert_data',inputs,{...changed,plan_refs:refs});
      const result=await observer.stage(format,'import','lightweight','import_expert_data',inputs,{...settings,plan_refs:refs});assert.equal(result.kind,'external_expert_dataset');assert.equal(result.status,'verified');
      const root=result.artifact_path,meta=await json(resolve(root,'artifact.json')),dataset=await json(resolve(root,'dataset.json')),scaler=await json(resolve(root,'scaler.json')),split=await json(resolve(root,'split.json')),report=await json(resolve(root,'report.json'));
      const arrays=JSON.parse(py(npz,[resolve(root,'expert.npz')]));assert.equal(arrays.X.length,1280);near(arrays.X,original.rows.map(row=>row.slice(2,11)),'same recorded causal features',1e-12,1e-12);near(arrays.y,original.rows.map(row=>row.slice(11,13)),'same actual external expert labels',1e-12,1e-12);
      assert.deepEqual(split.sample_counts,{train:800,validation:240,test:240});assert.equal(split.split_before_windowing,true);assert.equal(dataset.input_dim,9);assert.equal(dataset.output_dim,2);assert.equal(dataset.plant_dimensions_inferred,false);assert.equal(dataset.diagnostic_memory_is_feature,false);
      assert.equal((await json(resolve(root,'physical_context.json'))).interface,null);assert.ok(!Object.hasOwn(meta.interface,'nx'));assert.ok(!Object.hasOwn(meta.source,'controller_id'));
      near(arrays.actual_previous_u,arrays.X.map(row=>row.slice(7,9)),'source already previous action, no additional lag');
      const train=arrays.X.filter((_,k)=>arrays.split_id[k]===0),trainY=arrays.y.filter((_,k)=>arrays.split_id[k]===0);
      for(const [key,rows]of [['X',train],['y',trainY]]){
        const means=rows[0].map((_,j)=>rows.reduce((sum,row)=>sum+row[j],0)/rows.length),std=means.map((mean,j)=>Math.sqrt(rows.reduce((sum,row)=>sum+(row[j]-mean)**2,0)/rows.length)||1);
        near(scaler[key].location,means,'independent train-only mean',1e-12,1e-12);near(scaler[key].scale,std,'independent train-only population std',1e-12,1e-12);
        near(arrays[key+'_scaled'],arrays[key].map(row=>row.map((value,j)=>(value-means[j])/std[j])),'independent transformed rows',1e-12,1e-12);}
      for(const [file,expected]of Object.entries(meta.files))assert.equal(hash(await readFile(resolve(root,file))),expected);
      const reread=JSON.parse(py(inspectImported,[resolve(runRoot,'../../..','src'),workspace,result.artifact_id]));assert.equal(reread.rows,1280);assert.equal(reread.source.kind,'recorded_external_expert');assert.equal(reread.parents[0].id,material.artifact_id);
      assert.equal(report.training,'not_run');assert.ok(report.confirmation_sources.length===1&&report.confirmation_sources[0].confirmation_source==='test_fixture');
      const check={format,status:'PASS',workspace,artifact_id:result.artifact_id,artifact_path:root,material_id:material.artifact_id,rows:1280,dimensions:[9,2],source_npz_sha256:source.expert_npz_sha256,
        imported_npz_sha256:hash(await readFile(resolve(root,'expert.npz'))),metadata_sha256:hash(await readFile(resolve(root,'artifact.json'))),source_file_sha256:hashes[format],split_counts:split.sample_counts,independent_scaler_rows:1280,actual_labels_preserved:true,
        physical_interface:null,internal_controller_id:null,confirmation_sources:report.confirmation_sources,immutable_reresolve:reread};
      const old=evidence.checks.findIndex(row=>row.format===format);if(old<0)evidence.checks.push(check);else evidence.checks[old]=check;await save();
    }
    assert.equal(evidence.checks.length,2);assert.equal(evidence.rejections.length,4);evidence.status='PASS';evidence.error=null;evidence.finished_at=new Date().toISOString();
  }
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:observer.evidencePath,jobs:evidence.admissions.length,checks:evidence.checks.map(row=>({format:row.format,artifact_id:row.artifact_id,rows:row.rows}))}));}
