/** Narrow observer continuation for one recorded, self-corrected read-only call. No RPC. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {resolve,relative,isAbsolute,sep} from 'node:path';
import {isDeepStrictEqual} from 'node:util';

const sha=raw=>createHash('sha256').update(raw).digest('hex');
const json=async path=>JSON.parse((await readFile(path,'utf8')).replace(/^\uFEFF/,''));
const equal=(a,b)=>isDeepStrictEqual(a,b);

export const recoveredInspector={
 sessionId:'session-985a9345-0766-4266-9d95-6e4f0c84fdee',
 attemptId:'4ab63db1-49bb-4f09-9e02-ab08b02a2c35',
 nativeJobId:'control-engineering-d1d79d0c341a4056',
 correctedAttemptId:'4da07e16-eea9-4792-acaf-e218c9a52d79',
 correctedJobId:'control-engineering-a29b7f72514845e3',
 synthesisAttemptId:'1359ff2d-cddd-4d11-93ca-56f107e6f200',
 artifactId:'controller_design-7ad0266a91f9',
 detail:"KeyError: 'artifact_id'",
};

export function isRecordedReadOnlyFailure(attempt,revision){
 const x=revision?.recovered_read_only_inspector;
 return !!x&&attempt.attemptId===x.attemptId&&attempt.nativeJobId===x.nativeJobId&&attempt.owner===x.sessionId&&attempt.state==='failed'&&attempt.detail===x.detail&&attempt.request?.workspace===x.workspace&&attempt.request?.operation==='inspect_artifact'&&equal(attempt.request?.inputs,{artifact:x.artifactId})&&equal(attempt.request?.config,{})&&attempt.nativeSettlement?.status==='failed'&&attempt.nativeSettlement?.detail===x.detail;
}

export async function verifyRecoveredInspector(evidence,revision){
 const x=revision.recovered_read_only_inspector;
 assert.deepEqual(Object.fromEntries(Object.keys(recoveredInspector).map(k=>[k,x[k]])),recoveredInspector);
 assert.equal(evidence.status,'FAIL_RETAIN_ACTUAL_HANDLES_NO_AUTOMATIC_RETRY');
 assert.equal(evidence.error?.name,'AssertionError');
 assert.ok(evidence.error?.message.startsWith('A real native job failed; preserve and report before any further work'));
 assert.equal(evidence.roles.model.status,'PASS');
 assert.equal(evidence.roles.lightweight,undefined);
 const model=evidence.roles.model,modelCheck=evidence.checks.find(v=>v.role==='model');
 assert.equal(evidence.checks.length,1);assert.equal(model.final.artifact_id,modelCheck.artifact_id);
 const modelRoot=resolve(model.final.artifact_path),modelMeta=await json(resolve(modelRoot,'artifact.json'));
 assert.equal(sha(await readFile(resolve(modelRoot,'artifact.json'))),modelCheck.metadata_sha256);
 assert.equal(sha(await readFile(resolve(modelRoot,'report.json'))),modelCheck.report_sha256);
 assert.equal(modelMeta.validation.status,'verified');
 for(const [name,hash]of Object.entries(modelMeta.files)){const p=resolve(modelRoot,name),r=relative(modelRoot,p);assert.ok(r&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep));assert.equal(sha(await readFile(p)),hash);}
 const row=evidence.roles.controller,turn=row.turns.draft;
 assert.equal(row.sessionId,x.sessionId);assert.equal(row.workspace,x.workspace);
 assert.deepEqual(Object.keys(row.turns),['draft']);
 assert.equal(turn.wait.reason,'idle');assert.equal(turn.wait.state.running,false);
 assert.deepEqual(turn.wait.state.queue,[]);assert.deepEqual(turn.wait.state.pendingInput,[]);
 const all=Object.values(evidence.roles).flatMap(v=>v.history.attempts);
 const failures=all.filter(a=>['failed','killed','interrupted'].includes(a.state));
 assert.equal(failures.length,1);assert.ok(isRecordedReadOnlyFailure(failures[0],revision));
 const attempts=row.history.attempts;
 const corrected=attempts.find(a=>a.attemptId===x.correctedAttemptId);
 const synthesis=attempts.find(a=>a.attemptId===x.synthesisAttemptId);
 assert.ok(corrected&&synthesis,'The real successful correction and synthesis are mandatory');
 assert.equal(corrected.nativeJobId,x.correctedJobId);assert.equal(corrected.owner,x.sessionId);
 assert.equal(corrected.state,'completed');assert.equal(corrected.nativeSettlement.status,'completed');
 assert.equal(corrected.request.operation,'inspect_artifact');assert.equal(corrected.request.workspace,x.workspace);
 assert.deepEqual(corrected.request.inputs,{artifact_id:x.artifactId});assert.deepEqual(corrected.request.config,{});
 assert.equal(corrected.result.artifact_id,x.artifactId);assert.equal(corrected.result.status,'verified');
 assert.equal(synthesis.state,'completed');assert.equal(synthesis.request.operation,'synthesize_mpc');
 assert.equal(synthesis.result.artifact_id,x.artifactId);assert.equal(synthesis.result.status,'verified');
 assert.ok(Date.parse(synthesis.createdAt)<Date.parse(failures[0].createdAt));
 assert.ok(Date.parse(failures[0].createdAt)<Date.parse(corrected.createdAt));
 assert.equal(attempts.filter(a=>a.request.operation==='synthesize_mpc').length,1);
 const events=row.events.records.map(r=>r.event??r);
 const end=events.findLast(v=>v.type==='turn/end');assert.equal(end.data.reason.kind,'completed');
 const calls=events.filter(v=>v.type==='tool/call').map(v=>({...v,args:JSON.parse(v.data.arguments)}));
 const bad=calls.find(v=>v.data.name==='control_controller'&&v.args.operation==='inspect_artifact'&&equal(v.args.inputs,{artifact:x.artifactId}));
 const good=calls.find(v=>v.data.name==='control_controller'&&v.args.operation==='inspect_artifact'&&equal(v.args.inputs,{artifact_id:x.artifactId}));
 const help=calls.find(v=>v.data.name==='control_help'&&v.args.operation==='inspect_artifact'&&v.seq>bad?.seq&&v.seq<good?.seq);
 assert.ok(bad&&good&&help&&bad.seq<good.seq&&good.seq<end.seq);
 for(const [call,job]of [[bad,x.nativeJobId],[good,x.correctedJobId]])assert.ok(calls.some(v=>v.data.name==='job_output'&&v.args.job_id===job&&v.seq>call.seq&&v.seq<end.seq));
 const artifactRoot=resolve(synthesis.result.artifact_path),inside=relative(resolve(x.workspace,'artifacts'),artifactRoot);
 assert.ok(inside&&!isAbsolute(inside)&&inside!=='..'&&!inside.startsWith('..'+sep));
 const metadata=await json(resolve(artifactRoot,'artifact.json')),design=await json(resolve(artifactRoot,'design.json'));
 assert.equal(metadata.id,x.artifactId);assert.equal(metadata.kind,'controller_design');assert.equal(metadata.validation.status,'verified');
 assert.equal(design.model_instantiated,false);assert.equal(design.physical_interface,null);assert.equal(design.required_parameters.length,7);
 assert.equal(design.closed_loop.status,'not_run');assert.equal(design.closed_loop.ga_trials,0);assert.equal(design.solver_check.status,'not_run');
 for(const [name,hash]of Object.entries(metadata.files)){const p=resolve(artifactRoot,name),r=relative(artifactRoot,p);assert.ok(r&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep));assert.equal(sha(await readFile(p)),hash);}
 const refs=[];for(const item of revision.shared_reference_files){assert.equal(sha(await readFile(item.path)),item.sha256);refs.push(item);}
 const wrongReference=calls.find(v=>v.data.name==='read'&&v.args.file_path?.endsWith('design-and-tune-mpc\\references\\execution.md'));
 assert.ok(wrongReference,'Retain evidence of the actual relative-reference mistake');
 return{sessionId:x.sessionId,failed_attempt_preserved:x.attemptId,corrected_attempt:corrected.attemptId,artifact_id:x.artifactId,actual_turn_completed:true,actual_help_between_bad_and_good_calls:true,all_draft_file_hashes_verified:true,seven_missing_parameters:true,model_instantiated:false,ga_trials:0,closed_loop:'not_run',shared_reference_files:refs};
}

export const correctedMaterial={
 sessionId:'session-5fed7e1f-de96-479c-9315-7bddaa329baf',
 firstAttemptId:'583ed41e-fd31-4b53-9e39-08bc285f2ff8',firstJobId:'control-engineering-b84502f7df544e14',firstArtifactId:'material-932e13376159',
 selectedAttemptId:'fa17d947-a49b-411b-9bda-e2ff52f8f36d',selectedJobId:'control-engineering-ffcf3db694224aeb',selectedArtifactId:'material-f573656acc06',
 inspectionAttemptId:'67d6b853-c2a6-46b6-8c6e-b7c730ad51b1',inspectionJobId:'control-engineering-c1793b442fb14b48',
 sourceSha256:'e3399370fa48ba3895ccd489ad134d184f7b33ceaf5d747495da16a572f25b33',
 contentHash:'924b5b5bebc067ab0b4eda32f74869731eb8418ed1044c4a05a5a38febb91e11',
};

export async function verifyCorrectedMaterial(evidence,revision){
 const x=revision.corrected_material;
 assert.deepEqual(Object.fromEntries(Object.keys(correctedMaterial).map(k=>[k,x[k]])),correctedMaterial);
 assert.equal(evidence.status,'FAIL_RETAIN_ACTUAL_HANDLES_NO_AUTOMATIC_RETRY');
 assert.equal(evidence.error.name,'AssertionError');assert.ok(evidence.error.message.startsWith('lightweight/register_material must have exactly one actual accepted stage'));
 for(const role of ['model','controller']){
  const row=evidence.roles[role],check=evidence.checks.find(v=>v.role===role);assert.equal(row.status,'PASS');assert.equal(row.final.artifact_id,check.artifact_id);
  const root=resolve(row.final.artifact_path),meta=await json(resolve(root,'artifact.json'));
  assert.equal(sha(await readFile(resolve(root,'artifact.json'))),check.metadata_sha256);assert.equal(sha(await readFile(resolve(root,'report.json'))),check.report_sha256);
  for(const [name,hash]of Object.entries(meta.files)){const p=resolve(root,name),r=relative(root,p);assert.ok(r&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep));assert.equal(sha(await readFile(p)),hash);}
 }
 const row=evidence.roles.lightweight;assert.equal(row.sessionId,x.sessionId);assert.equal(row.workspace,x.workspace);
 assert.deepEqual(Object.keys(row.turns),['material']);assert.equal(row.turns.material.state,'COMPLETED');
 assert.equal(row.turns.material.wait.reason,'idle');assert.equal(row.turns.material.wait.state.running,false);assert.deepEqual(row.turns.material.wait.state.queue,[]);assert.deepEqual(row.turns.material.wait.state.pendingInput,[]);
 const attempts=row.history.attempts;assert.equal(attempts.length,5);assert.ok(attempts.every(a=>a.state==='completed'&&a.owner===x.sessionId));
 assert.ok(attempts.every(a=>['inspect_material','register_material','inspect_artifact'].includes(a.request.operation)));
 const byId=id=>attempts.find(a=>a.attemptId===id),first=byId(x.firstAttemptId),selected=byId(x.selectedAttemptId),inspected=byId(x.inspectionAttemptId);
 for(const [a,job,id]of [[first,x.firstJobId,x.firstArtifactId],[selected,x.selectedJobId,x.selectedArtifactId],[inspected,x.inspectionJobId,x.selectedArtifactId]]){assert.ok(a);assert.equal(a.nativeJobId,job);assert.equal(a.result.artifact_id,id);assert.equal(a.result.status,'verified');}
 assert.equal(first.request.operation,'register_material');assert.equal(selected.request.operation,'register_material');assert.equal(inspected.request.operation,'inspect_artifact');assert.deepEqual(inspected.request.inputs,{artifact_id:x.selectedArtifactId});
 assert.deepEqual(first.request.inputs,selected.request.inputs);assert.equal(first.request.inputs.local_path,resolve(x.workspace,'inputs/external-qp-records.csv'));
 const normalizeConfig=config=>{const copy=structuredClone(config);delete copy.name;delete copy.source_metadata;const manifest=copy.expected_manifest;if(Object.hasOwn(manifest,'dimensions_inferred')){assert.equal(manifest.dimensions_inferred,false);delete manifest.dimensions_inferred;}return copy;};
 assert.deepEqual(normalizeConfig(first.request.config),normalizeConfig(selected.request.config));
 assert.ok(first.request.config.name.includes('12-column'));assert.ok(selected.request.config.name.includes('13-column'));
 assert.equal(selected.request.config.source_metadata.header_verified_against_declared_column_index,true);
 assert.equal(selected.request.config.expected_manifest.content_hash,x.contentHash);
 assert.equal(selected.request.config.expected_manifest.source_executed,false);
 const primaryInventories=[],sourceManifests=[];
 for(const a of [first,selected]){const root=resolve(a.result.artifact_path),meta=await json(resolve(root,'artifact.json'));assert.equal(meta.id,a.result.artifact_id);assert.equal(meta.validation.status,'verified');assert.equal(meta.source.content_hash,x.contentHash);assert.equal(meta.source.source_executed,false);assert.equal(meta.files['source/external-qp-records.csv'],x.sourceSha256);for(const [name,hash]of Object.entries(meta.files)){const p=resolve(root,name),r=relative(root,p);assert.ok(r&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep));assert.equal(sha(await readFile(p)),hash);}primaryInventories.push(Object.fromEntries(Object.entries(meta.files).filter(([name])=>name.startsWith('source/'))));const manifest=await json(resolve(root,'material.json'));assert.equal(manifest.dimensions_inferred,false);assert.equal(manifest.source_executed,false);delete manifest.source.metadata;sourceManifests.push(manifest);}
 assert.deepEqual(primaryInventories[0],primaryInventories[1]);
 assert.deepEqual(sourceManifests[0],sourceManifests[1]);
 const bytes=await readFile(first.request.inputs.local_path);assert.equal(sha(bytes),x.sourceSha256);const lines=bytes.toString('utf8').trimEnd().split(/\r?\n/),header=lines[0];assert.equal(lines.length,1281);assert.equal(header.split(',').length,13);
 const events=row.events.records.map(v=>v.event??v),calls=events.filter(v=>v.type==='tool/call').map(v=>({...v,args:JSON.parse(v.data.arguments)}));
 const raw=calls.find(v=>v.seq===51&&v.data.name==='pwsh'),columns=calls.find(v=>v.seq===101&&v.data.name==='pwsh');assert.ok(raw&&columns);assert.ok(raw.args.command.includes(first.request.inputs.local_path));assert.ok(columns.args.command.includes(first.request.inputs.local_path));
 const resultFor=call=>events.find(v=>v.type==='tool/result'&&v.data.message.toolCallId===call.data.callId);
 const rawResult=resultFor(raw),columnResult=resultFor(columns);assert.ok(rawResult&&columnResult);assert.equal(rawResult.data.message.isError,false);assert.equal(columnResult.data.message.isError,false);
 const text=event=>event.data.message.content.filter(v=>v.type==='text').map(v=>v.text).join('\n');
 assert.ok(text(rawResult).includes(header));assert.ok(text(rawResult).includes(x.sourceSha256.toUpperCase()));
 for(const fragment of ['rows=1280','cols=13','trajectories=32 min_steps=40 max_steps=40','non-0.1s steps=0'])assert.ok(text(columnResult).includes(fragment));
 header.split(',').forEach((name,index)=>assert.ok(text(columnResult).includes(`idx ${index} = ${name}`)));
 const firstCall=calls.find(v=>v.data.name==='control_lightweight'&&v.args.operation==='register_material'&&v.args.config.name===first.request.config.name),selectedCall=calls.find(v=>v.data.name==='control_lightweight'&&v.args.operation==='register_material'&&v.args.config.name===selected.request.config.name);assert.ok(firstCall&&selectedCall&&raw.seq<firstCall.seq&&firstCall.seq<columns.seq&&columns.seq<selectedCall.seq);
 const final=events.findLast(v=>v.type==='assistant/message'),finalText=text(final);assert.ok(finalText.includes('请以 `'+x.selectedArtifactId+'` 为唯一有效材料'));assert.ok(finalText.includes(x.firstArtifactId));
 assert.equal(events.findLast(v=>v.type==='turn/end').data.reason.kind,'completed');
 return{sessionId:x.sessionId,selected_material:selected.result,first_material_preserved:x.firstArtifactId,source_sha256:x.sourceSha256,actual_header_raw_result_seq:rawResult.seq,actual_columns_result_seq:columnResult.seq,actual_final_selection_seq:final.seq,all_material_file_hashes_verified:true,metadata_differences:['name','source_metadata','expected_manifest.dimensions_inferred false omitted from second request'],original_material_turn_reused_no_new_calls:true,no_import_or_training:true};
}
