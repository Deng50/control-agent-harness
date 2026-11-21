/** Run the benchmark spreadsheet through the real native command and worker
 * services with a marked Session fixture. No model provider or Web key. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHost, project } from './history-host.mjs';

const benchmark = resolve(process.env.CONTROL_BENCHMARK_ROOT ?? resolve(project,'../control-agent-benchmark'));
const input = join(benchmark,'cases/B02_excel_only/inputs');
const out = join(project,'test-output/structured-confirmation');
const unpack = result => { assert.equal(result.kind,'success',result.text); return JSON.parse(result.text); };

test('B02 spreadsheet: one native user event covers the prepared dataset and the exact requested MLP export', {
  timeout:120000, skip:!existsSync(join(input,'spring_measurements.xlsx')) && 'B02 benchmark inputs unavailable',
}, async () => {
  await mkdir(out,{recursive:true});
  const workspace = await mkdtemp(join(out,'B02 case '));
  const host = await createHost(workspace,'structured-card-b02',{confirmFixturePlans:false});
  try {
    await mkdir(join(workspace,'.case-plans'),{recursive:true});
    await writeFile(join(workspace,'.case-plans/test-fixture.json'),JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace}));
    const command = (name,args) => host.command(name,{workspace,...args});
    const stage = async args => {
      const started = unpack(await command('control',args));
      const ended = await host.ctx.jobs.wait(started.jobId,45000,host.agent.id);
      assert.equal(ended.status,'completed',ended.detail);
      return unpack(await command('control-status',{attemptId:started.attemptId})).record.result;
    };
    const events = () => host.agent.session.snapshotEvents();
    const registrations = () => events().filter(e=>e.type==='job/registered').length;
    const material = await stage({agent:'model',operation:'register_material',inputs:{local_path:join(input,'spring_measurements.xlsx')}});
    const preparation = {agent:'model',operation:'prepare_supervised_dataset',inputs:{material:material.artifact_id},
      config:JSON.parse(await readFile(join(input,'data_mapping_example.json'),'utf8'))};
    const training = {epochs:1,hidden_dims:[4],input_dim:20,output_dim:2,seed:42};
    const card = unpack(await command('control-prepare-review',{...preparation,next_training:{config:training}}));
    assert.deepEqual(card.requirements.map(row=>row.scope),['signals','training']);
    assert.equal(card.requirements[1].effective_settings.input_dim,20);
    assert.equal(card.requirements[1].effective_settings.activation,'tanh');
    const before = registrations();
    assert.equal((await command('control',{...preparation,card_id:card.card_id})).kind,'error');
    assert.equal(registrations(),before);
    const confirmed = unpack(await command('control-confirm',{card_id:card.card_id}));
    assert.equal(confirmed.confirmation_source,'test_fixture');
    assert.equal(confirmed.statuses.training.source.command_id,confirmed.statuses.signals.source.command_id);

    // Concurrent engineering work in the same workspace must preserve B02.
    const mpc = JSON.parse(await readFile(join(benchmark,'cases/B04_language_design/inputs/design_request_round2.json'),'utf8'));
    const invalidDesign = await command('control-prepare-review',{agent:'controller',operation:'synthesize_mpc',...mpc,
      config:{...mpc.config,weights:{...mpc.config.weights,r:-1}}});
    assert.equal(invalidDesign.kind,'error'); assert.equal(registrations(),before);
    const other = unpack(await command('control-prepare-review',{agent:'controller',operation:'synthesize_mpc',...mpc}));
    assert.notEqual(other.intent_id,card.intent_id);
    assert.equal(other.statuses.signals.confirmation_state,'pending');
    const dataset = await stage({...preparation,card_id:card.card_id});
    const train = {agent:'model',operation:'train_supervised',inputs:{dataset:dataset.artifact_id},config:training,card_id:card.card_id};
    const atDataset = registrations();
    const wrong = await command('control',{...train,config:{...training,epochs:2}});
    assert.equal(wrong.kind,'error'); assert.equal(registrations(),atDataset);
    const model = await stage(train);
    assert.equal(model.kind,'supervised_model'); assert.equal(model.status,'verified');
    const network = JSON.parse(await readFile(join(model.artifact_path,'bundle/network.json'),'utf8'));
    assert.deepEqual([network.input_dim,network.output_dim],[20,2]);
    const exported = JSON.parse(await readFile(join(model.artifact_path,'bundle/bundle_manifest.json'),'utf8'));
    assert.ok(exported.files['model.pt']); assert.ok(exported.files['predict.py']);
    // 保留冻结包及一条留出记录，复跑推理时不再采集或训练。
    const evidenceRoot=join(project,'test-output/p0c-b02');
    await mkdir(evidenceRoot,{recursive:true});
    const isolated=await mkdtemp(join(evidenceRoot,'heldout-'));
    await cp(join(model.artifact_path,'bundle'),join(isolated,'bundle'),{recursive:true});
    const python = join(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
    const testPredictions=join(model.artifact_path,'test_predictions.npz');
    const datasetArrays=join(dataset.artifact_path,'supervised.npz');
    const extract=spawnSync(python,['-I','-c',`
import json, sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as model, np.load(sys.argv[2], allow_pickle=False) as data:
    test_rows = np.flatnonzero(data["split_id"] == 2)
    np.testing.assert_array_equal(model["row_index"], test_rows)
    row = int(test_rows[0])
    np.testing.assert_array_equal(model["truth"][0], data["y"][row])
    print(json.dumps({"row_index": row, "raw": data["X"][row].tolist(), "test_rows": len(test_rows),
                      "truth": model["truth"][0].tolist(), "saved_prediction": model["prediction"][0].tolist()}))
`,testPredictions,datasetArrays],{cwd:isolated,encoding:'utf8',timeout:30000,windowsHide:true});
    assert.equal(extract.status,0,extract.stderr);
    const sample=JSON.parse(extract.stdout);
    assert.equal(sample.raw.length,20);
    assert.equal(sample.truth.length,2);
    const sampleFile=join(isolated,'input.json'), resultFile=join(isolated,'actual.json');
    await writeFile(sampleFile,JSON.stringify({feature_labels:exported.feature_labels,features:[sample.raw]}));
    // -I 忽略 PYTHONPATH；禁止回退导入项目包，即使运行环境中恰好有 editable 安装。
    const probe=spawnSync(python,['-I','-c',`
import runpy, sys
class NoProjectPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'control_harness':
            raise RuntimeError("Frozen inference requested the source project")
sys.meta_path.insert(0, NoProjectPackage())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
assert not any(name.split('.')[0] == 'control_harness' for name in sys.modules)
`,'bundle/predict.py','--input','input.json','--output','actual.json'],
      {cwd:isolated,encoding:'utf8',timeout:30000,windowsHide:true});
    assert.equal(probe.status,0,probe.stderr);
    const prediction = JSON.parse(await readFile(resultFile,'utf8'));
    assert.deepEqual(prediction.output_units,['m','m/s']);
    assert.equal(prediction.predictions[0].length,2);
    const delta=prediction.predictions[0].map((value,index)=>Math.abs(value-sample.saved_prediction[index]));
    assert.ok(delta.every(value=>Number.isFinite(value) && value<=1e-12),JSON.stringify(delta));
    assert.deepEqual(prediction.target_labels,exported.target_labels);
    const digest=async path=>createHash('sha256').update(await readFile(path)).digest('hex');
    const materialManifest=JSON.parse(await readFile(join(material.artifact_path,'material.json'),'utf8'));
    const workbook=await digest(join(input,'spring_measurements.xlsx'));
    assert.equal(materialManifest.file_hashes[materialManifest.primary_file],workbook);
    const evidence={material:material.artifact_id,dataset:dataset.artifact_id,model:model.artifact_id,
      workbook_sha256:workbook,bundle_manifest_sha256:await digest(join(isolated,'bundle/bundle_manifest.json')),
      dataset_arrays_sha256:await digest(datasetArrays),saved_predictions_sha256:await digest(testPredictions),
      training_epochs:training.epochs,comparison_rows:1,available_test_rows:sample.test_rows,
      dataset_row_index:sample.row_index,truth:sample.truth,saved_prediction:sample.saved_prediction,
      isolated_prediction:prediction.predictions[0],absolute_difference:delta,absolute_tolerance:1e-12,
      output_units:prediction.output_units,verified:'isolated_prediction_matches_saved_test_prediction',
      replay:'python -I bundle/predict.py --input input.json --output actual.json'};
    await writeFile(join(isolated,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');
    console.log('B02 held-out inference',JSON.stringify({isolated,workbook_sha256:workbook,absolute_difference:delta}));
    assert.equal(events().filter(e=>e.type==='command/run' && e.data.name==='control-confirm').length,1);
    // Dropping training from this intent closes the original continuation.
    const shortened = unpack(await command('control-prepare-review',{...preparation,intent_id:card.intent_id}));
    assert.equal(shortened.statuses.signals.confirmation_state,'pending');
    assert.equal((await command('control',train)).kind,'error');
  } finally {
    await host.close();
    await rm(workspace,{recursive:true,force:true});
  }
});
