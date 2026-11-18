"""Independent real-GA baseline retention, held-out-test isolation and portable replay.

No evaluator, model loader, MPC solver or genetic operator is mocked. These are
numerical tests, not native Host/MATLAB/LLM acceptance.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import torch

from control_harness.artifacts import Workspace,read_json
from control_harness.control import design_controller
from control_harness.modeling import register_model
from control_harness.runtime import OperationContext

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'test-output/v2-development/20261117T002006/fixtures'
OUTPUT=Path(os.environ.get('CONTROL_TEST_OUTPUT',ROOT/'test-output/tuning-acceptance-tests'))


class TuningAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True,exist_ok=True);cls.root=OUTPUT/('real-ga-'+uuid.uuid4().hex[:8]);cls.root.mkdir()
        cls.addClassCleanup(cls.cleanup)
        old_threads=torch.get_num_threads();torch.set_num_threads(1);cls.addClassCleanup(torch.set_num_threads,old_threads)
        raw=(FIXTURES/'F02_existing_linear_4x2.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest()=='ac6364b26915a81abc790a88c6346e2fe91fdd62950898ecac0cc0c67f6ae7d9'
        cls.frozen=json.loads(raw)
        f05_raw=(FIXTURES/'F05_expert_and_tradeoff_candidates.json').read_bytes()
        assert hashlib.sha256(f05_raw).hexdigest()=='66493849a0fa57eaba303e576077d40e8ad5779118617fa0a34d982231a73d2b'
        f05=json.loads(f05_raw);cls.workspace=Workspace(cls.root/'workspace')
        nominal=register_model(cls.workspace,cls.frozen['nominal'],{'kind':'user_nominal','description':'Frozen synthetic F02 predictor'})
        reference=register_model(cls.workspace,cls.frozen['reference'],{'kind':'reference_simulator','description':'Frozen synthetic F02 evaluation model'})
        cls.config=copy.deepcopy(f05['source_expert_controller']['config'])
        cls.config['minimum_improvement']=1.0  # A strict >100% gain is impossible for this nonnegative objective.
        cls.config['evaluation']={'schema_version':1,'recipe':'tracking','energy_model':'action_proxy',
            'weights':copy.deepcopy(f05['manual_scoring_oracle']['weights']['tracking']),
            'scales':copy.deepcopy(f05['manual_scoring_oracle']['scales']),'hard_gates':{'constraint_tolerance':1e-6}}
        cls.changed=copy.deepcopy(cls.config)
        cls.changed['test_scenarios'][0].update(name='held-out-test-only-intervention',initial_state=[-.2,.1,.05,-.08],reference=[-.25,.1])
        # Both configurations are fixed before either candidate search starts.
        cls.pre_registration={'original':copy.deepcopy(cls.config),'test_only_intervention':copy.deepcopy(cls.changed),
            'expected':'same actual candidate and verification scores/selection; only final held-out test changes'}
        cls.results=[]
        for config in (cls.config,cls.changed):
            events=[];artifact_id=design_controller(cls.workspace,nominal,reference,config,OperationContext(emit=events.append))
            root,meta=cls.workspace.resolve(artifact_id,kind='controller')
            cls.results.append({'id':artifact_id,'root':root,'meta':meta,'report':read_json(root/'report.json'),
                'descriptor':read_json(root/'controller.json'),'events':events})

    @classmethod
    def cleanup(cls):
        if hasattr(cls,'root'):
            assert cls.root.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(cls.root,ignore_errors=True)

    def test_real_search_without_required_improvement_retains_actual_feasible_baseline(self):
        for config,result in zip((self.config,self.changed),self.results):
            report=result['report'];descriptor=result['descriptor']
            self.assertEqual(report['algorithm'],'genetic_algorithm');self.assertEqual(len(report['trials']),8)
            self.assertTrue(report['baseline']['hard_constraints_passed']);self.assertTrue(report['test']['hard_constraints_passed'])
            self.assertFalse(report['tuned_candidate_accepted']);self.assertEqual(report['selection'],'baseline_retained')
            expected={'max_iterations':80,'solver_tolerance':1e-8,**config['parameters']}
            self.assertEqual(report['baseline_parameters'],expected)
            self.assertEqual(report['chosen_parameters'],expected);self.assertEqual(descriptor['parameters'],expected)
            self.assertEqual(report['test_used_for_tuning'],False)
            self.assertTrue((result['root']/'bundle/controller.py').is_file())
            progress=[row for row in result['events'] if row.get('stage')=='controller_tuning']
            self.assertEqual(len(progress),8)
            self.assertIsInstance(report['failed_candidates'],list);self.assertIsNotNone(report['best']);self.assertIsNotNone(report['last'])

    def test_changing_only_final_test_never_changes_search_or_verification_selection(self):
        a,b=[r['report'] for r in self.results]
        self.assertEqual(self.config['tuning_scenarios'],self.changed['tuning_scenarios'])
        self.assertEqual(self.config['verification_scenarios'],self.changed['verification_scenarios'])
        self.assertEqual(self.config['evaluation'],self.changed['evaluation'])
        self.assertNotEqual(self.config['test_scenarios'],self.changed['test_scenarios'])
        for left,right in zip(a['trials'],b['trials']):
            self.assertEqual(left['parameters'],right['parameters']);self.assertEqual(left['score'],right['score'])
            self.assertEqual(left['hard_constraints_passed'],right['hard_constraints_passed'])
            self.assertEqual(left['generation'],right['generation']);self.assertEqual(left['lineage'],right['lineage'])
        for key in ('baseline','verification_baseline','verification_candidate'):
            self.assertEqual(a[key]['score'],b[key]['score']);self.assertEqual(a[key]['hard_constraints_passed'],b[key]['hard_constraints_passed'])
        self.assertEqual(a['chosen_parameters'],b['chosen_parameters']);self.assertEqual(a['selection'],b['selection'])
        self.assertEqual(a['test']['runs'][0]['scenario'],self.config['test_scenarios'][0])
        self.assertEqual(b['test']['runs'][0]['scenario'],self.changed['test_scenarios'][0])
        self.assertNotEqual(a['test']['score'],b['test']['score'])

    def test_chosen_bundle_reloads_without_harness_and_replays_independent_frozen_matrix_plant(self):
        code=r'''import importlib.abc,importlib.util,json,math,sys
from pathlib import Path
import numpy as np,torch
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('No Harness import')
sys.meta_path.insert(0,Block());torch.set_num_threads(1)
root=Path(sys.argv[1]);frozen=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));report=json.loads((root/'report.json').read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('portable_test_controller',root/'bundle/controller.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
policy=module.load_design(root/'bundle').instantiate();plant=frozen['reference'];A=plant['A'];B=plant['B'];E=plant['E'];C=plant['C'];i=plant['interface']
def product(M,v):return [math.fsum(m*v[k] for k,m in enumerate(row)) for row in M]
maximum=0.
for saved in report['test']['runs']:
 s=saved['scenario'];x=list(s['initial_state']);d=s['disturbance'];r=s['reference'];previous=[min(max(0.,v['lower']),v['upper']) for v in i['controls']];policy.warm=None
 for record in saved['trajectory']:
  action,diag=policy.act(x,r,d,previous);action=action.tolist();ax=product(A,x);bu=product(B,action);ed=product(E,d);xn=[a+b+e for a,b,e in zip(ax,bu,ed)];y=product(C,xn)
  for actual,expected in ((x,record['x']),(action,record['u']),(xn,record['next_x']),(y,record['output'])):
   np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5);maximum=max(maximum,float(np.max(np.abs(np.asarray(actual)-expected))))
  x=xn;previous=action
assert {k:policy.config[k] for k in report['chosen_parameters']}==report['chosen_parameters']
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
print(json.dumps({'status':'PASS','maximum_error':maximum,'independent_matrix_plant':True,'harness_imports':False}))'''
        result=self.results[0]
        run=subprocess.run([sys.executable,'-I','-B','-c',code,str(result['root']),str(FIXTURES/'F02_existing_linear_4x2.json')],
            cwd=self.root,env={**os.environ,'PYTHONPATH':'','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr);output=json.loads(run.stdout)
        self.assertEqual(output['status'],'PASS');self.assertFalse(output['harness_imports']);self.assertTrue(output['independent_matrix_plant'])

    def test_native_script_pure_auditor_and_embedded_replay_accept_real_report_and_reject_corruption(self):
        # Execute only the smoke's pure audit function and isolated numerical
        # child, never its native/RPC top level or a Host/MATLAB operation.
        source_path=ROOT/'scripts/smoke-goal-tuning.mjs';source=source_path.read_text(encoding='utf-8')
        report_path=self.root/'native-auditor-real-report.json';report_path.write_text(json.dumps(self.results[0]['report']),encoding='utf-8')
        node=os.environ.get('CONTROL_NODE') or shutil.which('node') or str(Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/node/bin/node.exe'))
        self.assertTrue(Path(node).is_file(),'Node is required for independent native-smoke auditor verification')
        code=r'''const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(process.argv[1],'utf8'),report=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),frozen=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const near=(a,b,label,atol=1e-6,rtol=1e-5)=>{if(Array.isArray(b)){assert.ok(Array.isArray(a),label);assert.equal(a.length,b.length);b.forEach((v,k)=>near(a[k],v,label,atol,rtol));}else assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=atol+rtol*Math.abs(b),label);};
const start=source.indexOf('function auditEvaluation('),end=source.indexOf('// The controller is loaded',start);assert.ok(start>=0&&end>start);
const audit=vm.runInNewContext('('+source.slice(start,end)+')',{assert,frozen,near});let count=0;
for(const key of ['baseline','verification_baseline','verification_candidate','test']){audit(report[key],report.evaluation,key);count++;}
for(const trial of report.trials){if(trial.tuning_evaluation){audit(trial.tuning_evaluation,report.evaluation,'trial'+trial.index);count++;}}
const broken=JSON.parse(JSON.stringify(report.test));broken.runs[0].objective_metrics.rmse+=.5;assert.throws(()=>audit(broken,report.evaluation,'corrupt recorded RMSE'));
console.log(JSON.stringify({status:'PASS',actual_evaluation_groups:count,corruption_rejected:true,native_calls:0}));'''
        run=subprocess.run([node,'-e',code,str(source_path),str(report_path),str(FIXTURES/'F02_existing_linear_4x2.json')],
            cwd=self.root,capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr);audit=json.loads(run.stdout)
        self.assertEqual(audit['actual_evaluation_groups'],12);self.assertTrue(audit['corruption_rejected']);self.assertEqual(audit['native_calls'],0)
        replay=re.search(r'const independentReplay=String.raw`([\s\S]*?)`;',source).group(1)
        child=subprocess.run([sys.executable,'-I','-B','-c',replay,str(self.results[0]['root']/'bundle'),str(FIXTURES/'F02_existing_linear_4x2.json'),str(report_path)],
            cwd=self.root,env={**os.environ,'PYTHONPATH':'','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(child.returncode,0,child.stdout+child.stderr);replayed=json.loads(child.stdout)
        self.assertEqual(replayed['status'],'PASS');self.assertFalse(replayed['project_imports'])


if __name__=='__main__':unittest.main()
