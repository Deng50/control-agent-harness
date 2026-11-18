"""Public S12 confirmation/dispatch; Native and final PGAC gates are separate."""
from __future__ import annotations
import copy
from pathlib import Path
import unittest
from control_harness.artifacts import read_json
from control_harness.case_plans import PendingPlanConfirmation,propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.inspection import prepare_handoff
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
import test_differentiable_validation as validation_fixtures


class PublicDifferentiableOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):validation_fixtures.DifferentiableArtifactTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):validation_fixtures.DifferentiableArtifactTests.tearDownClass.__func__(cls)
    def setUp(self):
        validation_fixtures.DifferentiableArtifactTests.setUp(self)
        self.context=validation_fixtures.fixture_context(self.ws)
        self.request={'operation':'validate_differentiable_model','workspace':str(self.ws.path),
            'inputs':{'model':self.candidate,'evaluation_model':self.original,'dataset':self.data},
            'config':{'mode':'exact_rewrite',**copy.deepcopy(self.config)}}
    def approve(self,request):
        refs={}
        for requirement in confirmation_requirements(self.ws,request):
            plan=propose_plan(self.ws,requirement['scope'],requirement['expected_payload'],requirement['bindings'])
            refs[requirement['scope']]={key:plan[key] for key in ['plan_id','version_hash']}
        request['config']['plan_refs']=refs
    def test_full_validation_summary_requires_confirmed_version_before_any_candidate_stage(self):
        before=copy.deepcopy(self.ws.manifest)
        with self.assertRaises(PendingPlanConfirmation):run_operation(self.request,OperationContext())
        self.assertEqual(self.ws.manifest,before)
        self.approve(self.request)
        for patch in [{'numeric':{'atol':1e-7,'rtol':1e-6}},{'steps':49},{'mode':'existing_torch'}]:
            revised=copy.deepcopy(self.request);revised['config'].update(patch)
            with self.assertRaises(PendingPlanConfirmation):run_operation(revised,self.context)
        self.assertEqual(self.ws.manifest,before)
    def test_actual_dispatch_binds_receipt_and_handoff_uses_exact_validated_model(self):
        self.approve(self.request);result=run_operation(self.request,self.context)
        self.assertEqual(result['kind'],'differentiable_validation');self.assertEqual(result['status'],'verified')
        root=Path(result['artifact_path']);report=read_json(root/'report.json');metadata=read_json(root/'artifact.json')
        self.assertTrue(report['qualified_for_pgac']);self.assertEqual(metadata['activation'],'none')
        self.assertEqual([row['scope'] for row in report['confirmation_sources']],['signals'])
        self.assertEqual(report['confirmation_sources'][0]['confirmation_source'],'test_fixture')
        handoff=prepare_handoff(self.ws,result['artifact_id'])['handoff']
        self.assertEqual(handoff['operation'],'optimize_external_policy')
        self.assertEqual(handoff['inputs'],{'model':self.candidate,'validation':result['artifact_id']})
        self.assertEqual(handoff['missing_inputs'],['policy'])
        self.assertEqual(handoff['config_suggestions']['domain'],report['coverage']['domain'])


class PublicRestrictedPgacOperations(unittest.TestCase):
    """Append-only real scalar public dispatch/authorization/recovery cases."""
    @classmethod
    def setUpClass(cls):
        import test_pgac_gradients as pgac_fixtures
        pgac_fixtures.RestrictedPgacTrainingTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        import test_pgac_gradients as pgac_fixtures
        pgac_fixtures.RestrictedPgacTrainingTests.tearDownClass.__func__(cls)

    def setUp(self):
        import test_pgac_gradients as pgac_fixtures
        from control_harness.artifacts import atomic_json
        from control_harness.case_plans import FIXTURE_KIND
        pgac_fixtures.RestrictedPgacTrainingTests.setUp(self)
        atomic_json(self.ws.path/'.case-plans/test-fixture.json',{'kind':FIXTURE_KIND,'workspace':str(self.ws.path)})
        self.context=OperationContext.test_fixture(self.ws)
        self.settings={**copy.deepcopy(self.config),'epochs':1}
        self.request={'operation':'optimize_external_policy','workspace':str(self.ws.path),
            'inputs':{'policy':self.policy,'model':self.model,'validation':self.validation},'config':self.settings}

    approve=PublicDifferentiableOperations.approve

    def candidate(self):
        self.approve(self.request);return run_operation(self.request,self.context)

    def test_public_actual_pgac_dispatch_records_fixture_scopes_and_exact_original_handoff(self):
        requirements=confirmation_requirements(self.ws,self.request)
        self.assertEqual([row['scope'] for row in requirements],['signals','training'])
        before=copy.deepcopy(self.ws.manifest)
        with self.assertRaises(PendingPlanConfirmation):run_operation(self.request,OperationContext())
        self.assertEqual(self.ws.manifest,before)
        result=self.candidate();self.assertEqual(result['kind'],'external_pgac_candidate');self.assertEqual(result['status'],'verified')
        root=Path(result['artifact_path']);report=read_json(root/'report.json');metadata=read_json(root/'artifact.json')
        self.assertEqual(metadata['activation'],'none');self.assertEqual(report['optimizer_updates'],3)
        self.assertEqual([row['scope'] for row in report['confirmation_sources']],['signals','training'])
        self.assertTrue(all(row['confirmation_source']=='test_fixture' for row in report['confirmation_sources']))
        self.assertEqual(report['model_validation'],self.validation);self.assertFalse(report['test_used_for_updates_or_checkpoint_selection'])
        handoff=prepare_handoff(self.ws,result['artifact_id'])['handoff']
        self.assertEqual(handoff['operation'],'evaluate_external_pgac');self.assertEqual(handoff['inputs'],{'candidate':result['artifact_id']})
        self.assertEqual(handoff['config_suggestions'],report['original_evaluation_config'])
        self.assertEqual(self.ws.manifest['active']['external_imitation_policy'],self.policy)

    def test_changed_optimizer_domain_and_real_validation_version_invalidate_old_refs(self):
        from control_harness.differentiable_validation import read_differentiable_validation,validate_differentiable_model
        initial=confirmation_requirements(self.ws,self.request);self.approve(self.request)
        for patch in ({'learning_rate':.0007},{'domain':{'x':[[-2.,2.]],'u':[[-1.,1.]],'d':[]}}):
            changed=copy.deepcopy(self.request);changed['config'].update(patch);rows=confirmation_requirements(self.ws,changed)
            self.assertEqual(rows[0],initial[0]);self.assertNotEqual(rows[1]['expected_payload'],initial[1]['expected_payload'])
            before=copy.deepcopy(self.ws.manifest)
            with self.assertRaises(PendingPlanConfirmation):run_operation(changed,self.context)
            self.assertEqual(self.ws.manifest,before)
        receipt=read_differentiable_validation(self.ws,self.validation);config=receipt['resolved_config']
        config={key:copy.deepcopy(config[key]) for key in ('mode','partition','steps','probes','numeric','gradient','domain','switching_planes')};config['steps']=49
        other=validate_differentiable_model(self.ws,self.plant,self.model,config,reference_dataset_id=receipt['bindings']['dataset']['id'])
        revised=copy.deepcopy(self.request);revised['inputs']['validation']=other
        self.assertNotEqual(confirmation_requirements(self.ws,revised)[0]['bindings'],initial[0]['bindings'])
        before=copy.deepcopy(self.ws.manifest)
        with self.assertRaises(PendingPlanConfirmation):run_operation(revised,self.context)
        self.assertEqual(self.ws.manifest,before)

    def test_request_json_cannot_forge_receipt_and_evaluation_requires_exact_frozen_conditions(self):
        from control_harness.contracts import ContractError
        from unittest.mock import patch
        self.approve(self.request);forged=copy.deepcopy(self.request);forged.update(confirmed=True,confirmation_source='native_user',command_id='invented')
        before=copy.deepcopy(self.ws.manifest)
        with self.assertRaises(PendingPlanConfirmation):run_operation(forged,OperationContext())
        self.assertEqual(self.ws.manifest,before)
        result=run_operation(self.request,self.context);report=read_json(Path(result['artifact_path'])/'report.json')
        evaluation={'operation':'evaluate_external_pgac','workspace':str(self.ws.path),'inputs':{'candidate':result['artifact_id']},'config':copy.deepcopy(report['original_evaluation_config'])}
        self.assertEqual([row['scope'] for row in confirmation_requirements(self.ws,evaluation)],['signals','sampling'])
        for config in ({},{'scenarios':[]}):
            with self.assertRaisesRegex(ContractError,'scenarios'):confirmation_requirements(self.ws,{**evaluation,'config':config})
        changed=copy.deepcopy(evaluation);changed['config']['evaluation']['q']*=1.1;self.approve(changed)
        before=copy.deepcopy(self.ws.manifest)
        with patch('control_harness.model_adapters.load_python_adapter',side_effect=AssertionError('Rejected configuration must not execute original source')):
            with self.assertRaisesRegex(ContractError,'changed after PGAC training'):run_operation(changed,self.context)
        self.assertEqual(self.ws.manifest,before)

    def test_actual_hard_exit_public_recovery_resumes_the_real_pgac_checkpoint_with_same_refs(self):
        import os,subprocess,sys
        from control_harness.artifacts import atomic_json
        self.approve(self.request);self.request['config']['epochs']=2;self.approve(self.request)
        atomic_json(self.ws.path/'hard-exit-request.json',self.request)
        source=r'''import json,os,sys,torch
from pathlib import Path
from control_harness.artifacts import Workspace,atomic_json
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
torch.set_num_threads(1);root=Path(sys.argv[1]);request=json.loads((root/'hard-exit-request.json').read_text())
def emit(event):
 if event.get('stage')=='restricted_pgac_training' and event.get('epoch')==1:
  atomic_json(root/'hard-exit-ready.json',{'stage_id':Path(event['checkpoint']).parent.name,'pid':os.getpid()});os._exit(73)
run_operation(request,OperationContext.test_fixture(Workspace(root),emit=emit))
raise RuntimeError('Expected a real hard exit after the first committed PGAC epoch')
'''
        project=Path(__file__).resolve().parents[1]
        child=subprocess.run([sys.executable,'-B','-c',source,str(self.ws.path)],capture_output=True,text=True,timeout=90,env={**os.environ,'PYTHONPATH':str(project/'src'),'PYTHONUTF8':'1'})
        self.assertEqual(child.returncode,73,child.stdout+child.stderr)
        marker=read_json(self.ws.path/'hard-exit-ready.json')
        recovered=run_operation({'operation':'recover_interrupted_stage','workspace':str(self.ws.path),'inputs':{'stage_id':marker['stage_id']},'config':{}},self.context)
        self.assertEqual(recovered['checkpoint_kind'],'external_pgac_candidate');self.assertEqual(recovered['algorithm'],'external_physical_policy_optimization')
        self.assertEqual(recovered['epoch'],1);self.assertEqual(recovered['optimizer_updates'],3)
        refs=copy.deepcopy(self.request['config']['plan_refs']);resume=copy.deepcopy(self.request)
        resume['config'].update(resume_checkpoint=recovered['checkpoint'],resume_checkpoint_sha256=recovered['checkpoint_sha256'])
        result=run_operation(resume,self.context);report=read_json(Path(result['artifact_path'])/'report.json')
        self.assertEqual(resume['config']['plan_refs'],refs);self.assertEqual(report['resumed_from_epoch'],1);self.assertEqual(report['optimizer_updates'],6)
        receipt=read_json(Path(recovered['checkpoint']).with_name('interrupted-stage-recovery.json'))
        self.assertEqual(receipt['owner_verification'],'os_confirmed_process_exit');self.assertEqual(receipt['original_owner']['pid'],marker['pid'])


if __name__=='__main__':unittest.main()
