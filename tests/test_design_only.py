"""Design-only contracts with actual predictors/SLSQP and forbidden-path probes."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace,atomic_json,read_json
from control_harness.contracts import ContractError,ModelInterface
from control_harness.control import MPCPolicy
from control_harness.design_only import synthesize_mpc
from control_harness.modeling import load_model,register_model
import control_harness.control as control


ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class DesignOnlyTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'test-output/design-only-tests';parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.ws=Workspace(self.root/'case')
        self.parameters={'mass':2.,'damping':.8,'stiffness':3.,'sample_time':.1,
            'state_bounds':[[-2.,2.],[-3.,3.]],'input_bounds':[[-4.,4.]],'output_bounds':[[-2.,2.]]}

    def numeric_spec(self,nu=1,nonlinear=False,bias=0.):
        var=lambda name,bound:{'name':name,'unit':'1','lower':-bound,'upper':bound}
        return {'representation':'discrete_polynomial','interface':{'schema_version':1,'name':'explicit design-only unit fixture',
            'sample_time':.2,'time_semantics':'discrete_state_transition','states':[var('state',3)],
            'controls':[var('control'+str(k),1) for k in range(nu)],'outputs':[var('sensor',3)],'disturbances':[]},
            'A':[[.7]],'B':[[.2]+[0.]*(nu-1)],'E':[[]],'C':[[1.]],'bias':[bias],
            'terms':[{'state_index':0,'coefficient':.08,'powers':[2]+[0]*nu}] if nonlinear else []}

    def model(self,**kwargs):return register_model(self.ws,self.numeric_spec(**kwargs),{'kind':'user_nominal'})

    def result(self,inputs,config=None):
        artifact=synthesize_mpc(self.ws,inputs,config or {})
        root,metadata=self.ws.resolve(artifact,allow_rejected=True)
        return artifact,root,metadata,read_json(root/'design.json'),read_json(root/'report.json')

    def test_natural_spring_description_stays_symbolic_without_any_model_or_numerical_run(self):
        with (patch('control_harness.predictors.resolve_predictor',side_effect=AssertionError('must not instantiate missing plant')) as resolver,
             patch.object(control.GeneticOperators,'_initial_population',side_effect=AssertionError('no GA')) as ga,
             patch.object(control,'closed_loop',side_effect=AssertionError('no plant loop')) as loop):
            _,root,metadata,design,report=self.result({'description':'为质量-弹簧-阻尼系统设计MPC，仅生成设计方案'})
        resolver.assert_not_called();ga.assert_not_called();loop.assert_not_called()
        self.assertFalse(design['model_instantiated']);self.assertIsNone(design['physical_interface'])
        self.assertEqual(design['parameters'],{})
        self.assertEqual(set(design['required_parameters']),set(self.parameters))
        self.assertEqual(design['equations']['continuous_matrices']['C'],[[1,0]])
        self.assertEqual(design['status'],'parameterized_draft')
        self.assertEqual(design['solver_check']['status'],'not_run');self.assertEqual(design['closed_loop']['status'],'not_run')
        self.assertEqual(metadata['interface']['interface_kind'],'material');self.assertEqual(metadata['interface']['format'],'bundle')
        self.assertNotIn('states',metadata['interface']);self.assertEqual(metadata['activation'],'none')
        self.assertEqual(report['closed_loop_validation'],'not_run')
        self.assertTrue((root/'bundle/controller.py').is_file());self.assertTrue((root/'bundle/mpc_control.m').is_file())
        self.assertFalse(read_json(root/'bundle/parameter_schema.json')['physical_example_defaults'])

    def test_partial_null_parameters_remain_required_and_text_numbers_are_not_silently_used(self):
        _,_,_,design,_=self.result({'description':'spring m=123 c=45 k=67 dt=.8'},
            {'parameters':{'mass':2.,'damping':None},'weights':{'q':3.},'controller':{'horizon':3}})
        self.assertEqual(design['parameters'],{'mass':2.})
        self.assertNotIn('mass',design['required_parameters']);self.assertIn('damping',design['required_parameters'])
        self.assertIn('sample_time',design['required_parameters']);self.assertEqual(design['weights']['q'],3.)
        self.assertIsNone(design['physical_interface'])
        values=copy.deepcopy(self.parameters);values['mass']=None
        with patch('control_harness.predictors.resolve_predictor',side_effect=AssertionError('null mass must remain draft')):
            _,_,_,still_draft,_=self.result({}, {'template':'spring_damper','parameters':values})
        self.assertEqual(still_draft['required_parameters'],['mass'])
        self.assertFalse(still_draft['model_instantiated'])

    def test_generic_description_is_a_contract_draft_until_an_explicit_predictor_is_given(self):
        _,_,_,draft,_=self.result({'description':'为热管理过程生成MPC设计，不进行调优'})
        self.assertEqual(draft['template'],'generic');self.assertFalse(draft['model_instantiated'])
        self.assertEqual(draft['required_parameters'],['prediction_contract','interface'])
        _,_,_,numeric,_=self.result({'description':'一个显式离散过程'},
            {'prediction':{'method':'equations','model_spec':self.numeric_spec()},'solver_check':False})
        self.assertTrue(numeric['model_instantiated']);self.assertIsNone(numeric['template'])
        self.assertEqual(numeric['required_parameters'],[]);self.assertEqual(numeric['solver_check']['status'],'not_run')

    def test_complete_spring_values_make_one_real_solve_without_GA_or_closed_loop(self):
        with (patch.object(control,'minimize',wraps=control.minimize) as numerical,
             patch.object(control.GeneticOperators,'_initial_population',side_effect=AssertionError('GA forbidden')) as ga,
             patch.object(control,'closed_loop',side_effect=AssertionError('plant closed loop forbidden')) as loop,
             patch.object(control,'evaluate_controller',side_effect=AssertionError('tuning evaluator forbidden')) as evaluation):
            _,root,metadata,design,report=self.result({}, {'template':'spring_damper','parameters':self.parameters,
                'controller':{'horizon':4,'max_iterations':60,'solver_tolerance':1e-9,'input_rate_bounds':[[-.8,.8]]},
                'weights':{'q':2.,'r':.03,'s':.02},'solver_check':{'x':[.4,-.1],'reference':[0.],'disturbance':[],'previous':[0.]}})
        self.assertEqual(numerical.call_count,1);ga.assert_not_called();loop.assert_not_called();evaluation.assert_not_called()
        self.assertTrue(design['model_instantiated']);self.assertEqual(design['status'],'instantiated_unverified')
        self.assertEqual(design['solver_check']['status'],'passed');self.assertEqual(design['solver_check']['solves'],1)
        self.assertEqual(design['solver_check']['point_source'],'explicit_diagnostic_point')
        self.assertEqual(design['closed_loop']['ga_trials'],0);self.assertEqual(design['closed_loop']['plant_rollout_steps'],0)
        self.assertFalse(design['controller_parameters_tuned']);self.assertEqual(design['closed_loop_validation'],'not_run')
        self.assertGreater(design['solver_check']['diagnostics']['iterations'],0)
        self.assertTrue(np.isfinite(design['solver_check']['action']).all())
        self.assertEqual(design['prediction']['spec']['A'],[[0.,1.],[-1.5,-.4]])
        self.assertEqual(design['physical_interface']['sample_time'],.1)
        self.assertEqual(design['cost']['range_normalization'],{'output':[4.],'control':[8.]})
        self.assertEqual(design['cost']['weights'],{'q':2.,'r':.03,'s':.02})
        self.assertEqual(metadata['activation'],'none');self.assertNotIn('controller',self.ws.manifest['active'])

    def test_direct_nonlinear_prediction_is_NMPC_not_automatic_linearization(self):
        model=self.model(nonlinear=True);before=copy.deepcopy(self.ws.manifest['active'])
        _,root,metadata,design,_=self.result({'model':model},{'controller':{'horizon':3},
            'solver_check':{'x':[.3],'reference':[0.],'disturbance':[],'previous':[0.]}})
        self.assertEqual(design['prediction']['route'],'direct')
        self.assertTrue(design['prediction']['spec']['terms'])
        self.assertEqual(design['solver_check']['status'],'passed')
        self.assertEqual([row['id'] for row in metadata['parents']],[model])
        self.assertEqual(before,self.ws.manifest['active'])
        self.assertFalse(any(row['kind']=='identified_model' for row in self.ws.manifest['artifacts'].values()))

    def test_solver_disabled_does_not_invent_validation_or_replace_existing_controller(self):
        model_id=self.model();model=load_model(self.ws,model_id)
        _,diagnostics=MPCPolicy(model,{'horizon':2}).act([.1],[0.],[],[0.])
        # A real previously solvable policy artifact is only a lifecycle fixture;
        # its metadata explicitly makes no closed-loop certification claim.
        with self.ws.stage('controller',model.interface,[model_id],{}) as stage:
            atomic_json(stage.root/'controller.json',{'prediction_model':model_id,'evaluation_model':model_id,'parameters':{'horizon':2}})
            previous=stage.publish({'status':'verified','checks':['actual_single_point_solver_fixture'],'closed_loop':'not_run'},source={'diagnostics':diagnostics})
        _,_,metadata,design,_=self.result({'model':model_id},{'solver_check':False})
        self.assertTrue(design['model_instantiated']);self.assertEqual(design['solver_check']['reason'],'not_requested')
        self.assertEqual(self.ws.manifest['active']['controller'],previous);self.assertEqual(metadata['activation'],'none')

    def test_infeasible_single_point_is_a_failed_diagnostic_not_a_closed_loop_failure(self):
        model=self.model(bias=100.)
        _,root,metadata,design,_=self.result({'model':model},{'controller':{'horizon':2,'max_iterations':10}})
        self.assertEqual(design['solver_check']['status'],'failed')
        self.assertEqual(metadata['validation']['status'],'rejected')
        self.assertEqual(design['status'],'solver_check_failed');self.assertEqual(design['closed_loop']['status'],'not_run')
        self.assertTrue((root/'bundle/controller.py').is_file())

    def test_large_diagnostic_can_export_but_is_explicitly_not_run_before_Jacobian_allocation(self):
        model=self.model(nu=3)
        with patch.object(control,'minimize',side_effect=AssertionError('large Jacobian must not allocate')) as numerical:
            _,root,metadata,design,_=self.result({'model':model},{'controller':{'horizon':100}})
        numerical.assert_not_called();self.assertEqual(design['solver_check']['status'],'not_run')
        self.assertEqual(design['solver_check']['reason'],'single_point_diagnostic_resource_budget')
        self.assertEqual(metadata['validation']['status'],'verified')
        self.assertTrue((root/'bundle/bundle_manifest.json').is_file())

    def sampled(self):
        spec=self.numeric_spec();interface=ModelInterface.parse(spec['interface']);rng=np.random.default_rng(999)
        records={key:[] for key in ('x','u','d','next_x','y','next_y','trajectory_id','time')}
        for trajectory in range(6):
            x=np.array([-.5+.17*trajectory])
            for step in range(10):
                u=rng.uniform(-.8,.8,size=1);next_x=.7*x+.2*u+.03
                for key,value in (('x',x.copy()),('u',u),('d',[]),('next_x',next_x),('y',x.copy()),('next_y',next_x.copy()),('trajectory_id',trajectory),('time',.2*step)):
                    records[key].append(value)
                x=next_x
        with self.ws.stage('sampled_dataset',interface,[],{'fixture':'independent affine recorded arrays'}) as stage:
            np.savez_compressed(stage.root/'transitions.npz',**{key:np.asarray(value,dtype=np.int64 if key=='trajectory_id' else np.float64) for key,value in records.items()})
            atomic_json(stage.root/'split.json',{'train':[0,1,2,3],'validation':[4],'test':[5]})
            return stage.publish({'status':'verified'},source={'kind':'plant_simulator','description':'independent fixed recurrence unit fixture'},activate=False)

    def test_single_sampled_dataset_identifies_affine_predictor_without_reference_twin(self):
        dataset=self.sampled()
        _,_,metadata,design,_=self.result({'dataset':dataset},{'prediction':{'C':[[1.]],'acceptance':{'atol':1e-6,'rtol':1e-5}},'solver_check':False})
        self.assertEqual(design['prediction']['route'],'identify_affine')
        np.testing.assert_allclose(design['prediction']['spec']['A'],[[.7]],rtol=0,atol=1e-12)
        np.testing.assert_allclose(design['prediction']['spec']['B'],[[.2]],rtol=0,atol=1e-12)
        np.testing.assert_allclose(design['prediction']['spec']['bias'],[.03],rtol=0,atol=1e-12)
        self.assertTrue(design['prediction']['report']['acceptance']['passed'])
        self.assertEqual(design['closed_loop']['evaluation_model'],None)
        self.assertIn(dataset,[row['id'] for row in metadata['parents']])

    def test_local_linearization_remains_explicit_and_has_its_validity_bounds(self):
        model=self.model(nonlinear=True)
        _,_,_,design,_=self.result({'model':model},{'prediction':{'method':'linearize',
            'operating_point':{'x':[.1],'u':[0.],'d':[]},'validity_bounds':{'x':[[0.,.2]],'u':[[-.2,.2]],'d':[]}},'solver_check':False})
        self.assertEqual(design['prediction']['route'],'linearize')
        self.assertEqual(design['physical_interface']['states'][0]['lower'],0.)
        self.assertEqual(design['physical_interface']['states'][0]['upper'],.2)
        self.assertEqual(design['prediction']['report']['scope'],'local_jacobian_not_global_linear_model')
        self.assertIn('not a full-box error bound',design['prediction']['report']['validation_method'])
        self.assertEqual(design['closed_loop']['status'],'not_run')

    def test_invalid_partial_values_weights_dimensions_and_old_GA_fields_are_rejected(self):
        configs=[{'template':'spring_damper','parameters':{'mass':0.}}, {'template':'spring_damper','parameters':{'damping':-1}},
            {'template':'spring_damper','parameters':{'sample_time':'.1'}}, {'template':'spring_damper','parameters':{'state_bounds':[[0.,0.],[-1.,1.]]}},
            {'template':'spring_damper','parameters':{'dt':.1}}, {'controller':{'horizon':1.5}}, {'weights':{'q':False}},
            {'weights':{'Q':[[1.]]}}, {'weights':{'r':-1}}, {'budget':8}, {'solver_check':{'confirmed':True}}, {'template':[]}]
        for config in configs:
            with self.subTest(config=config),self.assertRaises(ContractError):self.result({'description':'spring'},config)
        model=self.model()
        with self.assertRaises(ContractError):self.result({'model':model,'description':'ambiguous'})
        with self.assertRaises(ContractError):self.result({'model':model,'evaluation_model':model})
        with self.assertRaises(ContractError):self.result({'model':model},{'controller':{'input_rate_bounds':[[-1.,1.],[-1.,1.]]}})


if __name__=='__main__':unittest.main()
