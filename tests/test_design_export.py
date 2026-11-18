"""Portable design drafts and actual MPCPolicy numerical reuse, without MATLAB."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import torch

from control_harness.contracts import ContractError
from control_harness.control import MPCPolicy
from control_harness.design_export import export_controller_design,SPRING_KEYS
from control_harness.examples import example
from control_harness.modeling import DeclarativeDynamics,tensor
from control_harness.predictors import resolve_predictor,spring_damper_spec

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(os.environ.get('CONTROL_TEST_OUTPUT',ROOT/'test-output/design-export-tests'))
EXTERNAL=ROOT.parent/'controller-design-portability-tests'


def design(prediction=None,**changes):
    result={'schema_version':1,'template':'spring_damper' if prediction is None else 'explicit_predictor',
        'description':'Synthetic independent design-export test','parameters':{},'required_parameters':list(SPRING_KEYS) if prediction is None else [],
        'model_instantiated':prediction is not None,'physical_interface':None if prediction is None else prediction.interface.to_dict(),
        'prediction':None if prediction is None else prediction.to_dict(),'controller_config':{'horizon':3,'max_iterations':80,'solver_tolerance':1e-10},
        'weights':{'q':1.,'r':.02,'s':.01},'equations':{'spring':"m*position_ddot+c*position_dot+k*position=force"},
        'cost':'sum q||output-reference||^2/range_y^2+r||u||^2/range_u^2+s||du||^2/range_u^2',
        'constraints':'declared state/input/output and optional per-sample input increment',
        'solver_check':{'status':'not_run'},'closed_loop':{'status':'not_run','ga_trials':0}}
    result.update(changes);return result


def load(directory):
    name='test_portable_design_'+uuid.uuid4().hex
    spec=importlib.util.spec_from_file_location(name,Path(directory)/'controller.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


class DesignExportTests(unittest.TestCase):
    def setUp(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.root=OUTPUT/('d-'+uuid.uuid4().hex[:8]);self.root.mkdir()
        self.addCleanup(self.cleanup)
    def cleanup(self):
        assert self.root.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(self.root,ignore_errors=True)
    def predictor(self,case='nonlinear'):
        return resolve_predictor(None,{}, {'method':'equations','model_spec':example(case)['reference' if case=='nonlinear' else 'nominal']})
    def bundle(self,definition,prediction=None):
        target=self.root/('b-'+uuid.uuid4().hex[:6]);export_controller_design(target,definition,prediction);return target

    def test_missing_spring_parameters_remain_callable_draft_without_example_defaults(self):
        directory=self.bundle(design());module=load(directory);draft=module.load_design(directory)
        self.assertFalse(draft.model_instantiated)
        self.assertEqual(draft.definition['parameters'],{})
        with self.assertRaisesRegex(module.NotInstantiated,'mass.*damping.*stiffness.*sample_time'):
            draft.instantiate()
        with self.assertRaises(module.NotInstantiated):draft.instantiate({'mass':None})
        schema=json.loads((directory/'parameter_schema.json').read_text(encoding='utf-8'))
        self.assertFalse(schema['physical_example_defaults'])
        self.assertEqual(set(schema['required_when_instantiating_spring']),set(SPRING_KEYS))
        self.assertEqual(draft.definition['closed_loop']['status'],'not_run')

    def test_parameterized_spring_instantiates_user_values_and_matches_original_mpc(self):
        params={'mass':1.7,'damping':.3,'stiffness':4.,'sample_time':.1,
            'state_bounds':[[-2.,2.],[-3.,3.]],'input_bounds':[[-5.,5.]],'output_bounds':[[-2.,2.]]}
        directory=self.bundle(design());portable=load(directory).load_design(directory).instantiate(params)
        original=MPCPolicy(DeclarativeDynamics(spring_damper_spec(params)),{'horizon':3,'max_iterations':80,'solver_tolerance':1e-10,'q':1.,'r':.02,'s':.01})
        np.testing.assert_allclose(portable.model.A,original.model.A,atol=1e-14,rtol=0)
        np.testing.assert_allclose(portable.model.B,original.model.B,atol=1e-14,rtol=0)
        self.assertEqual(portable.interface.sample_time,.1)
        self.assertEqual([v.name for v in portable.interface.states],['position','velocity'])
        u,a=portable.act([.2,-.1],[.05],[]);v,b=original.act([.2,-.1],[.05],[])
        np.testing.assert_allclose(u,v,atol=1e-8,rtol=1e-7)
        self.assertAlmostEqual(a['predicted_objective'],b['predicted_objective'],places=12)
        other=load(directory).load_design(directory).instantiate({**params,'mass':2.5,'stiffness':0.})
        self.assertGreater(float(torch.max(torch.abs(other.model.A-portable.model.A))),1e-3)

    def test_actual_mpc_source_and_solver_are_reused_verbatim_in_ready_package(self):
        predictor=self.predictor();definition=design(predictor);directory=self.bundle(definition,predictor)
        source=(directory/'mpc_runtime.py').read_text(encoding='utf-8')
        self.assertIn(inspect.getsource(MPCPolicy),source)
        manifest=json.loads((directory/'bundle_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['numerical_reuse']['MPCPolicy']['source_sha256'],hashlib.sha256(inspect.getsource(MPCPolicy).encode()).hexdigest())
        portable=load(directory).load_design(directory).instantiate()
        original=MPCPolicy(predictor.runtime,{**definition['controller_config'],**definition['weights']})
        for x in ([.1,-.05],[-.12,.03]):
            u,a=portable.act(x,[.03],[.01],[0.]);v,b=original.act(x,[.03],[.01],[0.])
            np.testing.assert_allclose(u,v,rtol=0,atol=1e-12)
            self.assertEqual(a['predicted_objective'],b['predicted_objective'])
            np.testing.assert_array_equal(portable.warm,original.warm)
        self.assertEqual(json.loads((directory/'matlab_model.json').read_text())['kind'],'unsupported_nonlinear_predictor')

    def test_affine_mimo_cost_constraints_and_rate_semantics_match_original(self):
        predictor=self.predictor('linear');definition=design(predictor)
        definition['controller_config']['input_rate_bounds']=[[-.2,.2],[-.1,.1]]
        directory=self.bundle(definition,predictor);portable=load(directory).load_design(directory).instantiate()
        original=MPCPolicy(predictor.runtime,{**definition['controller_config'],**definition['weights']})
        sequence=tensor([[.01,-.02],[.05,.02],[.0,.03]])
        args=[tensor([.1,-.05,.08,.02]),tensor([0.,0.]),tensor([.01]),tensor([0.,0.])]
        a,ac=portable.rollout(sequence,*args);b,bc=original.rollout(sequence,*args)
        self.assertEqual(a.item(),b.item());torch.testing.assert_close(ac,bc,atol=0,rtol=0)
        u,diag=portable.act([.1,-.05,.08,.02],[0.,0.],[.01],[0.,0.])
        self.assertLessEqual(abs(u[0]),.2+1e-8);self.assertLessEqual(abs(u[1]),.1+1e-8)
        matlab=json.loads((directory/'matlab_model.json').read_text())
        self.assertEqual(matlab['kind'],'affine');near=np.array(matlab['A'])
        np.testing.assert_array_equal(near,predictor.runtime.A.numpy())
        self.assertLessEqual(diag['constraint_violation'],1e-6)

    def test_repo_independent_new_process_solves_ready_bundle_with_project_imports_blocked(self):
        predictor=self.predictor('linear');definition=design(predictor);directory=self.bundle(definition,predictor)
        EXTERNAL.mkdir(parents=True,exist_ok=True);detached=EXTERNAL/('d-'+uuid.uuid4().hex[:8])
        assert not detached.resolve().is_relative_to(ROOT.resolve())
        def cleanup_external():
            assert detached.resolve().is_relative_to(EXTERNAL.resolve())
            shutil.rmtree(detached,ignore_errors=True)
        self.addCleanup(cleanup_external);shutil.copytree(directory,detached)
        # Remove the original bundle so the copy cannot accidentally read it.
        assert directory.resolve().is_relative_to(self.root.resolve());shutil.rmtree(directory)
        case={'x':[.1,-.05,.08,.02],'reference':[0.,0.],'disturbance':[.01],'previous':[0.,0.]}
        (self.root/'case.json').write_text(json.dumps(case),encoding='utf-8')
        code='''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('Project imports blocked')
sys.meta_path.insert(0,Block());script=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
'''
        env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}
        run=subprocess.run([sys.executable,'-I','-B','-c',code,str(detached/'controller.py'),'--input',str(self.root/'case.json'),'--output',str(self.root/'result.json')],cwd=detached,env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        actual=json.loads((self.root/'result.json').read_text())
        expected,diagnostics=MPCPolicy(predictor.runtime,{**definition['controller_config'],**definition['weights']}).act(case['x'],case['reference'],case['disturbance'],case['previous'])
        np.testing.assert_allclose(actual['u'],expected,rtol=1e-10,atol=1e-10)
        self.assertAlmostEqual(actual['diagnostics']['predicted_objective'],diagnostics['predicted_objective'],places=12)
        self.assertEqual(actual['closed_loop'],'not_run');self.assertEqual(actual['ga_trials'],0)

    def test_corrupted_runtime_or_predictor_asset_hash_is_rejected_before_loading(self):
        predictor=self.predictor('linear');directory=self.bundle(design(predictor),predictor)
        (directory/'mpc_runtime.py').write_text('raise RuntimeError("must never execute")\n')
        with self.assertRaisesRegex(ValueError,'member changed'):load(directory).load_design(directory)
        other=self.bundle(design(predictor),predictor)
        (other/'prediction/model.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'member changed'):load(other).load_design(other)

    def test_generic_draft_never_invents_predictor_and_invalid_physics_is_rejected(self):
        directory=self.bundle(design(template='generic_description',required_parameters=['explicit predictor contract']))
        module=load(directory)
        with self.assertRaises(module.NotInstantiated):module.load_design(directory).instantiate({'mass':1.})
        spring=self.bundle(design());draft=load(spring).load_design(spring)
        p={'mass':1.,'damping':.2,'stiffness':2.,'sample_time':.1,'state_bounds':[[-1.,1.],[-1.,1.]],'input_bounds':[[-2.,2.]],'output_bounds':[[-1.,1.]]}
        for bad in ({'mass':0},{'sample_time':True},{'stiffness':-1},{'state_bounds':[[1.,-1.],[-1.,1.]]}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):draft.instantiate({**p,**bad})

    def test_matlab_templates_are_parameterized_and_report_unverified_toolbox_license(self):
        directory=self.bundle(design());manifest=json.loads((directory/'bundle_manifest.json').read_text())
        self.assertEqual(manifest['matlab']['license'],'not_checked')
        self.assertEqual(manifest['matlab']['numerical_validation'],'not_run')
        source=(directory/'mpc_control.m').read_text()
        self.assertIn('quadprog(Hessian,linear,A,b',source);self.assertIn('exitflag>0',source);self.assertIn('violation<=1e-6',source)
        self.assertIn('per-sample action increments',source)
        self.assertIn('expm(generator*dt)',(directory/'prediction_model.m').read_text())
        self.assertTrue(all((directory/name).is_file() for name in ['controller.py','example.py','parameter_schema.json','predict_step.m','prediction_model.m','mpc_cost.m','selfcheck_controller.m']))
        self.assertIn("license('test','Optimization_Toolbox')",(directory/'selfcheck_controller.m').read_text())
        self.assertIn("result.status='not_instantiated'",(directory/'selfcheck_controller.m').read_text())

    def test_single_point_resource_guard_runs_before_dense_solver(self):
        spec=copy.deepcopy(example('linear')['nominal'])
        spec['interface']['controls'].append({'name':'third_control','unit':'1','lower':-1.5,'upper':1.5})
        for row in spec['B']:row.append(0.)
        predictor=resolve_predictor(None,{}, {'method':'equations','model_spec':spec})
        directory=self.bundle(design(predictor),predictor)
        portable=load(directory).load_design(directory).instantiate(controller_config={'horizon':100})
        self.assertEqual(portable.horizon,100)
        self.assertEqual(portable.interface.nu,3)
        with self.assertRaisesRegex(ValueError,'resource guard'):
            portable.act([0.]*4,[0.]*2,[0.],[0.]*3)

    def test_matlab_large_output_weights_stay_sparse_with_identical_quadratic_cost(self):
        from scipy import sparse
        # A supported V*M domain must not construct a 13.4 GB dense output weight.
        horizon,ny,nx,nu=20,2048,2,1
        spec=copy.deepcopy(example('nonlinear')['nominal'])
        original_output=spec['interface']['outputs'][0]
        spec['interface']['outputs']=[{**original_output,'name':'output_'+str(i)} for i in range(ny)]
        spec['C']=[[1.,.001*(i%7)] for i in range(ny)]
        predictor=resolve_predictor(None,{}, {'method':'equations','model_spec':spec})
        definition=design(predictor);definition['controller_config']['horizon']=horizon
        directory=self.bundle(definition,predictor)
        self.assertEqual(json.loads((directory/'matlab_model.json').read_text())['ny'],ny)
        variables=horizon*nu;margins=horizon*(2*nx+2*ny)
        self.assertLessEqual(variables*margins,5_000_000)
        self.assertGreater((horizon*ny)**2*8,10_000_000_000)
        source=(directory/'mpc_control.m').read_text()
        self.assertIn('Q=kron(speye(H),spdiags(weights.q./yrange.^2,0,ny,ny))',source)
        self.assertIn('R=kron(speye(H),spdiags(weights.r./urange.^2,0,n,n))',source)
        self.assertIn('S=kron(speye(H),spdiags(weights.s./urange.^2,0,n,n))',source)
        self.assertNotIn('kron(eye(H),diag(',source)
        # Independent SciPy sparse algebra verifies the unchanged diagonal cost
        # and condensed Hessian without allocating the forbidden dense diagonal.
        scale=np.linspace(1.,3.,ny);weight=definition['weights']['q']/scale**2
        q=sparse.kron(sparse.eye(horizon),sparse.diags(weight),format='csr')
        self.assertEqual(q.nnz,horizon*ny)
        self.assertLess(q.data.nbytes+q.indices.nbytes+q.indptr.nbytes,1_000_000)
        rng=np.random.default_rng(20261117);yp=rng.normal(size=(horizon*ny,variables));error=rng.normal(size=horizon*ny)
        repeated=np.tile(weight,horizon)
        np.testing.assert_allclose(yp.T@(q@yp),yp.T@(repeated[:,None]*yp),rtol=1e-13,atol=1e-10)
        self.assertAlmostEqual(float(error@(q@error)),float(np.sum(error**2*repeated)),places=10)

    def test_real_corrected_predictor_bundle_preserves_residual_and_mpc_action(self):
        from control_harness.modeling import CorrectedDynamics,ResidualNetwork
        from control_harness.predictors import PredictionResult
        nominal=DeclarativeDynamics(example('linear')['nominal']);i=nominal.interface
        residual=ResidualNetwork(i,{'hidden_dims':[6]}, {'input_mean':[0.]*(i.nx+i.nu+i.nd),
            'input_scale':[1.]*(i.nx+i.nu+i.nd),'residual_scale':[1.]*i.nx})
        with torch.no_grad():
            for p in residual.parameters():p.zero_()
            residual.network.net[-1].bias.fill_(.01)
        model=CorrectedDynamics(nominal,residual)
        prediction=PredictionResult(model,i,{'representation':'nominal_plus_learned_residual'},None,'direct',{'source':'explicit test_fixture intervention'},True)
        definition=design(prediction);directory=self.bundle(definition,prediction)
        portable=load(directory).load_design(directory).instantiate()
        original=MPCPolicy(model,{**definition['controller_config'],**definition['weights']})
        x=[.1,-.04,.03,.02];u=[.02,-.01];d=[.01]
        torch.testing.assert_close(portable.model(tensor(x),tensor(u),tensor(d)),model(tensor(x),tensor(u),tensor(d)),atol=0,rtol=0)
        a,ad=portable.act(x,[0.,0.],d);b,bd=original.act(x,[0.,0.],d)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        self.assertEqual(ad['predicted_objective'],bd['predicted_objective'])

    def test_absolute_storage_locations_are_omitted_and_reexport_does_not_overwrite(self):
        definition=design();definition['workspace']=str(self.root);definition['solver_check']['source_root']=str(self.root/'private')
        directory=self.bundle(definition)
        self.assertNotIn(str(self.root),(directory/'design.json').read_text())
        with self.assertRaises(ContractError):export_controller_design(directory,definition)
        with self.assertRaises(ContractError):self.bundle({**definition,'model_instantiated':'false'})


if __name__=='__main__':unittest.main()
