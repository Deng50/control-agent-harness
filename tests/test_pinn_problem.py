"""Static SI thermal-ODE material preparation with real frozen F04 files."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from control_harness.artifacts import Workspace,read_json
from control_harness.contracts import ContractError
from control_harness.material_registration import register_material
from control_harness.pinn_contract import CAPABILITY,UNITS,pinn_signals,validate_pinn_config,validate_pinn_definition
from control_harness.pinn_problem import prepare_pinn_problem,read_pinn_problem,validate_pinn_arrays,_load_points
from control_harness.runtime import OperationContext


PROJECT=Path(__file__).resolve().parents[1]
FROZEN=PROJECT/'test-output/v2-development/20261117T002006/fixtures'
OUTPUT=PROJECT/'test-output/pinn-problem-tests';OUTPUT.mkdir(parents=True,exist_ok=True)


def configuration():
    f=read_json(FROZEN/'F04_conditional_ode.json')
    return {'schema_version':1,'equation':{'template':'thermal_first_order','forcing':'constant_per_condition',
        'parameters':{key:f['parameters'][key] for key in ['C_J_per_K','G_W_per_K','Ta_K']}},
        'domain':{'time_s':[0.,40.],'initial_temperature_K':[291.15,295.15],'power_W':[0.,200.]},
        'normalization':{'time_origin_s':0.,'time_scale_s':40.,'state_offset_K':293.15,'state_scale_K':4.,'power_scale_W':200.,'physics_scale_W':200.},
        'units':dict(UNITS),'conditions':[{'id':c['id'],'split':c['split'],'initial_temperature_K':c['T0_K'],'power_W':c['power_W']} for c in f['conditions']],
        'reference_file':'source/F04_analytic_trajectories.csv','reference_columns':{
            key:{'index':index,'header':header} for key,index,header in [('condition_id',0,'trajectory_id'),('split',1,'split'),('time_s',2,'time_s'),('initial_temperature_K',3,'T0_K'),('power_W',4,'power_W'),('temperature_K',5,'temperature_K')]},
        'collocation_file':'source/F04_training_physics_collocation.csv','collocation_columns':{
            key:{'index':index,'header':header} for key,index,header in [('condition_id',0,'trajectory_id'),('time_s',1,'time_s'),('initial_temperature_K',2,'T0_K'),('power_W',3,'power_W')]},
        'source':{'kind':'synthetic_analytic','description':'Immutable F04 synthetic test_fixture analytic thermal references; no experiment, neural training or Plant execution'}}


class PinnProblemTests(unittest.TestCase):
    confirmation_source='test_fixture'

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.inputs=self.root/'inputs';self.inputs.mkdir();self.ws=Workspace(self.root/'workspace')
        for name in ['F04_conditional_ode.json','F04_analytic_trajectories.csv','F04_training_physics_collocation.csv']:shutil.copyfile(FROZEN/name,self.inputs/name)
        self.config=configuration();self.context=OperationContext()

    def material(self):
        return register_material(self.ws,str(self.inputs/'F04_conditional_ode.json'),{'dependency_root':str(self.inputs),
            'dependencies':['F04_analytic_trajectories.csv','F04_training_physics_collocation.csv']},self.context)

    def prepare(self,config=None):
        material=self.material();artifact=prepare_pinn_problem(self.ws,material,self.config if config is None else config,self.context)
        return artifact,read_pinn_problem(self.ws,artifact),material

    def edit_csv(self,name,change):
        path=self.inputs/name
        with path.open(encoding='utf-8',newline='') as stream:rows=list(csv.reader(stream))
        change(rows)
        with path.open('w',encoding='utf-8',newline='') as stream:csv.writer(stream).writerows(rows)

    def assert_no_problem(self):self.assertFalse(any(r['kind']=='conditional_ode_problem' for r in self.ws.manifest['artifacts'].values()))

    def test_actual_frozen_f04_labels_collocation_units_condition_split_and_analytic_values(self):
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.inputs.iterdir()};artifact,result,material=self.prepare();d,a,m=result['definition'],result['arrays'],result['metadata']
        self.assertEqual(m['kind'],'conditional_ode_problem');self.assertEqual(m['interface']['interface_kind'],'material');self.assertEqual(m['interface']['format'],'bundle')
        self.assertFalse(any(key in d for key in ['nx','nu','ny','states','sample_time','transition']));self.assertEqual(d['capability'],CAPABILITY)
        self.assertEqual(a['time_s'].shape,(2412,1));self.assertEqual(a['temperature_K'].shape,(2412,1));self.assertEqual(np.bincount(a['split_id']).tolist(),[804,804,804])
        self.assertEqual(a['collocation_time_s'].shape,(1028,1));self.assertEqual(set(a['collocation_condition_index'].tolist()),{0,1,2,3})
        self.assertEqual(a['initial_value_K'].shape,(12,1));np.testing.assert_array_equal(a['initial_split_id'],[0]*4+[1]*4+[2]*4)
        f=read_json(FROZEN/'F04_conditional_ode.json');parameters=f['parameters'];ambient=parameters['Ta_K'];C=parameters['C_J_per_K'];G=parameters['G_W_per_K']
        expected=np.array([ambient+p/G+(t0-ambient-p/G)*math.exp(-G*t/C) for t,t0,p in zip(a['time_s'][:,0],a['initial_temperature_K'][:,0],a['power_W'][:,0])])
        np.testing.assert_allclose(a['temperature_K'][:,0],expected,atol=1e-12,rtol=0)
        train=a['temperature_K'][a['split_id']==0,0];self.assertAlmostEqual(d['normalization']['data_nrmse_scale_K'],float(train.std(ddof=0)),places=14)
        self.assertAlmostEqual(d['normalization']['data_nrmse_scale_K'],f['data_nrmse_scale_K'],places=14);self.assertEqual(d['normalization']['fit_samples'],804)
        self.assertEqual(d['normalization']['data_population_std_K'],float(train.std(ddof=0)))
        self.assertEqual(d['normalization']['data_nrmse_scale_source'],'training_population_std');self.assertIsNone(d['normalization']['data_nrmse_fallback_K'])
        self.assertIs(d['normalization']['exact_constant_training_targets'],False)
        self.assertNotIn('dT_dt_K_per_s',a);self.assertEqual(m['source']['pinn_signals'],d['source']['pinn_signals'])
        self.assertIn('(state_scale_K/time_scale_s)*dtheta/ds',m['source']['pinn_signals']['time_derivative']['chain_rule'])
        self.assertEqual(m['source']['pinn_signals']['equation_parameter_units'],{'C_J_per_K':'J/K','G_W_per_K':'W/K','Ta_K':'K'})
        self.assertEqual(m['validation']['accuracy'],'not_assessed');self.assertNotIn('conditional_ode_problem',self.ws.manifest['active'])
        self.assertEqual({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.inputs.iterdir()},before)
        self.assertEqual({p['id'] for p in m['parents']},{material});self.assertNotIn('gates',d)

    def test_exact_equilibrium_training_targets_use_predeclared_engineering_scale_without_heldout_leakage(self):
        # A legitimate low-coverage production problem: one equilibrium train
        # condition, with distinct dynamic validation and test conditions.
        self.config['conditions']=[{'id':name,'split':split,'initial_temperature_K':t0,'power_W':power}
            for name,split,t0,power in [('equilibrium','train',293.15,0.),('cooling','validation',291.15,0.),('heating','test',295.15,200.)]]
        rows=[['trajectory_id','split','time_s','T0_K','power_W','temperature_K']]
        for condition in self.config['conditions']:
            t0=condition['initial_temperature_K'];power=condition['power_W']
            for index in range(201):
                time=index*.2;temperature=293.15+power/50.+(t0-293.15-power/50.)*math.exp(-time/20.)
                rows.append([condition['id'],condition['split'],repr(time),repr(t0),repr(power),repr(temperature)])
        with (self.inputs/'F04_analytic_trajectories.csv').open('w',encoding='utf-8',newline='') as stream:csv.writer(stream).writerows(rows)
        with (self.inputs/'F04_training_physics_collocation.csv').open('w',encoding='utf-8',newline='') as stream:
            csv.writer(stream).writerows([['trajectory_id','time_s','T0_K','power_W'],*[[ 'equilibrium',time,293.15,0.] for time in [0.,10.,20.,30.,40.]]])
        _,result,_=self.prepare();definition=result['definition'];arrays=result['arrays'];norm=definition['normalization']
        train=arrays['temperature_K'][arrays['split_id']==0,0]
        np.testing.assert_array_equal(train,np.full(201,293.15))
        # Naive std on this exact decimal constant can be positive by roundoff;
        # the declared result must instead be mathematically zero.
        self.assertEqual(norm['data_population_std_K'],0.);self.assertIs(norm['exact_constant_training_targets'],True)
        self.assertEqual(norm['data_nrmse_scale_K'],4.);self.assertEqual(norm['data_nrmse_fallback_K'],4.)
        self.assertEqual(norm['data_nrmse_scale_source'],'engineering_state_scale_for_exact_constant_training_targets')
        self.assertEqual(norm['fit_samples'],201);self.assertEqual(norm['fit_split'],'train')
        self.assertNotEqual(norm['data_nrmse_scale_K'],float(arrays['temperature_K'].std()))
        self.assertEqual(result['metadata']['source']['pinn_signals']['normalization'],norm)
        self.assertEqual(validate_pinn_definition(definition),definition)
        report=read_json(result['root']/'report.json');self.assertEqual(report['statistics']['conditions_by_split'],{'train':1,'validation':1,'test':1})
        self.assertEqual(report['statistics']['data_population_std_K'],0.);self.assertEqual(report['training'],'not_run')
        self.assertEqual(result['metadata']['validation']['accuracy'],'not_assessed')
        for edit in [lambda n:n.update(data_population_std_K=1e-14),lambda n:n.update(data_nrmse_fallback_K=2.),
                lambda n:n.update(data_nrmse_scale_source='training_population_std'),lambda n:n.update(exact_constant_training_targets=1)]:
            bad=copy.deepcopy(definition);edit(bad['normalization']);bad['source']['pinn_signals']=pinn_signals(bad)
            with self.subTest(normalization=bad['normalization']),self.assertRaises(ContractError):validate_pinn_definition(bad)
        # A self-consistent but false nonconstant declaration cannot pass the
        # reader's independent inspection of the actual training target array.
        bad=copy.deepcopy(definition);bad['normalization'].update(data_population_std_K=1e-14,data_nrmse_scale_K=1e-14,
            data_nrmse_scale_source='training_population_std',data_nrmse_fallback_K=None,exact_constant_training_targets=False)
        bad['source']['pinn_signals']=pinn_signals(bad);validate_pinn_definition(bad)
        with self.assertRaisesRegex(ContractError,'training-only'):validate_pinn_arrays(bad,arrays)
        legacy=copy.deepcopy(definition)
        for name in ['data_population_std_K','data_nrmse_scale_source','data_nrmse_fallback_K','exact_constant_training_targets']:legacy['normalization'].pop(name)
        legacy['normalization']['data_nrmse_scale_K']=1e-14;legacy['source']['pinn_signals']=pinn_signals(legacy)
        validate_pinn_definition(legacy)
        with self.assertRaisesRegex(ContractError,'training-only'):validate_pinn_arrays(legacy,arrays)

    def test_pure_contract_is_portable_and_does_not_read_files_or_invent_plant_capabilities(self):
        _,result,_=self.prepare();definition=result['definition']
        source=(PROJECT/'src/control_harness/pinn_contract.py').read_text().replace('from .contracts import ContractError','ContractError=ValueError')
        namespace={};exec(compile(source,'portable_pinn_contract.py','exec'),namespace)
        with patch('builtins.open',side_effect=AssertionError('Pure validation cannot read source files')):
            self.assertEqual(namespace['validate_pinn_definition'](definition),definition)
        invalid=copy.deepcopy(definition);invalid['capability']['one_step_transition']=True
        with self.assertRaises(ValueError):namespace['validate_pinn_definition'](invalid)
        invalid=copy.deepcopy(definition);invalid['source']['pinn_signals']['units']['temperature_K']='degC'
        with self.assertRaises(ContractError):validate_pinn_definition(invalid)

    def test_missing_equation_wrong_units_and_normalization_are_rejected_before_publication(self):
        variants=[]
        raw=copy.deepcopy(self.config);raw.pop('equation');variants.append(raw)
        for edit in [lambda r:r['units'].update(temperature_K='degC'),lambda r:r['normalization'].update(state_offset_K=273.15),
            lambda r:r['normalization'].update(time_origin_s=1.),lambda r:r['normalization'].update(physics_scale_W=0.),
            lambda r:r['equation']['parameters'].update(C_J_per_K=True),lambda r:r['equation'].update(forcing='arbitrary_timeseries')]:
            raw=copy.deepcopy(self.config);edit(raw);variants.append(raw)
        material=self.material()
        for raw in variants:
            with self.subTest(raw=raw),self.assertRaises(ContractError):prepare_pinn_problem(self.ws,material,raw,self.context)
        self.assert_no_problem()

    def test_renaming_same_condition_or_moving_it_to_another_split_cannot_create_independence(self):
        raw=copy.deepcopy(self.config);raw['conditions'][4]['initial_temperature_K']=raw['conditions'][0]['initial_temperature_K'];raw['conditions'][4]['power_W']=raw['conditions'][0]['power_W']
        with self.assertRaisesRegex(ContractError,'same physical'):validate_pinn_config(raw)
        raw=copy.deepcopy(self.config);raw['conditions'][0]['initial_temperature_K']=290.
        with self.assertRaisesRegex(ContractError,'outside'):validate_pinn_config(raw)
        raw=copy.deepcopy(self.config)
        for condition in raw['conditions']:condition['split']='train'
        with self.assertRaisesRegex(ContractError,'partitions'):validate_pinn_config(raw)

    def test_duplicate_headers_are_resolved_by_declared_positions_not_names(self):
        self.edit_csv('F04_analytic_trajectories.csv',lambda rows:rows[0].__setitem__(3,'temperature_K'))
        self.config['reference_columns']['initial_temperature_K']['header']='temperature_K'
        _,result,_=self.prepare();a=result['arrays']
        self.assertGreater(float(np.max(np.abs(a['temperature_K']-a['initial_temperature_K']))),1.)
        raw=copy.deepcopy(self.config);raw['reference_columns']['temperature_K']['index']=3
        with self.assertRaisesRegex(ContractError,'positions'):validate_pinn_config(raw)

    def test_reference_split_condition_time_order_and_duplicate_points_conflicts_are_rejected(self):
        changes=[lambda rows:rows[1].__setitem__(1,'test'),lambda rows:rows[1].__setitem__(3,'294.15'),
            lambda rows:rows[1].__setitem__(4,'175'),lambda rows:rows[1].__setitem__(0,'unlisted_condition'),
            lambda rows:rows.insert(2,list(rows[1])),lambda rows:rows.__setitem__(slice(2,4),[rows[3],rows[2]])]
        original=(FROZEN/'F04_analytic_trajectories.csv').read_bytes()
        for index,change in enumerate(changes):
            (self.inputs/'F04_analytic_trajectories.csv').write_bytes(original);self.edit_csv('F04_analytic_trajectories.csv',change)
            with self.subTest(index=index),self.assertRaises(ContractError):prepare_pinn_problem(self.ws,self.material(),self.config,self.context)
        self.assert_no_problem()

    def test_nonfinite_missing_and_outside_domain_reference_values_are_rejected(self):
        original=(FROZEN/'F04_analytic_trajectories.csv').read_bytes()
        for column,value in [(2,'nan'),(5,'inf'),(5,''),(2,'40.1'),(5,'-1')]:
            (self.inputs/'F04_analytic_trajectories.csv').write_bytes(original)
            self.edit_csv('F04_analytic_trajectories.csv',lambda rows:rows[1].__setitem__(column,value))
            with self.subTest(column=column,value=value),self.assertRaises(ContractError):prepare_pinn_problem(self.ws,self.material(),self.config,self.context)
        self.assert_no_problem()

    def test_collocation_is_training_conditions_only_and_forcing_is_constant(self):
        original=(FROZEN/'F04_training_physics_collocation.csv').read_bytes()
        def heldout(rows):rows[1]=['validation_initial0_drive0','1.0','291.65','25.0']
        for change in [heldout,lambda rows:rows[1].__setitem__(3,'100'),lambda rows:rows[1].__setitem__(1,'-0.1'),lambda rows:rows.insert(2,list(rows[1]))]:
            (self.inputs/'F04_training_physics_collocation.csv').write_bytes(original);self.edit_csv('F04_training_physics_collocation.csv',change)
            with self.assertRaises(ContractError):prepare_pinn_problem(self.ws,self.material(),self.config,self.context)
        self.assert_no_problem()

    def test_only_bound_snapshot_members_are_read_and_later_parent_tamper_invalidates_problem(self):
        material=self.material();(self.inputs/'F04_analytic_trajectories.csv').write_text('arbitrary source text is data, not a Python instruction',encoding='utf-8')
        artifact=prepare_pinn_problem(self.ws,material,self.config,self.context);result=read_pinn_problem(self.ws,artifact);self.assertEqual(len(result['arrays']['time_s']),2412)
        for path in ['../unbound.csv','C:/outside.csv','source/missing.csv']:
            raw=copy.deepcopy(self.config);raw['reference_file']=path
            with self.assertRaises(ContractError):prepare_pinn_problem(self.ws,material,raw,self.context)
        source,_=self.ws.resolve(material);(source/self.config['reference_file']).write_bytes(b'changed immutable CSV')
        with self.assertRaises(ContractError):read_pinn_problem(self.ws,artifact)

    def test_reader_rejects_point_tamper_and_independent_semantic_split_or_scale_conflicts(self):
        artifact,result,_=self.prepare();a=result['arrays'];d=result['definition']
        bad={key:value.copy() for key,value in a.items()};bad['split_id'][0]=2
        with self.assertRaises(ContractError):validate_pinn_arrays(d,bad)
        bad={key:value.copy() for key,value in a.items()};bad['collocation_condition_index'][0]=4
        with self.assertRaises(ContractError):validate_pinn_arrays(d,bad)
        wrong=copy.deepcopy(d);wrong['normalization']['data_nrmse_scale_K']=float(a['temperature_K'].std())
        self.assertNotAlmostEqual(wrong['normalization']['data_nrmse_scale_K'],d['normalization']['data_nrmse_scale_K'],places=6)
        with self.assertRaisesRegex(ContractError,'training-only'):validate_pinn_arrays(wrong,a)
        path=result['root']/'points.npz';path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaises(ContractError):read_pinn_problem(self.ws,artifact)

    def test_malicious_npy_header_is_rejected_before_numpy_allocation_or_pickle(self):
        _,result,_=self.prepare();path=result['root']/'points.npz'
        with zipfile.ZipFile(path) as archive:members={e.filename:archive.read(e) for e in archive.infolist()}
        malformed=io.BytesIO();np.lib.format.write_array_header_1_0(malformed,{'descr':np.dtype('float64').str,'fortran_order':False,'shape':(10**12,1)})
        members['time_s.npy']=malformed.getvalue();target=self.root/'huge-declared-points.npz'
        with zipfile.ZipFile(target,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        with patch('numpy.load',side_effect=AssertionError('Do not deserialize an oversized declared shape')),self.assertRaises(ContractError):_load_points(target,result['definition'],digest)


if __name__=='__main__':unittest.main()
