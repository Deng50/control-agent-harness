"""Real stored arrays and complete ZIP bytes; synthetic test fixtures, no training."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from control_harness.artifacts import Workspace,atomic_json
from control_harness.contracts import ContractError,MaterialInterface,DataInterface
from control_harness.reporting import read_report_arrays,prepare_bundle_archive
from control_harness.supervised_export import build_supervised_network,export_supervised

OUTPUT=Path(__file__).resolve().parents[1]/'test-output/reporting-tests';OUTPUT.mkdir(parents=True,exist_ok=True)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.ws=Workspace(self.root/'workspace')

    def stored(self,arrays,kind='supervised_model',name='test_predictions.npz'):
        with self.ws.stage(kind,MaterialInterface('bundle','Synthetic report transport fixture'),[],{'synthetic':True}) as stage:
            np.savez_compressed(stage.root/name,**arrays)
            atomic_json(stage.root/'report.json',{'scope':'Synthetic report serialization only; not a trained-model acceptance claim'})
            return stage.publish({'status':'verified','scope':'array_fixture_only'},activate=False)

    def test_actual_group_filter_preserves_source_rows_time_and_physical_values(self):
        arrays={'trajectory_id':np.repeat([10,20,30],4),'time':np.tile([0.,.2,.4,.6],3),
            'prediction':np.arange(24,dtype=float).reshape(12,2)+100.,'truth':np.arange(24,dtype=float).reshape(12,2)+99.}
        artifact=self.stored(arrays)
        result=read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['time','trajectory_id','prediction','truth'],'max_rows':3,'filter':{'key':'trajectory_id','value':20}})['file_preview']
        row=result['array_rows'];self.assertEqual(row['source_rows'],[4,5,7]);self.assertEqual(row['matched_rows'],4);self.assertTrue(row['truncated'])
        self.assertEqual(row['values']['time'],[0.,.2,.6]);self.assertEqual(row['values']['prediction'],[[108.,109.],[110.,111.],[114.,115.]])
        self.assertFalse(row['metrics_recomputed']);self.assertEqual(row['groups']['trajectory_id']['values'],[10,20,30]);self.assertEqual(result['sha256'],hashlib.sha256((self.ws.resolve(artifact)[0]/'test_predictions.npz').read_bytes()).hexdigest())

    def test_optional_defaults_keep_aligned_actual_rows_and_do_not_mix_initial_condition_arrays(self):
        arrays={'time_s':np.arange(8,dtype=float).reshape(-1,1),'condition_index':np.repeat([2,3],4),
            'predicted_temperature_K':np.full((8,1),301.),'reference_temperature_K':np.full((8,1),300.),
            'initial_condition_index':np.array([2,3]),'normalized_initial_residual':np.array([[.01],[.02]])}
        artifact=self.stored(arrays,'pinn_model');result=read_report_arrays(self.ws,artifact,'test_predictions.npz',{})['file_preview']['array_rows']
        self.assertEqual(result['returned_rows'],8);self.assertNotIn('normalized_initial_residual',result['values'])
        with self.assertRaisesRegex(ContractError,'row axis'):read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['time_s','normalized_initial_residual']})

    def test_nonfinite_preview_becomes_explicit_gap_without_rewriting_full_report(self):
        artifact=self.stored({'prediction':np.array([[1.],[np.nan],[3.]]),'truth':np.array([[1.],[2.],[3.]])})
        result=read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['prediction','truth']})['file_preview']['array_rows']
        self.assertEqual(result['values']['prediction'],[[1.],[None],[3.]])

    def test_path_variable_filter_and_element_boundaries_reject(self):
        artifact=self.stored({'prediction':np.zeros((100,100)),'trajectory_id':np.zeros(100,dtype=int)})
        for name,config in [('../outside.npz',{'keys':['prediction']}),('test_predictions.npz',{'keys':['__proto__']}),
                ('test_predictions.npz',{'keys':['prediction'],'max_rows':True}),('test_predictions.npz',{'keys':['prediction']}),
                ('test_predictions.npz',{'keys':['prediction'],'filter':{'key':'prediction','value':0}}),
                ('test_predictions.npz',{'keys':['prediction'],'filter':{'key':'trajectory_id','value':'zero'}})]:
            with self.subTest(config=config),self.assertRaises(ContractError):read_report_arrays(self.ws,artifact,name,config)

    def test_hostile_npy_header_and_zip_duplicate_refused_before_np_load(self):
        header=io.BytesIO();np.lib.format.write_array_header_1_0(header,{'descr':'<f8','fortran_order':False,'shape':(100_000_000,1)})
        for duplicate in (False,True):
            with self.ws.stage('supervised_model',MaterialInterface('bundle'),[],{'synthetic':True}) as stage:
                with zipfile.ZipFile(stage.root/'test_predictions.npz','w') as archive:
                    archive.writestr('prediction.npy',header.getvalue())
                    if duplicate:
                        import warnings
                        with warnings.catch_warnings():warnings.simplefilter('ignore');archive.writestr('prediction.npy',header.getvalue())
                artifact=stage.publish({'status':'verified','scope':'malformed_negative_fixture'},activate=False)
            with patch('control_harness.reporting.np.load',side_effect=AssertionError('Header rejection must precede array allocation')),self.assertRaises(ContractError):
                read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['prediction']})

    def actual_bundle(self,kind='supervised_model',parents=None):
        data={'schema_version':1,'interface_kind':'tabular_supervised','name':'Synthetic untrained transport-only Tensor fixture','task':'static_regression','sample_time':None,
            'time_column':None,'trip_column':None,'signals':[{'name':name,'unit':'1','role':'auxiliary' if name!='y' else 'output','column':{'index':i,'header':name,'worksheet':None},'delay_steps':0,'observable':name!='y'} for i,name in enumerate(['a','b','y'])],
            'features':[{'signal':'a','window':1,'lag':0},{'signal':'b','window':1,'lag':0}],'targets':[{'signal':'y','offset':0}]}
        config={'schema_version':1,'architecture':'mlp','input_dim':2,'output_dim':1,'hidden_dims':[3],'activation':'tanh','dtype':'float64'}
        scaler={'method':'standard','fit_split':'train','fit_samples':10,'X':{'location':[0.,0.],'scale':[1.,1.]},'y':{'location':[0.],'scale':[1.]}}
        with torch.random.fork_rng():torch.manual_seed(51);network=build_supervised_network(config)
        with self.ws.stage(kind,DataInterface.parse(data),parents or [],{'synthetic':True,'training':'not_run'},compatible_parents=False) as stage:
            export_supervised(stage.root/'bundle',network,config,data,scaler,{'purpose':'Actual initialized Tensor serialization; no learning/performance claim'})
            artifact=stage.publish({'status':'verified','scope':'independent_export_transport_only'},activate=False)
        return artifact,network.state_dict()

    def test_complete_independent_zip_includes_real_weights_code_and_every_manifest_member(self):
        artifact,state=self.actual_bundle();descriptor=prepare_bundle_archive(self.ws,artifact)
        self.assertTrue(descriptor['complete']);self.assertFalse(descriptor['truncated']);self.assertEqual(descriptor['bytes'],Path(descriptor['host_path']).stat().st_size)
        self.assertEqual(descriptor['sha256'],hashlib.sha256(Path(descriptor['host_path']).read_bytes()).hexdigest())
        with zipfile.ZipFile(descriptor['host_path']) as archive:
            self.assertEqual(set(archive.namelist()),set(descriptor['member_sha256']))
            for name,digest in descriptor['member_sha256'].items():self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest)
            loaded=torch.load(io.BytesIO(archive.read('bundle/model.pt')),weights_only=True)
            self.assertEqual(set(loaded),set(state));self.assertTrue(all(torch.equal(loaded[k],state[k]) for k in state))
            self.assertIn('bundle/inference.py',archive.namelist());self.assertIn('bundle/predict.py',archive.namelist())
        self.assertEqual(prepare_bundle_archive(self.ws,artifact)['sha256'],descriptor['sha256'])

    def test_bundle_missing_tampered_cached_path_and_size_fail_without_partial_package(self):
        dataset=self.stored({'prediction':np.zeros((1,1))},kind='supervised_dataset')
        with self.assertRaisesRegex(ContractError,'no published standalone'):prepare_bundle_archive(self.ws,dataset)
        artifact,_=self.actual_bundle()
        with self.assertRaisesRegex(ContractError,'not be truncated'):prepare_bundle_archive(self.ws,artifact,max_bytes=1)
        descriptor=prepare_bundle_archive(self.ws,artifact)
        with zipfile.ZipFile(descriptor['host_path'],'a') as archive:archive.writestr('../outside','wrong')
        with self.assertRaisesRegex(ContractError,'inventory'):prepare_bundle_archive(self.ws,artifact)

    def test_artifact_symlink_is_rejected_before_file_read_even_when_bytes_match(self):
        artifact=self.stored({'prediction':np.ones((2,1))});root,_=self.ws.resolve(artifact);path=root/'test_predictions.npz';outside=self.root/'outside.npz';outside.write_bytes(path.read_bytes());path.unlink()
        try:path.symlink_to(outside)
        except OSError:
            path.write_bytes(outside.read_bytes())
            if os.name!='nt':raise
            # Windows without SeCreateSymbolicLinkPrivilege uses its native
            # unprivileged alias: same file bytes through a directory junction.
            relocated=self.root/'aliased-report-source';self.assertTrue(relocated.resolve().is_relative_to(self.root.resolve()))
            self.assertTrue(root.resolve().is_relative_to(self.ws.path.resolve()));root.rename(relocated);created=False
            try:
                result=subprocess.run(['cmd','/c','mklink','/J',str(root),str(relocated)],capture_output=True,text=True,check=False)
                self.assertEqual(result.returncode,0,'Windows alias negative must actually execute: '+result.stderr);created=True
                self.assertEqual((root/'test_predictions.npz').read_bytes(),outside.read_bytes())
                with self.assertRaisesRegex(ContractError,'symlinks|reparse'):read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['prediction']})
            finally:
                if created:os.rmdir(root)
                relocated.rename(root)
            return
        with self.assertRaisesRegex(ContractError,'symlinks|reparse'):read_report_arrays(self.ws,artifact,'test_predictions.npz',{'keys':['prediction']})

    @unittest.skipUnless(os.name=='nt','Windows directory-junction boundary')
    def test_real_directory_junction_cannot_redirect_artifact_reads_or_bundle_download(self):
        parent=self.stored({'prediction':np.ones((2,1))},kind='supervised_dataset');artifact,_=self.actual_bundle(parents=[parent]);root,_=self.ws.resolve(parent);outside=self.root/'same-bytes-outside-artifact-store'
        self.assertTrue(root.resolve().is_relative_to(self.ws.path.resolve()));self.assertTrue(outside.resolve().is_relative_to(self.root.resolve()))
        root.rename(outside)
        created=False
        try:
            result=subprocess.run(['cmd','/c','mklink','/J',str(root),str(outside)],capture_output=True,text=True,check=False)
            self.assertEqual(result.returncode,0,'Actual Windows junction creation must pass for this boundary check: '+result.stderr)
            created=True
            with self.assertRaisesRegex(ContractError,'symlinks|reparse'):prepare_bundle_archive(self.ws,artifact)
            with self.assertRaisesRegex(ContractError,'symlinks|reparse'):read_report_arrays(self.ws,artifact,'test_predictions.npz',{})
        finally:
            if created:os.rmdir(root)  # Remove the checked junction itself, never its target tree.
            outside.rename(root)


if __name__=='__main__':unittest.main()
