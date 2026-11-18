"""Independent real CPU MLP continuation and data-only identity boundaries.

Small synthetic CSV fixtures exercise actual material/data/checkpoint code.
These tests do not claim F03 quality acceptance, real LLM or native Host use.
"""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch

from control_harness.artifacts import Workspace, file_hash, read_json
from control_harness.contracts import ContractError
from control_harness.material_registration import register_material, prepare_supervised_dataset
from control_harness.runtime import Cancelled, OperationContext
from control_harness.supervised import train_supervised


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "supervised-resume-tests"))
HEADERS = ["trip", "f1", "f2", "f3", "target_a", "target_b"]


def interface(*, swap=False, unit="1"):
    column = lambda index: {"index": index, "header": HEADERS[index], "worksheet": None}
    signals = [{"name": name, "role": "auxiliary" if index <= 3 else "output", "unit": unit if index == 1 else "1",
        "column": column(index), "delay_steps": 0, "observable": index <= 3}
        for index,name in enumerate(HEADERS[1:],1)]
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "name": "independent synthetic resume fixture",
        "task": "static_regression", "sample_time": None, "time_column": None, "trip_column": column(0),
        "signals": signals, "features": [{"signal": name, "window": 1, "lag": 0} for name in
            (["f2", "f1", "f3"] if swap else ["f1", "f2", "f3"])],
        "targets": [{"signal": name, "offset": 0} for name in ["target_a", "target_b"]],
        "input_dim": 3, "output_dim": 2}


class SupervisedResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.fixture_root = OUTPUT / ("fixture-" + uuid.uuid4().hex)
        cls.fixture_root.mkdir()
        cls.fixture_ws = Workspace(cls.fixture_root / "case")
        source = cls.fixture_root / "synthetic.csv"
        rng = np.random.default_rng(20261117)
        with source.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.writer(stream); writer.writerow(HEADERS)
            for group in range(8):
                for _ in range(8):
                    a,b,c=rng.uniform(-1,1,3)
                    writer.writerow([f"synthetic-trip-{group}",a,b,c,.8*a-.5*b+.3*c+.2,-.2*a+.4*b+.6*c+.1*np.sin(np.pi*a)])
        cls.material_id = register_material(cls.fixture_ws,str(source),
            {"source_kind":"path","source_metadata":{"kind":"synthetic","description":"Independent real CPU resume tests"}},OperationContext())
        cls.dataset_config = {"interface":interface(),"split":{"mode":"trip","train":.5,"validation":.25,"test":.25},
            "scaler":"standard","source":{"kind":"synthetic","description":"Synthetic resume fixture, not vehicle data"}}
        cls.dataset_id = prepare_supervised_dataset(cls.fixture_ws,cls.material_id,cls.dataset_config,OperationContext())
        cls.config = {"hidden_dims":[7,5],"activation":"tanh","epochs":6,"learning_rate":.004,
            "batch_size":7,"patience":50,"seed":20261117,"weight_decay":.001,"min_delta":0.,"grad_clip":5.}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.fixture_root)
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.root=OUTPUT/("run-"+uuid.uuid4().hex)
        shutil.copytree(self.fixture_ws.path,self.root)
        self.addCleanup(lambda:shutil.rmtree(self.root))
        self.ws=Workspace(self.root)

    def train(self,config=None,context=None,dataset_id=None):
        return train_supervised(self.ws,dataset_id or self.dataset_id,config or copy.deepcopy(self.config),context)

    def artifact(self,artifact_id):
        return self.ws.resolve(artifact_id,kind="supervised_model")[0]

    def checkpoint(self,artifact_id):
        return torch.load(self.artifact(artifact_id)/"latest.pt",map_location="cpu",weights_only=True)

    def assert_nested_equal(self,actual,expected):
        if isinstance(expected,torch.Tensor):
            self.assertTrue(torch.equal(actual,expected),"Continuation changed a tensor")
        elif isinstance(expected,dict):
            self.assertEqual(set(actual),set(expected))
            for key in expected:self.assert_nested_equal(actual[key],expected[key])
        elif isinstance(expected,(tuple,list)):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):self.assert_nested_equal(a,b)
        else:self.assertEqual(actual,expected)

    def interrupted(self,config=None,exception=Cancelled):
        progress=[]
        def emit(event):
            progress.append(event)
            if event.get("epoch")==2:raise exception("independent interruption after committed epoch two")
        with self.assertRaises(exception):self.train(config,OperationContext(emit=emit))
        path=Path(progress[-1]["checkpoint"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.parent,self.ws.path/"failed")
        self.assertTrue((path.parent/"failure.json").is_file())
        self.assertTrue((path.parent/"latest.checkpoint.json").is_file())
        self.assertFalse(self.ws.lock_path.exists())
        self.assertEqual(torch.load(path,weights_only=True)["epoch"],2)
        return path

    def test_cancel_resume_matches_continuous_model_optimizer_rng_log_and_selected_model(self):
        continuous=self.train()
        path=self.interrupted()
        resumed=self.train({**self.config,"resume_checkpoint":str(path),"resume_checkpoint_sha256":file_hash(path)})
        self.assert_nested_equal(self.checkpoint(resumed),self.checkpoint(continuous))
        self.assert_nested_equal(torch.load(self.artifact(resumed)/"model.pt",weights_only=True),
            torch.load(self.artifact(continuous)/"model.pt",weights_only=True))
        report=read_json(self.artifact(resumed)/"report.json")
        self.assertEqual(report["resumed_from_epoch"],2)
        self.assertEqual(report["trained_epochs"],6)
        self.assertEqual(report["optimizer_updates"],self.checkpoint(resumed)["updates"])
        self.assertEqual(self.checkpoint(resumed)["contract"]["algorithm"],"supervised_training")
        self.assertGreater(report["optimizer_updates"],0)

    def test_runtime_failure_and_relative_checkpoint_path_resume_identically(self):
        continuous=self.train()
        path=self.interrupted(exception=RuntimeError)
        resumed=self.train({**self.config,"resume_checkpoint":path.relative_to(self.ws.path).as_posix()})
        self.assert_nested_equal(self.checkpoint(resumed),self.checkpoint(continuous))

    def test_partial_first_epoch_replays_from_committed_epoch_zero(self):
        continuous=self.train()
        checks=0
        def cancelled():
            nonlocal checks
            checks+=1
            return checks>=5
        with self.assertRaises(Cancelled):self.train(context=OperationContext(cancelled=cancelled))
        paths=list((self.ws.path/"failed").glob("*/latest.pt"))
        self.assertEqual(len(paths),1)
        self.assertEqual(torch.load(paths[0],weights_only=True)["epoch"],0)
        resumed=self.train({**self.config,"resume_checkpoint":str(paths[0])})
        self.assert_nested_equal(self.checkpoint(resumed),self.checkpoint(continuous))

    def test_target_total_epochs_and_confirmation_refs_do_not_reset_optimizer_identity(self):
        full=self.train()
        short=self.train({**self.config,"epochs":2,"plan_refs":{"training":{"plan_id":"first","version_hash":"a"*64}}})
        resumed=self.train({**self.config,"resume_checkpoint":str(self.artifact(short)/"latest.pt"),
            "plan_refs":{"training":{"plan_id":"second","version_hash":"b"*64}}})
        self.assert_nested_equal(self.checkpoint(resumed),self.checkpoint(full))
        contract=self.checkpoint(resumed)["contract"]
        for key in ("epochs","resume_checkpoint","resume_checkpoint_sha256","plan_refs"):
            self.assertNotIn(key,contract["config"])

    def test_early_stop_wait_survives_a_larger_total_epoch_budget(self):
        config={**self.config,"epochs":12,"patience":1,"min_delta":1e6}
        stopped=self.train(config)
        before=self.checkpoint(stopped)
        self.assertLess(before["epoch"],12)
        resumed=self.train({**config,"epochs":18,"resume_checkpoint":str(self.artifact(stopped)/"latest.pt")})
        self.assert_nested_equal(self.checkpoint(resumed),before)

    def test_changed_layers_learning_rate_activation_and_numerical_settings_cannot_force_resume(self):
        path=self.interrupted()
        for changed in ({"hidden_dims":[9,5]},{"learning_rate":.002},{"activation":"relu"},
                {"batch_size":8},{"weight_decay":.01},{"seed":22},{"min_delta":.1},{"grad_clip":2.}):
            with self.subTest(changed=changed),self.assertRaises(ContractError):
                self.train({**self.config,**changed,"resume_checkpoint":str(path)})

    def test_changed_dataset_features_and_units_cannot_reuse_the_optimizer(self):
        path=self.interrupted()
        for changed in ({"interface":interface(swap=True)},{"interface":interface(unit="m")},{"scaler":"minmax"}):
            config={**copy.deepcopy(self.dataset_config),**changed}
            other=prepare_supervised_dataset(self.ws,self.material_id,config,OperationContext())
            with self.subTest(changed=changed),self.assertRaises(ContractError):
                self.train({**self.config,"resume_checkpoint":str(path)},dataset_id=other)
        # Even identical features on a separately versioned material/dataset
        # cannot silently continue the old optimizer trajectory.
        another=prepare_supervised_dataset(self.ws,self.material_id,self.dataset_config,OperationContext())
        with self.assertRaises(ContractError):self.train({**self.config,"resume_checkpoint":str(path)},dataset_id=another)
        material_root,_=self.ws.resolve(self.material_id)
        description=read_json(material_root/"material.json")
        with (material_root/description["primary_file"]).open(encoding="utf-8",newline="") as stream:
            changed_rows=list(csv.reader(stream))
        a,b,c=map(float,changed_rows[1][1:4]);a+=.05
        changed_rows[1][1:]=[a,b,c,.8*a-.5*b+.3*c+.2,-.2*a+.4*b+.6*c+.1*np.sin(np.pi*a)]
        changed_source=self.ws.path/"changed-synthetic.csv"
        with changed_source.open("w",encoding="utf-8",newline="") as stream:csv.writer(stream).writerows(changed_rows)
        changed_material=register_material(self.ws,str(changed_source),{"source_kind":"path"},OperationContext())
        changed_dataset=prepare_supervised_dataset(self.ws,changed_material,self.dataset_config,OperationContext())
        with self.assertRaises(ContractError):
            self.train({**self.config,"resume_checkpoint":str(path)},dataset_id=changed_dataset)

    def test_wrong_hash_decreased_budget_external_path_and_tampered_data_reject(self):
        path=self.interrupted()
        for changed in ({"resume_checkpoint_sha256":"0"*64},{"epochs":1},
                {"resume_checkpoint":str(self.fixture_root/"outside-latest.pt")}):
            with self.subTest(changed=changed),self.assertRaises(ContractError):
                self.train({**self.config,"resume_checkpoint":str(path),**changed})
        root,_=self.ws.resolve(self.dataset_id)
        source=root/"supervised.npz"
        source.write_bytes(source.read_bytes()+b"tamper")
        with self.assertRaises(ContractError):self.train({**self.config,"resume_checkpoint":str(path)})


if __name__=="__main__":unittest.main()
