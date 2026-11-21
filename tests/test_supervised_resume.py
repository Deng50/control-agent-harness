"""Independent real CPU MLP continuation and data-only identity boundaries.

Small synthetic CSV fixtures exercise actual material/data/checkpoint code.
These tests do not claim F03 quality acceptance, real LLM or native Host use.
"""
from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace, failure_fields, file_hash, read_json
from control_harness import checkpoints
from control_harness.contracts import ContractError
from control_harness.material_registration import register_material, prepare_supervised_dataset
from control_harness.runtime import Cancelled, OperationContext
from control_harness.supervised import train_supervised


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "supervised-resume-tests"))
HEADERS = ["trip", "f1", "f2", "f3", "target_a", "target_b"]


def run_windows_share_probe(workspace, dataset, config):
    """子进程只持有自己新建的测试阶段指针句柄；退出之后父进程才可执行孤儿归档。"""
    import ctypes
    from ctypes import wintypes
    torch.set_num_threads(1)
    ws = Workspace(workspace)
    if not ws.path.is_relative_to((ROOT / "test-output").resolve()):
        raise ValueError("Sharing probe must run inside the repository test-output directory")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    handle = None
    observed = {}

    def hold_after_commit(event):
        nonlocal handle
        if event.get("stage") != "supervised_training" or event.get("epoch") != 1:
            return
        stage_id = Path(event["checkpoint"]).parent.name
        pointer = ws.path / ".staging" / stage_id / checkpoints.HEAD
        if pointer.resolve() != pointer or read_json(pointer.parent / "stage.intent.json")["owner"]["pid"] != os.getpid():
            raise RuntimeError("Refused to lock a checkpoint from another stage")
        # GENERIC_READ，允许其他进程读写，但本次持有期间不共享 DELETE。
        # OPEN_EXISTING 只打开真实训练器已提交的指针，不创建或改写它。
        handle = create(str(pointer), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
        if handle == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        # 另一个打开请求仅申请 DELETE 访问，不执行删除。记录内核返回的共享冲突码作为对照。
        probe = create(str(pointer), 0x00010000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
        if probe != invalid:
            close(probe)
            raise RuntimeError("The handle unexpectedly shared DELETE")
        observed.update(delete_access_winerror=ctypes.get_last_error(), committed=event)

    try:
        train_supervised(ws, dataset, config, OperationContext(emit=hold_after_commit))
    except checkpoints.CheckpointWriteError as error:
        observed["facts"] = failure_fields(error)
    finally:
        if handle not in (None, invalid) and not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    return observed


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

    def test_each_uncommitted_write_failure_preserves_previous_epoch_and_exact_resume(self):
        # 在相同小数据上，在下一轮的三个提交边界分别注入拒绝访问；校验冻结点再逐张量对比续训。
        baseline = self.checkpoint(self.train())
        real_open, real_publish = Path.open, checkpoints._publish_pointer
        for boundary in ("payload", "sidecar", "pointer"):
            progress = []
            def fail_open(path, mode="r", *args, **kwargs):
                if progress and path.stem.startswith("checkpoint-") and (
                        boundary == "payload" and mode == "xb" or boundary == "sidecar" and mode == "x"):
                    raise PermissionError("denied at " + boundary)
                return real_open(path, mode, *args, **kwargs)

            def fail_pointer(path, value):
                if progress and boundary == "pointer":
                    raise PermissionError("denied at pointer")
                return real_publish(path, value)

            def emit(event):
                if event.get("epoch") == 1:
                    progress.append(event)

            with self.subTest(boundary=boundary), patch.object(Path, "open", fail_open), \
                    patch.object(checkpoints, "_publish_pointer", fail_pointer), \
                    self.assertRaises(checkpoints.CheckpointWriteError) as caught:
                self.train(context=OperationContext(emit=emit))
            facts = failure_fields(caught.exception)
            path = Path(facts["checkpoint"])
            self.assertEqual(facts["error_code"], "CHECKPOINT_ACCESS_DENIED")
            self.assertEqual(facts["stage_failure"]["archive_state"], "archived")
            self.assertEqual(facts["checkpoint_status"], "archived")
            self.assertEqual((facts["epoch"], facts["target_epochs"]), (1, self.config["epochs"]))
            self.assertEqual(path, checkpoints.current_checkpoint(path.parent))
            self.assertEqual(facts["checkpoint_sha256"], progress[0]["checkpoint_sha256"])
            self.assertEqual(facts["checkpoint_sha256"], file_hash(path))
            head = read_json(path.parent / checkpoints.HEAD)
            self.assertTrue((path.parent / (head["previous"] + ".pt")).is_file())
            for orphan in path.parent.glob("checkpoint-*.pt"):
                if orphan.stem not in (head["current"], head["previous"]):
                    with self.assertRaisesRegex(ContractError, "not committed"):
                        self.train({**self.config, "resume_checkpoint": str(orphan)})
            resumed = self.train({**self.config, "resume_checkpoint": str(path),
                                  "resume_checkpoint_sha256": facts["checkpoint_sha256"]})
            self.assert_nested_equal(baseline, self.checkpoint(resumed))
            self.assertEqual(len(list(self.artifact(resumed).glob("checkpoint-*.pt"))), 2)

    def test_first_commit_failure_has_no_resume_and_archive_failure_keeps_primary_error(self):
        with patch.object(checkpoints, "_publish_pointer", side_effect=PermissionError("first pointer denied")), \
                self.assertRaises(checkpoints.CheckpointWriteError) as initial:
            self.train()
        self.assertIn("first pointer denied", str(initial.exception))
        self.assertNotIn("checkpoint", failure_fields(initial.exception))
        retained = Path(initial.exception.stage_failure["retained_path"])
        self.assertFalse((retained / checkpoints.HEAD).exists())
        self.assertFalse((retained / "latest.pt").exists())

        def zip_write_failure(*args):
            try:
                raise PermissionError("checkpoint volume refused write")
            except OSError:
                raise RuntimeError("unexpected ZIP footer position")

        with patch.object(checkpoints.torch, "save", zip_write_failure), self.assertRaises(checkpoints.CheckpointWriteError) as wrapped:
            self.train()
        self.assertEqual(wrapped.exception.code, "CHECKPOINT_ACCESS_DENIED")
        self.assertIn("checkpoint volume refused write", str(wrapped.exception))
        self.assertNotIn("checkpoint", failure_fields(wrapped.exception))

        real_rename = Path.rename
        def deny_archive(source, destination):
            if source.parent.name == ".staging" and destination.parent.name == "failed":
                raise PermissionError("archive denied")
            return real_rename(source, destination)

        def lose_event(event):
            if event.get("epoch") == 2:
                raise RuntimeError("original progress transport failure")

        with patch.object(Path, "rename", deny_archive), self.assertRaisesRegex(RuntimeError, "original progress") as caught:
            self.train(context=OperationContext(emit=lose_event))
        facts = failure_fields(caught.exception)
        self.assertEqual((facts["checkpoint_status"], facts["epoch"]), ("retained", 2))
        self.assertIn("archive denied", " ".join(facts["stage_failure"]["archive_errors"]))
        path = Path(facts["checkpoint"])
        self.assertEqual(path.parent.parent, self.ws.path / ".staging")
        self.assertEqual(facts["checkpoint_sha256"], file_hash(path))
        with self.assertRaises(ContractError):
            self.train({**self.config, "resume_checkpoint": str(path)})

    def test_pointer_commit_before_failed_acknowledgement_reports_actual_epoch(self):
        committed = 0
        progress = []
        real_publish = checkpoints._publish_pointer
        def fail_after_commit(path, head):
            nonlocal committed
            real_publish(path, head)
            committed += 1
            if committed == 3:  # 已提交第 2 轮，宿主只看到第 1 轮进度。
                raise OSError("lost pointer acknowledgement")
        with patch.object(checkpoints, "_publish_pointer", fail_after_commit), \
                self.assertRaises(checkpoints.CheckpointWriteError) as caught:
            self.train(context=OperationContext(emit=progress.append))
        facts = failure_fields(caught.exception)
        self.assertEqual((progress[-1]["epoch"], facts["epoch"], facts["checkpoint_selection"]), (1, 2, "current"))
        self.assertEqual(facts["checkpoint_sha256"], file_hash(Path(facts["checkpoint"])))

    def test_genuine_legacy_pair_is_read_only_and_generation_snapshot_requires_pointer(self):
        selected = self.interrupted()
        legacy = selected.with_name("latest.pt")
        sidecar = checkpoints.checkpoint_sidecar(legacy)
        metadata = read_json(sidecar)
        (legacy.parent / checkpoints.HEAD).unlink()
        with self.assertRaisesRegex(ContractError, "no committed pointer"):
            self.train({**self.config, "resume_checkpoint": str(legacy)})
        # 旧旁注本来没有代号或目标轮次，原字节和训练签名仍需完整校验。
        metadata.pop("generation")
        metadata.pop("target_epochs")
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        before = (file_hash(legacy), file_hash(sidecar))
        resumed = self.train({**self.config, "resume_checkpoint": str(legacy), "resume_checkpoint_sha256": before[0]})
        self.assertEqual(read_json(self.artifact(resumed) / "report.json")["resumed_from_epoch"], 2)
        self.assertEqual((file_hash(legacy), file_hash(sidecar)), before)

    def test_pointer_retry_is_limited_to_windows_sharing_and_lock_conflicts(self):
        busy = PermissionError("sharing violation")
        busy.winerror = 32
        denied = PermissionError("access denied")
        denied.winerror = 5
        for failures, expected_attempts in (([busy, None], 2), ([busy] * 3, 3), ([denied], 1)):
            with patch.object(checkpoints, "atomic_json", side_effect=failures) as write, \
                    patch.object(checkpoints.time, "sleep"):
                if failures[-1] is None:
                    checkpoints._publish_pointer(self.root / checkpoints.HEAD, {})
                else:
                    with self.assertRaises(PermissionError):
                        checkpoints._publish_pointer(self.root / checkpoints.HEAD, {})
                self.assertEqual(write.call_count, expected_attempts)

    @unittest.skipUnless(os.name == "nt", "Requires Windows CreateFileW share modes")
    def test_windows_no_share_delete_blocks_checkpoint_commit_and_resumes_exactly(self):
        if not self.root.resolve().is_relative_to((ROOT / "test-output").resolve()):
            self.skipTest("Real handle probe is confined to this repository's test-output")
        expected = self.checkpoint(self.train())
        child_code = ("import json,sys; from test_supervised_resume import run_windows_share_probe; "
                      "print(json.dumps(run_windows_share_probe(**json.load(sys.stdin))))")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(ROOT / name) for name in ("src", "tests"))}
        child = subprocess.run([sys.executable, "-c", child_code], text=True, capture_output=True, timeout=60,
                               env=env, input=json.dumps({"workspace": str(self.root), "dataset": self.dataset_id,
                                                          "config": self.config}))
        self.assertEqual(child.returncode, 0, child.stderr)
        observed = json.loads(child.stdout)
        self.assertEqual(observed["delete_access_winerror"], 32)
        facts = observed["facts"]
        replace_error = facts["io_failure"]["winerror"]
        # MoveFileExW 在部分 Windows 文件系统上把删除目标受限返回为 5。
        # 记录替换操作本身的原始码，不能把独立 DELETE 探针的 32 强加给训练器。
        self.assertIn(replace_error, (5, 32, 33))
        self.assertEqual(facts["error_code"], "CHECKPOINT_ACCESS_DENIED" if replace_error == 5 else "CHECKPOINT_FILE_BUSY")
        self.assertEqual(facts["io_failure"]["operation"], "commit checkpoint pointer")
        self.assertEqual(facts["epoch"], 1)
        self.assertEqual(facts["target_epochs"], self.config["epochs"])
        self.assertEqual(facts["checkpoint_sha256"], observed["committed"]["checkpoint_sha256"])
        if facts["checkpoint_status"] == "retained":
            from control_harness.recovery import recover_interrupted_stage
            archived = recover_interrupted_stage(self.ws, facts["stage_failure"]["stage_id"])
        else:
            self.assertEqual(facts["checkpoint_status"], "archived")
            archived = facts
        self.assertEqual(file_hash(Path(archived["checkpoint"])), facts["checkpoint_sha256"])
        resumed = self.train({**self.config, "resume_checkpoint": archived["checkpoint"],
                              "resume_checkpoint_sha256": facts["checkpoint_sha256"]})
        self.assert_nested_equal(self.checkpoint(resumed), expected)
        report = read_json(self.artifact(resumed) / "report.json")
        self.assertEqual(report["resumed_from_epoch"], 1)
        print(f"Windows checkpoint sharing: DELETE open=32, pointer replace={replace_error}, "
              f"{facts['error_code']}, stage={facts['checkpoint_status']}, resumed=1/{report['trained_epochs']}")

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
