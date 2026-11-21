"""B01/B05 实际对象与专家的低预算预检回归。不训练网络，不启动 MATLAB。"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.case_plans import FIXTURE_KIND, propose_plan_batch
from control_harness.case_requirements import confirmation_requirements
from control_harness.contracts import ContractError
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
from control_harness.stage_contracts import check_inputs


PROJECT = Path(__file__).resolve().parents[1]
CASES = PROJECT.parent / "control-agent-benchmark/cases"


@unittest.skipUnless(CASES.is_dir(), "需要与本仓并列的 control-agent-benchmark 原始案例")
class BenchmarkCapabilityPreflight(unittest.TestCase):
    def setUp(self):
        folder = PROJECT / "test-output/benchmark-preflight"
        folder.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(directory.cleanup)
        self.ws = Workspace(Path(directory.name))
        atomic_json(self.ws.path / ".case-plans/test-fixture.json",
                    {"kind": FIXTURE_KIND, "workspace": str(self.ws.path)})
        self.context = OperationContext.test_fixture(self.ws)

    def request(self, operation, inputs, config):
        return {"workspace": str(self.ws.path), "operation": operation, "inputs": inputs, "config": config}

    def execute(self, operation, inputs, config):
        # 经正式入口绑定所有要求的版本。测试夹具只能在本临时工作区内签收。
        request = self.request(operation, inputs, dict(config))
        requirements = confirmation_requirements(self.ws, request, structured=True, context=self.context)
        if requirements:
            rows = [{"scope": row["scope"], "reviewPayload": row["review_payload"],
                     "reviewBindings": row["review_bindings"]} for row in requirements]
            plans = propose_plan_batch(self.ws, operation, rows)["plans"]
            request["config"]["plan_refs"] = {p["scope"]: {k: p[k] for k in ("plan_id", "version_hash")} for p in plans}
        return run_operation(request, self.context)

    def register(self, folder, source, adapter, operation):
        material = self.execute("register_material", {"local_path": str(folder / source)}, {})["artifact_id"]
        return self.execute(operation, {"material": material}, read_json(folder / adapter))["artifact_id"]

    def host_checks(self, *requests):
        # 调用真实 Node Host 预检，它自行从 manifest 读产物及版本；不以 Python 伪造元数据。
        script = """
import {readFileSync} from 'node:fs';
import {stagePreflight} from './harness-plugin/src/case-requirements.mjs';
const requests = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify(await Promise.all(requests.map(stagePreflight))));
"""
        result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=PROJECT, timeout=60,
                                input=json.dumps(requests), capture_output=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_b01_same_physical_interface_cannot_substitute_another_source_or_revision(self):
        folder = CASES / "B01_model_training/inputs"
        original = self.register(folder, "plant_numpy.py", "plant_adapter.json", "register_plant")
        another_version = self.register(folder, "plant_numpy.py", "plant_adapter.json", "register_plant")
        matlab = self.register(folder, "spring_step.m", "matlab_function_adapter.json", "register_plant")
        # 三个对象的单位、维度、约束完全一致。另一个登记版本与 MATLAB 程序分别检验版本和来源。
        meta = [self.ws.resolve(key)[1] for key in (original, another_version, matlab)]
        self.assertEqual(len({item["interface_hash"] for item in meta}), 1)
        self.assertEqual(meta[0]["source"]["content_hash"], meta[1]["source"]["content_hash"])
        self.assertNotEqual(meta[0]["source"]["content_hash"], meta[2]["source"]["content_hash"])
        sampling = self.execute("recommend_sampling", {"model": original},
                                {"overrides": {"trajectory_count": 5, "steps": 3}})["sampling"]
        dataset = self.execute("sample_model", {"model": original}, {"sampling": sampling, "sampling_run": "full"})["artifact_id"]
        requests = [self.request("convert_plant_predictor", {"plant": key, "dataset": dataset},
                                 {"method": "sampled_affine"}) for key in (original, another_version, matlab)]
        statuses = self.host_checks(*requests)
        self.assertEqual([r["status"] for r in statuses], ["ready", "unsupported", "unsupported"])
        for request, status in zip(requests[1:], statuses[1:]):
            self.assertEqual(status["jobs_started"], 0)
            self.assertTrue(any(issue["field"] == "inputs.dataset" for issue in status["issues"]))
            with self.assertRaisesRegex(ContractError, "inputs.dataset"):
                self.execute(request["operation"], request["inputs"], request["config"])
        self.assertFalse(any(row["kind"] == "identified_model" for row in self.ws.manifest["artifacts"].values()))

    def test_b05_five_real_features_are_checked_before_training(self):
        folder = CASES / "B05_external_expert/inputs"
        plant = self.register(folder, "plant_numpy.py", "plant_adapter.json", "register_plant")
        expert = self.register(folder, "expert_mpc.py", "expert_adapter.json", "register_expert")
        collection = read_json(folder / "collection_conditions.json")
        # 保留三个原始分区、原始初态/目标/扰动，只缩短每个工况至两次决策，足以检查上一拍动作。
        scenarios = collection["scenarios"]
        collection["scenarios"] = [{**scenario, "steps": 2}
                                   for part, count in (("train", 3), ("validation", 1), ("test", 1))
                                   for scenario in [s for s in scenarios if s["split"] == part][:count]]
        dataset = self.execute("collect_external_expert", {"expert": expert, "plant": plant}, collection)["artifact_id"]
        root, meta = self.ws.resolve(dataset)
        self.assertEqual(meta["source"]["expert"], expert)
        self.assertEqual(meta["source"]["plant"], plant)
        self.assertEqual(meta["source"]["student_feature_scope"], "simulation_only")
        self.assertEqual(meta["interface"]["input_dim"], 5)
        self.assertEqual([feature["signal"] for feature in meta["interface"]["features"]], collection["student_features"])
        with np.load(root / "expert.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["observation"].shape, (10, 2))
            np.testing.assert_array_equal(arrays["previous_u"][1::2], arrays["u"][::2])
        requests = [self.request("train_external_imitation", {"dataset": dataset},
                                 {"network": {"input_dim": width, "output_dim": 1}}) for width in (4, 5)]
        bad, good = self.host_checks(*requests)
        self.assertEqual([bad["status"], good["status"]], ["unsupported", "ready"])
        self.assertEqual(bad["issues"][0]["field"], "config.network.input_dim")
        self.assertEqual(bad["jobs_started"], good["jobs_started"])
        self.assertEqual(good["jobs_started"], 0)
        with self.assertRaisesRegex(ContractError, "需要 5，收到 4"):
            self.execute(requests[0]["operation"], requests[0]["inputs"], requests[0]["config"])
        # 正向到达数值作业自己的预检边界即结束。此处的 ready 仅说明维度兼容。
        good_request = requests[1]
        check_inputs(self.ws, good_request["operation"], good_request["inputs"], good_request["config"])
        self.assertFalse(any("policy" in row["kind"] for row in self.ws.manifest["artifacts"].values()))
