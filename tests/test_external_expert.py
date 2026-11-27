"""Actual explicit-memory experts/Plants and independent causal collection oracles."""
from __future__ import annotations

import copy
import base64
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from control_harness.artifacts import Workspace, read_json
from control_harness.contracts import ContractError
from control_harness.expert_adapter import validate_expert_spec, load_external_expert
from control_harness.external_expert import (register_expert, collect_external_expert, build_expert_collection_plan,
    ExpertCollectionStopped, ExpertCollectionCancelled)
from control_harness.material_registration import register_material
from control_harness.plants import register_plant
from control_harness.runtime import OperationContext
from control_harness.runtime import Cancelled


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-output/external-expert-tests"
OUTPUT.mkdir(parents=True, exist_ok=True)


def physical(nu=1):
    variable = lambda name, unit, bound: {"name": name, "unit": unit, "lower": -bound, "upper": bound}
    return {"schema_version": 1, "name": "explicit synthetic external-expert Plant", "sample_time": .1,
        "time_semantics": "discrete_state_transition", "states": [variable("position", "m", 2.)],
        "controls": [variable("force" + str(index), "N", 1.) for index in range(nu)],
        "outputs": [variable("position_sensor", "m", 2.)], "disturbances": [variable("wind", "N", .2)]}


def declaration(nu=1):
    interface = physical(nu)
    return {"schema_version": 1, "backend": "python_callable", "source_file": "source/expert.py", "entrypoint": "expert",
        "interface": interface, "parameters": {"control_period": .2}, "time_unit": "s", "control_period": .2,
        "observations": [{"name": "position_measurement", "source": "output", "index": 0, "plant_name": "position_sensor", "unit": "m", "availability": "measured", "delay_steps": 0}],
        "reference_available": True, "disturbance_available": True, "previous_action_available": True,
        "initial_memory": {"integral": 0., "calls": 0}, "failure_policy": "stop",
        "action_mapping": [{"action_index": index, "plant_control_index": index, "plant_control_name": "force" + str(index), "unit": "N"} for index in range(nu)]}


EXPERT = '''import numpy as np
from scipy.optimize import minimize
def expert(obs,r,d,previous,memory,p):
    error=float(r[0]-obs[0])
    target=.4*error+.1*float(previous[0])+.05*memory['integral']
    solution=minimize(lambda a:float((a[0]-target)**2),np.array([0.],dtype=np.float64),jac=lambda a:np.array([2*(a[0]-target)],dtype=np.float64),bounds=[(-1.,1.)],method='SLSQP',options={'ftol':1e-12,'maxiter':20})
    if not solution.success:return (np.zeros(1,dtype=np.float64),memory,'solver_failed')
    next_memory={'integral':memory['integral']+p['control_period']*error,'calls':memory['calls']+1}
    return (np.asarray(solution.x,dtype=np.float64),next_memory,'ok')
'''


def configuration():
    return {"seed": 20261123, "scaler": "standard", "scenarios": [{"name": "synthetic-" + str(index),
        "initial_state": [-.3+.1*index], "reference": [.2+.02*index], "disturbance": [-.05+.01*index],
        "initial_previous_action": [.01*index], "steps": 4,
        "split": "train" if index < 4 else "validation" if index == 4 else "test"} for index in range(6)]}


class ExternalExpertTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.ws = Workspace(self.root / "workspace")

    def register(self, source=EXPERT, spec=None, plant_gain=.2):
        context = OperationContext(); spec = copy.deepcopy(spec or declaration())
        source_path = self.root / "expert.py"; source_path.write_text(source, encoding="utf-8")
        material = register_material(self.ws, str(source_path), {}, context)
        expert = register_expert(self.ws, material, {"adapter": spec}, context)
        nu = len(spec["interface"]["controls"])
        plant_path = self.root / "plant.py"
        plant_path.write_text("import numpy as np\ndef transition(x,u,d,dt,p):\n return np.array([.85*x[0]+p['gain']*u[0]+" + (".1*u[1]+" if nu == 2 else "") + ".05*d[0]],dtype=np.float64)\n", encoding="utf-8")
        plant_material = register_material(self.ws, str(plant_path), {}, context)
        plant = register_plant(self.ws, plant_material, {"adapter": {"schema_version": 1, "backend": "python_callable",
            "source_file": "source/plant.py", "interface": spec["interface"], "parameters": {"gain": plant_gain},
            "source_semantics": "transition", "entrypoint": "transition", "C": [[1.]]}}, context)
        return expert, plant, material, spec

    def collect(self, expert, plant, config=None, context=None):
        with patch("control_harness.control.design_controller", side_effect=AssertionError("no MPC redesign")), patch("control_harness.control.MPCPolicy", side_effect=AssertionError("no internal MPC")):
            result = collect_external_expert(self.ws, expert, plant, configuration() if config is None else config, context)
        root, metadata = self.ws.resolve(result)
        return root, metadata, read_json(root / "report.json")

    def test_dynamic_reference_disturbance_and_seeded_sensor_noise_at_decisions(self):
        expert,plant,_,_=self.register()
        config=configuration()
        first=config['scenarios'][0]
        first.update(family='reversal',reference_sequence=[first['reference'],[.1],[-.2],[-.2]],
                     disturbance_sequence=[first['disturbance'],[-.04],[.02],[.07]],
                     observation_noise_std=[.01],noise_seed=123)
        root,_,_=self.collect(expert,plant,config)
        with np.load(root/'expert.npz',allow_pickle=False) as rows:
            idx=rows['trajectory_id']==0
            np.testing.assert_array_equal(rows['reference'][idx],first['reference_sequence'])
            np.testing.assert_array_equal(rows['d'][idx],first['disturbance_sequence'])
            np.testing.assert_allclose(rows['observation'][idx]-rows['plant_y'][idx],
                                       np.random.default_rng(123).normal(size=(4,1))*.01,atol=1e-15)
            x,u,d=rows['x'][idx],rows['u'][idx],rows['d'][idx]
            next_x=.85*(.85*x+.2*u+.05*d)+.2*u+.05*d
            np.testing.assert_allclose(rows['next_x'][idx],next_x,atol=1e-15)

    def test_real_scipy_expert_memory_reset_period_and_causal_arrays_match_independent_oracle(self):
        expert, plant, _, _ = self.register()
        root, metadata, report = self.collect(expert, plant)
        self.assertEqual(metadata["kind"], "external_expert_dataset")
        self.assertEqual(metadata["interface"]["sample_time"], .2)
        self.assertEqual(metadata["interface"]["input_dim"], 4)
        self.assertNotIn("nx", metadata["interface"])
        self.assertEqual(report["controller_design"], "not_run"); self.assertFalse(report["internal_MPC_required"])
        self.assertEqual(report["plant_substeps"], 2); self.assertEqual(report["plant_dt"], .1)
        self.assertEqual(report["source"]["kind"], "declared_simulation"); self.assertFalse(report["source"]["measured"])
        self.assertFalse(any(row["kind"] in {"controller", "controller_design", "nominal_model"} for row in self.ws.manifest["artifacts"].values()))
        diagnostics = read_json(root / "diagnostics.json")
        self.assertEqual(report["expert"]["source_invocations_this_adapter"], 3 * len(diagnostics))
        with np.load(root / "expert.npz", allow_pickle=False) as arrays:
            row = 0
            for trajectory, scenario in enumerate(configuration()["scenarios"]):
                state, integral, previous = scenario["initial_state"][0], 0., scenario["initial_previous_action"][0]
                reference, disturbance = scenario["reference"][0], scenario["disturbance"][0]
                for step in range(4):
                    error = reference-state
                    action = max(-1., min(1., .4*error+.1*previous+.05*integral))
                    np.testing.assert_allclose(arrays["X"][row], [state, reference, disturbance, previous], rtol=0, atol=1e-12)
                    np.testing.assert_allclose(arrays["y"][row], [action], rtol=0, atol=1e-12)
                    self.assertEqual(diagnostics[row]["memory_before"]["calls"], step)
                    self.assertAlmostEqual(diagnostics[row]["memory_before"]["integral"], integral, places=13)
                    integral += .2*error
                    for _ in range(2): state = .85*state+.2*action+.05*disturbance
                    np.testing.assert_allclose(arrays["next_x"][row], [state], rtol=0, atol=1e-12)
                    self.assertAlmostEqual(arrays["time"][row], step*.2, places=13)
                    self.assertEqual(int(arrays["trajectory_id"][row]), trajectory)
                    previous=action; row+=1
            self.assertEqual(row, 24)
            split = arrays["split_id"]
            self.assertEqual([int(np.count_nonzero(split==index)) for index in range(3)], [16,4,4])
            expected_mean = arrays["X"][split==0].mean(0)
        np.testing.assert_allclose(read_json(root / "scaler.json")["X"]["location"], expected_mean, rtol=0, atol=1e-14)
        self.assertEqual(report["expert"]["resets"], 6)

    def test_resume_restores_expert_memory_diagnostics_and_matches_full_collection(self):
        expert, plant, _, _ = self.register()
        baseline, _, _ = self.collect(expert, plant)
        commits = []
        context = OperationContext(emit=lambda event: commits.append(event["collection_checkpoint"])
                                   if "collection_checkpoint" in event else None, cancelled=lambda: bool(commits))
        context._case_receipt_key = base64.b64encode(os.urandom(32)).decode()
        context._case_native_session = "isolated-expert-test"
        with self.assertRaises(ExpertCollectionCancelled):
            self.collect(expert, plant, context=context)
        context.cancelled = lambda: False
        with patch.dict(os.environ, {"CONTROL_COLLECTION_RESUME": json.dumps(commits[0])}):
            resumed, _, report = self.collect(expert, plant, context=context)
        self.assertEqual(report["restored_trajectories"], 1)
        self.assertEqual(report["expert"]["source_invocations_this_adapter"], 3 * (report["valid_rows"] - 4))
        with np.load(baseline / "expert.npz") as before, np.load(resumed / "expert.npz") as after:
            for key in before.files:
                np.testing.assert_array_equal(after[key], before[key])

    def test_registration_is_static_and_source_hashes_are_checked_before_execution(self):
        expert, _, material, _ = self.register(EXPERT + "\nraise RuntimeError('must not run during registration')\n")
        root, meta = self.ws.resolve(expert)
        self.assertFalse(read_json(root / "report.json")["source_executed"])
        source_root, _ = self.ws.resolve(material)
        (source_root / "source/expert.py").write_text(EXPERT + "\n# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "changed"):
            self.ws.resolve(expert)

    def test_status_nonfinite_and_out_of_bounds_stop_at_valid_prefix_without_clamping(self):
        cases = [("failed_status", "return (np.array([0.],dtype=np.float64),memory,'solver_failed')"),
                 ("nonfinite", "return (np.array([np.nan],dtype=np.float64),memory,'ok')"),
                 ("bounds", "return (np.array([2.],dtype=np.float64),memory,'ok')")]
        for name, failure in cases:
            source = EXPERT.replace("    error=float(r[0]-obs[0])", "    if memory['calls']>=2:" + failure + "\n    error=float(r[0]-obs[0])")
            expert, plant, _, _ = self.register(source)
            with self.subTest(name=name), self.assertRaises(ExpertCollectionStopped): self.collect(expert, plant)
            failed = sorted((self.ws.path / "failed").glob("external_expert_dataset-*"), key=lambda path:path.stat().st_mtime_ns)[-1]
            report = read_json(failed / "report.json")
            self.assertEqual(report["valid_rows"], 2); self.assertEqual(report["failure"]["trajectory_id"], 0)
            self.assertEqual(report["failure"]["step_index"], 2); self.assertFalse(report["later_trajectories_started"])
            with np.load(failed / "partial-expert.npz", allow_pickle=False) as prefix:
                self.assertEqual(len(prefix["u"]), 2); self.assertTrue(np.isfinite(prefix["u"]).all())
            self.assertFalse(any(row["kind"] == "external_expert_dataset" for row in self.ws.manifest["artifacts"].values()))

    def test_mutable_hidden_state_and_memory_nonfinite_are_rejected(self):
        hidden = "import numpy as np\nH=[0]\ndef expert(o,r,d,p,m,c):\n H[0]+=1\n return (np.array([H[0]/10],dtype=np.float64),m,'ok')\n"
        expert, plant, _, _ = self.register(hidden)
        with self.assertRaisesRegex(ExpertCollectionStopped, "hidden state"):
            self.collect(expert, plant)
        invalid = "import numpy as np\ndef expert(o,r,d,p,m,c):\n return (np.zeros(1,dtype=np.float64),{'bad':float('nan')},'ok')\n"
        expert, plant, _, _ = self.register(invalid)
        with self.assertRaisesRegex(ExpertCollectionStopped, "finite JSON"):
            self.collect(expert, plant)

    def test_availability_false_hides_arguments_and_simulation_only_observation_needs_explicit_feature_selection(self):
        source = "import numpy as np\ndef expert(o,r,d,p,m,c):\n assert len(r)==len(d)==len(p)==0\n return (np.array([.1*o[0]],dtype=np.float64),{'integral':0.,'calls':m['calls']+1},'ok')\n"
        spec = declaration(); spec.update(reference_available=False, disturbance_available=False, previous_action_available=False)
        spec["observations"][0]["availability"] = "simulation_only"
        expert, plant, _, _ = self.register(source, spec)
        with self.assertRaisesRegex(ContractError, "student_features"):
            self.collect(expert, plant)
        config = configuration(); config["student_features"] = ["observation::position_measurement"]
        root, meta, report = self.collect(expert, plant, config)
        self.assertEqual(meta["interface"]["input_dim"], 1)
        self.assertEqual(report["deployment_scope"], "simulation_only")
        self.assertEqual(report["source"]["student_feature_scope"], "simulation_only")

    def test_mimo_action_permutation_previous_and_rate_limits_use_plant_order(self):
        spec = declaration(2)
        spec["action_mapping"] = [{"action_index": 0, "plant_control_index": 1, "plant_control_name": "force1", "unit": "N"},
                                  {"action_index": 1, "plant_control_index": 0, "plant_control_name": "force0", "unit": "N"}]
        spec["input_rate_bounds"] = [[-.16,.1],[-.1,.21]]
        source = "import numpy as np\ndef expert(o,r,d,p,m,c):\n return (np.array([.5*p[1]+.1,.5*p[0]-.05],dtype=np.float64),{'integral':0.,'calls':m['calls']+1},'ok')\n"
        expert, plant, _, _ = self.register(source, spec)
        config = configuration()
        for row in config["scenarios"]: row["initial_previous_action"] = [.2,-.2]
        root, meta, _ = self.collect(expert, plant, config)
        with np.load(root / "expert.npz", allow_pickle=False) as arrays:
            np.testing.assert_allclose(arrays["previous_u"][0], [.2,-.2], atol=0, rtol=0)
            np.testing.assert_allclose(arrays["u"][0], [.05,0.], atol=1e-14, rtol=0)
            np.testing.assert_allclose(arrays["u"][1], [-.025,.1], atol=1e-14, rtol=0)
            np.testing.assert_array_equal(arrays["y"], arrays["u"])
        self.assertEqual([row["signal"] for row in meta["interface"]["targets"]], ["expert_action::force0","expert_action::force1"])

    def test_period_units_mapping_and_duplicate_conditions_fail_before_source_calls(self):
        for update in ({"control_period": .15}, {"time_unit": "ms"}, {"control_period": True}, {"backend": []}):
            with self.assertRaises(ContractError): validate_expert_spec(declaration() | update)
        for field in ("unit", "plant_name", "index"):
            spec = declaration(); spec["observations"][0][field] = "wrong" if field != "index" else 2
            with self.assertRaises(ContractError): validate_expert_spec(spec)
        spec=declaration();spec["action_mapping"][0]["unit"]="m"
        with self.assertRaises(ContractError):validate_expert_spec(spec)
        spec=declaration();config=configuration()
        config["scenarios"][1] = {**config["scenarios"][0],"name":"different label","steps":9,"split":"test"}
        with self.assertRaisesRegex(ContractError,"Duplicate physical"):
            build_expert_collection_plan(physical(),spec,config)
        config=configuration();config["scenarios"][0]["comment"]="irrelevant"
        with self.assertRaisesRegex(ContractError,"Unknown scenario"):
            build_expert_collection_plan(physical(),spec,config)
        config=configuration();config["scenarios"][0]["steps"]=10**9
        with self.assertRaises(ContractError):build_expert_collection_plan(physical(),spec,config)
        expert,plant,_,_=self.register()
        for selected in (["memory::integral"],["expert_action::force0"],["raw_state"],[["observation::position_measurement"]]):
            config=configuration();config["student_features"]=selected
            with self.assertRaises(ContractError):self.collect(expert,plant,config)

    def test_learned_plant_loads_directly_without_an_internal_controller_or_original_source(self):
        import torch
        from control_harness.contracts import ModelInterface
        from control_harness.supervised_export import build_supervised_network, export_supervised
        from control_harness.learned_dynamics import export_learned_model
        expert, _, _, spec = self.register()
        physical_interface = spec["interface"]
        signals = []
        for group, role in (("states", "state"), ("controls", "control"), ("disturbances", "disturbance")):
            for variable in physical_interface[group]:
                name = group+"::"+variable["name"]
                signals.append({"name":name,"role":role,"unit":variable["unit"],"observable":True,"delay_steps":0,
                    "column":{"index":len(signals),"header":name,"worksheet":None}})
        data = {"schema_version":1,"interface_kind":"tabular_supervised","task":"dynamic_prediction","sample_time":.1,"signals":signals,
            "features":[{"signal":signal["name"],"window":1,"lag":0} for signal in signals],
            "targets":[{"signal":"states::position","offset":1}],"state_mapping":{"position":"states::position"},
            "time_column":{"index":3,"header":"time","worksheet":None}}
        network_config={"schema_version":1,"architecture":"mlp","input_dim":3,"output_dim":1,"hidden_dims":[],"activation":"tanh","dtype":"float64"}
        network=build_supervised_network(network_config)
        with torch.no_grad():network[0].weight.copy_(torch.tensor([[.85,.2,.05]],dtype=torch.float64));network[0].bias.zero_()
        scaler={"method":"none","fit_split":"train","X":{"location":[0.,0.,0.],"scale":[1.,1.,1.]},"y":{"location":[0.],"scale":[1.]}}
        source=self.root/"nn-source"
        export_supervised(source/"bundle",network,network_config,data,scaler,{"kind":"manual_tensor_test_fixture_not_training_claim"})
        with self.ws.stage("learned_model",ModelInterface.parse(physical_interface),[],{"kind":"manual_tensor_fixture"}) as stage:
            export_learned_model(stage.root,source,physical_interface,[[1.]])
            plant=stage.publish({"status":"verified","scope":"manually assigned numeric unit fixture"},activate=False)
        import shutil
        shutil.rmtree(source)
        root,metadata,report=self.collect(expert,plant)
        self.assertEqual(report["plant"]["backend"],"registered_torch")
        self.assertEqual(report["plant"]["execution"]["source_representation"],"learned_discrete_mlp")
        self.assertEqual({row["id"] for row in metadata["parents"]},{expert,plant})
        with np.load(root/"expert.npz",allow_pickle=False) as arrays:
            for row in range(len(arrays["u"])):
                state=arrays["x"][row,0];action=arrays["u"][row,0];disturbance=arrays["d"][row,0]
                for _ in range(2):state=.85*state+.2*action+.05*disturbance
                np.testing.assert_allclose(arrays["next_x"][row],[state],rtol=0,atol=1e-12)

    def test_expert_and_plant_interface_mismatch_is_refused_before_source_loading(self):
        expert,plant,_,_=self.register()
        root,_=self.ws.resolve(plant);definition=read_json(root/"plant.json")
        definition["adapter_spec"]["interface"]["sample_time"] = .2
        mismatched=register_plant(self.ws,definition["material"],{"adapter":definition["adapter_spec"]},OperationContext())
        with patch("control_harness.external_expert.load_external_expert",side_effect=AssertionError("must reject before expert execution")),self.assertRaisesRegex(ContractError,"Interface mismatch"):
            self.collect(expert,mismatched)

    def test_plant_subtick_violation_and_cancellation_archive_prefix_without_training_artifact(self):
        expert, plant, _, _ = self.register(plant_gain=50.)
        with self.assertRaisesRegex(ExpertCollectionStopped,"subtick"):
            self.collect(expert,plant)
        expert,plant,_,_=self.register()
        events=[]
        def emit(event):
            events.append(event)
            if event.get("stage")=="external_expert_collection":raise Cancelled("fixed between-trajectory cancellation")
        with self.assertRaises(ExpertCollectionCancelled):self.collect(expert,plant,context=OperationContext(emit=emit))
        failed=sorted((self.ws.path/"failed").glob("external_expert_dataset-*"),key=lambda p:p.stat().st_mtime_ns)[-1]
        report=read_json(failed/"report.json")
        self.assertEqual(report["valid_rows"],4);self.assertEqual(len(report["trajectories"]),1)
        self.assertEqual(report["failure"]["phase"],"after_completed_trajectory")
        self.assertFalse(any(row["kind"]=="external_expert_dataset" for row in self.ws.manifest["artifacts"].values()))

    def test_mixed_python_matlab_backend_is_refused_without_starting_a_bridge(self):
        expert,_,_,spec=self.register()
        path=self.root/"plant.m";path.write_text("function xn=plant(x,u,d,dt,p)\nxn=x;\nend\n",encoding="utf-8")
        material=register_material(self.ws,str(path),{},OperationContext())
        plant=register_plant(self.ws,material,{"adapter":{"schema_version":1,"backend":"matlab_function","source_file":"source/plant.m",
            "interface":spec["interface"],"parameters":{},"source_semantics":"transition","entrypoint":"plant","C":[[1.]]}},OperationContext())
        with self.assertRaisesRegex(ContractError,"Mixed Python/MATLAB"):
            self.collect(expert,plant)


if __name__ == "__main__": unittest.main()
