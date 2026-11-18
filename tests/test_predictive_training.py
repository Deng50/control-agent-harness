"""Real next-state NN training from one trajectory-split sampled artifact."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError, ModelInterface
from control_harness.learned_dynamics import load_learned_model
from control_harness.inspection import prepare_handoff
from control_harness.material_registration import register_material
from control_harness.modeling import register_model
from control_harness.plants import register_plant
from control_harness.predictive import train_predictive_model, _data_interface, _prepare, _read_sampled
from control_harness.runtime import Cancelled, OperationContext


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class PredictiveTrainingTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "test-output" / "predictive-training-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ws = Workspace(self.root / "case")
        self.C = [[1.3, -.2]]
        self.network = {"hidden_dims": [5], "activation": "tanh", "epochs": 8, "learning_rate": .01,
            "batch_size": 13, "patience": 30, "seed": 20261121, "weight_decay": .001, "min_delta": 0, "grad_clip": 5}
        self.config = {"network": self.network, "C": self.C, "scaler": "standard"}

    def dataset(self, nd=1, mutate=None, split=None, units="1"):
        variable = lambda name, bound: {"name": name, "unit": units, "lower": -bound, "upper": bound}
        interface = ModelInterface.parse({"schema_version": 1, "name": "recorded linear two-state fixture", "sample_time": .2,
            "time_semantics": "discrete_state_transition", "states": [variable("position", 10), variable("velocity", 10)],
            "controls": [variable("position", 1)], "disturbances": [variable("wind", .2)] if nd else [],
            "outputs": [variable("sensor", 20)]})
        A, B, E, C = np.array([[.7,.1],[-.1,.65]]), np.array([.2,.1]), np.array([.03,.05]), np.asarray(self.C)
        rng = np.random.default_rng(12345)
        rows = {key: [] for key in ("x", "u", "d", "next_x", "y", "next_y", "trajectory_id", "time")}
        for trajectory in range(6):
            x = np.array([-.4 + .13 * trajectory, .2 - .07 * trajectory])
            for step in range(12):
                u = rng.uniform(-.7, .7)
                d = rng.uniform(-.1, .1) if nd else 0.
                next_x = A @ x + B * u + E * d
                for key, value in (("x", x.copy()), ("u", [u]), ("d", [d] if nd else []), ("next_x", next_x),
                                   ("y", C @ x), ("next_y", C @ next_x), ("trajectory_id", trajectory), ("time", step * .2)):
                    rows[key].append(value)
                x = next_x
        arrays = {key: np.asarray(value, dtype=np.int64 if key == "trajectory_id" else np.float64) for key, value in rows.items()}
        if mutate: mutate(arrays)
        split = split or {"train": [0,1,2,3], "validation": [4], "test": [5]}
        # Keep a real immutable material/NumPy plant provenance chain. Arrays
        # above are an independent golden recurrence; registration never runs it.
        source = self.root / "fixture_plant.py"
        source.write_text("import numpy as np\ndef transition(x,u,d,dt,p):\n z=d[0] if len(d) else 0.\n return np.array([.7*x[0]+.1*x[1]+.2*u[0]+.03*z,-.1*x[0]+.65*x[1]+.1*u[0]+.05*z],dtype=np.float64)\n", encoding="utf-8")
        context=OperationContext()
        material=register_material(self.ws,str(source),{},context)
        plant=register_plant(self.ws,material,{"adapter":{"schema_version":1,"backend":"python_callable",
            "source_file":"source/fixture_plant.py","interface":interface.to_dict(),"parameters":{},
            "source_semantics":"transition","entrypoint":"transition","C":self.C}},context)
        with self.ws.stage("sampled_dataset", interface, [plant], {"fixture": "independent A/B/E recurrence"}) as stage:
            np.savez_compressed(stage.root / "transitions.npz", **arrays)
            atomic_json(stage.root / "split.json", split)
            artifact = stage.publish({"status": "verified"}, source={"kind": "plant_simulator", "backend": "python_callable", "model":plant,
                "measured": False, "description": "Independent synthetic recorded trajectory fixture"}, activate=False)
        return artifact, arrays, interface, split

    def artifact(self, source, config=None, context=None):
        result = train_predictive_model(self.ws, source, config or self.config, context)
        root, metadata = self.ws.resolve(result, allow_rejected=True)
        return result, root, metadata, read_json(root / "report.json")

    def test_derived_npz_mapping_split_train_scaler_and_cache_identity_are_preserved(self):
        source, arrays, physical, split = self.dataset()
        data, masks, original, _, meta = _read_sampled(self.ws, source)
        first, reused = _prepare(self.ws, source, data, masks, original, physical, meta, "standard")
        second, reused_again = _prepare(self.ws, source, data, masks, original, physical, meta, "standard")
        self.assertFalse(reused); self.assertTrue(reused_again); self.assertEqual(first, second)
        root, metadata = self.ws.resolve(first)
        self.assertEqual(metadata["activation"], "none")
        self.assertEqual([row["id"] for row in metadata["parents"]], [source])
        declaration = metadata["interface"]
        self.assertEqual(declaration["task"], "dynamic_prediction")
        self.assertEqual(declaration["column_semantics"], "logical_signal_index_not_tabular_source_column")
        self.assertEqual(declaration["source_mapping"]["signals"]["states::position"], {"array":"x","column":0,"target_array":"next_x"})
        self.assertEqual(declaration["state_mapping"], {"position":"states::position","velocity":"states::velocity"})
        self.assertEqual([row["signal"] for row in declaration["features"]],
                         ["states::position","states::velocity","controls::position","disturbances::wind"])
        self.assertTrue(all(row["offset"] == 1 for row in declaration["targets"]))
        with np.load(root / "supervised.npz", allow_pickle=False) as loaded:
            np.testing.assert_array_equal(loaded["X"], np.concatenate([arrays[key] for key in ("x","u","d")], axis=1))
            np.testing.assert_array_equal(loaded["y"], arrays["next_x"])
            for index, part in enumerate(("train","validation","test")):
                np.testing.assert_array_equal(loaded["split_id"] == index, np.isin(arrays["trajectory_id"], split[part]))
        scaler = read_json(root / "scaler.json")
        X = np.concatenate([arrays[key] for key in ("x","u","d")], axis=1)
        np.testing.assert_allclose(scaler["X"]["location"], X[masks["train"]].mean(0), rtol=0, atol=1e-15)
        third, _ = _prepare(self.ws, source, data, masks, original, physical, meta, "minmax")
        self.assertNotEqual(first, third)

    def test_actual_training_packages_a_physical_next_state_model_and_independent_metrics(self):
        source, arrays, physical, split = self.dataset()
        artifact, root, metadata, report = self.artifact(source)
        self.assertEqual(metadata["kind"], "learned_model")
        self.assertEqual(metadata["interface"], read_json(root / "learned.json")["interface"])
        self.assertEqual(metadata["interface"]["sample_time"], .2)
        self.assertFalse(report["pinn"]); self.assertFalse(report["physics_constraints_in_training"])
        self.assertFalse(report["nominal_or_reference_model_required"])
        self.assertEqual(report["target_semantics"], "next_x_in_state_units_no_dt_multiplication")
        self.assertEqual(report["C"], self.C)
        self.assertEqual([row["id"] for row in metadata["parents"]], [source,report["supervised_dataset"],report["supervised_model"]])
        model = load_learned_model(root)
        model.interface.compatible(physical)
        test = arrays["trajectory_id"] == split["test"][0]
        train = np.isin(arrays["trajectory_id"], split["train"])
        state_scale = arrays["next_x"][train].std(0, ddof=0); state_scale[state_scale == 0] = 1
        output_scale = arrays["next_y"][train].std(0, ddof=0); output_scale[output_scale == 0] = 1
        with np.load(root / "one_step_predictions.npz") as saved:
            state_rmse = np.sqrt(np.mean((saved["predicted_state"] - arrays["next_x"][test]) ** 2, axis=0))
            output_rmse = np.sqrt(np.mean((saved["predicted_output"] - arrays["next_y"][test]) ** 2, axis=0))
            np.testing.assert_allclose(saved["predicted_output"], saved["predicted_state"] @ np.asarray(self.C).T, rtol=0, atol=1e-15)
            nn_root,_ = self.ws.resolve(report["supervised_model"])
            with np.load(nn_root / "test_predictions.npz") as supervised:
                # The S05 inverse-scaled prediction is already next_x.  The
                # physical wrapper must not multiply by dt or integrate again.
                np.testing.assert_allclose(saved["predicted_state"],supervised["prediction"],rtol=0,atol=1e-14)
            np.testing.assert_allclose(saved["prediction_time"],arrays["time"][test]+physical.sample_time,rtol=0,atol=1e-15)
        np.testing.assert_allclose(report["one_step"]["test"]["state"]["nrmse"], state_rmse/state_scale, rtol=0, atol=1e-14)
        np.testing.assert_allclose(report["one_step"]["test"]["output"]["nrmse"], output_rmse/output_scale, rtol=0, atol=1e-14)
        x = arrays["x"][test][0].copy(); recursive = []
        for u,d in zip(arrays["u"][test], arrays["d"][test]):
            x = model.step(x,u,d); recursive.append(x)
        with np.load(root / "rollout_predictions.npz") as saved:
            np.testing.assert_allclose(saved["predicted_state"], recursive, rtol=0, atol=1e-15)
            np.testing.assert_array_equal(saved["true_state"], arrays["next_x"][test])
        self.assertTrue(report["rollout"]["all_trajectories_completed"])
        self.assertTrue(all(not row["teacher_forcing_after_initial_state"] for row in report["rollout"]["trajectories"]))
        self.assertFalse(report["test_used_for_training_or_selection"])
        self.assertNotIn("supervised_model", self.ws.manifest["active"])
        self.assertNotIn("supervised_dataset", self.ws.manifest["active"])
        self.assertTrue((root / "bundle" / "simulate.py").is_file())
        # An unrelated active reference must not silently become this result's
        # evaluation source. NumPy plant provenance is retained but not a Torch
        # runtime adapter for the legacy controller evaluation path.
        unrelated=register_model(self.ws,{"interface":physical.to_dict(),"representation":"discrete_polynomial",
            "A":[[.7,.1],[-.1,.65]],"B":[[.2],[.1]],"E":[[.03],[.05]],"C":self.C},
            {"kind":"reference_simulator","description":"unrelated active handoff negative control"})
        handoff=prepare_handoff(self.ws,artifact)
        self.assertEqual(handoff["handoff"]["agent"],"controller")
        self.assertEqual(handoff["handoff"]["operation"],"synthesize_mpc")
        self.assertEqual(handoff["handoff"]["inputs"]["model"],artifact)
        self.assertEqual(handoff["handoff"]["missing_inputs"],[])
        self.assertEqual(handoff["optional_handoffs"][0]["missing_inputs"],["evaluation_model"])
        self.assertNotEqual(handoff["optional_handoffs"][0]["inputs"]["evaluation_model"],unrelated)
        lineage_kinds={self.ws.manifest["artifacts"][row["id"]]["kind"] for row in handoff["source_versions"]}
        self.assertTrue({"learned_model","supervised_model","supervised_dataset","sampled_dataset","plant_model","material"}<=lineage_kinds)
        source_handoff=prepare_handoff(self.ws,source)
        self.assertEqual(source_handoff["handoff"]["operation"],"train_predictive_model")
        self.assertEqual(source_handoff["handoff"]["config_suggestions"]["C"],self.C)

    def assert_nested_equal(self, left, right):
        if isinstance(right, torch.Tensor): self.assertTrue(torch.equal(left, right))
        elif isinstance(right, dict):
            self.assertEqual(set(left), set(right))
            for key in right: self.assert_nested_equal(left[key], right[key])
        elif isinstance(right, (tuple,list)):
            self.assertEqual(len(left), len(right))
            for a,b in zip(left,right): self.assert_nested_equal(a,b)
        else: self.assertEqual(left,right)

    def test_interrupted_nn_reuses_the_same_prepared_artifact_and_resumes_exactly(self):
        source, *_ = self.dataset()
        _, _, _, continuous = self.artifact(source)
        nn_root, _ = self.ws.resolve(continuous["supervised_model"])
        expected = torch.load(nn_root / "latest.pt", weights_only=True)
        events = []
        def emit(event):
            events.append(event)
            if event.get("stage") == "supervised_training" and event.get("epoch") == 3: raise Cancelled("fixed committed-epoch interruption")
        with self.assertRaises(Cancelled): self.artifact(source, context=OperationContext(emit=emit))
        checkpoint = events[-1]["checkpoint"]
        self.assertTrue(Path(checkpoint).is_file())
        _, _, _, resumed = self.artifact(source, {**self.config,"resume_checkpoint":checkpoint})
        self.assertEqual(resumed["supervised_dataset"], continuous["supervised_dataset"])
        self.assertTrue(resumed["supervised_dataset_reused"])
        self.assertEqual(resumed["network"]["resumed_from_epoch"], 3)
        resumed_root, _ = self.ws.resolve(resumed["supervised_model"])
        self.assert_nested_equal(torch.load(resumed_root / "latest.pt",weights_only=True), expected)

    def test_published_nn_checkpoint_progress_survives_a_later_outer_failure(self):
        source, *_ = self.dataset()
        published = []
        def emit(event):
            if event.get("checkpoint_status") == "published":
                published.append(event); raise RuntimeError("intentional post-NN calibration interruption")
        with self.assertRaisesRegex(RuntimeError, "post-NN"):
            self.artifact(source, context=OperationContext(emit=emit))
        path = Path(published[0]["checkpoint"])
        self.assertEqual(path.parent.parent.name, "artifacts")
        self.assertTrue(path.is_file())
        _, _, _, resumed = self.artifact(source, {**self.config,"resume_checkpoint":str(path)})
        self.assertEqual(resumed["network"]["resumed_from_epoch"], self.network["epochs"])

    def test_C_is_required_and_must_match_recorded_current_and_next_outputs_before_training(self):
        source, *_ = self.dataset()
        for C in (None, [[1.,0.]], [[True,0.]], [[1.,0.],[0.,1.]], [[float("nan"),0.]]):
            with self.subTest(C=C), self.assertRaises(ContractError):
                self.artifact(source, {**self.config,"C":C})
        self.assertFalse(any(row["kind"] == "supervised_model" for row in self.ws.manifest["artifacts"].values()))

    def test_bad_source_split_time_continuity_or_missing_output_is_not_reinterpreted(self):
        cases = [lambda rows: rows["time"].__setitem__(1,.3), lambda rows: rows["x"].__setitem__((1,0),2.),
                 lambda rows: rows.pop("next_y"), lambda rows: rows["u"].__setitem__((0,0),2.)]
        for mutate in cases:
            with self.subTest(mutate=mutate):
                source,*_ = self.dataset(mutate=mutate)
                with self.assertRaises(ContractError): self.artifact(source)
        source,*_ = self.dataset(split={"train":[0,1,2,3],"validation":[3,4],"test":[5]})
        with self.assertRaisesRegex(ContractError,"disjoint"):
            self.artifact(source)

    def test_zero_disturbance_width_and_nonidentity_output_mapping_are_preserved(self):
        source, arrays, physical, _ = self.dataset(nd=0)
        _, root, _, report = self.artifact(source, {**self.config,"network":{**self.network,"epochs":2},"scaler":"none"})
        model=load_learned_model(root)
        self.assertEqual(model.interface.nd,0)
        actual=model.step(arrays["x"][0],arrays["u"][0])
        np.testing.assert_allclose(model.output(torch.from_numpy(actual)).numpy(),np.asarray(self.C)@actual,rtol=0,atol=1e-15)
        self.assertEqual(report["C"],self.C)

    def test_predeclared_prediction_gate_rejects_without_replacing_a_verified_model(self):
        source,*_ = self.dataset()
        first,_,_,_ = self.artifact(source)
        rejected,root,meta,report = self.artifact(source,{**self.config,"max_rollout_state_nrmse":0.})
        self.assertEqual(meta["validation"]["status"],"rejected")
        self.assertIn("max_rollout_state_nrmse",report["acceptance_failures"])
        self.assertEqual(report["structural_finite_bounds_acceptance"],"passed")
        self.assertEqual(self.ws.manifest["active"]["learned_model"],first)
        self.assertTrue((root/"report.json").is_file())
        _,_,other_meta,other_report = self.artifact(source,{**self.config,"scaler":"minmax","max_rollout_state_nrmse":0.})
        self.assertEqual(other_meta["validation"]["status"],"rejected")
        self.assertNotEqual(other_report["supervised_dataset"],report["supervised_dataset"])
        self.assertEqual(self.ws.manifest["active"]["learned_model"],first)

    def test_invalid_network_dimensions_limits_or_nested_authority_are_refused_before_derivation(self):
        source,*_ = self.dataset()
        patches = [{"network":{**self.network,"input_dim":3}}, {"network":{**self.network,"output_dim":1}},
                   {"network":{**self.network,"resume_checkpoint":"wrong-layer"}}, {"scaler":"fit_all"},
                   {"max_rollout_state_nrmse":[.1]}, {"max_test_output_nrmse":True}, {"physics_weight":.1}]
        for patch in patches:
            with self.subTest(patch=patch),self.assertRaises(ContractError):
                self.artifact(source,{**self.config,**patch})
        self.assertFalse(any(row["kind"]=="supervised_dataset" for row in self.ws.manifest["artifacts"].values()))

    def test_resume_rejects_a_changed_source_or_feature_units_instead_of_rebuilding_checkpoint_authority(self):
        source,*_ = self.dataset()
        _,_,_,report = self.artifact(source)
        nn_root,_ = self.ws.resolve(report["supervised_model"])
        other,*_ = self.dataset(units="different_unit")
        with self.assertRaisesRegex(ContractError,"configuration differs"):
            self.artifact(other,{**self.config,"resume_checkpoint":str(nn_root/"latest.pt")})

    def test_cached_prepared_data_are_fully_revalidated_before_resume(self):
        source,*_ = self.dataset()
        arrays,masks,split,physical,meta = _read_sampled(self.ws,source)
        prepared,_ = _prepare(self.ws,source,arrays,masks,split,physical,meta,"standard")
        root,_ = self.ws.resolve(prepared)
        (root/"scaler.json").write_text("{}",encoding="utf-8")
        with self.assertRaisesRegex(ContractError,"missing or changed"):
            self.artifact(source)


if __name__ == "__main__": unittest.main()
