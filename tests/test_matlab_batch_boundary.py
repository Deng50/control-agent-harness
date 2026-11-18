"""Independent synthetic batch-protocol tests; NEVER real MATLAB execution.

The JSON files below emulate an authenticated exchange for boundary testing.
Their synthetic provenance is explicit; passing these tests cannot certify MCP,
Simulink log timing, physical fidelity or MATLAB numerical execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile

import numpy as np

from control_harness.contracts import ContractError, canonical_hash
from control_harness.matlab_adapter import (MatlabBatchAdapter, MatlabBatchError,
    load_matlab_batch_adapter, validate_matlab_spec, validate_simulink_source)
from control_harness.matlab_source import validate_matlab_sources
from control_harness.sampling import SamplingStopped, build_sampling_plan, collect


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "matlab-batch-boundary"))


def interface(nd=1):
    def variable(name): return {"name": name, "unit": "1", "lower": -10., "upper": 10.}
    return {"schema_version": 1, "name": "SYNTHETIC protocol fixture; no MATLAB run",
        "sample_time": .2, "time_semantics": "discrete_state_transition",
        "states": [variable("position"), variable("velocity")], "controls": [variable("force")],
        "disturbances": [variable("disturbance")] if nd else [], "outputs": [variable("readout")]}


class MatlabBatchBoundaryTests(unittest.TestCase):
    def setUp(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.root = OUTPUT / ("synthetic-exchange-" + uuid.uuid4().hex)
        self.root.mkdir()  # Inherits the authorized workspace ACL on Windows.
        self.addCleanup(lambda: shutil.rmtree(self.root))

    def fixture(self, nd=1, trajectories=3, steps=4):
        raw = interface(nd)
        self.spec = validate_matlab_spec({"schema_version": 1, "backend": "matlab_function",
            "source_file": "source/protocol_fixture.m", "interface": raw, "parameters": {},
            "source_semantics": "transition", "entrypoint": "protocol_fixture", "C": [[1., .2]]})
        self.plan = build_sampling_plan(raw, {"seed": 17, "steps": steps,
            "trajectories": [{"trajectory_id": 7 + 4*k, "x0": [.1 + .1*k, -.05*k]} for k in range(trajectories)],
            "u0": [0.], "d0": [0.] if nd else [],
            "u_profile": {"kind": "step", "times": [0, .4] if steps >= 3 else [0],
                "values": [[.1], [-.15]] if steps >= 3 else [[.1]]},
            "d_profile": {"kind": "step", "times": [0], "values": [[.03]]} if nd else None})
        arrays = {key: [] for key in ("x", "u", "d", "next_x", "y", "next_y", "time", "trajectory_id")}
        summaries = []
        # Independent synthetic scalar recurrence; the .m text is never run.
        for trajectory in self.plan["trajectories"]:
            x, v = trajectory["x0"]
            for k, (u, d) in enumerate(zip(trajectory["u"], trajectory["d"])):
                disturbance = d[0] if nd else 0.
                nx, nv = .8*x + .1*v + .2*u[0] + .05*disturbance, .9*v - .1*u[0] + .02*disturbance
                for key, value in {"x": [x, v], "u": u, "d": d, "next_x": [nx, nv],
                        "y": [x + .2*v], "next_y": [nx + .2*nv],
                        "time": trajectory["t"][k], "trajectory_id": trajectory["trajectory_id"]}.items():
                    arrays[key].append(value)
                x, v = nx, nv
            summaries.append({"trajectory_id": trajectory["trajectory_id"], "completed_steps": steps,
                "status": "completed", "stop_detection": "SYNTHETIC_NO_MATLAB",
                "error_identifier": "", "error_message": ""})
        self.binding = {"request_sha256": "c"*64, "nonce": "11111111-2222-4333-8444-555555555555",
            "model_version": {"id": "synthetic_model", "metadata_hash": "a"*64},
            "material_version": {"id": "synthetic_material", "metadata_hash": "b"*64}}
        source = self.root / "input" / self.spec["source_file"]
        source.parent.mkdir(parents=True)
        source.write_text("function next = protocol_fixture(x,u,d,dt,p)\nnext = x;\nend\n", encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.manifest = {"schema_version": 1, "kind": "control-matlab-batch-v1", **self.binding,
            "spec_hash": canonical_hash(self.spec), "spec": self.spec, "plan": self.plan,
            "source_root": str(self.root / "input"), "source_file": str(source), "source_sha256": digest,
            "source_inventory": {self.spec["source_file"]: digest}, "result_path": str(self.root / "result.json")}
        self.result = {"schema_version": 1, "kind": "control-matlab-batch-result-v1", **self.binding,
            "spec_hash": canonical_hash(self.spec), "plan_hash": self.plan["plan_hash"],
            "status": "completed", "arrays": arrays, "trajectories": summaries, "failure": [],
            "stop_detection": "SYNTHETIC_NO_MATLAB", "matlab_release": "NOT_RUN_SYNTHETIC",
            "matlab_pid": 0, "session_id": "synthetic-protocol-fixture", "hardware_execution": False}
        return self.result

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, result=None, manifest=None):
        digest = self.write_json("result.json", self.result if result is None else result)
        self.write_json("manifest.json", self.manifest if manifest is None else manifest)
        return load_matlab_batch_adapter(self.spec, self.plan, result_path=self.root / "result.json",
            result_sha256=digest, **self.binding)

    def partial(self, completed_trajectories=1, valid_steps=2):
        result = copy.deepcopy(self.result)
        size = completed_trajectories * self.plan["steps"] + valid_steps
        for key in result["arrays"]:
            result["arrays"][key] = result["arrays"][key][:size]
        result["trajectories"] = result["trajectories"][:completed_trajectories + 1]
        last = result["trajectories"][-1]
        last.update(completed_steps=valid_steps, status="partial" if valid_steps else "failed",
            error_identifier="Synthetic:Stop", error_message="synthetic deliberate prefix stop")
        result.update(status="partial" if size else "failed", failure={"trajectory_id": last["trajectory_id"],
            "step_index": valid_steps, "time": valid_steps*self.plan["dt"], "reason": "synthetic deliberate prefix stop"})
        return result

    def simulink_spec(self):
        return {"schema_version": 1, "backend": "simulink", "source_file": "source/synthetic.slx",
            "source_semantics": "transition", "interface": interface(0), "parameters": {}, "simulink": {
                "controls": [{"port": 1, "sid": "1"}], "disturbances": [], "outputs": [{"port": 1, "sid": "4"}],
                "states": [{"sid": "2", "initial_condition_variable": "initial_position"},
                           {"sid": "3", "initial_condition_variable": "initial_velocity"}],
                "sample_time_variable": "sample_dt", "solver": "FixedStepDiscrete"}}

    def slx(self, *, feedthrough=False, hidden_state=False, callback=False):
        blocks = ''.join(f'<Block SID="{sid}" BlockType="{kind}"/>' for sid,kind in
            [("1","Inport"),("2","UnitDelay"),("3","UnitDelay"),("4","Outport"),("5","Sum")])
        if hidden_state: blocks += '<Block SID="6" BlockType="UnitDelay"/>'
        source = "1" if feedthrough else "2"
        lines = f'<Line><P Name="Src">{source}#out:1</P><P Name="Dst">4#in:1</P></Line>'
        code = '<P Name="InitFcn">disp(1);</P>' if callback else ''
        path = self.root / ("synthetic-" + uuid.uuid4().hex + ".slx")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("simulink/blockdiagram.xml", '<Model>'+code+'<System>'+blocks+lines+'</System></Model>')
        return path

    def test_synthetic_completed_exchange_replays_declared_rows_and_has_safe_close(self):
        self.fixture()
        adapter = self.load()
        arrays, report = collect(adapter, self.plan)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["valid_rows"], 12)
        for key, value in arrays.items():
            np.testing.assert_allclose(value, self.result["arrays"][key], rtol=0, atol=1e-15)
        self.assertEqual(report["adapter"]["matlab_batch"]["matlab_release"], "NOT_RUN_SYNTHETIC")
        adapter.close(); adapter.close()
        with self.assertRaises(ContractError): adapter.reset([.1, 0.])

    def test_native_failed_or_partial_status_never_replays_full_rows_as_success(self):
        self.fixture()
        for status in ("failed", "partial", "running", "verified", None):
            result = copy.deepcopy(self.result); result["status"] = status
            with self.subTest(status=status), self.assertRaises(ContractError): self.load(result)

    def test_dt_and_initial_state_variables_cannot_alias(self):
        spec = self.simulink_spec()
        validate_matlab_spec(spec)
        spec["simulink"]["states"][0]["initial_condition_variable"] = "sample_dt"
        with self.assertRaisesRegex(ContractError, "alias|conflict"): validate_matlab_spec(spec)

    def test_nd_zero_matlab_empty_json_d_is_restored_as_n_by_zero(self):
        self.fixture(nd=0)
        # MATLAB serializes an N-by-zero array as [] rather than N empty rows.
        self.result["arrays"]["d"] = []
        adapter = self.load()
        self.assertEqual(adapter.arrays["d"].shape, (12, 0))
        arrays, report = collect(adapter, self.plan)
        self.assertEqual(arrays["d"].shape, (12, 0))
        self.assertEqual(report["valid_rows"], 12)

    def test_one_step_one_trajectory_matlab_scalar_channel_encodings(self):
        self.fixture(nd=0, trajectories=1, steps=1)
        self.result["arrays"]["d"] = []
        self.result["arrays"]["trajectory_id"] = 7
        self.result["arrays"]["time"] = 0.
        self.result["arrays"]["x"] = self.result["arrays"]["x"][0]
        self.result["arrays"]["next_x"] = self.result["arrays"]["next_x"][0]
        for key in ("u", "y", "next_y"):
            self.result["arrays"][key] = self.result["arrays"][key][0][0]
        arrays, report = collect(self.load(), self.plan)
        self.assertEqual(arrays["x"].shape, (1, 2))
        self.assertEqual(arrays["u"].shape, (1, 1))
        self.assertEqual(report["status"], "completed")

    def test_wrong_array_widths_nonfinite_ids_and_time_are_rejected(self):
        self.fixture()
        def bad_width(result): result["arrays"]["x"][0].pop()
        def bad_count(result): result["arrays"]["trajectory_id"].pop()
        def bad_finite(result): result["arrays"]["next_x"][2][0] = None
        def bad_id_type(result): result["arrays"]["trajectory_id"][0] = 7.5
        def bad_time(result): result["arrays"]["time"][1] = None
        for mutate in (bad_width, bad_count, bad_finite, bad_id_type, bad_time):
            result = copy.deepcopy(self.result); mutate(result)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ContractError): self.load(result)

    def test_wrong_row_order_controls_disturbances_times_and_state_continuity_are_rejected(self):
        self.fixture()
        def wrong_order(result):
            for key in result["arrays"]:
                result["arrays"][key][:4],result["arrays"][key][4:8] = result["arrays"][key][4:8],result["arrays"][key][:4]
        def wrong_u(result): result["arrays"]["u"][1][0] += .02
        def wrong_d(result): result["arrays"]["d"][1][0] += .02
        def wrong_time(result): result["arrays"]["time"][2] += .01
        def discontinuous_x(result): result["arrays"]["x"][1][0] += .01
        def discontinuous_y(result): result["arrays"]["y"][1][0] += .01
        def wrong_initial(result): result["arrays"]["x"][0][0] += .01
        def wrong_output(result): result["arrays"]["next_y"][-1][0] += .01
        for mutate in (wrong_order, wrong_u, wrong_d, wrong_time, discontinuous_x, discontinuous_y, wrong_initial, wrong_output):
            result = copy.deepcopy(self.result); mutate(result)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ContractError): self.load(result)

    def test_nonmatlab_extra_array_axes_strings_and_boolean_numbers_are_rejected(self):
        self.fixture()
        def extra_axis(result): result["arrays"]["x"] = [result["arrays"]["x"]]
        # First trajectory's second state is numerically zero, so these would
        # pass numeric continuity after coercion. Their JSON types are invalid.
        def string_number(result): result["arrays"]["x"][0][1] = "0.0"
        def boolean_number(result): result["arrays"]["x"][0][1] = False
        for mutate in (extra_axis,string_number,boolean_number):
            result = copy.deepcopy(self.result); mutate(result)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ContractError): self.load(result)

    def test_wrong_nonce_plan_request_versions_and_spec_are_rejected_even_with_current_file_hash(self):
        self.fixture()
        for key, value in {"nonce":"00000000-0000-4000-8000-000000000000", "plan_hash":"0"*64,
                "request_sha256":"0"*64, "spec_hash":"0"*64,
                "model_version":{"id":"other", "metadata_hash":"a"*64},
                "material_version":{"id":"other", "metadata_hash":"b"*64}}.items():
            result = copy.deepcopy(self.result); result[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ContractError, "bound"): self.load(result)

    def test_manifest_plan_and_private_source_bytes_cannot_change_after_prepare(self):
        self.fixture()
        manifest = copy.deepcopy(self.manifest); manifest["plan"]["trajectories"][0]["x0"][0] += .01
        with self.assertRaisesRegex(ContractError, "manifest"): self.load(manifest=manifest)
        source = Path(self.manifest["source_file"])
        source.write_text(source.read_text(encoding="utf-8") + "% modified after prepare\n", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "source copy changed"): self.load()

    def test_result_hash_cannot_be_self_declared_after_bytes_change(self):
        self.fixture()
        digest = self.write_json("result.json", self.result)
        self.write_json("manifest.json", self.manifest)
        changed = copy.deepcopy(self.result); changed["matlab_release"] = "edited"
        self.write_json("result.json", changed)
        with self.assertRaisesRegex(ContractError, "Host-observed file hash"):
            load_matlab_batch_adapter(self.spec,self.plan,result_path=self.root/"result.json",result_sha256=digest,**self.binding)

    def test_partial_valid_prefix_is_preserved_and_later_trajectory_is_never_completed(self):
        self.fixture()
        partial = self.partial()
        adapter = self.load(partial)
        with self.assertRaises(SamplingStopped) as caught: collect(adapter, self.plan)
        self.assertEqual(caught.exception.report["status"], "failed")
        self.assertEqual(caught.exception.report["valid_rows"], 6)
        self.assertEqual(caught.exception.report["completed_trajectories"], 1)
        self.assertEqual(adapter._trajectory, 1)
        self.assertEqual(set(caught.exception.arrays["trajectory_id"]), {7, 11})
        self.assertFalse(caught.exception.report["invalid_transition_in_arrays"])
        np.testing.assert_allclose(caught.exception.arrays["x"],partial["arrays"]["x"],rtol=0,atol=0)
        with self.assertRaises(MatlabBatchError): adapter.reset(self.plan["trajectories"][2]["x0"])

    def test_zero_row_failure_retains_empty_correctly_dimensioned_arrays(self):
        self.fixture(nd=0)
        partial = self.partial(completed_trajectories=0, valid_steps=0)
        partial["arrays"]["d"] = []
        with self.assertRaises(SamplingStopped) as caught: collect(self.load(partial), self.plan)
        self.assertEqual(caught.exception.report["valid_rows"], 0)
        self.assertEqual(caught.exception.arrays["x"].shape, (0, 2))
        self.assertEqual(caught.exception.arrays["d"].shape, (0, 0))

    def test_partial_inconsistent_summary_failure_or_later_rows_are_rejected(self):
        self.fixture()
        def missing_failure(result): result["failure"] = []
        def wrong_failure_trajectory(result): result["failure"]["trajectory_id"] = 7
        def wrong_failure_step(result): result["failure"]["step_index"] = 1
        def incorrectly_completed(result): result["status"] = "completed"
        def incorrectly_failed(result): result["status"] = "failed"
        def later_trajectory(result): result["trajectories"].append(copy.deepcopy(self.result["trajectories"][2]))
        def incomplete_marked_complete(result): result["trajectories"][-1]["status"] = "completed"
        for mutate in (missing_failure,wrong_failure_trajectory,wrong_failure_step,incorrectly_completed,
                incorrectly_failed,later_trajectory,incomplete_marked_complete):
            partial = self.partial(); mutate(partial)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ContractError): self.load(partial)

    def test_completed_result_cannot_carry_failure_or_incomplete_summaries(self):
        self.fixture()
        for mutation in ("failure", "missing summary", "incomplete", "fractional count"):
            result = copy.deepcopy(self.result)
            if mutation == "failure": result["failure"] = {"reason":"synthetic contradiction"}
            if mutation == "missing summary": result["trajectories"].pop()
            if mutation == "incomplete": result["trajectories"][-1]["completed_steps"] = 3
            if mutation == "fractional count": result["trajectories"][-1]["completed_steps"] = 4.0
            with self.subTest(mutation=mutation), self.assertRaises(ContractError): self.load(result)

    def test_static_synthetic_slx_rejects_feedthrough_hidden_state_and_callbacks(self):
        spec = self.simulink_spec()
        approved = validate_simulink_source(self.slx(), spec)
        self.assertFalse(approved["source_executed"])
        self.assertEqual(approved["output_semantics"], "state_only")
        for options in ({"feedthrough": True}, {"hidden_state": True}, {"callback": True}):
            with self.subTest(options=options), self.assertRaises(ContractError):
                validate_simulink_source(self.slx(**options), spec)

    def test_static_matlab_hidden_memory_in_main_or_bound_dependency_is_rejected(self):
        self.fixture()
        source_root = self.root / "input"
        main = source_root / self.spec["source_file"]
        main.write_text("function next=protocol_fixture(x,u,d,dt,p)\nnext=helper(x);\nend\n",encoding="utf-8")
        helper = source_root / "source/helper.m"
        helper.write_text("function value=helper(x)\nvalue=x;\nend\n",encoding="utf-8")
        def inventory():
            return {path.relative_to(source_root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (main,helper)}
        report = validate_matlab_sources(source_root,self.spec,inventory())
        self.assertFalse(report["source_executed"])
        for keyword in ("persistent memory", "global memory"):
            helper.write_text(f"function value=helper(x)\n{keyword}\nvalue=x;\nend\n",encoding="utf-8")
            with self.subTest(keyword=keyword), self.assertRaisesRegex(ContractError,"hidden-state"):
                validate_matlab_sources(source_root,self.spec,inventory())
        helper.write_text("function value=helper(x)\nvalue=evalin('base','memory');\nend\n",encoding="utf-8")
        with self.assertRaisesRegex(ContractError,"hidden-state"):
            validate_matlab_sources(source_root,self.spec,inventory())

    def test_static_matlab_comments_and_strings_do_not_fake_hidden_state_or_hide_changed_bytes(self):
        self.fixture()
        source_root = self.root / "input"
        main = source_root / self.spec["source_file"]
        main.write_text("function next=protocol_fixture(x,u,d,dt,p)\n% persistent global evalin\n"
            "message='persistent global';\nnext=x;\nend\n",encoding="utf-8")
        inventory = {self.spec["source_file"]:hashlib.sha256(main.read_bytes()).hexdigest()}
        report = validate_matlab_sources(source_root,self.spec,inventory)
        self.assertEqual(report["status"],"supported_subset")
        main.write_text(main.read_text(encoding="utf-8") + "% changed bytes\n",encoding="utf-8")
        with self.assertRaisesRegex(ContractError,"hash changed"):
            validate_matlab_sources(source_root,self.spec,inventory)


if __name__ == "__main__": unittest.main()
