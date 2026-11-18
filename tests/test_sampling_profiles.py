"""Synthetic mathematical sampling references; no training, vehicle or MATLAB claim."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from control_harness.contracts import ContractError, ModelInterface
from control_harness.model_adapters import load_python_adapter
from control_harness.runtime import Cancelled, OperationContext
from control_harness.sampling import (SamplingCancelled, SamplingStopped,
    build_sampling_plan, collect, prefix_plan, recommend_sampling)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "sampling-tests"))


def interface(nx=2, nu=1, nd=1, ny=1, dt=.2):
    def variable(name, bound):
        return {"name": name, "unit": "fixture-unit", "lower": -bound, "upper": bound}
    return {"schema_version": 1, "name": "Independent synthetic linear fixture",
        "time_semantics": "discrete_state_transition", "sample_time": dt,
        "states": [variable(f"x{k}", 10.) for k in range(nx)],
        "controls": [variable(f"u{k}", 2.) for k in range(nu)],
        "disturbances": [variable(f"d{k}", 1.) for k in range(nd)],
        "outputs": [variable(f"y{k}", 20.) for k in range(ny)]}


def constant(value):
    return {"kind": "step", "times": [0], "values": [value]}


def config(raw=None, **overrides):
    raw = raw or interface()
    settings = {"mode": "expert", "seed": 123, "steps": 6, "trajectory_count": 2,
        "x0": [.1] * len(raw["states"]), "u0": [0.] * len(raw["controls"]),
        "d0": [0.] * len(raw["disturbances"]),
        "u_profile": constant([.2] * len(raw["controls"])),
        "d_profile": constant([.05] * len(raw["disturbances"])) if raw["disturbances"] else None,
        "control_rate_limits": None, "disturbance_rate_limits": None, "coupled_constraints": []}
    settings.update(overrides)
    return settings


class LinearAdapter:
    def __init__(self, raw=None):
        self.interface = ModelInterface.parse(raw or interface())
        nx, nu, nd, ny = self.interface.nx, self.interface.nu, self.interface.nd, self.interface.ny
        self.A = np.diag(np.linspace(.7, .9, nx))
        self.B = np.arange(1, nx * nu + 1).reshape(nx, nu) * .015
        self.E = np.arange(1, nx * nd + 1).reshape(nx, nd) * .01
        self.C = np.zeros((ny, nx))
        for k in range(ny):
            self.C[k, k % nx] = 1.
        self.C[:, -1] += .1
        self.calls = {"reset": 0, "step": 0, "output": 0, "describe": 0}
        self._state = np.zeros(nx)

    def reset(self, x0):
        self.calls["reset"] += 1
        self._state = x0.copy()
        return self.state

    @property
    def state(self):
        return self._state.copy()

    def step(self, u, d=None):
        self.calls["step"] += 1
        self._state = self.A @ self._state + self.B @ u + self.E @ d
        return self.state

    def output(self):
        self.calls["output"] += 1
        return self.C @ self._state

    def describe(self):
        self.calls["describe"] += 1
        return {"backend": "independent_test_fixture", "source_semantics": "transition"}


class SamplingProfilesTests(unittest.TestCase):
    def plan(self, raw=None, **overrides):
        return build_sampling_plan(raw or interface(), config(raw, **overrides))

    def test_step_grid_interval_times_and_previous_command_are_explicit(self):
        plan = self.plan(steps=5, u_profile={"kind": "step", "times": [0, .4, .8], "values": [[.1], [-.2], [.3]]})
        np.testing.assert_allclose(plan["trajectories"][0]["t"], [0, .2, .4, .6, .8, 1.])
        np.testing.assert_allclose(plan["trajectories"][0]["u"], [[.1], [.1], [-.2], [-.2], [.3]])
        self.assertEqual(plan["trajectories"][0]["u0"], [0.])
        self.assertEqual(plan["duration"], 1.)

    def test_chirp_matches_independent_analytic_values_and_envelope(self):
        profile = {"kind": "chirp", "offset": [.1], "amplitude": [.3], "f0_hz": .1, "f1_hz": 1., "phase": [.25]}
        plan = self.plan(steps=10, u_profile=profile)
        expected = [[.1 + .3 * math.sin(2 * math.pi * (.1 * t + .5 * .9 / 2. * t*t) + .25)] for t in np.arange(10)*.2]
        np.testing.assert_allclose(plan["trajectories"][0]["u"], expected, rtol=0, atol=2e-16)
        for changed in ({"f1_hz": 3.}, {"amplitude": [2.]}, {"f0_hz": -.1}, {"amplitude": [-.1]}):
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                self.plan(u_profile={**profile, **changed})

    def test_random_is_seeded_held_and_rate_limited_without_clipping_user_values(self):
        profile = {"kind": "bounded_random", "low": [-1.], "high": [1.], "hold_steps": 3}
        plan = self.plan(steps=20, u_profile=profile, control_rate_limits=[.25])
        self.assertEqual(plan, self.plan(steps=20, u_profile=profile, control_rate_limits=[.25]))
        u = np.asarray(plan["trajectories"][0]["u"])
        self.assertLessEqual(np.max(np.abs(np.diff(np.vstack([[0.], u]), axis=0))) / .2, .25 + 1e-12)
        for k in range(20):
            np.testing.assert_array_equal(u[k], u[(k//3)*3])
        other = self.plan(steps=20, seed=124, u_profile=profile, control_rate_limits=[.25])
        self.assertFalse(np.array_equal(u, other["trajectories"][0]["u"]))
        self.assertFalse(np.array_equal(u, plan["trajectories"][1]["u"]))
        with self.assertRaisesRegex(ContractError, "per-second"):
            self.plan(u_profile=constant([.3]), control_rate_limits=[1.])

    def test_timeseries_zoh_linear_and_no_extrapolation(self):
        series = {"kind": "timeseries", "time": [0, .4, .8], "values": [[0.], [.4], [0.]]}
        zoh = self.plan(steps=5, u_profile=series)
        linear = self.plan(steps=5, u_profile={**series, "interpolation": "linear"})
        np.testing.assert_allclose(zoh["trajectories"][0]["u"], [[0], [0], [.4], [.4], [0]])
        np.testing.assert_allclose(linear["trajectories"][0]["u"], [[0], [.2], [.4], [.2], [0]], atol=1e-15)
        with self.assertRaisesRegex(ContractError, "covering"):
            self.plan(steps=6, u_profile=series)

    def test_recommendation_is_complete_reviewable_and_uses_same_expert_validator(self):
        raw = interface()
        candidate = recommend_sampling(raw)
        self.assertEqual(candidate["trajectory_count"], 5)
        self.assertTrue(candidate["recommendation"]["requires_user_review"])
        self.assertFalse(candidate["recommendation"]["physical_model_inferred"])
        self.assertFalse(candidate["recommendation"]["sufficient_excitation_proved"])
        self.assertEqual(build_sampling_plan(raw, candidate)["config"], candidate)
        overridden = recommend_sampling(raw, {"steps": 4, "trajectory_count": 1, "x0": [.3, -.2],
            "u0": [.1], "u_profile": constant([.15]), "control_rate_limits": [.5], "seed": 7})
        condition = overridden["trajectories"][0]
        self.assertEqual(condition["x0"], [.3, -.2])
        self.assertEqual(build_sampling_plan(raw, overridden)["trajectories"][0]["u"], [[.15]]*4)
        expert = copy.deepcopy(overridden); expert["mode"] = "expert"
        self.assertEqual(build_sampling_plan(raw, expert)["trajectories"], build_sampling_plan(raw, overridden)["trajectories"])
        with self.assertRaises(ContractError):
            recommend_sampling(raw, {"x0": [11., 0.]})
        self.assertEqual(recommend_sampling(raw, {"duration": .8})["steps"], 4)

    def test_preexecution_bounds_width_dt_count_and_type_rejections(self):
        cases = [{"x0": [11., 0.]}, {"u0": [3.]}, {"d0": [2.]}, {"x0": [0.]},
            {"u_profile": constant([3.])}, {"d_profile": constant([2.])}, {"u_profile": constant(.1)},
            {"dt": .1}, {"duration": 1.1}, {"steps": True}, {"steps": "6"}, {"trajectory_count": True},
            {"u0": [False]}, {"x0": [False, .1]}, {"control_rate_limits": [-1.]},
            {"control_rate_limits": [False]}, {"disturbance_rate_limits": [.1, .2]},
            {"trajectories": [{}, {}], "trajectory_count": 1},
            {"trajectories": [{"trajectory_id": 0}, {"trajectory_id": 0}]},
            {"trajectories": [{"trajectory_id": 2**53}], "trajectory_count": 1},
            {"seed": 2**53}, {"recommendation": {"bad": float("nan")}}]
        for bad in cases:
            adapter = LinearAdapter()
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                collect(adapter, self.plan(**bad))
            self.assertEqual(sum(adapter.calls.values()), 0)

    def test_profile_bad_times_shapes_nonfinite_fields_rejected(self):
        cases = [{"kind": "step", "times": [0, .3], "values": [[0], [.1]]},
            {"kind": "step", "times": [0, 2], "values": [[0], [.1]]},
            {"kind": "step", "times": [0, 0], "values": [[0], [.1]]},
            {"kind": "step", "times": 0, "values": [[0]]},
            {"kind": "step", "times": [0], "values": [[float("nan")]]},
            {"kind": "step", "times": [0], "values": [[True]]},
            {"kind": "bounded_random", "low": [.1], "high": [.2]},
            {"kind": "bounded_random", "low": [1.], "high": [-1.]},
            {"kind": "timeseries", "time": [0, 1], "values": [[0], [0]], "interpolation": "cubic"},
            {"kind": "step", "times": [0], "values": [[0]], "unrecognized": 1}, {"kind": "anything"}]
        for profile in cases:
            with self.subTest(profile=profile), self.assertRaises(ContractError): self.plan(u_profile=profile)

    def test_batch_budget_rejects_before_condition_or_profile_array_allocation(self):
        # These calls must exit before a huge condition list or signal array.
        with patch("control_harness.sampling._profile", side_effect=AssertionError("profile allocation reached")) as profile:
            with patch("control_harness.sampling.np.empty", side_effect=AssertionError("array allocation reached")) as empty:
                for bad in ({"trajectory_count": 10**12}, {"steps": 10**12},
                        {"duration": 10**12, "steps": 5_000_000_000_000}):
                    with self.subTest(bad=bad), self.assertRaisesRegex(ContractError, "allocation budget.*batches"):
                        self.plan(**bad)
                for bad in ({"trajectory_count": 10**12}, {"steps": 10**12}, {"duration": 10**12}):
                    with self.subTest(recommend=bad), self.assertRaisesRegex(ContractError, "allocation budget.*batches"):
                        recommend_sampling(interface(), bad)
                profile.assert_not_called()
                empty.assert_not_called()

    def test_rates_are_per_second_and_include_initial_control_and_disturbance(self):
        self.plan(u_profile=constant([.1]), control_rate_limits=[.5], d_profile=constant([.04]), disturbance_rate_limits=[.2])
        for bad in ({"control_rate_limits": [.499]}, {"disturbance_rate_limits": [.199]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ContractError, "per-second"):
                self.plan(**{"u_profile": constant([.1]), "control_rate_limits": [.5],
                    "d_profile": constant([.04]), "disturbance_rate_limits": [.2], **bad})

    def test_input_coupling_and_initial_mixed_constraint_reject_before_execution(self):
        raw = interface(4, 2, 1, 2)
        for constraints in ([{"name": "sum", "u": [1, 1], "upper": .3}],
                [{"name": "initial mixed", "x": [1, 0, 0, 0], "u": [1, 0], "upper": .2}],
                [{"x": [0, 0, 0, 0], "upper": 1}], [{"u": [1, 0], "lower": 2, "upper": 1}],
                [{"u": [1], "upper": 1}]):
            with self.subTest(constraints=constraints), self.assertRaises(ContractError):
                self.plan(raw, coupled_constraints=constraints)

    def assert_reference(self, adapter, plan, arrays):
        row = 0
        for trajectory in plan["trajectories"]:
            x = np.array(trajectory["x0"])
            for k, (u, d) in enumerate(zip(trajectory["u"], trajectory["d"])):
                expected = adapter.A @ x + adapter.B @ np.array(u) + adapter.E @ np.array(d)
                for key, value in (("x", x), ("u", u), ("d", d), ("next_x", expected),
                        ("y", adapter.C @ x), ("next_y", adapter.C @ expected)):
                    np.testing.assert_allclose(arrays[key][row], value, rtol=0, atol=2e-15)
                self.assertEqual(arrays["trajectory_id"][row], trajectory["trajectory_id"])
                self.assertAlmostEqual(arrays["time"][row], k * plan["dt"])
                x, row = expected, row + 1

    def test_two_state_single_plant_reference_output_width_and_independent_resets(self):
        adapter = LinearAdapter()
        plan = self.plan()
        arrays, report = collect(adapter, plan)
        self.assert_reference(adapter, plan, arrays)
        self.assertEqual(arrays["x"].shape, (12, 2))
        self.assertEqual(arrays["y"].shape, (12, 1))
        np.testing.assert_array_equal(arrays["x"][:6], arrays["x"][6:])
        self.assertEqual(adapter.calls["reset"], 2)
        self.assertEqual(adapter.calls["step"], 12)
        self.assertEqual(report["completed_trajectories"], 2)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(report["stability_proved"])
        json.dumps(report, allow_nan=False)

    def test_four_state_two_control_no_disturbance_preserves_every_column(self):
        raw = interface(4, 2, 0, 2, .1)
        adapter = LinearAdapter(raw)
        plan = self.plan(raw, steps=7, u_profile={"kind": "step", "times": [0, .3], "values": [[.2, -.1], [-.3, .4]]})
        arrays, _ = collect(adapter, plan)
        self.assert_reference(adapter, plan, arrays)
        for key, width in (("x", 4), ("u", 2), ("d", 0), ("y", 2)):
            self.assertEqual(arrays[key].shape, (14, width))
        with self.assertRaisesRegex(ContractError, "width is zero"):
            self.plan(raw, d_profile=constant([]))

    def test_trial_is_exact_random_and_chirp_prefix_including_original_config(self):
        for profile in ({"kind": "bounded_random", "low": [-.5], "high": [.5], "hold_steps": 2},
                {"kind": "chirp", "offset": [0], "amplitude": [.5], "f0_hz": .1, "f1_hz": 1.}):
            full = self.plan(steps=20, trajectory_count=3, u_profile=profile)
            trial = prefix_plan(full, {"trajectories": 2, "steps": 4})
            self.assertEqual(trial["config"], full["config"])
            self.assertEqual(trial["parent_plan_hash"], full["plan_hash"])
            self.assertNotEqual(trial["plan_hash"], full["plan_hash"])
            for k in range(2):
                self.assertEqual(trial["trajectories"][k]["u"], full["trajectories"][k]["u"][:4])
                self.assertEqual(trial["trajectories"][k]["t"], full["trajectories"][k]["t"][:5])
            arrays, report = collect(LinearAdapter(), json.loads(json.dumps(trial)))
            self.assertEqual(len(arrays["x"]), 8)
            self.assertEqual(report["parent_plan_hash"], full["plan_hash"])
            with self.assertRaises(ContractError): prefix_plan(trial, {"steps": 1})
            with self.assertRaises(ContractError): prefix_plan(full, {"steps": 21})

    def test_json_roundtrip_and_changed_plan_rejected_before_adapter_execution(self):
        plan = self.plan(control_rate_limits=[1.])
        collect(LinearAdapter(), json.loads(json.dumps(plan)))
        changes = [("u", lambda p: p["trajectories"][0]["u"][0].__setitem__(0, .3)),
            ("hash", lambda p: p.__setitem__("plan_hash", "0"*64)),
            ("config", lambda p: p["config"].__setitem__("steps", 7)),
            ("rate bool", lambda p: p.__setitem__("control_rate_limits", [True])),
            ("numeric bool", lambda p: p["trajectories"][0]["x0"].__setitem__(0, False)),
            ("dt bool", lambda p: p.__setitem__("dt", True))]
        for name, change in changes:
            changed, adapter = copy.deepcopy(plan), LinearAdapter()
            change(changed)
            with self.subTest(name=name), self.assertRaises(ContractError): collect(adapter, changed)
            self.assertEqual(sum(adapter.calls.values()), 0)

    def test_state_bound_failure_is_immediate_with_only_valid_prefix(self):
        adapter = LinearAdapter()
        adapter.A, adapter.B, adapter.E = np.eye(2), np.array([[30.], [0.]]), np.zeros((2, 1))
        with self.assertRaises(SamplingStopped) as caught:
            collect(adapter, self.plan(steps=5, trajectory_count=3))
        failure = caught.exception
        self.assertEqual(adapter.calls["step"], 2)
        self.assertEqual(adapter.calls["reset"], 1)
        self.assertEqual(len(failure.arrays["x"]), 1)
        self.assertEqual(failure.report["failure"]["step_index"], 1)
        self.assertEqual(failure.report["status"], "failed")
        self.assertFalse(failure.report["invalid_transition_in_arrays"])
        self.assertGreater(failure.report["failure"]["attempted"]["next_x"][0], 10)

    def test_runtime_mixed_constraint_uses_next_state_with_held_command(self):
        adapter = LinearAdapter()
        adapter.A, adapter.B = np.eye(2), np.array([[1.], [0.]])
        plan = self.plan(coupled_constraints=[{"name": "x plus u", "x": [1, 0], "u": [1], "upper": .6}])
        with self.assertRaisesRegex(SamplingStopped, "x plus u") as caught: collect(adapter, plan)
        self.assertEqual(caught.exception.report["valid_rows"], 1)
        self.assertEqual(adapter.calls["step"], 2)

    def test_nonfinite_attempt_is_sanitized_and_failing_row_is_excluded(self):
        adapter = LinearAdapter()
        original = adapter.step
        def bad_step(u, d):
            value = original(u, d)
            if adapter.calls["step"] == 3:
                adapter._state[1] = float("nan")
                value = adapter.state
            return value
        adapter.step = bad_step
        with self.assertRaises(SamplingStopped) as caught: collect(adapter, self.plan())
        failure = caught.exception
        self.assertEqual(len(failure.arrays["x"]), 2)
        self.assertEqual(adapter.calls["step"], 3)
        self.assertEqual(adapter.calls["reset"], 1)
        self.assertEqual(failure.report["failure"]["nonfinite_indices"], {"next_x": [[1]]})
        self.assertIsNone(failure.report["failure"]["attempted"]["next_x"][1])
        json.dumps(failure.report, allow_nan=False)

    def test_current_and_next_output_bounds_are_enforced_without_fabricating_rows(self):
        for output_when in ("current", "next"):
            adapter = LinearAdapter()
            original = adapter.output
            def bad_output():
                if output_when == "current" or adapter.calls["step"]:
                    return np.array([21.])
                return original()
            adapter.output = bad_output
            with self.subTest(output_when=output_when), self.assertRaises(SamplingStopped) as caught:
                collect(adapter, self.plan())
            self.assertEqual(caught.exception.report["valid_rows"], 0)
            self.assertEqual(adapter.calls["step"], 0 if output_when == "current" else 1)
            self.assertEqual(caught.exception.report["failure"]["phase"], "output" if output_when == "current" else "next_output")

    def test_bad_reset_shape_return_state_and_exceptions_stop_collection(self):
        for kind in ("reset", "shape", "state", "exception"):
            adapter = LinearAdapter()
            if kind == "reset":
                adapter.reset = lambda x: np.zeros(2)
            elif kind == "shape":
                adapter.step = lambda u, d: np.zeros(3)
            elif kind == "state":
                adapter.step = lambda u, d: np.ones(2)
            else:
                def throw(u, d): raise RuntimeError("synthetic failure")
                adapter.step = throw
            with self.subTest(kind=kind), self.assertRaises(SamplingStopped) as caught:
                collect(adapter, self.plan())
            self.assertEqual(caught.exception.report["valid_rows"], 0)

    def test_cancel_after_step_preserves_native_cancel_semantics_and_valid_prefix(self):
        adapter = LinearAdapter()
        context = OperationContext(cancelled=lambda: adapter.calls["step"] >= 2)
        with self.assertRaises(SamplingCancelled) as caught: collect(adapter, self.plan(), context)
        self.assertIsInstance(caught.exception, Cancelled)
        self.assertEqual(caught.exception.report["status"], "cancelled")
        self.assertEqual(caught.exception.report["valid_rows"], 1)
        self.assertEqual(adapter.calls["step"], 2)
        self.assertEqual(adapter.calls["reset"], 1)

    def test_plan_adapter_interface_mismatch_does_not_reset_or_step(self):
        adapter = LinearAdapter(interface(dt=.1))
        with self.assertRaisesRegex(ContractError, "Interface mismatch"): collect(adapter, self.plan())
        self.assertEqual(sum(adapter.calls.values()), 0)

    def test_actual_python_callable_adapter_parameters_and_multitrajectory_reference(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        directory = OUTPUT / ("ordinary-source-" + uuid.uuid4().hex)
        directory.mkdir()
        source = '''import numpy as np
def transition(x,u,d,dt,parameters):
    return np.array([.8*x[0]+dt*parameters['gain']*u[0]+.02*d[0],.9*x[1]-.1*u[0]])
def observe(x,parameters):
    return np.array([x[0]+.2*x[1]])
'''
        try:
            (directory / "plant.py").write_text(source, encoding="utf-8")
            spec = {"schema_version": 1, "backend": "python_callable", "source_file": "plant.py",
                "interface": interface(), "parameters": {"gain": 1.5}, "source_semantics": "transition",
                "entrypoint": "transition", "output_entrypoint": "observe"}
            plan = self.plan(steps=4, trajectories=[{"trajectory_id": 7, "x0": [.4, -.2]}, {"trajectory_id": 9, "x0": [-.1, .3]}])
            with load_python_adapter(directory, spec) as adapter:
                arrays, report = collect(adapter, plan)
            index = 0
            for trajectory in plan["trajectories"]:
                a, b = trajectory["x0"]
                for k in range(4):
                    na, nb = .8*a + .2*1.5*.2 + .02*.05, .9*b - .1*.2
                    np.testing.assert_allclose(arrays["x"][index], [a, b], rtol=0, atol=2e-16)
                    np.testing.assert_allclose(arrays["next_x"][index], [na, nb], rtol=0, atol=2e-16)
                    np.testing.assert_allclose(arrays["y"][index], [a+.2*b], rtol=0, atol=2e-16)
                    a, b, index = na, nb, index + 1
            self.assertEqual(report["adapter"]["backend"], "python_callable")
            self.assertEqual(report["valid_rows"], 8)
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
