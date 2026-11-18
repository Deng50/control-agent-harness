"""Golden NumPy plants, independent derivative solutions and explicit reset/state contracts."""
from __future__ import annotations
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from control_harness.contracts import ContractError
from control_harness.model_adapters import load_python_adapter, validate_adapter_spec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "model-adapter-tests"))
OUTPUT.mkdir(parents=True, exist_ok=True)


def interface(nx=2, nu=1, nd=1, ny=2, dt=.2):
    variable = lambda name, unit: {"name": name, "unit": unit, "lower": -100, "upper": 100}
    return {"schema_version": 1, "name": "independent NumPy fixture", "sample_time": dt,
        "time_semantics": "discrete_state_transition", "states": [variable(f"x{k}", "m" if k % 2 == 0 else "m/s") for k in range(nx)],
        "controls": [variable(f"u{k}", "N") for k in range(nu)], "disturbances": [variable(f"d{k}", "N") for k in range(nd)],
        "outputs": [variable(f"y{k}", "m") for k in range(ny)]}


TRANSITION = '''import numpy as np
from .coefficients import matrices
def transition(x,u,d,dt,parameters):
    assert dt == .2
    A,B,E=matrices()
    return A@x+B@u+E@d+np.asarray(parameters['bias'],dtype=np.float64)
def observe(x,parameters):
    return np.array([x[0]+.1*x[1],x[1]],dtype=np.float64)
'''
COEFFICIENTS = '''import numpy as np
def matrices():
    return (np.array([[.8,.1],[0.,.9]],dtype=np.float64),np.array([[.2],[-.1]],dtype=np.float64),np.array([[.05],[.02]],dtype=np.float64))
'''
FACTORY = '''import numpy as np
class Plant:
    def __init__(self,p):
        self.sample_time=p['dt']; self.p=p; self.steps=0
    def reset(self,x0):
        self.state=np.asarray(x0,dtype=np.float64).copy(); self.steps=0
    def step(self,u,d):
        self.steps+=1
        self.state=np.asarray(self.p['A'],dtype=np.float64)@self.state+np.asarray(self.p['B'],dtype=np.float64)@u
        return self.state.copy()
    def observe(self):
        return self.state[[0,2]].copy()
def create(p): return Plant(p)
'''
DERIVATIVE = '''import numpy as np
def derivative(x,u,d,dt,parameters):
    assert dt == .2
    return np.array([-.4*x[0]+.3*u[0]+.2*d[0]+.01,.3*x[1]-.2*u[0]+.1*d[0]-.02],dtype=np.float64)
'''


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.root = Path(self.temporary.name)
    def tearDown(self): self.temporary.cleanup()
    def source(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return name
    def spec(self, **overrides):
        value = {"schema_version": 1, "backend": "python_callable", "source_file": "ordinary plant.py",
            "interface": interface(), "parameters": {"bias": [.01, -.02]}, "source_semantics": "transition",
            "entrypoint": "transition", "output_entrypoint": "observe"}
        value.update(overrides)
        return value
    def ordinary(self):
        self.source("ordinary plant.py", TRANSITION)
        self.source("coefficients.py", COEFFICIENTS)
        return self.spec()
    def factory(self, strategy="recreate", text=FACTORY):
        self.source("factory.py", text)
        A = [[.8,.1,0,0],[0,.7,.05,0],[0,0,.9,.1],[0,0,0,.6]]
        B = [[.2,0],[0,.1],[-.1,.05],[.03,.2]]
        return {"schema_version": 1, "backend": "python_factory", "source_file": "factory.py", "factory": "create",
            "interface": interface(4, 2, 0, 2, .1), "parameters": {"A": A, "B": B, "dt": .1},
            "source_semantics": "transition", "reset_strategy": strategy, "output_entrypoint": "observe"}

    def test_static_validation_never_imports_user_source(self):
        self.source("ordinary plant.py", "raise RuntimeError('source-imported-only-at-authorized-runtime')\n")
        self.assertEqual(validate_adapter_spec(self.spec())["backend"], "python_callable")
        with self.assertRaisesRegex(RuntimeError, "authorized-runtime"):
            load_python_adapter(self.root, self.spec())

    def test_two_state_numpy_callable_matches_independent_multitrajectory_reference_and_copies(self):
        spec = self.ordinary()
        A, B, E = np.array([[.8,.1],[0,.9]]), np.array([.2,-.1]), np.array([.05,.02])
        controls, disturbances = [.3, -.2, .1, .4], [.05, -.03, .02, 0]
        with load_python_adapter(self.root, spec) as adapter:
            with self.assertRaises(ContractError): adapter.output()
            for x0 in ([.4, -.2], [-.1, .3], [.4, -.2]):
                initial = np.array(x0, dtype=np.float64)
                adapter.reset(initial); initial[:] = 99
                expected = np.array(x0, dtype=np.float64)
                for u, d in zip(controls, disturbances):
                    expected = A @ expected + B*u + E*d + np.array([.01,-.02])
                    actual = adapter.step([u], [d])
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(adapter.output(), [expected[0]+.1*expected[1], expected[1]], rtol=0, atol=1e-15)
                    actual[:] = 100
                    detached = adapter.state; detached[:] = 200
                    np.testing.assert_allclose(adapter.state, expected, rtol=0, atol=1e-15)
            description = adapter.describe()
            self.assertEqual(description["state_order"], ["x0", "x1"])
            self.assertEqual(set(description["source_hashes"]), {"ordinary plant.py", "coefficients.py"})
            namespace = adapter.runtime.namespace
        self.assertFalse(any(key == namespace or key.startswith(namespace + '.') for key in sys.modules))

    def test_four_state_two_control_factory_matches_reference_with_both_explicit_reset_strategies(self):
        for strategy in ("reset", "recreate"):
            with self.subTest(strategy=strategy):
                spec = self.factory(strategy)
                A, B = np.asarray(spec["parameters"]["A"]), np.asarray(spec["parameters"]["B"])
                with load_python_adapter(self.root, spec) as adapter:
                    sequences = []
                    first_instance = None
                    for iteration in range(2):
                        expected = np.array([.2,-.1,.4,.3])
                        adapter.reset(expected)
                        if iteration == 0: first_instance = adapter.instance
                        elif strategy == "reset": self.assertIs(adapter.instance, first_instance)
                        else: self.assertIsNot(adapter.instance, first_instance)
                        trajectory = [adapter.state]
                        for u in ([.2,-.1], [-.3,.4], [.1,.2]):
                            expected = A @ expected + B @ np.asarray(u)
                            np.testing.assert_allclose(adapter.step(u), expected, rtol=0, atol=1e-15)
                            np.testing.assert_allclose(adapter.output(), expected[[0,2]], rtol=0, atol=1e-15)
                            trajectory.append(adapter.state)
                        sequences.append(np.asarray(trajectory))
                    np.testing.assert_array_equal(*sequences)

    def test_derivative_euler_and_rk4_are_explicit_next_state_integrators_with_held_inputs(self):
        self.source("derivative.py", DERIVATIVE)
        for method, substeps in (("euler", 16), ("rk4", 8)):
            spec = self.spec(source_file="derivative.py", entrypoint="derivative", source_semantics="derivative", C=[[1,0],[0,1]],
                time_unit="s", derivative_units=["m/s", "(m/s)/s"],
                integration={"method": method, "substeps": substeps, "input_hold": "zero_order_hold"})
            spec.pop("output_entrypoint")
            with self.subTest(method=method), load_python_adapter(self.root, spec) as adapter:
                initial = [.4,-.2]
                adapter.reset(initial); exact = np.array(initial)
                for u, d in ((.2,.05), (-.1,.03), (.3,-.02)):
                    force = [.3*u+.2*d+.01, -.2*u+.1*d-.02]
                    rates = [-.4,.3]
                    expected = []
                    for x, rate, offset in zip(exact, rates, force):
                        factor = (1+rate*.2/substeps)**substeps if method == "euler" else math.exp(rate*.2)
                        expected.append(factor*x+(factor-1)/rate*offset)
                    actual = adapter.step([u], [d]); exact = np.array(expected)
                    np.testing.assert_allclose(actual, exact, rtol=0, atol=2e-10)
                    np.testing.assert_array_equal(adapter.output(), adapter.state)
                self.assertEqual(adapter.describe()["integration"]["input_hold"], "zero_order_hold")

    def test_factory_hidden_memory_affecting_dynamics_is_rejected_as_undeclared_state(self):
        source = FACTORY.replace("return self.state.copy()", "self.state=self.state+.01*self.steps\n        return self.state.copy()")
        with load_python_adapter(self.root, self.factory(text=source)) as adapter:
            adapter.reset([.2,-.1,.4,.3]); adapter.step([.1,.2])
            with self.assertRaisesRegex(ContractError, "hidden memory"): adapter.step([.1,.2])
            with self.assertRaisesRegex(ContractError, "reset before use"): adapter.state

    def test_factory_hidden_memory_affecting_output_is_rejected(self):
        source = FACTORY.replace("return self.state[[0,2]].copy()", "return self.state[[0,2]].copy()+self.steps*.01")
        with load_python_adapter(self.root, self.factory(text=source)) as adapter:
            adapter.reset([.2,-.1,.4,.3]); adapter.step([.1,.2])
            with self.assertRaisesRegex(ContractError, "hidden memory"): adapter.output()

    def test_callable_module_memory_cannot_masquerade_as_declared_state(self):
        source = "import numpy as np\ncounter=0\ndef transition(x,u,d,dt,parameters):\n global counter\n counter+=1\n return x+counter*.01\n"
        self.source("memory.py", source)
        spec = self.spec(source_file="memory.py", C=[[1,0],[0,1]])
        spec.pop("output_entrypoint")
        with load_python_adapter(self.root, spec) as adapter:
            adapter.reset([0,0])
            with self.assertRaisesRegex(ContractError, "hidden memory"): adapter.step([0], [0])

    def test_wrong_shapes_nonfinite_and_lower_precision_outputs_are_rejected(self):
        cases = {"column": "np.zeros((2,1),dtype=np.float64)", "nonfinite": "np.array([np.inf,0.],dtype=np.float64)",
                 "float32": "np.zeros(2,dtype=np.float32)", "boolean": "np.array([True,False])"}
        for name, result in cases.items():
            with self.subTest(name=name):
                self.source("bad.py", f"import numpy as np\ndef transition(x,u,d,dt,parameters): return {result}\n")
                spec = self.spec(source_file="bad.py", C=[[1,0],[0,1]]); spec.pop("output_entrypoint")
                with load_python_adapter(self.root, spec) as adapter:
                    adapter.reset([0,0])
                    with self.assertRaises((ContractError, FloatingPointError)): adapter.step([0],[0])

    def test_controls_disturbances_dt_and_reset_are_not_implicitly_broadcast_or_guessed(self):
        with load_python_adapter(self.root, self.ordinary()) as adapter:
            with self.assertRaises(ContractError): adapter.reset([[0],[0]])
            adapter.reset([0,0])
            with self.assertRaises(ContractError): adapter.step(.1, [0])
            adapter.reset([0,0])
            with self.assertRaises(ContractError): adapter.step([.1])
        wrong_dt = self.factory(); wrong_dt["parameters"]["dt"] = .3
        with load_python_adapter(self.root, wrong_dt) as adapter:
            with self.assertRaisesRegex(ContractError, "sample_time"): adapter.reset([0,0,0,0])
        bad_reset = FACTORY.replace("self.state=np.asarray(x0,dtype=np.float64).copy();", "self.state=np.zeros(4,dtype=np.float64);")
        with load_python_adapter(self.root, self.factory(text=bad_reset)) as adapter:
            with self.assertRaisesRegex(ContractError, "exact declared initial state"): adapter.reset([1,2,3,4])

    def test_unknown_semantics_integrators_units_factory_derivatives_and_feedthrough_fail_statically(self):
        original = self.ordinary()
        for patch in ({"source_semantics": "guess"}, {"source_file": "../escape.py"}, {"source_file": "C:/outside.py"},
                      {"C": [[1,0]], "output_entrypoint": "observe"}, {"D": [[1]]}, {"parameters": []},
                      {"schema_version": True}, {"project_autodetect": True},
                      {"interface": {**interface(), "sample_time": ".2"}}):
            with self.subTest(patch=patch), self.assertRaises(ContractError): validate_adapter_spec({**original, **patch})
        derivative = copy.deepcopy(original); derivative.update(source_semantics="derivative", time_unit="s",
            derivative_units=["m/s", "(m/s)/s"], integration={"method":"rk4","substeps":2,"input_hold":"zero_order_hold"})
        for patch in ({"time_unit":"ms"}, {"derivative_units":["m","m/s"]}, {"integration":{"method":"zoh","substeps":1,"input_hold":"zero_order_hold"}},
                      {"integration":{"method":"rk4","substeps":True,"input_hold":"zero_order_hold"}}):
            with self.subTest(patch=patch), self.assertRaises(ContractError): validate_adapter_spec({**derivative, **patch})
        factory = self.factory(); factory.update(source_semantics="derivative")
        with self.assertRaisesRegex(ContractError, "factory derivatives"): validate_adapter_spec(factory)

    def test_undeclared_absolute_imports_external_io_and_missing_relative_dependencies_are_rejected(self):
        for body in ("import os\n", "from .missing import coefficients\n", "open('external.txt')\n"):
            with self.subTest(body=body):
                self.source("unsupported.py", body + "def transition(x,u,d,dt,parameters): return x\n")
                spec = self.spec(source_file="unsupported.py", C=[[1,0],[0,1]]); spec.pop("output_entrypoint")
                with self.assertRaises((ContractError, ModuleNotFoundError)): load_python_adapter(self.root, spec)

    def test_factory_lazy_properties_use_the_same_verified_snapshot_import_loader(self):
        source = FACTORY.replace("self.sample_time=p['dt']; self.p=p; self.steps=0", "self.p=p; self.steps=0")
        source = source.replace("    def reset(self,x0):", "    @property\n    def sample_time(self):\n        from .settings import DT\n        return DT\n    def reset(self,x0):")
        self.source("settings.py", "DT=.1\n")
        with load_python_adapter(self.root, self.factory(text=source)) as adapter:
            adapter.reset([0,0,0,0])
            self.assertIn("settings.py", adapter.describe()["source_hashes"])
        self.source("settings.py", "import os\nDT=.1\n")
        with load_python_adapter(self.root, self.factory(text=source)) as adapter:
            with self.assertRaisesRegex(ContractError, "math imports"):
                adapter.reset([0,0,0,0])

    def test_mixed_boolean_vectors_are_not_silently_promoted_to_numeric_inputs(self):
        with load_python_adapter(self.root, self.ordinary()) as adapter:
            with self.assertRaisesRegex(ContractError, "boolean"):
                adapter.reset([False, .1])

    def test_bound_primary_hash_is_checked_before_tampered_code_executes(self):
        spec = self.ordinary()
        hashes = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                  for name in ("ordinary plant.py", "coefficients.py")}
        with load_python_adapter(self.root, spec, expected_file_hashes=hashes) as adapter:
            adapter.reset([0, 0]); adapter.step([0], [0])
            self.assertEqual(adapter.describe()["material_hash_verification"], "every_loaded_module_before_execution")
        self.source("ordinary plant.py", "raise RuntimeError('TAMPERED_PRIMARY_EXECUTED')\n")
        with self.assertRaisesRegex(ContractError, "source hash changed before execution"):
            load_python_adapter(self.root, spec, expected_file_hashes=hashes)

    def test_lazy_helpers_are_hash_checked_and_must_belong_to_the_bound_closure(self):
        source = "def transition(x,u,d,dt,parameters):\n from .helper import offset\n return x+offset\n"
        self.source("source/primary.py", source)
        self.source("source/helper.py", "offset=0.01\n")
        hashes = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                  for name in ("source/primary.py", "source/helper.py")}
        spec = self.spec(source_file="source/primary.py", C=[[1, 0], [0, 1]])
        spec.pop("output_entrypoint")
        with load_python_adapter(self.root, spec, expected_file_hashes=hashes) as adapter:
            adapter.reset([0, 0])
            # Modify a helper after the primary module was loaded.  The helper
            # is read lazily and rejected before its top-level code runs.
            self.source("source/helper.py", "raise RuntimeError('TAMPERED_HELPER_EXECUTED')\n")
            with self.assertRaisesRegex(ContractError, "source hash changed before execution"):
                adapter.step([0], [0])
        self.source("source/helper.py", "offset=0.01\n")
        with load_python_adapter(self.root, spec, expected_file_hashes={"source/primary.py": hashes["source/primary.py"]}) as adapter:
            adapter.reset([0, 0])
            with self.assertRaisesRegex(ContractError, "outside the bound material file closure"):
                adapter.step([0], [0])

    def test_expected_material_hash_map_is_explicit_and_confined(self):
        spec = self.ordinary()
        for hashes in ({}, {"../ordinary plant.py": "a" * 64}, {"ordinary plant.py": "not-sha256"}, []):
            with self.subTest(hashes=hashes), self.assertRaises(ContractError):
                load_python_adapter(self.root, spec, expected_file_hashes=hashes)


if __name__ == "__main__": unittest.main()
