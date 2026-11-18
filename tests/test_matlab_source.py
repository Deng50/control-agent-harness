"""MATLAB source support checks run without MATLAB or source execution."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from control_harness.contracts import ContractError
from control_harness.matlab_source import validate_matlab_sources


ROOT = Path(__file__).resolve().parents[1]


class MatlabSourceTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "test-output" / "matlab-source-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def hashes(self):
        return {path.relative_to(self.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def spec(self, **overrides):
        return {"backend": "matlab_function", "source_file": "source/plant.m", "entrypoint": "plant", **overrides}

    def plant(self, body="y = x + dt*(u(1)+d(1))*ones(size(x));"):
        self.source("source/plant.m", f"function y = plant(x,u,d,dt,p)\n{body}\nend\n")

    def review(self, spec=None, hashes=None):
        return validate_matlab_sources(self.root, spec or self.spec(), self.hashes() if hashes is None else hashes)

    def test_real_f01_numerical_sources_are_supported_without_execution(self):
        for name in ("f01_exact_step", "f01_derivative"):
            with self.subTest(name=name):
                self.source(f"source/{name}.m", (ROOT / f"matlab/fixtures/{name}.m").read_text(encoding="utf-8"))
                report = self.review(self.spec(source_file=f"source/{name}.m", entrypoint=name))
                self.assertEqual(report["status"], "supported_subset")
                self.assertFalse(report["source_executed"])
                self.assertIn(name, report["function_inventory"])
        self.assertEqual(report["file_count"], 2)
        self.assertTrue(any("expm" in row["numerical_builtins"] for row in report["files"]))

    def test_comments_strings_escaping_and_transpose_do_not_create_false_forbidden_calls(self):
        self.plant("""% persistent global evalin('base','x')
%{
system('bad'); load external.mat
%{
global nested
%}
%}
text = ['load(' 'persistent' 'it''s harmless'];
other = "evalin('base', ""x"") and % not a comment";
A = [1 0; 0 1]; B = A'; C = A.';
y = B*C*x + dt*(u(1)+d(1))*ones(size(x)); % global
""")
        report = self.review()
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["files"][0]["functions"][0]["inputs"], ["x", "u", "d", "dt", "p"])

    def test_transpose_does_not_hide_a_following_real_forbidden_construct(self):
        for transpose in ("x'", "x.'"):
            with self.subTest(transpose=transpose):
                self.plant(f"y={transpose}; evalin('base','x');")
                with self.assertRaisesRegex(ContractError, "evalin"):
                    self.review()

    def test_continued_signature_arguments_and_struct_fields_are_explicit_data(self):
        self.source("source/plant.m", """function y = plant(x,u, ... rest is a comment mentioning system
d,dt,p)
arguments
 x (2,1) double {mustBeFinite, mustBeReal}
 u (1,1) double
 d (1,1) double
 dt (1,1) double
 p struct
end
y=x+dt*(u+d)*p.load+p.persistent;
end
""")
        self.assertEqual(self.review()["files"][0]["functions"][0]["inputs"], ["x", "u", "d", "dt", "p"])

    def test_local_functions_and_bound_helpers_have_a_function_inventory(self):
        self.plant("y = helper(x,u,d,dt,p) + local_scale(x);")
        self.source("source/plant.m", (self.root / "source/plant.m").read_text() + "function z=local_scale(v)\nz=.01*v;\nend\n")
        self.source("source/helper.m", "function y=helper(x,u,d,dt,p)\ny=x+dt*(u(1)+d(1))*ones(size(x));\nend\n")
        self.source("source/observe.m", "function y=observe(x,p)\ny=x(1);\nend\n")
        report = self.review(self.spec(output_entrypoint="observe"))
        self.assertEqual(report["function_inventory"], ["helper", "observe", "plant"])
        plant = next(row for row in report["files"] if row["primary_function"] == "plant")
        self.assertEqual(plant["snapshot_dependencies"], ["helper"])
        self.assertEqual([row["name"] for row in plant["functions"]], ["plant", "local_scale"])

    def test_package_function_dependencies_are_bound_and_named(self):
        self.source("source/+numeric/plant.m", "function y=plant(x,u,d,dt,p)\ny=numeric.offset(x,u,d,dt,p);\nend\n")
        self.source("source/+numeric/offset.m", "function y=offset(x,u,d,dt,p)\ny=x+dt*(u(1)+d(1))*ones(size(x));\nend\n")
        report = self.review(self.spec(source_file="source/+numeric/plant.m", entrypoint="numeric.plant"))
        self.assertEqual(report["function_inventory"], ["numeric.offset", "numeric.plant"])

    def test_helper_local_variables_do_not_hide_an_external_call_in_another_function(self):
        self.source("source/plant.m", """function y=plant(x,u,d,dt,p)
y=outside_dependency(x);
end
function y=local_helper(x)
outside_dependency=[1 2];
y=outside_dependency(1)*x;
end
""")
        with self.assertRaisesRegex(ContractError, "Unknown/external MATLAB dependency outside_dependency"):
            self.review()

    def test_snapshot_helpers_must_be_reachable_on_the_explicit_function_path(self):
        self.plant("y=helper(x,u,d,dt,p);")
        self.source("source/unrelated_directory/helper.m", "function y=helper(x,u,d,dt,p)\ny=x;\nend\n")
        with self.assertRaisesRegex(ContractError, "outside the explicit function path"):
            self.review()
        (self.root / "source/unrelated_directory/helper.m").unlink()
        self.source("source/private/helper.m", "function y=helper(x,u,d,dt,p)\ny=x;\nend\n")
        report = self.review()
        self.assertIn("helper", report["function_inventory"])

    def test_nested_closures_script_trailers_and_unbalanced_blocks_are_not_claimed_supported(self):
        sources = ["function y=plant(x,u,d,dt,p)\nfunction z=nested(v)\nz=v;\nend\ny=nested(x);\nend\n",
                   "function y=plant(x,u,d,dt,p)\ny=x;\nend\nother=1;\n",
                   "function y=plant(x,u,d,dt,p)\nif x(1)>0\ny=x;\n"]
        for source in sources:
            with self.subTest(source=source):
                self.source("source/plant.m", source)
                with self.assertRaises(ContractError): self.review()

    def test_every_bound_m_file_is_checked_including_unused_scripts(self):
        self.plant()
        self.source("source/unrelated.m", "global bad\nx=1;\n")
        with self.assertRaisesRegex(ContractError, "function file"):
            self.review()

    def test_hidden_memory_workspace_process_loading_and_nondeterminism_are_rejected(self):
        bodies = ["persistent count", "global shared", "y=eval('x');", "y=evalin('base','x');",
                  "assignin('base','x',x); y=x;", "clear plant; y=x;", "clearvars; y=x;", "run('external.m'); y=x;",
                  "load external.mat", "save('out.mat','x'); y=x;", "system('bad'); y=x;", "mex('external.c'); y=x;",
                  "y=rand(size(x));", "y=py.some_function(x);", "y=javaObject('External');", "y=now;",
                  "feval('external',x); y=x;", "f=str2func('external'); y=f(x);", "y=matfile('data.mat');",
                  "y=Simulink.SimulationInput('external');", "y=timer;", "onCleanup('bad'); y=x;", "y=handle(x);"]
        for body in bodies:
            with self.subTest(body=body):
                self.plant(body)
                with self.assertRaisesRegex(ContractError, "Unsupported MATLAB"):
                    self.review()

    def test_unknown_dependencies_bare_functions_and_object_constructors_require_thin_adapters(self):
        for body in ("y=external_model(x,u);", "y=external_value;", "y=SomeHandleClass(x);", "y=foreign.package(x);"):
            with self.subTest(body=body):
                self.plant(body)
                with self.assertRaisesRegex(ContractError, "Unknown/external MATLAB dependency"):
                    self.review()

    def test_function_handles_shell_escapes_and_variable_arity_are_unsupported(self):
        for body in ("f=@sin; y=f(x);", "!external_command\ny=x;"):
            with self.subTest(body=body):
                self.plant(body)
                with self.assertRaisesRegex(ContractError, "Unsupported MATLAB construct"):
                    self.review()
        self.source("source/plant.m", "function y=plant(x,u,d,dt,varargin)\ny=x;\nend\n")
        with self.assertRaisesRegex(ContractError, "fixed-interface"):
            self.review()

    def test_tampered_primary_or_helper_is_rejected_before_reviewing_its_new_code(self):
        self.plant()
        self.source("source/helper.m", "function y=helper(x)\ny=x;\nend\n")
        expected = self.hashes()
        self.source("source/helper.m", "this is not a function and must fail hash first")
        with self.assertRaisesRegex(ContractError, "source hash changed"):
            self.review(hashes=expected)
        self.source("source/helper.m", "function y=helper(x)\ny=x;\nend\n")
        self.source("source/plant.m", "system('MUST_NEVER_BE_EXECUTED')")
        with self.assertRaisesRegex(ContractError, "source hash changed"):
            self.review(hashes=expected)

    def test_unbound_m_files_opaque_dependencies_and_inventory_escape_are_rejected(self):
        self.plant(); expected = self.hashes()
        self.source("source/extra.m", "function y=extra(x)\ny=x;\nend\n")
        with self.assertRaisesRegex(ContractError, "inventory differs"):
            self.review(hashes=expected)
        (self.root / "source/extra.m").unlink()
        self.source("source/helper.mexw64", "opaque")
        with self.assertRaisesRegex(ContractError, "Opaque MEX"):
            self.review()
        with self.assertRaisesRegex(ContractError, "confined relative"):
            self.review(hashes={"../escape.m": "a" * 64})

    def test_primary_and_output_arity_and_filename_must_match_declared_fixed_interface(self):
        for source in ("function y=wrong(x,u,d,dt,p)\ny=x;\nend\n", "function y=plant(x,u,d,p)\ny=x;\nend\n"):
            with self.subTest(source=source):
                self.source("source/plant.m", source)
                with self.assertRaises(ContractError): self.review()
        self.plant()
        self.source("source/observe.m", "function y=observe(x)\ny=x;\nend\n")
        with self.assertRaisesRegex(ContractError, r"g\(x,parameters\)"):
            self.review(self.spec(output_entrypoint="observe"))

    def test_simulink_without_m_sources_reports_zero_functions_and_does_not_execute(self):
        self.source("source/model.slx", "fixture binary placeholder; static validator does not open SLX")
        report = self.review({"backend": "simulink", "source_file": "source/model.slx"})
        self.assertEqual(report["file_count"], 0)
        self.assertFalse(report["source_executed"])

    def test_unterminated_literals_and_block_comments_are_explicit_errors(self):
        for body in ("y='unterminated", "%{\ny=x;", 'y="unterminated'):
            with self.subTest(body=body):
                self.plant(body)
                with self.assertRaisesRegex(ContractError, "Unterminated MATLAB"):
                    self.review()


if __name__ == "__main__": unittest.main()
