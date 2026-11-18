"""Verify a copied three-role delivery, then execute each portable API in python -I.

Uses only relative paths and explicit SHA256 values in delivery-manifest.json.
It neither imports control_harness nor trains, tunes, starts a server or runs MATLAB.
The caller prepares a new delivery directory and supplies fixed inference probes.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def member(root, name):
    assert isinstance(name, str) and name and "\\" not in name and ":" not in name
    parts = PurePosixPath(name)
    assert not parts.is_absolute() and all(p not in {"", ".", ".."} for p in name.split("/"))
    path = (root / name).resolve(strict=True)
    assert path.is_relative_to(root)
    return path


def same_or_under(path, root):
    return path == root or path.is_relative_to(root)


def child(request_path, output):
    assert sys.flags.isolated == 1, "Use a new isolated interpreter"
    request = read(request_path)
    root = Path(request["delivery_root"]).resolve(strict=True)
    bundle = member(root, request["case"]["bundle"])
    forbidden = [Path(value).resolve() for value in request["forbidden_roots"]]
    assert not any(same_or_under(root, value) for value in forbidden)
    counters = {"file_access_denied": 0, "unexpected_denials": 0, "harness_import_denied": 0}
    negative_control = False

    def audit(event, args):
        if event not in {"open", "os.listdir", "os.scandir", "os.chdir"} or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        target = Path(os.fsdecode(value)).resolve()
        if any(same_or_under(target, folder) for folder in forbidden):
            counters["file_access_denied"] += 1
            if not negative_control:
                counters["unexpected_denials"] += 1
            raise PermissionError("Independent delivery cannot read the original project/run")

    sys.addaudithook(audit)

    class BlockHarness(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "control_harness" or fullname.startswith("control_harness."):
                counters["harness_import_denied"] += 1
                raise ImportError("Portable delivery cannot import control_harness")

    sys.meta_path.insert(0, BlockHarness())
    sys.path[:] = [p for p in sys.path if p and not any(same_or_under(Path(p).resolve(), r) for r in forbidden)]
    negative_control = True
    try:
        try:
            (forbidden[0] / "__migration_negative_control_no_read__").open("rb")
        except PermissionError:
            pass
        else:
            raise AssertionError("File-access negative control was not blocked")
        try:
            __import__("control_harness")
        except ImportError:
            pass
        else:
            raise AssertionError("Project-import negative control was not blocked")
    finally:
        negative_control = False

    import numpy as np
    import torch
    torch.set_num_threads(1)

    def module(filename, name):
        source = member(bundle, filename)
        spec = importlib.util.spec_from_file_location(name, source)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        assert Path(value.__file__).resolve().is_relative_to(bundle)
        return value

    case = request["case"]
    point = case["probe"]["input"]
    kind = case["kind"]
    result = {"kind": kind, "artifact_id": case["artifact_id"], "isolated": True,
              "python": sys.version, "executable": sys.executable, "bundle": str(bundle),
              "torch_threads": torch.get_num_threads(), "closed_loop": "not_rerun"}
    if kind == "supervised_model":
        runtime = module("inference.py", "_delivery_supervised").load_bundle(bundle)
        values = np.asarray(point["raw_features"], dtype=np.float64)
        result["prediction"] = runtime.predict(values, feature_labels=point["feature_labels"]).tolist()
        result["api"] = "inference.load_bundle(...).predict(raw_features,feature_labels)"
    elif kind == "controller":
        runtime = module("controller.py", "_delivery_controller").load_design(bundle)
        action, diagnostics = runtime.solve(point["x"], point["reference"], point["disturbance"], point["previous"])
        result.update(action=np.asarray(action).tolist(), diagnostics=diagnostics,
                      api="controller.load_design(...).solve(x,reference,disturbance,previous)")
    elif kind == "external_optimized_policy":
        runtime = module("policy_runtime.py", "_delivery_policy").load_external_policy(bundle)
        runtime.reset()
        previous = np.asarray(point["previous"], dtype=np.float64)
        features = np.asarray(point["raw_features"], dtype=np.float64)
        projected = runtime.predict(features, previous, feature_labels=point["feature_labels"])
        action = runtime.act(torch.as_tensor(point["x"], dtype=torch.float64),
                             torch.as_tensor(point["reference"], dtype=torch.float64),
                             torch.as_tensor(point["disturbance"], dtype=torch.float64),
                             torch.as_tensor(previous, dtype=torch.float64)).detach().cpu().numpy()
        np.testing.assert_allclose(action, projected, atol=1e-12, rtol=1e-12)
        result.update(action=action.tolist(), projected=projected.tolist(),
                      api="policy_runtime.load_external_policy(...).predict and .act")
    else:
        raise ValueError("Unsupported portable kind: " + kind)
    errors = {}
    for key, expected in case["probe"]["expected"].items():
        actual = np.asarray(result[key], dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=case["probe"]["atol"], rtol=case["probe"]["rtol"])
        errors[key] = float(np.max(np.abs(actual - expected)))
    assert counters == {"file_access_denied": 1, "unexpected_denials": 0, "harness_import_denied": 1}
    assert not any(name == "control_harness" or name.startswith("control_harness.") for name in sys.modules)
    result.update(status="PASS", maximum_absolute_error=errors, access_guards=counters,
                  training=False, MATLAB=False, native_RPC=False,
                  module_files={name: value.__file__ for name, value in list(sys.modules.items())
                                if name.startswith("_delivery_") and getattr(value, "__file__", None)})
    write_new(output, result)


def verify(manifest_path, output, python):
    root = manifest_path.resolve(strict=True).parent
    manifest = read(manifest_path)
    assert manifest["schema_version"] == 1 and manifest["kind"] == "control-engineering-delivery-v1"
    cases = manifest["portable_cases"]
    assert len(cases) == 3 and {v["role"] for v in cases} == {"model", "controller", "lightweight"}
    assert len({v["artifact_id"] for v in cases}) == 3
    forbidden = [str(Path(value).resolve()) for value in manifest["forbidden_runtime_roots"]]
    assert forbidden and not any(same_or_under(root, Path(value)) for value in forbidden)
    proof = {"schema_version": 1, "status": "RUNNING", "started_at": time.time(),
             "manifest_sha256": sha(manifest_path), "source_sha256": sha(Path(__file__)),
             "delivery_root": str(root), "runtime": str(python), "cases": [],
             "scope": "Fresh local directory and isolated portable inference; audited local runtime reused, not a new-machine installation"}
    try:
        before = {}
        for name, digest in manifest["files"].items():
            path = member(root, name)
            assert len(digest) == 64 and sha(path) == digest, "Delivery member changed: " + name
            before[name] = digest
        for case in cases:
            directory = member(root, case["bundle"])
            bundle_manifest = read(member(directory, case["bundle_manifest"]))
            assert bundle_manifest["files"] == case["payload_sha256"]
            for name, digest in case["payload_sha256"].items():
                assert sha(member(directory, name)) == digest
            prefix = root / ("migration-" + case["role"])
            request = Path(str(prefix) + "-input.json")
            result_path = Path(str(prefix) + "-result.json")
            write_new(request, {"delivery_root": str(root), "forbidden_roots": forbidden, "case": case})
            env = {k: v for k, v in os.environ.items() if k.upper() not in {"PYTHONPATH", "CONTROL_CODE_ROOT", "CONTROL_PYTHON"}}
            env["PYTHONNOUSERSITE"] = "1"
            cmd = [str(python), "-I", str(Path(__file__).resolve()), "--probe", str(request), "--output", str(result_path)]
            started = time.time()
            completed = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=180, check=False)
            stdout = Path(str(prefix) + ".stdout.log")
            stderr = Path(str(prefix) + ".stderr.log")
            for path, text in [(stdout, completed.stdout), (stderr, completed.stderr)]:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text)
            row = {"role": case["role"], "artifact_id": case["artifact_id"], "command": cmd,
                   "exit_code": completed.returncode, "duration_seconds": time.time() - started,
                   "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr)}
            proof["cases"].append(row)
            assert completed.returncode == 0, "Inspect the original migration process logs; do not silently rerun"
            result = read(result_path)
            assert result["status"] == "PASS" and result["isolated"] is True
            row.update(result=result, result_sha256=sha(result_path))
        assert before == {name: sha(member(root, name)) for name in before}
        proof.update(status="PASS", verified_files=len(before), payload_unchanged_after_inference=True,
                     original_project_reads=0, project_imports=0, training=False, MATLAB=False,
                     original_closed_loop="not_rerun", content_integrity="SHA256, not a digital signature")
    except Exception as error:
        proof.update(status="FAIL", error={"type": type(error).__name__, "message": str(error),
                                           "traceback": traceback.format_exc()})
    proof["finished_at"] = time.time()
    write_new(output, proof)
    print(json.dumps({k: proof.get(k) for k in ["status", "verified_files", "delivery_root"]}))
    return 0 if proof["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--probe", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    assert not args.output.exists(), "Never overwrite an earlier result or repeat an uncertain process"
    if args.probe:
        child(args.probe, args.output)
        return 0
    return verify(args.manifest, args.output, args.python.resolve(strict=True))


if __name__ == "__main__":
    sys.exit(main())
