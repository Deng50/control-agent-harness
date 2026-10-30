# Model source inspection without execution

`inspect_model_source(path, context)` returns a read-only report containing source path/SHA-256, evidence locations, missing fields, uncertainty and any complete editable model candidates. It does not import or execute uploaded Python, start MATLAB, modify the source or register an artifact.

## Supported material

| Source | Inspected evidence | Boundary |
| --- | --- | --- |
| `.json` | Explicit declarative model schema, ordered interface, units, ranges, sample time and A/B/C/E/bias/terms dimensions | Shape/finite/interface validation, not physical fidelity certification |
| `.py` | AST classes, callable signatures and source lines; finite literal constants and `ENV_SPEC`, `MANIFEST`, `MODEL_SPEC`, `DYNAMICS_SPEC` | No import, `exec`, function invocation, decorators or top-level statements are executed; names such as x/u/NX alone do not establish engineering semantics |
| `.slx` | ZIP-contained Simulink XML block declarations and explicitly labelled Inport/Outport blocks | No compilation, dimension propagation, linked-library resolution, model simulation or inferred state semantics |
| `.mdl` | Recognized text header and read-only source identity | Native interface resolution requires a future MATLAB adapter |

Explicit Python dictionaries can directly declare the standard model spec, or contain `model`, `model_spec`, or `dynamics_spec`. The inspector validates those candidates with the same interface and declarative-dynamics contracts used by execution. Manifests depending on calls, other names, NumPy constructors or arbitrary computation remain uncertain. The tool does not manufacture a replacement using guessed units, sample times or signal roles.

A single valid explicit model candidate sets `registration_ready=true` and provides `editable_model_spec`. Multiple valid models require an explicit selection. This readiness means the supplied declaration satisfies the numeric schema, not that a Python function is proven equivalent to that declaration. Declarative equations use `register_model`. A standalone `.py` with an explicit `python_torch_transition` or `python_torch_derivative` candidate sets `registration_operation=register_python_model`; explicit registration executes and freezes the reviewed source, then checks its numeric and gradient contract. Inspection itself never executes or publishes it. See [Python dynamics](python-dynamics.md) for the supported callable boundary and execution permissions.

MATLAB discovery checks only configured executable files, PATH and standard installation directories. The report always identifies license/product status as unverified and `matlab_started=false`. Finding `matlab.exe` is not evidence of an available MATLAB/Simulink license or a working model. The current module does not provide native Simulink execution.

Source size, AST nodes, ZIP entries and XML payloads are bounded. SLX files are inspected in memory, without extraction; XML DTD/entity declarations are rejected. Common credential-named constants are redacted from reports.

## Dispatcher integration

Suggested read-only operation:

```json
{
  "operation": "inspect_model_source",
  "workspace": "D:/projects/control-agent-harness/workspaces/example",
  "inputs": {"model_path": "D:/projects/example/plant.py"}
}
```

The dispatcher should return `inspect_model_source(inputs["model_path"], context)` directly. Its result is a report, not an artifact ID, because incomplete model materials cannot satisfy an executable artifact interface.

## Validation

`tests/test_model_inspection.py` verifies JSON schema extraction and incomplete interfaces, source immutability, malicious/top-level Python not being executed, dynamic/ambiguous manifests remaining unresolved, output credential redaction, SLX XML boundaries, absent license claims, invalid sources and cancellation.
