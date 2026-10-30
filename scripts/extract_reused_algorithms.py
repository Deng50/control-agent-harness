"""Copy only numerical classes/operators, with exact source hashes for audit."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT.parent
DEST = ROOT / "src/control_harness/reused"
DEST.mkdir(parents=True, exist_ok=True)
entries = []


def extract(relative, class_name, output, replacement_base, methods=None):
    path = SOURCES / relative
    text = path.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.decorator_list = []
    node.bases = [ast.Name(id=replacement_base, ctx=ast.Load())] if replacement_base else []
    node.body = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and (methods is None or n.name in methods)]
    if class_name == "ParameterTuningSkill":
        node.name = "GeneticOperators"
    code = ast.unparse(node).replace("ParameterTuningSkill._fitness", "GeneticOperators._fitness")
    header = ('"""Extracted numerical code. See reuse-manifest.json; original registries are excluded."""\n'
              'from __future__ import annotations\nimport math\nimport random\nfrom typing import Sequence\n'
              'import torch\nfrom torch import nn\n\n')
    (DEST / output).write_text(header + code + "\n", encoding="utf-8")
    entries.append({"source": path.relative_to(SOURCES).as_posix(), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_class": class_name, "source_line": node.lineno,
                    "output": f"src/control_harness/reused/{output}",
                    "adaptation": "AST extraction; remove application registries; use nn.Module; preserve method bodies"})


for name in ("mlp", "lstm", "rnn"):
    extract(f"NMPC-light/plugins/networks/{name}.py", name.upper() + "Network", name + ".py", "nn.Module")
extract("AIagent/pgac_auto_train_agent_v1/kernels/pgac_core.py", "PerceptionNet", "perception.py", "nn.Module")
# Remove legacy globals in defaults: dimensions are always explicit in the adapter.
perception = DEST / "perception.py"
perception.write_text(perception.read_text(encoding="utf-8").replace("x_dim: int=STATE_DIM", "x_dim: int")
                     .replace("u_dim: int=ACTION_DIM", "u_dim: int"), encoding="utf-8")
extract("mpc_agent_v5/app/skills/tuning.py", "ParameterTuningSkill", "genetic.py", None,
        {"_fitness", "_initial_population", "_tournament", "_crossover", "_mutate"})
(DEST / "__init__.py").write_text('"""Source-tracked numerical components extracted from the user repositories."""\n', encoding="utf-8")
(ROOT / "reuse-manifest.json").write_text(json.dumps({"schema_version": 1, "entries": entries},
                                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
