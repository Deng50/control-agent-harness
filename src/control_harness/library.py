"""Read-only access to the bundled control engineering reference library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
import unicodedata

from .library_routes import ALIASES, CONCEPTS, SCENES


LIBRARY = Path(__file__).resolve().parents[2] / "harness-plugin" / "control-skill-library"
ROLES = {
    "train-object-model": "model",
    "design-and-tune-mpc": "controller",
    "distill-controller": "lightweight",
}
RESOURCE_FIELDS = ("instructions", "knowledge", "case", "code", "evidence")
# The source library's .ts files are serialized TorchScript models.
TEXT_FORMATS = {".md", ".json", ".csv", ".m", ".py", ".txt", ".mjs"}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_QUERY = 4096


class LibraryError(ValueError):
    """A missing, altered or unsupported bundled resource."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[\s_-]+", " ", text).strip()
    # Often mixed in engineering requests, e.g. "一阶 MPC" and "MATLAB 预测矩阵".
    return re.sub(r"(?<=[\u3400-\u9fff]) | (?=[\u3400-\u9fff])", "", text)


def _matches(query: str, term: str) -> bool:
    value = _normalized(term)
    # Keep short Latin terms such as MPC, CSV and Actor out of unrelated words.
    left = r"(?<![a-z0-9])" if re.match(r"[a-z0-9]", value) else ""
    right = r"(?![a-z0-9])" if re.search(r"[a-z0-9]$", value) else ""
    return bool(value and re.search(left + re.escape(value) + right, query))


def _public_path(value: str) -> bool:
    parts = value.split("/")
    return (bool(value) and not any(char in value for char in ("\\", ":", "\x00"))
            and not any(part in {"", ".", ".."} for part in parts)
            and parts[0] != "evaluation"
            and parts[:2] not in (["runs", "paired"], ["runs", "valve-evaluation"]))


class ControlLibrary:
    def __init__(self, root: Path = LIBRARY):
        self.root = root.resolve(strict=True)
        manifest = (self.root / "snapshot.json").read_bytes()
        snapshot = json.loads(manifest)
        if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("files"), dict):
            raise LibraryError("Unsupported control library snapshot format")
        self.files = snapshot["files"]
        self.snapshot_sha256 = _digest(manifest)

    def _resource_path(self, name: str) -> Path:
        if not _public_path(name) or name not in self.files:
            raise LibraryError("Unknown library resource. Use a path returned by search or read.")
        path = self.root.joinpath(*name.split("/")).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise LibraryError(f"Invalid bundled file: {name}")
        return path

    def _text(self, name: str) -> str:
        path = self._resource_path(name)
        if path.suffix.casefold() not in TEXT_FORMATS:
            raise LibraryError(f"Binary asset. Open its case description for the model interface: {name}")
        metadata = self.files[name]
        expected_size = metadata["bytes"]
        if path.stat().st_size != expected_size or expected_size > MAX_FILE_BYTES:
            raise LibraryError(f"Resource size differs from snapshot or exceeds the text limit: {name}")
        with path.open("rb") as stream:
            data = stream.read(expected_size + 1)
        if len(data) != expected_size or _digest(data) != metadata["sha256"]:
            raise LibraryError(f"Resource checksum differs from snapshot: {name}")
        return data.decode("utf-8-sig")

    def _routes(self) -> list[dict]:
        routes = json.loads(self._text("index.json"))
        # Reuse the source skill's own role instead of assigning authority by scene.
        roles_by_skill = {row["skill"]: row["role"] for row in routes}
        for scene, skill, chapter, code, report, terms in SCENES:
            directory = f"cases/matlab-scenes/{scene}"
            routes.append({
                "skill": skill, "role": roles_by_skill[skill], "terms": list(terms),
                "instructions": f".agents/skills/{skill}/SKILL.md",
                "knowledge": f"knowledge/{chapter}.md", "case": f"{directory}/README.md",
                "code": f"{directory}/{code}" if code else "cases/matlab-scenes/run_scene.m",
                "evidence": f"runs/matlab-scenes/{scene}/fixed/{report}.json",
            })
        return routes

    def search(self, query: str, role: str | None = None, limit: int = 3) -> dict:
        if not query.strip() or len(query) > MAX_QUERY:
            raise LibraryError(f"Query must contain 1 to {MAX_QUERY} characters")
        if not 1 <= limit <= 8 or (role is not None and role not in ROLES.values()):
            raise LibraryError("Use a known role and a limit between 1 and 8")
        normalized = _normalized(query)
        matlab = any(_matches(normalized, word) for word in ("matlab", "simulink", "slx"))
        python = _matches(normalized, "python")
        ranked = []
        for row in self._routes():
            native_role = ROLES[row["role"]]
            if role and role != native_role:
                continue
            code, skill = row["code"], row["skill"]
            terms = list(row["terms"]) + list(ALIASES.get(code, ()))
            concepts = CONCEPTS.get(code, ())
            # A role/skill identifier selects its general entry. Case-specific
            # variants sharing that identifier must match the task or language.
            scene_specific = row["case"].startswith("cases/matlab-scenes/")
            if not scene_specific:
                terms += [skill, *ALIASES.get(skill, ())]
                concepts = concepts or CONCEPTS.get(skill, ())
            if matlab and skill == "verify-mpc-prediction" and code.endswith(".m"):
                terms += ["预测矩阵", "增量核验", "增量预测", "prediction matrix", "control increments", "increment matrix", skill]
                concepts = CONCEPTS["controlkit/prediction.py"]
            if matlab and code == "cases/native-mpc/run_native_mpc.m":
                terms.append("MPC")
            hits = [term for term in dict.fromkeys(terms) if _matches(normalized, term)]
            specificity = max((len(_normalized(term)) for term in hits), default=0)
            clues = [next((term for term in group if _matches(normalized, term)), "") for group in concepts]
            combined = bool(clues) and all(clues)
            if combined:
                hits += [clue for clue in clues if _normalized(clue) not in {_normalized(term) for term in hits}]
                specificity = max(specificity, sum(len(_normalized(term)) for term in clues))
            if not hits:
                continue
            # Strongest specific phrase first. A requested ecosystem breaks ties
            # between equivalent implementations (e.g. either prediction checker).
            platform = int((matlab and code.endswith(".m")) or (python and code.endswith(".py")))
            plant_case = combined and code.startswith("cases/")
            key = (-specificity, -platform, -int(plant_case), -len(hits), skill, row["case"], code)
            ranked.append((key, row, native_role, hits))

        matches = []
        for _, row, native_role, hits in sorted(ranked, key=lambda item: item[0])[:limit]:
            for name in RESOURCE_FIELDS:
                self._resource_path(row[name])
            identity = "\n".join(row[k] for k in ("skill", "case", "code")).encode("utf-8")
            matches.append({**row, "asset_id": _digest(identity)[:16], "role": native_role,
                            "matched_terms": hits, "evidence_scope": "historical_reference_run"})
        return {"query": query, "role": role, "matches": matches,
                "snapshot_sha256": self.snapshot_sha256}

    def read(self, path: str, offset: int = 0, length: int = 4000) -> dict:
        if offset < 0 or not 1 <= length <= 12000:
            raise LibraryError("Use a nonnegative character offset and length between 1 and 12000")
        content = self._text(path)
        if offset > len(content):
            raise LibraryError(f"Offset exceeds document length of {len(content)} characters")
        excerpt = content[offset:offset + length]
        end = offset + len(excerpt)
        links = []
        if path.endswith(".md"):
            for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", excerpt):
                relative = target.partition("#")[0]
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), relative))
                if relative and _public_path(resolved) and resolved in self.files and resolved not in links:
                    links.append(resolved)
        return {"path": path, "sha256": self.files[path]["sha256"], "snapshot_sha256": self.snapshot_sha256,
                "source_scope": "historical_reference_run" if path.startswith("runs/") else "bundled_reference",
                "offset": offset, "next_offset": end if end < len(content) else None,
                "total_chars": len(content), "text": excerpt,
                "links": [{"path": ref, "readable": PurePosixPath(ref).suffix.lower() in TEXT_FORMATS}
                          for ref in links[:24]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--query")
    action.add_argument("--read", metavar="PATH")
    parser.add_argument("--role", choices=sorted(ROLES.values()))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=4000)
    args = parser.parse_args()
    try:
        library = ControlLibrary()
        result = (library.search(args.query, args.role, args.limit) if args.query is not None
                  else library.read(args.read, args.offset, args.length))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Control library: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
