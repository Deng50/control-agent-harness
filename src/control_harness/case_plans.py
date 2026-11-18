"""Editable engineering plans and request-bound confirmation verification.

Python's existing canonical_hash owns plan versions. Native user confirmation
belongs to Harness; request JSON can select a plan but cannot confer authority.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
import uuid

from .contracts import ContractError, canonical_hash
from .locking import kernel_mutex

SCOPES = ("signals", "sampling", "training")
FIXTURE_KIND = "control-case-plan-test-fixture-v1"


class PendingPlanConfirmation(ContractError):
    pass


def _root(workspace) -> Path:
    return Path(getattr(workspace, "path", workspace)).expanduser().resolve()


def canonical_payload(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _same_json(left, right) -> bool:
    # JSON transport normalizes -0/0 and integral floats, but never bool/number.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _document(workspace) -> dict:
    path = _root(workspace) / ".case-plans" / "plans.json"
    if not path.exists():
        return {"schema_version": 1, "plans": {}, "active_by_scope": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ContractError("Unsupported case plan schema")
    return data


def _write(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_payload(data))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_plan(workspace, plan_id: str | None = None, scope: str | None = None) -> dict:
    data = _document(workspace)
    if plan_id is None and scope is None:
        return copy.deepcopy(data)
    plan_id = plan_id or data["active_by_scope"].get(scope)
    if not plan_id or plan_id not in data["plans"]:
        raise PendingPlanConfirmation(f"No proposed {scope or 'requested'} plan; propose and review it first")
    return copy.deepcopy(data["plans"][plan_id])


def propose_plan(workspace, scope: str, payload: dict, bindings: dict | None = None,
                 dependencies: dict | None = None) -> dict:
    """Edit the active scope plan; unchanged content/version is idempotent."""
    if (scope not in SCOPES or not isinstance(payload, dict)
            or (bindings is not None and not isinstance(bindings, dict))
            or (dependencies is not None and not isinstance(dependencies, dict))):
        raise ContractError("Case plans require a signals/sampling/training scope and JSON object payload/bindings")
    directory = _root(workspace) / ".case-plans"
    directory.mkdir(parents=True, exist_ok=True)
    with kernel_mutex(directory / "plans.mutex"):
        data = _document(workspace)
        if dependencies is None:
            dependencies = {}
            for upstream in SCOPES[:SCOPES.index(scope)]:
                current = data["active_by_scope"].get(upstream)
                if current:
                    record = data["plans"][current]
                    dependencies[upstream] = {"plan_id": current, "version_hash": record["version_hash"]}
        for dependency, reference in dependencies.items():
            if dependency not in SCOPES[:SCOPES.index(scope)]:
                raise ContractError("Plan dependencies must refer to earlier engineering scopes")
            upstream = read_plan(workspace, reference.get("plan_id"))
            if upstream["scope"] != dependency or upstream["version_hash"] != reference.get("version_hash") or upstream["status"] == "stale":
                raise PendingPlanConfirmation("Plan dependency version is stale")
        envelope = {"scope": scope, "payload": payload, "bindings": bindings or {}, "dependencies": dependencies}
        text, version = canonical_payload(envelope), canonical_hash(envelope)
        old_id = data["active_by_scope"].get(scope)
        old = data["plans"].get(old_id)
        if old and old["version_hash"] == version and old["status"] != "stale":
            return copy.deepcopy(old)
        plan_id = old_id or f"plan-{uuid.uuid4().hex}"
        record = {"plan_id": plan_id, **copy.deepcopy(envelope), "canonical_payload_json": text,
                  "version_hash": version, "revision": (old or {}).get("revision", 0) + 1,
                  "status": "proposed", "updated_at": time.time()}
        data["plans"][plan_id] = record
        data["active_by_scope"][scope] = plan_id
        # A downstream summary is no longer what the user reviewed when an
        # upstream engineering scope changes, even if its own settings did not.
        for downstream in SCOPES[SCOPES.index(scope) + 1:]:
            dependent_id = data["active_by_scope"].get(downstream)
            if dependent_id:
                data["plans"][dependent_id]["status"] = "stale"
                data["plans"][dependent_id]["stale_reason"] = f"{scope} plan version changed"
        _write(directory / "plans.json", data)
        return copy.deepcopy(record)


def validate_plan(workspace, reference: dict, scope: str, *, expected_payload=None, bindings=None) -> dict:
    if not isinstance(reference, dict) or not isinstance(reference.get("plan_id"), str) or not isinstance(reference.get("version_hash"), str):
        raise PendingPlanConfirmation(f"Missing config.plan_refs.{scope}; propose and confirm the current plan")
    record = read_plan(workspace, reference.get("plan_id"))
    envelope = {key: record[key] for key in ("scope", "payload", "bindings", "dependencies")}
    if (record["scope"] != scope or record["version_hash"] != reference.get("version_hash")
            or record["status"] == "stale" or canonical_hash(envelope) != record["version_hash"]
            or canonical_payload(envelope) != record["canonical_payload_json"]):
        raise PendingPlanConfirmation(f"The {scope} plan version changed; review and confirm its current version")
    if expected_payload is not None and not _same_json(record["payload"], expected_payload):
        raise PendingPlanConfirmation(f"The {scope} request differs from the reviewed plan payload")
    if bindings is not None and not _same_json(record["bindings"], bindings):
        raise PendingPlanConfirmation(f"The {scope} material/data/dependency versions changed")
    for dependency, ref in record["dependencies"].items():
        active = read_plan(workspace, scope=dependency)
        if active["plan_id"] != ref["plan_id"] or active["version_hash"] != ref["version_hash"] or active["status"] == "stale":
            raise PendingPlanConfirmation(f"The upstream {dependency} plan changed")
    return record


def fixture_workspace(workspace) -> None:
    root = _root(workspace)
    try:
        marker = json.loads((root / ".case-plans" / "test-fixture.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError("Explicit test_fixture authority requires an isolated marked test-output workspace") from error
    if marker.get("kind") != FIXTURE_KIND or _root(marker.get("workspace", ".")) != root or "test-output" not in root.parts:
        raise ContractError("Explicit test_fixture authority requires an isolated marked test-output workspace")


def require_confirmation(context, request: dict, requirements: list[dict]) -> list[dict]:
    """Recheck the current plans and trusted Host envelope before numerical work."""
    if not requirements:
        return []
    workspace = _root(request["workspace"])
    plans = []
    for requirement in requirements:
        scope = requirement["scope"]
        plans.append(validate_plan(workspace, request.get("config", {}).get("plan_refs", {}).get(scope), scope,
            expected_payload=requirement.get("expected_payload", requirement.get("expectedPayload")), bindings=requirement.get("bindings")))
    if getattr(context, "_test_fixture_workspace", None) == workspace:
        fixture_workspace(workspace)
        context.case_confirmation_sources = [{"scope": plan["scope"], "plan_id": plan["plan_id"],
            "version_hash": plan["version_hash"], "confirmation_source": "test_fixture",
            "source": {"kind": "explicit_python_test_context"}} for plan in plans]
        return plans
    body_text = getattr(context, "_case_receipt_body", None)
    key_text = getattr(context, "_case_receipt_key", None)
    signature = getattr(context, "_case_receipt_signature", None)
    raw = getattr(context, "_case_raw_request", None)
    if not all(isinstance(value, str) and value for value in (body_text, key_text, signature, raw)):
        raise PendingPlanConfirmation("A native user confirmation receipt is required; request JSON flags are not authority")
    try:
        key = base64.b64decode(key_text, validate=True)
        expected = hmac.new(key, body_text.encode("utf-8"), hashlib.sha256).hexdigest()
        if len(key) != 32 or not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        body = json.loads(body_text)
        if (body.get("schema_version") != 1 or _root(body["workspace"]) != workspace
                or body.get("session_id") != getattr(context, "_case_native_session", None)
                or body.get("request_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()
                or not _same_json(json.loads(raw), request) or body.get("operation") != request["operation"]):
            raise ValueError("request binding")
        expected_refs = [{"scope": plan["scope"], "plan_id": plan["plan_id"], "version_hash": plan["version_hash"]} for plan in plans]
        if body.get("plans") != expected_refs:
            raise ValueError("plan versions")
        sources = body.get("confirmation_sources")
        if (not isinstance(sources, list) or len(sources) != len(plans)
                or any(not isinstance(source, dict) or not isinstance(source.get("source"), dict)
                       or source["source"].get("kind") != "native_user_command"
                       or source.get("confirmation_source") not in {"native_user", "test_fixture"} for source in sources)):
            raise ValueError("native confirmation sources")
    except (ValueError, KeyError, TypeError) as error:
        raise PendingPlanConfirmation("Host confirmation receipt does not match this request, Session or current plan versions") from error
    context.case_confirmation_sources = copy.deepcopy(sources)
    return plans
