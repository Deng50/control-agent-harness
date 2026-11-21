"""将实际生成的数据及其父材料绑定到一次已通过原生确认的准备作业。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

from .artifacts import file_hash
from .case_plans import _same_json, canonical_payload, read_plan
from .contracts import ContractError

FILES = ("supervised.npz", "dataset.json", "scaler.json", "split.json", "config.json")
KIND = "control-supervised-preparation-v2"


def _key(context):
    value = getattr(context, "_case_receipt_key", None)
    if not value:
        return None
    key = base64.b64decode(value, validate=True)
    if len(key) != 32:
        raise ContractError("Invalid preparation proof authority")
    return key


def sign_preparation(stage, context, material, preparation):
    """在 publish 前签实际文件、当前原料版本和用户确认过的工程配置。"""
    plan = getattr(context, "_verified_preparation_plan", None)
    if not plan:
        return None
    if (not _same_json(plan["bindings"], {"material": material})
            or not _same_json(plan["payload"], {"operation": "prepare_supervised_dataset", "config": preparation})):
        raise ContractError("Prepared material or config differs from the admitted engineering review")
    body = canonical_payload({"kind": KIND, "workspace": str(stage.workspace.path),
        "session_id": context._case_native_session, "intent_id": plan["intent_id"],
        "signals": {k: plan[k] for k in ("plan_id", "version_hash")}, "dataset_id": stage.id,
        "material": material, "config": preparation, "interface": stage.interface.to_dict(),
        "files": {name: file_hash(stage.root / name) for name in FILES}})
    return {"body": body, "signature": hmac.new(_key(context), body.encode("utf-8"), hashlib.sha256).hexdigest()}


def verified_preparation(workspace, dataset, request, context):
    """缺输入来源或跨 Session/任务仅能按准确数据集版本确认。损坏的本地凭证拒绝执行。"""
    proof = dataset.get("source", {}).get("preparation_proof")
    key = _key(context)
    if proof is None or key is None:
        return None
    # 外部工作区的凭证没有本地授权能力。按准确数据集版本走独立确认。
    try:
        body = json.loads(proof["body"])
        if body["workspace"] != str(workspace.path):
            return None
        signature = hmac.new(key, proof["body"].encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(proof["signature"], signature) or body["kind"] != KIND:
            raise ValueError("signature")
        material = body["material"]
        parent_id = material["id"]
        _, parent = workspace.resolve(parent_id, kind="material")
        version = workspace.manifest["artifacts"][parent_id]["metadata_hash"]
        content = parent["source"]["content_hash"]
        if (body["dataset_id"] != dataset["id"] or dataset["kind"] != "supervised_dataset"
                or dataset["parents"] != [{"id": parent_id, "metadata_hash": version}]
                or dataset["source"].get("material_artifact") != parent_id
                or not content or dataset["source"].get("content_hash") != content
                or material != {"id": parent_id, "metadata_hash": version, "content_hash": content}
                or not _same_json(body["interface"], dataset["interface"])
                or body["files"] != {name: dataset["files"][name] for name in FILES}):
            raise ValueError("dataset or material changed")
        # 固定 config.json 的 SHA，并校验其中实际审批的接口及采数配置。
        from .case_requirements import _config
        from .contracts import DataInterface
        path = workspace.path / workspace.manifest["artifacts"][dataset["id"]]["path"] / "config.json"
        raw = path.read_bytes()
        prep = json.loads(raw)
        if (hashlib.sha256(raw).hexdigest() != body["files"]["config.json"]
                or not _same_json(_config(prep), body["config"])
                or not _same_json(DataInterface.parse(prep["interface"]).to_dict(), body["interface"])
                or not _same_json(prep.get("plan_refs", {}).get("signals"), body["signals"])):
            raise ValueError("preparation config changed")
        signals = request.get("config", {}).get("plan_refs", {}).get("signals")
        if context._case_native_session != body["session_id"] or not _same_json(signals, body["signals"]):
            return None
        plan = read_plan(workspace, scope="signals", intent_id=body["intent_id"])
        if (plan["plan_id"], plan["version_hash"]) != (signals["plan_id"], signals["version_hash"]):
            return None
    except (TypeError, KeyError, ValueError, OSError) as exc:
        raise ContractError("The signed dataset preparation proof does not match its verified inputs and files") from exc
    return body
