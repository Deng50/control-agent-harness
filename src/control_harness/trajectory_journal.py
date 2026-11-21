"""按独立 reset 边界提交采集进度。只有 Host 恢复原作业时才会选择旧轨迹。"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import re

import numpy as np

from .artifacts import atomic_json, file_hash, read_json
from .contracts import ContractError, canonical_hash
from .locking import process_alive


class TrajectoryJournal:
    def __init__(self, stage, context, plan, dt):
        self.stage, self.context = stage, context
        self.directory = stage.root / "trajectory-commits"
        self.directory.mkdir()
        self.scope = {"workspace": str(stage.workspace.path), "project_id": stage.workspace.manifest["project_id"],
                      "session": context._case_native_session, "kind": stage.kind, "parents": stage.parents,
                      "interface": stage.interface.fingerprint, "plan_hash": plan["plan_hash"], "plant_dt": dt}
        self.key = base64.b64decode(context._case_receipt_key) if context._case_receipt_key else None
        self.source = None
        self.intent_digest = file_hash(stage.root / "stage.intent.json")
        self.source_intent_digest = None
        self.count = 0
        self.restored = 0
        selector = os.environ.get("CONTROL_COLLECTION_RESUME")
        if selector:
            self._open_source(json.loads(selector), len(plan["trajectories"]))

    def _seal(self, body):
        return hmac.new(self.key, canonical_hash(body).encode("ascii"), hashlib.sha256).hexdigest()

    def _open_source(self, selector, total):
        # 此选择仅来自作业历史；路径、工况和来源字节在 Python 中独立复核。
        stage_id = selector.get("stage_id", "")
        count = selector.get("completed")
        if (not self.key or not re.fullmatch(re.escape(self.stage.kind) + r"-[0-9a-f]{12}", stage_id)
                or type(count) is not int or not 1 <= count <= total):
            raise ContractError("Invalid trajectory resume selection")
        workspace = self.stage.workspace
        root = workspace.path / "failed" / stage_id
        if not root.exists():
            root = workspace.path / ".staging" / stage_id
        intent_path = root / "stage.intent.json"
        if intent_path.resolve() != intent_path:
            raise ContractError("Trajectory source redirects outside its original stage")
        intent_bytes = intent_path.read_bytes()
        intent = json.loads(intent_bytes)
        if (intent.get("stage_id") != stage_id or intent.get("workspace") != str(workspace.path)
                or intent.get("project_id") != self.scope["project_id"] or intent.get("dependencies") != self.stage.parents
                or stage_id in workspace.manifest["artifacts"] or (root / "artifact.json").exists()):
            raise ContractError("Trajectory source identity or publication state changed")
        if root.parent.name == ".staging":
            owner = intent["owner"]
            if owner.get("host") != platform.node() or process_alive(owner.get("pid")) is not False:
                raise ContractError("Previous trajectory producer has no verified exit")
        self.source, self.count = root, count
        self.source_intent_digest = hashlib.sha256(intent_bytes).hexdigest()

    def load(self, index, condition):
        """核验整轨的元数据、二进制字节及 reset 配置，半写文件没有提交记录。"""
        if index >= self.count:
            return None
        root = self.source / "trajectory-commits"
        descriptor, data = root / f"{index}.json", root / f"{index}.npz"
        if descriptor.resolve() != descriptor or data.resolve() != data:
            raise ContractError("Trajectory checkpoint contains a filesystem alias")
        record = read_json(descriptor)
        body = record["body"]
        if (not hmac.compare_digest(record["seal"], self._seal(body)) or body["scope"] != self.scope
                or body["stage_id"] != self.source.name or body["stage_intent_sha256"] != self.source_intent_digest
                or body["index"] != index
                or body["condition_hash"] != canonical_hash(condition)):
            raise ContractError("Trajectory checkpoint source, reset or plan changed")
        # 同一个已打开文件先散列再读取，防止两次打开间被替换。
        with data.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != body["sha256"]:
                raise ContractError("Trajectory checkpoint data hash changed")
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as archive:
                rows = {key: archive[key] for key in archive.files}
        self.restored += 1
        return rows, body["diagnostics"]

    def commit(self, index, condition, rows, diagnostics=None):
        """先写完数组，再原子提交独立描述。进度只在提交成功后发送。"""
        times = condition["t"][:-1] if "t" in condition else condition["time"]
        count = len(times)
        if (not count or any(len(values) != count for values in rows.values())
                or not np.array_equal(rows["time"], times)
                or not np.all(rows["trajectory_id"] == condition["trajectory_id"])):
            raise ContractError("Only one full independent trajectory can be committed")
        path = self.directory / f"{index}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **rows)
            stream.flush()
            os.fsync(stream.fileno())
        body = {"scope": self.scope, "stage_id": self.stage.id, "stage_intent_sha256": self.intent_digest, "index": index,
                "condition_hash": canonical_hash(condition), "reset": condition["x0"],
                "initial_action": condition.get("initial_action"), "rows": count,
                "sha256": file_hash(path), "diagnostics": diagnostics}
        atomic_json(path.with_suffix(".json"), {"body": body, "seal": self._seal(body) if self.key else None})
        if self.key:
            self.context.emit({"event": "progress", "stage": "trajectory_commit",
                               "collection_checkpoint": {"stage_id": self.stage.id, "completed": index + 1},
                               "restored_trajectories": self.restored})
