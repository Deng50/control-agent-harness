"""Host 与数值进程共用的输入种类表。每次执行均核对当前版本及物理接口。"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import ContractError, ModelInterface

_SPEC = json.loads(Path(__file__).with_name("stage_contracts.json").read_text(encoding="utf-8"))
STAGE_CONTRACTS = _SPEC["stages"]


def accepted(kind):
    """普通字符串标记文件资源，其余产物类型展开为集合供工作流规划器比较。"""
    if isinstance(kind, list):
        return set(kind)
    if kind.startswith("@"):
        return set(_SPEC["groups"][kind[1:]])
    return kind if kind in ("file", "artifact_directory") else {kind}


def check_inputs(workspace, operation, inputs, config):
    """持作业锁重查不可变依赖。未通过构造、单位或绑定检查时不进入数值算法。"""
    stage = STAGE_CONTRACTS.get(operation)
    if stage is None:
        return
    metadata = {}
    selected_closures = {}
    for name, kind in {**stage["inputs"], **stage.get("optional", {})}.items():
        kinds = accepted(kind)
        if not isinstance(kinds, set):
            continue
        value = inputs.get(name)
        if not value:
            if name in stage["inputs"]:
                raise ContractError(f"clarify inputs.{name}: 请选择该步骤所需产物")
            continue
        try:
            closure = workspace.resolve_closure(value)
            _, meta = closure[value]
        except (OSError, ValueError, KeyError) as exc:
            raise ContractError(f"stale_dependency inputs.{name}: {exc}") from exc
        if meta["kind"] not in kinds:
            raise ContractError(f"unsupported inputs.{name}: expected {sorted(kinds)}, received {meta['kind']}")
        missing = [file for file in _SPEC["payloads"].get(meta["kind"], []) if file not in meta["files"]]
        if missing:
            raise ContractError(f"unsupported inputs.{name}: 缺少已登记的载荷文件 {missing}")
        if meta["kind"] in {"controller", "plant_model", "controller_design"} | accepted("@predictor"):
            from .inspection import artifact_capabilities
            info = artifact_capabilities(workspace, value, _closure=closure)
            capability = {"controller": "executable_controller", "plant_model": "sampling",
                          "controller_design": "controller_design"}.get(meta["kind"], "prediction")
            if not info["capabilities"][capability]:
                raise ContractError(f"unsupported inputs.{name}: 缺少 {capability} 能力，需核对描述文件和来源")
        metadata[name] = meta
        selected_closures[name] = closure

    physical = [(name, ModelInterface.parse(meta["interface"])) for name, meta in metadata.items()
                if "states" in meta["interface"]]
    for name, interface in physical[1:]:
        try:
            physical[0][1].compatible(interface)
        except ContractError as exc:
            raise ContractError(f"unsupported inputs.{name}.interface: {exc}") from exc
    if operation == "collect_expert" and metadata["evaluation_model"]["kind"] == "plant_model":
        # 原对象回放沿控制器记录的预测器取来源，防止同尺寸的另一个对象被当作原对象。
        from .plants import controller_callable_source
        original = controller_callable_source(inputs["controller"], selected_closures["controller"])
        if original != inputs["evaluation_model"]:
            raise ContractError("unsupported inputs.evaluation_model: 请选择预测器来源闭包中的同一 python_callable 原对象")
    if operation in {"convert_plant_predictor", "validate_differentiable_model"}:
        source = inputs["plant" if operation == "convert_plant_predictor" else "evaluation_model"]
        data = metadata["dataset"]
        if data["source"].get("model") != source or source not in {p["id"] for p in data["parents"]}:
            raise ContractError("unsupported inputs.dataset: 数据必须来自所选原对象的同一版本")
    if operation == "convert_plant_predictor" and config.get("method") == "declared_equations" and not inputs.get("equations"):
        raise ContractError("clarify inputs.equations: 请登记明确状态、单位和采样周期的方程模型")
    if operation in {"train_external_imitation", "train_supervised", "train_predictive_model"}:
        data = metadata["dataset"]["interface"]
        if operation == "train_predictive_model":
            widths = (sum(len(data[k]) for k in ("states", "controls", "disturbances")), len(data["states"]))
        else:
            widths = (data.get("input_dim"), data.get("output_dim"))
        network = config if operation == "train_supervised" else config.get("network", {})
        for key, width in zip(("input_dim", "output_dim"), widths):
            if width is not None and key in network and network[key] != width:
                raise ContractError(f"unsupported config.{key}: 已登记特征需要 {width}，收到 {network[key]}")
