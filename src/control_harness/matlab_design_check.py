"""通过专用 MCP 的固定批处理对项目导出的 MPC 做一次数值检查。"""
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from .artifacts import atomic_json, file_hash, read_json
from .contracts import ContractError, MaterialInterface
from .design_export import _MATLAB_CONTROL, _MATLAB_COST, _MATLAB_MODEL, _MATLAB_SELFCHECK, _MATLAB_STEP

# 此表既是允许执行的文件名单，又是本机发行版本的实际源代码。
# 工作区 manifest 只能证明文件版本，不能把用户写的任意 .m 加入此表。
_SOURCES = {"mpc_control.m": _MATLAB_CONTROL, "mpc_cost.m": _MATLAB_COST,
            "prediction_model.m": _MATLAB_MODEL, "predict_step.m": _MATLAB_STEP,
            "selfcheck_controller.m": _MATLAB_SELFCHECK}


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(config, design, model):
    """在写出交换文件前拒绝无界的求解规模、未知字段和不匹配的物理接口。"""
    point = config.get("case")
    if not isinstance(point, dict) or set(point) - {"x", "reference", "disturbance", "previous"} or not {"x", "reference"} <= set(point):
        raise ContractError("单点检查需要一个显式状态 x 和目标 reference。")
    if model.get("kind") == "affine":
        nx, nu, ny, nd = (model[key] for key in ("nx", "nu", "ny", "nd"))
    elif model.get("template") == "spring_damper":
        nx, nu, ny, nd = 2, 1, 1, 0
    else:
        raise ContractError("此设计没有可供 MATLAB 单点检查的仿射预测模型，请使用适用的导出接口。")
    if not all(type(v) is int and 0 < v <= 256 for v in (nx, nu, ny)) or type(nd) is not int or not 0 <= nd <= 256:
        raise ContractError("单点检查的模型维度超出支持范围。")
    horizon = design["controller_config"]["horizon"]
    if type(horizon) is not int or not 1 <= horizon <= 100 or horizon * nu > 256 or (horizon * nu) * (2 * horizon * (nx+ny+nu)) > 5_000_000:
        raise ContractError("单点检查规模超出既有稠密求解预算。")
    if nd and "disturbance" not in point:
        raise ContractError("该设计要求提供扰动向量 disturbance。")
    for name, width in (("x", nx), ("reference", ny), ("disturbance", nd), ("previous", nu)):
        if name in point and (not isinstance(point[name], list) or len(point[name]) != width or not all(map(_finite, point[name]))):
            raise ContractError(f"单点检查 {name} 应是长度为 {width} 的有限数值向量。")
    return point


def check_design(workspace, design_id, config, context):
    # 同样的请求和业务确认贯穿准备与归档两个沙箱进程。此入口不启动 MATLAB。
    phase = os.environ.get("CONTROL_MATLAB_PHASE")
    raw = getattr(context, "_case_raw_request", None)
    request_hash = os.environ.get("CONTROL_MATLAB_REQUEST_SHA256")
    nonce = os.environ.get("CONTROL_MATLAB_NONCE", "")
    if not getattr(context, "_case_receipt_body", None) or phase not in {"prepare", "finalize"} or not raw or _sha(raw) != request_hash:
        raise ContractError("单点检查需要已确认的原生 MATLAB 作业。")
    if str(uuid.UUID(nonce)) != nonce:
        raise ContractError("交换目录标识无效。")
    parent = workspace.path / ".case-plans" / "matlab-exchanges"
    exchange = parent / nonce
    if exchange.resolve() != exchange or Path(os.environ["CONTROL_MATLAB_EXCHANGE_DIR"]).resolve() != exchange:
        raise ContractError("MATLAB 单点检查交换目录不在当前工作区。")
    root, metadata = workspace.resolve(design_id, kind="controller_design")
    snapshot = exchange / "bundle"
    source = root / "bundle"
    case = _point(config, read_json(source / "design.json"), read_json(source / "matlab_model.json"))
    plan_hash = _sha(json.dumps({"case": case, "parameters": config.get("parameters", {})}, sort_keys=True, allow_nan=False))
    spec_hash = workspace.manifest["artifacts"][design_id]["metadata_hash"]
    manifest_path, result_path = exchange / "manifest.json", exchange / "result.json"
    bindings = {"nonce": nonce, "request_sha256": request_hash, "spec_hash": spec_hash}
    if phase == "prepare":
        snapshot.mkdir(parents=True, exist_ok=False)
        for name, text in _SOURCES.items():
            path = source / name
            if "bundle/"+name not in metadata["files"] or path.read_text(encoding="utf-8") != text:
                raise ContractError(f"设计包 {name} 与已安装版本的固定求解函数不一致。")
            (snapshot/name).write_text(text, encoding="utf-8")
        for name in ("design.json", "matlab_model.json"):
            data = (source/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != metadata["files"].get("bundle/"+name):
                raise ContractError(f"设计数据 {name} 的内容已变化。")
            (snapshot/name).write_bytes(data)
        atomic_json(snapshot/"case.json", case)
        atomic_json(snapshot/"parameters.json", config.get("parameters", {}))
        manifest = {**bindings, "kind": "control-matlab-design-check-v1", "schema_version": 1,
                    "members": [{"name": p.name, "sha256": file_hash(p)} for p in sorted(snapshot.iterdir())],
                    "result_path": str(result_path), "plan": {"plan_hash": plan_hash}}
        workspace.resolve(design_id, kind="controller_design")
        atomic_json(manifest_path, manifest)
        return {**bindings, "matlab_phase": "prepared", "batch_kind": "controller_design", "status": "verified",
                "manifest_path": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
                "exchange_dir": str(exchange), "result_path": str(result_path), "plan_hash": plan_hash}
    # 不读取调用方指定的任意文件，只接受该请求、该目录、该批次返回的数值结果。
    if result_path.resolve() != result_path or Path(os.environ["CONTROL_MATLAB_RESULT_PATH"]).resolve() != result_path:
        raise ContractError("单点检查结果路径与请求不一致。")
    data = result_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != os.environ.get("CONTROL_MATLAB_RESULT_SHA256"):
        raise ContractError("MATLAB 单点检查结果发生变化。")
    result = json.loads(data)
    if any(result.get(k) != v for k, v in {**bindings, "plan_hash": plan_hash}.items()):
        raise ContractError("MATLAB 单点检查结果不属于本次作业。")
    report = result["report"]
    if report.get("wrapper_nonce") != nonce or report.get("closed_loop") != "not_run" or report.get("ga_trials") != 0:
        raise ContractError("MATLAB 固定求解入口未返回匹配的报告。")
    solved = report.get("status") == "single_point_solved"
    if solved:
        action = report.get("action")
        values = action if isinstance(action, list) else [action]
        expected_nu = read_json(source/"matlab_model.json").get("nu", 1)
        if len(values) != expected_nu or not all(map(_finite, values)) or not _finite(report.get("cost")) or report.get("solve", {}).get("exitflag", 0) <= 0:
            raise ContractError("MATLAB 报告没有有效的控制量、代价与求解状态。")
    elif report.get("status") not in {"blocked_toolbox_license", "not_instantiated"}:
        raise ContractError("MATLAB 单点检查结果缺少可识别的求解状态。")
    with workspace.stage("matlab_design_check", MaterialInterface("json", "MATLAB 单点 MPC 检查"), [design_id], config, compatible_parents=False) as stage:
        atomic_json(stage.root/"report.json", {**report, "case": case, "design": design_id})
        return stage.publish({"status": "verified" if solved else "rejected", "solver_status": report["status"],
                              "closed_loop": "not_run"}, source={"design": design_id, "request_sha256": request_hash}, activate=False)
