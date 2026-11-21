"""把有状态定义的采样对象及其真实采样结果转换为可供 MPC 加载的预测器。"""
from __future__ import annotations

import math

import numpy as np

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError
from .identification import fit_affine_transitions
from .modeling import DeclarativeDynamics
from .plants import _selection
from .predictive import _read_sampled
from .predictors import compare_predictor_trajectories
from .runtime import OperationContext


def _limits(config):
    """门限作用于各个状态、各个输出；每一维除以该变量声明的物理范围。"""
    limits = {}
    for label, default in (("one_step", .01), ("rollout", .05)):
        value = config.get(f"max_{label}_nrmse", default)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ContractError(f"max_{label}_nrmse 需要有限正数，按各通道物理范围归一化。")
        limits[label] = value
    return limits


def _ranges(values):
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def convert_plant_predictor(workspace: Workspace, inputs: dict, config: dict, context: OperationContext) -> str:
    """保留原对象，对另外发布的动力学逐项核验。没有模型选择或有效状态数据时拒绝转换。"""
    context.check()
    if set(config) - {"method", "max_one_step_nrmse", "max_rollout_nrmse", "plan_refs"}:
        raise ContractError("预测器转换配置包含未识别的参数。")
    method = config.get("method")
    if method not in ("declared_equations", "sampled_affine"):
        raise ContractError("请选择核验已登记的明确方程，或确认只在采样工况内辨识仿射近似。")
    expected = {"plant", "dataset", "equations"} if method == "declared_equations" else {"plant", "dataset"}
    if set(inputs) != expected or any(not isinstance(inputs[k], str) or not inputs[k] for k in expected):
        raise ContractError("方程核验需要 plant、dataset、equations 三个产物；采样辨识仅需要 plant 和 dataset。")
    limits = _limits(config)
    physical, adapter, _, plant_meta, _ = _selection(workspace, inputs["plant"])
    if not adapter or adapter["backend"] != "python_callable":
        raise ContractError("本转换入口需要已登记的 python_callable plant_model 及有完整状态的采样数据。")
    # 先检查状态到输出的投影。单有自定义测量函数时，盲读整套轨迹无法补全这个物理契约。
    if method == "sampled_affine" and "C" not in adapter:
        raise ContractError("缺少显式线性输出映射 C。请登记已知输出方程或选用保留输出函数的建模路线。")
    arrays, masks, split, physical_data, data_meta = _read_sampled(workspace, inputs["dataset"])
    physical.compatible(physical_data)
    data_source = data_meta.get("source", {})
    if ([parent["id"] for parent in data_meta["parents"]] != [inputs["plant"]]
            or data_source.get("model") != inputs["plant"] or data_source.get("kind") != "plant_simulator"
            or data_source.get("backend") != adapter["backend"]):
        raise ContractError("采样数据必须来自选定的原对象版本，请选择其完整 sample_model 产物。")
    unique, counts = np.unique(arrays["trajectory_id"], return_counts=True)
    lengths = dict(zip(unique.tolist(), counts.tolist()))
    if any(lengths[trajectory] < 2 for part in ("validation", "test") for trajectory in split[part]):
        raise ContractError("独立验证与测试的每条轨迹至少需要两步才能检验滚动误差，请补充连续状态采样。")

    parents = [inputs["plant"], inputs["dataset"]]
    identification = None
    if method == "declared_equations":
        # 对方程路线只读取注册的声明式数据。不会把第三方 Python 代码当作方程加载。
        equations_root, meta = workspace.resolve(inputs["equations"])
        if meta["kind"] not in ("nominal_model", "reference_model"):
            raise ContractError("equations 需要用 register_model 登记的离散或显式积分多项式方程。")
        spec = read_json(equations_root / "model.json")
        if spec.get("representation") not in ("discrete_polynomial", "continuous_polynomial"):
            raise ContractError("仅支持明确的离散多项式或声明积分方法的连续多项式方程。")
        model = DeclarativeDynamics(spec)
        physical.compatible(model.interface)
        if "C" in adapter and not np.array_equal(model.C.numpy(), adapter["C"]):
            raise ContractError("方程的输出矩阵 C 与原对象已登记的输出映射不一致，请核对输出顺序和单位。")
        kind = "nominal_model"
        parents.append(inputs["equations"])
    else:
        # f(x,u,d) 不一定是线性函数。拟合仅使用训练轨迹，绝不从文件名或数组列数推断 C。
        try:
            coefficients, identification = fit_affine_transitions(physical, arrays, masks, {}, context)
        except ContractError as error:
            raise ContractError(f"无法从训练轨迹辨识仿射模型：{error}。请核对状态定义、独立激励和样本数，或提供明确方程。") from error
        spec = {"representation": "discrete_polynomial", "interface": physical.to_dict(),
                **coefficients, "C": adapter["C"], "terms": []}
        model = DeclarativeDynamics(spec)
        kind = "identified_model"

    # 构造验证尺度和适用范围。物理约束保持原样；样本的坐标包围盒不能证明整盒均被验证。
    scales = {key: np.array([v.upper-v.lower for v in variables])
              for key, variables in (("state", physical.states), ("output", physical.outputs))}
    def metric(prediction, truth, axis, test):
        relative = (prediction-truth) / scales[axis]
        nrmse = np.sqrt(np.mean(relative**2, axis=0))
        if not np.isfinite(nrmse).all():
            raise FloatingPointError("预测误差超出有限数值范围。")
        return {"nrmse_per_channel": nrmse.tolist(), "limit": limits[test],
                "passed": bool(np.all(nrmse <= limits[test])), "rows": len(truth)}

    reason = None
    current_output = None
    try:
        current_output = metric(arrays["x"] @ model.C.numpy().T, arrays["y"], "output", "one_step")
        one_step, rollouts, passed = compare_predictor_trajectories(model, arrays, masks, split, metric, context)
        passed = passed and current_output["passed"]
        if not passed:
            reason = "预测误差超过约定门限。请按通道检查输出映射和独立轨迹误差，修正方程或选择非线性预测器。"
    except FloatingPointError as error:
        # 发散模型留下拒收报告。保持不完整的开环指标为空，避免无穷值污染标准 JSON。
        one_step, rollouts, passed, reason = {}, {}, False, str(error)
    index = workspace.manifest["artifacts"]
    # 按工况和物理时间定位首帧，包括磁盘行顺序与工况顺序不同的情况。
    sorted_rows = np.lexsort((arrays["time"], arrays["trajectory_id"]))
    _, first = np.unique(arrays["trajectory_id"][sorted_rows], return_index=True)
    applicability = {
        "scope": "declared_equations_checked_on_sampled_trajectories" if method == "declared_equations"
                 else "affine_approximation_within_sampled_conditions_only",
        "sample_time": physical.sample_time, "physical_interface": physical.to_dict(),
        "parameter_values": adapter["parameters"], "initial_state_source": "recorded_x_at_first_timestamp_per_trajectory",
        "observed_initial_state_ranges": _ranges(arrays["x"][sorted_rows[first]]),
        "observed_training_ranges": {key: _ranges(arrays[key][masks["train"]]) for key in ("x", "u", "d", "y")},
        "sampling_plan_hash": data_source["sampling_plan_hash"], "global_equivalence_proved": False,
        "closed_loop_validation": "not_run"}
    source = {"kind": "plant_predictor_conversion", "method": method, "plant": inputs["plant"],
              "dataset": inputs["dataset"], "material": plant_meta["source"]["material"],
              "source_content_hash": plant_meta["source"]["content_hash"], "source_backend": adapter["backend"],
              "source_semantics": adapter["source_semantics"], "source_integration": adapter.get("integration"),
              "equations": inputs.get("equations"), "parent_versions": {p: index[p]["metadata_hash"] for p in parents},
              "applicability": applicability}
    report = {"method": method, "source": source, "applicability": applicability, "limits": limits,
              "normalization": "RMSE per component divided by declared upper minus lower",
              "trajectory_split": split, "identification": identification, "one_step": one_step,
              "rollout": rollouts, "current_output_mapping": current_output,
              "acceptance": {"passed": passed, "reason": reason}, "confirmation_sources": context.case_confirmation_sources}
    context.check()
    with workspace.stage(kind, physical, parents, config) as stage:
        atomic_json(stage.root / "model.json", spec)
        if model.transition_adapter is not None:
            atomic_json(stage.root / "transition_adapter.json", model.transition_adapter)
        atomic_json(stage.root / "report.json", report)
        return stage.publish({"status": "verified" if passed else "rejected",
                              "heldout_one_step_and_rollout": passed, "scope": applicability["scope"]},
                             source=source, activate=False)
