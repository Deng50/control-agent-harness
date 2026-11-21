"""MPC 交付验收。源设计保持冻结，通过的工作区控制器另行发布并保留父链。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import os
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

from .artifacts import atomic_json, read_json
from .contracts import ContractError, MaterialInterface, ModelInterface
from .control import SolverFailure, controller_from_descriptor
from .design_export import _CONTROLLER, _portable, _runtime_source
from .design_only import _controller, _vector, _MAX_DIAGNOSTIC_JACOBIAN, _MAX_DIAGNOSTIC_VARIABLES
from .modeling import tensor
from .predictors import _execution_spec, export_predictor_assets, resolve_predictor
from .reporting import DELIVERY_CHECKS, MAX_BUNDLE_BYTES, _bundle, _no_alias
from .runtime import OperationContext


class _MissingEvidence(ValueError):
    """可读的设计文件缺少本次验收所需的可执行资产或工况。"""

    def __init__(self, message, requirement=None):
        super().__init__(message)
        self.requirement = requirement


# 由受信发行版提供的固定驱动。-I 排除当前目录、用户站点及 PYTHONPATH，
# 包内代码只能由已核对的固定入口装载。输出留在独立临时目录，禁止加载平台模块。
_DRIVER = '''
import importlib.util,json,sys
from pathlib import Path
root=Path.cwd()
def restrict(event,args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo'}:
        raise PermissionError('Delivery verification uses offline local assets only')
sys.addaudithook(restrict)
class BlockHarness:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0]=='control_harness':
            raise ImportError('Standalone package requested project code')
sys.meta_path.insert(0,BlockHarness())
def save(name,value):
    (root/name).write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
phase='load'
try:
    bundle=root/'bundle'
    spec=importlib.util.spec_from_file_location('qualified_controller',bundle/'controller.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy=module.load_design(bundle).instantiate()
    save('loaded.json',True)
    phase='single_point'
    case=json.loads((root/'case.json').read_text(encoding='utf-8'))
    action,diagnostics=policy.act(case['x'],case['reference'],case['disturbance'],case['previous'])
    result={'action':action.tolist(),'diagnostics':diagnostics}
    if case.get('measure_domain'):
        # MPCPolicy 的 warm 是末行补齐的移位序列。恢复本次序列并记录预测范围，
        # 不仿真原对象、不改变已求出的首拍控制，也不向用户传输整批矩阵。
        import numpy as np
        sequence=np.vstack([action,policy.warm[:-1]])
        x=np.array(case['x'])
        lo=x.copy()
        hi=x.copy()
        for u in sequence:
            x=policy.model.step(x,u,case['disturbance'])
            lo=np.minimum(lo,x)
            hi=np.maximum(hi,x)
        result['predicted_ranges']={'x':{'min':lo.tolist(),'max':hi.tolist()},
                                    'u':{'min':sequence.min(0).tolist(),'max':sequence.max(0).tolist()}}
    save('result.json',result)
except Exception as error:
    save('error.json',{'phase':phase,'type':type(error).__name__,'message':str(error)[:400]})
    sys.exit(1)
'''


@contextmanager
def _detached_directory(workspace):
    # 按工作区 ACL 创建独占目录，兼容 Windows 沙箱。清理仅涉及本次创建的目录。
    staging = workspace.path / '.staging'
    staging.mkdir(exist_ok=True)
    _no_alias(workspace.path, staging)
    scratch = staging / ('delivery-' + uuid.uuid4().hex)
    scratch.mkdir()
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch)


def _request(config):
    if not isinstance(config, dict) or set(config) - {
            'target', 'case', 'required_checks', 'expected_action', 'action_atol', 'plan_refs'}:
        raise ContractError('交付验收仅接受 target、case、required_checks、expected_action 和 action_atol。')
    target = config.get('target')
    if target not in ('standalone_python', 'workspace_controller'):
        raise ContractError('请明确选择独立 Python 包或可供其他角色使用的工作区控制器。')
    extra = config.get('required_checks', [])
    if not isinstance(extra, list) or any(type(k) is not str or k not in DELIVERY_CHECKS for k in extra):
        raise ContractError('required_checks 包含未定义的验收项。')
    if 'workspace_controller' in extra and target != 'workspace_controller':
        raise ContractError('需要工作区控制器时请选择 target=workspace_controller。')
    required = list(dict.fromkeys(['files', 'load', 'standalone', 'single_point', *extra]))
    if target == 'workspace_controller' and 'workspace_controller' not in required:
        required.append('workspace_controller')
    if 'action_atol' in config and 'expected_action' not in config:
        raise ContractError('action_atol 应和 expected_action 一起提供。')
    atol = config.get('action_atol', 1e-6)
    if type(atol) not in (float, int) or not math.isfinite(atol) or atol < 0:
        raise ContractError('action_atol 必须是以控制量单位表示的非负有限绝对误差。')
    return target, required, atol


def _point(raw, interface, rate_limited):
    if not isinstance(raw, dict):
        raise _MissingEvidence('请提供独立验收工况 case，至少给出有序状态 x 和目标 reference。')
    if set(raw) - {'x', 'reference', 'disturbance', 'previous'}:
        raise ContractError('工况只接受 x、reference、disturbance 和 previous。')
    needed = ['x', 'reference'] + (['disturbance'] if interface.nd else []) + (['previous'] if rate_limited else [])
    missing = [key for key in needed if key not in raw]
    if missing:
        raise _MissingEvidence('独立验收工况还需明确字段 ' + '、'.join(missing))
    default_previous = [min(max(0.0, item.lower), item.upper) for item in interface.controls]
    values = {**{'disturbance': [], 'previous': default_previous}, **raw}
    for key, group in [('x', interface.states), ('reference', interface.outputs),
                       ('disturbance', interface.disturbances), ('previous', interface.controls)]:
        vector = _vector(values[key], len(group), key)
        if any(not signal.lower <= value <= signal.upper for signal, value in zip(group, vector)):
            raise ContractError(f'验收工况 {key} 超出已声明的信号界限，请核对工况和单位。')
        values[key] = vector.tolist()
    return values


def _snapshot(bundle, files, destination):
    # 重新逐字节校验读到的数据，使后续执行绑定到此次快照；所有文件只按清单复制。
    for name, expected in files.items():
        data = (bundle / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ContractError('复制过程中发现交付包文件版本变化。')
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _check_budget(interface, controller, spec):
    horizon = controller['horizon']
    variables = horizon * interface.nu
    constraints = 2 * horizon * (interface.nx + interface.ny + (interface.nu if 'input_rate_bounds' in controller else 0))
    width = interface.nx + interface.nu + interface.nd + 1
    # 沿用控制器单点雅可比预算，并在解算前限制连续矩阵指数及非线性积分工作量。
    if (variables > _MAX_DIAGNOSTIC_VARIABLES or variables * constraints > _MAX_DIAGNOSTIC_JACOBIAN
            or (spec['representation'] == 'continuous_polynomial' and width > 256)
            or horizon * (1 + len(spec.get('terms', []))) * spec.get('integration', {}).get('substeps', 1) > 4096):
        raise _MissingEvidence('该工况解算规模超过有界独立烟测预算；请使用能覆盖此规模的验收环境。')


def _sampled_ranges(report, source):
    """辨识模型的训练包络只限定外推；范围内的单点仍没有原对象闭环证明。"""
    if source.get('kind') == 'operating_point_linearization':
        source = source.get('source_conversion', {})
    if source.get('kind') == 'plant_predictor_conversion' and source.get('method') == 'sampled_affine':
        limits = source.get('applicability', {}).get('observed_training_ranges')
    elif report.get('route') == 'identify_affine':
        limits = report.get('observed_training_ranges')
    else:
        return None
    if not isinstance(limits, dict) or not {'x', 'u', 'd'}.issubset(limits):
        raise _MissingEvidence('采样辨识预测器缺少已记录的训练范围。', 'sampled_domain')
    return limits


def _within_sampled_ranges(ranges, limits, interface, label):
    groups = {'x': interface.states, 'u': interface.controls, 'd': interface.disturbances}
    for key, actual in ranges.items():
        bound = limits[key]
        width = len(groups[key])
        lo = _vector(bound.get('min', bound.get('lower')), width, key + ' 训练下限')
        hi = _vector(bound.get('max', bound.get('upper')), width, key + ' 训练上限')
        if np.any(hi < lo):
            raise ContractError('采样训练范围上下限次序错误。')
        for n, signal in enumerate(groups[key]):
            low, high = actual['min'][n], actual['max'][n]
            # 仅容许浮点舍入，常量通道不会自动获得任意物理容差。
            eps = 32 * np.finfo(float).eps * max(1., abs(lo[n]), abs(hi[n]))
            if low < lo[n]-eps or high > hi[n]+eps:
                raise _MissingEvidence(f'{label}的 {signal.name} 范围 [{low:.8g}, {high:.8g}] {signal.unit} '
                                       f'超出训练包络 [{lo[n]:.8g}, {hi[n]:.8g}]。请更换工况或补采数据。',
                                       'sampled_domain')


def _qualify_templates(bundle, design, predictor, scratch):
    # 清单哈希证明内部完整性，包内 Python 的执行授权来自已安装的固定发行模板。
    # 老包仍可下载；遇到其他版本的源代码时留下明确缺项，由该版本校验器接手。
    trusted = scratch / 'trusted'
    export_predictor_assets(predictor, trusted)
    expected = read_json(trusted / 'predictor_manifest.json')
    assets = bundle / 'prediction'
    manifest = read_json(assets / 'predictor_manifest.json')
    names = set(expected['files'])
    if (manifest.get('kind') != expected['kind'] or set(manifest.get('files', {})) != names
            or manifest.get('representation') != expected['representation']
            or manifest.get('interface') != expected['interface']):
        raise ContractError('包内预测器清单与冻结的预测接口不一致。')
    if read_json(assets / 'model.json') != read_json(trusted / 'model.json'):
        raise ContractError('包内预测方程或参数与源设计不一致。')
    for path, text in [(bundle / 'controller.py', _CONTROLLER), (bundle / 'mpc_runtime.py', _runtime_source()),
                       (assets / 'predictor.py', (trusted / 'predictor.py').read_text(encoding='utf-8'))]:
        if path.read_text(encoding='utf-8') != text:
            raise _MissingEvidence('交付包的执行模板版本与当前发行版不一致；请使用匹配版本的校验器或重新导出。')
    if read_json(bundle / 'design.json') != _portable(design):
        raise ContractError('工作区设计与交付包中的设计参数不一致。')


def _probe(scratch, point, context, checks):
    atomic_json(scratch / 'case.json', point)
    # 不继承业务确认收据、API 配置、用户 Python 启动环境。固定依赖由同一解释器装载。
    env = {key: os.environ[key] for key in ('SystemRoot', 'WINDIR', 'PATH') if key in os.environ}
    env.update({key: str(scratch) for key in ('TEMP', 'TMP', 'TMPDIR')})
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    deadline = time.monotonic() + 60
    with subprocess.Popen([sys.executable, '-I', '-B', '-c', _DRIVER], cwd=scratch, env=env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as proc:
        try:
            while proc.poll() is None:
                context.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('独立 Python 验收超过 60 秒预算。')
                try:
                    proc.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if (scratch / 'loaded.json').is_file() and read_json(scratch / 'loaded.json') is True:
        checks['load'] = 'passed'
    if proc.returncode or not (scratch / 'result.json').is_file():
        phase = 'single_point' if checks['load'] == 'passed' else 'load'
        checks[phase] = checks['standalone'] = 'failed'
        error = read_json(scratch / 'error.json') if (scratch / 'error.json').is_file() else {}
        raise ContractError(f"独立 Python 验收失败（{phase}，退出码 {proc.returncode}）。{error.get('message', '')}")
    checks['standalone'] = 'passed'
    return read_json(scratch / 'result.json')


def _valid_action(result, interface, previous, rate):
    action = _vector(result['action'], interface.nu, '验收动作')
    tolerance = 1e-6  # 与项目 MPCPolicy 的硬约束容差一致，单位为相应约束的工程量纲。
    if any(value < bound.lower - tolerance or value > bound.upper + tolerance for value, bound in zip(action, interface.controls)):
        raise ContractError('独立控制动作超过幅值界限。')
    if rate is not None and any(value < lo - tolerance or value > hi + tolerance for value, (lo, hi) in zip(action - previous, rate)):
        raise ContractError('独立控制动作超过每拍增量界限。')
    violation = result['diagnostics']['constraint_violation']
    if not math.isfinite(violation) or not 0 <= violation <= tolerance:
        raise ContractError('独立求解的预测约束残差超出容差。')
    return action


def verify_design_delivery(workspace, design_id, config, context=None):
    """按调用方明确选择的目标执行有界验收；如需跨角色执行，则额外复核原预测器。"""
    context = context or OperationContext()
    target, required, atol = _request(config)
    checks = dict.fromkeys(DELIVERY_CHECKS, 'not_run')
    with workspace.lock('verify_design_delivery'):
        context.check()
        root, meta, bundle, manifest, files, size = _bundle(workspace, design_id, MAX_BUNDLE_BYTES)
        if meta['kind'] != 'controller_design':
            raise ContractError('请选择 controller_design 交付包。')
        workspace.resolve(design_id, kind='controller_design')
        design = read_json(root / 'design.json')
        checks['files'] = 'passed'
        evidence = {'manifest_sha256': files[manifest], 'bundle_bytes': size, 'required_checks': required,
                    'requested_case': config.get('case'), 'target': target, 'expected_action': config.get('expected_action'),
                    'entrypoint': 'controller.py / load_design(bundle).instantiate().act(x, reference, disturbance, previous)',
                    'action_atol': atol if 'expected_action' in config else None,
                    'scope': '本次交付只验证显式工况的有界控制动作。闭环性能、硬件时延及 MATLAB/Simulink 需要独立证据。'}
        reasons = []
        additional_missing = []
        descriptor = None
        failed = False
        try:
            if not design.get('model_instantiated'):
                raise _MissingEvidence('设计尚未实例化，请补齐 ' + '、'.join(design.get('required_parameters') or ['物理模型和接口']))
            prediction = design['prediction']
            spec = prediction.get('spec') or {}
            if spec.get('representation') not in ('continuous_polynomial', 'discrete_polynomial'):
                raise _MissingEvidence('当前独立验收器支持离散多项式及显式积分的连续多项式；该预测器需配套的独立校验器。')
            interface = ModelInterface.parse(design['physical_interface'])
            declared = {**spec.get('interface', {}), 'time_semantics': 'discrete_state_transition'}
            if ModelInterface.parse(declared).to_dict() != interface.to_dict():
                raise ContractError('设计的物理接口与冻结方程声明的信号、单位或周期不一致。')
            controller, weights, _ = _controller({'controller': design['controller_config'], 'weights': design['weights']}, interface.nu)
            point = _point(config.get('case'), interface, 'input_rate_bounds' in controller)
            expected_action = _vector(config['expected_action'], interface.nu, 'expected_action') if 'expected_action' in config else None
            _check_budget(interface, controller, spec)
            predictor = resolve_predictor(None, {}, {'method': 'equations', 'model_spec': spec}, context)
            if predictor.interface.to_dict() != interface.to_dict():
                raise ContractError('实际预测器接口与交付设计声明不一致。')
            initial_y = predictor.runtime.output(tensor(point['x'])).detach().cpu().numpy()
            if any(not signal.lower <= y <= signal.upper for y, signal in zip(initial_y, interface.outputs)):
                raise ContractError('验收初态的预测输出超出已声明的输出约束。')
            evidence['case'] = point
            report = prediction.get('report', {})
            source = report.get('source', {})
            model_id = prediction.get('artifact_id')
            if model_id is not None:
                if model_id not in [parent['id'] for parent in meta['parents']]:
                    raise ContractError('设计声明的预测器没有对应的冻结父版本。')
                model_root, model_meta = workspace.resolve(model_id)
                # 同时检查独立包的来源。工作区模型如包含一般 Python 代码，不能走本验收器。
                if model_meta['kind'] not in {'identified_model', 'nominal_model', 'reference_model'}:
                    raise _MissingEvidence('工作区预测器类型需要专用的装载验收器。')
                if _execution_spec(read_json(model_root / 'model.json')) != _execution_spec(spec):
                    raise ContractError('设计中冻结的动力学与登记父模型不一致。')
                source = model_meta['source']
            source = source if isinstance(source, dict) else {}
            evidence['applicability'] = source.get('applicability') or {
                key: report[key] for key in ('scope', 'observed_training_ranges', 'validity_bounds', 'source_conversion') if key in report}
            limits = _sampled_ranges(report, source)
            if limits:
                evidence['sampled_domain'] = {'checked_against': limits, 'semantics': '逐通道训练包络仅用于识别外推，不能证明联合工况覆盖。'}
                _within_sampled_ranges({key: {'min': point[field], 'max': point[field]} for key, field in
                                       [('x', 'x'), ('u', 'previous'), ('d', 'disturbance')]}, limits, interface, '指定工况')
            with _detached_directory(workspace) as scratch:
                snapshot = scratch / 'bundle'
                _snapshot(bundle, files, snapshot)
                _qualify_templates(snapshot, design, predictor, scratch)
                context.progress('standalone_delivery_check', source_artifact=design_id)
                result = _probe(scratch, {**point, 'measure_domain': bool(limits)}, context, checks)
            evidence['standalone_result'] = result
            try:
                action = _valid_action(result, interface, point['previous'], controller.get('input_rate_bounds'))
            except ContractError:
                checks['single_point'] = 'failed'
                raise
            checks['single_point'] = 'passed'
            if limits:
                _within_sampled_ranges(result['predicted_ranges'], limits, interface, '该单点的预测序列')
            if expected_action is not None:
                evidence['max_action_error'] = float(np.max(np.abs(action - expected_action)))
                if evidence['max_action_error'] > atol:
                    raise ContractError('实际控制动作未满足指定目标动作及绝对误差容差。')
            if target == 'workspace_controller':
                if not model_id:
                    raise _MissingEvidence('转为工作区控制器需绑定一个已登记的数值预测器；请登记同一方程后用该版本生成设计。')
                descriptor = {'prediction_model': model_id, 'parameters': {**controller, **weights},
                              'design_artifact': design_id, 'delivery_scope': 'explicit_single_point',
                              'applicability': evidence['applicability'],
                              'closed_loop': 'not_run'}
                # 同一工况再从真实工作区入口加载，保留父作业取消与单点时间上限。
                def cancelled():
                    context.check()
                    return False
                local_context = OperationContext(deadline=time.monotonic() + 60, cancelled=cancelled)
                policy = controller_from_descriptor(workspace, descriptor, local_context)
                try:
                    native, diagnostics = policy.act(point['x'], point['reference'], point['disturbance'], point['previous'])
                except (SolverFailure, FloatingPointError):
                    checks['workspace_controller'] = 'failed'
                    raise
                evidence['workspace_result'] = {'action': native.tolist(), 'diagnostics': diagnostics}
                if not np.isfinite(native).all() or not np.allclose(native, action, rtol=0, atol=1e-6):
                    checks['workspace_controller'] = 'failed'
                    raise ContractError('相同工况下工作区控制器和独立包的动作不一致。')
                checks['workspace_controller'] = 'passed'
        except _MissingEvidence as error:
            reasons.append(str(error))
            if error.requirement:
                additional_missing.append(error.requirement)
        except (ContractError, SolverFailure, FloatingPointError) as error:
            failed = True
            reasons.append(str(error))
        missing = [key for key in required if checks[key] != 'passed'] + additional_missing
        satisfied = not failed and not missing and not reasons
        status = 'failed' if failed else ('artifact_ready' if satisfied else 'partial')
        delivery = {'status': status, 'source_artifact': design_id, 'bundle_artifact': design_id,
                    'manifest_sha256': files[manifest], 'target': target, 'request_satisfied': satisfied,
                    'checks': checks, 'missing_requirements': missing, 'details': ' '.join(reasons) or evidence['scope']}
        if failed and 'expected_action' in config and checks['single_point'] == 'passed' and evidence.get('max_action_error', 0) > atol:
            delivery['missing_requirements'].append('expected_action')
        promote = satisfied and target == 'workspace_controller'
        delivery['workspace_controller_published'] = promote
        parents = [design_id, descriptor['prediction_model']] if promote else [design_id]
        # 下游采数会避开原设计预留的工况；单点验收没有预留闭环训练/测试工况。
        stage_config = {**config, 'verification_scenarios': [], 'test_scenarios': []} if promote else config
        kind = 'controller' if promote else 'delivery_validation'
        output_interface = interface if promote else MaterialInterface('json', 'MPC design delivery evidence')
        with workspace.stage(kind, output_interface, parents, stage_config, compatible_parents=False) as stage:
            if promote:
                atomic_json(stage.root / 'controller.json', descriptor)
            atomic_json(stage.root / 'delivery.json', delivery)
            atomic_json(stage.root / 'report.json', {'algorithm': 'verify_design_delivery', 'delivery': delivery,
                        'evidence': evidence, 'confirmation_sources': context.case_confirmation_sources})
            context.check()
            return stage.publish({'status': 'verified' if satisfied else 'rejected', 'delivery_status': status,
                                  'scope': 'explicit_delivery_requirements', 'closed_loop': 'not_run'},
                                 source={'kind': 'design_delivery_verification', 'design_artifact': design_id}, activate=False)
