"""Bounded physical-rollout policy gradients from actual external imitation.

The numerical kernel adapts policy.py's physical rollout/anchor/Adam algorithm
(original source SHA256 1225ba0b3a387bd5cb01f3bf68a9578b4235d9072f8387dc72ca3843e5f5edc9).
The old optimizer is unchanged. This branch has explicit causal feature and
model-domain contracts; no critic, PPO, SAC, Plant differentiation or test-based
checkpoint selection is introduced.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import math
import numpy as np
import torch

from .artifacts import atomic_json, file_hash, read_json
from .checkpoints import current_checkpoint, load_checkpoint, restore_training, save_checkpoint, seed_training, training_contract
from .contracts import ContractError, DataInterface, ModelInterface, canonical_hash
from .runtime import OperationContext
from .pgac_replay import SegmentReplay, normalize_replay
from .pgac_scenarios import scenario_series, observation_noise
from .pgac_tracking import tracking_gate, tracking_index, tracking_less, tracking_summary


class RolloutDomainError(ContractError):
    """A numerical prediction would leave the explicitly qualified domain."""


def _number(value, label, *, positive=False, maximum=None):
    try: result = float(value) if type(value) in {int, float} else float('nan')
    except (ValueError, OverflowError): result = float('nan')
    if not math.isfinite(result) or result < 0 or positive and result == 0 or maximum is not None and result > maximum:
        raise ContractError(f'{label} must be a finite {"positive" if positive else "nonnegative"} number')
    return result


def _integer(value, label, maximum=100_000):
    if type(value) is not int or not 1 <= value <= maximum: raise ContractError(f'{label} must be an integer in [1,{maximum}]')
    return value


def _box(raw, variables, label):
    try: value = np.asarray(raw)
    except (ValueError, TypeError) as error: raise ContractError(f'{label} requires an explicit finite physical box') from error
    if not variables and isinstance(raw, list) and not raw: return []
    if value.dtype.kind not in 'fiu' or value.shape != (len(variables), 2) or not np.isfinite(value).all() or np.any(value[:, 0] >= value[:, 1]):
        raise ContractError(f'{label} requires one finite increasing [low,high] per ordered channel')
    bounds = np.asarray([[v.lower, v.upper] for v in variables])
    if np.any(value[:, 0] < bounds[:, 0]) or np.any(value[:, 1] > bounds[:, 1]): raise ContractError(f'{label} exceeds the explicit physical interface')
    return value.astype(np.float64).tolist()


def normalize_pgac_config(config, interface, coverage):
    allowed = {'epochs','learning_rate','batch_size','horizon','seed','q','r','s','imitation_weight','constraint_weight',
        'domain','original_evaluation','replay','resume_checkpoint','resume_checkpoint_sha256','plan_refs'}
    if not isinstance(config, dict) or set(config) - allowed: raise ContractError('Unknown restricted PGAC setting')
    if not isinstance(config.get('domain'), dict) or set(config['domain']) != {'x','u','d'}: raise ContractError('PGAC domain must explicitly contain x,u,d physical boxes')
    domain = {key: _box(config['domain'][key], variables, 'domain.' + key) for key, variables in
        [('x', interface.states), ('u', interface.controls), ('d', interface.disturbances)]}
    if not isinstance(coverage, dict) or set(coverage) != {'x','u','d'}: raise ContractError('Qualified model coverage has no explicit x,u,d domain')
    for key, box in domain.items():
        declared = np.asarray(coverage[key], dtype=np.float64).reshape(-1, 2); value = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        if declared.shape != value.shape or np.any(value[:, 0] < declared[:, 0]) or np.any(value[:, 1] > declared[:, 1]):
            raise ContractError('PGAC domain must be a subset of qualified model coverage: ' + key)
    result = {'epochs': _integer(config.get('epochs', 40), 'epochs'), 'epochs_semantics': 'target_total',
        'learning_rate': _number(config.get('learning_rate', .0003), 'learning_rate', positive=True, maximum=1.),
        'batch_size': _integer(config.get('batch_size', 24), 'batch_size', 100_000),
        'horizon': _integer(config.get('horizon'), 'horizon', 256), 'seed': config.get('seed', 42),
        'imitation_weight': _number(config.get('imitation_weight', .1), 'imitation_weight'),
        'constraint_weight': _number(config.get('constraint_weight', 100.), 'constraint_weight'),
        'domain': domain, 'grad_clip': 5., 'optimizer': 'adam', 'device': 'cpu', 'dtype': 'float64',
        'rollout_semantics': 'external_current_full_state_features_held_subticks_v1',
        'constraint_penalty_semantics': 'mean_normalized_state_output_penalty_over_plant_subticks'}
    if type(result['seed']) is not int or not 0 <= result['seed'] < 2**32: raise ContractError('seed must be an explicit uint32 integer')
    for key in ('q','r','s'):
        if key not in config: raise ContractError('External PGAC requires explicit internal q,r,s weights')
        result[key] = _number(config[key], key)
    if not 0 < math.fsum(result[key] for key in ('q','r','s')) < float('inf'): raise ContractError('At least one finite q,r,s objective weight must be positive')
    if 'replay' in config:
        result['replay'] = normalize_replay(config['replay'])
        if result['replay']['mode'] != 'balanced_starts':
            result['rollout_semantics'] = 'episode_local_current_policy_with_recorded_exogenous_sequences_v1'
    return result


def require_current_full_state(policy, interface):
    physical = ModelInterface.parse(policy.physical_context['interface']) if policy.physical_context.get('interface') else None
    if physical is None: raise ContractError('PGAC requires an explicit physical source, not inferred feature dimensions')
    interface.compatible(physical)
    binding = policy.binding
    if binding.get('status') != 'supported_current_mapping': raise ContractError('PGAC requires explicit current causal features; history, delay or unknown mappings are unsupported')
    states = {entry['physical_index'] for entry in binding['current_features'] if entry['kind'] == 'observation' and entry['source'] == 'state'}
    if states != set(range(interface.nx)): raise ContractError('Restricted PGAC needs every physical state explicitly selected as a current state observation; partial observability remains supported for imitation')
    return binding


def tensor_policy_action(policy, model, state, reference, disturbance, previous, noise=None):
    """Assemble exactly the causal DataInterface order without NumPy/no_grad."""
    values = []; output = None
    for entry in policy.binding['current_features']:
        kind = entry['kind']
        if kind == 'observation':
            if entry['source'] == 'state': value = state[..., entry['physical_index']]
            else:
                if output is None: output = model.output(state)
                value = output[..., entry['physical_index']]
            if noise is not None:
                value = value + noise[..., entry['index']]
        elif kind == 'reference': value = reference[..., entry['index']]
        elif kind == 'disturbance': value = disturbance[..., entry['index']]
        elif kind == 'previous_action': value = previous[..., entry['index']]
        else: raise ContractError('Unsupported causal PGAC feature source')
        values.append(value)
    return policy(torch.stack(values, dim=-1), previous_action=previous)


def check_domain(state, action, disturbance, domain, phase):
    for name, value in [('x', state), ('u', action), ('d', disturbance)]:
        if name == 'u' and value is None: continue  # Action is not known before causal features are assembled.
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite PGAC ' + phase + ' ' + name)
        bounds = value.new_tensor(domain[name]).reshape(-1, 2)
        if torch.any(value < bounds[:, 0]) or torch.any(value > bounds[:, 1]): raise RolloutDomainError('PGAC ' + phase + ' leaves qualified ' + name + ' domain; baseline is not replaced')


def _window_costs(policy, model, state, reference, disturbance, previous, settings, substeps, context,
                  reference_sequence, disturbance_sequence, noise_sequence, tracked):
    """Each policy evolves its own state and applied action within the same exogenous window."""
    i = model.interface
    xlo = state.new_tensor([v.lower for v in i.states]); xhi = state.new_tensor([v.upper for v in i.states])
    ylo = state.new_tensor([v.lower for v in i.outputs]); yhi = state.new_tensor([v.upper for v in i.outputs])
    yscale = yhi - ylo; uscale = state.new_tensor([v.upper - v.lower for v in i.controls])
    horizon = settings['horizon']
    sums = {name:state.new_zeros(()) for name in ('tracking','action','increment','constraint')}
    windows = {key: [] for key in ('tracking', 'action', 'increment')}
    original_state = state; last_action = previous
    for k in range(horizon):
        reference = reference if reference_sequence is None else reference_sequence[:,k]
        disturbance = disturbance if disturbance_sequence is None else disturbance_sequence[:,k]
        noise = None if noise_sequence is None else noise_sequence[:,k]
        context.check(); check_domain(state, None, disturbance, settings['domain'], 'before observation')
        action = tensor_policy_action(policy, model, state, reference, disturbance, last_action, noise)
        if k==0: first_action=action
        check_domain(state, action, disturbance, settings['domain'], 'rollout input')
        for _ in range(substeps):
            context.check(); state = model(state, action, disturbance)
            if not isinstance(state,torch.Tensor) or state.shape != original_state.shape or state.dtype != torch.float64 or state.device.type != 'cpu': raise FloatingPointError('PGAC dynamics returned an invalid next-state batch')
            check_domain(state, action, disturbance, settings['domain'], 'predicted subtick')
            output = model.output(state)
            if output.shape != reference.shape or not torch.isfinite(output).all(): raise FloatingPointError('PGAC dynamics returned an invalid output batch')
            sums['constraint'] = sums['constraint'] + settings['constraint_weight'] / substeps * (((torch.relu(xlo-state)+torch.relu(state-xhi))/(xhi-xlo)).square().mean()
                + ((torch.relu(ylo-output)+torch.relu(output-yhi))/yscale).square().mean())
        if tracked is None:
            sums['tracking'] = sums['tracking'] + settings['q'] * ((output-reference)/yscale).square().mean()
            sums['action'] = sums['action'] + settings['r'] * (action/uscale).square().mean()
            sums['increment'] = sums['increment'] + settings['s'] * ((action-last_action)/uscale).square().mean()
        else:
            windows['tracking'].append((output[:, tracked]-reference[:, tracked])/yscale[tracked])
            windows['action'].append(action/uscale)
            windows['increment'].append((action-last_action)/uscale)
        last_action = action
    terms = {key:value/horizon for key,value in sums.items()}
    if tracked is not None:
        for key, trace in windows.items():
            values = torch.stack(trace, dim=1).reshape(len(state), -1)
            # vector_norm has a finite zero subgradient for perfect tracking or zero actions.
            terms[key] = torch.linalg.vector_norm(values, dim=1) / math.sqrt(values.shape[1])
    return terms, first_action


def physical_rollout_loss(policy, model, state, reference, disturbance, previous, expert_action, settings, *, substeps=1, context=None, reference_sequence=None, disturbance_sequence=None,
                          noise_sequence=None, anchor_mask=None, components=None, baseline_policy=None):
    """Compute the configured control objective through the projected Actor and frozen Plant model."""
    context = context or OperationContext(); i = model.interface
    if (state.ndim != 2 or state.shape[1] != i.nx or reference.shape != (len(state), i.ny) or disturbance.shape != (len(state), i.nd)
            or previous.shape != (len(state), i.nu) or expert_action.shape != previous.shape
            or any(value.dtype != torch.float64 or value.device.type != 'cpu' or not torch.isfinite(value).all() for value in (state,reference,disturbance,previous,expert_action))):
        raise ContractError('PGAC loss requires finite ordered CPU float64 physical batches')
    if type(substeps) is not int or substeps < 1: raise ContractError('Plant subticks must be explicit positive integers')
    for values,width in ((reference_sequence,i.ny),(disturbance_sequence,i.nd),
                         (noise_sequence,len(policy.physical_context['physical_mapping']['observations']))):
        if values is not None and (values.shape!=(len(state),settings['horizon'],width) or values.dtype!=state.dtype
                                   or values.device!=state.device or not torch.isfinite(values).all()):
            raise ContractError('PGAC exogenous windows must match batch, horizon and channel dimensions')
    objective = settings.get('original_evaluation', {}).get('evaluation', {})
    tracked = tracking_index(objective, i) if objective.get('primary_metric') == 'tracking_rmse' else None
    window = (model, state, reference, disturbance, previous, settings, substeps, context,
              reference_sequence, disturbance_sequence, noise_sequence, tracked)
    terms, first_action = _window_costs(policy, *window)
    if tracked is not None:
        if baseline_policy is None:
            raise ContractError('Tracking-primary training requires the actual frozen imitation policy for matched effort budgets')
        with torch.no_grad():
            baseline, _ = _window_costs(baseline_policy, *window)
        terms['tracking'] = settings['q'] * terms['tracking'].mean()
        for name, weight, budget in [('action', 'r', 'max_action_rms_relative_increase'),
                                     ('increment', 's', 'max_increment_rms_relative_increase')]:
            limit = baseline[name] * (1 + objective[budget])
            terms[name] = settings[weight] * torch.relu(terms[name] - limit).mean()
    uscale = state.new_tensor([v.upper - v.lower for v in i.controls])
    protection = ((first_action-expert_action)/uscale).square().mean(dim=1)
    if anchor_mask is not None:
        if anchor_mask.dtype!=torch.bool or anchor_mask.shape!=(len(state),):
            raise ContractError('Imitation anchors must identify actual expert rows')
        protection = torch.where(anchor_mask,protection,0.)
    terms['imitation'] = settings['imitation_weight'] * protection.mean()
    loss = sum(terms.values())
    if components is not None:
        components.update({key:float(value.detach()) for key,value in terms.items()})
    if not torch.isfinite(loss): raise FloatingPointError('PGAC physical rollout loss is nonfinite')
    return loss


def _segment_loss(policy, model, sampler, samples, settings, substeps, context, *, baseline_policy=None):
    total_count = sum(map(len, samples.values()))
    loss = next(policy.network.parameters()).new_zeros(())
    parts = Counter()
    for horizon, items in samples.items():
        for start in range(0, len(items), settings['batch_size']):
            chunk = items[start : start + settings['batch_size']]
            components = {}
            if sampler.settings['mode'] == 'balanced_starts':
                # This ablation changes only the sampling distribution. The old kernel
                # holds reference/disturbance and starts an H-step prediction at any row.
                batch = {
                    key: torch.stack([ep[key][t] for ep, t in chunk])
                    for key in ('x', 'previous_u', 'u', 'reference', 'd')
                }
                value = physical_rollout_loss(
                    policy,
                    model,
                    batch['x'],
                    batch['reference'],
                    batch['d'],
                    batch['previous_u'],
                    batch['u'],
                    settings,
                    substeps=substeps,
                    context=context,
                    components=components,
                    baseline_policy=baseline_policy,
                )
            else:
                current, series, anchors = sampler.batch(chunk, horizon)
                value = physical_rollout_loss(
                    policy,
                    model,
                    current['x'],
                    series['reference'][:, 0],
                    series['d'][:, 0],
                    current['previous_u'],
                    current['u'],
                    {**settings, 'horizon': horizon},
                    substeps=substeps,
                    context=context,
                    reference_sequence=series['reference'],
                    disturbance_sequence=series['d'],
                    noise_sequence=series['noise'],
                    anchor_mask=anchors,
                    components=components,
                    baseline_policy=baseline_policy,
                )
            weight = len(chunk) / total_count
            loss = loss + weight * value
            parts.update({key: weight * v for key, v in components.items()})
    return loss, dict(parts)


def _state_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode()); value.update(str(tensor.dtype).encode()); value.update(str(tuple(tensor.shape)).encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def _predictor_validation(policy, model, original, settings, substeps, context, *, split='validation'):
    i = model.interface; evaluation = original['evaluation']; runs = []
    tracked = tracking_index(evaluation, i) if evaluation.get('primary_metric') == 'tracking_rmse' else None
    output_bounds = torch.tensor([[v.lower, v.upper] for v in i.outputs], dtype=torch.float64)
    rate = policy.action_constraints['rate_bounds']
    if rate is not None:
        rate = torch.tensor(rate, dtype=torch.float64)

    def inside(values, bounds):
        return bool(torch.all(values >= bounds[:, 0] - 1e-12) and torch.all(values <= bounds[:, 1] + 1e-12))

    def observe(state):
        values = model.output(state)
        if values.shape != (1, i.ny) or not torch.isfinite(values).all():
            raise FloatingPointError('PGAC validation output shape/values are invalid')
        return values, inside(values, output_bounds)

    for trajectory_id, scenario in enumerate(original['scenarios']):
        if scenario['split'] != split: continue
        context.check(); x = torch.tensor(scenario['initial_state'],dtype=torch.float64).unsqueeze(0)
        refs = torch.tensor(scenario_series(scenario,'reference'),dtype=torch.float64)
        disturbances = torch.tensor(scenario_series(scenario,'disturbance'),dtype=torch.float64)
        noises = torch.tensor(observation_noise(scenario,len(policy.physical_context['physical_mapping']['observations'])),dtype=torch.float64)
        previous = torch.tensor(scenario['initial_previous_action'],dtype=torch.float64).unsqueeze(0); costs = []
        outputs_valid = increments_valid = True
        tracking_squared, action_squared, increment_squared = [], [], []
        with torch.no_grad():
            for k in range(scenario['steps']):
                r=refs[k:k+1]; d=disturbances[k:k+1]
                context.check(); check_domain(x,None,d,settings['domain'],'before validation observation')
                if k == 0:
                    _, outputs_valid = observe(x)
                u = tensor_policy_action(policy,model,x,r,d,previous,noises[k:k+1]); check_domain(x,u,d,settings['domain'],'validation input')
                if rate is not None:
                    increments_valid &= inside(u-previous, rate)
                for _ in range(substeps):
                    x = model(x,u,d)
                    if not isinstance(x,torch.Tensor) or x.shape!=(1,i.nx) or x.dtype!=torch.float64 or x.device.type!='cpu': raise FloatingPointError('PGAC validation next-state shape/dtype is invalid')
                    check_domain(x,u,d,settings['domain'],'validation predicted subtick')
                    y, in_bounds = observe(x)
                    outputs_valid &= in_bounds
                yscale = y.new_tensor([v.upper-v.lower for v in i.outputs]); uscale = u.new_tensor([v.upper-v.lower for v in i.controls])
                score = evaluation['q']*((y-r)/yscale).square().mean()+evaluation['r']*(u/uscale).square().mean()+evaluation['s']*((u-previous)/uscale).square().mean()
                if not torch.isfinite(score): raise FloatingPointError('PGAC validation objective is nonfinite')
                if tracked is not None:
                    tracking_squared.append(float((y[0, tracked] - r[0, tracked]).square()))
                    action_squared.append(float((u/uscale).square().mean()))
                    increment_squared.append(float(((u-previous)/uscale).square().mean()))
                costs.append(float(score)); previous = u
        runs.append({'name':scenario.get('name'), 'split':split,'steps':scenario['steps'],'score':math.fsum(costs)/len(costs),'hard_feasible':outputs_valid and increments_valid,
                     **({'family':scenario['family']} if 'family' in scenario else {})})
        if tracked is not None:
            runs[-1].update(trajectory_id=trajectory_id, output_bounds_passed=outputs_valid, increment_bounds_passed=increments_valid,
                            tracking_rmse=math.sqrt(math.fsum(tracking_squared)/len(costs)),
                            action_rms=math.sqrt(math.fsum(action_squared)/len(costs)),
                            increment_rms=math.sqrt(math.fsum(increment_squared)/len(costs)))
    if not runs: raise ContractError('PGAC requires explicit original validation conditions before fitting')
    feasible = all(row['hard_feasible'] for row in runs)
    result = {'score':math.fsum(row['score'] for row in runs)/len(runs) if feasible else None,'hard_feasible':feasible,'runs':runs,
              'source':'qualified_prediction_model_only_not_original_Plant','test_used':False}
    if tracked is not None:
        result['tracking'] = tracking_summary(runs)
    return result


def train_restricted_pgac(workspace, policy_id, model_id, config, context=None, *, validation_id):
    from .differentiable_validation import read_differentiable_validation
    from .external_policy import load_external_policy
    from .external_policy_evaluation import _selection
    from .external_pgac_evaluation import normalize_original_evaluation
    from .external_expert_data import read_external_expert_dataset
    from .external_imitation import _load_members
    from .modeling import load_model
    from .policy_export import export_external_policy_state

    context = context or OperationContext()
    with workspace.lock('external_physical_policy_optimization'):
        selected = _selection(workspace,policy_id)
        if not selected['applicable']: raise ContractError('PGAC requires the exact executable original expert and Plant ancestry: '+selected['reason'])
        qualification = read_differentiable_validation(workspace,validation_id,candidate_id=model_id,original_id=selected['plant_id'])
        model = load_model(workspace,model_id); selected['physical'].compatible(model.interface)
        baseline = load_external_policy(selected['policy_root']); policy = load_external_policy(selected['policy_root'])
        binding = require_current_full_state(policy,model.interface)
        original = normalize_original_evaluation(selected,config.get('original_evaluation'))
        evaluation = original['evaluation']
        track_primary = evaluation.get('primary_metric') == 'tracking_rmse'
        resolved = normalize_pgac_config(config,model.interface,qualification['coverage']['domain']); resolved['original_evaluation'] = original
        if track_primary and not resolved['q']:
            raise ContractError('Tracking-primary PGAC requires positive training q for the selected output')
        substeps = binding['plant_substeps']; dataset_id = selected['dataset_id']
        arrays, interface, dataset, scaler, dataset_meta = read_external_expert_dataset(workspace,dataset_id)
        n = len(arrays['X']); i = model.interface
        shapes={'x':(n,i.nx),'reference':(n,i.ny),'d':(n,i.nd),'u':(n,i.nu),'previous_u':(n,i.nu),'trajectory_id':(n,)}
        if 'replay' in resolved:
            shapes.update(next_x=(n,i.nx),observation=(n,len(selected['expert']['observations'])),plant_y=(n,i.ny),time=(n,))
        raw = _load_members(selected['dataset_root']/'expert.npz',shapes,dataset_meta['files']['expert.npz'])
        identifiers = raw['trajectory_id'].astype(np.int64)
        if not np.array_equal(identifiers,raw['trajectory_id']) or not np.array_equal(raw['u'],arrays['y']): raise ContractError('PGAC physical rows/actions do not match the bound expert dataset')
        plan = read_json(selected['dataset_root']/'collection-plan.json'); split = plan['split']
        for name,index in [('train',0),('validation',1),('test',2)]:
            if not np.array_equal(np.isin(identifiers,split[name]),arrays['split_id']==index): raise ContractError('PGAC must retain exact original trajectory partitions')
        groups = np.unique(identifiers[arrays['split_id']==0])
        if not len(groups): raise ContractError('No training trajectories available for PGAC')
        model_widths = sum(layer.in_features+layer.out_features for layer in model.modules() if isinstance(layer,torch.nn.Linear))
        if sum(p.numel() for p in model.parameters())>2_000_000 or sum(v.numel() for v in model.state_dict().values())>8_000_000:
            raise ContractError('PGAC differentiable model exceeds the single-job parameter/state budget')
        widths = sum([policy.input_dim,*policy.config['hidden_dims'],policy.output_dim])+i.nx+i.ny+i.nu+i.nd+model_widths
        effective_batch = resolved['batch_size'] if 'replay' in resolved else min(resolved['batch_size'],int((arrays['split_id']==0).sum()))
        if effective_batch*resolved['horizon']*substeps*widths*8 > 5_000_000:
            raise ContractError('PGAC rollout exceeds its five-million logical activation budget; reduce batch_size or horizon')
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval(); initial_state = copy.deepcopy(policy.network.state_dict())
        if any(not torch.equal(initial_state[key],baseline.network.state_dict()[key]) for key in initial_state): raise ContractError('PGAC must initialize from actual imitation weights')
        policy.network.requires_grad_(True); policy.eval(); seed_training(resolved['seed']); rng = np.random.default_rng(resolved['seed'])
        optimizer = torch.optim.Adam(policy.network.parameters(),lr=resolved['learning_rate'])
        parents = list(dict.fromkeys([policy_id,model_id,validation_id,dataset_id,selected['expert_id'],selected['plant_id']]))
        effective = {**resolved,'network_config':policy.config,'physical_binding':binding,'source_mapping':dataset['source_mapping'],
            'scaler':policy.scaler,'action_constraints':policy.action_constraints,'physical_context':policy.physical_context,
            'gradient_semantics':qualification.get('gradient_semantics'),'control_substeps':substeps,'initial_weights_sha256':_state_hash(initial_state)}
        contract = training_contract(workspace,'external_physical_policy_optimization',parents,effective,interface.fingerprint)
        start = updates = 0; log = []; best_state = copy.deepcopy(initial_state); best_score = None
        replay = (SegmentReplay(resolved['replay'],selected,raw,plan,original,resolved['domain'])
                  if 'replay' in resolved else None)
        if config.get('resume_checkpoint'):
            checkpoint = load_checkpoint(workspace,config['resume_checkpoint'],contract,resolved['epochs'],expected_sha256=config.get('resume_checkpoint_sha256'))
            restore_training(checkpoint,policy.network,optimizer,numpy_generator=rng)
            if replay is not None:
                replay.restore(checkpoint.get('replay_state'), checkpoint['epoch'], _state_hash(checkpoint['model_state']))
            start,updates,log,best_state,best_score = (checkpoint[key] for key in ('epoch','updates','log','best_state','best_metric'))
        tensors = {name:torch.from_numpy(raw[name]) for name in ('x','reference','d','u','previous_u')}
        with workspace.stage('external_pgac_candidate',DataInterface.parse(interface.to_dict()),parents,config,compatible_parents=False) as stage:
            def checkpoint_epoch(epoch):
                return save_checkpoint(stage.root, contract, target_epochs=resolved["epochs"], model=policy.network,optimizer=optimizer,epoch=epoch,updates=updates,
                    best_state=best_state,best_metric=best_score,wait=0,log=log,numpy_generator=rng,
                    replay_state=replay.state if replay is not None else None)
            for name,value in [('resolved_config',resolved),('network_config',policy.config),('scaler',policy.scaler),('data_interface',policy.data_interface),
                ('action_constraints',policy.action_constraints),('physical_context',policy.physical_context)]: atomic_json(stage.root/(name+'.json'),value)
            torch.save(initial_state,stage.root/'initial_imitation_state.pt')
            saved = checkpoint_epoch(start)
            context.progress('pgac_initial_checkpoint',epoch=start, optimizer_updates=updates, **saved)
            baseline_validation = _predictor_validation(baseline,model,original,resolved,substeps,context)
            if start == 0 and best_score is None:
                best_score = baseline_validation['score']
                if track_primary and baseline_validation['hard_feasible']:
                    best_score = baseline_validation['tracking']['tracking_rmse']
                saved = checkpoint_epoch(start)
                context.progress('pgac_initial_checkpoint', epoch=start, optimizer_updates=updates, **saved)
            # Fixed expert training rows are a repeatable deployment/action parity probe.
            probe = torch.as_tensor(arrays['X'][arrays['split_id']==0][:64],dtype=torch.float64)
            probe_u=tensors['previous_u'][arrays['split_id']==0][:64]
            with torch.no_grad():
                baseline_actions=baseline(probe,previous_action=probe_u)
                if start==0 and not torch.equal(policy(probe,previous_action=probe_u),baseline_actions):
                    raise ContractError('Zero-step policy deployment actions differ from actual imitation')
            if replay is not None:
                training_eval = {'scenarios':[dict(row,split='train') for row in replay.train_scenarios],
                                 'evaluation':original['evaluation']}
                baseline_training = _predictor_validation(baseline,model,training_eval,resolved,substeps,context,split='train')
            first_update = None
            for epoch in range(start,resolved['epochs']):
                context.check(); epoch_losses = []
                if replay is not None:
                    replay.refresh(workspace,policy,epoch,_state_hash(policy.network.state_dict()),context,checkpoint_epoch)
                    sampler=replay.sampler()
                    order=range(resolved['replay']['updates_per_epoch'])
                    drawn=Counter()
                    sampled_terms=Counter()
                else:
                    order=rng.permutation(groups).tolist()
                for group in order:
                    context.check()
                    before = copy.deepcopy(policy.network.state_dict()); before_optimizer = copy.deepcopy(optimizer.state_dict())
                    if replay is not None:
                        samples,counts=sampler.draw(rng,resolved['batch_size'],resolved['horizon']); drawn.update(counts)
                        loss,batch_terms=_segment_loss(policy,model,sampler,samples,resolved,substeps,context,baseline_policy=baseline)
                        sampled_terms.update(batch_terms)
                        source={'replay_starts':[{ 'source':ep['source'],'episode':ep['id'],'t':t,'horizon':h}
                                                 for h,items in samples.items() for ep,t in items]}
                    else:
                        indexes=np.flatnonzero((identifiers==group)&(arrays['split_id']==0))
                        batch=rng.choice(indexes,size=min(len(indexes),resolved['batch_size']),replace=False)
                        loss=physical_rollout_loss(policy,model,tensors['x'][batch],tensors['reference'][batch],tensors['d'][batch],tensors['previous_u'][batch],tensors['u'][batch],resolved,substeps=substeps,context=context,baseline_policy=baseline)
                        source={'training_trajectory_id':int(group),'source_rows':batch.tolist()}
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(policy.network.parameters(),5.,error_if_nonfinite=True)); optimizer.step()
                    if any(not torch.isfinite(p).all() for p in policy.network.parameters()):
                        policy.network.load_state_dict(before); optimizer.load_state_dict(before_optimizer); raise FloatingPointError('Nonfinite PGAC update rolled back; baseline unchanged')
                    if first_update is None: first_update = {'state_sha256_before':_state_hash(before),'learning_rate':optimizer.param_groups[0]['lr'],
                        'weight_decay':optimizer.param_groups[0]['weight_decay'],'gradient_norm_before_clip':norm,**source}
                    updates += 1; epoch_losses.append(float(loss.detach()))
                result = _predictor_validation(policy,model,original,resolved,substeps,context)
                score = result['score']
                eligible = result['hard_feasible']
                if track_primary:
                    gate = tracking_gate(baseline_validation['tracking'], result['tracking'], evaluation)
                    eligible = eligible and gate['passed']
                    score = result['tracking']['tracking_rmse']
                if track_primary:
                    improved = eligible and (best_score is None or tracking_less(score, best_score))
                else:
                    improved = eligible and (best_score is None or score < best_score-evaluation['comparison_atol'])
                if improved: best_score,best_state = score,copy.deepcopy(policy.network.state_dict())
                event = {'epoch':epoch+1,'training_loss':math.fsum(epoch_losses)/len(epoch_losses),'predictor_validation_score':result['score'],
                    'selected_by_validation':improved,'optimizer_updates':updates}
                if track_primary:
                    event.update(predictor_validation_tracking=result['tracking'], predictor_validation_gate=gate,
                                 predictor_validation_hard_feasible=result['hard_feasible'], eligible_for_best_checkpoint=eligible)
                if replay is None:
                    event.update(training_conditions_updated=len(groups),training_trajectory_order=order)
                else:
                    with torch.no_grad():
                        # Use the same expert hold-in probe windows each epoch/seed, independent of draw RNG.
                        fixed={min(resolved['horizon'],len(ep['x'])):[] for ep in replay.expert}
                        for ep in replay.expert:
                            fixed[min(resolved['horizon'],len(ep['x']))].append((ep,0))
                        measured,components=_segment_loss(policy,model,sampler,fixed,resolved,substeps,context,baseline_policy=baseline)
                        training_score=_predictor_validation(policy,model,training_eval,resolved,substeps,context,split='train')
                        event.update(post_update_training_J=training_score['score'],post_update_training_runs=training_score['runs'],
                            post_update_training_probe_terms=components,
                            sampled_optimization_terms={key:value/len(order) for key,value in sampled_terms.items()},
                            post_update_action_delta_max=float((policy(probe,previous_action=probe_u)-baseline_actions).abs().max()),
                            sampled_strata=[{'source':a,'family':b,'phase':c,'count':v} for (a,b,c),v in sorted(drawn.items())],
                            student_control_steps=replay.state['total_steps'],student_episode_count=replay.state['next_id'])
                        if track_primary:
                            event['post_update_training_tracking'] = training_score['tracking']
                            event['post_update_training_probe_loss'] = float(measured)
                        else:
                            event['post_update_training_probe_J'] = float(measured)
                log.append(event)
                saved = checkpoint_epoch(epoch+1)
                context.progress('restricted_pgac_training',**event,**saved)
            last_sha=_state_hash(policy.network.state_dict())
            torch.save(policy.network.state_dict(),stage.root/'last_policy_state.pt')
            policy.network.load_state_dict(best_state,strict=True); policy.eval()
            selected_validation = _predictor_validation(policy,model,original,resolved,substeps,context)
            refs = {'imitation_policy':policy_id,'rollout_model':model_id,'model_validation':validation_id,'external_expert_dataset':dataset_id,
                'expert':selected['expert_id'],'plant':selected['plant_id']}
            torch.save(best_state,stage.root/'model.pt')
            manifest = export_external_policy_state(stage.root/'bundle',best_state,policy.config,policy.data_interface,policy.scaler,policy.action_constraints,policy.physical_context,
                provenance={**refs,'source_mapping':dataset['source_mapping'],'deployment_scope':policy.deployment_scope,'training_algorithm':'external_physical_policy_optimization'})
            atomic_json(stage.root/'policy.json',{'schema_version':1,'representation':'external_pgac_candidate','bundle':'bundle',**refs})
            report = {'algorithm':'restricted_differentiable_physical_rollout_policy_gradient','checkpoint_algorithm':'external_physical_policy_optimization',**refs,
                'resolved_config':resolved,'original_evaluation_config':original,'rollout_domain_checks_passed':True,'domain':resolved['domain'],
                'physical_interface':i.to_dict(),'physical_binding':binding,'gradient_semantics':qualification.get('gradient_semantics'),
                'initialized_from_actual_imitation':True,'initial_weights_sha256':_state_hash(initial_state),'initial_imitation_bundle_weights_sha256':file_hash(selected['policy_root']/'bundle/model.pt'),
                'first_update_in_this_run':first_update,'trained_epochs':len(log),'optimizer_updates':updates,'resumed_from_epoch':start,'training_signature':canonical_hash(contract),
                'training_trajectory_ids':groups.tolist(),'validation_trajectory_ids':[row['trajectory_id'] for row in plan['trajectories'] if row['split']=='validation'],
                'selection':'prediction_model_validation_only_including_unchanged_imitation_baseline','baseline_predictor_validation':baseline_validation,'selected_predictor_validation':selected_validation,
                'selected_imitation_baseline':_state_hash(best_state)==_state_hash(initial_state),'learned_critic':False,'ppo':False,'sac':False,
                'model_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'trainable_parameters':'only actual imitation raw MLP Parameters',
                'test_used_for_updates_or_checkpoint_selection':False,'original_plant_closed_loop':'not_run','policy_activation':'not_changed','requires_original_plant_gate':True,
                'resource_boundary':{'max_logical_rollout_activations':5_000_000,'max_model_parameters':2_000_000,'max_model_state_values':8_000_000,
                    'estimate':'effective_batch*horizon*plant_substeps*(sum(policy widths)+sum(model Linear input/output widths)+nx+ny+nu+nd)*8'},
                'export':{'directory':'bundle','manifest':'bundle/policy_manifest.json','details':manifest},'confirmation_sources':context.case_confirmation_sources,'log':log}
            report.update(best_epoch=max((e['epoch'] for e in log if e['selected_by_validation']),default=0),
                          last_weights_sha256=last_sha,selected_weights_sha256=_state_hash(best_state),
                          initial_action_parity='exact_same_immutable_weights_and_feature_contract')
            report['expert_validation_trajectory_ids']=report['validation_trajectory_ids']
            if track_primary:
                output = model.interface.outputs[tracking_index(evaluation, model.interface)]
                report.update(checkpoint_metric='tracking_rmse', tracking_output=output.name, tracking_unit=output.unit,
                              selection='minimum_validation_RMSE_passing_frozen_physical_and_baseline_relative_gates',
                              effort_reference='frozen_imitation_on_same_predictor_start_window_and_noise_no_expert_calls',
                              optimization_objective='mean_start_window_RMSE_in_output_spans_plus_matched_imitation_RMS_budget_excess_and_existing_protections')
            if original.get('condition_source')=='fresh_holdout':
                report['validation_trajectory_ids']=[]
            report['checkpoint_validation_conditions']=[
                {'name':row.get('name'),'family':row.get('family'),'steps':row['steps'],
                 'frozen_scenario_sha256':canonical_hash(row)}
                for row in original['scenarios'] if row['split']=='validation']
            if replay is not None:
                report['replay']=replay.summary()
                report['replay']['last_committed_checkpoint_bytes']=current_checkpoint(stage.root).stat().st_size
                report['baseline_predictor_training']=baseline_training
                report['original_plant_closed_loop']='student_training_only' if replay.state['total_steps'] else 'not_run'
            atomic_json(stage.root/'report.json',report); atomic_json(stage.root/'training_log.json',log); context.check()
            return stage.publish({'status':'verified','scope':'numerical_candidate_only','original_plant_gate':'pending','predictor_feasible':selected_validation['hard_feasible'],
                'checks':['qualified_differentiable_model','exact_imitation_initialization','current_full_state_causal_features','inherited_projection','bounded_real_rollout_gradients','validation_only_selection']},
                source={'algorithm':report['algorithm'],**refs,'requires_original_plant_gate':True},activate=False)
