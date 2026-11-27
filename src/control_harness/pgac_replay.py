"""Episode-local, stratified starting states for qualified external PGAC.

Original-Plant collection is a CPU inference task. Only the qualified dynamics
participates in the later differentiable window. Retained episodes are immutable
CPU tensors and are included in the same atomic checkpoint as Adam and RNG.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re

import numpy as np
import torch

from .contracts import ContractError
from .pgac_scenarios import condition_hash, normalize_schedule, scenario_series, observation_noise
from .external_expert import _bounded, _vector
from .external_policy_evaluation import _features, _feature_vector


def normalize_replay(value):
    allowed = {
        'mode',
        'capacity_steps',
        'refresh_every',
        'updates_per_epoch',
        'student_fraction',
        'scenarios',
        'expert_families',
        'family_weights',
        'phase_weights',
    }
    if (
        not isinstance(value, dict)
        or set(value) - allowed
        or value.get('mode') not in ('expert_segments', 'balanced_starts', 'online')
    ):
        raise ContractError('replay requires an explicit supported mode and settings')
    result = {
        'capacity_steps': 4096,
        'refresh_every': 1,
        'updates_per_epoch': 6,
        'student_fraction': 0.5 if value['mode'] == 'online' else 0.0,
        **value,
    }
    for key, maximum in [('capacity_steps', 100000), ('refresh_every', 100000), ('updates_per_epoch', 10000)]:
        if type(result[key]) is not int or not 1 <= result[key] <= maximum:
            raise ContractError(f'replay.{key} must be an integer in [1,{maximum}]')
    fraction = result['student_fraction']
    if type(fraction) not in (float, int) or not 0 <= fraction <= 1:
        raise ContractError('replay.student_fraction must be in [0,1]')
    if result['mode'] != 'online' and (fraction or 'scenarios' in result):
        raise ContractError('Expert-only replay draws exclusively from existing expert training trajectories')
    for key in ('family_weights', 'phase_weights', 'expert_families'):
        if key in result and not isinstance(result[key], dict):
            raise ContractError(f'replay.{key} must be an explicit mapping')
    if any(
        not isinstance(k, str) or not k.strip() or type(v) not in (float, int) or not np.isfinite(v) or v <= 0
        for weights in ('family_weights', 'phase_weights')
        for k, v in result.get(weights, {}).items()
    ):
        raise ContractError('Stratum sampling weights must be finite and positive')
    if set(result.get('phase_weights', {})) - {
        'initial',
        'interior',
        'near_boundary',
        'reference_change',
        'reversal',
    }:
        raise ContractError('Unknown replay sampling phase')
    if any(
        not isinstance(k, str) or not k.isdecimal() or not isinstance(v, str) or not v.strip()
        for k, v in result.get('expert_families', {}).items()
    ):
        raise ContractError(
            'Expert family keys must be training trajectory IDs and values nonempty family names'
        )
    return result


def case_from_plan(row):
    return {
        'name': row.get('name', 'expert-' + str(row['trajectory_id'])),
        'family': row.get('family', 'expert-' + str(row['trajectory_id'])),
        'initial_state': row['x0'],
        'initial_previous_action': row['initial_action'],
        **{
            k: row[k]
            for k in (
                'reference',
                'disturbance',
                'steps',
                'reference_sequence',
                'disturbance_sequence',
                'observation_noise_std',
                'noise_seed',
            )
            if k in row
        },
    }


def _case(case, interface, expert):
    allowed = {
        'name',
        'family',
        'steps',
        'initial_state',
        'initial_previous_action',
        'reference',
        'disturbance',
        'reference_sequence',
        'disturbance_sequence',
        'observation_noise_std',
        'noise_seed',
    }
    if (
        not isinstance(case, dict)
        or set(case) - allowed
        or type(case.get('steps')) is not int
        or not 1 <= case['steps'] <= 100000
    ):
        raise ContractError('Invalid replay scenario or steps')
    name = case.get('name')
    family = case.get('family', name)
    if not isinstance(name, str) or not name.strip() or not isinstance(family, str) or not family.strip():
        raise ContractError('Replay scenarios require a nonempty name and family')
    fields = {
        'x0': _vector(case.get('initial_state'), interface.nx, 'initial_state'),
        'initial_action': _vector(case.get('initial_previous_action'), interface.nu, 'initial_previous_action'),
        'reference': _vector(case.get('reference'), interface.ny, 'reference'),
        'disturbance': _vector(case.get('disturbance'), interface.nd, 'disturbance'),
    }
    for key, variables in [
        ('x0', interface.states),
        ('initial_action', interface.controls),
        ('reference', interface.outputs),
        ('disturbance', interface.disturbances),
    ]:
        _bounded(fields[key], variables, key)
    row = {**fields, 'steps': case['steps'], **normalize_schedule(case, interface, expert)}
    ref = scenario_series(row, 'reference')
    d = scenario_series(row, 'disturbance')
    noise = observation_noise(row, len(expert['observations']))
    return {
        'name': name,
        'family': family,
        'x0': np.asarray(row['x0']),
        'initial_action': np.asarray(row['initial_action']),
        'reference': ref,
        'd': d,
        'noise': noise,
        'physical_signature': condition_hash(row),
        'scenario': case,
    }


def _same_prefix(a, b):
    if not np.array_equal(a['x0'], b['x0']) or not np.array_equal(a['initial_action'], b['initial_action']):
        return False
    length = min(len(a['reference']), len(b['reference']))
    return all(np.array_equal(a[key][:length], b[key][:length]) for key in ('reference', 'd'))


def _phase_groups(episode, interface):
    """Stratify on the events and distances actually present in this episode."""
    r = episode['reference'].numpy()
    x = episode['x'].numpy()
    lower = np.array([v.lower for v in interface.states])
    upper = np.array([v.upper for v in interface.states])
    boundary = np.any(np.minimum(x - lower, upper - x) / (upper - lower) < 0.08, axis=1)
    changes = np.flatnonzero(np.any(r[1:] != r[:-1], axis=1)) + 1
    phases = np.full(len(x), 'interior', dtype=object)
    phases[: min(3, len(x))] = 'initial'
    for start in changes:
        label = 'reversal' if np.any(r[start] * r[start - 1] < 0) else 'reference_change'
        phases[max(0, start - 1) : start + 3] = label
    phases[boundary] = 'near_boundary'
    return {key: np.flatnonzero(phases == key) for key in sorted(set(phases))}


def _episode(arrays, **metadata):
    return {
        **metadata,
        **{
            key: torch.as_tensor(np.array(values, copy=True), dtype=torch.float64)
            for key, values in arrays.items()
        },
    }


class SegmentReplay:
    def __init__(self, settings, selection, raw, plan, original, domain):
        self.settings = settings
        self.selection = selection
        self.domain = domain
        self.expert = []
        self.columns, reason = _features(selection)
        if self.columns is None:
            raise ContractError(reason)
        self.interface = selection['physical']
        self.observations = selection['expert']['observations']
        self.substeps = plan['plant_substeps']
        i = self.interface
        self.widths = {
            'x': i.nx,
            'next_x': i.nx,
            'u': i.nu,
            'previous_u': i.nu,
            'observation': len(self.observations),
            'noise': len(self.observations),
            'reference': i.ny,
            'd': i.nd,
        }
        self.cases = []
        self.state = {
            'episodes': [],
            'total_steps': 0,
            'next_id': 0,
            'epoch_collecting': None,
            'completed_cases': 0,
        }
        train_rows = [row for row in plan['trajectories'] if row['split'] == 'train']
        families = settings.get('expert_families', {})
        if set(families) - {str(row['trajectory_id']) for row in train_rows}:
            raise ContractError('Expert families must refer only to training trajectories')
        defaults = []
        for row in train_rows:
            tid = row['trajectory_id']
            ix = np.flatnonzero(raw['trajectory_id'] == tid)
            if len(ix) != row['steps'] or not np.allclose(raw['time'][ix], row['time'], rtol=0, atol=1e-10):
                raise ContractError('Expert time rows do not match the bound contiguous trajectory')
            case = case_from_plan(row)
            case['family'] = families.get(str(tid), case['family'])
            defaults.append(case)
            clean = np.column_stack(
                [
                    raw['x' if obs['source'] == 'state' else 'plant_y'][ix, obs['index']]
                    for obs in self.observations
                ]
            )
            arrays = {
                key: raw[key][ix] for key in ('x', 'next_x', 'u', 'previous_u', 'reference', 'd', 'observation')
            }
            if (
                not np.array_equal(arrays['x'][0], row['x0'])
                or not np.array_equal(arrays['previous_u'][0], row['initial_action'])
                or not np.array_equal(arrays['next_x'][:-1], arrays['x'][1:])
                or not np.array_equal(arrays['u'][:-1], arrays['previous_u'][1:])
            ):
                raise ContractError('Expert replay requires a contiguous own-state and actual-action chain')
            arrays['noise'] = arrays['observation'] - clean
            self.expert.append(
                _episode(arrays, family=case['family'], name=case['name'], id=tid, source='expert')
            )
        self.train_scenarios = defaults
        self.declared_conditions = 0
        if settings['mode'] == 'online':
            spec = selection['plant_spec']
            if spec is not None and spec['backend'] not in ('python_callable', 'python_factory'):
                raise ContractError(
                    'Online replay requires the bound Python or registered Torch original Plant; use expert_segments for this adapter'
                )
            rows = settings.get('scenarios', defaults)
            if not isinstance(rows, list) or not 1 <= len(rows) <= 10000:
                raise ContractError('Online collection requires explicit nonempty training scenarios')
            self.declared_conditions = len(rows)
            held = [
                _case(
                    {k: v for k, v in row.items() if k != 'split'} | {'name': row.get('name', 'held')},
                    self.interface,
                    selection['expert'],
                )
                for row in original['scenarios']
            ]
            if original.get('condition_source') == 'fresh_holdout':
                # Previously evaluated expert partitions stay held out even when final
                # evaluation moves to a new blind set. Names, seeds and duration do not
                # make the same commanded physical prefix available for student updates.
                held.extend(
                    _case(case_from_plan(row), self.interface, selection['expert'])
                    for row in plan['trajectories']
                    if row['split'] in {'validation', 'test'}
                )
            distinct = {}
            total_declared = 0
            width = (
                2 * self.interface.nx
                + 3 * self.interface.nu
                + self.interface.ny
                + self.interface.nd
                + 2 * len(self.observations)
            )
            if settings['capacity_steps'] * width > 5_000_000:
                raise ContractError('Replay capacity exceeds the five-million stored scalar budget')
            for row in rows:
                total_declared += row['steps'] if isinstance(row, dict) and type(row.get('steps')) is int else 0
                if total_declared * width > 5_000_000:
                    raise ContractError('Replay condition declarations exceed the collection scalar budget')
                if (
                    isinstance(row, dict)
                    and type(row.get('steps')) is int
                    and row['steps'] > settings['capacity_steps']
                ):
                    raise ContractError('capacity_steps must hold each complete student episode')
                case = _case(row, self.interface, selection['expert'])
                if any(_same_prefix(case, c) for c in held):
                    raise ContractError(
                        'Student training scenario overlaps a frozen validation/test physical prefix'
                    )
                key = case['physical_signature']
                if key not in distinct or len(case['reference']) > len(distinct[key]['reference']):
                    distinct[key] = case
            self.cases = list(distinct.values())
            self.train_scenarios = [case['scenario'] for case in self.cases]
            total = sum(len(case['reference']) for case in self.cases)
            if total > settings['capacity_steps'] or total * self.substeps > 500000:
                raise ContractError(
                    'Replay capacity or original Plant subtick budget cannot hold one full condition refresh'
                )
        names = {ep['family'] for ep in self.expert} | {c['family'] for c in self.cases}
        if set(settings.get('family_weights', {})) - names:
            raise ContractError('Unknown replay family weight')

    def refresh(self, workspace, policy, epoch, policy_hash, context, save):
        if self.settings['mode'] != 'online' or epoch % self.settings['refresh_every']:
            return
        from .external_pgac_evaluation import _CoveragePlant
        from .model_adapters import load_python_adapter
        from .plants import _TorchAdapter

        selected = self.selection
        if self.state['epoch_collecting'] != epoch:
            self.state.update(epoch_collecting=epoch, completed_cases=0)
        for index in range(self.state['completed_cases'], len(self.cases)):
            case = self.cases[index]
            plant = (
                load_python_adapter(
                    selected['plant_root'],
                    selected['plant_spec'],
                    expected_file_hashes=selected['plant_material']['files'],
                )
                if selected['plant_spec'] is not None
                else _TorchAdapter(workspace, selected['plant_id'])
            )
            try:
                episode = self.collect(
                    _CoveragePlant(plant, self.domain), policy, case, epoch, policy_hash, context
                )
            finally:
                plant.close()
            pool = self.state['episodes']
            pool.append(episode)
            while sum(len(e['x']) for e in pool) > self.settings['capacity_steps']:
                pool.pop(0)
            self.state['completed_cases'] = index + 1
            self.state['next_id'] += 1
            self.state['total_steps'] += len(episode['x'])
            # Persist each completed original-Plant episode before announcing it.
            saved = save(epoch)
            context.progress(
                'pgac_student_episode_committed',
                training_epoch=epoch + 1,
                episode=episode['id'],
                condition=case['name'],
                student_control_steps=self.state['total_steps'],
                **saved,
            )

    def restore(self, stored, epoch, policy_sha256):
        if not isinstance(stored, dict) or set(stored) != set(self.state):
            raise ContractError('Online checkpoint missing the expected replay state')
        if any(
            type(stored[k]) is not int or stored[k] < 0 for k in ('total_steps', 'next_id', 'completed_cases')
        ):
            raise ContractError('Invalid replay checkpoint counters')
        collecting = stored['epoch_collecting']
        if collecting is not None and (type(collecting) is not int or not 0 <= collecting <= epoch):
            raise ContractError('Replay checkpoint collection cursor disagrees with the optimizer epoch')
        if stored['completed_cases'] > len(self.cases):
            raise ContractError('Replay checkpoint exceeds the frozen training condition list')
        episodes = stored['episodes']
        if not isinstance(episodes, list) or len(episodes) > stored['next_id']:
            raise ContractError('Replay checkpoint has an invalid episode inventory')
        completed = stored['completed_cases']
        if collecting is None:
            if episodes or completed or stored['next_id'] or stored['total_steps'] or (self.cases and epoch):
                raise ContractError('Replay checkpoint has data without a committed collection')
        else:
            frequency = self.settings['refresh_every']
            if (
                not self.cases
                or not completed
                or collecting % frequency
                or epoch - collecting > frequency
                or (epoch > collecting and completed != len(self.cases))
            ):
                raise ContractError('Replay checkpoint refresh does not match the frozen collection schedule')
            sweeps = collecting // frequency
            lengths = [len(case['reference']) for case in self.cases]
            if stored['next_id'] != sweeps * len(self.cases) + completed or stored[
                'total_steps'
            ] != sweeps * sum(lengths) + sum(lengths[:completed]):
                raise ContractError('Replay counters would skip or repeat committed original-Plant decisions')
        retained = 0
        hashes = {}
        first_index = stored['next_id'] - len(episodes)
        for index, ep in enumerate(episodes, start=first_index):
            if not self.cases or not isinstance(ep, dict):
                raise ContractError('Invalid replay episode entry')
            collection, slot = divmod(index, len(self.cases))
            case = self.cases[slot]
            source_epoch = collection * self.settings['refresh_every']
            stamp = ep.get('policy_sha256')
            if (
                ep.get('physical_signature') != case['physical_signature']
                or ep.get('source') != 'student'
                or ep.get('id') != index
                or ep.get('epoch') != source_epoch
                or ep.get('name') != case['name']
                or ep.get('family') != case['family']
                or not isinstance(stamp, str)
                or not re.fullmatch('[0-9a-f]{64}', stamp)
                or hashes.setdefault(source_epoch, stamp) != stamp
            ):
                raise ContractError(
                    'Replay checkpoint episode contradicts its frozen condition or policy provenance'
                )
            if source_epoch == epoch and stamp != policy_sha256:
                raise ContractError('Partial refresh was collected with different checkpoint policy weights')
            n = len(case['reference'])
            for key, width in self.widths.items():
                t = ep.get(key)
                if (
                    not isinstance(t, torch.Tensor)
                    or t.shape != (n, width)
                    or t.dtype != torch.float64
                    or not torch.isfinite(t).all()
                ):
                    raise ContractError('Replay checkpoint contains an invalid physical series: ' + key)
            if not torch.equal(ep['x'][1:], ep['next_x'][:-1]) or not torch.equal(
                ep['u'][:-1], ep['previous_u'][1:]
            ):
                raise ContractError('Replay checkpoint has a broken state/action chain')
            if (
                not np.array_equal(ep['x'][0].numpy(), case['x0'])
                or not np.array_equal(ep['previous_u'][0].numpy(), case['initial_action'])
                or any(not np.array_equal(ep[key].numpy(), case[key]) for key in ('reference', 'd', 'noise'))
            ):
                raise ContractError(
                    'Replay checkpoint differs from the frozen starting state/action or exogenous sequence'
                )
            for j, observation in enumerate(self.observations):
                if observation['source'] == 'state':
                    measured = ep['x'][:, observation['index']] + ep['noise'][:, j]
                    if not torch.equal(measured, ep['observation'][:, j]):
                        raise ContractError(
                            'Replay observation does not match the own state and committed observation noise'
                        )
            retained += n
        if (
            first_index
            and retained + len(self.cases[(first_index - 1) % len(self.cases)]['reference'])
            <= self.settings['capacity_steps']
        ):
            raise ContractError(
                'Replay checkpoint discarded an episode that fits in the configured FIFO capacity'
            )
        if retained > min(stored['total_steps'], self.settings['capacity_steps']):
            raise ContractError('Replay checkpoint exceeds the committed collection or capacity')
        self.state = stored

    def collect(self, plant, policy, case, epoch, policy_hash, context):
        steps = len(case['reference'])
        # Contiguous arrays bound per-episode memory by channels * steps. A list of
        # tiny NumPy arrays for every sample would dominate the payload on small plants.
        arrays = {key: np.empty((steps, width), dtype=np.float64) for key, width in self.widths.items()}
        for key in ('reference', 'd', 'noise'):
            arrays[key][:] = case[key]
        plant.reset(case['x0'])
        previous = case['initial_action'].copy()
        y = _bounded(plant.output(), self.interface.outputs, 'current student output')
        for k in range(steps):
            context.check()
            reference = arrays['reference'][k]
            d = arrays['d'][k]
            x = plant.state
            obs = (
                np.array(
                    [x[o['index']] if o['source'] == 'state' else y[o['index']] for o in self.observations]
                )
                + arrays['noise'][k]
            )
            features = _feature_vector(self.columns, obs, reference, d, previous)
            action = np.asarray(policy.predict(features, previous_action=previous)).reshape(self.interface.nu)
            for _ in range(self.substeps):
                context.check()
                xn = plant.step(action, d)
                # The Python adapter verifies state-only visibility at the actual
                # internal state. A transient violation may vanish by the next decision.
                y = _bounded(plant.output(), self.interface.outputs, 'student output at Plant subtick')
            arrays['x'][k] = x
            arrays['next_x'][k] = xn
            arrays['observation'][k] = obs
            arrays['previous_u'][k] = previous
            arrays['u'][k] = action
            previous = action
        return dict(
            **{key: torch.from_numpy(values) for key, values in arrays.items()},
            family=case['family'],
            name=case['name'],
            source='student',
            id=self.state['next_id'],
            epoch=epoch,
            policy_sha256=policy_hash,
            physical_signature=case['physical_signature'],
        )

    def sampler(self):
        return SegmentSampler(self.expert, self.state['episodes'], self.settings, self.interface)

    def summary(self):
        episodes = self.state['episodes']
        return {
            'mode': self.settings['mode'],
            'expert_training_steps': sum(len(ep['x']) for ep in self.expert),
            'expert_training_episodes': len(self.expert),
            'declared_student_conditions': self.declared_conditions,
            'effective_student_conditions': len(self.cases),
            'student_control_steps': self.state['total_steps'],
            'original_plant_substeps': self.state['total_steps'] * self.substeps,
            'extra_mpc_calls': 0,
            'student_episodes_collected': self.state['next_id'],
            'retained_steps': sum(len(ep['x']) for ep in episodes),
            'retained_tensor_bytes': sum(
                ep[key].numel() * ep[key].element_size() for ep in episodes for key in self.widths
            ),
            'capacity_steps': self.settings['capacity_steps'],
            'capacity_tensor_bytes': self.settings['capacity_steps'] * sum(self.widths.values()) * 8,
            'checkpoint_frequency': 'after_each_committed_student_episode_and_completed_optimizer_epoch',
            'retained_episodes': [
                {
                    'id': ep['id'],
                    'family': ep['family'],
                    'policy_sha256': ep['policy_sha256'],
                    'steps': len(ep['x']),
                    'condition': ep['name'],
                }
                for ep in episodes
            ],
        }


class SegmentSampler:
    def __init__(self, expert, student, settings, interface):
        self.settings = settings
        self.groups = {}
        for source, episodes in [('expert', expert), ('student', student)]:
            families = defaultdict(lambda: defaultdict(list))
            for episode in episodes:
                for phase, indices in _phase_groups(episode, interface).items():
                    families[episode['family']][phase].append((episode, indices))
            self.groups[source] = families

    def draw(self, rng, count, horizon):
        samples = defaultdict(list)
        stats = Counter()
        for _ in range(count):
            source = 'student' if rng.random() < self.settings['student_fraction'] else 'expert'
            families = self.groups[source]
            names = sorted(families)
            weights = np.array([self.settings.get('family_weights', {}).get(key, 1.0) for key in names])
            weights = weights / weights.max()
            family = names[int(rng.choice(len(names), p=weights / weights.sum()))]
            phases = families[family]
            if self.settings.get('phase_weights'):
                names = sorted(phases)
                weights = np.array([self.settings['phase_weights'].get(k, 1.0) for k in names])
                weights = weights / weights.max()
                phase = names[int(rng.choice(len(names), p=weights / weights.sum()))]
            else:
                phase = rng.choice(sorted(phases))
            options = phases[phase]
            ep, ix = options[rng.integers(len(options))]
            start = int(rng.choice(ix))
            length = (
                horizon if self.settings['mode'] == 'balanced_starts' else min(horizon, len(ep['x']) - start)
            )
            samples[length].append((ep, start))
            stats[(source, family, str(phase))] += 1
        return samples, stats

    @staticmethod
    def batch(items, length):
        current = {key: torch.stack([ep[key][t] for ep, t in items]) for key in ('x', 'previous_u', 'u')}
        series = {
            key: torch.stack([ep[key][t : t + length] for ep, t in items])
            for key in ('reference', 'd', 'noise')
        }
        mask = torch.tensor([ep['source'] == 'expert' for ep, t in items], dtype=torch.bool)
        return current, series, mask
