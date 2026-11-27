"""Physical scenario schedules shared by expert collection, replay and evaluation."""
from collections import defaultdict

import numpy as np

from .contracts import ContractError, canonical_hash


SCHEDULE_FIELDS = {'family', 'reference_sequence', 'disturbance_sequence',
                   'observation_noise_std', 'noise_seed'}


def scenario_series(row, key):
    """Value available to the controller at each decision, before Plant substeps."""
    sequence = row.get(key + '_sequence')
    if sequence is not None:
        return np.asarray(sequence, dtype=np.float64)
    value = np.asarray(row[key], dtype=np.float64)
    return np.broadcast_to(value, (row['steps'], len(value)))


def observation_noise(row, width):
    std = row.get('observation_noise_std')
    if std is None or not np.any(std):
        return np.zeros((row['steps'], width), dtype=np.float64)
    return np.random.default_rng(row['noise_seed']).normal(size=(row['steps'], width)) * std


def normalize_schedule(row, interface, expert):
    result = {}
    if 'family' in row:
        family = row['family']
        if not isinstance(family, str) or not family.strip():
            raise ContractError('Scenario family must be a nonempty label')
        result['family'] = family
    for key, variables in [('reference', interface.outputs), ('disturbance', interface.disturbances)]:
        name = key + '_sequence'
        if name not in row:
            continue
        try:
            values = np.asarray(row[name])
        except (ValueError, TypeError) as error:
            raise ContractError(name + ' requires a rectangular numeric array') from error
        if (values.dtype.kind not in 'fiu' or values.shape != (row['steps'], len(variables))
                or not np.isfinite(values).all() or not np.array_equal(values[0], row[key])):
            raise ContractError(name + ' must have one finite vector per decision and start at the declared initial value')
        if np.any(values < [v.lower for v in variables]) or np.any(values > [v.upper for v in variables]):
            raise ContractError(name + ' exceeds physical bounds')
        result[name] = values.astype(float).tolist()
    if 'observation_noise_std' in row or 'noise_seed' in row:
        try:
            std = np.asarray(row.get('observation_noise_std'))
        except (ValueError, TypeError) as error:
            raise ContractError('Observation noise requires a numeric vector in observation order') from error
        seed = row.get('noise_seed')
        if (std.dtype.kind not in 'fiu' or std.shape != (len(expert['observations']),)
                or not np.isfinite(std).all() or np.any(std < 0)
                or type(seed) is not int or not 0 <= seed < 2**32):
            raise ContractError('Observation noise requires nonnegative standard deviations in observation order and an explicit uint32 noise_seed')
        result.update(observation_noise_std=std.astype(float).tolist(), noise_seed=seed)
    if expert['backend'] == 'matlab_function' and set(result) - {'family'}:
        raise ContractError('Time-varying schedules and observation noise require the Python closed-loop adapter')
    return result


def condition_hash(row):
    """Ignore names, duration, redundant constant samples and ineffective noise seeds."""
    fields = {key: [float(v) if v else 0.0 for v in row[key]]
              for key in ('x0', 'reference', 'disturbance', 'initial_action')}
    for key in ('reference', 'disturbance'):
        if key + '_sequence' in row:
            values = scenario_series(row, key)
            changes = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
            if len(changes):
                fields[key + '_changes'] = [[int(k), [float(v) if v else 0.0 for v in values[k]]] for k in changes]
    if np.any(row.get('observation_noise_std', [])):
        fields['noise'] = [row['noise_seed'], row['observation_noise_std']]
    return canonical_hash(fields)


def ensure_new_physical_conditions(source_rows, new_rows):
    """A shorter run or a different noise realization cannot make a seen command profile blind."""
    prefixes = defaultdict(list)

    def key(row):
        return tuple(tuple(row[field]) for field in ('x0', 'initial_action', 'reference', 'disturbance'))

    for row in source_rows:
        prefixes[key(row)].append(row)
    for row in new_rows:
        bucket = prefixes[key(row)]
        for prior in bucket:
            length = min(row['steps'], prior['steps'])
            if all(np.array_equal(scenario_series(row, k)[:length], scenario_series(prior, k)[:length])
                   for k in ('reference', 'disturbance')):
                raise ContractError('Fresh evaluation condition overlaps a previous physical prefix (initial state/action and exogenous schedule)')
        bucket.append(row)
