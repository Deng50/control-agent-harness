"""Paired engineering metrics shared by predictor selection and original-plant release."""
import math
import sys

from .contracts import ContractError


RMS_FIELDS = ('tracking_rmse', 'action_rms', 'increment_rms')
BUDGET_FIELDS = ('max_action_rms_relative_increase', 'max_increment_rms_relative_increase')


def tracking_less(value, reference):
    """Strict decrease at the scale of the two metrics, including physical signals near zero."""
    return value < reference and not math.isclose(value, reference, rel_tol=16*sys.float_info.epsilon, abs_tol=0.)


def _relative_change(before, after):
    if before == 0:
        return None
    change = (after-before)/before
    return change if math.isfinite(change) else None


def tracking_index(objective, interface):
    """A physical RMSE has one output and one unit. Never pool unlike signals."""
    names = [output.name for output in interface.outputs]
    target = objective.get('tracking_output')
    if target is None and len(names) == 1:
        return 0
    if target not in names:
        raise ContractError('tracking_output must identify the physical output whose RMSE is the primary metric')
    return names.index(target)


def tracking_summary(rows):
    """Equal mean of per-trajectory RMS; the caller supplies only complete trajectories."""
    if not rows or len({row['trajectory_id'] for row in rows}) != len(rows):
        raise ContractError('Tracking comparison requires distinct complete trajectories')
    for row in rows:
        for field in RMS_FIELDS:
            value = row[field]
            if not math.isfinite(value) or value < 0:
                raise ContractError('Tracking comparison requires finite nonnegative RMS values')
    return {**{field: math.fsum(row[field] / len(rows) for row in rows) for field in RMS_FIELDS},
            'per_scenario': rows}


def tracking_gate(baseline, candidate, objective, *, validation=True):
    """Apply paired baseline-relative budgets identically on the predictor and the plant."""
    if baseline is None or candidate is None:
        return {'primary_metric': 'tracking_rmse', 'passed': False, 'checks': {'complete_pair': False}}
    base_rows = {row['trajectory_id']: row for row in baseline['per_scenario']}
    cand_rows = {row['trajectory_id']: row for row in candidate['per_scenario']}
    if not base_rows or base_rows.keys() != cand_rows.keys():
        return {'primary_metric': 'tracking_rmse', 'passed': False, 'checks': {'paired_trajectories': False}}
    base_rmse = baseline['tracking_rmse']
    measured = candidate['tracking_rmse']
    minimum = objective.get('min_relative_improvement', 0.)
    checks = {'minimum_improvement': not tracking_less(base_rmse*(1-minimum), measured)
              and (not validation or tracking_less(measured,base_rmse)) and (not minimum or base_rmse>0)}
    if 'max_worst_case_relative_regression' in objective:
        ratio = 1 + objective['max_worst_case_relative_regression']
        regressions = [key for key, row in cand_rows.items()
                       if tracking_less(base_rows[key]['tracking_rmse']*ratio, row['tracking_rmse'])]
        checks['worst_case_tracking'] = not regressions
    else:
        regressions = []
    limits = {}
    for metric, budget in zip(('action_rms', 'increment_rms'), BUDGET_FIELDS):
        limits[metric] = min(sys.float_info.max, baseline[metric] * (1 + objective[budget]))
        checks[metric] = not tracking_less(limits[metric],candidate[metric])
    change = _relative_change(base_rmse, measured)
    return {'primary_metric': 'tracking_rmse', 'passed': all(checks.values()), 'checks': checks,
            'baseline_rmse': base_rmse, 'candidate_rmse': candidate['tracking_rmse'],
            'relative_improvement': -change if change is not None else None,
            'action_limits': limits, 'regressing_trajectory_ids': regressions}


def paired_tracking_report(baseline, candidate):
    """Expose signed changes on the plant. A positive tracking improvement is a reduction."""
    if baseline is None or candidate is None:
        return {'status':'incomplete_pair'}
    base = {row['trajectory_id']:row for row in baseline['per_scenario']}
    pairs = []
    families = {}
    for row in candidate['per_scenario']:
        other = base.get(row['trajectory_id'])
        if other is None:
            raise ContractError('Cannot report changes for unmatched original-Plant trajectories')
        pairs.append((other,row))
        if row.get('family') is not None:
            families.setdefault(row['family'],[]).append((other,row))
    if len(pairs)!=len(base):
        raise ContractError('Incomplete original-Plant comparison')

    def changes(group):
        result = {}
        for metric in RMS_FIELDS:
            before = math.fsum(left[metric]/len(group) for left,_ in group)
            after = math.fsum(right[metric]/len(group) for _,right in group)
            result[metric] = {'imitation':before,'candidate':after,'delta':after-before,
                              'relative_change':_relative_change(before,after)}
        return result

    return {'status':'complete','overall':changes(pairs),
            'by_family':{name:changes(group) for name,group in families.items()},
            'per_scenario':[{**{key:right.get(key) for key in ('trajectory_id','name','family','noise_seed')},
                             **changes([(left,right)])} for left,right in pairs]}
