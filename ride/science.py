"""Preregistered, simulated intervention comparisons for RIDE.

Do not infer human causal effects from simulator trajectories. Trials are
eligible only when initial state, route progress, and physically executed ego
interventions are comparable. Missing measurements fail closed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Protocol:
    progress_m: float = 3.0
    # V5 shared-PID warmup should support tighter starting-state parity.
    # Fix BEFORE looking at V5 outcome rollouts, not post-hoc.
    init_position_m: float = 0.5
    init_speed_mps: float = 0.5
    min_action_xy_difference_m: float = 0.5
    min_action_speed_difference_mps: float = 0.35
    braking_mps2: float = -4.0
    ttc_s: float = 1.5
    contrast_actions: tuple[tuple[str, str], ...] = (
        ('early_slow', 'late_slow'), ('mild_early', 'mild_late')
    )
    families: tuple[str, ...] = ('tm', 'behavior')


def same_initial(a: dict, b: dict, p: Protocol) -> tuple[bool, str]:
    left, right = a.get('initial_states') or {}, b.get('initial_states') or {}
    for role in ('ego', 'follower_a', 'follower_b'):
        s, t = left.get(role), right.get(role)
        if s is None or t is None:
            return False, 'missing_initial_state:' + role
        try:
            xyz1 = [float(v) for v in s['xyz']]
            xyz2 = [float(v) for v in t['xyz']]
            v1, v2 = float(s['speed_mps']), float(t['speed_mps'])
            if len(xyz1) != 3 or len(xyz2) != 3 or not all(
                math.isfinite(v) for v in xyz1 + xyz2 + [v1, v2]
            ):
                return False, 'malformed_initial_state:' + role
            if math.dist(xyz1, xyz2) > p.init_position_m:
                return False, 'initial_position_mismatch:' + role
            if abs(v1-v2) > p.init_speed_mps:
                return False, 'initial_speed_mismatch:' + role
        except (TypeError, KeyError, ValueError):
            return False, 'malformed_initial_state:' + role
    return True, ''


def distinct_executed_ego_actions(a: dict, b: dict, p: Protocol) -> tuple[bool, str]:
    """Verify physical action contrast, not merely different profile names.

    In particular, a CARLA lane-change command that does not actually move the
    ego vehicle must NOT be counted as a successful intervention.
    """
    ta, tb = a.get('trace'), b.get('trace')
    if not ta or not tb or len(ta) != len(tb):
        return False, 'missing_or_unpaired_action_trace'
    speed_diff = pos_diff = 0.0
    for x, y in zip(ta, tb):
        try:
            if abs(float(x['t'])-float(y['t'])) > 1e-3:
                return False, 'mismatched_action_trace_time'
            speed_diff = max(speed_diff, abs(float(x['ego_speed_mps'])-float(y['ego_speed_mps'])))
            pos_diff = max(pos_diff, math.dist(x['xy_by_role']['ego'], y['xy_by_role']['ego']))
        except (ValueError, TypeError, KeyError):
            return False, 'malformed_action_trace'
    if not math.isfinite(speed_diff) or not math.isfinite(pos_diff):
        return False, 'nonfinite_action_trace'
    if speed_diff < p.min_action_speed_difference_mps and pos_diff < p.min_action_xy_difference_m:
        return False, 'ego_actions_not_physically_distinct'
    return True, ''


def validate_pair(a: dict, b: dict, p: Protocol) -> tuple[bool, str, float | None]:
    am, bm = a.get('metrics') or {}, b.get('metrics') or {}
    if not am.get('valid') or not bm.get('valid'):
        return False, 'invalid_trial', None
    # The exploratory intersection builder only emits a red-trigger-motion
    # proxy and goal-distance-reduction. Neither certifies signal legality,
    # right-of-way or route arc progress. Prevent scientific false positives.
    if a.get('archetype')=='intersection' or b.get('archetype')=='intersection':
        if (am.get('traffic_light_legality_verified') is not True or
                bm.get('traffic_light_legality_verified') is not True):
            return False, 'intersection_legality_unverified', None
        if (am.get('progress_measure')!='verified_route_arc_length' or
                bm.get('progress_measure')!='verified_route_arc_length'):
            return False, 'intersection_route_progress_unverified', None
    # For map-grounded non-convoy scenes, unrecorded mechanism execution
    # fails closed. Legacy synthetic convoy fixtures remain compatible.
    for trial in (a,b):
        if trial.get('archetype') in ('cut_in','support_gap','intersection') and \
                trial.get('metrics',{}).get('mechanism_executed') is not True:
            return False, 'scenario_maneuver_not_executed', None
    if not am.get('mechanism_executed', True) or not bm.get('mechanism_executed', True):
        return False, 'scenario_maneuver_not_executed', None
    ok, why = same_initial(a, b, p)
    if not ok:
        return False, why, None
    if am.get('ego_collision') or bm.get('ego_collision'):
        return False, 'ego_collision_in_either_arm', None
    try:
        # These must be route-goal progress, not odometer/path length.
        d1, d2 = float(am['ego_progress_m']), float(bm['ego_progress_m'])
        if not math.isfinite(d1) or not math.isfinite(d2):
            return False, 'nonfinite_progress', None
        delta = abs(d1-d2)
    except (KeyError, TypeError, ValueError):
        return False, 'missing_progress', None
    if delta > p.progress_m:
        return False, 'progress_unmatched', delta
    ok, why = distinct_executed_ego_actions(a, b, p)
    if not ok:
        return False, why, delta
    return True, '', delta


def _below(row: dict, metric: str, threshold: float) -> bool:
    value = row.get(metric)
    if value is None:
        return False
    try:
        number = float(value)
        return number < threshold if math.isfinite(number) else False
    except (ValueError, TypeError):
        return False


def assess_pair(a: dict, b: dict, p: Protocol) -> dict[str, Any]:
    """Signed effect for a preregistered (a,b) action pair; no cherry-picking."""
    am, bm = a.get('metrics') or {}, b.get('metrics') or {}
    ok, reason, delta = validate_pair(a, b, p)
    if not ok:
        return {'eligible': False, 'reason': reason, 'delta_progress_m': delta,
                'a_reasons': am.get('reasons'), 'b_reasons': bm.get('reasons')}
    ca, cb = bool(am.get('non_ego_pair_collision')), bool(bm.get('non_ego_pair_collision'))
    ha, hb = bool(am.get('hard_brake')), bool(bm.get('hard_brake'))
    ta = _below(am, 'min_rear_pair_ttc_s', p.ttc_s)
    tb = _below(bm, 'min_rear_pair_ttc_s', p.ttc_s)
    return {'eligible': True, 'reason': '', 'delta_progress_m': delta,
            'third_party_collision_a': ca, 'third_party_collision_b': cb,
            'signed_collision_effect': int(ca)-int(cb),
            'hard_brake_a': ha, 'hard_brake_b': hb,
            'braking_roles_a': am.get('hard_brake_roles', []),
            'braking_roles_b': bm.get('hard_brake_roles', []),
            'signed_brake_proxy_effect': int(ha)-int(hb),
            'signed_short_ttc_proxy_effect': int(ta)-int(tb)}
