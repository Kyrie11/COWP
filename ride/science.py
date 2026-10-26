"""Scientific protocol, named outcomes, and fixed preregistered comparisons.

This is an audit protocol: it never trains a planner or asserts human causation.
It deliberately separates simulated physical collision from comfort/risk proxies.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Protocol:
    progress_m: float = 3.0
    init_position_m: float = 1.0
    init_speed_mps: float = 1.0
    braking_mps2: float = -4.0
    ttc_s: float = 1.5
    # This pair is set BEFORE looking at outcomes, and the sign is preserved.
    contrast_actions: tuple[tuple[str, str], ...] = (
        ('early_slow', 'late_slow'), ('mild_early', 'mild_late'))
    families: tuple[str, ...] = ('tm', 'behavior')


def same_initial(a: dict, b: dict, p: Protocol) -> tuple[bool, str]:
    x, y = a.get('initial_states') or {}, b.get('initial_states') or {}
    for role in ('ego', 'follower_a', 'follower_b'):
        s, t = x.get(role), y.get(role)
        if s is None or t is None: return False, 'missing_initial_state:' + role
        try:
            if math.dist(s['xyz'], t['xyz']) > p.init_position_m: return False, 'initial_position_mismatch:' + role
            if abs(s['speed_mps']-t['speed_mps']) > p.init_speed_mps: return False, 'initial_speed_mismatch:' + role
        except (TypeError, KeyError, ValueError):
            return False, 'malformed_initial_state:' + role
    return True, ''


def assess_pair(a: dict, b: dict, p: Protocol) -> dict[str, Any]:
    """a and b are two interventions, in *preregistered* order.

    Output is a signed contrast; no selection of a 'bad' action after seeing data.
    Neither an invalid trial nor an unexecuted trial counts as a negative event.
    """
    out = {'eligible': False}
    am, bm = a.get('metrics') or {}, b.get('metrics') or {}
    if not am.get('valid') or not bm.get('valid'):
        return {**out, 'reason': 'invalid_trial', 'a_reasons': am.get('reasons'), 'b_reasons': bm.get('reasons')}
    ok, reason = same_initial(a, b, p)
    if not ok: return {**out, 'reason': reason}
    if am.get('ego_collision') or bm.get('ego_collision'):
        return {**out, 'reason': 'ego_collision_in_either_arm'}
    try:
        progress_a = float(am['ego_progress_m']); progress_b=float(bm['ego_progress_m'])
        if not all(map(math.isfinite, (progress_a,progress_b))):
            return {**out,'reason':'nonfinite_progress'}
        d = abs(progress_a-progress_b)
    except (KeyError,TypeError,ValueError):
        return {**out,'reason':'missing_progress'}
    if d>p.progress_m: return {**out,'reason':'progress_unmatched','delta_progress_m':d}
    # STRICT: same explicitly identified third-party pair in the baseline convoy.
    collision_a = bool(am.get('non_ego_pair_collision'))
    collision_b = bool(bm.get('non_ego_pair_collision'))
    hard_a = bool(am.get('hard_brake'))
    hard_b = bool(bm.get('hard_brake'))
    try: ttc_a=float(am.get('min_rear_pair_ttc_s') or 9999.)
    except (TypeError,ValueError):ttc_a=9999.
    try: ttc_b=float(bm.get('min_rear_pair_ttc_s') or 9999.)
    except (TypeError,ValueError):ttc_b=9999.
    return {**out,'eligible':True,'reason':'', 'delta_progress_m':d,
            'third_party_collision_a':collision_a,'third_party_collision_b':collision_b,
            'signed_collision_effect':int(collision_a)-int(collision_b),
            'hard_brake_a':hard_a,'hard_brake_b':hard_b,
            'braking_roles_a':am.get('hard_brake_roles',[]),
            'braking_roles_b':bm.get('hard_brake_roles',[]),
            'signed_brake_proxy_effect':int(hard_a)-int(hard_b),
            'signed_short_ttc_proxy_effect':int(ttc_a<p.ttc_s)-int(ttc_b<p.ttc_s)}
