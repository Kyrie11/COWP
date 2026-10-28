"""Explicit and auditable longitudinal speed ceiling for CARLA BehaviorAgent.

BehaviorAgent normally selects a speed from the road limit and its private
behavior profile, not from the research nominal speed. We adjust only its
maximum cruise speed and the speed-limit margin, so both responder
families target the same nominal speed; car-following, obstacle response and
braking logic remain the original BehaviorAgent. No claim of human behavioral calibration is implied.
"""
from __future__ import annotations
import math


def cap_behavior_agent(agent, nominal_mps):
    """Set BehaviorAgent 0.9.16 max_speed (km/h); fail closed if API differs."""
    speed = float(nominal_mps)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError('BEHAVIOR_NOMINAL_SPEED_INVALID')
    params = getattr(agent, '_behavior', None)
    if params is None or not hasattr(params, 'max_speed') or not hasattr(params, 'speed_lim_dist'):
        raise RuntimeError('BEHAVIOR_SPEED_CAP_API_MISMATCH: verify CARLA 0.9.16 behavior_agent.py')
    original = float(params.max_speed)
    if not math.isfinite(original) or original <= 0:
        raise RuntimeError('BEHAVIOR_MAX_SPEED_INVALID')
    original_margin = float(params.speed_lim_dist)
    cap = min(original, 3.6 * speed)
    params.max_speed = cap
    params.speed_lim_dist = 0.0
    return {'original_max_kmh': original, 'capped_max_kmh': cap,
            'original_speed_limit_margin_kmh': original_margin,
            'matched_speed_limit_margin_kmh': 0.0,
            'nominal_mps': speed, 'note': 'BehaviorAgent may still slow down according to its own car-following rules'}
