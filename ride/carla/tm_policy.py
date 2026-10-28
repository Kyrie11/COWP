"""CARLA TrafficManager speed policy with explicit physical units.

`vehicle_percentage_speed_difference(veh, 0)` runs at the map speed LIMIT.
A requested 9m/s is 32.4km/h and is ILLEGAL in a 30km/h Town04 corridor.
For prospective v9 use cruise=8.0m/s with 0 rear speed delta there.
Never silently clamp TM while leaving BehaviorAgent and ego at the old speed.
It is NOT comparable with a 9 m/s PID-controlled responder. CARLA vehicle
speed limits are returned in km/h. The percentage represents fraction below
that limit; log the effective speed target and verify observed rollout speeds.
"""
from __future__ import annotations
import math


def speed_limit_percentage(limit_kmh: float, nominal_mps: float) -> float:
    limit_kmh = float(limit_kmh)
    nominal_mps = float(nominal_mps)
    if not math.isfinite(limit_kmh) or limit_kmh < 5.:
        raise ValueError('TM_SPEED_LIMIT_UNAVAILABLE: %r km/h' % limit_kmh)
    if not math.isfinite(nominal_mps) or nominal_mps <= 0:
        raise ValueError('TM_NOMINAL_SPEED_INVALID: %r m/s' % nominal_mps)
    if nominal_mps * 3.6 > limit_kmh + 0.5:
        raise ValueError('TM_DESIRED_SPEED_EXCEEDS_LIMIT: %.2f m/s > %.2f km/h' % (nominal_mps, limit_kmh))
    return max(0., min(95., 100. * (1. - nominal_mps * 3.6 / limit_kmh)))


def configure_tm_speed(tm, vehicle, nominal_mps):
    limit_kmh=float(vehicle.get_speed_limit())
    pct=speed_limit_percentage(limit_kmh, nominal_mps)
    tm.vehicle_percentage_speed_difference(vehicle,pct)
    return pct,limit_kmh


def nominal_target_speeds(cruise,delta,archetype):
    return {'follower_a':float(cruise),
            'follower_b':float(cruise)+(float(delta) if archetype in ('cut_in','support_gap') else 0.)}
