"""Pure, testable scientific protocol for CARLA intervention studies.

Pre-register metrics before experiments. An effect is a SIMULATED intervention
result and never a human counterfactual. TM and BehaviorAgent are distinct
implementations but both operate in CARLA physics/map.
"""
from __future__ import annotations
import math

DT=0.05
DURATION_S=8.0
PROFILES=('cruise','early_slow','late_slow','mild_early','mild_late')


def target_speed_mps(profile: str, t: float, base=11.0):
    """Paired commands share speed-time integral but NOT guaranteed actual travel."""
    if profile=='cruise':return base
    if profile not in PROFILES:raise ValueError(profile)
    if t<1.5 or t>=5.5:return base
    amplitude=3.0 if profile.startswith('mild') else 5.5
    low,high=max(0.5,base-amplitude),base+amplitude
    early=t<3.5
    if profile.endswith('early') or profile=='early_slow':return low if early else high
    return high if early else low


def closing_ttc(rear_s,front_s,rear_v,front_v,vehicle_length=4.5):
    # Front/rear center projections converted into approximate bumper gap.
    gap=front_s-rear_s-vehicle_length
    closing=rear_v-front_v
    return max(0.,gap/closing) if closing>0.1 and gap>0 else float('inf')


def summarise_trial(row):
    ts=row.get('trace',[])
    if not ts:return {'valid':False,'reasons':['empty_trace']}
    reasons=[]
    if row.get('initial_overlap'):reasons.append('initial_overlap')
    if row.get('unexpected_actor'):reasons.append('uncontrolled_actor')
    if row.get('simulation_exception'):reasons.append('simulation_exception')
    if any(not x['road_valid_all'] for x in ts):reasons.append('offroad_or_lane_departure')
    if row.get('initial_state_mismatch'):reasons.append('unpaired_initial_state')
    road_end=float(ts[-1]['ego_progress_m'])
    out={
      'valid':not reasons,'reasons':reasons,
      'ego_progress_m':road_end,
      'follower_a_progress_m':float(ts[-1]['follower_a_progress_m']),
      'follower_b_progress_m':float(ts[-1]['follower_b_progress_m']),
      'ego_collision':bool(row.get('ego_collision',False)),
      'non_ego_collision':bool(row.get('non_ego_collision',False)),
      'non_ego_pair_collision':bool(row.get('non_ego_pair_collision',False)),
      'min_rear_pair_gap_m':min(x['rear_pair_gap_m'] for x in ts),
      'min_rear_pair_ttc_s':min(x['rear_pair_ttc_s'] for x in ts),
      'min_follower_accel_mps2':min(min(x['follower_a_accel'],x['follower_b_accel']) for x in ts),
      'hard_brake':any(min(x['follower_a_accel'],x['follower_b_accel']) < -4.0 for x in ts),
      'ego_action_tracking_mse':sum((x['ego_target_speed_mps']-x['ego_speed_mps'])**2 for x in ts)/len(ts),
    }
    return out


def check_contrast(a,b, *, progress_tolerance=3., ttc_threshold=1.5):
    """a=putatively harmful; b=alternative; no statistical interpretation."""
    if not a.get('valid') or not b.get('valid'):return {'eligible':False,'why':'invalid_trial'}
    delta=abs(a['ego_progress_m']-b['ego_progress_m'])
    if delta>progress_tolerance:return {'eligible':False,'why':'task_progress_mismatch','delta_progress_m':delta}
    # Predetermine two endpoints. Collision requires a consistent named pair;
    # hard brake/short TTC is exploratory, NEVER re-label as a collision.
    risk=(a['non_ego_pair_collision'] and not b['non_ego_pair_collision'])
    burden=(a['min_follower_accel_mps2']<-4 and b['min_follower_accel_mps2']>=-4)
    near=(a['min_rear_pair_ttc_s']<ttc_threshold and b['min_rear_pair_ttc_s']>=ttc_threshold)
    return {'eligible':True,'delta_progress_m':delta,
      'strict_third_party_collision_preventable':bool(risk and not a['ego_collision'] and not b['ego_collision']),
      'hard_brake_preventable_proxy':bool(burden),'short_ttc_preventable_proxy':bool(near)}
