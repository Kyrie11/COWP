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
    if gap<=0:return 0.0
    return gap/closing if closing>0.1 else float('inf')


def summarise_trial(row):
    ts=row.get('trace',[])
    reasons=[]
    # Preserve the actionable failure reason when a trial aborts before the
    # first physics frame.  The previous early return collapsed CARLA API
    # errors into ``empty_trace`` and made the audit look like an ordinary
    # invalid scene instead of an engineering failure.
    if not ts:
        if row.get('simulation_exception'):
            reasons.append('simulation_exception')
        else:
            reasons.append('empty_trace')
        return {'valid':False,'reasons':reasons,
                'traffic_light_legality_verified':False,
                'mechanism_executed':False}
    if row.get('initial_overlap'):reasons.append('initial_overlap')
    if row.get('unexpected_actor'):reasons.append('uncontrolled_actor')
    if row.get('simulation_exception'):reasons.append('simulation_exception')
    if any(not x['road_valid_all'] for x in ts):reasons.append('offroad_or_lane_departure')
    # CARLA is_at_traffic_light() denotes a traffic-light trigger area, not
    # a verified crossing of the red stop line. This is a warning proxy only.
    red_trigger_moving=any(x.get('red_light_motion_roles') for x in ts)
    if row.get('initial_state_mismatch'):reasons.append('unpaired_initial_state')
    if row.get('sensor_overlap_disagreement_pairs'):
        reasons.append('collision_sensor_vs_obb_disagreement')
    road_end=float(ts[-1]['ego_progress_m'])
    ttc_values=[float(x['rear_pair_ttc_s']) for x in ts if x.get('rear_pair_ttc_s') is not None]
    gap_values=[float(x['rear_pair_gap_m']) for x in ts if x.get('rear_pair_gap_m') is not None]
    archetype=row.get('archetype','convoy')
    if archetype in ('cut_in','support_gap'):
        # A valid merge must start in the adjacent lane and later occupy the
        # target lane.  Merely spawning in the target lane or choosing a target
        # waypoint without a physical lane transition is not mechanism
        # execution.
        first_lane=ts[0].get('ego_lane_id')
        target_lane=ts[0].get('target_lane_id')
        switched=(first_lane is not None and target_lane is not None and
                  int(first_lane) != int(target_lane) and
                  any(bool(x.get('ego_in_target_lane')) for x in ts))
        mechanism=bool(switched and all(x.get('ego_in_target_lane',False) for x in ts[-3:]))
    else:
        mechanism=(any(x.get('ego_in_junction',False) for x in ts) and
                   any(x.get('follower_a_in_junction',False) for x in ts) and
                   any(x.get('ego_follower_a_distance_m',9999.)<18. for x in ts)) if archetype=='intersection' else True
    out={
      'valid':not reasons,'reasons':reasons,
      'traffic_light_legality_verified':False,
      'red_light_trigger_motion_proxy_observed':bool(red_trigger_moving),
      'ego_progress_m':road_end,
      'ego_path_length_m':float(ts[-1]['ego_path_length_m']),
      'progress_measure':ts[-1]['progress_measure'],
      'mechanism_executed':bool(mechanism),
      'follower_a_progress_m':float(ts[-1]['follower_a_progress_m']),
      'follower_b_progress_m':float(ts[-1]['follower_b_progress_m']),
      'ego_collision':bool(row.get('ego_collision',False)),
      'non_ego_collision':bool(row.get('non_ego_collision',False)),
      'non_ego_pair_collision':bool(row.get('non_ego_pair_collision',False)),
      'min_rear_pair_gap_m':min(gap_values) if gap_values else None,
      'min_rear_pair_ttc_s':min(ttc_values) if ttc_values else None,
      'ttc_applicable_steps':len(ttc_values),
      'min_follower_accel_mps2':min(min(x['follower_a_accel'],x['follower_b_accel']) for x in ts),
      'min_follower_a_accel_mps2':min(x['follower_a_accel'] for x in ts),
      'min_follower_b_accel_mps2':min(x['follower_b_accel'] for x in ts),
      'hard_brake_roles':[r for r,k in [('follower_a','follower_a_accel'),('follower_b','follower_b_accel')]
                          if min(x[k] for x in ts)<-4.0],
      'hard_brake':any(min(x['follower_a_accel'],x['follower_b_accel']) < -4.0 for x in ts),
      'ego_action_tracking_mse':sum((x['ego_target_speed_mps']-x['ego_speed_mps'])**2 for x in ts)/len(ts),
    }
    return out

