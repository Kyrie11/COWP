"""Pure, testable scientific protocol for CARLA intervention studies.

Pre-register metrics before experiments. An effect is a SIMULATED intervention
result and never a human counterfactual. TM and BehaviorAgent are distinct
implementations but both operate in CARLA physics/map.
"""
from __future__ import annotations
import math

DT=0.05
DURATION_S=8.0
# Record all acceleration, but exclude the common start-up control transient
# from the decision-dependent braking endpoint. Must be frozen before trials.
BRAKE_OBSERVATION_START_S=1.0
PROFILES=('cruise','early_slow','late_slow','mild_early','mild_late',
          'merge_early','merge_late')


def target_speed_mps(profile: str, t: float, base=11.0):
    """Paired commands share speed-time integral but NOT guaranteed actual travel."""
    if profile in ('cruise','merge_early','merge_late'):return base
    if profile not in PROFILES:raise ValueError(profile)
    if t<1.5 or t>=5.5:return base
    # V4 yielded 4.6-13.3m paired-progress gaps at 8s. The new v5
    # prespecified smaller amplitudes seek informative, task-matched pairs;
    # empirical matching is STILL checked, never assumed from target speeds.
    amplitude=1.8 if profile.startswith('mild') else 3.0
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
    if row.get('pre_action_collision_events'):reasons.append('collision_before_ego_intervention')
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
        # Require at least 0.5s continuous occupancy, not merely the last
        # frames. A valid merge may subsequently cross an OpenDRIVE road-id
        # boundary, so checking the final three frames creates false negatives.
        max_dwell = dwell = 0
        for x in ts:
            dwell = dwell + 1 if x.get('ego_in_target_lane') else 0
            max_dwell = max(max_dwell, dwell)
        mechanism=bool(switched and max_dwell >= round(0.5/DT))
    else:
        mechanism=(any(x.get('ego_in_junction',False) for x in ts) and
                   any(x.get('follower_a_in_junction',False) for x in ts) and
                   any(x.get('ego_follower_a_distance_m',9999.)<18. for x in ts)) if archetype=='intersection' else True
    # Direct set_target_velocity + controller handoff creates a non-physical
    # ~-48 m/s² spike around t=0.2 in the uploaded panel, independent of ego
    # action. Keep the raw values and an explicit startup diagnostic; compare
    # post-startup outcomes only. A true collision is NEVER censored here.
    tbrake=[x for x in ts if float(x['t']) >= BRAKE_OBSERVATION_START_S]
    if not tbrake:
        reasons.append('insufficient_post_startup_observation')
        tbrake=ts
    full_min=min(min(x['follower_a_accel'],x['follower_b_accel']) for x in ts)
    post_min=min(min(x['follower_a_accel'],x['follower_b_accel']) for x in tbrake)
    # One noisy acceleration sample is not a real emergency-braking episode.
    # Require >=0.20 seconds continuously below threshold; retain the raw
    # one-sample flag for sensitivity analysis. This is a new v4 metric.
    def longest_run(role):
        longest=run=0
        for frame in tbrake:
            run = run + 1 if frame[role] < -4.0 else 0
            longest=max(longest,run)
        return longest
    duration_a=longest_run('follower_a_accel') * DT
    duration_b=longest_run('follower_b_accel') * DT
    hard_roles=[name for name,duration in (('follower_a',duration_a),
                                           ('follower_b',duration_b)) if duration >= 0.20-1e-8]
    out={
      'valid':not reasons,'reasons':reasons,
      'brake_observation_start_s':BRAKE_OBSERVATION_START_S,
      'startup_brake_proxy_observed':any(
         min(x['follower_a_accel'],x['follower_b_accel']) < -4.0
         for x in ts if float(x['t']) < BRAKE_OBSERVATION_START_S),
      'min_follower_accel_full_mps2':full_min,
      'traffic_light_legality_verified':False,
      'red_light_trigger_motion_proxy_observed':bool(red_trigger_moving),
      'ego_progress_m':road_end,
      'ego_path_length_m':float(ts[-1]['ego_path_length_m']),
      'progress_measure':ts[-1]['progress_measure'],
      'mechanism_executed':bool(mechanism),
      'target_lane_dwell_max_s':(max_dwell*DT if archetype in ('cut_in','support_gap') else None),
      'follower_a_progress_m':float(ts[-1]['follower_a_progress_m']),
      'follower_b_progress_m':float(ts[-1]['follower_b_progress_m']),
      'ego_collision':bool(row.get('ego_collision',False)),
      'non_ego_collision':bool(row.get('non_ego_collision',False)),
      'non_ego_pair_collision':bool(row.get('non_ego_pair_collision',False)),
      'min_rear_pair_gap_m':min(gap_values) if gap_values else None,
      'min_rear_pair_ttc_s':min(ttc_values) if ttc_values else None,
      'ttc_applicable_steps':len(ttc_values),
      'min_follower_accel_mps2':post_min,
      'min_follower_a_accel_mps2':min(x['follower_a_accel'] for x in tbrake),
      'min_follower_b_accel_mps2':min(x['follower_b_accel'] for x in tbrake),
      'hard_brake_roles':hard_roles,
      'hard_brake':bool(hard_roles),
      'hard_brake_instantaneous_post_startup_proxy':post_min < -4.0,
      'follower_a_max_sustained_brake_s':duration_a,
      'follower_b_max_sustained_brake_s':duration_b,
      'ego_action_tracking_mse':sum((x['ego_target_speed_mps']-x['ego_speed_mps'])**2 for x in ts)/len(ts),
    }
    return out

