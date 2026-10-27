"""v4 regression gates: original discovery had invalid merge geometry and braking artifacts."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

from ride.carla.protocol import DT,summarise_trial
from ride.carla_scenarios import cut_in_candidate
from ride.forensics import summarize,_stream
from ride.support import derive_narrow_from_wide,compare
from ride.baseline_inventory import inventory

class Pt:
    def __init__(self,x,y=0): self.x=x;self.y=y
class Tr:
    def __init__(self,x,y): self.location=Pt(x,y);self.rotation=type('R',(),{'yaw':0})()
class Waypoint:
    road_id=1;is_junction=False;lane_type=1
    def __init__(self,x=0,y=0,lane=1):self.transform=Tr(x,y);self.lane_id=lane
    def next(self,d):return [Waypoint(self.transform.location.x+d,self.transform.location.y,self.lane_id)]
    def previous(self,d):return [Waypoint(self.transform.location.x-d,self.transform.location.y,self.lane_id)]
    def get_left_lane(self):return Waypoint(self.transform.location.x,self.transform.location.y+3.5,2)
    def get_right_lane(self):return None
class Carla:
    class LaneType:Driving=1


def _trace(brake_at=()):
    frames=[]
    for i in range(40):
        t=round((i+1)*DT,3)
        frames.append({'t':t,'road_valid_all':True,'ego_progress_m':float(t)*9,
            'ego_path_length_m':float(t)*9,'progress_measure':'goal_axis_projection',
            'follower_a_progress_m':t*9,'follower_b_progress_m':t*9,
            'follower_a_accel':-5. if i in brake_at else 0.,
            'follower_b_accel':0.,'ego_target_speed_mps':9.,'ego_speed_mps':9.,
            'ego_in_target_lane':i in range(24,36),
            'ego_lane_id':2 if i>=24 else 1, 'target_lane_id':2})
    return frames


def test_merge_witness_requires_lane_occupancy_not_last_three_frames():
    r={'trace':_trace(),'archetype':'cut_in'}
    m=summarise_trial(r)
    assert m['mechanism_executed'] is True
    assert m['target_lane_dwell_max_s'] >= 0.5
    for x in r['trace']:x['ego_in_target_lane']=False
    assert summarise_trial(r)['mechanism_executed'] is False


def test_brake_requires_post_startup_sustained_event():
    r={'trace':_trace(brake_at=(3,4,5,6)),'archetype':'convoy'}
    m=summarise_trial(r)
    assert m['startup_brake_proxy_observed'] is True
    assert m['hard_brake'] is False
    r['trace'][25]['follower_a_accel']=-5.
    m=summarise_trial(r)
    assert m['hard_brake'] is False
    assert m['hard_brake_instantaneous_post_startup_proxy'] is True
    for j in (25,26,27,28):r['trace'][j]['follower_a_accel']=-5.
    m=summarise_trial(r)
    assert m['hard_brake'] is True and m['follower_a_max_sustained_brake_s'] >= 0.2


def test_cut_in_front_ego_rear_correct_longitudinal_order():
    case=cut_in_candidate(Waypoint(),Carla)
    assert case is not None
    assert case['follower_a'].transform.location.x>case['ego'].transform.location.x
    assert case['ego'].transform.location.x>case['follower_b'].transform.location.x
    assert case['ego'].lane_id!=case['follower_a'].lane_id


def test_partial_forensic_fails_closed(tmp_path):
    file=tmp_path/'trial.jsonl'
    file.write_text(json.dumps({'type':'manifest','simulator':'CARLA','duration_s':8.,'dt':.05})+'\n'+
                    json.dumps({'type':'trial','family':'tm','archetype':'support_gap',
                        'profile':'early_slow','anchor':1,'seed':17,'trace':[],
                        'metrics':{'valid':False,'reasons':['simulation_exception']}})+'\n')
    r=summarize(_stream(str(file)))
    assert r['run_completed_and_count_verified'] is False
    assert r['counts']['invalid_reason_simulation_exception']==1
    assert r['scientific_claim_allowed'] is False


def test_subset_uses_identical_candidate_and_recorded_provenance():
    m={'type':'manifest','simulator':'CARLA','joint_target_grid':[6.,9.,12.],
       'joint_limit':4,'carla_server_version':'0.9.16','carla_client_version':'0.9.16',
       'map':'Town04','archetype':'support_gap','dt':.05,'cruise_mps':9.,
       'duration_s':.05,'seed':17,'repetitions':1,'warmup_ticks':20,
       'anchor_protocol_version':'v4'}
    def trial(target):
        return {'type':'trial','family':'joint_witness','archetype':'support_gap',
                'profile':'early_slow','anchor':0,'seed':17,'joint_targets':list(target),
                'initial_states':{k:{'xyz':[0.,0.,0.],'speed_mps':9.}
                                  for k in ('ego','follower_a','follower_b')},
                'trace':[{'t':.05,'xy_by_role':{'ego':[0.,0.],'follower_a':[9.,0.],
                                          'follower_b':[-9.,0.]}}],
                'metrics':{'valid':True,'mechanism_executed':True,
                           'ego_collision':False,'non_ego_collision':target!=(12.,12.),
                           'hard_brake':False,'ego_progress_m':1.,
                           'follower_a_progress_m':1.,'follower_b_progress_m':1.}}
    wide=[m,trial((6.,6.)),trial((6.,12.)),trial((12.,6.)),trial((12.,12.)),
          {'type':'run_end','completed':True,'trial_count':4}]
    narrow=derive_narrow_from_wide(wide,[(6.,6.)])
    assert len([r for r in narrow if r['type']=='trial'])==1
    assert narrow[0]['subset_provenance'].startswith('EXACT_SUBSET')
    compared=compare(narrow,wide)
    assert compared['summary']['valid_nested_bank_comparisons']==1
    assert compared['summary']['finite_bank_misses']==1


def test_read_only_checkpoint_inventory_marks_git_lfs_pointer(tmp_path):
    root=tmp_path/'baseline'; d=root/'GameFormer';d.mkdir(parents=True)
    (d/'weights.pth').write_text('version https://git-lfs.github.com/spec/v1\n')
    inventory_result=inventory(root)
    weights=inventory_result['methods']['GameFormer']['checkpoints']
    assert weights[0]['likely_git_lfs_pointer'] is True
    assert inventory_result['methods']['R3R_Julia']['exists'] is False

def test_intersection_red_trigger_and_goal_distance_proxy_do_not_certify_legality():
    from ride.science import Protocol,validate_pair
    row={'archetype':'intersection','metrics':{'valid':True,
         'mechanism_executed':True,'traffic_light_legality_verified':False,
         'progress_measure':'goal_distance_reduction_proxy'}}
    ok,why,_=validate_pair(row,deepcopy(row),Protocol())
    assert ok is False and why=='intersection_legality_unverified'

def test_collision_before_action_is_invalid_and_not_part_of_ego_effect():
    row={'archetype':'convoy','trace':_trace(),
         'pre_action_collision_events':[{'frame':11,'role_a':'follower_a','role_b':'follower_b'}],
         'collision_events':[],'ego_collision':False,'non_ego_pair_collision':False}
    m=summarise_trial(row)
    assert m['valid'] is False
    assert 'collision_before_ego_intervention' in m['reasons']
    assert m['non_ego_pair_collision'] is False
