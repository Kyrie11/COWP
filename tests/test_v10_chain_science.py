"""Tests exercise the new scene topology and causal gate without a CARLA server."""
from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from ride.carla.roles import all_roles,primary_pair,nominal_speed
from ride.carla.behavior_policy import cap_behavior_agent
from ride.carla_scenarios import chain_cut_in_candidate
from ride.carla.pair_geometry import longitudinal_pair_eligible
from ride.carla.protocol import summarise_trial
from ride.science import Protocol,same_initial
from ride.protocol_lock import check,FIELDS,TEXT_FIELDS
from ride.v10_mechanism_report import _risk_key,response_delta_against_cruise


class Pos:
    def __init__(self,x=0.,y=0.,z=0.):self.x=x;self.y=y;self.z=z
class WP:
    lane_type=1;road_id=45;is_junction=False;lane_width=3.5
    def __init__(self,x=0.,y=0.,lane=6):
        self.transform=NS(location=Pos(x,y),rotation=NS(yaw=0.))
        self.lane_id=lane
        self.road_id=6 if x>5 else 45
    def next(self,m):return [WP(self.transform.location.x+m,self.transform.location.y,self.lane_id)]
    def previous(self,m):return [WP(self.transform.location.x-m,self.transform.location.y,self.lane_id)]
    def get_left_lane(self):return WP(self.transform.location.x,self.transform.location.y+3.5,5)
    def get_right_lane(self):return None
class Carla:
    class LaneType:Driving=1


def test_chain_candidate_exposes_physical_b_c_not_a_b():
    c=chain_cut_in_candidate(WP(),Carla,20,35,25)
    assert c is not None
    assert c['follower_a'].transform.location.x==0.
    assert c['ego'].transform.location.x==-20.
    assert c['follower_b'].transform.location.x==-35.
    assert c['follower_c'].transform.location.x==-60.
    assert c['_target_corridor_xy'][0]==[-60.,0.]
    assert len(c['_target_corridor_xy'])==18+1+7+5
    assert primary_pair('chain_cut_in')==('follower_b','follower_c')
    assert all_roles('chain_cut_in')==('ego','follower_a','follower_b','follower_c')


def test_chain_unsupported_short_road_never_fakes_coordinates():
    class End(WP):
        def previous(self,m):return []
    class Broken(WP):
        def previous(self,m):return [] if self.transform.location.x < -30 else super().previous(m)
    # The simple fake WP returns WP(), so explicitly unit-test the spacing validation.
    with pytest.raises(ValueError):chain_cut_in_candidate(WP(),Carla,20,35,19)
    with pytest.raises(ValueError):chain_cut_in_candidate(WP(),Carla,20,35,22)


def test_cross_opendrive_road_id_still_same_frozen_lane():
    front,rear=WP(8,0),WP(3,0)
    assert front.road_id != rear.road_id
    corridor=[[x,0.] for x in range(-40,11,5)]
    assert longitudinal_pair_eligible(front,rear,front.transform.location,
        rear.transform.location,target_corridor_xy=corridor)
    shifted=Pos(3,3.5)
    assert not longitudinal_pair_eligible(front,rear,front.transform.location,shifted,
        target_corridor_xy=corridor)


def test_nominal_velocity_roles_and_cap():
    assert nominal_speed('follower_b',archetype='chain_cut_in',cruise_mps=8,
                         rear_speed_delta_mps=1)==9
    assert nominal_speed('follower_c',archetype='chain_cut_in',cruise_mps=8,
                         chain_c_speed_delta_mps=.5)==8.5
    obj=NS(_behavior=NS(max_speed=80.,speed_lim_dist=10.))
    cfg=cap_behavior_agent(obj,8)
    assert cfg['capped_max_kmh']==pytest.approx(28.8)
    assert obj._behavior.max_speed==pytest.approx(28.8)
    assert obj._behavior.speed_lim_dist==0.0
    with pytest.raises(RuntimeError,match='MISMATCH'):cap_behavior_agent(NS(),8)


def chain_frames():
    out=[]
    for i in range(160):
        t=round((i+1)*0.05,3)
        out.append({'t':t,'road_valid_all':True,'ego_progress_m':t*8,
                    'ego_path_length_m':t*8,'progress_measure':'goal_axis_projection',
                    'follower_a_progress_m':t*8,'follower_b_progress_m':t*8,
                    'ego_lane_id':5,'target_lane_id':6,
                    'ego_in_target_lane':3<t<7,
                    'follower_a_accel':0.,'follower_b_accel':0.,
                    'follower_c_accel':-5. if 4.1<t<4.5 else 0.,
                    'rear_pair_ttc_s':1.0 if t>4 else 3.,
                    'rear_pair_gap_m':8.,'red_light_motion_roles':[],
                    'ego_target_speed_mps':8.,'ego_speed_mps':8.,
                    'follower_b_speed_mps':8.,'follower_c_speed_mps':8.,
                    'ego_target_waypoint_lane_id':6 if t>=1.5 else 5,
                    'xy_by_role':{'ego':[t*8,3.5], 'follower_b':[t*8-10,0],
                                  'follower_c':[t*8-35,0]}})
    return out


def test_chain_primary_hardbrake_is_b_c_not_a_b():
    row={'archetype':'chain_cut_in','trace':chain_frames(),'non_ego_pair_collision':False}
    m=summarise_trial(row)
    assert m['valid'] and m['mechanism_executed']
    assert m['hard_brake_roles']==['follower_c']
    assert m['primary_third_party_pair_roles']==['follower_b','follower_c']
    assert m['min_rear_pair_ttc_s']==1.0
    assert m['follower_c_max_sustained_brake_s']>=.2


def test_initial_matching_must_include_c():
    src={'archetype':'chain_cut_in','initial_states':{role:{'xyz':[i*20.,0.,0.],'speed_mps':8.}
          for i,role in enumerate(all_roles('chain_cut_in'))}}
    other=deepcopy(src);other['initial_states']['follower_c']['xyz'][0]+=1.1
    assert same_initial(src,other,Protocol())==(False,'initial_position_mismatch:follower_c')


def test_protocol_freezes_behavior_cap_and_c_spacing():
    d={key:3.5 for key in FIELDS}
    d.update({key:'nominal_cap' for key in TEXT_FIELDS})
    frozen={'type':'manifest','anchor_protocol_version':'ride_lane_corridor_v10_20261028',**d}
    assert check(frozen,d)=={}
    altered={**d,'chain_bc_gap_m':25.,'behavior_speed_protocol':'native'}
    diff=check(frozen,altered)
    assert 'chain_bc_gap_m' in diff and 'behavior_speed_protocol' in diff


def test_cruise_control_must_be_initially_matched():
    source={'archetype':'chain_cut_in','trace':chain_frames(),
         'metrics':{'valid':True},
         'initial_states':{r:{'xyz':[i*25,0,0], 'speed_mps':8.}
                           for i,r in enumerate(all_roles('chain_cut_in'))}}
    cruise=deepcopy(source)
    cruise['initial_states']['follower_c']['xyz'][0]+=1.0
    assert response_delta_against_cruise(source,cruise)['eligible'] is False


def test_pre_action_drift_cannot_be_claimed_as_ego_induced_effect():
    """Even matched initial positions cannot excuse diverging pre-action histories."""
    init={r:{'xyz':[i*25.,0.,0.],'speed_mps':8.}
          for i,r in enumerate(all_roles('chain_cut_in'))}
    src={'archetype':'chain_cut_in','trace':chain_frames(),
         'metrics':{'valid':True},'initial_states':init}
    base=deepcopy(src)
    # Declare a commanded merge only after t=1.5s.
    for f in src['trace']:
        f['ego_target_waypoint_lane_id']=6 if f['t']>=1.5 else 5
    for f in base['trace']:
        f['ego_target_waypoint_lane_id']=5
    # Introduce mismatch at t=1.1s, before the merge command at t=1.5s.
    for f in base['trace']:
        if f['t']>=1.1:
            f['xy_by_role']['follower_c'][0] += 2.
    result=response_delta_against_cruise(src,base)
    assert result['eligible'] is False
    assert result['reason']=='pre_action_trajectory_mismatch'
    assert result['pre_action_max_responder_xy_m']>1.


def test_same_initial_checks_heading_when_recorded():
    p=Protocol()
    src={'archetype':'chain_cut_in','initial_states':{r:{'xyz':[i*25,0,0],
       'speed_mps':8.,'yaw_deg':0.} for i,r in enumerate(all_roles('chain_cut_in'))}}
    dst=deepcopy(src)
    dst['initial_states']['follower_c']['yaw_deg']=10.0
    assert same_initial(src,dst,p)==(False,'initial_heading_mismatch:follower_c')
