"""Science-gate regression tests; no CARLA runtime is needed."""
from types import SimpleNamespace as NS
from copy import deepcopy
import json
from pathlib import Path
import pytest
from ride.v11_chain_evidence import (
    sustained_first, audit_intervention, evaluate, count_spatial_corridors,
)
from ride.carla.roles import per_role_response_modes
from ride.carla_scenarios import _anchor_roundtrip_valid, ANCHOR_PROTOCOL_VERSION


def test_sustained_response_screening_needs_four_frames():
    times=[.05*(i+1) for i in range(100)]
    signal=[0.]*100
    for i in range(40,43):signal[i]=.8  # only 0.15s
    assert sustained_first(times,signal,.5) is None
    signal[43]=.8
    assert sustained_first(times,signal,.5)==pytest.approx(2.05)
    signal[43]=float('nan')
    assert sustained_first(times,signal,.5) is None


def frames(kind):
    out=[]
    for i in range(160):
        t=round((i+1)*.05,3)
        b=8.0-(.8 if kind=='treatment' and t>=2.0 else 0.)
        c=8.0-(.5 if kind=='treatment' and t>=2.5 else 0.)
        out.append({'t':t,'ego_target_waypoint_lane_id':6 if kind=='treatment' and t>=1.5 else 5,
            'target_lane_id':6,'ego_lane_id':5,
            'xy_by_role':{'ego':[t*8.,3.5], 'follower_b':[t*8.-15,0.],
                          'follower_c':[t*8.-40,0.]},
            'follower_b_speed_mps':b,'follower_c_speed_mps':c,
            'rear_pair_ttc_s':7.,'rear_pair_gap_m':19.5})
    return out


def row(kind):
    return {'archetype':'chain_cut_in','profile':'early_slow' if kind=='treatment' else 'cruise',
        'trace':frames(kind),'metrics':{'valid':True,'non_ego_pair_collision':False,'hard_brake':False},
        'initial_states':{r:{'xyz':[float(j*10),0.,0.],'yaw_deg':0.,'speed_mps':8.}
                          for j,r in enumerate(('ego','follower_a','follower_b','follower_c'))}}


def test_ordered_response_is_screening_not_proof():
    result=audit_intervention(row('treatment'),row('cruise'))
    assert result['eligible']
    assert result['first_B_extra_slowdown_s']==pytest.approx(2.)
    assert result['first_C_extra_slowdown_s']==pytest.approx(2.5)
    assert result['ordered_B_then_C_slowdown_proxy']
    assert not result['new_hazard_vs_cruise_not_task_matched']
    assert 'NOT' not in result['mechanism_stage']


def test_precommand_mismatch_rejects_simulation_drift():
    treatment=row('treatment');baseline=row('cruise')
    treatment['trace'][24]['xy_by_role']['follower_c'][0]+=1.5
    result=audit_intervention(treatment,baseline)
    assert not result['eligible']
    assert result['reason']=='pre_intervention_drift'


def test_missing_c_does_not_get_silent_zero():
    treatment=row('treatment');baseline=row('cruise')
    del treatment['trace'][80]['follower_c_speed_mps']
    result=audit_intervention(treatment,baseline)
    assert not result['eligible'] and result['reason']=='malformed_trace_KeyError'


def test_exploratory_mixed_modes_are_explicit():
    assert per_role_response_modes('chain_cut_in','mixed_b_tm_c_behavior') == {
        'follower_a':'tm','follower_b':'tm','follower_c':'behavior'}
    assert per_role_response_modes('chain_cut_in','mixed_b_behavior_c_tm') == {
        'follower_a':'behavior','follower_b':'behavior','follower_c':'tm'}
    with pytest.raises(ValueError):per_role_response_modes('cut_in','mixed_b_behavior_c_tm')


def test_freeze_boundary_roundtrip_rejects_ambiguous_c_identity():
    class Carla:
        class LaneType:Driving=1
        @staticmethod
        def Location(x,y,z):return NS(x=x,y=y,z=z)
    class Map:
        def get_waypoint(self,loc,**kwargs):
            # 3 decimals may move a value on the boundary to another road.
            road=41 if loc.x<1.0005 else 6
            return NS(road_id=road,lane_id=-3,transform=NS(
                location=loc,rotation=NS(yaw=90.)))
    identities={'follower_c':{'road_id':41,'lane_id':-3,'yaw':90.}}
    assert _anchor_roundtrip_valid(Map(),Carla,{'follower_c':[1.00049,0.,0.]},identities)[0]
    assert not _anchor_roundtrip_valid(Map(),Carla,{'follower_c':[1.001,0.,0.]},identities)[0]
    assert ANCHOR_PROTOCOL_VERSION=='ride_lane_corridor_v11_20261028'


def test_spatial_cluster_does_not_count_neighbor_anchors_as_independent():
    keys=[('Town04',0.,0.),('Town04',55.,0.),('Town04',130.,0.),('Town04',500.,0.),('Town05',0.,0.)]
    assert count_spatial_corridors(keys)==3 # 0--55--130 transitive; 500; other town


def test_read_only_reaudits_real_v10_if_available():
    paths=[Path('/mnt/data/ride_v10_audit/results')/q for q in (
        'rq12_v10_threecar_m0/carla/cut_in.jsonl',
        'rq12_v10_chain_dev_m0/carla/chain_cut_in.jsonl',
        'rq12_v10_chain_confirm/seed_23/carla/chain_cut_in.jsonl')]
    if not all(p.exists() for p in paths):pytest.skip('user original uploaded files not in this runtime')
    r=evaluate(paths)
    assert r['counts']['trials']==100
    assert r['counts']['valid_trials']==71
    assert r['counts']['cross_family_pairs']==9
    assert r['counts']['strict_primary_rank_reversals']==0
    assert r['counts']['ordered_B_then_C_response_proxies']==0
    assert r['counts']['spatial_corridor_clusters_90m_all']==7
