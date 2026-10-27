"""Regression tests for path independence and decision-evidence validity.

These tests use SYNTHETIC fixtures or fake CARLA APIs. They are NOT physical
simulator validation, WOMD observation studies, or research results.
"""
from __future__ import annotations
import copy, json, math, subprocess, sys
from pathlib import Path

import pytest
from ride.synthetic_fixture import rows
from ride.carla.protocol import closing_ttc, summarise_trial
from ride.science import Protocol, assess_pair
from ride.ranking import risk_key
from ride.support import compare as compare_banks
from ride.decision import score
from ride.combine import combine
from ride.burden import compare as compare_burden


def test_runtime_module_entrypoints_have_no_optional_simulator_import_dependency():
    # A relocated package must import and show CLI help without the old tree.
    for module in ('ride.carla.run', 'ride.carla.analyze',
                   'ride.womd.native', 'ride.womd.observational',
                   'ride.waymax.paired_idm', 'ride.evidence', 'ride.ranking',
                   'ride.support', 'ride.burden', 'ride.freeze_ids'):
        done=subprocess.run([sys.executable,'-m',module,'--help'],
                            capture_output=True,text=True,timeout=20)
        assert done.returncode==0,(module,done.stderr)
        assert 'usage:' in done.stdout


def test_zero_gap_ttc_is_urgent_not_infinity():
    assert closing_ttc(0.,4.,12.,6.,vehicle_length=4.5)==0.
    assert closing_ttc(0.,15.,12.,6.,vehicle_length=4.5)==pytest.approx(10.5/6.)
    assert closing_ttc(0.,15.,5.,6.,vehicle_length=4.5)==math.inf


def test_profile_names_without_physical_ego_difference_not_eligible():
    a,b=rows()[1:3]
    b=copy.deepcopy(b)
    b['trace']=copy.deepcopy(a['trace'])
    measured=assess_pair(a,b,Protocol())
    assert not measured['eligible']
    assert measured['reason']=='ego_actions_not_physically_distinct'


def test_risk_key_preserves_zero_ttc():
    r={'metrics':{'valid':True,'min_rear_pair_ttc_s':0.,
                  'non_ego_pair_collision':False,'hard_brake':False}}
    assert risk_key(r)[2]==1
    r['metrics']['min_rear_pair_ttc_s']=None
    assert risk_key(r)[2]==0


def test_partial_run_cannot_enter_combined_scientific_panel(tmp_path):
    path=tmp_path/'a.jsonl'
    m={'type':'manifest','simulator':'CARLA','archetype':'convoy',
       'carla_server_version':'0.9.16','carla_client_version':'0.9.16'}
    r={'type':'trial','archetype':'convoy','seed':17,'anchor':0,'family':'tm','profile':'early_slow'}
    path.write_text('\n'.join(map(json.dumps,[m,r]))+'\n')
    with pytest.raises(ValueError,match='Incomplete'):
        combine([path])
    with path.open('a') as f:f.write(json.dumps({'type':'run_end','completed':True,'trial_count':1})+'\n')
    assert len(combine([path]))==2


def test_no_false_braking_regret_beyond_lexicographic_collision_minimum():
    trial=rows()
    selected=trial[1]['profile']
    # The collision-free chosen action hard-brakes, but a colliding action does
    # not. Braking regret must not treat the colliding action as a better option.
    row=next(x for x in trial if x.get('family')=='tm' and x.get('profile')==selected)
    row['metrics']['non_ego_pair_collision']=False
    row['metrics']['hard_brake']=True
    alt=next(x for x in trial if x.get('family')=='tm' and x.get('profile')=='late_slow')
    alt['metrics']['non_ego_pair_collision']=True
    alt['metrics']['hard_brake']=False
    # Other alternatives set to also collide, to isolate lexicographic ordering.
    for x in trial:
        if x.get('family')=='tm' and x is not row:
            x['metrics']['non_ego_pair_collision']=True
    choice={'method_id':'PROXY_TEST','archetype':'convoy','anchor':'0','seed':'7',
            'response_family':'tm','selected_action':selected,
            'choice_locked_before_rollout':'true'}
    result=score(trial,[choice])['results'][0]
    assert result['valid']
    assert result['third_party_collision_regret']==0
    assert result['hard_brake_proxy_regret']==0


def test_finite_bank_identical_status_but_different_trajectories_is_inconclusive():
    manifest={'type':'manifest','simulator':'CARLA','archetype':'support_gap',
              'carla_server_version':'0.9.16','carla_client_version':'0.9.16',
              'map':'Town04','dt':.05,'cruise_mps':9.}
    t={'t':.05,'xy_by_role':{'ego':[20.,0.],'follower_a':[0.,0.],'follower_b':[-18.,0.]}}
    r={'type':'trial','archetype':'support_gap','family':'joint_witness',
       'profile':'early_slow','anchor':0,'seed':17,'joint_targets':[6.,9.],
       'initial_states':{k:{'xyz':[i,0.,0.],'speed_mps':9.} for k,i in
                        (('ego',20),('follower_a',0),('follower_b',-18))},
       'metrics':{'valid':True,'ego_collision':False,'non_ego_collision':False,'hard_brake':False,
                  'ego_progress_m':5.,'follower_a_progress_m':5.,'follower_b_progress_m':5.},
       'trace':[t]}
    drift=copy.deepcopy(r);drift['trace'][0]['xy_by_role']['ego']=[21.,0.]
    result=compare_banks([manifest,r],[manifest,drift])
    assert result['summary']['valid_nested_bank_comparisons']==0
    assert result['summary']['inconclusive_shared_candidate_inconsistency']==1


def test_burden_incremental_audit_preserves_other_thresholds():
    common={'archetype':'convoy','anchor':'0','seed':'17',
            'pred_dynamic_valid':'true','pred_rule_valid':'true',
            'pred_ego_collision_prob':'.01',
            'pred_third_party_collision_prob':'.05'}
    a=dict(common,action='a',pred_progress_m='50.',pred_burden_cost='.4')
    b=dict(common,action='b',pred_progress_m='48.',pred_burden_cost='.1')
    out=compare_burden([a,b])
    assert out['summary']['decision_switches']==1
    assert out['summary']['task_matched_switch_proxies']==1
    assert out['results'][0]['burden_on']=='b'
    assert out['results'][0]['burden_off']=='a'


def test_carla_mechanism_not_executed_is_ineligible():
    r=rows();a=copy.deepcopy(r[1]);b=copy.deepcopy(r[2])
    a['metrics']['mechanism_executed']=False
    assert assess_pair(a,b,Protocol())['reason']=='scenario_maneuver_not_executed'


class _Loc:
    def __init__(self,x=0.,y=0.,z=0.):self.x=x;self.y=y;self.z=z
class _Rotation:
    def __init__(self,yaw=0.):self.yaw=yaw
class _Transform:
    def __init__(self,x,y):self.location=_Loc(x,y);self.rotation=_Rotation()
class _Waypoint:
    road_id=1
    is_junction=False
    lane_type=1
    def __init__(self,x=0.,y=0.,lane_id=1):
        self.transform=_Transform(x,y);self.lane_id=lane_id
    def next(self,d):return [_Waypoint(self.transform.location.x+d,self.transform.location.y,self.lane_id)]
    def previous(self,d):return [_Waypoint(self.transform.location.x-d,self.transform.location.y,self.lane_id)]
    def get_left_lane(self):return _Waypoint(self.transform.location.x,self.transform.location.y+3.5,2) if self.lane_id==1 else None
    def get_right_lane(self):return None
class _Carla:
    class LaneType:Driving=1
    Location=_Loc


def test_ego_target_waypoint_uses_carla_yaw_attribute():
    """CARLA 0.9.16 uses Rotation.yaw; the old Rotation.y path aborted trials."""
    from ride.carla_scenarios import ego_target_waypoint

    class Vehicle:
        def get_location(self):
            return _Loc(0., 0., 0.)

    class Map:
        def get_waypoint(self, *args, **kwargs):
            return _Waypoint(0., 0., 1)

    class World:
        def get_map(self):
            return Map()

    scene = {'follower_a': _Waypoint(0., 0., 1)}
    nxt = ego_target_waypoint(World(), Vehicle(), scene, 'convoy', 'mild_early', 0., _Carla)
    assert nxt is not None


def test_ordinary_adjacent_lane_is_not_rejected_by_cut_in_constructor():
    from ride.carla_scenarios import cut_in_candidate
    case=cut_in_candidate(_Waypoint(),_Carla)
    assert case is not None and case['ego'].lane_id==2
    assert case['follower_a'].lane_id==1


def test_support_gap_is_a_merge_geometry_not_a_convoy_alias():
    from ride.carla_scenarios import support_gap_candidate
    case = support_gap_candidate(_Waypoint(), _Carla)
    assert case is not None
    assert case['ego'].lane_id != case['follower_a'].lane_id
    assert case['follower_b'].lane_id == case['follower_a'].lane_id


def test_restore_requires_original_lane_identity_not_just_nearby_coordinates():
    from ride.carla_scenarios import restore
    wp=_Waypoint(5.,3.5,lane_id=1)
    class Map:
        def get_waypoint(self,*args,**kwargs):return wp
    class World:
        def get_map(self):return Map()
    anchor={'coords':{'ego':[5.,3.5,0.]},
            'waypoint_identities':{'ego':{'road_id':1,'lane_id':2,'yaw':0.}}}
    with pytest.raises(RuntimeError,match='road/lane'):
        restore(World(),anchor,_Carla)


def test_manifest_only_entrypoint_runs_without_an_old_source_tree(tmp_path,monkeypatch):
    """Exercise the exact modern Python entrypoint using a fake CARLA client.

    This verifies CLI/frozen-manifest wiring, not CARLA physics.
    """
    from ride.carla import run
    class Map:
        name='Town04'
    class World:
        def get_map(self):return Map()
    class Client:
        def __init__(self,*args):pass
        def set_timeout(self,*args):pass
        def get_server_version(self):return '0.9.16'
        def get_client_version(self):return '0.9.16'
        def get_world(self):return World()
        def load_world(self,*args):return World()
    StubCarla=type('StubCarla',(),{'Client':Client})
    monkeypatch.setattr(run,'_carla',lambda:StubCarla)
    monkeypatch.setattr('ride.carla_scenarios.build_anchors',
                        lambda world, archetype, count, seed, **kwargs:([
                            {'anchor':0,'key':[0.,0.,0.],'map':'Town04',
                             'waypoint_identities':{'ego':{'road_id':1,'lane_id':1,'yaw':0.}},
                             'coords':{'ego':[0.,0.,0.]},'archetype':archetype}
                        ], {'anchors_found':1}))
    freeze=tmp_path/'anchors.jsonl'
    monkeypatch.setattr(sys,'argv',['ride.carla.run','--archetype','convoy',
                                    '--anchors','1','--seed','17','--cruise-mps','9',
                                    '--manifest-only','--output',str(freeze)])
    run.main()
    obj=json.loads(freeze.read_text().splitlines()[0])
    assert obj['type']=='manifest'
    assert obj['archetype']=='convoy'
    assert len(obj['anchors'])==1
    assert obj['version_requested']=='0.9.16'
    # Frozen scene map/seed must survive relocation to a new output directory.
    target=tmp_path/'reused.jsonl'
    monkeypatch.setattr(sys,'argv',['ride.carla.run','--archetype','convoy',
                                    '--anchors','1','--seed','17','--cruise-mps','9',
                                    '--anchors-from',str(freeze),'--manifest-only',
                                    '--output',str(target)])
    run.main()
    assert json.loads(target.read_text().splitlines()[0])['anchors']==obj['anchors']


def test_direct_analysis_refuses_interrupted_carla_outcome(tmp_path):
    from ride.evidence import read_jsonl
    fp=tmp_path/'partial.jsonl'
    fp.write_text(json.dumps({'type':'manifest','simulator':'CARLA','archetype':'convoy'})+'\n')
    with pytest.raises(ValueError,match='Incomplete'):
        read_jsonl(fp)
    with fp.open('a') as f:f.write(json.dumps({'type':'run_end','completed':True,'trial_count':0})+'\n')
    assert len(read_jsonl(fp))==2


def test_structurally_distinct_ttc_braking_rule_is_audited_as_separate_model():
    import numpy as np
    from ride.womd.reactive import REGIMES, simulate
    assert 'ttc_gap_rule' in REGIMES
    # Three lane-aligned cars: one ego, a slower first follower, a second
    # follower closing up. Only a one-step synthetic model smoke test.
    st=np.array([[35.,0.,8.,0.,0.],[12.,0.,4.,0.,0.],[-1.,0.,12.,0.,0.]])
    out=simulate(st,np.full(3,4.5),np.full(3,1.9),np.ones(3,dtype=bool),
                 0,[10,11,12],0.,'ttc_gap_rule',steps=1)
    assert out['accelerations'].shape==(1,3)
    assert all(np.isfinite(v) for v in out['accelerations'].ravel())


def test_invalid_narrow_candidate_does_not_prove_finite_bank_miss():
    manifest={'type':'manifest','simulator':'CARLA','archetype':'support_gap',
              'carla_server_version':'0.9.16','carla_client_version':'0.9.16',
              'map':'Town04','cruise_mps':9.,'dt':.05}
    base={'type':'trial','archetype':'support_gap','anchor':0,'seed':17,
          'family':'joint_witness','profile':'early_slow','joint_targets':[6.,6.],
          'initial_states':{x:{'xyz':[z,0.,0.],'speed_mps':9.}
                            for x,z in [('ego',30),('follower_a',10),('follower_b',-10)]},
          'trace':[{'t':.05,'xy_by_role':{'ego':[30.,0.],
                      'follower_a':[10.,0.],'follower_b':[-10.,0.]}}],
          'metrics':{'valid':False,'ego_collision':False,'non_ego_collision':False,
                     'hard_brake':False,'ego_progress_m':1.,
                     'follower_a_progress_m':1.,'follower_b_progress_m':1.}}
    good=copy.deepcopy(base)
    good['joint_targets']=[9.,9.]
    good['metrics']['valid']=True
    result=compare_banks([manifest,base],[manifest,base,good])
    assert result['summary']['finite_bank_misses']==0
    assert result['details'][0]['comparison_valid'] is False


class _ContinuousRoadWaypoint:
    """A physically continuous lane whose OpenDRIVE road ID changes ahead."""
    lane_type=1
    lane_id=1
    is_junction=False
    def __init__(self,x):
        self.x=float(x)
        self.road_id=1 if x<155. else 2
        self.s=float(x)
        self.transform=_Transform(self.x,0.)
        self.transform.rotation.yaw=self.x*.15
    def next(self,d):
        # The spawn point at x=5 is unsuitable, but the sampled map at x=100
        # contains a valid 100m lane segment.
        x=self.x+float(d)
        return [_ContinuousRoadWaypoint(x)] if self.x>=60. and x<=205. else []
    def previous(self,d):
        x=self.x-float(d)
        return [_ContinuousRoadWaypoint(x)] if x>=60. else []


class _ContinuousRoadMap:
    name='Town04'
    def get_spawn_points(self):return [_Transform(5.,0.)]
    def get_waypoint(self,location,**kwargs):return _ContinuousRoadWaypoint(location.x)
    def generate_waypoints(self,spacing):return [_ContinuousRoadWaypoint(100.)]


class _ContinuousRoadWorld:
    def get_map(self):return _ContinuousRoadMap()


def test_convoy_scans_map_waypoints_beyond_spawn_points_and_follows_road_transitions(monkeypatch):
    """A long no-junction lane may not include a recommended spawn point.

    The previous implementation rejected this valid geometry because the
    goal crossed an OpenDRIVE road boundary and had >12deg heading variation.
    """
    from ride.carla_scenarios import build_anchors
    monkeypatch.setitem(sys.modules,'carla',_Carla)
    rows,diag=build_anchors(_ContinuousRoadWorld(),'convoy',1,17,
                            with_diagnostics=True)
    assert len(rows)==1
    assert rows[0]['coords']['ego'][0]==120.
    assert rows[0]['coords']['follower_b'][0]==82.
    assert rows[0]['coords']['end'][0]==200.
    assert rows[0]['waypoint_identities']['end']['road_id']==2
    assert diag['spawn_points_scanned']==1
    assert diag['map_waypoints_scanned']==1
    assert diag['rejections']['rear_lane_end_or_branch']==1


def test_convoy_rejects_hidden_junction_inside_100m_instead_of_landing_after_it():
    from ride.carla_scenarios import convoy_candidate
    class JunctionCorridor(_ContinuousRoadWaypoint):
        def next(self,d):
            x=self.x+d
            if 140.<=x<145.:
                new=JunctionCorridor(x)
                new.is_junction=True
                return [new]
            return [JunctionCorridor(x)] if self.x>=60. and x<=205. else []
        def previous(self,d):
            x=self.x-d
            return [JunctionCorridor(x)] if x>=60. else []
    reasons={}
    assert convoy_candidate(JunctionCorridor(100.),_Carla,reasons) is None
    assert reasons['forward_junction_or_non_driving']==1


def test_inspect_anchors_is_read_only_and_reports_rejections(tmp_path,monkeypatch,capsys):
    """The inspect command must not create or freeze scientific outcomes."""
    from ride.carla import run
    class Client:
        def __init__(self,*args):pass
        def set_timeout(self,*args):pass
        def get_server_version(self):return '0.9.16'
        def get_client_version(self):return '0.9.16'
        def get_world(self):return _ContinuousRoadWorld()
        def load_world(self,*args):return _ContinuousRoadWorld()
    monkeypatch.setattr(run,'_carla',lambda:type('Carla',(),{'Client':Client}))
    monkeypatch.setitem(sys.modules,'carla',_Carla)
    output=tmp_path/'should_not_exist.jsonl'
    monkeypatch.setattr(sys,'argv',['ride.carla.run','--archetype','convoy',
             '--inspect-anchors','--town','Town04','--output',str(output)])
    run.main()
    out=capsys.readouterr().out
    assert 'CARLA_MAP_ANCHOR_DIAGNOSTICS' in out
    assert '"eligible_anchors": 1' in out
    assert not output.exists()
