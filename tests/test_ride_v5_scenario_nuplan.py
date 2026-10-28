"""Offline protocol tests: no CARLA server / nuPlan DB / GPU required."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from ride.carla.lane_geometry import distance_to_polyline,on_centerline,map_lane_center_valid
from ride.carla_scenarios import cut_in_candidate
from ride.carla.protocol import target_speed_mps
from ride.reactsim_setup import inspect,list_dbs


class Pos:
    def __init__(self,x=0.,y=0.,z=0.):self.x=x;self.y=y;self.z=z
class WP:
    lane_id=6;lane_type=1;road_id=45;is_junction=False;lane_width=3.5
    def __init__(self,x=0.,y=0.,lane_id=6):
        self.transform=NS(location=Pos(x,y),rotation=NS(yaw=0.))
        self.lane_id=lane_id
    def next(self,m):
        p=WP(self.transform.location.x+m,self.transform.location.y,self.lane_id)
        if p.transform.location.x>5:p.road_id=6  # OpenDRIVE boundary
        return [p]
    def previous(self,m):return [WP(self.transform.location.x-m,self.transform.location.y,self.lane_id)]
    def get_left_lane(self):return WP(self.transform.location.x,self.transform.location.y+3.5,5)
    def get_right_lane(self):return None
class Carla:
    class LaneType:Driving=1


def test_v5_cut_in_has_actual_gap_and_crosses_frozen_road_segment():
    r=cut_in_candidate(WP(),Carla)
    assert r is not None
    assert r['follower_a'].transform.location.x-r['ego'].transform.location.x==20
    assert r['ego'].transform.location.x-r['follower_b'].transform.location.x==25
    assert len(r['_target_corridor_xy'])==28
    assert on_centerline(20,0,r['_target_corridor_xy'])  # target road 6
    assert not on_centerline(20,3.5,r['_target_corridor_xy'])
    assert r['_target_lane_id']==6


def test_line_distance_segments_not_endpoints_only():
    assert distance_to_polyline(4,2,[(0,0),(10,0)])==pytest.approx(2)
    with pytest.raises(ValueError):distance_to_polyline(1,1,[(0,0)])
    with pytest.raises(ValueError):distance_to_polyline(1,1,[(0,0),(0,0)])


def test_center_validity_tracks_lane_ribbon_not_frozen_road_id():
    class Map:
        def get_waypoint(self,loc,**kwargs):
            return NS(transform=NS(location=Pos(loc.x,0)),lane_width=3.5,
                      lane_type=1,is_junction=False,road_id=6,lane_id=6)
    ok,d=map_lane_center_valid(Pos(10,1.80),Map(),Carla)
    assert ok and d==pytest.approx(1.80)
    ok,d=map_lane_center_valid(Pos(10,1.90),Map(),Carla)
    assert not ok


def test_v5_speed_pair_commands_symmetric_integral():
    times=[i*.05 for i in range(160)]
    a=sum(target_speed_mps('mild_early',t,9) for t in times)
    b=sum(target_speed_mps('mild_late',t,9) for t in times)
    assert a==pytest.approx(b)
    assert target_speed_mps('merge_early',3,9)==9


def make_temp_nuplan(tmp):
    src=tmp/'capplan'/'nuplan';repo=tmp/'repo';work=tmp/'workspace'
    repo.mkdir()
    maps=src/'maps';maps.mkdir(parents=True)
    (maps/'nuplan-maps-v1.0.json').write_text('{}')
    for city in ('sg-one-north','us-ma-boston','us-nv-las-vegas-strip','us-pa-pittsburgh-hazelwood'):
        folder=maps/city/'version';folder.mkdir(parents=True)
        (folder/'map.gpkg').write_bytes(b'test')
    val=src/'nuplan-v1.1'/'splits'/'val';val.mkdir(parents=True)
    (val/'log_a.db').write_bytes(b'fake')
    behavior=tmp/'behaviors';behavior.mkdir();(behavior/'sample.pkl').write_bytes(b'NOT UNPICKLED')
    return src,repo,work,behavior


def test_reactsim_stage_only_creates_symlinks_without_copying(tmp_path):
    src,repo,work,behavior=make_temp_nuplan(tmp_path)
    dry=inspect(src,repo,work,behavior,['val'])
    assert dry['db_link_status']=={'would_link':1}
    assert 'missing_source' not in dry['root_link_status'].values()
    assert not work.exists()
    ready=inspect(src,repo,work,behavior,['val'],apply=True)
    assert ready['ready_for_preprocess'] is True
    db=work/'nuplan_db_trainval_links'/'log_a.db'
    assert db.is_symlink() and db.resolve()==src/'nuplan-v1.1'/'splits'/'val'/'log_a.db'
    assert (repo/'data'/'nuplan'/'maps').is_symlink()
    assert (repo/'data'/'nuplan'/'nuplan-v1.1'/'splits'/'trainval').is_symlink()
    again=inspect(src,repo,work,behavior,['val'],apply=True)
    assert again['db_link_status']=={'ready':1}
    assert (src/'nuplan-v1.1'/'splits'/'val'/'log_a.db').read_bytes()==b'fake'


def test_reactsim_refuses_test_split_and_conflict(tmp_path):
    src,repo,work,behavior=make_temp_nuplan(tmp_path)
    with pytest.raises(ValueError,match='test'):list_dbs(src,['test'])
    (repo/'data'/'custom_behaviors').mkdir(parents=True)
    outcome=inspect(src,repo,work,behavior,['val'],apply=False)
    assert outcome['root_link_status'][str(repo/'data'/'custom_behaviors')]=='conflict_existing_path'


def test_smoke_requires_safe_physical_merge_not_just_valid_trace():
    from ride.carla.smoke_validate import validate
    src=[{'type':'manifest'}, {'type':'trial','trace':[{'t':1}],
            'metrics':{'valid':True,'mechanism_executed':False,'ego_collision':False}},
         {'type':'run_end','completed':True}]
    ok,status=validate(src,require_mechanism=True)
    assert not ok and status['valid_executed_ego_safe_mechanism_trials']==0
    src[1]['metrics']['mechanism_executed']=True
    src[1]['metrics']['ego_collision']=True
    assert not validate(src,require_mechanism=True)[0]
    src[1]['metrics']['ego_collision']=False
    assert validate(src,require_mechanism=True)[0]
