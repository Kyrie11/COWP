"""V12 exact fixes: route queue, new balanced speed pair, sensitivity honesty."""
from __future__ import annotations
import json,sys,types
from pathlib import Path
from types import SimpleNamespace as N
import pytest
from ride.carla.route_protocol import forward_lane_plan,install_chain_lane_route
from ride.carla.protocol import compensated_speed_pair
from ride.protocol_lock import check,FIELDS,TEXT_FIELDS
from ride.v12_sensitivity import evaluate


class Wp:
    lane_type=1
    lane_id=6
    is_junction=False
    def __init__(self,x=0.,y=0.,n=40,*,branched=False):
        self.transform=N(location=N(x=x,y=y,z=0.),rotation=N(yaw=0.))
        self._n=n;self.branched=branched
    def next(self,d):
        if self.branched:return [Wp(),Wp()]
        return [Wp(self.transform.location.x+d,0.,self._n-1)] if self._n>0 else []


class Carla:
    class LaneType:Driving=1


def test_frozen_corridor_unique_nonjunction_route():
    path,d=forward_lane_plan(Wp(),carla=Carla,horizon_m=75.)
    assert d>=75 and len(path)>15
    with pytest.raises(RuntimeError,match='END_OR_BRANCH'):
        forward_lane_plan(Wp(n=3),carla=Carla,horizon_m=75.)
    with pytest.raises(RuntimeError,match='END_OR_BRANCH'):
        forward_lane_plan(Wp(branched=True),carla=Carla,horizon_m=75.)


def test_behavior_global_plan_installed_without_changing_run_step(monkeypatch):
    # Do not require CARLA in offline tests.
    nav=types.ModuleType('agents.navigation.local_planner')
    nav.RoadOption=N(LANEFOLLOW='LANEFOLLOW')
    monkeypatch.setitem(sys.modules,'agents',types.ModuleType('agents'))
    monkeypatch.setitem(sys.modules,'agents.navigation',types.ModuleType('agents.navigation'))
    monkeypatch.setitem(sys.modules,'agents.navigation.local_planner',nav)
    class Agent:
        def __init__(self):self.plan=[];self._local_planner=self
        def set_global_plan(self,plan,**kwargs):
            self.plan=list(plan);self.options=kwargs
        def get_incoming_waypoint_and_direction(self,steps=3):
            return (self.plan[steps] if steps<len(self.plan) else (None,''))
    class Road:
        def get_waypoint(self,loc,**kwargs):return Wp(n=60)
    a=Agent()
    details=install_chain_lane_route(a,N(get_location=lambda:N(x=0,y=0,z=0)),Road(),Carla,
                                      cruise_mps=7.5,duration_s=8.)
    assert details['route_length_m']>=78
    assert details['protocol']=='explicit_frozen_map_corridor_v12'
    assert a.plan[0][1]=='LANEFOLLOW'
    assert a.options=={'stop_waypoint_creation':True,'clean_queue':True}


def test_speed_pair_has_matching_nominal_integral_but_distinct_commands():
    dt=.05
    ts=[i*dt for i in range(160)]
    first=[compensated_speed_pair('speed_early',t,7.5,.5) for t in ts]
    second=[compensated_speed_pair('speed_late',t,7.5,.5) for t in ts]
    assert sum(first)*dt==pytest.approx(sum(second)*dt)
    assert first[40]<second[40]
    assert first[85]>second[85]
    assert min(first)>0
    with pytest.raises(ValueError):compensated_speed_pair('speed_late',2,7.5,2)


def test_v12_confirmation_rejects_route_or_amplitude_drift():
    frozen={'type':'manifest','anchor_protocol_version':'ride_lane_corridor_v12_20261028',
       **{x:3.5 for x in FIELDS},**{x:'frozen_corridor' for x in TEXT_FIELDS}}
    d={x:frozen[x] for x in FIELDS+TEXT_FIELDS}
    assert check(frozen,d,required_version='ride_lane_corridor_v12_20261028')=={}
    d['behavior_route_protocol']='native_shortest'
    assert 'behavior_route_protocol' in check(frozen,d,required_version='ride_lane_corridor_v12_20261028')


def test_v12_readonly_sensitivity_is_exploratory(tmp_path):
    from test_v11_chain_evidence import row
    t=row('treatment');t.update(type='trial',anchor=0,seed=17,family='tm')
    b=row('cruise');b.update(type='trial',anchor=0,seed=17,family='tm')
    manifest={'type':'manifest','map':'Town04','anchors':[{'key':[0.,0.,0.]}],
              'anchor_protocol_version':'ride_lane_corridor_v12_20261028',
              'behavior_route_protocol':'frozen_corridor'}
    with (tmp_path/'trial.jsonl').open('w') as f:
        for d in [manifest,t,b,{'type':'run_end','completed':True,'trial_count':2}]:
            f.write(json.dumps(d)+'\n')
    a=evaluate([tmp_path/'trial.jsonl'])
    assert a['counts']['eligible_negative_controls']==1
    assert a['counts']['verified_ex_ante_planner_choices']==0
    assert a['threshold_sensitivity'][0]['by_family'][0]['B_before_C']==1
    assert a['primary_RQ2']=='NOT_ESTABLISHED_BY_TEMPORAL_ORDER_OR_CRUISE_CONTROL'
