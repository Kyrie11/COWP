"""Critical regression guards added after the uploaded v7 scientific audit."""
import json
import math
from pathlib import Path
import pytest
from ride.carla.tm_policy import speed_limit_percentage,configure_tm_speed,nominal_target_speeds
from ride.nuplan_recovery_plan import make_plan
from ride.v8_gate_audit import compare_panels
from ride.gameformer_headless import patch


class FakeTM:
    def __init__(self):self.value=None
    def vehicle_percentage_speed_difference(self,veh,pct):self.value=pct

class FakeVeh:
    def get_speed_limit(self):return 72.0  # km/h


def test_tm_nominal_not_road_speed_limit():
    tm=FakeTM();pct,lim=configure_tm_speed(tm,FakeVeh(),9.)
    assert pct==pytest.approx(55.0)
    assert lim==72
    assert tm.value==pytest.approx(55.0)
    assert pct!=0.

@pytest.mark.parametrize('limit,nominal',[(0,9),(-1,9),(72,0),(20,9),(72,float('nan'))])
def test_invalid_tm_nominal_halts(limit,nominal):
    with pytest.raises(ValueError):speed_limit_percentage(limit,nominal)


def test_tm_dynamic_role_speed_and_missing_split_separation():
    assert nominal_target_speeds(9.,1.0,'cut_in')=={'follower_a':9.,'follower_b':10.}
    assert nominal_target_speeds(9.,1.0,'convoy')=={'follower_a':9.,'follower_b':9.}
    cov={'ids_file':'read-only.json','groups':[
        {'db_name':'a.db','log_name':'2021.06.01','scenario_count':2,'status':'MISSING_IN_ALL_LOCAL_SPLITS'},
        {'db_name':'b.db','log_name':'2021.08.01','scenario_count':4,'status':'AVAILABLE_IN_STAGED_SPLITS'}]}
    r=make_plan(cov)
    assert r['missing_log_filenames']==1 and r['missing_scenarios_filename_only']==2
    assert r['missing_by_month']=={'2021.06':1}
    assert r['missing_db_filenames']==['a.db']


def test_protocol_drift_and_physical_holdout_are_separate_failures():
    dev={'manifest':{'map':'Town04','cutin_late_switch_s':3.5,'cutin_lateral_lookahead_m':20},
         'anchor_keys':[{'anchor':1,'key':[10,10,0]}], 'complete':True}
    repeated={'path':'repeat','manifest':{'map':'Town04','cutin_late_switch_s':4.5,'cutin_lateral_lookahead_m':15},
              'anchor_keys':[{'anchor':9,'key':[10,15,0]}],'complete':True}
    result=compare_panels([dev,repeated])[0]
    assert result['protocol_drift'] and result['not_spatially_independent_of_dev']
    assert result['independent_confirmation'] is False
    independent=dict(repeated,manifest=dict(dev['manifest']),anchor_keys=[{'anchor':9,'key':[110,110,0]}])
    assert compare_panels([dev,independent])[0]['independent_confirmation'] is True


def test_gameformer_headless_only_suppresses_gui():
    s='def main():\n    simulate()\n    build_nuboard(builder, simulation_file)\n'
    out=patch(s)
    assert 'simulate()' in out and "RIDE_HEADLESS" in out
    assert "build_nuboard(builder, simulation_file)" in out
    with pytest.raises(ValueError):patch('no GUI call here')


def test_dev_confirmation_frozen_parameter_lock():
    from ride.protocol_lock import check,FIELDS
    src={x:3.5 for x in FIELDS}
    src.update(type='manifest',anchor_protocol_version='ride_lane_corridor_v8_20261028')
    actual={x:3.5 for x in FIELDS}
    assert check(src,actual)=={}
    actual['cutin_lateral_lookahead_m']=15.0
    assert 'cutin_lateral_lookahead_m' in check(src,actual)
    src['anchor_protocol_version']='ride_lane_corridor_v7_20261028'
    assert 'anchor_protocol_version' in check(src,actual)


def test_sustained_speed_drift_not_one_frame_spike():
    from ride.carla.response_calibration import sustained_overspeed
    trace=[{'t':i*.05,'follower_a_speed_mps':20. if 60<=i<75 else 9.} for i in range(100)]
    assert sustained_overspeed(trace,'follower_a',9.)
    trace=[{'t':i*.05,'follower_a_speed_mps':20. if i==70 else 9.} for i in range(100)]
    assert not sustained_overspeed(trace,'follower_a',9.)
