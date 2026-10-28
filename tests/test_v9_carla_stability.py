import json
from pathlib import Path
from ride.carla.m0_gate import audit_rows
from ride.carla.tm_policy import speed_limit_percentage
from ride.carla_scenarios import ANCHOR_PROTOCOL_VERSION

def test_valid_30kmh_cruise():
    assert abs(speed_limit_percentage(30.,8.)-4.)<1e-8

def test_9mps_violates_30kmh():
    import pytest
    with pytest.raises(ValueError,match='TM_DESIRED_SPEED_EXCEEDS_LIMIT'):
        speed_limit_percentage(30.,9.)

def test_manifest_protocol_bumped():
    assert ANCHOR_PROTOCOL_VERSION=='ride_lane_corridor_v9_20261028'

def test_m0_fails_on_missing_tm_trials():
    rows=[{'type':'manifest','simulator':'CARLA'},
          {'type':'trial','anchor':0,'seed':17,'family':'behavior','profile':'early_slow',
             'metrics':{'valid':True,'mechanism_executed':True},'trace':[{'t':0.05}]},
          {'type':'run_end','completed':True,'trial_count':1}]
    result=audit_rows(rows)
    assert result['status']=='M0_MECHANISM_OR_VALIDITY_BLOCKER'
    assert result['valid_cross_family_task_matched_pairs']==0

def test_m0_incomplete_panel_not_negative():
    r=audit_rows([{'type':'manifest','simulator':'CARLA'}])
    assert r['status']=='INCOMPLETE_PANEL_NOT_SCIENTIFIC_DATA'

def test_no_automatic_map_swap_and_no_default_low_quality():
    run=(Path(__file__).resolve().parents[1]/'ride/carla/run.py').read_text()
    assert 'RIDE_MAP_MISMATCH_REFUSE_AUTO_LOAD' in run
    assert 'if not a.allow_world_switch' in run
    sh=(Path(__file__).resolve().parents[1]/'scripts/02_start_carla_headless.sh').read_text()
    assert 'RIDE_CARLA_QUALITY:-Epic' in sh
    assert '-quality-level=Low \\n' not in sh
