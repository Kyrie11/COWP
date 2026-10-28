"""Offline regression fixtures: no CARLA, torch or nuPlan installation required."""
import json
import sqlite3
from pathlib import Path

from ride.reactsim_tokens import inspect_token, audit_ids
from ride.carla.mechanism_audit import summarize_one, _sustained_start, audit


def make_db(p):
    con=sqlite3.connect(p)
    con.executescript('''
    CREATE TABLE scene(token BLOB PRIMARY KEY, name TEXT);
    CREATE TABLE lidar_pc(token BLOB PRIMARY KEY, timestamp INTEGER, scene_token BLOB, lidar_token BLOB);
    CREATE TABLE lidar(token BLOB PRIMARY KEY, log_token BLOB);
    CREATE TABLE log(token BLOB PRIMARY KEY, map_version TEXT);
    CREATE TABLE scenario_tag(lidar_pc_token BLOB, type TEXT);
    ''')
    for i in range(1,7):
        con.execute('INSERT INTO scene VALUES (?,?)', (i.to_bytes(8,'big'), f'scene_{i:02}'))
    con.execute('INSERT INTO log VALUES (?,?)',(b'log_one_', 'us-nv-las-vegas-strip'))
    con.execute('INSERT INTO lidar VALUES (?,?)',(b'lidar_01',b'log_one_'))
    con.execute('INSERT INTO lidar_pc VALUES (?,?,?,?)',(bytes.fromhex('2cdb26865dfa5261'),123, (3).to_bytes(8,'big'), b'lidar_01'))
    con.execute('INSERT INTO lidar_pc VALUES (?,?,?,?)',(bytes.fromhex('bbbbbbbbbbbbbbbb'),456, (1).to_bytes(8,'big'), b'lidar_01'))
    con.commit();con.close()


def test_sqlite_diagnostics_classify_raw_vs_query_filter(tmp_path):
    p=tmp_path/'one.db';make_db(p)
    assert inspect_token(p,'2cdb26865dfa5261')['status']=='ELIGIBLE_BY_SQL_JOINS'
    assert inspect_token(p,'bbbbbbbbbbbbbbbb')['status']=='FILTERED_SCENE_BOUNDARY'
    assert inspect_token(p,'0000000000000000')['status']=='RAW_TOKEN_ABSENT'
    assert inspect_token(p,'abc')['status']=='MALFORMED_TOKEN'
    assert inspect_token(tmp_path/'none.db','2cdb26865dfa5261')['status']=='DB_MISSING'


def test_official_ids_audit_does_not_drop_failures(tmp_path):
    db_dir=tmp_path/'dbs';db_dir.mkdir();make_db(db_dir/'one.db')
    ids=tmp_path/'ids.json'
    ids.write_text(json.dumps({'groups':[{'log_name':'one','scenario_tokens':
           ['2cdb26865dfa5261','bbbbbbbbbbbbbbbb','0000000000000000']}]}))
    r=audit_ids(db_dir,ids)
    assert r['checked_ids']==3
    assert not r['eligible_for_full_official_preprocess']
    assert r['status_counts']=={'ELIGIBLE_BY_SQL_JOINS':1,'FILTERED_SCENE_BOUNDARY':1,'RAW_TOKEN_ABSENT':1}
    r2=audit_ids(db_dir,ids,max_tokens=1)
    assert not r2['full_list_audited']
    assert not r2['eligible_for_full_official_preprocess']


def test_braking_timing_distinguishes_command_from_full_lane_entry():
    trace=[]
    for i in range(160):
        t=round((i+1)*.05,4)
        trace.append({'t':t, 'ego_lane_id':5, 'target_lane_id':6,
             'ego_target_waypoint_lane_id':6 if t>=1.5 else 5,
             'ego_in_target_lane':t>=3,
             'follower_a_accel':0.,'follower_b_accel':-5. if 1.7<=t<=2.15 else 0.})
    row={'type':'trial','archetype':'cut_in','anchor':0,'seed':17,'family':'tm',
         'profile':'merge_early','trace':trace,
         'metrics':{'valid':True,'mechanism_executed':True,'ego_progress_m':71.}}
    r=summarize_one(row)
    assert r['first_lateral_command_s']==1.5
    assert r['first_ego_target_lane_s']==3.
    assert r['first_sustained_brake_b_s']==1.7
    assert r['burden_b_mps']>0


def test_negative_control_nonmerged_brake_is_not_causal_proof():
    trace=[]
    for i in range(160):
        t=round((i+1)*.05,4)
        trace.append({'t':t,'follower_a_accel':0.,'follower_b_accel':-5. if 1.5<=t<=2.0 else 0.,
            'target_lane_id':6,'ego_lane_id':5,'ego_target_waypoint_lane_id':5,
            'ego_in_target_lane':False})
    row={'type':'trial','archetype':'cut_in','profile':'cruise','family':'tm','anchor':0,
         'seed':17,'trace':trace,'metrics':{'valid':True,'mechanism_executed':False}}
    rows=[{'type':'manifest','simulator':'CARLA','anchor_protocol_version':'v5'},row,
          {'type':'run_end','completed':True}]
    r=audit(rows)
    assert r['complete']
    assert r['summary']['cruise_rear_braking_trials']==1
    assert r['summary']['valid_task_matched_mechanism_pairs']==0


def test_ego_execution_parity_is_not_implied_by_matching_labels():
    from ride.science import same_ego_execution
    a={'trace':[{'t':.05,'ego_target_speed_mps':9., 'ego_speed_mps':9.,
                 'xy_by_role':{'ego':[0.,0.]}}]}
    b={'trace':[{'t':.05,'ego_target_speed_mps':9.,'ego_speed_mps':9.,
                 'xy_by_role':{'ego':[1.,0.]}}]}
    ok,why=same_ego_execution(a,b)
    assert not ok and why=='cross_family_ego_execution_mismatch'
    b['trace'][0]['xy_by_role']['ego']=[0.05,0.]
    assert same_ego_execution(a,b)[0]


def test_cut_in_gap_grid_has_controlled_physical_separation():
    # Reuse the in-memory CARLA waypoint fixture; no physics required.
    from test_ride_v5_scenario_nuplan import WP,Carla
    from ride.carla_scenarios import cut_in_candidate
    narrow=cut_in_candidate(WP(),Carla,rear_gap_m=35.)
    assert narrow is not None
    assert narrow['follower_a'].transform.location.x - narrow['ego'].transform.location.x==20
    assert narrow['ego'].transform.location.x-narrow['follower_b'].transform.location.x==15
    wide=cut_in_candidate(WP(),Carla,rear_gap_m=45.)
    assert wide['ego'].transform.location.x-wide['follower_b'].transform.location.x==25
    import pytest
    with pytest.raises(ValueError):cut_in_candidate(WP(),Carla,rear_gap_m=30.)


def test_pilot_report_does_not_promote_invalid_missing_results(tmp_path):
    from ride.carla.pilot_report import report
    r=report(tmp_path)
    assert r['cells_found']==0 and r['valid_cells']==0
