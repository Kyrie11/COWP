import json
import math
import sqlite3
import zipfile
from pathlib import Path
import pytest
from ride.carla.lane_geometry import project_to_polyline
from ride.science import same_initial,Protocol
from ride.choice_contract import validate_record
from ride.nuplan_extract_missing import catalog,extract
from ride.reactsim_recovery import plan
from ride.reactsim_subset import subset
from ride.carla.pilot_report import report
from ride.ranking import audit_rankings


def test_station_straight():
    assert project_to_polyline(3,2,[(0,0),(10,0)])[:2] == (3.,2.)


def test_station_curve_and_road_boundary():
    s,d,tx,ty=project_to_polyline(11,4,[(0,0),(10,0),(10,10)])
    assert math.isclose(s,14) and d==1 and (tx,ty)==(0,1)


def test_station_degenerate_or_nonfinite_fails():
    with pytest.raises(ValueError):project_to_polyline(2,0,[(0,0),(0,0)])
    with pytest.raises(ValueError):project_to_polyline(float('nan'),0,[(0,0),(10,0)])


def _initial(yaw=359.9):
    return {'initial_states':{x:{'xyz':[0,0,0],'speed_mps':7,'yaw_deg':yaw}
                              for x in ('ego','follower_a','follower_b')}}


def test_yaw_wrap_and_yaw_mismatch():
    assert same_initial(_initial(359.9),_initial(.1),Protocol())[0]
    assert not same_initial(_initial(180),_initial(10),Protocol())[0]
    assert 'initial_yaw_mismatch' in same_initial(_initial(180),_initial(10),Protocol())[1]


def test_partial_yaw_rejected():
    right=_initial();del right['initial_states']['ego']['yaw_deg']
    assert same_initial(_initial(),right,Protocol())[1].startswith('partial_initial_yaw')


def test_choice_contract_fails_closed():
    row={'scene_id':'s','initial_state_sha256':'a'*64,'checkpoint_sha256':'b'*64,
         'candidates':['a','b'],'candidate_scores':{'a':1,'b':2},'selected_action':'a',
         'planner':'someplanner','planner_commit':'tag','response_model_assumption':'m',
         'choice_locked_before_rollout':True,'information_cutoff_us':1000}
    assert validate_record(row)[0]
    assert not validate_record({**row,'choice_locked_before_rollout':False})[0]
    assert not validate_record({**row,'candidate_scores':{'a':1}})[0]


def test_zip_catalog_and_selective_extract(tmp_path):
    zipf=tmp_path/'nuplan-v1.1_val.zip'
    with zipfile.ZipFile(zipf,'w') as z:z.writestr('val/2021.a.db',b'not_sqlite')
    targets=tmp_path/'missing.txt';targets.write_text('2021.a.db\n2021.b.db\n')
    splits=tmp_path/'splits';(splits/'val').mkdir(parents=True)
    assert len(catalog(zipf,targets,'val'))==1
    assert extract(zipf,targets,splits,'val')['matched_in_archive']==1
    assert not list((splits/'val').iterdir())
    with pytest.raises(ValueError):extract(zipf,targets,splits,'test')


def test_missing_by_log_not_by_token(tmp_path):
    cov={'groups':[{'db_name':'foo.db','status':'MISSING_IN_ALL_LOCAL_SPLITS'},
                   {'db_name':'bar.db','status':'MISSING_IN_ALL_LOCAL_SPLITS'},
                   {'db_name':'yes.db','status':'AVAILABLE_IN_STAGED_SPLITS'}]}
    p=tmp_path/'c.json';p.write_text(json.dumps(cov));(tmp_path/'foo.db').write_bytes(b'x')
    d=plan(p,[tmp_path]);assert d['missing_distinct_logs']==2
    assert d['status_counts']=={'FOUND_IN_EXTRA_ROOT':1,'NOT_FOUND_IN_SCANNED_SOURCES':1}


def test_ranking_v8_does_not_drop_ego_execution_parity():
    from ride.ranking import audit_rankings
    man={'type':'manifest','simulator':'CARLA','archetype':'cut_in',
         'anchor_protocol_version':'ride_lane_corridor_v8_20261028'}
    # Stub trial with absent trajectories cannot be upgraded to valid v8 ranking.
    rows=[man]
    for fam in ('tm','behavior'):
        for action in ('early_slow','late_slow'):
            rows.append({'type':'trial','family':fam,'profile':action,'anchor':0,'seed':17,
                         'archetype':'cut_in','metrics':{'valid':True,'mechanism_executed':True,
                           'ego_progress_m':40,'ego_collision':False},
                         'initial_states':_initial()['initial_states'],
                         'trace':[]})
    r=audit_rankings(rows)
    assert r['summary']['task_matched_cross_model_pairs']==0
    assert any('missing_or_unpaired_ego_execution' in reason
               for x in r['all_pairs'] for reason in x['reasons'])


def test_v8_secondary_metric_rejects_unsupported_margin():
    from ride.secondary_effect_gate import audit_secondary
    import pytest
    with pytest.raises(ValueError):
        audit_secondary([],margin=-.25)


def test_v8_anchor_overlap_precommit_only(tmp_path):
    import json
    from ride.anchor_disjoint import compare
    a=tmp_path/'a.jsonl'; b=tmp_path/'b.jsonl'
    base={'type':'manifest','map':'Carla/Maps/Town04','anchors':[{'coords':{'ego':[0,0,0]}}]}
    a.write_text(json.dumps(base)+'\n')
    changed={**base,'anchors':[{'coords':{'ego':[6,5,0]}}]}
    b.write_text(json.dumps(changed)+'\n')
    assert compare(a,b)['overlap_count']==1
    changed['anchors']=[{'coords':{'ego':[100,50,0]}}]
    b.write_text(json.dumps(changed)+'\n')
    assert compare(a,b)['position_disjoint_heuristic']
