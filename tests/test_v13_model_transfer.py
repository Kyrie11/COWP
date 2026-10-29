import json
import pytest
from pathlib import Path

from ride.v13_model_transfer import (
    extract_trial_cost,lock_choices,evaluate_lock,_do_lock,_do_evaluate,
    write_new,sha256,
)


def frame(i, action, burden):
    t=(i+1)*0.05
    # synthetic future C braking vs no additional braking, after startup
    cacc=-(2.8 if burden and t>=2.0 else 0.5)
    ego_speed=7.2 if (action=='speed_early' and 1.5<=t<3.5) else 7.8 if (action=='speed_late' and 1.5<=t<3.5) else 7.5
    return {'t':t,'follower_c_accel':cacc, 'ego_speed_mps':ego_speed,
            'ego_target_speed_mps':ego_speed,
            'xy_by_role':{'ego':[ego_speed*t,0.]}}


def trial(action='speed_early',family='behavior',burden=False):
    return {'type':'trial','archetype':'chain_cut_in', 'anchor':0,'seed':17,
            'family':family,'profile':action, 'joint_targets':None,'recovery':None,
            'metrics':{'valid':True,'mechanism_executed':True,
                'ego_collision':False,'non_ego_pair_collision':False,
                'min_rear_pair_ttc_s':15.,'ttc_applicable_steps':160,
                'follower_c_max_sustained_brake_s':0.,
                'ego_progress_m':50.0 if action=='speed_late' else 51.0},
            'initial_states':{r:{'xyz':[float(i),0.,0.], 'speed_mps':7.5,'yaw_deg':0.0}
                              for i,r in enumerate(('ego','follower_a','follower_b','follower_c'))},
            'trace':[frame(i,action,burden) for i in range(160)]}


def meta():
    return {'type':'manifest','simulator':'CARLA','archetype':'chain_cut_in',
        'map':'Carla/Maps/Town04','seed':17,'anchor_protocol_version':'ride_lane_corridor_v12_20261028',
        'anchors':[{'anchor':0,'key':[1,2,3]}], 'duration_s':8.0,'cruise_mps':7.5,
        'warmup_ticks':20,'rear_speed_delta_mps':0,'cutin_ego_gap_m':20.,
        'cutin_rear_gap_m':35.,'chain_bc_gap_m':20.,'chain_c_speed_delta_mps':0.5,
        'cutin_late_switch_s':3.5,'cutin_lateral_lookahead_m':20.,
        'cutin_paired_amplitude_mps':0.5,'behavior_speed_protocol':'nominal_cap',
        'behavior_route_protocol':'frozen_corridor'}


def rows(family,burden_for_early=False):
    return [meta(),trial('speed_early',family,burden_for_early),
            trial('speed_late',family,False),
            {'type':'run_end','completed':True,'trial_count':2}]


def test_source_tie_induced_choice_error_detectable():
    src=rows('behavior')
    tgt=rows('tm',True)
    choices=lock_choices(src,source_family='behavior',target_family='tm',
                         frozen_sha='1',source_sha='2',expected_target_path='/tmp/target.jsonl')
    result=evaluate_lock(choices,src,tgt)
    pair=[r for r in result['results'] if r['pair']==['speed_early','speed_late'] and r['policy']=='SOURCE_THIRD_PARTY'][0]
    assert pair['selected_action']=='speed_early'
    assert pair['scientific_status']=='AVOIDABLE_C_BURDEN_LOSS'
    assert pair['model_blind_distinction']
    assert pair['paired_model_induced_selection_error']
    assert pair['selected_C_burden_excess_mps']>0.25


def test_no_false_positive_when_no_target_difference():
    src=rows('behavior')
    choices=lock_choices(src,source_family='behavior',target_family='tm',
                         frozen_sha='1',source_sha='2',expected_target_path='/tmp/target.jsonl')
    result=evaluate_lock(choices,src,rows('tm'))
    assert result['summary']['diagnostic_model_induced_choice_errors']==0


def test_placebo_never_counts_model_induced_error():
    src=rows('tm')
    choices=lock_choices(src,source_family='tm',target_family='tm',
                         frozen_sha='1',source_sha='2',expected_target_path='/tmp/t',
                         placebo_same_family=True)
    outcome=evaluate_lock(choices,src,rows('tm',True))
    assert outcome['summary']['same_model_placebo'] is True
    assert outcome['summary']['diagnostic_model_induced_choice_errors']==0

def test_same_family_refused():
    with pytest.raises(ValueError,match='distinct'):
        lock_choices(rows('tm'),source_family='tm',target_family='tm',
                     frozen_sha='1',source_sha='2',expected_target_path='/tmp/t')


def test_manifest_mismatch_fails_closed():
    choices=lock_choices(rows('behavior'),source_family='behavior',target_family='tm',
                         frozen_sha='1',source_sha='2',expected_target_path='/tmp/t')
    bad=rows('tm');bad[0]['map']='Carla/Maps/Town05'
    with pytest.raises(ValueError,match='manifest_mismatch'):
        evaluate_lock(choices,rows('behavior'),bad)


def test_missing_c_is_unknown_not_zero():
    row=trial()
    del row['trace'][23]['follower_c_accel']
    with pytest.raises(KeyError):
        extract_trial_cost(row)


def test_cross_model_ego_drift_is_unknown():
    source=rows('behavior');target=rows('tm',True)
    target[1]['trace'][25]['xy_by_role']['ego'][0]+=2.0
    choices=lock_choices(source,source_family='behavior',target_family='tm',
                         frozen_sha='1',source_sha='2',expected_target_path='/tmp/t')
    result=evaluate_lock(choices,source,target)
    assert all(r['scientific_status']=='UNKNOWN' for r in result['results'] if r['pair']==['speed_early','speed_late'])


def test_lock_refuses_prewritten_target(tmp_path):
    target=tmp_path/'target.jsonl';target.write_text('anything')
    source=tmp_path/'src.jsonl';src=rows('behavior');source.write_text('\n'.join(json.dumps(r) for r in src)+'\n')
    frozen=tmp_path/'frozen.jsonl';frozen.write_text(json.dumps(meta())+'\n')
    Args=type('A',(),{})
    a=Args();a.target=str(target);a.source=str(source);a.frozen=str(frozen)
    a.choices=str(tmp_path/'choices.json');a.commit=str(tmp_path/'commit.json')
    a.source_family='behavior';a.target_family='tm';a.min_decisions=1
    a.tie_resolution=0.05;a.meaningful_delta=0.25;a.comfort_limit=2.0
    with pytest.raises(RuntimeError,match='TARGET_ALREADY_EXISTS'):
        _do_lock(a)


def test_commit_hash_blocks_tampering(tmp_path):
    src=rows('behavior');target_rows=rows('tm',True)
    source=tmp_path/'src.jsonl';target=tmp_path/'target.jsonl';frozen=tmp_path/'frozen.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in src)+'\n')
    frozen.write_text(json.dumps(meta())+'\n')
    Args=type('A',(),{})
    a=Args();a.target=str(target);a.source=str(source);a.frozen=str(frozen)
    a.choices=str(tmp_path/'choices.json');a.commit=str(tmp_path/'commit.json')
    a.output=str(tmp_path/'audit.json');a.source_family='behavior';a.target_family='tm'
    a.min_decisions=1;a.tie_resolution=0.05;a.meaningful_delta=0.25;a.comfort_limit=2.0
    _do_lock(a)
    target.write_text('\n'.join(json.dumps(r) for r in target_rows)+'\n')
    summary=_do_evaluate(a)
    assert summary['diagnostic_model_induced_choice_errors'] == 1
    c=json.loads(Path(a.choices).read_text());c['warning']='altered'
    Path(a.choices).write_text(json.dumps(c))
    with pytest.raises(RuntimeError,match='HASH_MISMATCH'):
        _do_evaluate(a)
