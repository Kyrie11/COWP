import copy
import pytest
from ride.reactsim_paired_audit import analyze


def row(action,model,progress,collision,brake=False):
    return dict(scene_id='scene_0',action_id=action,response_model=model,
        initial_state_digest='init_same',affected_actor_pair_digest='pair_same',
        action_trajectory_digest='action_'+action,source_rollout_digest='0'*64,
        progress_m=progress,ego_collision=False,third_party_collision=collision,
        hard_brake=brake,rule_valid=True,mechanism_executed=True,simulator_completed=True)


def audit(rows):
    return analyze(rows,action_a='merge',action_b='yield',model_a='SMART',model_b='MTR')


def test_normalized_strict_reversal_requires_all_four_arms():
    rows=[row('merge','SMART',56,True),row('yield','SMART',55,False),
          row('merge','MTR',56,False),row('yield','MTR',55,True)]
    report=audit(rows)
    assert report['summary']['eligible_scenes']==1
    assert report['summary']['strict_posthoc_reversals']==1
    assert report['summary']['rq1_ex_ante_planner_choice_proven'] is False
    assert audit(rows[:3])['summary']['invalid_reasons']=={'missing_arm':1}


def test_normalized_refuses_leaky_or_unmatched_futures():
    rows=[row('merge','SMART',56,True),row('yield','SMART',55,False),
          row('merge','MTR',56,False),row('yield','MTR',55,True)]
    modified=copy.deepcopy(rows);modified[0]['action_trajectory_digest']='different_action'
    assert audit(modified)['summary']['invalid_reasons']=={'different_ego_action_across_models':1}
    modified=copy.deepcopy(rows);modified[0]['ego_collision']=True
    assert audit(modified)['summary']['invalid_reasons']=={'unsafe_illegal_or_unexecuted_action':1}
    modified=copy.deepcopy(rows);modified[0]['progress_m']=90
    assert audit(modified)['summary']['invalid_reasons']=={'progress_unmatched':1}


def test_bad_schema_fails_closed():
    rows=[row('merge','SMART',56,True)]
    rows[0]['simulator_completed']='true'
    with pytest.raises(ValueError,match='boolean'):audit(rows)
