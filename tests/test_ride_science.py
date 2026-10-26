import json
from copy import deepcopy
import pytest
from ride.synthetic_fixture import rows
from ride.evidence import audit
from ride.science import Protocol, assess_pair
from ride.decision import score
from ride.gates import evaluate
from ride.ranking import audit_rankings
from ride.carla_scenarios import yaw_diff


def test_synthetic_is_only_fixture():
    r=audit(rows())
    assert r['summary']['total_trials']==8
    assert r['summary']['valid_task_matched_contrasts']==4
    assert len(r['cross_family_same_effect'])==2
    assert r['summary']['calibrated_independent_human_model'] is False
    assert evaluate(r)['RQ3']=='NOT_TESTED_NO_LOCKED_STRONG_BASELINE_CHOICES'


def test_initial_mismatch_censors_instead_of_false_negative():
    r=rows();b=deepcopy(r[2]);b['initial_states']['ego']['xyz'][0]+=3
    first=next(x for x in r if x.get('profile')=='early_slow')
    p=assess_pair(first,b,Protocol())
    assert not p['eligible'] and 'initial' in p['reason']


def test_progress_and_ego_collision_censor():
    r=rows();a=r[1];b=deepcopy(r[2]);b['metrics']['ego_progress_m']=70
    assert assess_pair(a,b,Protocol())['reason']=='progress_unmatched'
    b['metrics']['ego_progress_m']=50;b['metrics']['ego_collision']=True
    assert assess_pair(a,b,Protocol())['reason']=='ego_collision_in_either_arm'


def test_missing_arm_does_not_count_as_uneventful():
    r=rows();r=[q for q in r if not(q.get('profile')=='late_slow' and q.get('family')=='tm')]
    s=audit(r)
    assert s['summary']['missing_contrast_arms']==1
    assert s['summary']['valid_task_matched_contrasts']==3


def test_external_planner_choice_locked_before_rollout():
    row={'method_id':'RACP_ADAPTED','archetype':'convoy','anchor':'0','seed':'7',
         'response_family':'tm','selected_action':'early_slow','choice_locked_before_rollout':'true'}
    s=score(rows(),[row])
    assert s['total_valid']==1
    assert s['results'][0]['third_party_collision_regret']==1
    unsafe=dict(row,choice_locked_before_rollout='false')
    s=score(rows(),[unsafe])
    assert s['total_valid']==0
    assert s['results'][0]['reason']=='not_precommitted'


def test_car_reproducible_angles():
    assert yaw_diff(359,1)==2
    assert yaw_diff(-170,170)==20


def test_support_empty_is_unknown_and_expansion_can_find_one():
    from ride.support import compare
    base={'type':'trial','family':'joint_witness','archetype':'support_gap','anchor':0,'seed':17,
          'profile':'early_slow','joint_targets':[6.,6.],
          'metrics':{'valid':True,'ego_collision':False,'non_ego_collision':True,'hard_brake':False}}
    manifest={'type':'manifest','simulator':'CARLA','carla_server_version':'0.9.16',
              'carla_client_version':'0.9.16','archetype':'support_gap','map':'Town04',
              'cruise_mps':9.,'dt':.05}
    base['initial_states']={role:{'xyz':[pos,0.,0.],'speed_mps':9.} for role,pos in
                            (('ego',30),('follower_a',10),('follower_b',-10))}
    base['trace']=[{'t':0.05,'xy_by_role':{'ego':[30.,0.],'follower_a':[10.,0.],
        'follower_b':[-10.,0.]}}]
    base['metrics'].update({'ego_progress_m':1.0,'follower_a_progress_m':1.0,
                             'follower_b_progress_m':1.0})
    narrow=[manifest,base]
    better=deepcopy(base)
    better['joint_targets']=[4.,6.]
    better['metrics']['non_ego_collision']=False
    result=compare(narrow,[manifest,base,better])
    assert result['summary']['finite_bank_misses']==1
    assert result['details'][0]['small_bank_no_witness_status']=='UNKNOWN'
    assert result['summary']['global_UNSAT_proven'] is False


def test_ranking_audit_discriminates_model_disagreement_from_real_planner():
    base=rows()
    r=audit_rankings(base)
    assert r['summary']['strict_risk_order_reversals']==0
    assert r['summary']['posthoc_outcome_ranking_not_actual_planner_choice']
    for x in base:
        if x.get('family')=='behavior' and x.get('profile') in ('early_slow','mild_early'):
            x['metrics']['non_ego_pair_collision']=False
            x['metrics']['hard_brake']=False
            x['metrics']['min_rear_pair_ttc_s']=4.0
        elif x.get('family')=='behavior' and x.get('profile') in ('late_slow','mild_late'):
            x['metrics']['non_ego_pair_collision']=True
            x['metrics']['hard_brake']=True
            x['metrics']['min_rear_pair_ttc_s']=0.4
    changed=audit_rankings(base)
    assert changed['summary']['strict_risk_order_reversals']>=1
    assert evaluate(audit(base),ranking=changed)['RQ1'].startswith('SIMULATOR_')


def test_ego_collision_choice_retained_in_denominator():
    trial=rows()
    trial[1]['metrics']['ego_collision']=True
    choice={'method_id':'TEST_ONLY','archetype':'convoy','anchor':'0','seed':'7',
            'response_family':'tm','selected_action':'early_slow','choice_locked_before_rollout':'true'}
    res=score(trial,[choice])
    assert res['total_submitted']==1
    assert res['total_unsafe_selected']==1
    assert res['summary']['TEST_ONLY']['unsafe_ego_collision_selected']==1
    assert res['summary']['TEST_ONLY']['valid_decisions']==0


def test_finite_bank_inconsistent_shared_outcome_cannot_be_claimed_as_positive():
    from ride.support import compare
    manifest={'type':'manifest','simulator':'CARLA','carla_server_version':'0.9.16',
              'carla_client_version':'0.9.16','archetype':'support_gap','map':'Town04',
              'cruise_mps':9.,'dt':.05}
    initial={role:{'xyz':[pos,0,0],'speed_mps':9.} for role,pos in
             (('ego',30),('follower_a',10),('follower_b',-10))}
    bad={'type':'trial','family':'joint_witness','archetype':'support_gap','anchor':0,
         'seed':17,'profile':'early_slow','joint_targets':[6.,9.],
         'metrics':{'valid':True,'ego_collision':False,'non_ego_collision':True,'hard_brake':False},
         'initial_states':initial,
         'trace':[{'t':0.05,'xy_by_role':{'ego':[30.,0.],'follower_a':[10.,0.],
         'follower_b':[-10.,0.]}}]}
    bad['metrics'].update({'ego_progress_m':1.,'follower_a_progress_m':1.,
                          'follower_b_progress_m':1.})
    good=deepcopy(bad);good['metrics']['non_ego_collision']=False
    new=deepcopy(good);new['joint_targets']=[4.,6.]
    result=compare([manifest,bad],[manifest,good,new])
    assert result['summary']['finite_bank_misses']==0
    assert result['summary']['inconclusive_shared_candidate_inconsistency']==1


def test_exante_score_selectors_are_deterministic_and_honestly_named():
    from ride.selectors import choose, make_choices
    rows=[{'archetype':'convoy','anchor':'0','seed':'17','action':str(i),
           'pred_ego_collision_prob':'0.01',
           'pred_third_party_collision_prob':str(risk),
           'pred_burden_cost':str(burden),
           'pred_progress_m':str(prog),'pred_dynamic_valid':'true','pred_rule_valid':'true'}
          for i,risk,burden,prog in [(1,.3,.4,50),(2,.05,.1,48)]]
    assert choose(rows,'safety_only')['selected_action']=='1'
    assert choose(rows,'hard_courtesy')['selected_action']=='2'
    submitted,unavailable=make_choices(rows,'hard_courtesy')
    assert len(submitted)==2 and not unavailable
    assert all(r['method_id']=='PROXY_HARD_COURTESY' for r in submitted)


def test_oriented_box_not_false_positive_across_adjacent_lanes():
    from ride.geometry import rects_intersect
    import math
    assert not rects_intersect(0,0,0,4.7,2.0,0,3.5,0,4.7,2.0)
    assert rects_intersect(0,0,0,4.7,2.0,2.,0,0,4.7,2.0)
    assert not rects_intersect(0,0,0,4.7,2.0,8.,0,0,4.7,2.0)
    assert rects_intersect(0,0,math.pi/2,4.7,2.0,0,0,0,4.7,2.0)


def test_freeze_womd_ids_ignores_logged_future_diagnostics(tmp_path):
    from ride.freeze_ids import freeze
    cohort=tmp_path/'data';cohort.mkdir()
    base={'entries':[{'scenario_id':str(i),'evaluation_partition':'development',
                      'observed_future_diagnostic':{'collision':bool(i%2)}} for i in range(25)]}
    (cohort/'cohort_manifest.json').write_text(json.dumps(base))
    a=freeze(cohort,22)
    for row in base['entries']:row['observed_future_diagnostic']={'collision':not bool(row['observed_future_diagnostic']['collision'])}
    (cohort/'cohort_manifest.json').write_text(json.dumps(base))
    b=freeze(cohort,22)
    assert a==b and len(a)==22
