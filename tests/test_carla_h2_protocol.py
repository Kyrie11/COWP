import math
import pytest
from research_probe.carla_h2.protocol import target_speed_mps,check_contrast,closing_ttc,summarise_trial
from research_probe.carla_h2.analyze_carla import analyze

def test_equal_command_integrals():
    dt=0.05
    for base in (8.,9.,11.):
        integral=[sum(target_speed_mps(x,i*dt,base)*dt for i in range(160)) for x in
                  ('cruise','early_slow','late_slow','mild_early','mild_late')]
        assert max(integral)-min(integral)<1e-5

def test_ttc():
    assert abs(closing_ttc(-10,0,10,5)-1.1)<1e-9
    assert closing_ttc(-10,0,5,10)==float('inf')

def test_matching_progress_required():
    a={'valid':True,'ego_progress_m':40,'non_ego_pair_collision':True,'ego_collision':False,'min_follower_accel_mps2':-6,'min_rear_pair_ttc_s':0.9}
    b={'valid':True,'ego_progress_m':41,'non_ego_pair_collision':False,'ego_collision':False,'min_follower_accel_mps2':-2,'min_rear_pair_ttc_s':4}
    assert check_contrast(a,b)['strict_third_party_collision_preventable']
    b['ego_progress_m']=47
    assert check_contrast(a,b)['why']=='task_progress_mismatch'

def test_no_false_claims_or_fake_independence():
    def trial(family,profile,coll):
        base={'valid':True,'ego_progress_m':40,'non_ego_pair_collision':coll,'ego_collision':False,
            'min_follower_accel_mps2':-3,'min_rear_pair_ttc_s':2,
            'follower_a_progress_m':35,'follower_b_progress_m':32}
        init={r:{'xyz':[i*10.,0.,0.],'speed_mps':9.} for i,r in enumerate(('ego','follower_a','follower_b'))}
        return {'type':'trial','anchor':0,'seed':7,'family':family,'profile':profile,'metrics':base,'initial_states':init}
    rows=[{'type':'manifest','simulator':'CARLA'}]
    for fam in ('tm','behavior'):
        rows+=[trial(fam,'early_slow',True),trial(fam,'late_slow',False)]
    result=analyze(rows)
    assert result['summary']['cross_family_same_action_strict_cases']==1
    assert result['summary']['human_H2_proven'] is False
    assert result['summary']['H4_baseline_run'] is False

def test_joint_search_is_positive_witness_only():
    rows=[{'type':'manifest'}]
    ans=analyze(rows)
    assert ans['summary']['bounded_joint_safe_witnesses']==0
    assert not ans['summary']['human_H2_proven']

def test_bad_offroad_rejects_trial():
    row={'ego_collision':False,'non_ego_collision':False,'trace':[
        {'road_valid_all':False,'ego_progress_m':1,'follower_a_progress_m':0,'follower_b_progress_m':-1,
         'rear_pair_ttc_s':9999.,'rear_pair_gap_m':10.,
         'follower_a_accel':0.,'follower_b_accel':0.,'ego_target_speed_mps':8.,'ego_speed_mps':8.}]}
    assert not summarise_trial(row)['valid']
