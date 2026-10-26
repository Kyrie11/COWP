import numpy as np
from research_probe.matched_response_test import make_profiles
from research_probe.reactive_stress import _ttc_gap_rule_accel, simulate
from research_probe.h1_joint_witness_gate import check_row
from research_probe.h4_planner_choice_gate import check


def test_fixed_4s_prefix_exact_across_8s_extension():
    four={p['id']:p['controls'] for p in make_profiles(20)}
    eight={p['id']:p['controls'] for p in make_profiles(40,pulse_steps=20)}
    assert all(np.array_equal(v,eight[k][:20]) for k,v in four.items())
    assert np.all(eight['b+1_cos2_+1'][20:]==1)
    # Old protocol was not the same 4s action even with the same ID.
    previous={p['id']:p['controls'] for p in make_profiles(40)}
    assert not np.allclose(previous['b+0_cos2_+1'][:20],four['b+0_cos2_+1'])


def test_simulation_prefix_equivalence():
    s=np.array([[0.,0.,12.,0.,0.],[45.,0.,9.,0.,0.],[85.,0.,8.,0.,0.]])
    L=np.ones(3)*4.5;W=np.ones(3)*1.9;valid=np.ones(3,bool)
    p=next(x for x in make_profiles(20) if x['id']=='b+0_cos2_+1')['controls']
    q=next(x for x in make_profiles(40,pulse_steps=20) if x['id']=='b+0_cos2_+1')['controls']
    for r in ('cautious','ttc_gap_rule'):
        a=simulate(s,L,W,valid,0,[11,22,33],p,r,steps=20)
        b=simulate(s,L,W,valid,0,[11,22,33],q,r,steps=40)
        assert np.allclose(a['paths_xy'],b['paths_xy'][:21])
        assert np.allclose(a['accelerations'],b['accelerations'][:20])


def test_ttc_is_distinct_rule_and_safe_scalar():
    pos=np.array([[0.,0.],[10.,0.]])
    vel=np.array([12.,3.]);yaw=np.zeros(2);L=np.array([4.,4.]);W=np.array([2.,2.]);valid=np.ones(2,bool)
    braking=_ttc_gap_rule_accel(0,pos,vel,yaw,L,W,valid)
    ghost=_ttc_gap_rule_accel(0,pos,vel,yaw,L,W,valid,excluded_leader=1)
    assert braking<0 and ghost==0.


def test_h1_valid_witness_and_unsat_not_inferred():
    T=4;x=np.zeros((T,2,2));x[:,1,0]=20.+np.arange(T)*0.1
    row={'scenario_id':'a','ego_action_id':'pulse','dt':0.1,'object_ids':[99,100],
         'ego_object_id':99,'xy':x.tolist(),'yaw':np.zeros((T,2)).tolist(),
         'length':[4.,4.],'width':[2.,2.],'valid':np.ones((T,2),bool).tolist(),
         'road_valid':np.ones((T,2),bool).tolist(),'source':'test'}
    assert check_row(row)['status']=='SAT_WITNESS_WITHIN_DECLARED_MODEL'
    x[:,1,0]=0.
    row['xy']=x.tolist()
    assert check_row(row)['status']=='UNKNOWN_COLLIDING_PROPOSAL'


def test_planner_gate_requires_common_action_bank():
    r={'scenario_id':'s','planner':'method','source_implementation':'real','checkpoint_sha256':'valid-hash',
       'evaluated_actions':['bad'],'chosen_action':'bad','bad_action_id':'bad','alternative_action_id':'good',
       'bad_progress_m':10.,'alternative_progress_m':11.,'bad_valid':True,'alternative_valid':True,
       'bad_pair_risk':1.,'alternative_pair_risk':0.}
    assert check([r])['rows'][0]['choice_evidence_valid'] is False
    r['evaluated_actions'].append('good')
    assert check([r])['rows'][0]['choice_evidence_valid'] is True
