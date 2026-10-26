import numpy as np
from research_probe.probe import (synthetic_selftest,joint_response,path_ok,rollout,examine_scene,HORIZON)

def test_synthetic():
    assert synthetic_selftest()['passed']

def test_joint_counterexample_is_not_merely_ego_collision():
    ego=np.full((HORIZON+1,2),100.)
    zero=np.zeros((HORIZON+1,2))
    a=[[(0.,zero)],[(0.,zero+np.array([1.,0.]))]]
    assert all(path_ok(ego,v[0][1],1.,.5) for v in a)
    assert joint_response(a,[1.,1.])[0] is None

def test_independent_responses_and_successor_not_safety_guarantee():
    x=np.array([0.,0.,5.,0.,0.,2.,1.])
    p=rollout(x,-4.)
    assert p.shape==(HORIZON+1,2)
    assert abs(p[0,0])<1e-6

def test_missing_sdc_is_rejected_not_fabricated():
    from research_probe.probe import _states
    try:_states({'state/current/x':np.array([0.,1.]),'state/current/y':np.array([0.,0.])})
    except ValueError as ex:assert 'is_sdc' in str(ex)
    else:assert False

def test_oriented_rectangles_reject_circle_false_positive():
    from research_probe.footprints import circle_clear,oriented_clear
    # Two parallel vehicle bodies 3 m apart laterally are collision-free as
    # 4.5 x 1.9 rectangles, but overlap with circumscribed disc geometry.
    a=np.array([[t,0.] for t in range(10)])
    b=np.array([[t,3.] for t in range(10)])
    radius=(4.5**2+1.9**2)**.5/2
    assert not circle_clear(a,b,radius,radius,0.05)
    fp=(4.5,1.9,0.,False)
    assert oriented_clear(a,b,fp,fp,0.05)

def test_oriented_boxes_detect_crossing_overlap():
    from research_probe.footprints import oriented_clear
    a=np.array([[0.,0.],[0.5,0.]])
    b=np.array([[0.,0.],[0.,0.5]])
    assert not oriented_clear(a,b,(4.,2.,0.,True),(4.,2.,np.pi/2,True),0.05)

def test_previous_data_full_bank_incompatibility():
    from research_probe.audit_previous import audit
    row={'eligible':True,'scenario_id':'x','pairwise_joint_discordance':2,
         'joint_but_local_conflict':0,'temporal_collapse_proxy':0,'avoidable_collapse_proxy':0,
         'candidates':[{'pairwise':True,'joint':False,'local_compatible':False,'successor_viable':None}]*2}
    old={'scenarios':[row]};out=audit(old,old)
    assert out['base']['discordance_with_joint_candidate_alternative_scenes']==0
    assert out['base']['discordance_all_candidates_positive_scenes']==1
