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
