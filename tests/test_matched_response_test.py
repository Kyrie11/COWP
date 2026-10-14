import numpy as np
from research_probe.matched_response_test import make_profiles
from research_probe.reactive_stress import simulate


def test_frozen_smooth_profiles_displacement_match_without_clipping():
    profiles=make_profiles(steps=20,base_accels=(0.,),amplitudes=(1.5,))
    assert len(profiles)==3
    plus=next(p for p in profiles if p['amplitude']>0)
    minus=next(p for p in profiles if p['amplitude']<0)
    assert np.max(np.abs(plus['controls']))<=1.5+1e-9
    assert abs(float(np.sum(plus['controls'])))<1e-10
    assert abs(float(np.sum(minus['controls'])))<1e-10
    s=np.array([[0.,0.,12.,0.,0.],[80.,0.,12.,0.,0.],[100.,0.,12.,0.,0.]])
    L=np.array([4.5]*3);W=np.array([1.9]*3);valid=np.ones(3,bool)
    pos=[]
    for profile in (plus,minus):
        o=simulate(s,L,W,valid,0,[1,2,3],profile['controls'],'cautious',steps=20)
        pos.append(float(o['ego_progress_m']))
    assert abs(pos[0]-pos[1])<1e-8


def test_profiles_frozen_and_well_typed():
    ids=[p['id'] for p in make_profiles()]
    assert len(set(ids))==len(ids)
    assert len(ids)==15
