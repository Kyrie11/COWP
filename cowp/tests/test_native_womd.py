import numpy as np
from research_probe.native_womd import scene_features, observational_metrics, _stable_hash
from research_probe.reactive_stress import _pair_conflicts, simulate

def test_geometry_initial_non_overlap_and_reactive():
    s=np.array([[0,0,8,0,0],[25,0,8,0,0],[55,0,8,0,0]],float)
    L=np.array([4.5]*3);W=np.array([1.9]*3);v=np.ones(3,dtype=bool)
    assert _pair_conflicts(s[:,:2],s[:,4],L,W,v)==[]
    r=simulate(s,L,W,v,0,[1,2,3],-1,'cautious',steps=8)
    assert r['collision'] is False
    assert r['near_term_ego_clear'] is True

def test_scene_selection_past_current_only():
    n=3
    s={'ego_index':0,'types':np.ones(n,int),
       'current_valid':np.ones((n,1)),'current_x':np.array([[0],[10],[15.]]),
       'current_y':np.array([[0],[0],[0]]),
       'current_velocity_x':np.array([[5.],[1.],[3.]]),
       'current_velocity_y':np.zeros((n,1))}
    f=scene_features(s)
    assert f['nearby_count']==2
    assert f['past_only_interactive']
    assert _stable_hash('a')==_stable_hash('a')

def test_native_scenario_proto_decoding_without_protobuf_install():
    from types import SimpleNamespace as O
    from research_probe.native_womd import decode_scenario_proto
    tracks=[]
    for i in range(3):
        states=[O(center_x=float(i*10+k),center_y=0.,velocity_x=3.,velocity_y=0.,heading=0.,
                  length=4.5,width=2.,valid=True) for k in range(91)]
        tracks.append(O(id=i+1,object_type=1,states=states))
    sc=O(tracks=tracks,sdc_track_index=0,current_time_index=10,scenario_id='raw_scenario_test')
    d=decode_scenario_proto(sc)
    assert d['past_x'].shape==(3,10)
    assert d['future_x'].shape==(3,80)
    assert d['current_x'][0,0]==10
    assert d['future_x'][0,0]==11
    assert d['ego_index']==0
