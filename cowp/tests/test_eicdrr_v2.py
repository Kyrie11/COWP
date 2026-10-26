"""Logic-level regression tests for EICDRR V2 (synthetic, NOT empirical)."""
import numpy as np
import pytest
from research_probe.h1_compatibility_csp import exact_pairwise_csp
from research_probe.h3_successor_proxy import analyze_scene as h3_analyze
from research_probe.native_womd import verify_shard_inventory
from research_probe.paired_intervention import analyze_scene as paired_analyze
from research_probe.reactive_stress import simulate


def _scene(tmp_path):
    x=np.asarray([[0.],[22.],[-19.]])
    y=np.zeros((3,1))
    d={'agent_ids':np.array([1,2,3],np.int64),'ego_index':np.array(0,dtype=np.int16)}
    for key,val in [('x',x),('y',y),('velocity_x',np.asarray([[8.],[6.],[10.]])),
                    ('velocity_y',y),('bbox_yaw',y),('length',np.full((3,1),4.4)),
                    ('width',np.full((3,1),1.9)),('valid',np.ones((3,1)))]:
        d['current_'+key]=val
    d['future_x']=np.repeat(x,80,axis=1)
    d['future_y']=np.repeat(y,80,axis=1)
    d['future_valid']=np.ones((3,80))
    p=tmp_path/'test.npz'
    np.savez_compressed(p,**d)
    return p


def test_csp_three_valued():
    d={1:[0,1],2:[0,1]}
    ban_all={(1,2,a,b):True for a in range(2) for b in range(2)}
    assert exact_pairwise_csp(d,ban_all)[0]=='UNSAT'
    assert exact_pairwise_csp(d,ban_all,max_nodes=1)[0]=='UNKNOWN'
    ban_equal={(1,2,a,b):a==b for a in range(2) for b in range(2)}
    kind,_,witness=exact_pairwise_csp(d,ban_equal)
    assert kind=='SAT' and witness[1]!=witness[2]


def test_shard_integrity_guard():
    from pathlib import Path
    shards=[Path(f'validation_tfexample.tfrecord-{i:05d}-of-00150') for i in range(150)]
    assert verify_shard_inventory(shards)['complete']
    with pytest.raises(ValueError):verify_shard_inventory(shards[:-1])
    with pytest.raises(ValueError):verify_shard_inventory(shards[:-1]+[shards[0]])


def test_piecewise_control_and_delayed_policy(tmp_path):
    p=_scene(tmp_path)
    with np.load(p) as d:
        st=np.c_[d['current_x'][:,0],d['current_y'][:,0],d['current_velocity_x'][:,0],
                 d['current_velocity_y'][:,0],d['current_bbox_yaw'][:,0]]
        l=d['current_length'][:,0];w=d['current_width'][:,0]
        v=d['current_valid'][:,0]>0.5;ids=d['agent_ids']
        a=simulate(st,l,w,v,0,ids,-1.,'cautious',steps=6)
        b=simulate(st,l,w,v,0,ids,[-1.]*6,'cautious',steps=6)
        np.testing.assert_allclose(a['paths_xy'],b['paths_xy'])
        c=simulate(st,l,w,v,0,ids,[-1.]*3+[0.]*3,'cautious',steps=6,responder_delay_steps=2)
        assert c['paths_xy'].shape==(7,3,2)
        with pytest.raises(ValueError):simulate(st,l,w,v,0,ids,[0,0], 'cautious',steps=6)


def test_h3_model_relative(tmp_path):
    p=_scene(tmp_path)
    r=h3_analyze(p,prefix_steps=3,total_steps=8)
    assert r['eligible'] and len(r['regimes']['heterogeneous']['actions'])==4
    assert r['claim_level']=='MODEL_RELATIVE_PREFIX_BACKUP_SURVIVAL_ONLY'


def test_pair_fine_grid(tmp_path):
    p=_scene(tmp_path)
    r=paired_analyze(p,steps=6,controls=[-1.,-.75,-.5],responder_delay_steps=2)
    assert r['eligible'] and len(r['regimes']['cautious']['actions'])==3
    assert r['responder_delay_steps']==2
    assert r['mechanism_claim_level'].startswith('DELAYED_IDM')
