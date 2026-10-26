import json
from pathlib import Path
import numpy as np
from research_probe.h2_evidence_gate import wilson,audit
from research_probe.independent_model_gate import check
from research_probe.native_womd import locate_shards
from research_probe.reactive_stress import simulate


def test_split_exact_not_validation_interactive(tmp_path):
    for split in ('validation','validation_interactive'):
        p=tmp_path/'tf_example'/split
        p.mkdir(parents=True)
        (p/f'{split}_tfexample.tfrecord-00000-of-00150').write_bytes(b'x')
    a,_=locate_shards(tmp_path,'validation',0,'tfexample')
    b,_=locate_shards(tmp_path,'validation_interactive',0,'tfexample')
    assert len(a)==len(b)==1
    assert 'validation_interactive' not in str(a[0])
    assert 'validation_interactive' in str(b[0])


def test_wilson_bounds():
    assert wilson(0,100)[0]==0
    assert wilson(2,147)[0]<2/147<wilson(2,147)[1]


def trial(family,seed,success=True):
    return {'scenario_id':'scene','family':family,'implementation':family+'-policy',
    'checkpoint_sha256':'none_rule_based' if family=='rules' else 'abcdef123456',
    'seed':seed,'bad_action_id':'a','alternative_action_id':'b','pair_object_ids':[44,22],
    'bad_pair_collision':success,'alternative_pair_collision':False,
    'bad_ego_collision':False,'alternative_any_collision':False,
    'bad_progress_m':10.0,'alternative_progress_m':11.0,
    'bad_road_valid':True,'alternative_road_valid':True,
    'bad_kinematics_valid':True,'alternative_kinematics_valid':True}


def test_distinct_family_gate():
    one=check([trial('rules',i) for i in range(5)])
    assert one['n_two_family_screens']==0
    two=check([trial(x,i) for x in ('rules','learned_policy') for i in range(5)])
    assert two['n_two_family_screens']==1
    bad=check([trial('rules',0),trial('rules',0)])
    assert bad['invalid_rows']


def test_replay_requires_actual_start_and_path_shape():
    state=np.array([[0.,0.,1.,0.,0.],[10.,0.,1.,0.,0.],[20.,0.,1.,0.,0.]])
    d=np.ones(3)*4;w=np.ones(3)*2;v=np.ones(3,dtype=bool)
    kwargs={'steps':10,'logged_ego_path':np.column_stack((.2*np.arange(11),np.zeros(11)))}
    r=simulate(state,d,w,v,0,[100,101,102],0,'cautious',**kwargs)
    assert np.allclose(r['paths_xy'][:,0,0],.2*np.arange(11))
    try:
        simulate(state,d,w,v,0,[100,101,102],0,'cautious',steps=10,logged_ego_path=np.zeros((10,2)))
        assert False
    except ValueError:pass


def test_gate_sample(tmp_path):
    row={'scenario_id':'abc','eligible':True,'profile_controls':{},'regimes':{}}
    for k in ('cautious','assertive','heterogeneous'):
        row['regimes'][k]={'actions':{},'strict_matches':[],'matched_burden_proxy_cases':[],'near_pair_proxy_cases':[]}
    row['all_three_regimes_shared_matched_pair_keys']=[]
    j={'summary':{'attempted':1},'rows':[row],'errors':[]}
    p=tmp_path/'m.json';p.write_text(json.dumps(j))
    result=audit(p)
    assert result['strict_h2']['scene_any_regime']==0
