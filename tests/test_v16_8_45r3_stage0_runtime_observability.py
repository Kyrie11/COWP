from __future__ import annotations

import importlib
from pathlib import Path


def _mod():
    return importlib.import_module('cowp.scripts.106_eval_rcrso_support')


def _partial(shard:int, scenarios:list[int], *, examples:int, groups:int, oracle:int, base_full:int, ana_full:int, k2_full:int, k4_full:int):
    def rs(full:int, root:int | None=None, csp:int=0):
        return {'root_num':(full if root is None else root),'full_num':full,'csp_num':csp,'burden_sum':0.0,'burden_count':0,'learned_verified_profile_count':0,'learned_root_nonempty_num':0}
    return {
        'version':'test','scientific_method':'V16.8.45 RCRSO unchanged','split':'val','num_shards':2,'shard_index':shard,
        'k_values':[2,4], 'scenario_hashes':scenarios, 'checkpoint_sha256':'x','sidecar_summary_sha256':'y',
        'assigned_examples':examples,'processed_examples':examples,
        'timing_seconds':{'wall_seconds':1.0+shard,'model_forward_s':0.1},
        'stats':{
            'examples':examples,'eligible_examples':examples,'hypothesis_groups':groups,'oracle_positive_roots':oracle,'verifier_calls':10*examples,
            'baseline':{
                'fixed_bank':{'root_num':0,'full_num':base_full,'csp_num':0},
                'v44_analytic_extension':{'root_num':0,'full_num':ana_full,'csp_num':0},
            },
            'rcrso':{'2':rs(k2_full),'4':rs(k4_full)},
        },
    }


def test_scenario_partition_is_disjoint_and_group_complete():
    m=_mod()
    paths=[
        Path('0000000000000001_000_001_00.npz'),
        Path('0000000000000001_000_001_01.npz'),
        Path('0000000000000001_001_001_00.npz'),
        Path('0000000000000002_000_001_00.npz'),
        Path('0000000000000002_000_001_01.npz'),
    ]
    a=m._assigned_groups(paths,2,0); b=m._assigned_groups(paths,2,1)
    sa={k[0] for k,_ in a}; sb={k[0] for k,_ in b}
    assert sa.isdisjoint(sb)
    assert sa|sb=={1,2}
    got={(k,len(pp)) for bucket in (a,b) for k,pp in bucket}
    assert got=={((1,0),2),((1,1),1),((2,0),2)}


def test_parallel_raw_count_merge_matches_combined_counts_and_k_selection():
    m=_mod()
    p0=_partial(0,[2,4],examples=10,groups=4,oracle=5,base_full=1,ana_full=1,k2_full=2,k4_full=3)
    p1=_partial(1,[1,3],examples=8,groups=4,oracle=3,base_full=1,ana_full=2,k2_full=2,k4_full=3)
    s=m._partial_to_summary([p0,p1],3.0)
    assert s['examples']==18
    assert s['hypothesis_groups']==8
    assert s['baseline']['fixed_bank']['FullHypothesisRootCoverage']==2/8
    assert s['baseline']['v44_analytic_extension']['FullHypothesisRootCoverage']==3/8
    assert s['rcrso_curve'][0]['FullHypothesisRootCoverage']==4/8
    assert s['rcrso_curve'][1]['FullHypothesisRootCoverage']==6/8
    # Plateau=0.75, 95% target=0.7125 => K=4, not K=2.
    assert s['selected_k']==4
    assert s['coverage_lift_over_best_frozen_baseline']==3/8
    assert s['stage0_support_gate']['pass'] is True


def test_merge_summary_keeps_frozen_gate_threshold_and_scientific_method():
    m=_mod()
    p0=_partial(0,[2],examples=2,groups=2,oracle=1,base_full=1,ana_full=1,k2_full=1,k4_full=1)
    p1=_partial(1,[1],examples=2,groups=2,oracle=1,base_full=1,ana_full=1,k2_full=1,k4_full=1)
    # No lift over best baseline => fail even though the run is otherwise valid.
    s=m._partial_to_summary([p0,p1],3.0)
    assert s['minimum_required_lift_pp']==3.0
    assert s['stage0_support_gate']['pass'] is False
    assert s['scientific_method']=='V16.8.45 RCRSO unchanged'
    assert s['contract']['performance_repairs_change_hard_boolean'] is False


def test_static_cache_namespace_changes_when_candidate_local_roadgraph_changes():
    import numpy as np
    m=_mod()
    base={
        'agent_index':np.asarray([1],np.int64),'root_index':np.asarray([0],np.int64),
        'blocker_object_type':np.asarray([1],np.int64),'beta':np.asarray([0.5],np.float32),
        'root_trajectory':np.zeros((4,7),np.float32),'blocker_state_global':np.zeros((11,),np.float32),
        'roadgraph_xy':np.zeros((3,2),np.float32),'roadgraph_heading':np.zeros((3,),np.float32),
        'roadgraph_types':np.zeros((3,),np.int32),'roadgraph_valid':np.ones((3,),bool),
        'environment_current':np.zeros((2,4,7),np.float32),'environment_shifted':np.zeros((2,4,7),np.float32),
        'environment_object_type':np.zeros((2,),np.int64),'environment_agent_index':np.asarray([2,3],np.int64),
    }
    a=m._stage0_static_namespace(base)
    changed={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in base.items()}
    changed['roadgraph_xy'][0,0]=1.0
    b=m._stage0_static_namespace(changed)
    assert a!=b
