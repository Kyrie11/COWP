from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from cowp.core.config import load_config
from cowp.models.recourse_set_operator import build_rcrso_features_np, RCRSOConfig
from cowp.geometry.collision import unsafe_between
from cowp.waymax_eval.policy_wrapper import (
    _trajectory_waymax_kinematic_safe_np,
    _trajectory_waymax_kinematic_safe_literal_np,
    _verified_root_conditioned_recourse_set_profiles_np,
)

def make_case(cfg):
    H=80; dt=float(cfg.get('time',{}).get('dt',0.1))
    state=np.zeros((26,11),np.float32); state[:,10]=1.; state[:,7:10]=[4.5,1.8,1.6]
    state[1,3]=5.; state[1,5]=5.
    root=np.zeros((H,7),np.float32); t=np.arange(1,H+1,dtype=np.float32)*dt
    root[:,0]=5*t; root[:,3]=5; root[:,5:7]=[4.5,1.8]
    env=[]
    for e in range(2,26):
        tr=root.copy(); tr[:,1]=100+e*5
        env.append({'agent_index':e,'object_type':1,'trajectory':tr,'shifted_trajectory':np.vstack([tr[1:],tr[-1:]])})
    road={'xy':np.zeros((0,2),np.float32),'heading':np.zeros(0,np.float32),'valid':np.zeros(0,bool),'types':np.zeros(0,np.int32)}
    rng=np.random.default_rng(20260845); props=rng.uniform(-.3,.3,(32,8)).astype(np.float32)
    egos=[]
    for i in range(10):
        ego=root.copy(); ego[:,1]=50+i
        egos.append((ego,np.vstack([ego[1:],ego[-1:]])))
    return state,root,env,road,props,egos

def bench(fn, n=1000):
    t=time.perf_counter();
    for _ in range(n): fn()
    return time.perf_counter()-t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='V16_8_45R2_SIDECAR_MICROBENCHMARK.json'); args=ap.parse_args()
    cfg=load_config('configs/label_cowp_v16_8.yaml','configs/data.yaml','configs/eval_cowp_v16_8.yaml')
    state,root,env,road,props,egos=make_case(cfg)
    cur=state[1].copy(); tr=root.copy()
    lit=bench(lambda:_trajectory_waymax_kinematic_safe_literal_np(cur,tr,cfg),1000)
    vec=bench(lambda:_trajectory_waymax_kinematic_safe_np(cur,tr,cfg),1000)
    def fresh():
        for ego,shift in egos:
            _verified_root_conditioned_recourse_set_profiles_np(state,1,1,root,100.,ego,shift,env,road,cfg,props,root_ordinal=0,compatibility_cache={})
    def shared():
        cache={}
        for ego,shift in egos:
            _verified_root_conditioned_recourse_set_profiles_np(state,1,1,root,100.,ego,shift,env,road,cfg,props,root_ordinal=0,compatibility_cache=cache)
    fresh_s=bench(fresh,1); shared_s=bench(shared,1)
    # Feature construction used to recompute four directional root/environment
    # unsafe-event masks even though V44 analytic completion had already produced
    # the identical event indices.
    event_steps=[]; shifted_root=np.vstack([root[1:],root[-1:]])
    for actor in env:
        for left,right,right_type in ((root,actor['trajectory'],actor['object_type']),(actor['trajectory'],root,1),(shifted_root,actor['shifted_trajectory'],actor['object_type']),(actor['shifted_trajectory'],shifted_root,1)):
            event_steps.extend(np.flatnonzero(np.asarray(unsafe_between(left,right,cfg,agent_type=int(right_type)).event_mask,bool)).tolist())
    fcfg=RCRSOConfig()
    feature_recompute=bench(lambda:[build_rcrso_features_np(root=root,root_mass=.8,root_source=1,blocker_state=state[1],current_ego_trajectory=e,shifted_ego_trajectory=s,environment=env,cfg=fcfg,verifier_cfg=cfg,blocker_object_type=1) for e,s in egos[:5]],1)
    feature_reuse=bench(lambda:[build_rcrso_features_np(root=root,root_mass=.8,root_source=1,blocker_state=state[1],current_ego_trajectory=e,shifted_ego_trajectory=s,environment=env,cfg=fcfg,verifier_cfg=cfg,blocker_object_type=1,precomputed_environment_event_steps=event_steps) for e,s in egos[:5]],1)
    out={
        'version':'V16.8.45R2-engineering',
        'kinematics_literal_1000_s':lit,'kinematics_vectorized_1000_s':vec,'kinematics_speedup':lit/max(vec,1e-12),
        'verifier_10_candidates_fresh_cache_s':fresh_s,'verifier_10_candidates_shared_scene_cache_s':shared_s,
        'verifier_cache_speedup':fresh_s/max(shared_s,1e-12),
        'feature_5_candidates_recompute_environment_events_s':feature_recompute,
        'feature_5_candidates_reuse_environment_events_s':feature_reuse,
        'feature_event_reuse_speedup':feature_recompute/max(feature_reuse,1e-12),
        'note':'synthetic component benchmark only; not a server end-to-end claim',
    }
    Path(args.output).write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
