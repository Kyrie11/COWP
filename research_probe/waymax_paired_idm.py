"""Experimental Waymax native-TFExample SAME-SCENE paired IDM policy probe.

IMPORTANT: version-sensitive adapter.  Requires server's Waymax/TensorFlow/JAX;
NOT run or validated in the supplied analysis sandbox.  Treat failures as
integration diagnostics, never substitute synthetic simulation labels.

All actors are simulated under Waymax StateDynamics. Non-ego VEHICLES use the
same non-communicating IDMRoutePolicy in both branches; nonvehicles use logged
expert action (a declared limitation). Ego is a separate IDMRoutePolicy with
varied target speed; this is an intervention on a closed-loop POLICY, not the
same as an actual action from COWP/GameFormer/RA-CBF. IDM follows logged spatial
routes, so this is NOT an independent learned-reaction model or human ground truth.
"""
from __future__ import annotations
import argparse,dataclasses,hashlib,json
from pathlib import Path
import numpy as np
from .reactive_stress import _pair_conflicts


def _to_scalar(x):
    return float(np.asarray(x).reshape(-1)[0])


def _extract(state, local_ids):
    """Obtain oriented geometry of tracked agent subset at the *simulated* time."""
    tr=state.current_sim_trajectory
    all_ids=np.asarray(state.object_metadata.ids).reshape(-1)
    targets=[]
    for oid in local_ids:
        candidates=np.flatnonzero(all_ids==int(oid))
        if len(candidates)!=1:
            raise RuntimeError(f'Agent id {oid} missing/nonunique in Waymax state; raise max_num_objects')
        targets.append(int(candidates[0]))
    def take(field):return np.asarray(field).reshape(-1)[targets].astype(float)
    p=np.stack([take(tr.x),take(tr.y)],axis=1)
    return p,take(tr.yaw),take(tr.length),take(tr.width),take(tr.valid)>.5,take(tr.speed),targets


def _read_target_npz(cohort,entry):
    with np.load(cohort/entry['file'],allow_pickle=False) as f:
        return (np.asarray(f['agent_ids']).astype(np.int64).tolist(),
                np.array([float(f['current_x'][0,0]),float(f['current_y'][0,0])]))


def _load_waymax_state(shard, npz_ids, sdc_xy, *, max_num_objects=128, record_index=None):
    # Imports are intentionally lazy so analysis/audit CLI does not require Waymax.
    from waymax import config, dataloader
    cfg=dataclasses.replace(config.WOD_1_1_0_VALIDATION,
        path=str(shard), repeat=1, shuffle_seed=None, num_shards=1,
        max_num_objects=max_num_objects, include_sdc_paths=False,
        deterministic=True)
    candidates=[]
    for idx,state in enumerate(dataloader.simulator_state_generator(cfg)):
        if record_index is not None and idx!=record_index:continue
        ids=np.asarray(state.object_metadata.ids).reshape(-1)
        is_sdc=np.asarray(state.object_metadata.is_sdc).reshape(-1).astype(bool)
        if is_sdc.sum()!=1:continue
        if int(ids[np.flatnonzero(is_sdc)[0]])!=int(npz_ids[0]):continue
        pos,_,_,_,_,_,_= _extract(state,[npz_ids[0]])
        if float(np.linalg.norm(pos[0]-sdc_xy))>0.25:continue
        candidates.append(state)
        if len(candidates)>1:raise RuntimeError('Ambiguous scenario match by SDC ID+initial xy')
    if len(candidates)!=1:
        raise RuntimeError(f'No exact Waymax scene match in {shard.name}; check DatasetConfig/version/record_index')
    return candidates[0]


def _rollout(scenario, targets, *, ego_desired_vel, regime, steps):
    from waymax import env,config,dynamics,agents
    # StateDynamics exactly matches IDMRoutePolicy actor action semantics.
    dynamics_model=dynamics.StateDynamics()
    environment=env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(config.EnvironmentConfig(),
            max_num_objects=len(np.asarray(scenario.object_metadata.ids).reshape(-1)),
            controlled_object=config.ObjectType.VALID,
            compute_reward=False,allow_new_objects_after_warmup=False))
    response_cfg={
        'cautious': dict(desired_vel=15.,safe_time_headway=1.8,max_accel=1.1,max_decel=4.),
        'assertive':dict(desired_vel=24.,safe_time_headway=0.8,max_accel=1.8,max_decel=5.),
    }[regime]
    # Shapes/masks derived from state, never expected number of objects hardcoded.
    def ego_mask(s):return s.object_metadata.is_sdc
    def vehicle_mask(s):return (s.object_metadata.object_types==1) & (~s.object_metadata.is_sdc)
    def nonvehicle_mask(s):return (s.object_metadata.object_types!=1) & (~s.object_metadata.is_sdc)
    actors=[
        agents.IDMRoutePolicy(is_controlled_func=ego_mask,desired_vel=float(ego_desired_vel),
                              safe_time_headway=1.2),
        agents.IDMRoutePolicy(is_controlled_func=vehicle_mask,**response_cfg),
        agents.create_expert_actor(dynamics_model=dynamics_model,is_controlled_func=nonvehicle_mask),
    ]
    st=environment.reset(scenario)
    initial=_extract(st,targets)
    p,y,L,W,valid,speed,_=initial
    if _pair_conflicts(p,y,L,W,valid):
        return {'eligible':False,'reason':'initial_oriented_overlap'}
    xy0=p[0].copy();pairs=set();ego_pairs=set();initial_speed=speed.copy()
    min_accel=np.full(len(targets),np.inf)
    collisions_early=False;sim_steps=0
    trace=[]
    for k in range(steps):
        if bool(np.asarray(st.is_done).reshape(-1)[0]):break
        outputs=[actor.select_action({},st,None,None) for actor in actors]
        action=agents.merge_actions(outputs)
        st=environment.step(st,action)
        p,y,L,W,valid,new_speed,_=_extract(st,targets)
        min_accel=np.minimum(min_accel,(new_speed-speed)/0.1)
        speed=new_speed
        conflicting=_pair_conflicts(p,y,L,W,valid)
        for a,b in conflicting:
            if 0 in (a,b):
                ego_pairs.add(tuple(sorted((a,b))))
                if k<5:collisions_early=True
            else:pairs.add(tuple(sorted((a,b))))
        trace.append({'step':k+1,'ego_xy':p[0].tolist(),'ego_speed_mps':float(speed[0]),
                      'non_ego_pairs':[list(x) for x in conflicting if 0 not in x],
                      'ego_pairs':[list(x) for x in conflicting if 0 in x]})
        sim_steps+=1
    return {'eligible':True,'steps_executed':sim_steps,
        'non_ego_collision_pairs':[list(x) for x in sorted(pairs)],
        'ego_collision_pairs':[list(x) for x in sorted(ego_pairs)],
        'any_collision':bool(pairs or ego_pairs),'ego_collision':bool(ego_pairs),
        'non_ego_collision':bool(pairs),
        'near_term_ego_clear':not collisions_early,
        'ego_displacement_m':float(np.linalg.norm(p[0]-xy0)),
        'initial_ego_speed_mps':float(initial_speed[0]),
        'non_ego_min_accel_by_agent':[float(x) if np.isfinite(x) else None for x in min_accel[1:]],
        'trajectory_trace':trace}


def _shard_path(root,entry):
    found=list(root.rglob(entry['source_shard']))
    if len(found)!=1:raise FileNotFoundError(f'Need unique shard {entry["source_shard"]} beneath {root}: found {len(found)}')
    return found[0]


def run(cohort,raw_root,output,*,max_scenes=12,steps=40,velocities=(6.,12.,18.),
        progress_tolerance=3.,regimes=('cautious','assertive')):
    cohort=Path(cohort);raw_root=Path(raw_root)
    manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    if manifest.get('source_format')!='tfexample':
        raise ValueError('This adapter requires original WOMD TFExample, not Scenario proto')
    entries=sorted(manifest['entries'],key=lambda e:hashlib.sha256(('waymax-pilot-v1|'+e['scenario_id']).encode()).hexdigest())
    rows=[];errors=[]
    for e in entries[:max_scenes or None]:
        sid=e['scenario_id']
        try:
            target,xy=_read_target_npz(cohort,e)
            scenario=_load_waymax_state(_shard_path(raw_root,e),target,xy,record_index=e.get('source_record_index'))
            d={'scenario_id':sid,'status':'WAYMAX_IDM_POLICY_PILOT_NOT_HUMAN_CAUSALITY','regimes':{}}
            for regime in regimes:
                a=[]
                for vel in velocities:
                    x=_rollout(scenario,target,ego_desired_vel=vel,regime=regime,steps=steps)
                    x['ego_desired_vel_mps']=vel;a.append(x)
                cases=[];noninferior=[]
                for x in a:
                    if not x.get('eligible') or not x['non_ego_collision'] or not x['near_term_ego_clear']:continue
                    for y in a:
                        if y is x or not y.get('eligible') or y['any_collision']:continue
                        if y['ego_displacement_m']+progress_tolerance<x['ego_displacement_m']:continue
                        contrast=[x['ego_desired_vel_mps'],y['ego_desired_vel_mps']]
                        noninferior.append(contrast)
                        if abs(y['ego_displacement_m']-x['ego_displacement_m'])<=progress_tolerance:
                            cases.append(contrast)
                d['regimes'][regime]={'actions':a,'paired_candidate_actions':cases,
                                       'progress_noninferior_candidate_actions':noninferior}
            common=set.intersection(*(set(map(tuple,d['regimes'][r]['paired_candidate_actions'])) for r in regimes))
            d['all_regimes_matched_pairs']=[list(z) for z in sorted(common)]
            rows.append(d)
        except Exception as ex:
            errors.append({'scenario_id':sid,'error':type(ex).__name__+': '+str(ex)})
    result={'summary':{'attempted':len(entries[:max_scenes or None]),'complete':len(rows),
              'errors':len(errors),'any_all_regime_matched':sum(bool(r['all_regimes_matched_pairs']) for r in rows),
              'scientific_claim_allowed':False,
              'limitations':['IDMRoutePolicy follows logged spatial routes','Not independently validated learned-human reaction',
                'Non-vehicle actors are log experts','Descriptive Euclidean ego displacement only',
                'Only same-shard pilot; no reproduced GameFormer or RA-CBF']},'rows':rows,'errors':errors}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result['summary']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=12)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--ego-speeds',type=float,nargs='+',default=[6.,12.,18.])
    p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args()
    print(json.dumps(run(a.cohort,a.raw_root,a.output,max_scenes=a.max_scenes,steps=a.steps,
        velocities=tuple(a.ego_speeds),progress_tolerance=a.progress_tolerance),indent=2))
if __name__=='__main__':main()
