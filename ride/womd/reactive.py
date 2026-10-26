"""Independent-reactive *hypothesis generator* on raw WOMD observed states.

Each agent sees current geometry only; no shared plans, messages, or joint command.
Constant-heading longitudinal IDM-inspired accelerations and SAT rectangles are
screening approximations, NOT Waymax and NOT a causal estimate from WOMD logs.
Pre-registered ego controls are identical in every simulated policy regime.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from .footprints import oriented_clear

DT=.2
STEPS=20
CONTROLS=(-3.,-1.5,0.,1.5)
REGIMES=('cautious','assertive','heterogeneous','ttc_gap_rule')


def _load_scene(path):
    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
    n=len(d['agent_ids']); ego=int(d['ego_index'])
    if ego<0 or ego>=n:raise ValueError('bad ego')
    state=np.column_stack([d['current_x'][:,0],d['current_y'][:,0],
          d['current_velocity_x'][:,0],d['current_velocity_y'][:,0],d['current_bbox_yaw'][:,0]])
    length=d['current_length'][:,0];width=d['current_width'][:,0]
    valid=d['current_valid'][:,0]>.5
    if not np.isfinite(state[valid]).all():raise ValueError('non-finite state')
    return d,state,length,width,valid,ego


def _collide(p1,p2,l1,w1,h1,l2,w2,h2,margin=.05):
    return not oriented_clear(np.asarray([p1]),np.asarray([p2]),(l1,w1,h1,False),(l2,w2,h2,False),margin)


def _pair_conflicts(pos,yaw,length,width,valid):
    res=[]
    for i in range(len(pos)):
        if not valid[i]:continue
        for j in range(i+1,len(pos)):
            if not valid[j]:continue
            if np.linalg.norm(pos[i]-pos[j])>(length[i]+length[j])+3:continue
            if _collide(pos[i],pos[j],length[i],width[i],yaw[i],length[j],width[j],yaw[j]):res.append((i,j))
    return res


def _idm_accel(i,pos,vel,yaw,length,width,valid,params,excluded_leader=None):
    speed=max(0.,float(vel[i])); h=np.array([math.cos(yaw[i]),math.sin(yaw[i])]); side=np.array([-h[1],h[0]])
    gap=np.inf; leader=None
    for j in range(len(pos)):
        if j==i or j==excluded_leader or not valid[j]:continue
        diff=pos[j]-pos[i];along=float(np.dot(diff,h))
        lateral=abs(float(np.dot(diff,side)))
        if along<=0 or lateral>0.5*(width[i]+width[j])+0.85:continue
        edge_gap=along-0.5*(length[i]+length[j])
        if edge_gap<gap:gap=edge_gap;leader=j
    a,b,T,v0=params
    free=a*(1-(speed/max(v0,1.))**4)
    if leader is None:return float(np.clip(free,-6.,2.5))
    closing=speed-float(vel[leader])
    desired=2+max(0.,speed*T+speed*closing/(2*math.sqrt(a*b)))
    return float(np.clip(free-a*(desired/max(gap,.2))**2,-6.,2.5))


def _ttc_gap_rule_accel(i,pos,vel,yaw,length,width,valid,excluded_leader=None):
    """An independent geometric gap/TTC braking RULE, not a learned human driver.

    This has no IDM free-speed term or desired headway. It retains the same
    fixed-heading simulator, collision checker, and observation limitations.
    """
    h=np.array([math.cos(yaw[i]), math.sin(yaw[i])]);side=np.array([-h[1],h[0]])
    best=0.0
    for j in range(len(pos)):
        if j==i or j==excluded_leader or not valid[j]:continue
        d=pos[j]-pos[i];along=float(d@h)
        if along<=0 or abs(float(d@side))>0.5*(width[i]+width[j])+0.85:continue
        gap=along-0.5*(length[i]+length[j])
        leader_projected_speed=float(vel[j]*math.cos(yaw[j]-yaw[i]))
        closing=max(0.,float(vel[i]-leader_projected_speed))
        ttc=gap/max(closing,1e-6)
        if gap<=2.0:
            candidate=-6.0
        elif closing>0.25 and ttc<3.5:
            candidate=-(closing*closing)/(2*max(gap-2.0,0.5))
        else: candidate=0.0
        best=min(best,candidate)
    return float(np.clip(best,-6.,0.))


def _params(regime,agent_key):
    if regime=='cautious':return (1.1,2.2,1.8,15.)
    if regime=='assertive':return (1.8,3.2,.8,24.)
    if regime=='heterogeneous':
        h=int(hashlib.sha256(str(agent_key).encode()).hexdigest()[:8],16)
        return ((1.1,2.2,1.8,15.),(1.8,3.2,.8,24.),(1.4,2.5,1.2,19.))[h%3]
    raise ValueError(regime)


def simulate(state,length,width,valid,ego,ids,ego_accel,regime,steps=STEPS,*,ghost_ego_to_others=False,responder_delay_steps=0,logged_ego_path=None):
    ego_controls=np.asarray(ego_accel,dtype=float)
    if ego_controls.ndim==0:
        ego_controls=np.full(steps,float(ego_controls))
    elif ego_controls.shape!=(steps,):
        raise ValueError('ego_accel must be scalar or a length-steps sequence')
    pos=state[:,:2].astype(float).copy()
    yaw=state[:,4].copy()
    # velocity norm is used under vehicle heading; large mismatch => exclude before science claims
    vel=np.linalg.norm(state[:,2:4],axis=1).astype(float)
    n=len(pos);initial=_pair_conflicts(pos,yaw,length,width,valid)
    if logged_ego_path is not None:
        logged_ego_path=np.asarray(logged_ego_path,dtype=float)
        if logged_ego_path.shape!=(steps+1,2) or not np.isfinite(logged_ego_path).all():
            raise ValueError('logged_ego_path must be finite [(steps+1),2]')
        if np.linalg.norm(logged_ego_path[0]-pos[ego])>0.2:
            raise ValueError('logged ego path initial pose mismatch')
    if not isinstance(responder_delay_steps,int) or responder_delay_steps<0:
        raise ValueError('responder_delay_steps must be non-negative integer')
    obs_history=[]
    paths=np.zeros((steps+1,n,2),float);paths[0]=pos
    acc=np.zeros((steps,n));collision_steps=[];other_other=[];ego_other=[];ego_steps=[];non_ego_steps=[]
    for t in range(steps):
        obs_history.append((pos.copy(),vel.copy()))
        obs_pos,obs_vel=obs_history[max(0,t-responder_delay_steps)]
        for i in range(n):
            if not valid[i]:continue
            if i==ego:
                acc[t,i]=float(ego_controls[t])
            elif regime=='ttc_gap_rule':
                acc[t,i]=_ttc_gap_rule_accel(i,obs_pos,obs_vel,yaw,length,width,valid,
                             excluded_leader=ego if ghost_ego_to_others else None)
            else:
                acc[t,i]=_idm_accel(i,obs_pos,obs_vel,yaw,length,width,valid,_params(regime,ids[i]),
                             excluded_leader=ego if ghost_ego_to_others else None)
        next_v=np.clip(vel+acc[t]*DT,0.,37.)
        unit=np.column_stack([np.cos(yaw),np.sin(yaw)])
        pos=pos+(vel+next_v)[:,None]*unit*DT*.5
        vel=next_v
        if logged_ego_path is not None:
            # Observational response-model fit ONLY: replay the logged ego;
            # this is not a counterfactual ego planner action.
            displacement=logged_ego_path[t+1]-logged_ego_path[t]
            pos[ego]=logged_ego_path[t+1]
            vel[ego]=float(np.linalg.norm(displacement)/DT)
            if vel[ego]>0.2:
                yaw[ego]=float(math.atan2(displacement[1],displacement[0]))
        paths[t+1]=pos
        for a,b in _pair_conflicts(pos,yaw,length,width,valid):
            collision_steps.append(t+1)
            
            if ego in (a,b):
                ego_other.append([int(a),int(b)]); ego_steps.append(t+1)
            else:
                other_other.append([int(a),int(b)]); non_ego_steps.append(t+1)
    return {'initial_overlap_pairs':[list(x) for x in initial],
            'collision':bool(collision_steps),'first_collision_step':min(collision_steps) if collision_steps else None,
            'ego_collision':bool(ego_other),'non_ego_collision':bool(other_other),
            'non_ego_collision_pairs':sorted(set(tuple(p) for p in other_other)),
            'ego_collision_pairs':sorted(set(tuple(p) for p in ego_other)),
            'non_ego_collision_first_step':min(non_ego_steps) if non_ego_steps else None,
            'non_ego_min_accel_by_agent':{str(i):float(np.min(acc[:,i])) for i in range(n)
                                             if i != ego and valid[i]},
            'ego_progress_m':float(np.linalg.norm(paths[-1,ego]-paths[0,ego])),
            'near_term_ego_clear':not any(t<=5 for t in ego_steps),
            'max_braking_mps2':float(np.min(acc[:,np.arange(n)!=ego])) if n>1 else None,
            'paths_xy':paths,'accelerations':acc}


def observed_ade(data,pred,agents):
    # compare independent simulated responses to logged WOMD future using aligned 10Hz->5Hz
    k=min(5,STEPS)
    errs=[]
    for t in range(1,k+1):
        time_idx=2*t-1 # 0.2s ahead is future index 1
        if time_idx>=data['future_x'].shape[1]:break
        for j in agents:
            if data['future_valid'][j,time_idx]>.5:
                logged=np.array([data['future_x'][j,time_idx],data['future_y'][j,time_idx]])
                errs.append(float(np.linalg.norm(pred[t,j]-logged)))
    return float(np.mean(errs)) if errs else None


def analyze_scene(path,progress_tolerance=3.,ade_max=4.):
    data,st,L,W,valid,ego=_load_scene(path)
    ids=data['agent_ids'].tolist()
    initial=_pair_conflicts(st[:,:2],st[:,4],L,W,valid)
    if initial: return {'scenario_id':path.stem,'eligible':False,'reason':'initial_oriented_overlap','initial_pairs':initial}
    if valid.sum()<3:return {'scenario_id':path.stem,'eligible':False,'reason':'fewer_than_three'}
    # avoid strongly nonholonomic heading-velocity contradictions in simple longitudinal simulator
    velocities=st[:,2:4];sp=np.linalg.norm(velocities,axis=1)
    heading=np.column_stack([np.cos(st[:,4]),np.sin(st[:,4])])
    mismatch=(sp>3)&(np.sum(velocities*heading,axis=1)/np.maximum(sp,1e-5)<.65)&valid
    if mismatch.any():return {'scenario_id':path.stem,'eligible':False,'reason':'velocity_heading_mismatch'}
    regimes={}
    for regime in REGIMES:
        result=[]
        for a in CONTROLS:
            raw=simulate(st,L,W,valid,ego,ids,a,regime)
            non_ego=[i for i in range(len(ids)) if i!=ego and valid[i]]
            ade=observed_ade(data,raw['paths_xy'],non_ego)
            x={k:v for k,v in raw.items() if k not in ('paths_xy','accelerations')}
            x.update({'ego_acceleration_mps2':a,'response_ade_1s_m':ade})
            result.append(x)
        # all controls share regime but their measured predictive fit can differ across ego interventions.
        candidate=[]
        for x in result:
            if not x['near_term_ego_clear'] or not x['non_ego_collision']:continue
            alternatives=[y for y in result if y['ego_acceleration_mps2']!=x['ego_acceleration_mps2']
                          and not y['collision'] and y['ego_progress_m']+progress_tolerance>=x['ego_progress_m']]
            if alternatives:candidate.append({'action':x['ego_acceleration_mps2'],
                'alternative':max(alternatives,key=lambda v:v['ego_progress_m'])['ego_acceleration_mps2']})
        regimes[regime]={'actions':result,'preventable_proxy_candidates':candidate,
                         'any_reactive_risk':any(x['non_ego_collision'] for x in result),
                         'behavior_fit_reasonable':all(x['response_ade_1s_m'] is not None and x['response_ade_1s_m']<=ade_max for x in result)}
    # Screen across three IDM parametrizations AND a structurally distinct TTC
    # rule. This is stronger sensitivity testing, not a calibrated human model.
    stable_actions=set.intersection(*(set(x['action'] for x in regimes[r]['preventable_proxy_candidates'])
                                       for r in REGIMES))
    fits=all(x['behavior_fit_reasonable'] for x in regimes.values())
    return {'scenario_id':path.stem,'eligible':True,'initial_oriented_overlap':False,
       'stable_intervention_proxy_actions':sorted(stable_actions),'all_regimes_fit_observations':fits,
       'regimes':regimes,'scientific_status':'SIMULATOR_HYPOTHESIS_ONLY'}


def run(cohort,output,max_scenes,progress_tolerance,ade_max,selection_order="hash"):
    if selection_order not in ("hash", "manifest"):raise ValueError(selection_order)
    cohort=Path(cohort);data=json.loads((cohort/'cohort_manifest.json').read_text())
    rows=[];err=[]
    entries=data['entries']
    if selection_order=='hash':
        entries=sorted(entries,key=lambda e:hashlib.sha256(('native-eval-v1|'+e['scenario_id']).encode()).hexdigest())
    selected=entries[:max_scenes or None]
    for e in selected:
        path=cohort/e['file']
        try: rows.append(analyze_scene(path,progress_tolerance,ade_max))
        except Exception as ex:err.append({'id':e['scenario_id'],'error':str(ex)})
    eligible=[x for x in rows if x['eligible']]
    raw_proxies=[x for x in eligible if x['stable_intervention_proxy_actions']]
    accepted=[x for x in raw_proxies if x['all_regimes_fit_observations']]
    summary={'scientific_claim_allowed':False,'observed_input':'WOMD historical positions only; interventions generated by simplified simulator',
             'n_checked':len(rows),'n_eligible':len(eligible),'excluded':len(rows)-len(eligible),
             'selection_order':selection_order,'n_selected_entries':len(selected),
             'selection_protocol':'sha256(native-eval-v1|scenario_id)' if selection_order=='hash' else 'manifest_prefix',
             'stable_proxy_scenes':len(raw_proxies),'calibrated_stable_proxy_scenes':len(accepted),
             'selected_intervention_ids':[x['scenario_id'] for x in raw_proxies],
             'calibrated_ids':[x['scenario_id'] for x in accepted],
             'limits':['Not Waymax; static headings and longitudinal IDM-like policy only','Observed-future ADE is descriptive, not sufficient calibration','No ego actual projected controller action','No lane/map/traffic-light rules','No human ground truth for interventions','IDM family and TTC-gap braking rule share fixed-heading physics/observations, not independent calibrated world models']}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'summary':summary,'rows':rows,'errors':err},indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=80)
    p.add_argument('--progress-tolerance',type=float,default=3.)
    p.add_argument('--ade-max',type=float,default=4.)
    p.add_argument('--selection-order',choices=['hash','manifest'],default='hash')
    a=p.parse_args();print(json.dumps(run(a.cohort,a.output,a.max_scenes,a.progress_tolerance,a.ade_max,a.selection_order),indent=2))
if __name__=='__main__':main()
