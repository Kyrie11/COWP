"""Falsification-first longitudinal recourse probe for WOMD-derived COWP tensor caches.

This tool is NOT Waymax, not a traffic-rule verifier, and not a deployable planner.
It uses observed current states, independent constant-acceleration responders,
all-agent collision checks and one-step successor recomputation. No future GT,
COWP witness score, or natural-root pseudo-label is used to decide feasibility.
The candidate bank comes from the frozen COWP dataset, so sampling limitations
are explicitly recorded and this is development evidence only.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np
from .footprints import circle_clear, oriented_clear

DT = 0.2
HORIZON = 16             # 3.2 seconds; diagnostic only
SHIFT = 4               # 0.8 seconds; not a guarantee
ACCELS = (-4., -2., 0., 1.5)
EGO_BACKUP_ACCELS = (-4., -2., 0., 1.5)
FOOTPRINT_MODE = "circle"


def _get(d, key, default=None):
    for k in (key, 'womd/' + key):
        if k in d:
            return np.asarray(d[k])
    return default


def _load(path):
    with np.load(path, allow_pickle=False) as n:
        return {k.replace('__', '/'): n[k] for k in n.files if
                k.startswith(('womd__state__','state__','cowp__candidates__','scenario__')) or
                k.startswith(('state/','womd/state/','cowp/candidates/','scenario/'))}


def _scalar_agent(a, n, default):
    if a is None: return np.full(n, default, dtype=float)
    a=np.asarray(a,dtype=float).reshape(-1)
    if a.size < n: raise ValueError(f'Expected {n} agents, found {a.size}')
    return a[:n]


def _states(d):
    x=_get(d,'state/current/x');y=_get(d,'state/current/y')
    if x is None or y is None: raise ValueError('missing current state x/y')
    x=np.asarray(x).reshape(-1);y=np.asarray(y).reshape(-1);n=len(x)
    vx=_scalar_agent(_get(d,'state/current/velocity_x'),n,0.)
    vy=_scalar_agent(_get(d,'state/current/velocity_y'),n,0.)
    yaw=_scalar_agent(_get(d,'state/current/bbox_yaw'),n,0.)
    length=_scalar_agent(_get(d,'state/current/length'),n,4.5)
    width=_scalar_agent(_get(d,'state/current/width'),n,1.9)
    valid=_scalar_agent(_get(d,'state/current/valid'),n,1.)>.5
    is_sdc=_get(d,'state/is_sdc')
    if is_sdc is None: raise ValueError('state/is_sdc required: no guessing ego identity')
    is_sdc=np.asarray(is_sdc).reshape(-1)
    if is_sdc.size<n or (is_sdc[:n]>.5).sum()!=1: raise ValueError('invalid/missing unique is_sdc')
    ego=int(np.flatnonzero(is_sdc[:n]>.5)[0])
    if not valid[ego]: raise ValueError('ego current invalid')
    # Simple circular overlap approximation; NOT oriented-box safety.
    radius=np.maximum(np.hypot(length,width)*.5,1.0)
    return np.column_stack([x,y,vx,vy,yaw,radius,valid.astype(float),length,width]),ego


def rollout(initial, accel, steps=HORIZON, dt=DT):
    """Simple longitudinal dynamics, constant yaw (units SI). Includes t=0."""
    x,y,vx,vy,yaw=map(float,initial[:5]); speed=float(np.hypot(vx,vy))
    direction=np.array([np.cos(yaw),np.sin(yaw)])
    xy=np.empty((steps+1,2),dtype=float); xy[0]=[x,y]
    for i in range(steps):
        v1=np.clip(speed+accel*dt,0.,38.)
        xy[i+1]=xy[i]+direction*(speed+v1)*.5*dt
        speed=v1
    return xy


def path_ok(a,b,ra,rb,margin=0.0):
    if len(a)!=len(b): raise ValueError('path horizon mismatch')
    if isinstance(ra,tuple) and isinstance(rb,tuple):
        return oriented_clear(a,b,ra,rb,margin)
    return circle_clear(a,b,ra,rb,margin)


def ego_paths(d,states,ego,max_candidates):
    c=_get(d,'cowp/candidates/trajectory'); valid=_get(d,'cowp/candidates/valid')
    if c is None or valid is None: raise ValueError('missing candidate trajectory/valid')
    c=np.asarray(c); valid=np.asarray(valid).reshape(-1).astype(bool)
    if c.ndim!=3 or c.shape[-1]<2: raise ValueError(f'bad candidates shape {c.shape}')
    paths=[]
    for k in np.flatnonzero(valid)[:max_candidates]:
        xy=np.asarray(c[k,:,:2],float)
        if not np.isfinite(xy).all():continue
        if np.linalg.norm(xy[0]-states[ego,:2])>7.:continue
        # Resample the frozen candidate temporal grid (WOMD 10 Hz) to diagnostic 5 Hz.
        origin=states[ego,:2][None,:]
        sample=np.concatenate([origin,xy],axis=0)
        times=np.arange(len(sample))*.1
        target=np.arange(HORIZON+1)*DT
        if times[-1]+1e-9<target[-1]:continue
        res=np.column_stack([np.interp(target,times,sample[:,j]) for j in range(2)])
        # Reject discretely implausible 5 Hz candidate controls (guard against unit/schema errors).
        if np.max(np.linalg.norm(np.diff(res,axis=0),axis=-1))/DT>42.:continue
        paths.append((int(k),res))
    return paths


def options_for_agent(state, ego_xy,ego_radius,agent_radius,steps=HORIZON):
    opts=[]
    for a in ACCELS:
        xy=rollout(state,a,steps)
        if path_ok(xy,ego_xy,agent_radius,ego_radius,0.05):
            opts.append((float(a),xy))
    return opts


def joint_response(options, radii, cap=50000):
    """Exact backtracking for explicitly enumerated options, NOT full continuous support."""
    if any(not x for x in options):return None,0
    visited=0
    def dfs(i,chosen):
        nonlocal visited
        if i==len(options):return tuple(chosen)
        for ai,trajectory in options[i]:
            visited+=1
            if visited>cap:raise RuntimeError("joint CSP search budget exceeded; UNKNOWN, not infeasible")
            if all(path_ok(trajectory,other,radii[i],radii[j],0.05)
                   for j,(_,other) in enumerate(chosen)):
                sol=dfs(i+1,chosen+[(ai,trajectory)])
                if sol is not None:return sol
        return None
    return dfs(0,[]),visited


def independent_response(options):
    """Agent-local minimum-|acceleration| response, no inter-agent message exchange."""
    return tuple(min(o,key=lambda x:(abs(x[0]),x[0])) for o in options)


def compatible_responses(chosen,radii):
    return all(path_ok(chosen[i][1],chosen[j][1],radii[i],radii[j],0.05)
               for i in range(len(chosen)) for j in range(i+1,len(chosen)))


def next_state(st,xy,t,dt=DT):
    state=st.copy(); state[0:2]=xy[t]
    dr=(xy[t]-xy[max(t-1,0)])/dt if t else st[2:4]
    state[2:4]=dr
    if np.linalg.norm(dr)>0.25:state[4]=np.arctan2(dr[1],dr[0])
    return state


def successor_has_joint_option(ego_state, agent_states, radii, max_backup=4):
    """One-step replan with longitudinal backups: conservative *proxy* for options."""
    for a in EGO_BACKUP_ACCELS[:max_backup]:
        ego_xy=rollout(ego_state,a)
        opts=[options_for_agent(s,ego_xy,radii[0],radii[i+1]) for i,s in enumerate(agent_states)]
        sol,_=joint_response(opts,radii[1:])
        if sol is not None:return True
    return False


def _evaluate_candidate(ego_path,states,ego,others):
    agent_states=[states[i] for i in others]
    if FOOTPRINT_MODE == 'oriented-box':
        radii=[(float(states[ego,7]),float(states[ego,8]),float(states[ego,4]),True)]+[
            (float(states[i,7]),float(states[i,8]),float(states[i,4]),False) for i in others]
    else:
        radii=[float(states[ego,5])]+[float(states[i,5]) for i in others]
    options=[options_for_agent(s,ego_path,radii[0],radii[i+1]) for i,s in enumerate(agent_states)]
    pairwise=all(bool(o) for o in options)
    joint,checked=joint_response(options,radii[1:]) if pairwise else (None,0)
    # Replan after the *same deterministic local response*; avoid selecting a joint oracle.
    local=independent_response(options) if pairwise else tuple()
    local_joint=bool(local) and compatible_responses(local,radii[1:])
    successor=None
    if joint is not None and local_joint:
        es=next_state(states[ego],ego_path,SHIFT)
        ns=[next_state(s,local[i][1],SHIFT) for i,s in enumerate(agent_states)]
        successor=successor_has_joint_option(es,ns,radii)
    hard_pairs=[]
    if pairwise and joint is None:
        for i in range(len(options)):
            for j in range(i+1,len(options)):
                if not any(path_ok(a[1],b[1],radii[i+1],radii[j+1],0.05)
                           for a in options[i] for b in options[j]):
                    hard_pairs.append([int(others[i]),int(others[j])])
    return {'pairwise':pairwise,'joint':joint is not None,'local_compatible':local_joint,
            'successor_viable':successor,'search_nodes':checked,
            'individual_option_counts':[len(o) for o in options],
            'unconditionally_incompatible_response_pairs':hard_pairs}


def examine_scene(data,max_candidates=10,max_neighbors=3,radius_m=32.):
    st,ego=_states(data)
    candidates=ego_paths(data,st,ego,max_candidates)
    nearby=[i for i in range(len(st)) if i!=ego and st[i,6]>.5 and
            np.all(np.isfinite(st[i,:6])) and np.linalg.norm(st[i,:2]-st[ego,:2])<radius_m]
    nearby=sorted(nearby,key=lambda i:(np.linalg.norm(st[i,:2]-st[ego,:2]),i))[:max_neighbors]
    if len(nearby)<2: return {'eligible':False,'reason':'less_than_two_neighbors','n_candidates':len(candidates)}
    initial_circle=[];initial_box=[]
    for i in range(len(nearby)):
        for j in range(i+1,len(nearby)):
            aa=nearby[i];bb=nearby[j]
            pa=st[aa,:2][None,:];pb=st[bb,:2][None,:]
            if not circle_clear(pa,pb,float(st[aa,5]),float(st[bb,5]),0.05):
                initial_circle.append([int(aa),int(bb)])
            fa=(float(st[aa,7]),float(st[aa,8]),float(st[aa,4]),False)
            fb=(float(st[bb,7]),float(st[bb,8]),float(st[bb,4]),False)
            if not oriented_clear(pa,pb,fa,fb,0.05):
                initial_box.append([int(aa),int(bb)])
    results=[]
    for k,xy in candidates:
        r=_evaluate_candidate(xy,st,ego,nearby)
        r.update({'candidate_index':k,'ego_displacement_m':float(np.linalg.norm(xy[-1]-xy[0]))})
        results.append(r)
    if not results:return {'eligible':False,'reason':'no_usable_candidate','n_candidates':0}
    pairwise_discordance=[r for r in results if r['pairwise'] and not r['joint']]
    temporal=[r for r in results if r['joint'] and r['local_compatible'] and r['successor_viable'] is False]
    # Alternative for collapse must have verified sampled successor, and similar progress.
    avoidable=[]
    for bad in temporal:
        alternatives=[good for good in results if good['successor_viable'] is True and
                      good['ego_displacement_m']+3.0>=bad['ego_displacement_m']]
        if alternatives:avoidable.append({'candidate_index':bad['candidate_index'],
            'counterfactual_index':max(alternatives,key=lambda a:a['ego_displacement_m'])['candidate_index']})
    # Three frozen selection comparators using the same ego candidate bank and response primitives.
    # These are diagnostic baselines, not implementations of published RA-CBF or contingency MPC.
    def best(predicate):
        sub=[r for r in results if predicate(r)]
        return max(sub,key=lambda r:(r['ego_displacement_m'],-r['candidate_index'])) if sub else None
    selections={
        'pairwise_utility':best(lambda r:r['pairwise']),
        'joint_existence_utility':best(lambda r:r['joint']),
        'discrete_successor_backup':best(lambda r:r['joint'] and r['local_compatible'] and r['successor_viable'] is True),
    }
    summary_select={name:None if r is None else {
        'candidate_index':r['candidate_index'],'progress_proxy_m':r['ego_displacement_m'],
        'joint':r['joint'],'local_compatible':r['local_compatible'],
        'successor_viable':r['successor_viable']} for name,r in selections.items()}
    all_discordant=bool(results) and all(r['pairwise'] and not r['joint'] for r in results)
    return {'eligible':True,'n_candidates':len(results),'n_neighbors':len(nearby),
            'initial_overlap_circle_pairs':initial_circle,
            'initial_overlap_oriented_box_pairs':initial_box,
            'all_candidates_joint_incompatible':all_discordant,
            'selected_agent_indices':[int(i) for i in nearby],
            'baseline_selection':summary_select,
            'pairwise_joint_discordance':len(pairwise_discordance),
            'joint_but_local_conflict':sum(r['joint'] and not r['local_compatible'] for r in results),
            'temporal_collapse_proxy':len(temporal),'avoidable_collapse_proxy':len(avoidable),
            'avoidable_pairs':avoidable,'candidates':results}


def _id(d,path):
    v=_get(d,'scenario/id')
    if v is None:return path.stem
    a=np.asarray(v).reshape(-1)
    if a.size==0:return path.stem
    item=a[0];return item.decode() if isinstance(item,bytes) else str(item)


def aggregate(rows):
    valid=[r for r in rows if r.get('eligible')]
    selection_stats={name:{'available_scenes':sum(v.get('baseline_selection',{}).get(name) is not None for v in valid),
                        'selected_joint_incompatible':sum((v.get('baseline_selection',{}).get(name) or {}).get('joint') is False for v in valid),
                        'selected_local_conflict':sum((v.get('baseline_selection',{}).get(name) or {}).get('local_compatible') is False for v in valid),
                        'selected_temporal_collapse':sum((v.get('baseline_selection',{}).get(name) or {}).get('successor_viable') is False for v in valid)}
                     for name in ('pairwise_utility','joint_existence_utility','discrete_successor_backup')}
    pos=[r for r in valid if r['pairwise_joint_discordance']>0]
    geometric={'positive_scenes_with_t0_circle_overlap':sum(bool(r.get('initial_overlap_circle_pairs')) for r in pos),
               'positive_scenes_with_t0_oriented_overlap':sum(bool(r.get('initial_overlap_oriented_box_pairs')) for r in pos),
               'positive_scenes_all_candidates_incompatible':sum(bool(r.get('all_candidates_joint_incompatible')) for r in pos),
               'positive_scenes_without_t0_circle_overlap':sum(not r.get('initial_overlap_circle_pairs') for r in pos),
               'positive_scenes_without_t0_oriented_overlap':sum(not r.get('initial_overlap_oriented_box_pairs') for r in pos),
               'positive_scenes_with_hard_pair_witness':sum(any(c.get('unconditionally_incompatible_response_pairs') for c in r['candidates']) for r in pos)}
    return {'scenes_scanned':len(rows),'scenes_eligible':len(valid),
            'geometry_causal_diagnostics':geometric,
            'comparator_selection_diagnostics':selection_stats,
            'scenes_pairwise_joint_discordance':sum(r['pairwise_joint_discordance']>0 for r in valid),
            'scenes_joint_but_local_conflict':sum(r['joint_but_local_conflict']>0 for r in valid),
            'scenes_temporal_collapse_proxy':sum(r['temporal_collapse_proxy']>0 for r in valid),
            'scenes_avoidable_collapse_proxy':sum(r['avoidable_collapse_proxy']>0 for r in valid),
            'interpretation':'All rates are proposal-conditioned diagnostics from simplified longitudinal simulation. NOT a publication-level real traffic failure rate.'}


def synthetic_selftest():
    t=np.arange(HORIZON+1)*DT
    ego=np.column_stack([100+t*0,100+t*0])
    opts=[[(-2.,np.column_stack([t*0, t*0]))],[(0.,np.column_stack([t*0+0.4,t*0]))]]
    assert all(bool(o) for o in opts)
    assert joint_response(opts,[.6,.6])[0] is None
    assert path_ok(ego,opts[0][0][1],1.,.6)
    assert joint_response([[(0.,np.column_stack([t*0,t*0]))],[(0.,np.column_stack([t*0+4,t*0]))]],[.6,.6])[0] is not None
    s=np.array([0.,0.,5.,0.,0.,2.,1.]); x=rollout(s,-4.)
    assert np.linalg.norm(x[0]-s[:2])<1e-9
    return {'passed':True,'note':'Toy geometric counterexamples only. They DO NOT establish WOMD prevalence.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/research_probe/diagnostics.json'))
    p.add_argument('--max-scenes',type=int,default=80)
    p.add_argument('--max-candidates',type=int,default=10)
    p.add_argument('--max-neighbors',type=int,default=3)
    p.add_argument('--salt',default='research-probe-fixed-20261011')
    p.add_argument('--geometry',choices=['circle','oriented-box'],default='circle',
                   help='Same response bank; compare circle conservatism to oriented rectangles')
    p.add_argument('--exclude-ids',type=Path,default=None)
    p.add_argument('--selftest',action='store_true')
    p.add_argument('--response-accels',default='-4,-2,0,1.5',
                   help='Comma-separated acceleration samples; vary in preregistered sensitivity checks.')
    p.add_argument('--backup-accels',default='-4,-2,0,1.5')
    args=p.parse_args()
    global ACCELS,EGO_BACKUP_ACCELS,FOOTPRINT_MODE
    FOOTPRINT_MODE=args.geometry
    ACCELS=tuple(float(x) for x in args.response_accels.split(','))
    EGO_BACKUP_ACCELS=tuple(float(x) for x in args.backup_accels.split(','))
    if not ACCELS or not EGO_BACKUP_ACCELS: p.error('empty acceleration bank')
    if args.selftest:
        print(json.dumps(synthetic_selftest(),indent=2));return
    if not args.cache_dir or not args.cache_dir.is_dir():p.error('Provide existing --cache-dir')
    files=[p for p in args.cache_dir.glob('*.npz') if not p.name.startswith('.')]
    if not files:p.error('No .npz found in cache dir')
    exclude=set()
    if args.exclude_ids and args.exclude_ids.is_file():exclude=set(x.strip() for x in args.exclude_ids.read_text().splitlines() if x.strip())
    files.sort(key=lambda p:hashlib.sha256((args.salt+'|'+p.stem).encode()).hexdigest())
    results=[];errors=[]
    for path in files:
        try:
            d=_load(path); sid=_id(d,path)
            if sid in exclude:continue
            row=examine_scene(d,args.max_candidates,args.max_neighbors)
            row['scenario_id']=sid
            results.append(row)
        except Exception as e:
            errors.append({'filename':path.name,'error':str(e)})
        if len(results)>=args.max_scenes:break
    summary=aggregate(results)
    if len(results)<args.max_scenes:
        summary['sampling_warning']='Fewer usable scenes than requested after filtering/exclusions.'
    summary.update({'collision_geometry':FOOTPRINT_MODE,'response_accels':ACCELS,'backup_accels':EGO_BACKUP_ACCELS,'errors':len(errors),'selection':'deterministic salted filename ordering; not label/outcome based',
        'action_semantics':'frozen candidate path + independent constant-acceleration non-ego response; backup is longitudinal-only',
        'limitations':['no Waymax execution','no reaction-model calibration','no roadgraph/rule verification',
                       'simplified collision geometry: circle or static-vehicle-yaw boxes with path-tangent ego','candidate-bank selection bias','existence does not imply execution',
                       'successor is deterministic local-response diagnostic','data may be historically selected']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'scenarios':results,'errors':errors[:50]},indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
