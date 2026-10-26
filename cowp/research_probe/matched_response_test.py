"""Pre-specified progress-matched ego impulse contrasts for native WOMD.

This is a cheap, DETERMINISTIC, straight-heading IDM-family *mechanism screen*.
It is neither a Waymax rollout nor an estimate of human potential outcomes.
Matched pairs use equal base acceleration and opposite smooth cosine pulses:
  a(t) = base +/- amplitude*cos(2*pi*(t+0.5)/N).
For non-saturating ego dynamics, these pulses have equal final velocity and
net longitudinal displacement; whether matching holds is checked on the
EXECUTED simulator trajectory, never assumed from the profile design.
The ghost control prevents the ego from being an IDM leader while keeping its
PHYSICAL trajectory; its use provides a model-relative mediation diagnostic.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene,_pair_conflicts,simulate,REGIMES,DT


def make_profiles(steps=20,base_accels=(-1.0,0.0,1.0), amplitudes=(1.0,2.0), pulse_steps=None):
    if steps<8 or steps%2: raise ValueError('even steps>=8 required')
    pulse_steps = steps if pulse_steps is None else int(pulse_steps)
    if pulse_steps<8 or pulse_steps>steps or pulse_steps%2: 
        raise ValueError('pulse_steps must be even, at least 8, and no larger than steps')
    if any(a<=0 for a in amplitudes):raise ValueError('amplitudes must be positive')
    # FIXED-PREFIX extension: append base acceleration after the same 4s pulse.
    # Old steps=40 recomputed a slower 8s pulse, so it was NOT the same intervention.
    wave=np.zeros(steps,dtype=float)
    wave[:pulse_steps]=np.cos(2*np.pi*(np.arange(pulse_steps)+0.5)/pulse_steps)
    out=[]
    for base in base_accels:
        out.append({'id':f'b{base:+g}_flat','base':float(base),'amplitude':0.0,
                    'controls':np.full(steps,float(base))})
        for amp in amplitudes:
            for sign in (-1,1):
                out.append({'id':f'b{base:+g}_cos{amp:g}_{sign:+d}',
                            'base':float(base),'amplitude':float(sign*amp),
                            'controls':float(base)+float(sign*amp)*wave})
    return out


def _near_pair_min_center(paths,valid,ego,range_m=15.0):
    """Minimum center distances: an exploratory sensitivity proxy, NOT collision probability."""
    out={}
    indices=[i for i,v in enumerate(valid) if v and i!=ego]
    for j,i in enumerate(indices):
        for k in indices[j+1:]:
            dist=np.linalg.norm(paths[1:,i]-paths[1:,k],axis=1)
            mind=float(np.min(dist))
            if mind<=range_m:out[f'{i}-{k}']=mind
    return out


def _pairs(raw):
    return {tuple(sorted(map(int,p))) for p in raw['non_ego_collision_pairs']}


def _eligible(data,state,L,W,valid,ego):
    if _pair_conflicts(state[:,:2],state[:,4],L,W,valid):return 'initial_oriented_overlap'
    if valid.sum()<3:return 'fewer_than_three'
    velocities=state[:,2:4]; speed=np.linalg.norm(velocities,axis=1)
    directions=np.column_stack([np.cos(state[:,4]),np.sin(state[:,4])])
    mismatch=(speed>3)&(np.sum(velocities*directions,axis=1)/np.maximum(speed,1e-5)<.65)&valid
    if mismatch.any():return 'velocity_heading_mismatch'
    if int(ego)!=0:return 'unexpected_local_ego_index_not_zero'
    return None


def evaluate_scene(scene_path,*,steps=20,tolerance=3.,delay=0,base_accels=(-1.,0.,1.),amplitudes=(1.,2.),pulse_steps=None):
    path=Path(scene_path)
    data,state,L,W,valid,ego=_load_scene(path)
    ids=data['agent_ids'].tolist()
    why=_eligible(data,state,L,W,valid,ego)
    row={'scenario_id':path.stem,'eligible':why is None,'protocol':'matched_cosine_longitudinal_proxy',
         'independent_model_validation':False,'causal_human_outcome':False,
         'exclusion_reason':why,'steps':steps,'pulse_horizon_steps':(steps if pulse_steps is None else pulse_steps),'responder_delay_steps':delay}
    if why:return row
    profiles=make_profiles(steps,base_accels,amplitudes,pulse_steps=pulse_steps)
    row['profile_controls']={p['id']:[float(x) for x in p['controls']] for p in profiles}
    row['regimes']={}
    for regime in REGIMES:
        scored={}
        for p in profiles:
            raw=simulate(state,L,W,valid,ego,ids,p['controls'],regime,steps=steps,responder_delay_steps=delay)
            ghost=simulate(state,L,W,valid,ego,ids,p['controls'],regime,steps=steps,
                           ghost_ego_to_others=True,responder_delay_steps=delay)
            actual_pairs=_pairs(raw);ghost_pairs=_pairs(ghost)
            near_dist=_near_pair_min_center(raw['paths_xy'],valid,ego)
            ghost_near_dist=_near_pair_min_center(ghost['paths_xy'],valid,ego)
            ego_heading=state[ego,4]
            ego_dir=np.array([math.cos(ego_heading),math.sin(ego_heading)])
            progress=float((raw['paths_xy'][-1,ego]-raw['paths_xy'][0,ego])@ego_dir)
            braking={i:float(v) for i,v in raw['non_ego_min_accel_by_agent'].items()}
            ghost_braking={i:float(v) for i,v in ghost['non_ego_min_accel_by_agent'].items()}
            scored[p['id']]={
                'profile':p['id'],'base_acceleration':p['base'],'pulse_amplitude':p['amplitude'],
                'ego_initial_speed':float(np.linalg.norm(state[ego,2:4])),
                'ego_route_tangent_progress_proxy_m':progress,
                'ego_any_collision':bool(raw['ego_collision']),
                'any_collision':bool(raw['collision']),
                'near_term_ego_clear':bool(raw['near_term_ego_clear']),
                'full_horizon_ego_clear':not bool(raw['ego_collision']),
                'near_pair_min_center_m':near_dist,
                'ghost_near_pair_min_center_m':ghost_near_dist,
                'non_ego_pairs':[list(pair) for pair in sorted(actual_pairs)],
                'ghost_non_ego_pairs':[list(pair) for pair in sorted(ghost_pairs)],
                'ego_response_mediated_pairs':[list(pair) for pair in sorted(actual_pairs-ghost_pairs)],
                'max_extra_braking_mps2':max((ghost_braking[i]-braking[i] for i in braking),default=0.),
                'per_agent_extra_braking_vs_ghost_mps2':{
                    i:float(ghost_braking[i]-braking[i]) for i in braking},
            }
        # Pre-declared counterfactual pairs within SAME base acceleration:
        # opposite pulse signs and equal amplitude. Compare both directions.
        matches=[];nonmatched=[];burden_matches=[];near_pair_matches=[]
        for base in base_accels:
            for amp in amplitudes:
                pminus=f'b{base:+g}_cos{amp:g}_-1'
                pplus =f'b{base:+g}_cos{amp:g}_+1'
                for b_id,g_id in [(pminus,pplus),(pplus,pminus)]:
                    bad=scored[b_id];good=scored[g_id]
                    # Identity-preserving same-pair event must have been
                    # absent in the no-ego-leader ghost control, not just an
                    # unrelated collision elsewhere in the scene.
                    new_pairs={tuple(p) for p in bad['ego_response_mediated_pairs']}
                    delta=good['ego_route_tangent_progress_proxy_m']-bad['ego_route_tangent_progress_proxy_m']
                    if new_pairs and bad['full_horizon_ego_clear'] and not good['any_collision']:
                        item={'bad_profile':b_id,'alternative_profile':g_id,
                              'pair_ids':[list(p) for p in sorted(new_pairs)],
                              'pair_object_ids':[list(sorted((int(ids[a]),int(ids[b])))) for a,b in sorted(new_pairs)],
                              'progress_delta_m':float(delta),
                              'ego_response_mediated_within_proxy':True}
                        if abs(delta)<=tolerance:matches.append(item)
                        else:nonmatched.append(item)
                    # Continuous RESPONSE-BURDEN proxy, separately from H2;
                    # test BOTH directions of every frozen action pair.
                    if abs(delta)>tolerance or not bad['full_horizon_ego_clear'] or good['any_collision']:
                        continue
                    # Exploratory H2 sensitivity: the same non-ego pair becomes
                    # closer under one matched ego action; a ghost-ego contrast
                    # must also show that leader-response coupling mattered.
                    # This is NOT a validated physical collision or human effect.
                    for key,d_bad in bad['near_pair_min_center_m'].items():
                        d_good=good['near_pair_min_center_m'].get(key,float('inf'))
                        d_ghost=bad['ghost_near_pair_min_center_m'].get(key,float('inf'))
                        if d_bad<=8.0 and d_good-d_bad>=1.5 and d_ghost-d_bad>=1.0:
                            a,b=map(int,key.split('-'))
                            near_pair_matches.append({
                                'bad_profile':b_id,'alternative_profile':g_id,
                                'pair_local_indices':[a,b],
                                'pair_object_ids':[int(ids[a]),int(ids[b])],
                                'bad_pair_min_center_m':round(d_bad,4),
                                'alternative_pair_min_center_m':round(d_good,4),
                                'ghost_pair_min_center_m':round(d_ghost,4),
                                'progress_delta_m':float(delta),
                                'status':'EXPLORATORY_DISTANCE_PROXY_NOT_H2'})
                    for i,bval in bad['per_agent_extra_braking_vs_ghost_mps2'].items():
                        gval=good['per_agent_extra_braking_vs_ghost_mps2'][i]
                        burden_delta=bval-gval
                        if burden_delta>=1.0:
                            burden_matches.append({'bad_profile':b_id,'alternative_profile':g_id,
                                'agent_id_local':int(i),'burden_reduction_mps2':float(burden_delta),
                                'progress_delta_m':float(delta),
                                'risk_proxy_not_collision':True})
        row['regimes'][regime]={'actions':scored,'strict_matches':matches,
                                'unmatched_pairs':nonmatched,
                                'matched_burden_proxy_cases':burden_matches,
                                'near_pair_proxy_cases':near_pair_matches,
                                'n_strict_matches':len(matches)}
    keys=lambda k:{(x['bad_profile'],x['alternative_profile'],tuple(map(tuple,x.get('pair_object_ids',x['pair_ids']))))
                    for x in row['regimes'][k]['strict_matches']}
    common=set.intersection(*(keys(r) for r in REGIMES))
    burden_keys=lambda k:{(x['bad_profile'],x['alternative_profile'],x['agent_id_local'])
                            for x in row['regimes'][k]['matched_burden_proxy_cases']}
    burden_shared=set.intersection(*(burden_keys(r) for r in REGIMES))
    near_keys=lambda k:{(x['bad_profile'],x['alternative_profile'],tuple(x['pair_object_ids']))
                       for x in row['regimes'][k]['near_pair_proxy_cases']}
    common_near=set.intersection(*(near_keys(r) for r in REGIMES))
    row['all_three_regimes_shared_near_pair_keys']=[
        {'bad_profile':b,'alternative_profile':g,'pair_object_ids':list(ids)}
        for b,g,ids in sorted(common_near)]
    row['all_three_regimes_shared_burden_proxy_keys']=[
        {'bad_profile':b,'alternative_profile':g,'agent_id_local':i}
        for b,g,i in sorted(burden_shared)]
    row['all_three_regimes_shared_matched_pair_keys']=[
        {'bad_profile':b,'alternative_profile':g,'pair_ids':[list(p) for p in pairs]}
        for b,g,pairs in sorted(common)]
    return row


def run(cohort,output,*,max_scenes=0,steps=20,tolerance=3.,delay=0,
        base_accels=(-1.,0.,1.),amplitudes=(1.,2.),pulse_steps=None,selection_order='hash',scenario_ids=None,partition='all'):
    cohort=Path(cohort);manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    entries=manifest['entries']
    if partition!='all':
        entries=[e for e in entries if e.get('evaluation_partition','development')==partition]
    if selection_order=='hash':
        entries=sorted(entries,key=lambda e:hashlib.sha256(('matched-proxy-v1|'+e['scenario_id']).encode()).hexdigest())
    elif selection_order!='manifest':raise ValueError('selection_order must be hash or manifest')
    if scenario_ids is not None:
        wanted=set(scenario_ids);entries=[e for e in entries if e['scenario_id'] in wanted]
        if len(entries)!=len(wanted):raise ValueError(f'missing requested IDs: {sorted(wanted-{x["scenario_id"] for x in entries})}')
    entries=entries[:max_scenes or None]
    rows=[];errors=[]
    for i,e in enumerate(entries):
        try:
            r=evaluate_scene(cohort/e['file'],steps=steps,tolerance=tolerance,delay=delay,
                             base_accels=base_accels,amplitudes=amplitudes,pulse_steps=pulse_steps)
            # NPZ local actor identity MUST match cohort provenance.
            if r['scenario_id']!=e['scenario_id']:raise RuntimeError('scenario-ID/file mismatch')
            rows.append(r)
        except Exception as ex:errors.append({'scenario_id':e['scenario_id'],'error':repr(ex)})
        if (i+1)%25==0:print(f'[matched-response] {i+1}/{len(entries)}',flush=True)
    valid=[x for x in rows if x.get('eligible')]
    summary={'scientific_claim_allowed':False,'attempted':len(entries),'completed':len(rows),
             'eligible':len(valid),'errors':len(errors),'selection_order':selection_order,
             'evaluation_partition':partition,
             'n_regime_matched':{r:sum(bool(x['regimes'][r]['strict_matches']) for x in valid) for r in REGIMES},
             'n_shared_three_regime':sum(bool(x['all_three_regimes_shared_matched_pair_keys']) for x in valid),
             'n_matched_burden_proxy':{r:sum(bool(x['regimes'][r]['matched_burden_proxy_cases']) for x in valid) for r in REGIMES},
             'n_shared_three_regime_burden_proxy':sum(bool(x['all_three_regimes_shared_burden_proxy_keys']) for x in valid),
             'n_near_pair_proxy':{r:sum(bool(x['regimes'][r]['near_pair_proxy_cases']) for x in valid) for r in REGIMES},
             'n_shared_near_pair_proxy':sum(bool(x.get('all_three_regimes_shared_near_pair_keys')) for x in valid),
             'interpretation':'finite IDM-family counterfactual only; not real-human do-effect or Waymax; route-tangent approximate',
             'protocol':{'steps':steps,'pulse_horizon_steps':(steps if pulse_steps is None else pulse_steps),'dt':DT,'tolerance_m':tolerance,'base_accels':list(base_accels),
                         'amplitudes':list(amplitudes),'responder_delay_steps':delay,'burden_contrast_threshold_mps2':1.0}}
    result={'summary':summary,'rows':rows,'errors':errors}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--pulse-steps',type=int,default=None,help='Keep first N steps identical when extending horizon; suffix returns to base acceleration')
    p.add_argument('--progress-tolerance',type=float,default=3.)
    p.add_argument('--responder-delay-steps',type=int,default=0)
    p.add_argument('--selection-order',choices=('hash','manifest'),default='hash')
    p.add_argument('--partition',choices=('all','development','sealed_holdout'),default='all')
    p.add_argument('--scenario-ids-file',type=Path)
    a=p.parse_args(); ids=None
    if a.scenario_ids_file:ids=[x.strip().split()[0] for x in a.scenario_ids_file.read_text().splitlines() if x.strip()]
    print(json.dumps(run(a.cohort,a.output,max_scenes=a.max_scenes,steps=a.steps,
          tolerance=a.progress_tolerance,delay=a.responder_delay_steps,pulse_steps=a.pulse_steps,selection_order=a.selection_order,
          scenario_ids=ids,partition=a.partition),indent=2))
if __name__=='__main__':main()
