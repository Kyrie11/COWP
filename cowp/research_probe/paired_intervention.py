"""Same-initial-state interventions with a ghost-ego response control.

This probe does not implement a behavioral causal ground truth.  Ghost mode
makes surrounding agents ignore ego *as a leader* but keeps ego's physical
trajectory in collision detection.  Comparing the two separates physical
trajectory conflicts from reactive effects in this simplified model family.
"""
from __future__ import annotations
import argparse,json,hashlib,math
from pathlib import Path
from .reactive_stress import (_load_scene,_pair_conflicts,REGIMES,CONTROLS,
                              simulate,observed_ade)


def _pair_set(row):
    return {tuple(sorted(p)) for p in row['non_ego_collision_pairs']}


def analyze_scene(path, progress_tolerance=3., steps=20, controls=None, responder_delay_steps=0):
    data,state,L,W,valid,ego=_load_scene(Path(path))
    first=_pair_conflicts(state[:,:2],state[:,4],L,W,valid)
    out={'scenario_id':Path(path).stem,'eligible':False, 'protocol':'longitudinal_screen_not_waymax',
         'model_independence':False,'causal_intervention_identified_from_logs':False}
    if first:out['reason']='initial_overlap';return out
    if valid.sum()<3:out['reason']='too_few_agents';return out
    velocities=state[:,2:4]
    import numpy as np
    import math
    sp=np.linalg.norm(velocities,axis=1)
    heading=np.column_stack([np.cos(state[:,4]),np.sin(state[:,4])])
    mismatch=(sp>3)&(np.sum(velocities*heading,axis=1)/np.maximum(sp,1e-5)<.65)&valid
    if mismatch.any():out['reason']='velocity_heading_mismatch';return out
    out['eligible']=True
    out['regimes']={}
    ids=data['agent_ids'].tolist()
    controls=tuple(float(x) for x in (CONTROLS if controls is None else controls))
    if len(set(controls))!=len(controls):raise ValueError('Duplicate ego actions')
    if len(controls)<2:raise ValueError('Need at least two ego actions')
    out['ego_action_grid_mps2']=list(controls)
    out['responder_delay_steps']=int(responder_delay_steps)
    for regime in REGIMES:
        rows=[]
        for accel in controls:
            actual=simulate(state,L,W,valid,ego,ids,accel,regime,steps=steps,responder_delay_steps=responder_delay_steps)
            ghost=simulate(state,L,W,valid,ego,ids,accel,regime,steps=steps,ghost_ego_to_others=True,responder_delay_steps=responder_delay_steps)
            base={k:actual[k] for k in ('collision','ego_collision','non_ego_collision','non_ego_collision_pairs',
                        'ego_collision_pairs','ego_progress_m','near_term_ego_clear','non_ego_min_accel_by_agent')}
            base.update({'ego_acceleration_mps2':accel,
                'ghost_non_ego_collision_pairs':ghost['non_ego_collision_pairs'],
                'ghost_non_ego_min_accel_by_agent':ghost['non_ego_min_accel_by_agent'],
                'new_non_ego_pairs_vs_ghost':[list(p) for p in sorted(_pair_set(actual)-_pair_set(ghost))],
                'new_non_ego_pair_count_vs_ghost':len(_pair_set(actual)-_pair_set(ghost)),
                'response_ade_1s_m':observed_ade(data,actual['paths_xy'],
                                      [i for i in range(len(ids)) if i!=ego and valid[i]])})
            # A reaction-burden proxy can change without any collision.
            # Never turn it into a physical or human causal label.
            actual_b=actual['non_ego_min_accel_by_agent'];ghost_b=ghost['non_ego_min_accel_by_agent']
            base['additional_hard_braking_response_agents']=[int(i) for i in actual_b
                 if float(actual_b[i])<=-4.0 and float(actual_b[i])<float(ghost_b[i])-1.5]
            rows.append(base)
        cases=[];noninferior=[]
        for bad in rows:
            if not bad['near_term_ego_clear'] or not bad['new_non_ego_pairs_vs_ghost']:continue
            for good in rows:
                if good is bad or good['collision'] or good['new_non_ego_pairs_vs_ghost']:continue
                if good['ego_progress_m']+progress_tolerance<bad['ego_progress_m']:continue
                case={'intervention':bad['ego_acceleration_mps2'],'matched_alternative':good['ego_acceleration_mps2'],
                              'new_non_ego_pairs_vs_ghost':bad['new_non_ego_pairs_vs_ghost'],
                              'progress_difference_m':float(good['ego_progress_m']-bad['ego_progress_m'])}
                noninferior.append(case)
                if abs(case['progress_difference_m'])<=progress_tolerance:cases.append(case)
        out['regimes'][regime]={'actions':rows,'progress_noninferior_proxy_cases':noninferior,
                               'symmetric_progress_matched_proxy_cases':cases,'strict_proxy_cases':cases}
    # Require the exact same bad ego action and a common safe progress-matched
    # alternative in each shared family regime; not "independent validation".
    out['mechanism_claim_level']='DELAYED_IDM_FAMILY_SCREEN_ONLY_NOT_HUMAN_CAUSAL'
    joint=set.intersection(*(set((x['intervention'],x['matched_alternative'])
                               for x in out['regimes'][r]['strict_proxy_cases']) for r in REGIMES))
    out['three_regime_shared_action_pairs']=[list(x) for x in sorted(joint)]
    return out


def run(cohort, output, *, max_scenes=0, progress_tolerance=3., selection_order="hash", controls=None, responder_delay_steps=0):
    cohort=Path(cohort); manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    rows=[];errors=[]
    entries=manifest['entries']
    if selection_order not in ('hash','manifest'):raise ValueError(selection_order)
    if selection_order=='hash':
        entries=sorted(entries,key=lambda e:hashlib.sha256(('native-eval-v1|'+e['scenario_id']).encode()).hexdigest())
    for e in entries[:max_scenes or None]:
        try:rows.append(analyze_scene(cohort/e['file'], progress_tolerance,controls=controls,responder_delay_steps=responder_delay_steps))
        except Exception as exc:errors.append({'scenario_id':e['scenario_id'],'error':repr(exc)})
    result={'summary':{'n_checked':len(rows),'n_eligible':sum(r['eligible'] for r in rows),
                       'n_three_regime_shared':sum(bool(r.get('three_regime_shared_action_pairs')) for r in rows),
                       'errors':len(errors),'scientific_claim_allowed':False,
                       'not_waymax':True,'not_independent_behavioral_families':True,
                       'not_full_causal_identification':True,'selection_order':selection_order,
                       'ego_controls_mps2':list(CONTROLS if controls is None else controls),
                       'responder_delay_steps':responder_delay_steps},'rows':rows,'errors':errors}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result['summary']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-scenes',type=int,default=0)
    p.add_argument('--progress-tolerance',type=float,default=3.)
    p.add_argument('--selection-order',choices=['hash','manifest'],default='hash')
    p.add_argument('--ego-accelerations',nargs='+',type=float,default=None,help='Pre-register dense action grid e.g. -3 to +1.5 by 0.25')
    p.add_argument('--responder-delay-steps',type=int,default=0,help='Delay 0.2s/step; only same-model sensitivity')
    a=p.parse_args()
    print(json.dumps(run(a.cohort,a.output,max_scenes=a.max_scenes,progress_tolerance=a.progress_tolerance,selection_order=a.selection_order,controls=a.ego_accelerations,responder_delay_steps=a.responder_delay_steps),indent=2))

if __name__=='__main__':main()
