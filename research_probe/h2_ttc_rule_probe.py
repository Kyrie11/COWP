"""STRUCTURALLY DIFFERENT rule-based model sanity check on shortlisted cases.

Geometric TTC braking vs longitudinal IDM. They share simplified collision
physics and cannot count as two independent, calibrated closed-loop platforms.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from .reactive_stress import _load_scene,simulate,_pair_conflicts,DT
from .matched_response_test import make_profiles

def single_case(path,steps=20,pulse_steps=None,tolerance=3.,delay=0):
    d,st,L,W,valid,ego=_load_scene(path);ids=list(map(int,d['agent_ids'].tolist()))
    if _pair_conflicts(st[:,:2],st[:,4],L,W,valid):return {'scenario_id':Path(path).stem,'eligible':False,'reason':'initial_overlap'}
    models={}
    for family in ('cautious','assertive','heterogeneous','ttc_gap_rule'):
        actions={}
        for p in make_profiles(steps,pulse_steps=pulse_steps):
            bad=simulate(st,L,W,valid,ego,ids,p['controls'],family,steps=steps,responder_delay_steps=delay)
            ghost=simulate(st,L,W,valid,ego,ids,p['controls'],family,steps=steps,
                           ghost_ego_to_others=True,responder_delay_steps=delay)
            h=st[ego,4];direction=np.array([np.cos(h),np.sin(h)])
            progress=float((bad['paths_xy'][-1,ego]-bad['paths_xy'][0,ego])@direction)
            pairs={tuple(sorted(map(int,p))) for p in bad['non_ego_collision_pairs']}
            gpairs={tuple(sorted(map(int,p))) for p in ghost['non_ego_collision_pairs']}
            actions[p['id']]={'progress':progress,'pairs':pairs-gpairs,'any_collision':bad['collision'],
                              'ego_collision':bad['ego_collision']}
        witnesses=[]
        for base in (-1.,0.,1.):
            for amp in (1.,2.):
                a,b=f'b{base:+g}_cos{amp:g}_-1',f'b{base:+g}_cos{amp:g}_+1'
                for bad_name,good_name in ((a,b),(b,a)):
                    x,y=actions[bad_name],actions[good_name]
                    if (x['pairs'] and not x['ego_collision'] and not y['any_collision']
                      and abs(x['progress']-y['progress'])<=tolerance):
                        for pair in x['pairs']:
                            witnesses.append({'bad':bad_name,'good':good_name,'pair_object_ids':sorted([ids[q] for q in pair])})
        models[family]=witnesses
    witness_sets={model:{(x['bad'],x['good'],tuple(x['pair_object_ids'])) for x in v} for model,v in models.items()}
    return {'scenario_id':Path(path).stem,'eligible':True,'models':models,
            'ttc_idm_same_case_overlap':{
                model:len(witness_sets[model]&witness_sets['ttc_gap_rule'])
                for model in ('cautious','assertive','heterogeneous')}}

def run(cohort,ids_file,output,steps=20,pulse_steps=None,delay=0):
    cohort=Path(cohort);manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    entries={e['scenario_id']:e for e in manifest['entries']}
    ids=[x.strip() for x in Path(ids_file).read_text().splitlines() if x.strip()]
    rows=[];errors=[]
    for sid in ids:
        if sid not in entries:errors.append({'scenario_id':sid,'error':'missing from cohort'});continue
        e=entries[sid]
        if e.get('evaluation_partition')=='sealed_holdout':errors.append({'scenario_id':sid,'error':'sealed holdout forbidden'});continue
        try:rows.append(single_case(cohort/e['file'],steps,pulse_steps,delay=delay))
        except Exception as ex:errors.append({'scenario_id':sid,'error':repr(ex)})
    v=[x for x in rows if x['eligible']]
    summary={'attempted':len(ids),'eligible':len(v),'errors':errors,'steps':steps,'pulse_steps':pulse_steps or steps,
             'ttc_positive_scenes':sum(bool(x['models']['ttc_gap_rule']) for x in v),
             'ttc_idm_same_case_overlap_scenes':sum(any(x['ttc_idm_same_case_overlap'].values()) for x in v),
             'status':'STRUCTURALLY_DIFFERENT_RULE_WITHIN_SHARED_SIMULATOR_NOT_INDEPENDENT_CLOSED_LOOP_VALIDATION',
             'limitations':['Shared fixed-heading physics, not Waymax or human do-effect',
                            'TTC rule must undergo same observed trajectory calibration before scientific use',
                            'Selected dev IDs are discovery biased']}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--ids-file',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20);p.add_argument('--pulse-steps',type=int);p.add_argument('--delay',type=int,default=0)
    a=p.parse_args();print(json.dumps(run(a.cohort,a.ids_file,a.output,a.steps,a.pulse_steps,a.delay),indent=2))
if __name__=='__main__':main()
