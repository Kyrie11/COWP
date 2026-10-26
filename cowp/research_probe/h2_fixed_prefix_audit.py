"""Re-test 4s H2 cases at 8s while preserving the EXACT first 4s ego input.

NO observed human counterfactual / no map-based safety / no proof of H2.
The old 8s 'cos1' re-optimized the pulse over 8s, changing its first 4s.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .matched_response_test import make_profiles, evaluate_scene
from .reactive_stress import REGIMES


def keys(row,regime):
    return {(x['bad_profile'],x['alternative_profile'],tuple(tuple(p) for p in x['pair_object_ids']))
            for x in row['regimes'][regime]['strict_matches']}


def run(cohort,ids_file,output,short_steps=20,long_steps=40,tolerance=3.):
    if long_steps<=short_steps:raise ValueError('long_steps must be greater than short_steps')
    assert short_steps%2==0 and long_steps%2==0
    short=make_profiles(short_steps)
    long=make_profiles(long_steps,pulse_steps=short_steps)
    byid={x['id']:x for x in long}
    assert all((p['controls']==byid[p['id']]['controls'][:short_steps]).all() for p in short), 'EGO CONTROLS CHANGED'
    cohort=Path(cohort)
    manifest=json.loads((cohort/'cohort_manifest.json').read_text())
    lookup={e['scenario_id']:e for e in manifest['entries']}
    ids=[x.strip() for x in Path(ids_file).read_text().splitlines() if x.strip()]
    if len(set(ids))!=len(ids): raise ValueError('duplicate scene IDs')
    rows=[];errors=[]
    for sid in ids:
        e=lookup.get(sid)
        if not e:errors.append({'scenario_id':sid,'reason':'not in cohort'});continue
        if e.get('evaluation_partition')=='sealed_holdout':errors.append({'scenario_id':sid,'reason':'sealed holdout forbidden'});continue
        try:
            a=evaluate_scene(cohort/e['file'],steps=short_steps,tolerance=tolerance,pulse_steps=short_steps)
            b=evaluate_scene(cohort/e['file'],steps=long_steps,tolerance=tolerance,pulse_steps=short_steps)
            if not a['eligible'] or not b['eligible']:
                rows.append({'scenario_id':sid,'eligible':False,'reason':a.get('exclusion_reason') or b.get('exclusion_reason')});continue
            rs={}
            for regime in REGIMES:
                ka,kb=keys(a,regime),keys(b,regime)
                rs[regime]={'short_cases':len(ka),'long_cases':len(kb),
                            'shared_cases':len(ka&kb),'short_only_cases':len(ka-kb),'long_only_cases':len(kb-ka),
                            'shared_pairs':[{'bad':i,'alternative':j,'pair_object_ids':[list(p) for p in pairs]} for i,j,pairs in sorted(ka&kb)]}
            rows.append({'scenario_id':sid,'eligible':True,'regimes':rs})
        except Exception as ex:errors.append({'scenario_id':sid,'reason':repr(ex)})
    valid=[r for r in rows if r.get('eligible')]
    summary={'attempted':len(ids),'eligible':len(valid),'errors':len(errors),'pulse_steps':short_steps,
             'short_horizon_steps':short_steps,'long_horizon_steps':long_steps,
             'prefix_control_parity':True,'scientific_claim_allowed':False,
             'shared_any_regime_cases':sum(any(z['shared_cases']>0 for z in r['regimes'].values()) for r in valid),
             'limitations':['IDM-family fixed-heading simulation only','Not road-valid/Waymax','Endpoint 8s condition can fail if late ego crash, even when earlier pair collision remains']}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps({'summary':summary,'rows':rows,'errors':errors},indent=2),encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',required=True);p.add_argument('--ids-file',required=True);p.add_argument('--output',required=True)
    p.add_argument('--short-steps',type=int,default=20);p.add_argument('--long-steps',type=int,default=40)
    a=p.parse_args();print(json.dumps(run(a.cohort,a.ids_file,a.output,a.short_steps,a.long_steps),indent=2))
if __name__=='__main__':main()
