"""Read-only v12 speed-only pair: unchanged v11 safety endpoints.

IMPORTANT: these are EX POST rollout oracle risk comparisons.  No learned
planner's pre-rollout choice is present; do not label them RQ3 failures.
"""
from __future__ import annotations
import argparse,json,collections
from pathlib import Path
from ride.v11_chain_evidence import read_panel
from ride.science import Protocol, validate_pair, same_initial, same_ego_execution
from ride.v10_mechanism_report import _risk_key,_lexicographic_order


def evaluate(paths):
    p=Protocol();pairs=[];counters=collections.Counter();panels=[]
    for path in paths:
        m,ts,complete,broken=read_panel(path)
        panels.append({'path':str(path),'complete':complete,'broken':broken,'anchor_version':m.get('anchor_protocol_version')})
        by={(x.get('anchor'),x.get('seed'),x.get('family'),x.get('profile')):x for x in ts}
        keys={(k[0],k[1]) for k in by}
        for anchor,seed in sorted(keys,key=str):
            family_results={}
            for f in ('tm','behavior'):
                a=by.get((anchor,seed,f,'speed_early'))
                b=by.get((anchor,seed,f,'speed_late'))
                if a is None or b is None:
                    family_results[f]={'eligible':False,'reason':'missing_v12_speed_pair'}
                    continue
                ok,reason,progress=validate_pair(a,b,p)
                family_results[f]={'eligible':ok,'reason':reason,'progress_delta_m':progress}
                if ok:
                    family_results[f]['risk_order']=_lexicographic_order(_risk_key(a,p),_risk_key(b,p))
                    counters['within_family_eligible']+=1
            eligible=all(x.get('eligible') for x in family_results.values())
            errors=[]
            if eligible:
                for action in ('speed_early','speed_late'):
                    a=by[(anchor,seed,'tm',action)]
                    b=by[(anchor,seed,'behavior',action)]
                    ok,why=same_initial(a,b,p)
                    if not ok:errors.append(action+':'+why)
                    ok,why=same_ego_execution(a,b)
                    if not ok:errors.append(action+':'+why)
            eligible=bool(eligible and not errors)
            reversal=bool(eligible and family_results['tm']['risk_order'] in (-1,1) and
                          family_results['tm']['risk_order']==-family_results['behavior']['risk_order'])
            counters['contexts_seen']+=1
            counters['cross_family_eligible']+=int(eligible)
            counters['strict_primary_endpoint_reversal']+=int(reversal)
            pairs.append({'map':m.get('map'),'anchor':anchor,'seed':seed,'cross_family_eligible':eligible,
                          'strict_risk_order_reversal':reversal,'arm':family_results,'cross_errors':errors})
    for key in ('within_family_eligible','contexts_seen','cross_family_eligible','strict_primary_endpoint_reversal'):
        counters.setdefault(key,0)
    return {'scope':'EXPLORATORY_V12_NEW_SPEED_PAIR_NOT_RETROACTIVE_RQ1',
            'panels':panels,'summary':dict(counters),'pairs':pairs,
            'ex_ante_planner_choices':0,'RQ3':'NOT_TESTED',
            'limits':['Constant final progress is checked but not assumed from target speed integrals.',
             'A risk ranking computed after an executed rollout is not an ex-ante planner choice.',
             'Do not pool scene repeats and multiple arms as IID independent road observations.']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();o=evaluate(a.input)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(o['summary'],indent=2))
    return 0 if all(x['complete'] for x in o['panels']) else 4
if __name__=='__main__':raise SystemExit(main())
