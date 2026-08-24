from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np


def _load(path: str) -> dict:
    p=json.load(open(path, encoding='utf-8'))
    if 'scenario_results' in p:
        return p
    ids=[str(x) for x in p.get('scenario_ids_resolved',[])]
    mets=p.get('standard_metrics',[])
    diags=p.get('scenario_diagnostics',[])
    p['scenario_results']=[{'scenario_id':sid,'standard_metrics':mets[i] if i<len(mets) else {},'diagnostics':diags[i] if i<len(diags) else {}} for i,sid in enumerate(ids)]
    return p


def _mcnemar_exact(b:int,c:int)->float:
    n=b+c
    if n==0:return 1.0
    k=min(b,c)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)


def _paired(base:dict, other:dict)->dict:
    bm={r['scenario_id']:r for r in base['scenario_results']}; om={r['scenario_id']:r for r in other['scenario_results']}
    ids=sorted(set(bm)&set(om))
    out={'paired_scenarios':len(ids)}
    for key in ['CR','CollisionRate','OffroadRate','KinematicsInfeasibilityRate']:
        b=np.asarray([float(bm[s]['standard_metrics'].get(key,0))>0 for s in ids],bool)
        o=np.asarray([float(om[s]['standard_metrics'].get(key,0))>0 for s in ids],bool)
        wors=int((~b&o).sum()); improve=int((b&~o).sum())
        out[key]={'base_rate':float(b.mean()),'other_rate':float(o.mean()),'delta':float(o.mean()-b.mean()),'base_safe_to_other_fail':wors,'base_fail_to_other_safe':improve,'mcnemar_exact_p':_mcnemar_exact(wors,improve)}
    bd=[]; od=[]
    for s in ids:
        a=float(bm[s]['standard_metrics'].get('EP',np.nan)); b=float(om[s]['standard_metrics'].get('EP',np.nan))
        if np.isfinite(a) and np.isfinite(b): bd.append(a); od.append(b)
    if bd:
        d=np.asarray(od)-np.asarray(bd)
        rng=np.random.default_rng(16826)
        boot=np.asarray([rng.choice(d,size=len(d),replace=True).mean() for _ in range(5000)])
        out['EP']={'paired_finite':len(d),'base_mean':float(np.mean(bd)),'other_mean':float(np.mean(od)),'delta_mean':float(d.mean()),'delta_median':float(np.median(d)),'bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}
    return out


def _failure_localization(p:dict)->dict:
    rows=p.get('scenario_results',[]); out={}
    mapping={'Collision':'CollisionRate','Offroad':'OffroadRate','Kinematics':'KinematicsInfeasibilityRate'}
    for label,key in mapping.items():
        pos=[r for r in rows if float(r.get('standard_metrics',{}).get(key,0))>0]
        if not pos: continue
        vals=[]; at=[]; conv=[]; valid=[]; macros={}; reasons={}
        suffix=label.lower()
        for r in pos:
            d=r.get('diagnostics',{}) or {}
            v=d.get(f'fallback_rate_before_first_{suffix}')
            if v is not None: vals.append(float(v))
            a=d.get(f'fallback_at_action_before_first_{suffix}')
            if a is not None: at.append(bool(a))
            c=d.get(f'selected_conventional_safe_at_action_before_first_{suffix}')
            if c is not None: conv.append(bool(c))
            vv=d.get(f'selected_candidate_valid_at_action_before_first_{suffix}')
            if vv is not None: valid.append(bool(vv))
            macro=str(d.get(f'selected_macro_name_at_action_before_first_{suffix}','unknown'))
            macros[macro]=macros.get(macro,0)+1
            reason=str(d.get(f'fallback_reason_at_action_before_first_{suffix}','none'))
            reasons[reason]=reasons.get(reason,0)+1
        both=[bool(a) and bool(c) for a,c in zip(at,conv)] if len(at)==len(conv) else []
        out[label]={
            'positive_episodes':len(pos),
            'mean_fallback_rate_before_first_event':float(np.mean(vals)) if vals else None,
            'fallback_action_immediately_before_first_event_rate':float(np.mean(at)) if at else None,
            'conventional_safe_action_immediately_before_first_event_rate':float(np.mean(conv)) if conv else None,
            'fallback_and_conventional_safe_immediately_before_first_event_rate':float(np.mean(both)) if both else None,
            'valid_action_immediately_before_first_event_rate':float(np.mean(valid)) if valid else None,
            'selected_macro_histogram_before_first_event':dict(sorted(macros.items())),
            'fallback_reason_histogram_before_first_event':dict(sorted(reasons.items())),
        }
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cowp',required=True)
    ap.add_argument('--guard')
    ap.add_argument('--conventional')
    ap.add_argument('--planner')
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); base=_load(a.cowp)
    out={'schema_version':'cowp_v16_8_27_waymax_physical_compare_v2','cowp_failure_localization':_failure_localization(base)}
    for name,path in [('fallback_outcome',a.guard),('conventional_safety',a.conventional),('planner_score_only',a.planner)]:
        if path:
            obj=_load(path); out[f'cowp_vs_{name}']=_paired(base,obj); out[f'{name}_failure_localization']=_failure_localization(obj)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
