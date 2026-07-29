#!/usr/bin/env python3
"""Select an OC-RSC candidate only after calibration and offline use gates pass."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any] | None:
    try:
        return json.load(path.open())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def method_result(path: Path) -> dict[str, Any] | None:
    d=load(path)
    return None if not d else (d.get('methods',{}) or {}).get('ocrap')


def direct_reasons(r: dict[str, Any] | None) -> int:
    return sum(int(v) for k,v in ((r or {}).get('selection_reason_counts',{}) or {}).items() if 'direct_value' in str(k))


def fv(r: dict[str, Any] | None, key: str, default: float) -> float:
    try: return float((r or {}).get(key, default))
    except (TypeError,ValueError): return default


def infer_offline(base: Path) -> Path:
    candidates=[Path(str(base)+'_v43_offline'), Path(str(base)+'_offline'), base.parent/(base.name+'_rsc_offline')]
    return next((p for p in candidates if p.exists()), candidates[0])


def find_eval(offline: Path, bucket: str) -> dict[str, Any] | None:
    for name in (f'eval_{bucket}_v43_v43.json', f'eval_{bucket}_v42_v43.json', f'eval_{bucket}_v43.json'):
        r=method_result(offline/name)
        if r is not None: return r
    matches=sorted(offline.glob(f'eval_{bucket}_*v43*.json'))
    return method_result(matches[0]) if matches else None


def evaluate(base: Path, offline: Path | None=None) -> tuple[list[str],float,dict[str,Any]]:
    offline=offline or infer_offline(base)
    failures=[]; score=0.0; diag={'base_run':str(base),'offline_run':str(offline)}
    for bucket in ('near','contact'):
        c=load(base/'calibration'/f'direct_value_risk_{bucket}_v43.json')
        if not c:
            failures.append(f'{bucket}: missing v43 risk calibration'); continue
        valid=bool(c.get('valid_for_deployment',False)); threshold=fv(c,'direct_value_threshold',float('inf'))
        verify=c.get('verify',{}) or {}; selected=int(verify.get('num_selected',0)); precision=verify.get('challenge_precision')
        harm_ucb=fv(verify,'harmful_group_exposure_ucb90',1.0)
        diag[f'{bucket}_cal']={'valid':valid,'threshold':threshold,'verify_selected':selected,'verify_precision':precision,'harmful_group_exposure_ucb90':harm_ucb,'warnings':c.get('warnings')}
        if not valid: failures.append(f'{bucket}: certificate is not valid_for_deployment')
        if not math.isfinite(threshold): failures.append(f'{bucket}: non-finite threshold')
        if selected < 2: failures.append(f'{bucket}: fewer than two held-out challenges')
        if precision is None or float(precision)<0.50: failures.append(f'{bucket}: held-out precision < 50%')
        if harm_ucb>0.05: failures.append(f'{bucket}: harmful group-exposure UCB > 5%')
        score += selected + 5.0*float(precision or 0.0) - 10.0*harm_ucb

    results={b:find_eval(offline,b) for b in ('safe','near_contact','contact')}
    if any(r is None for r in results.values()): failures.append('missing one or more v43 offline evaluations')
    safe,near,contact=results['safe'],results['near_contact'],results['contact']
    if safe:
        if fv(safe,'intervention_rate',1)>1e-12: failures.append('safe offline intervention is non-zero')
        if fv(safe,'bounded_NUP',0)<0.999: failures.append('safe offline NUP < 0.999')
    for name,r,nup_floor,int_cap in (('near',near,0.995,0.04),('contact',contact,0.985,0.08)):
        if r:
            if fv(r,'bounded_NUP',0)<nup_floor: failures.append(f'{name} offline NUP < {nup_floor}')
            if fv(r,'intervention_rate',1)>int_cap: failures.append(f'{name} offline intervention > {int_cap}')
            if direct_reasons(r)<=0: failures.append(f'{name}: offline selector never used direct-value certificate')
            score += 4.0*direct_reasons(r) + 10.0*(fv(r,'post_contact_deployability',0)-fv(r,'FRA_exec',1))
    diag['offline']={b:None if r is None else {'FRA':r.get('FRA_exec'),'DRS':r.get('DRS'),'NUP':r.get('bounded_NUP'),'PCD':r.get('post_contact_deployability'),'intervention':r.get('intervention_rate'),'direct_reasons':direct_reasons(r),'reason_counts':r.get('selection_reason_counts')} for b,r in results.items()}
    return failures,score,diag


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('base_runs',nargs='+',type=Path)
    ap.add_argument('--offline-runs',nargs='*',type=Path)
    ap.add_argument('--write-choice',type=Path)
    args=ap.parse_args()
    if args.offline_runs and len(args.offline_runs)!=len(args.base_runs):
        ap.error('--offline-runs must match base_runs count')
    passed=[]
    for i,base in enumerate(args.base_runs):
        offline=args.offline_runs[i] if args.offline_runs else None
        failures,score,diag=evaluate(base,offline)
        print(json.dumps(diag,ensure_ascii=False,indent=2))
        if failures:
            for x in failures: print(f'FAIL {base.name}: {x}')
        else:
            print(f'PASS {base.name}: score={score:.6f}'); passed.append((score,base))
    if not passed:
        print('RESULT: FAIL — do not enter Waymax closed loop'); return 2
    passed.sort(key=lambda x:(-x[0],str(x[1]))); chosen=passed[0][1]
    print(f'RESULT: PASS — chosen {chosen}')
    if args.write_choice:
        args.write_choice.parent.mkdir(parents=True,exist_ok=True); args.write_choice.write_text(str(chosen)+'\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
