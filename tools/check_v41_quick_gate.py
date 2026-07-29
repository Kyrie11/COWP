#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def load(p: Path):
    return json.load(open(p)) if p.exists() else None

def metric(d,k,default=None):
    try: return float(d.get(k,default))
    except Exception: return default

def direct_count(d):
    return sum(int(v) for k,v in (d.get('selection_reason_counts',{}) or {}).items() if 'direct_value' in str(k))

def no_op_interventions(d, threshold):
    bad=[]; total=0
    for scene in d.get('scenes',[]) or []:
        for x in scene.get('decisions',[]) or []:
            if str(x.get('selected_macro','nominal')).lower() in {'nominal','keep',''}: continue
            total+=1
            dev=float(x.get('selected_nominal_deviation') or 0.0)
            if dev < threshold: bad.append((x.get('scene_id'),x.get('step_index'),dev,x.get('selection_reason')))
    return total,bad

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run',type=Path); ap.add_argument('--base-run',type=Path,default=None); ap.add_argument('--min-deviation',type=float,default=0.002); args=ap.parse_args()
    root=args.run; failures=[]; warnings=[]; rows=[]
    # calibration diagnostics
    base=args.base_run if args.base_run is not None else root.parent / root.name.replace('_eval','').replace('_micro','').replace('_confirm12','').replace('_offline','') / 'calibration'
    # allow explicit evaluation directory whose BASE_RUN is elsewhere
    for b in ('near','contact'):
        candidates=list(root.glob(f'**/direct_value_advantage_{b}_v41.json'))+list(base.glob(f'direct_value_advantage_{b}_v41.json'))
        if candidates:
            c=load(candidates[0]); rows.append((f'cal_{b}_groups',c.get('num_calibration_groups')))
            if not math.isfinite(float(c.get('direct_value_additive_q',float('inf')))): failures.append(f'{b}: non-finite additive q')
            if int(c.get('num_calibration_groups',0))<30: warnings.append(f'{b}: fewer than 30 calibration groups')
            if float(c.get('challenge_rate') or 0.0)<=0.0: failures.append(f'{b}: calibrated value head has zero challenge opportunity')
            if c.get('negative_challenge_rate') is not None and float(c['negative_challenge_rate'])>0.10: failures.append(f'{b}: negative challenge rate > 10%')
    # offline safe
    safe=load(root/'eval_safe_v41_v41.json')
    if safe:
        r=(safe.get('methods',{}) or {}).get('ocrap',{})
        if float(r.get('intervention_rate',1.0))>1e-12: failures.append('safe offline intervention is non-zero')
        if float(r.get('bounded_NUP',0.0))<0.999: failures.append('safe offline NUP < 0.999')
    # paired micro audits
    for b in ('near_contact','contact'):
        v=load(root/f'audit_{b}_selected_topk_v41_v41.json'); s=load(root/f'audit_{b}_selected_topk_v41_scalar.json')
        if not v: continue
        total,bad=no_op_interventions(v,args.min_deviation)
        rows.extend([(f'{b}_scenes',v.get('num_scenes')),(f'{b}_pcd',v.get('closed_loop_post_contact_deployability')),(f'{b}_nup',v.get('closed_loop_bounded_NUP')),(f'{b}_direct_reasons',direct_count(v)),(f'{b}_interventions',total)])
        if bad: failures.append(f'{b}: {len(bad)} intervention(s) below actionability threshold')
        if metric(v,'closed_loop_bounded_NUP',0)<(0.995 if b=='near_contact' else 0.985): failures.append(f'{b}: NUP below quick gate')
        if metric(v,'intervention_rate',1)>(0.04 if b=='near_contact' else 0.06): failures.append(f'{b}: intervention rate too high')
        if total>0 and direct_count(v)==0: warnings.append(f'{b}: interventions occurred but none came from the v41 value path')
        if direct_count(v)==0: failures.append(f'{b}: OC-CAVA direct value path never entered final decisions')
        if s:
            vp=metric(v,'closed_loop_post_contact_deployability',0); sp=metric(s,'closed_loop_post_contact_deployability',0)
            if vp+0.01<sp: failures.append(f'{b}: PCD worse than paired scalar by >0.01')
            vr=metric(v,'closed_loop_audit_paper_selected_PCD_regret',1); sr=metric(s,'closed_loop_audit_paper_selected_PCD_regret',1)
            if vr>sr+0.01: failures.append(f'{b}: paper PCD regret worse than scalar by >0.01')
    print('V41 QUICK GATE')
    for k,v in rows: print(f'{k}: {v}')
    for x in warnings: print('WARNING:',x)
    for x in failures: print('FAIL:',x)
    if failures: print('RESULT: FAIL — do not expand'); return 2
    print('RESULT: PASS — proceed to 12-rollout confirmation, not publication scale'); return 0
if __name__=='__main__': raise SystemExit(main())
