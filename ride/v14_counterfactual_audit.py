"""Independent *read-only* forensic audit of RIDE v13 CARLA model transfer.

This module replays existing decision audits from frozen source/target JSONL,
counts each candidate pair only once, checks physical acceleration plausibility,
and probes continuous-burden/sustained-brake sensitivity.  No post-hoc threshold
is an additional prospective success.  No model is human counterfactual truth.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
from collections import Counter

from .evidence import read_jsonl
from .v13_model_transfer import evaluate_lock, sha256


def _finite(x):
    v=float(x)
    if not math.isfinite(v):
        raise ValueError('nonfinite trace measurement')
    return v


def burdens(trace, *, role='follower_c', comfort=2.0, cap=None, median_window=1,
            after_s=1.0):
    """Integrate comfort deceleration exceedance; cap and median are diagnostics."""
    if median_window<1 or median_window % 2 != 1:
        raise ValueError('median window must be positive and odd')
    ts=[_finite(f['t']) for f in trace]
    vals=[_finite(f[role+'_accel']) for f in trace]
    if not ts or len(ts)<2: raise ValueError('short trace')
    intervals=[t-s for s,t in zip(ts,ts[1:])]
    if any(d<=0 or d>0.2 for d in intervals):raise ValueError('invalid timebase')
    if max(intervals)-min(intervals)>0.002:raise ValueError('nonuniform timebase')
    dt=statistics.median(intervals)
    if median_window>1:
        h=median_window//2
        vals=[statistics.median(vals[max(0,i-h):i+h+1]) for i in range(len(vals))]
    if cap is not None:
        if cap<=0:raise ValueError('nonpositive acceleration cap')
        vals=[max(-cap,min(cap,v)) for v in vals]
    return round(sum(max(0.,-a-comfort)*dt for a,t in zip(vals,ts) if t>=after_s),6)


def sustained_s(trace, *, role='follower_c', braking=4.0, after_s=1.0):
    if braking<=0: raise ValueError('threshold must be positive')
    vals=[(_finite(f['t']),_finite(f[role+'_accel'])) for f in trace]
    if len(vals)<2: raise ValueError('short trace')
    dt=statistics.median([vals[i][0]-vals[i-1][0] for i in range(1,len(vals))])
    best=run=0
    for t,a in vals:
        run=run+1 if t>=after_s and a< -braking else 0
        best=max(best,run)
    return round(best*dt,3)


def trace_integrity(trace):
    if not trace:return {'valid':False,'reason':'missing_trace'}
    try:
        accel=[];phys_diff=[]
        for f in trace:
            if _finite(f['t'])<1.0:continue
            a=_finite(f['follower_c_accel']);accel.append((f['t'],a))
            v=(f.get('physics_forward_accel_by_role') or {}).get('follower_c')
            if v is not None:phys_diff.append(abs(a-_finite(v)))
        outliers=[{'t':round(t,3),'long_accel_mps2':round(a,3)} for t,a in accel if abs(a)>10.0]
        return {'valid':True, 'poststartup_abs_accel_gt_10mps2':outliers,
                'poststartup_extreme_accel_count':len(outliers),
                'min_c_accel_mps2':round(min(a for _,a in accel),3),
                'max_speed_derivative_vs_physics_accel_diff_mps2':(round(max(phys_diff),3) if phys_diff else None),
                'brake_4mps2_max_continuous_s':sustained_s(trace),
                'brake_3_5mps2_max_continuous_s':sustained_s(trace,braking=3.5),
                'brake_4_5mps2_max_continuous_s':sustained_s(trace,braking=4.5)}
    except (ValueError,KeyError,TypeError) as exc:
        return {'valid':False,'reason':str(exc)}


def _index(panel):
    out={}
    for row in read_jsonl(panel):
        if row.get('type')=='trial':
            key=(str(row['anchor']),str(row['seed']),row['profile'])
            if key in out:raise ValueError('Duplicate arm '+repr(key))
            out[key]=row
    return out


def one_root(root):
    root=Path(root)
    choices=json.loads((root/'locks/source_model_choices.json').read_text())
    report=json.loads((root/'v13_target_decision_audit.json').read_text())
    frozen=root/'freeze/frozen/chain_cut_in_anchors.jsonl'
    source=root/'source/carla/chain_cut_in.jsonl'
    target=root/'target/carla/chain_cut_in.jsonl'
    commit=json.loads((root/'locks/source_model_choices.commit.json').read_text())
    if sha256(frozen)!=commit['frozen_sha256'] or sha256(source)!=commit['source_sha256']:
        raise ValueError('Source/frozen SHA256 no longer matches prospectively committed data')
    if sha256(root/'locks/source_model_choices.json')!=commit['decision_sha256']:
        raise ValueError('Choice lock SHA256 mismatch')
    recalculated=evaluate_lock(choices,read_jsonl(source),read_jsonl(target))
    if recalculated['summary']!=report['summary']:
        raise ValueError('Published v13 audit does not reproduce on raw source/target data')
    raw=_index(target)
    src={ (str(x['anchor']),str(x['seed']),tuple(x['pair']),x['policy']):x
          for x in choices['records'] }
    cases=[]
    statuses=Counter()
    selected=[r for r in report['results'] if r['scientific_status']!='UNKNOWN'
              and r['policy']=='SOURCE_THIRD_PARTY']
    for r in selected:
        anchor=str(r['anchor']);seed=str(r['seed']);a,b=r['pair']
        lock=src[(anchor,seed,tuple(r['pair']),r['policy'])]
        rows=(raw.get((anchor,seed,a)),raw.get((anchor,seed,b)))
        if None in rows:raise ValueError('Missing target arm for valid decision')
        cost0,cost1=rows[0]['trace'],rows[1]['trace']
        selected_action=r['selected_action'];alternative=b if selected_action==a else a
        srcdiff=(lock['source_scores'][a]['c_burden_mps']-
                 lock['source_scores'][b]['c_burden_mps'])
        targetdiff=(r['target_scores'][a]['c_burden_mps']-
                    r['target_scores'][b]['c_burden_mps'])
        robustness={}
        for method,param in (('raw',None),('cap6',6.),('cap5',5.),('median5',5)):
            if method=='median5':
                va,vb=burdens(cost0,median_window=param),burdens(cost1,median_window=param)
            else:va,vb=burdens(cost0,cap=param),burdens(cost1,cap=param)
            robustness[method]={a:va,b:vb,'chosen_minus_alternative':round(
                va-vb if selected_action==a else vb-va,6)}
        stability={}
        for threshold in (3.5,4.0,4.5):
            va,vb=sustained_s(cost0,braking=threshold),sustained_s(cost1,braking=threshold)
            stability[str(threshold)]={a:va,b:vb}
        merge={}
        for action,row in zip((a,b),rows):
            merge[action]=next((f['t'] for f in row['trace']
                                if f.get('ego_in_target_lane')),None)
        ccase={
          'anchor':anchor,'seed':seed,'pair':[a,b], 'selected_action':selected_action,
          'alternative':alternative, 'decision_status':r['scientific_status'],
          'model_error_reported':bool(r.get('paired_model_induced_selection_error')),
          'source_burden_diff_a_minus_b':round(srcdiff,6),
          'target_burden_diff_a_minus_b':round(targetdiff,6),
          'counterfactual_sensitivity_error':round(targetdiff-srcdiff,6),
          'response_amplitude_ratio':round(abs(targetdiff)/(abs(srcdiff)+.05),3),
          'target_primary_by_action':{x:r['target_scores'][x]['primary'] for x in (a,b)},
          'target_min_BC_ttc_by_action':{x:r['target_scores'][x]['min_BC_ttc_s'] for x in (a,b)},
          'ego_progress_m':{x:r['target_scores'][x]['ego_progress_m'] for x in (a,b)},
          'raw_trajectory_sensitivity_not_a_replacement_of_preregistered_metric':robustness,
          'brake_duration_threshold_sensitivity':stability,
          'trace_physics':{a:trace_integrity(cost0),b:trace_integrity(cost1)},
          'ego_first_target_lane_t_s':merge,
          'no_collision_claim_verified':all(not row.get('ego_collision') and
                                            not row.get('non_ego_pair_collision') for row in rows),
        }
        ccase['poststartup_extreme_c_accel']=(sum(v['poststartup_extreme_accel_count']
            for v in ccase['trace_physics'].values() if v['valid'])>0)
        ccase['caveat'] = ('KINEMATIC_OUTLIER: cannot treat this as a validated realistic braking event'
                          if ccase['poststartup_extreme_c_accel'] else
                          'CARLA conditional outcome only; C braking is not proven B-mediated')
        cases.append(ccase)
        statuses[r['scientific_status']]+=1
    status=report['summary']
    return {'root':root.name,'source_family':report['source_family'],
            'target_family':report['target_family'],
            'original_report_summary':status,
            'unique_valid_action_pairs':len(selected),
            'unique_pairs_with_model_errors':sum(c['model_error_reported'] for c in cases),
            'cases_with_extreme_acceleration':sum(c['poststartup_extreme_c_accel'] for c in cases),
            'status_by_unique_pair':dict(statuses),'cases':cases,
            'limits':['Same-model placebo on another road/seed is NOT paired repeatability.',
                      'Posthoc acceleration caps and median filtering are sensitivity analyses, not new precommitted endpoints.',
                      'B and C braking time order does not identify a causal mediated effect.',
                      'The source selector is a full CARLA model lookahead, not GameFormer or RACP.',
                      'No human counterfactual response is observed.']}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--roots',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(argv)
    reports=[one_root(p) for p in a.roots]
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    result={'protocol':'V14_READONLY_POSTHOC_SENSITIVITY_AUDIT',
            'reports':reports,'caveat':'This does not provide independent human validation or prove a third-party collision.'}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps([{'root':x['root'],'unique_valid_action_pairs':x['unique_valid_action_pairs'],
         'model_errors':x['unique_pairs_with_model_errors'], 'extreme_acceleration_cases':x['cases_with_extreme_acceleration']}
        for x in reports],indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
