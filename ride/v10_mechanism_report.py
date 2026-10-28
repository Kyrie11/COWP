"""Decisional-mechanism audit for frozen CARLA paired action panels.

The PRIMARY RQ1 endpoints remain the preregistered tuple collision/brake/TTC;
continuous burden and speed-response diagnostics are explicitly EXPLORATORY.
No human counterfactual inference, learned-planner error, or RQ3 claim is made.

Usage:
  python -m ride.v10_mechanism_report --input path/cut_in.jsonl --output path/mechanism_v10.json
"""
from __future__ import annotations
import argparse,collections,json,math
from pathlib import Path
from ride.science import Protocol, same_initial, same_ego_execution,validate_pair,assess_pair
from ride.carla.mechanism_audit import _treatment_time, _burden, _sustained_start


def _time_window(trace, begin):
    return [x for x in trace if float(x.get('t',-1))>=max(1.0,float(begin))]


def _risk_key(row,p):
    m=row.get('metrics') or {}
    ttc=m.get('min_rear_pair_ttc_s')
    try:small=ttc is not None and math.isfinite(float(ttc)) and float(ttc)<p.ttc_s
    except (ValueError,TypeError):small=False
    return [int(bool(m.get('non_ego_pair_collision'))),int(bool(m.get('hard_brake'))),int(bool(small))]


def _lexicographic_order(a,b):
    return int(tuple(a) > tuple(b)) - int(tuple(a) < tuple(b))


def response_delta_against_cruise(row,cruise):
    """Matched initial state, fixed absolute post-command time and frozen roles."""
    p=Protocol()
    m=row.get('metrics') or {}; c=cruise.get('metrics') or {}
    ok,why=same_initial(row,cruise,p)
    if not (m.get('valid') and c.get('valid') and ok):
        return {'eligible':False,'reason':why if not ok else 'invalid_action_or_cruise'}
    ts=row.get('trace') or []; baseline=cruise.get('trace') or []
    if not ts or not baseline or len(ts)!=len(baseline):
        return {'eligible':False,'reason':'missing_or_unpaired_trace'}
    t0=_treatment_time(row,ts)
    if t0 is None:
        return {'eligible':False,'reason':'no_verified_lateral_command'}
    for frame1,frame0 in zip(ts,baseline):
        if abs(float(frame1['t'])-float(frame0['t'])) > 1e-3:
            return {'eligible':False,'reason':'unmatched_simulation_timestamps'}
    post=[(a,b) for a,b in zip(ts,baseline) if float(a['t'])>=max(1.0,t0)]
    before=[(a,b) for a,b in zip(ts,baseline) if 1.0 <= float(a['t']) < t0]
    if not post:
        return {'eligible':False,'reason':'empty_post_action_window'}
    # An unobserved simulator/control-state drift *before* the ego intervention
    # cannot be attributed to the intervention. Match the pre-command executed
    # history, not just the first timestamp's pose. Fixed a priori audit
    # tolerances; do not tune them against third-party outcomes.
    if not before:
        return {'eligible':False,'reason':'no_pre_action_diagnostic_window'}
    responder_roles=['follower_b'] + (['follower_c'] if row.get('archetype')=='chain_cut_in' else [])
    max_pre_ego_xy=max(math.dist(a['xy_by_role']['ego'],b['xy_by_role']['ego']) for a,b in before)
    max_pre_responder_xy=max(math.dist(a['xy_by_role'][role],b['xy_by_role'][role])
        for a,b in before for role in responder_roles)
    max_pre_responder_dv=max(abs(float(a[f'{role}_speed_mps'])-float(b[f'{role}_speed_mps']))
        for a,b in before for role in responder_roles)
    pre_diagnostics={'pre_action_max_ego_xy_m':round(max_pre_ego_xy,5),
                     'pre_action_max_responder_xy_m':round(max_pre_responder_xy,5),
                     'pre_action_max_responder_speed_diff_mps':round(max_pre_responder_dv,5)}
    if max_pre_ego_xy>0.75 or max_pre_responder_xy>1.0 or max_pre_responder_dv>0.75:
        return {'eligible':False,'reason':'pre_action_trajectory_mismatch',**pre_diagnostics}
    out={'eligible':True,**pre_diagnostics,'lateral_command_s':t0,
         'first_target_lane_s':next((float(x['t']) for x in ts if x.get('ego_in_target_lane')),None),
         'baseline_cruise_has_hard_brake':bool(c.get('hard_brake')),
         'max_follower_b_speed_delta_post_mps':max(abs(float(a['follower_b_speed_mps'])-float(b['follower_b_speed_mps'])) for a,b in post),
         'mean_abs_follower_b_speed_delta_post_mps':round(sum(abs(float(a['follower_b_speed_mps'])-float(b['follower_b_speed_mps'])) for a,b in post)/len(post),5),
         'mean_abs_follower_b_speed_delta_pre_mps':round(sum(abs(float(a['follower_b_speed_mps'])-float(b['follower_b_speed_mps'])) for a,b in before)/len(before),5) if before else None,
         'post_command_incremental_burden_b_mps':round(_burden([x for x,_ in post],'follower_b') - _burden([x for _,x in post],'follower_b'),5),
         'full_horizon_incremental_burden_b_mps':round(_burden(ts,'follower_b')-_burden(baseline,'follower_b'),5),
         'candidate_b_brake_s':_sustained_start(ts,'follower_b'),
         'cruise_b_brake_s':_sustained_start(baseline,'follower_b'),
         'primary_pair_roles':m.get('primary_third_party_pair_roles')}
    if row.get('archetype')=='chain_cut_in':
        out['post_command_incremental_burden_c_mps']=round(
            _burden([x for x,_ in post],'follower_c') - _burden([x for _,x in post],'follower_c'),5)
        out['max_follower_c_speed_delta_post_mps']=max(abs(float(a['follower_c_speed_mps'])-
            float(b['follower_c_speed_mps'])) for a,b in post)
        out['cruise_primary_pair_collision']=bool(c.get('non_ego_pair_collision'))
    return out


def audit(rows):
    manifests=[r for r in rows if r.get('type')=='manifest']
    trials=[r for r in rows if r.get('type')=='trial' and not r.get('joint_targets') and not r.get('recovery')]
    end=[r for r in rows if r.get('type')=='run_end']
    finished=(len(manifests)==1 and len(end)==1 and end[0].get('completed') is True and
        end[0].get('trial_count')==len(trials))
    p=Protocol()
    lookup={};dupes=[]
    for r in trials:
        key=(r.get('archetype'),str(r.get('anchor')),str(r.get('seed')),r.get('family'),r.get('profile'))
        if key in lookup:dupes.append(key)
        lookup[key]=r
    valid=sum(r.get('metrics',{}).get('valid') is True for r in trials)
    mechanism=sum(r.get('metrics',{}).get('valid') is True and r.get('metrics',{}).get('mechanism_executed') is True for r in trials)
    primary_pairs=sum(r.get('metrics',{}).get('non_ego_pair_collision') is True and
        r.get('metrics',{}).get('valid') is True for r in trials)
    # Speed-protocol diagnostics are not a replacement for physical safety
    # endpoints. The cap is a controller parameter, not an observed-speed guarantee.
    speed_by_family={}
    nominal_reference=float(manifests[0].get('cruise_mps',8.0)) if manifests else 8.0
    for r in trials:
        fam=r.get('family')
        speeds=r.get('max_observed_follower_speed_mps') or {}
        if fam not in speed_by_family:
            speed_by_family[fam]={'trials_with_observed_speed':0,'trials_exceed_nominal_by_over_1p5_mps':0,
                                  'maximum_observed_responder_speed_mps':None}
        if not speeds:continue
        vs=[float(v) for v in speeds.values() if v is not None]
        if not vs:continue
        out=speed_by_family[fam]
        out['trials_with_observed_speed']+=1
        out['trials_exceed_nominal_by_over_1p5_mps']+=int(max(vs)>nominal_reference+1.5)
        out['maximum_observed_responder_speed_mps']=round(max(
            max(vs),out['maximum_observed_responder_speed_mps'] or 0.0),4)
    comparisons=[];negative_controls=[]
    contexts=sorted({k[:3] for k in lookup},key=str)
    for ctx in contexts:
        armkeys=[k for k in lookup if k[:3]==ctx]
        families=sorted({k[3] for k in armkeys if k[3] in p.families})
        for family in families:
            cruise=lookup.get((*ctx,family,'cruise'))
            if cruise is None:continue
            for action in ('early_slow','late_slow','mild_early','mild_late','merge_early','merge_late'):
                target=lookup.get((*ctx,family,action))
                if target is None:continue
                negative_controls.append({'context':[ *ctx,family ], 'action':action,
                    **response_delta_against_cruise(target,cruise)})
        for a,b in p.contrast_actions:
            per_family={}; cross_errors=[]; rows_for_fams={}
            for family in p.families:
                ra=lookup.get((*ctx,family,a));rb=lookup.get((*ctx,family,b))
                if ra is None or rb is None:
                    per_family[family]={'eligible':False,'reason':'missing_action_arm'}
                    continue
                rows_for_fams[family]=(ra,rb)
                result=assess_pair(ra,rb,p)
                per_family[family]={'eligible':bool(result['eligible']),
                    'reason':result.get('reason'),
                    'progress_difference_m':result.get('delta_progress_m'),
                    'primary_endpoint_a':_risk_key(ra,p),
                    'primary_endpoint_b':_risk_key(rb,p),
                    'primary_risk_order':_lexicographic_order(_risk_key(ra,p),_risk_key(rb,p)) if result['eligible'] else None,
                    'continuous_burden_b_delta_mps':round(_burden(ra['trace'],'follower_b') -
                                                    _burden(rb['trace'],'follower_b'),5) if result['eligible'] else None}
                if result['eligible'] and ctx[0]=='chain_cut_in':
                    per_family[family]['continuous_burden_c_delta_mps']=round(
                        _burden(ra['trace'],'follower_c')-_burden(rb['trace'],'follower_c'),5)
            both=all(per_family.get(f,{}).get('eligible',False) for f in p.families)
            if both:
                for action_idx,action in enumerate((a,b)):
                    ra=rows_for_fams[p.families[0]][action_idx]
                    rb=rows_for_fams[p.families[1]][action_idx]
                    ok,why=same_initial(ra,rb,p)
                    if not ok:cross_errors.append(action+':'+why)
                    ok,why=same_ego_execution(ra,rb)
                    if not ok:cross_errors.append(action+':'+why)
            matched=both and not cross_errors
            sign1=per_family.get(p.families[0],{}).get('primary_risk_order')
            sign2=per_family.get(p.families[1],{}).get('primary_risk_order')
            comparisons.append({'context':list(ctx),'actions':[a,b], 'families':per_family,
                'cross_family_eligible':matched,'cross_family_errors':cross_errors,
                'strict_primary_endpoint_reversal':bool(matched and sign1 in (-1,1) and sign1==-sign2)})
    covered={r['context'][1] for r in comparisons if r['cross_family_eligible']}
    matches=sum(r['cross_family_eligible'] for r in comparisons)
    reversals=sum(r['strict_primary_endpoint_reversal'] for r in comparisons)
    return {'protocol':'RIDE_V10_CARLA_SIMULATED_SCREENING','complete':finished,'duplicate_keys':list(map(list,dupes)),
        'counts':{'trials':len(trials),'valid_trials':valid,'executed_valid_mechanisms':mechanism,
            'valid_primary_pair_collisions':primary_pairs,'task_matched_cross_family_pairs':matches,
            'independent_physical_anchors_with_cross_family_pairs':len(covered),
            'strict_primary_endpoint_ranking_reversals':reversals},
        'comparisons':comparisons,'cruise_negative_controls':negative_controls,
        'exploratory_observed_speed_by_family':speed_by_family,
        'RQ1':'NOT_ESTABLISHED' if reversals==0 else 'EXPLORATORY_REVERSAL_REQUIRES_HOLDOUT',
        'RQ2':'NOT_ESTABLISHED_REQUIRES_MATCHED_AVOIDABLE_EFFECT_AND_INDEPENDENT_REPLICATION',
        'RQ3':'NOT_TESTED_NO_LOCKED_PLANNER_CHOICES',
        'scope_limits':['Simulated intervention, not human causal ground truth',
            'Posthoc outcome ordering, not pre-rollout planner decision',
            'Continuous burden and cruise relative deltas are exploratory diagnostics',
            'The cruise no-merge control is not a task-matched RQ2 alternative',
            'An ineligible trial cannot become a positive causal witness',
            'One corridor under multiple policies/seeds is not multiple independent physical scenes']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    rows=[];broken=0
    for line in args.input.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except ValueError:broken+=1
    result=audit(rows)
    result['broken_json_lines']=broken
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'counts':result['counts'],'complete':result['complete'],
        'RQ1':result['RQ1'],'RQ2':result['RQ2']},ensure_ascii=False,indent=2))
    return 0 if result['complete'] and broken==0 and not result['duplicate_keys'] else 4

if __name__=='__main__':raise SystemExit(main())
