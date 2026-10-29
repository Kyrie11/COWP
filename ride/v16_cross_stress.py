"""V16 independent CARLA evidence audit for response latency, no-op caps and decisions.

Reads completed CARLA JSONL, a directory containing it, or an uploaded ZIP.
Outputs only descriptive simulator evidence: no synthetic ground-truth, no
claim of human causality, no planner baseline impersonation. The unit is a
fixed anchor, not an individual action, family, seed, or selector row.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
import zipfile
from pathlib import Path
from collections import Counter
from .evidence import read_jsonl
from .science import Protocol, same_initial, same_ego_execution, validate_pair
from .v13_model_transfer import extract_trial_cost

ACTIONS=('cruise','early_slow','late_slow','mild_early','mild_late','speed_early','speed_late')
NORMAL='mixed_b_tm_c_behavior'
PERTURBED='mixed_b_tm_c_delayed_behavior'


def load_panel(path):
    p=Path(path)
    if p.is_dir():
        matches=list(p.rglob('carla/chain_cut_in.jsonl'))
        if len(matches)!=1: raise ValueError(f'Expected one complete CARLA panel under {p}; found {len(matches)}')
        rows=read_jsonl(matches[0]); identity=str(matches[0])
    elif p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            matches=[n for n in z.namelist() if n.endswith('/carla/chain_cut_in.jsonl') and
                     '/source/' not in n and '/target/' not in n]
            if len(matches)!=1:raise ValueError('Expected one completed stress panel in ZIP: '+str(matches))
            rows=[json.loads(x) for x in z.read(matches[0]).decode('utf-8').splitlines() if x.strip()]
        identity=str(p)
        _check_completed(rows)
    else:
        rows=read_jsonl(p);identity=str(p)
    m=next(x for x in rows if x.get('type')=='manifest')
    if m.get('archetype')!='chain_cut_in':raise ValueError('Four-car chain required')
    index={}
    for r in rows:
        if r.get('type')!='trial':continue
        key=(str(r['anchor']),str(r['seed']),r['family'],r['profile'])
        if key in index:raise ValueError('Duplicate trial key: '+str(key))
        index[key]=r
    return dict(path=identity, manifest=m, trials=index, rows=rows)


def _check_completed(rows):
    manifests=[r for r in rows if r.get('type')=='manifest']
    endings=[r for r in rows if r.get('type')=='run_end']
    trials=[r for r in rows if r.get('type')=='trial']
    if len(manifests)!=1 or manifests[0].get('simulator')!='CARLA' or len(endings)!=1 or\
       rows[-1].get('type')!='run_end' or endings[0].get('completed') is not True or\
       endings[0].get('trial_count')!=len(trials):
        raise ValueError('Incomplete CARLA panel: cannot use for science')


def _bc_sensor_event(row):
    return any({e.get('role_a'),e.get('role_b')} == {'follower_b','follower_c'}
               for e in (row.get('collision_events') or []))


def _max_brake(row,role='follower_c'):
    values=[float(t['controls_by_role'][role]['brake']) for t in row.get('trace') or []]
    return max(values) if values else None


def _b_parity(a,b):
    ta=a.get('trace') or [];tb=b.get('trace') or []
    if not ta or len(ta)!=len(tb):return False,None,None
    try:
        pos=max(math.dist(x['xy_by_role']['follower_b'],y['xy_by_role']['follower_b'])
                for x,y in zip(ta,tb))
        speed=max(abs(float(x['follower_b_speed_mps'])-float(y['follower_b_speed_mps']))
                  for x,y in zip(ta,tb))
    except (ValueError,KeyError,TypeError):return False,None,None
    return pos<=.5 and speed<=.75,round(pos,4),round(speed,4)


def _safe_cost(row):
    try:return extract_trial_cost(row)
    except (KeyError,TypeError,ValueError):return None


def action_contrast_diagnostic(panel,*,tie=.05,meaningful=.25):
    """RETROSPECTIVE contrast check, not an ex-ante choice or original planner."""
    rows=panel['trials']
    contexts=sorted({(a,s) for a,s,_,_ in rows})
    out=[]
    for anchor,seed in contexts:
        for a,b in (('early_slow','late_slow'),('mild_early','mild_late'),
                    ('speed_early','speed_late')):
            arms=[]
            for fam in (NORMAL,PERTURBED):
                x=rows.get((anchor,seed,fam,a));y=rows.get((anchor,seed,fam,b))
                if x is None or y is None:break
                valid,reason,prog=validate_pair(x,y,Protocol())
                if not valid:break
                cx=_safe_cost(x);cy=_safe_cost(y)
                if not cx or not cy:break
                arms.append((x,y,cx,cy))
            if len(arms)!=2:continue
            sx,sy,csx,csy=arms[0];tx,ty,ctx,cty=arms[1]
            checks=[(sx,tx),(sy,ty)]
            matched=all(same_initial(x,y,Protocol())[0] and
                        same_ego_execution(x,y)[0] and _b_parity(x,y)[0]
                        for x,y in checks)
            if not matched:continue
            ds=round(csx['c_burden_mps']-csy['c_burden_mps'],6)
            dt=round(ctx['c_burden_mps']-cty['c_burden_mps'],6)
            chosen=a if csx['ego_progress_m']>=csy['ego_progress_m'] else b
            regret=(ctx['c_burden_mps']-cty['c_burden_mps']) if chosen==a else (cty['c_burden_mps']-ctx['c_burden_mps'])
            out.append({'anchor':anchor,'seed':seed,'pair':[a,b],
                'source_C_burden_contrast_mps':ds,
                'target_C_burden_contrast_mps':dt,
                'source_burden_tie':abs(ds)<=tie,
                'target_meaningful_C_distinction':abs(dt)>=meaningful,
                'false_tie_exploratory':abs(ds)<=tie and abs(dt)>=meaningful,
                'hypothetical_progress_tiebreak':chosen,
                'posthoc_choice_C_burden_regret_mps':round(max(0,regret),6),
                'WARNING':'RETROSPECTIVE ONLY. Actual choice must be locked before target rollout.'})
    return out


def summarize_panel(p):
    manifest=p['manifest']; trials=list(p['trials'].values())
    valid=[r for r in trials if (r.get('metrics') or {}).get('valid')]
    ttcs=[float(r['metrics']['min_rear_pair_ttc_s']) for r in valid
          if r['metrics'].get('min_rear_pair_ttc_s') is not None]
    normal=[r for r in trials if r['family']==NORMAL]
    normal_c_max=[_max_brake(r) for r in normal if _max_brake(r) is not None]
    cap=float(manifest.get('chain_c_brake_command_cap',1.0))
    delay=float(manifest.get('chain_c_control_delay_s',0.0))
    hits=sum(maxbrake>cap+1e-5 for maxbrake in normal_c_max)
    reports=[r.get('v16_C_brake_control_actuation') for r in trials
             if r['family']==PERTURBED and r.get('v16_C_brake_control_actuation') is not None]
    logged_actuation={'reporting_trials':len(reports),
                      'C_command_samples':sum(int(x['command_samples']) for x in reports),
                      'cap_binding_ticks':sum(int(x['command_cap_binding_ticks']) for x in reports),
                      'delay_changed_brake_ticks':sum(int(x['delay_changed_command_ticks']) for x in reports),
                      'status':'NO_V16_ACTUATION_LOG_IN_HISTORICAL_PANEL' if not reports else 'MEASURED_COMMAND_ACTUATION'}
    valid_arms=[]; parity=[]
    pairs={(a,s) for a,s,_,_ in p['trials']}
    for anchor,seed in sorted(pairs):
        for action in ACTIONS:
            x=p['trials'].get((anchor,seed,NORMAL,action))
            y=p['trials'].get((anchor,seed,PERTURBED,action))
            if x is None or y is None:continue
            initial,reason=same_initial(x,y,Protocol())
            ego,why=same_ego_execution(x,y)
            b_equal,bpos,bspeed=_b_parity(x,y)
            cx=_safe_cost(x);cy=_safe_cost(y)
            ok=bool(initial and ego and b_equal and cx and cy)
            outcome={'anchor':anchor,'seed':seed,'action':action,'eligible':ok,
                     'initial_match':initial,'ego_execution_match':ego,'B_execution_match':b_equal,
                     'max_B_pos_diff_m':bpos,'max_B_speed_diff_mps':bspeed,
                     'reason':('' if ok else reason if not initial else why if not ego else
                              'B_execution_mismatch' if not b_equal else 'cost_unavailable'),
                     'normal_C_burden_mps':cx['c_burden_mps'] if cx else None,
                     'modified_C_burden_mps':cy['c_burden_mps'] if cy else None,
                     'C_burden_delta_mps':round(cy['c_burden_mps']-cx['c_burden_mps'],6) if ok else None,
                     'normal_C_max_brake_cmd':_max_brake(x),'modified_C_max_brake_cmd':_max_brake(y)}
            # Never elevate cruise to a task-matched alternative. It is only
            # an exogenous negative control of the follower policy.
            parity.append(outcome)
            if ok and action!='cruise':valid_arms.append(outcome)
    deltas=[x['C_burden_delta_mps'] for x in valid_arms]
    mmax=max(normal_c_max) if normal_c_max else None
    if cap>=1.0:
        cap_state='NOT_REQUESTED'
    elif normal_c_max and hits==0:
        cap_state='UNTRIGGERED_IN_NORMAL_C_CONTROLLER; DO_NOT_ATTRIBUTE_EFFECT_TO_CAP'
    elif normal_c_max:
        cap_state='POTENTIALLY_BINDING; VERIFY_RAW_REQUEST_VS_APPLIED_COMMAND'
    else:
        cap_state='UNKNOWN_NO_NORMAL_C_CONTROL_TRACES'
    return {'input':p['path'],'protocol':str(manifest.get('anchor_protocol_version')),
            'seed':manifest.get('seed'),'map':manifest.get('map'),
            'fixed_anchor_contexts':len({(a,s) for a,s,_,_ in p['trials']}),
            'C_delay_requested_s':delay,'C_brake_cap_requested':cap,
            'trial_count':len(trials),'valid_trial_count':len(valid),
            'mechanism_executed_trial_count':sum(bool((r.get('metrics') or {}).get('mechanism_executed')) for r in valid),
            'B_C_physical_collision_trials':sum(_bc_sensor_event(r) for r in trials),
            'collision_flag_without_BC_sensor_support':sum(bool((r.get('metrics') or {}).get('non_ego_pair_collision')) and not _bc_sensor_event(r) for r in trials),
            'ego_collision_trials':sum(bool((r.get('metrics') or {}).get('ego_collision')) for r in trials),
            'C_sustained_brake_event_trials':sum((r.get('metrics') or {}).get('follower_c_max_sustained_brake_s',0)>=.2 for r in valid),
            'dangerous_BC_TTC_below_1_5_s_trials':sum(t<1.5 for t in ttcs),
            'minimum_observed_BC_TTC_s':min(ttcs) if ttcs else None,
            'normal_C_max_brake_command':mmax,
            'normal_C_actions_with_brake_command_above_cap':hits,
            'cap_actuation_status':cap_state,
            'v16_logged_C_command_actuation':logged_actuation,
            'normal_vs_modified_C_parity':parity,
            'retrospective_candidate_contrasts':action_contrast_diagnostic(p),
            'eligible_noncruise_action_count':len(valid_arms),
            'eligible_C_burden_differences_mps':deltas,
            'absolute_C_burden_differences_ge_0_25_count':sum(abs(x)>=.25 for x in deltas),
            'largest_abs_C_burden_difference_mps':max(map(abs,deltas),default=None)}


def _frozen_signature(p):
    m=p['manifest'];result=[]
    for a in m.get('anchors') or []:
        coords=a.get('coords') or {}
        result.append((str(a.get('anchor')),tuple(tuple(round(float(v),3) for v in coords.get(k,[])) for k in
                                                   ('ego','follower_a','follower_b','follower_c'))))
    return (m.get('map'),m.get('seed'),tuple(sorted(result)))


def compare_cells(data):
    roots=[(k,v) for k,v in data.items() if v is not None]
    if not roots:return {}
    ref_name,ref=roots[0]
    deviations=[]
    for name,p in roots[1:]:
        consistent=(_frozen_signature(ref)==_frozen_signature(p))
        for key,x in sorted(ref['trials'].items()):
            y=p['trials'].get(key)
            if y is None:continue
            initial,reason=same_initial(x,y,Protocol())
            ego,why=same_ego_execution(x,y)
            b_ok,bpos,bspeed=_b_parity(x,y)
            cx=_safe_cost(x);cy=_safe_cost(y)
            ok=bool(consistent and initial and ego and b_ok and cx and cy)
            deviations.append({'from':ref_name,'to':name,'anchor':key[0],'seed':key[1],
                               'family':key[2],'action':key[3],
                               'same_frozen_geometry':consistent,
                               'eligible':ok,
                               'reason':('' if ok else 'frozen_identity_mismatch' if not consistent else
                                         reason if not initial else why if not ego else
                                         'B_trajectory_mismatch' if not b_ok else 'no_valid_C_cost'),
                               'C_burden_delta_mps':round(cy['c_burden_mps']-cx['c_burden_mps'],6) if ok else None})
    return deviations


def locked_summary(path):
    if path is None:return None
    p=Path(path)
    if p.is_dir():
        f=p/'v13_target_decision_audit.json'
        j=json.loads(f.read_text())
    elif p.suffix=='.zip':
        with zipfile.ZipFile(p) as z:
            found=[n for n in z.namelist() if n.endswith('/v13_target_decision_audit.json')]
            if len(found)!=1:raise ValueError('No unique locked choice audit')
            j=json.loads(z.read(found[0]))
    else:j=json.loads(p.read_text())
    results=j['results']
    uniq={ (str(r['anchor']),str(r['seed']),tuple(r['pair']))
           for r in results if r.get('status')=='SELECTED' and
           r.get('scientific_status') not in ('UNKNOWN',)}
    wrong={ (str(r['anchor']),str(r['seed']),tuple(r['pair']))
            for r in results if r.get('scientific_status')=='AVOIDABLE_C_BURDEN_LOSS'
            and r.get('policy')=='SOURCE_THIRD_PARTY'}
    return {'summary_as_recorded':j.get('summary',{}),'unique_evaluable_action_pairs':len(uniq),
            'unique_avoidable_controlled_policy_C_burden_action_pairs':len(wrong),
            'avoidable_pairs':[{'anchor':a,'seed':s,'actions':list(t)} for a,s,t in sorted(wrong)],
            'control_change_is_not_model_misspecification':True}


def analyze(panels,locked=None):
    summaries={k:summarize_panel(v) for k,v in panels.items() if v is not None}
    comparison=compare_cells({k:v for k,v in panels.items() if v is not None})
    prior=panels.get('baseline');cap=panels.get('cap');delay=panels.get('delay');both=panels.get('both')
    notes=[]
    if cap and summaries['cap']['cap_actuation_status'].startswith('UNTRIGGERED'):
        notes.append('cap=0.85 experiment did not bind C BehaviorAgent brake commands; cannot infer brake capacity effect')
    if all(x['B_C_physical_collision_trials']==0 for x in summaries.values()):
        notes.append('No measured B-C physical collisions across these completed panels')
    if prior and delay:
        notes.append('Action records and fixed road count are not independent scenes; cluster by physical corridor')
    return {'protocol':'V16_DESCRIPTIVE_CARLA_RESPONSE_FIDELITY_AND_DECISION_AUDIT',
            'panels':summaries,'cross_cell_matched_arms':comparison,
            'locked':locked_summary(locked), 'notes':notes,
            'scientific_gates':{'controlled_C_delay_affects_burden':'OBSERVED_IN_TWO_CARLA_CORRIDORS_NOT_HUMAN_CALIBRATED' if
                 delay and summaries['delay']['absolute_C_burden_differences_ge_0_25_count'] else 'UNKNOWN',
                'C_brake_cap_causes_risk':'NOT_TESTED_IF_CAP_UNTRIGGERED',
                'avoidable_BC_collision':'NOT_OBSERVED',
                'B_mediates_C_response':'NOT_IDENTIFIED',
                'actual_strong_planner_failure':'NOT_TESTED',
                'human_counterfactual_validity':'NOT_ESTABLISHED'},
            'limitations':['A deliberate C-only latency intervention is not an independent human ground truth.',
                 'No-op control-family comparisons estimate nonzero simulator reset noise.',
                 'B-before-C temporal order does not prove causal mediation.',
                 'Only independent scenes, not action/seed/family records, support population statistics.',
                 'No failure claim for GameFormer, RACP, HYPE or other planners.']}


def preflight_cap(panel,requested_cap):
    p=load_panel(panel)
    c=[_max_brake(r) for r in p['trials'].values() if r['family']==NORMAL]
    c=[v for v in c if v is not None]
    if not c:raise ValueError('Normal C policy trace missing')
    n=sum(v>requested_cap+1e-5 for v in c)
    return {'reference':p['path'],'cap':requested_cap,'C_normal_policy_max_observed_brake':max(c),
            'eligible_arms_with_brake_exceeding_cap':n,'total_normal_C_arms':len(c),
            'status':('INACTIVE_CAP_DO_NOT_USE_AS_LIMITED_BRAKING_EXPERIMENT' if n==0 and requested_cap<1
                      else 'CAP_CAN_BIND_BUT_MUST_VERIFY_ONLINE_REQUEST_VS_APPLIED' if requested_cap<1
                      else 'NO_CAP_REQUESTED'),
            'caveat':'Observation of requested command is not validation of maximum physical deceleration.'}


def actuation_gate(panel,allow_noop=False):
    s=summarize_panel(load_panel(panel))
    info=s['v16_logged_C_command_actuation']
    delay=s['C_delay_requested_s'];cap=s['C_brake_cap_requested']
    if info['reporting_trials']==0:
        status='UNVERIFIED_NO_V16_RAW_C_COMMAND_LOG'
    elif delay==0 and cap==1:
        status=('IDENTITY_NEGATIVE_CONTROL_VERIFIED' if info['delay_changed_brake_ticks']==0 and
                info['cap_binding_ticks']==0 else 'FAILED_IDENTITY_CONTROL_CHANGED_BRAKE')
    elif delay>0 and info['delay_changed_brake_ticks']==0:
        status='FAILED_REQUESTED_BRAKE_DELAY_NEVER_CHANGED_COMMAND'
    elif cap<1 and info['cap_binding_ticks']==0:
        status='INTENTIONAL_INACTIVE_CAP_PLACEBO' if allow_noop else 'FAILED_BRAKE_CAP_NEVER_BOUND'
    else:status='ACTUATION_VERIFIED_IN_CARLA_COMMAND_LOG'
    return {'status':status,'requested_delay_s':delay,'requested_cap':cap,
            'command_actuation':info,
            'not_a_physical_deceleration_or_human_calibration_proof':True}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('report');a.add_argument('--baseline');a.add_argument('--cap');a.add_argument('--delay');a.add_argument('--both');a.add_argument('--locked');a.add_argument('--output',required=True)
    p=sub.add_parser('cap-preflight');p.add_argument('--panel',required=True);p.add_argument('--cap',type=float,required=True);p.add_argument('--output')
    ag=sub.add_parser('actuation-gate');ag.add_argument('--panel',required=True);ag.add_argument('--allow-inactive-placebo',action='store_true');ag.add_argument('--output')
    args=ap.parse_args(argv)
    if args.cmd=='actuation-gate':
        result=actuation_gate(args.panel,args.allow_inactive_placebo)
        out=Path(args.output) if args.output else None
    elif args.cmd=='report':
        d={k:load_panel(v) if v else None for k,v in [('baseline',args.baseline),('cap',args.cap),('delay',args.delay),('both',args.both)]}
        result=analyze(d,args.locked);out=Path(args.output)
    else:
        if not math.isfinite(args.cap) or not .15<=args.cap<=1:ap.error('cap must be 0.15..1')
        result=preflight_cap(args.panel,args.cap)
        out=Path(args.output) if args.output else None
    if out:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result if args.cmd!='report' else {'panels':{k:{
        x:v[x] for x in ('trial_count','valid_trial_count','minimum_observed_BC_TTC_s',
         'B_C_physical_collision_trials','cap_actuation_status','eligible_noncruise_action_count',
         'absolute_C_burden_differences_ge_0_25_count')} for k,v in result['panels'].items()},
        'locked':result.get('locked'), 'scientific_gates':result['scientific_gates']},
        indent=2,ensure_ascii=False))
    if args.cmd=='cap-preflight' and result['status'].startswith('INACTIVE_CAP'):return 4
    if args.cmd=='actuation-gate' and result['status'].startswith(('FAILED','UNVERIFIED')):return 5
    return 0

if __name__=='__main__':raise SystemExit(main())
