"""Conservative trial-level CARLA secondary-safety stress audit.

A collision is a POSITIVE *simulator* witness only if the exact follower B--C
pair collides under one task-matched ego action, the alternate action doesn't,
and the no-merge cruise negative control under the SAME response family is
collision-free.  Time ordering is only a mediator proxy, never causal proof.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from collections import Counter
from .evidence import read_jsonl
from .science import Protocol,validate_pair,same_initial,same_ego_execution

PAIRS=(('early_slow','late_slow'),('mild_early','mild_late'),('speed_early','speed_late'))


def _pair_event(row):
    m=row.get('metrics') or {}
    if not m.get('non_ego_pair_collision'):return False
    pair={'follower_b','follower_c'}
    events=row.get('collision_events') or []
    # Sensor event with exact physical actor roles. An OBB overlap is not enough.
    return any({q.get('role_a'),q.get('role_b')}==pair for q in events)


def _evidence_time_order(case,cruise,*,speed_delta=0.3,hold_s=.15):
    """B change before C change vs SAME-family cruise, exploratory only."""
    first_B=first_C=None
    tr=case.get('trace',[]);base=cruise.get('trace',[])
    if len(tr)!=len(base) or not tr:return {'status':'INVALID_TRACE_ALIGNMENT'}
    dt=float(tr[1]['t'])-float(tr[0]['t']) if len(tr)>1 else .05
    frames=max(1,round(hold_s/dt));b_run=c_run=0
    for x,y in zip(tr,base):
        if abs(float(x['t'])-float(y['t']))>1e-4:return {'status':'INVALID_TRACE_TIME'}
        if x['t']<1:continue
        b_delta=float(y['follower_b_speed_mps'])-float(x['follower_b_speed_mps'])
        c_delta=float(y['follower_c_speed_mps'])-float(x['follower_c_speed_mps'])
        b_run=(b_run+1 if b_delta>=speed_delta else 0)
        c_run=(c_run+1 if c_delta>=speed_delta else 0)
        if b_run>=frames and first_B is None:first_B=round(float(x['t'])-(frames-1)*dt,3)
        if c_run>=frames and first_C is None:first_C=round(float(x['t'])-(frames-1)*dt,3)
    return {'status':'EXPLORATORY_ONLY_NO_MEDIATION_IDENTIFICATION',
            'first_B_extra_slow_t':first_B,'first_C_extra_slow_t':first_C,
            'B_before_C_proxy':bool(first_B is not None and first_C is not None and first_B<first_C)}


def analyze(panel):
    rows=read_jsonl(panel)
    manifest=next(x for x in rows if x.get('type')=='manifest')
    if manifest.get('archetype')!='chain_cut_in':raise ValueError('Four-car chain required')
    index={}
    for row in rows:
        if row.get('type')!='trial':continue
        k=(str(row['anchor']),str(row['seed']),row['family'],row['profile'])
        if k in index:raise ValueError('Duplicate outcome key')
        index[k]=row
    results=[]
    contexts=sorted({(a,s,f) for a,s,f,_ in index})
    for anchor,seed,family in contexts:
        base=index.get((anchor,seed,family,'cruise'))
        for a,b in PAIRS:
            x=index.get((anchor,seed,family,a));y=index.get((anchor,seed,family,b))
            row={'anchor':anchor,'seed':seed,'family':family,'pair':[a,b],
                 'classification':'UNKNOWN', 'reason':''}
            if base is None or x is None or y is None:
                row['reason']='missing_cruise_or_action';results.append(row);continue
            valid,reason,progress=validate_pair(x,y,Protocol())
            row['ego_progress_diff_m']=progress
            if not valid:
                row['reason']='invalid_task_pair:'+reason;results.append(row);continue
            if not (base.get('metrics') or {}).get('valid') or base.get('ego_collision'):
                row['reason']='invalid_cruise_negative_control';results.append(row);continue
            bstate,reason=same_initial(base,x,Protocol())
            cstate,reason2=same_initial(base,y,Protocol())
            if not bstate or not cstate:
                row['reason']='cruise_initial_state_mismatch:'+reason+','+reason2
                results.append(row);continue
            if _pair_event(base):
                row['classification']='BACKGROUND_BC_COLLISION_NOT_EGO_ATTRIBUTABLE'
                results.append(row);continue
            # Reject a numerical collision flag not backed by role-identified physical sensors.
            if any((r.get('metrics') or {}).get('non_ego_pair_collision') and not _pair_event(r)
                   for r in (x,y)):
                row['reason']='collision_flag_without_exact_role_sensor_event'
                results.append(row);continue
            events={a:_pair_event(x),b:_pair_event(y)}
            row['measured_pair_collisions']=events
            row['B_to_C_temporal_proxy_by_action']={a:_evidence_time_order(x,base),
                                                    b:_evidence_time_order(y,base)}
            row['min_BC_TTC_s']={a:(x.get('metrics') or {}).get('min_rear_pair_ttc_s'),
                                  b:(y.get('metrics') or {}).get('min_rear_pair_ttc_s')}
            if events[a] != events[b]:
                row['classification']='AVOIDABLE_SIMULATED_BC_COLLISION_CANDIDATE'
                row['bad_ego_action']=a if events[a] else b
                row['safe_ego_alternative']=b if events[a] else a
                row['reason']='Still requires B-mediator intervention and independently calibrated C response'
            else:
                row['classification']='NO_ACTION_DEPENDENT_BC_COLLISION' if not events[a] else 'BOTH_ACTIONS_BC_COLLISION'
            results.append(row)
    # SAME-CONTROLLER baseline (TM B; BehaviorAgent C) versus delayed/capped
    # BehaviorAgent C. Comparing to all-TM would confound policy identity
    # with latency; all-TM is only a contextual traffic-family baseline.
    # A valid controlled comparison requires *B as well as ego* remain
    # nearly identical; changing C behavior can feed back through CARLA.
    parity=[]
    for anchor,seed,_ in contexts:
        for action in ('cruise','early_slow','late_slow','mild_early','mild_late',
                       'speed_early','speed_late'):
            a=index.get((anchor,seed,'mixed_b_tm_c_behavior',action))
            b=index.get((anchor,seed,'mixed_b_tm_c_delayed_behavior',action))
            if a is None or b is None:continue
            init,init_why=same_initial(a,b,Protocol())
            ego,ego_why=same_ego_execution(a,b)
            ta,tb=a.get('trace',[]),b.get('trace',[])
            if len(ta)!=len(tb) or not ta:
                b_equal=False;distance=None;speed_delta=None
            else:
                distance=max(math.dist(x['xy_by_role']['follower_b'],y['xy_by_role']['follower_b'])
                             for x,y in zip(ta,tb))
                speed_delta=max(abs(float(x['follower_b_speed_mps'])-
                                    float(y['follower_b_speed_mps'])) for x,y in zip(ta,tb))
                b_equal=distance<=.5 and speed_delta<=.75
            parity.append({'anchor':anchor,'seed':seed,'profile':action,
                 'initial_match':init,'initial_reason':init_why,
                 'ego_execution_match':ego,'ego_reason':ego_why,
                 'B_execution_match':b_equal,
                 'max_B_xy_difference_m':distance,'max_B_speed_difference_mps':speed_delta,
                 'controlled_C_policy_effect_identifiable_in_CARLA':init and ego and b_equal})
    # Every key in contexts can repeat across response families.
    parity=list({(x['anchor'],x['seed'],x['profile']):x for x in parity}.values())
    counts=Counter(r['classification'] for r in results)
    return {'protocol':'V14_CARLA_PHYSICAL_STRESS_NOT_HUMAN_GROUND_TRUTH',
            'manifest_snapshot':{'map':manifest.get('map'),
             'anchor_protocol_version':manifest.get('anchor_protocol_version'),
             'c_delay_s':manifest.get('chain_c_control_delay_s'),
             'c_brake_command_cap':manifest.get('chain_c_brake_command_cap'),
             'response_families':manifest.get('families')},
            'summary':{'pairs_checked':len(results),'status_counts':dict(counts),
             'independent_corridors_with_simulated_avoidable_collision':len({r['anchor'] for r in results if r['classification']=='AVOIDABLE_SIMULATED_BC_COLLISION_CANDIDATE'})},
            'results':results,'same_controller_C_vs_lagged_C_parity':parity,
            'limits':['Controlled C delay is a stress condition, NOT a validated human driver.',
                      'Cruise is a no-merge negative control, NOT an equal-task substitute.',
                      'Use mixed_b_tm_c_behavior (not all-TM) to isolate the lag intervention.',
                      'B-before-C time sequence cannot identify B as a causal mediator.',
                      'Do not report a hypothetical stopping-margin violation as a collision.',
                      'A single collision witness does not establish systematic failure or strong planner deficit.']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    j=analyze(args.panel);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:f.write(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(j['summary'],ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
