"""Read-only v12 re-audit: threshold sensitivity is EXPLORATORY ONLY.

Use it to distinguish B response, C subsequent response, third-party hazard,
and ex-ante planner choice.  It does NOT re-label v11 pre-registered failures.
Physical corridor/seed correlations prevent treating comparisons as IID.
"""
from __future__ import annotations
import argparse, collections, json, math
from pathlib import Path
from ride.v11_chain_evidence import read_panel, audit_intervention, sustained_first
from ride.carla.mechanism_audit import _treatment_time

THRESHOLDS=((.1,.1),(.2,.2),(.3,.3),(.4,.3),(.5,.3))
ACTIONS=('early_slow','late_slow','mild_early','mild_late','speed_early','speed_late')


def evaluate(paths):
    panels=[];out=[];all_trials=0;good_trials=0;ex_count=0
    for path in paths:
        path=Path(path)
        meta, trials, complete, broken=read_panel(path)
        tby={(str(t.get('anchor')),str(t.get('seed')),t.get('family'),t.get('profile')):t
             for t in trials}
        errors=sum(bool(t.get('simulation_exception')) for t in trials)
        good=sum(bool((t.get('metrics') or {}).get('valid')) for t in trials)
        ex_count+=errors;all_trials+=len(trials);good_trials+=good
        panels.append({'file':str(path),'complete':complete,'broken_lines':broken,
                       'map':meta.get('map'),'anchors':len(meta.get('anchors') or []),
                       'trials':len(trials),'valid':good,'exceptions':errors,
                       'frozen_protocol':meta.get('anchor_protocol_version'),
                       'route_protocol':meta.get('behavior_route_protocol')})
        for (anchor,seed,family,profile),t in tby.items():
            if profile not in ACTIONS or family not in ('tm','behavior'):
                continue
            baseline=tby.get((anchor,seed,family,'cruise'))
            audit=(audit_intervention(t,baseline) if baseline else
                   {'eligible':False,'reason':'no_cruise_negative_control'})
            entry={'file':str(path),'map':meta.get('map'),'anchor':anchor,'seed':seed,
                   'family':family,'profile':profile,'eligible':bool(audit.get('eligible')),
                   'reason':audit.get('reason',''),
                   'primary_BC_collision':bool((t.get('metrics') or {}).get('non_ego_pair_collision')),
                   'observed_min_BC_ttc_s':(t.get('metrics') or {}).get('min_rear_pair_ttc_s')}
            if entry['eligible']:
                frames=t['trace'];base=baseline['trace']
                times=[float(x['t']) for x in frames]
                t0=1.5 if profile in ('speed_early','speed_late') else _treatment_time(t,frames)
                diffs={role:[float(y[f'{role}_speed_mps'])-float(x[f'{role}_speed_mps'])
                             for x,y in zip(frames,base)] for role in ('follower_b','follower_c')}
                # Comfort-exceedance effort for C (not B) is a distinct
                # third-party burden endpoint. It is POST-HOC on v11 data;
                # register any confirmatory comfort threshold prospectively.
                dt=float(times[1]-times[0])
                def effort(frs,role):
                    z=0.0
                    for x in frs:
                        if float(x['t'])>=max(1.0,t0):
                            acc=float(x[f'{role}_accel'])
                            if not math.isfinite(acc):raise ValueError('nonfinite_acceleration')
                            z+=max(0.,-acc-2.0)*dt
                    return z
                try:
                    c_effort=effort(frames,'follower_c')-effort(base,'follower_c')
                    b_effort=effort(frames,'follower_b')-effort(base,'follower_b')
                    effort_reason=''
                except (KeyError,TypeError,ValueError):
                    # Missing acceleration measurements are UNKNOWN, not zero.
                    c_effort=b_effort=None
                    effort_reason='MISSING_OR_INVALID_ACCELERATION_EVIDENCE'
                metrics=t.get('metrics') or {};bm=baseline.get('metrics') or {}
                def c_specific_hazard(mm):
                    ttc=mm.get('min_rear_pair_ttc_s')
                    ttc_hazard=(ttc is not None and math.isfinite(float(ttc)) and float(ttc)<1.5)
                    return bool(mm.get('non_ego_pair_collision')) or \
                           ('follower_c' in (mm.get('hard_brake_roles') or [])) or ttc_hazard
                c_hazard=c_specific_hazard(metrics)
                b_hazard=c_specific_hazard(bm)
                entries=[]
                for bth,cth in THRESHOLDS:
                    bfirst=sustained_first(times,diffs['follower_b'],bth,start=t0)
                    cfirst=sustained_first(times,diffs['follower_c'],cth,start=t0)
                    entries.append({'B_mps':bth,'C_mps':cth,'first_B_s':bfirst,'first_C_s':cfirst,
                       'B_then_C_proxy':(bfirst is not None and cfirst is not None and
                                         bfirst<=cfirst<=bfirst+3.)})
                entry.update({'t0_s':t0,'pre_BC_max_xy_m':audit.get('pre_BC_max_xy_m'),
                    'max_extra_B_speed_loss_mps':audit.get('max_extra_B_speed_loss_mps'),
                    'max_extra_C_speed_loss_mps':audit.get('max_extra_C_speed_loss_mps'),
                    'new_BC_primary_hazard_vs_cruise_v11_any_brake':audit.get('new_hazard_vs_cruise_not_task_matched'),
                    'new_C_specific_hazard_vs_cruise_exploratory':bool(c_hazard and not b_hazard),
                    'delta_C_comfort_exceedance_mps':round(c_effort,5) if c_effort is not None else None,
                    'delta_B_comfort_exceedance_mps':round(b_effort,5) if b_effort is not None else None,
                    'effort_evidence_unavailable_reason':effort_reason,
                    'sensitivity':entries})
            out.append(entry)
    stat=[]
    for bth,cth in THRESHOLDS:
        group=[]
        for family in ('tm','behavior'):
            xs=[x for x in out if x['eligible'] and x['family']==family]
            ix=THRESHOLDS.index((bth,cth))
            b=sum(x['sensitivity'][ix]['first_B_s'] is not None for x in xs)
            c=sum(x['sensitivity'][ix]['first_C_s'] is not None for x in xs)
            ordered=[x for x in xs if x['sensitivity'][ix]['B_then_C_proxy']]
            anchor_keys={(x['map'],x['anchor'],x['seed'],x['file']) for x in ordered}
            group.append({'family':family,'eligible_pairs_to_cruise':len(xs),
                          'B_response':b,'C_response':c,'B_before_C':len(ordered),
                          'within_file_anchor_contexts_with_proxy':len(anchor_keys)})
        stat.append({'threshold_B_mps':bth,'threshold_C_mps':cth,'by_family':group})
    return {'audit_type':'V12_POST_HOC_EXPLORATORY_THRESHOLD_SENSITIVITY',
       'primary_RQ1_endpoint':'UNCHANGED_AND_NOT_REDEFINED_BY_THIS_REPORT',
       'primary_RQ2':'NOT_ESTABLISHED_BY_TEMPORAL_ORDER_OR_CRUISE_CONTROL',
       'panels':panels,'counts':{'trial_rows':all_trials,'valid_trial_rows':good_trials,
          'simulation_exception_trial_rows':ex_count,'eligible_negative_controls':sum(x['eligible'] for x in out),
          'exploratory_new_C_specific_hazards_vs_cruise':sum(x.get('eligible') and x.get('new_C_specific_hazard_vs_cruise_exploratory') for x in out),
          'exploratory_C_effort_delta_over_0p1_mps':sum(x.get('eligible') and x.get('delta_C_comfort_exceedance_mps') is not None and x['delta_C_comfort_exceedance_mps']>0.1 for x in out),
          'verified_ex_ante_planner_choices':0},'threshold_sensitivity':stat,
       'records':out,'limitations':[
         'Post-hoc threshold sweep is hypothesis generation, never heldout confirmation.',
         'Velocity ordering does not prove B-mediated causal effects on C.',
         'cruise is NOT a progress-matched alternative ego action.',
         'Trial/arm/seed counts do not represent IID road corridors.',
         'Two CARLA agents are not independent human counterfactuals.',
         'No ex-ante planner selection or strong planner failures have been observed.']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    r=evaluate(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'counts':r['counts'],'thresholds':r['threshold_sensitivity']},indent=2))
    return 0 if all(x['complete'] for x in r['panels']) else 4

if __name__=='__main__':raise SystemExit(main())
