"""CARLA-trace-conditioned one-dimensional stopping-margin stress screen.

This is an ANALYTICAL hypothetical B-brakes-now scenario, not a CARLA crash,
not an RSS safety guarantee, and not calibrated human reaction statistics.
Use to choose *predeclared* CARLA counterfactual stress cells, then run physics.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from .evidence import read_jsonl
from .science import Protocol,validate_pair


def required_gap(v_rear,v_front,*,reaction_s,b_rear,b_front,standstill_m=2.):
    """Net bumper gap to avoid overlap if B brakes immediately and C reacts late.

    Constant decelerations; no steering, no grade, no brake buildup time.
    Negative returned required gap is floored at standstill. Never use as proof
    of safety or of real-world crash frequency.
    """
    vals=(v_rear,v_front,reaction_s,b_rear,b_front,standstill_m)
    if any(not math.isfinite(float(z)) for z in vals) or min(v_rear,v_front,reaction_s,standstill_m)<0 or b_rear<=0 or b_front<=0:
        raise ValueError('Invalid stopping-distance assumptions')
    stop_rear=v_rear*reaction_s+v_rear**2/(2*b_rear)
    stop_front=v_front**2/(2*b_front)
    return max(standstill_m,standstill_m+stop_rear-stop_front)


def screen_trace(trace,*,reaction_s,b_rear,b_front,standstill_m=2.,
                 after_t=1.0,require_b_deceleration=1.5,after_merge=True):
    candidates=[]
    for f in trace:
        if float(f['t'])<after_t:continue
        if after_merge and not f.get('ego_in_target_lane'):continue
        if not f.get('rear_pair_ttc_applicable'):continue
        if float(f['follower_b_accel']) > -require_b_deceleration:continue
        gap=f.get('rear_pair_gap_m')
        if gap is None:continue
        vC=float(f['follower_c_speed_mps']);vB=float(f['follower_b_speed_mps'])
        g=required_gap(vC,vB,reaction_s=reaction_s,b_rear=b_rear,b_front=b_front,
                       standstill_m=standstill_m)
        margin=float(gap)-g
        candidates.append({'t':round(float(f['t']),3), 'gap_m':round(float(gap),3),
            'b_velocity_mps':round(vB,3),'c_velocity_mps':round(vC,3),
            'b_observed_accel_mps2':round(float(f['follower_b_accel']),3),
            'hypothetical_required_gap_m':round(g,3),'margin_m':round(margin,3)})
    if not candidates:return {'status':'NO_QUALIFYING_B_BRAKE_OBSERVATION','frames':0,'min_margin_m':None}
    worst=min(candidates,key=lambda x:x['margin_m'])
    return {'status':('HYPOTHETICAL_STOP_MARGIN_DEFICIT' if worst['margin_m']<0 else
                      'NO_STOP_MARGIN_DEFICIT_IN_SCREEN'),
            'frames':len(candidates), 'min_margin_m':worst['margin_m'],
            'worst_frame':worst}


def evaluate_panel(path,*,reaction_s,b_rear,b_front,standstill_m=2.):
    rows=read_jsonl(path)
    m=next(r for r in rows if r.get('type')=='manifest')
    if m['archetype']!='chain_cut_in':raise ValueError('Only chain_cut_in scenarios supported')
    indexed={}
    for r in rows:
        if r.get('type')=='trial':
            k=(str(r['anchor']),str(r['seed']),r['family'],r['profile'])
            if k in indexed:raise ValueError('Duplicated panel trial')
            indexed[k]=r
    arms=[]
    for (anchor,seed,fam,act),r in sorted(indexed.items()):
        valid=bool(r.get('metrics',{}).get('valid') and not r.get('metrics',{}).get('ego_collision'))
        outcome=(screen_trace(r.get('trace',[]),reaction_s=reaction_s,b_rear=b_rear,b_front=b_front,
                              standstill_m=standstill_m,after_merge=(act!='cruise'))
                 if valid else {'status':'INVALID_TRIAL','frames':0,'min_margin_m':None})
        arms.append({'anchor':anchor,'seed':seed,'family':fam,'profile':act,'valid':valid,
          'measured_C_sustained_brake_s':r.get('metrics',{}).get('follower_c_max_sustained_brake_s'),
          'measured_BC_collision':r.get('metrics',{}).get('non_ego_pair_collision'),
          'hypothetical_screen':outcome})
    by={(r['anchor'],r['seed'],r['family'],r['profile']):r for r in arms}
    signed=[]
    for anchor,seed,fam,_ in sorted({(x['anchor'],x['seed'],x['family'],'') for x in arms}):
        for aa,bb in (('early_slow','late_slow'),('mild_early','mild_late'),('speed_early','speed_late')):
            A=by.get((anchor,seed,fam,aa));B=by.get((anchor,seed,fam,bb))
            if A is None or B is None:continue
            srcA=indexed[(anchor,seed,fam,aa)];srcB=indexed[(anchor,seed,fam,bb)]
            valid,reason,progress_delta=validate_pair(srcA,srcB,Protocol())
            ma=A['hypothetical_screen']['min_margin_m'];mb=B['hypothetical_screen']['min_margin_m']
            signed.append({'anchor':anchor,'seed':seed,'family':fam,'pair':[aa,bb],
              'valid_task_matched':valid,'invalid_reason':reason,'ego_progress_delta_m':progress_delta,
              'margin_a_m':ma,'margin_b_m':mb,
              'margin_diff_a_minus_b_m':(round(ma-mb,3) if valid and ma is not None and mb is not None else None)})
    return {'protocol':'V14_ANALYTIC_BRAKING_ENVELOPE_NOT_OBSERVED_COLLISIONS',
       'assumptions':{'reaction_s':reaction_s,'c_max_decel_mps2':b_rear,
                      'b_hypothetical_decel_mps2':b_front,'standstill_clearance_m':standstill_m,
                      'fixed_1d_lane_no_steering':True,'no_proven_human_parameter_calibration':True},
       'panel':str(path),'summary':{'arms':len(arms),'task_matched_pairs':sum(x['valid_task_matched'] for x in signed),
           'measured_BC_collisions':sum(x['measured_BC_collision'] is True for x in arms),
           'hypothetical_deficit_arms':sum(x['hypothetical_screen']['status']=='HYPOTHETICAL_STOP_MARGIN_DEFICIT' for x in arms)},
       'candidate_arms':arms,'paired_margins':signed,
       'WARNING':'A negative analytic stopping margin is conditional stress vulnerability, NOT an observed CARLA or human collision.'}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--reaction-s',type=float,default=0.6)
    ap.add_argument('--c-brake-mps2',type=float,default=3.5)
    ap.add_argument('--b-brake-mps2',type=float,default=5.)
    ap.add_argument('--standstill-gap-m',type=float,default=2.)
    args=ap.parse_args(argv)
    report=evaluate_panel(args.panel,reaction_s=args.reaction_s,
       b_rear=args.c_brake_mps2,b_front=args.b_brake_mps2,standstill_m=args.standstill_gap_m)
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps(report['summary'],indent=2,ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
