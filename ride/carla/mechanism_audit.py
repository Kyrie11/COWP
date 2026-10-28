"""Exploratory CARLA interference checks for RIDE RQ1/RQ2.

Measures whether rear-vehicle braking is already present in the no-merge
control, plus brake timing relative to the first lateral target command and
physical target-lane entry. This is post-hoc diagnostics, NOT a replacement for
locked RQ1 strict ranking or task-matched RQ2 tests.

Only a paired independent response evaluation can provide *simulated* effects;
none of these metrics is human counterfactual ground truth.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
from ride.evidence import read_jsonl
from ride.science import Protocol, same_initial, validate_pair

DT = .05

def _sustained_start(frames, role: str, *, threshold: float = -4., n: int = 4):
    streak = 0
    first = None
    for x in frames:
        if float(x.get('t', 0)) < 1.:
            continue
        if x.get(f'{role}_accel', 0) <= threshold:
            if not streak:
                first = float(x['t'])
            streak += 1
            if streak >= n:
                return first
        else:
            streak, first = 0, None
    return None

def _burden(frames, role: str, *, threshold: float = 3.) -> float:
    """Integral of acceleration magnitude above a comfort threshold (m/s)."""
    ts = [x for x in frames if float(x.get('t', 0)) >= 1.]
    if not ts:
        return 0.
    dt = float(ts[1]['t'])-float(ts[0]['t']) if len(ts)>1 else DT
    return round(sum(max(0., -float(x.get(f'{role}_accel',0)) - threshold)*dt for x in ts), 5)

def _treatment_time(row, ts):
    if row.get('archetype') not in ('cut_in', 'support_gap'):
        return None
    for x in ts:
        dest=x.get('target_lane_id')
        # A target-lane waypoint is evidence of a commanded lateral action,
        # unlike merely joining the lane's geometry a second later.
        if dest is not None and x.get('ego_target_waypoint_lane_id') == dest and x.get('ego_lane_id') != dest:
            return float(x['t'])
    return None

def summarize_one(row):
    ts = row.get('trace') or []
    m = row.get('metrics') or {}
    target_enter = next((float(x['t']) for x in ts if x.get('ego_in_target_lane')), None)
    cmd_time = _treatment_time(row, ts)
    brake_a = _sustained_start(ts,'follower_a')
    brake_b = _sustained_start(ts,'follower_b')
    return {
        'archetype':row.get('archetype'), 'anchor':str(row.get('anchor')),
        'seed':str(row.get('seed')), 'family':row.get('family'), 'profile':row.get('profile'),
        'valid':m.get('valid') is True,'mechanism_executed':m.get('mechanism_executed') is True,
        'ego_collision':bool(m.get('ego_collision')),'third_party_pair_collision':bool(m.get('non_ego_pair_collision')),
        'ego_progress_m':m.get('ego_progress_m'),
        'first_lateral_command_s':cmd_time, 'first_ego_target_lane_s':target_enter,
        'first_sustained_brake_a_s':brake_a,'first_sustained_brake_b_s':brake_b,
        'burden_a_mps':_burden(ts,'follower_a'), 'burden_b_mps':_burden(ts,'follower_b'),
        'min_ttc_s':m.get('min_rear_pair_ttc_s'), 'reasons':m.get('reasons',[])
    }

def audit(rows: list[dict]):
    manifest = next((x for x in rows if x.get('type')=='manifest'), {})
    records = [x for x in rows if x.get('type')=='trial' and not x.get('recovery') and not x.get('joint_targets')]
    complete = bool(rows and rows[-1].get('type')=='run_end' and rows[-1].get('completed'))
    contexts=collections.defaultdict(dict)
    trial_out=[]
    for r in records:
        x = summarize_one(r)
        trial_out.append(x)
        ctx = (x['archetype'],x['anchor'],x['seed'],x['family'])
        if x['profile'] in contexts[ctx]:
            raise ValueError('Duplicate action in context '+str((ctx,x['profile'])))
        contexts[ctx][x['profile']] = (r,x)
    matched=[]
    comparisons=[]
    for ctx, arms in sorted(contexts.items()):
        base=arms.get('cruise')
        for name,(r,s) in arms.items():
            if name=='cruise' or not base:
                continue
            control,control_metrics=base
            same,why=same_initial(r,control,Protocol())
            diff=round(s['burden_b_mps']-control_metrics['burden_b_mps'],5)
            matched.append({**{k:v for k,v in zip(('archetype','anchor','seed','family'),ctx)},
                            'action':name,'same_initial_as_cruise':same,'initial_mismatch_reason':None if same else why,
                            'cruise_has_sustained_brake':control_metrics['first_sustained_brake_b_s'] is not None,
                            'delta_burden_b_mps_vs_cruise':diff,
                            'candidate_brake_b_s':s['first_sustained_brake_b_s'],
                            'lateral_command_s':s['first_lateral_command_s'],
                            'target_entry_s':s['first_ego_target_lane_s'],
                            'candidate_brake_before_lateral_command':(
                                s['first_sustained_brake_b_s'] is not None and
                                s['first_lateral_command_s'] is not None and
                                s['first_sustained_brake_b_s'] < s['first_lateral_command_s']),
                            'candidate_brake_before_full_entry':(
                                s['first_sustained_brake_b_s'] is not None and
                                s['first_ego_target_lane_s'] is not None and
                                s['first_sustained_brake_b_s'] < s['first_ego_target_lane_s']),
                            'interpretation':'CRUISE IS A MECHANISTIC NEGATIVE CONTROL, NOT A TASK-MATCHED RQ2 ALTERNATIVE'})
        p=Protocol()
        for a,b in p.contrast_actions:
            if a in arms and b in arms:
                r1,s1=arms[a];r2,s2=arms[b]
                eligible,why,_=validate_pair(r1,r2,p)
                eligible=eligible and s1['mechanism_executed'] and s2['mechanism_executed']
                comparisons.append({**{k:v for k,v in zip(('archetype','anchor','seed','family'),ctx)},
                                    'actions':[a,b],'eligible':eligible,
                                    'ineligible_reason':None if eligible else (why or 'mechanism_not_executed'),
                                    'delta_burden_b_mps_a_minus_b':round(s1['burden_b_mps']-s2['burden_b_mps'],5),
                                    'binary_brake_pair':[r1['metrics'].get('hard_brake'),r2['metrics'].get('hard_brake')],
                                    'WARNING':'EXPLORATORY scalar sensitivity only; not a preregistered RQ1 rank reversal'})
    c=collections.Counter()
    for x in trial_out:
        c['trials']+=1
        c['valid']+=int(x['valid'])
        c['mechanism_executed']+=int(x['mechanism_executed'])
        c['third_party_collision']+=int(x['third_party_pair_collision'])
        c['sustained_brake_b']+=int(x['first_sustained_brake_b_s'] is not None)
    c['matched_vs_cruise'] = sum(x['same_initial_as_cruise'] for x in matched)
    c['valid_task_matched_mechanism_pairs'] = sum(x['eligible'] for x in comparisons)
    c['cruise_rear_braking_trials'] = sum(x['profile']=='cruise' and x['first_sustained_brake_b_s'] is not None for x in trial_out)
    return {'complete':complete,'manifest_protocol':manifest.get('anchor_protocol_version'),
            'summary':dict(c), 'trials':trial_out,'cruise_negative_controls':matched,
            'task_matched_continuous_diagnostics':comparisons,
            'scientific_claim_allowed':False,
            'warnings':['Posthoc burden integral is a diagnostic, NOT a new primary RQ1/RQ2 endpoint',
                        'Braking before lane entry may still be caused by lateral approach; compare command onset and cruise',
                        'CARLA response models are not independently validated human behavior',
                        'A saturated binary brake flag cannot establish no continuous interaction effects']}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    result=audit(read_jsonl(args.input))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'complete':result['complete'],**result['summary']},indent=2,ensure_ascii=False))
    if not result['complete']:
        raise SystemExit('Incomplete JSONL: diagnostic saved, scientific conclusions forbidden')
if __name__=='__main__': main()
