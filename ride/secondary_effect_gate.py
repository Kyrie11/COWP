"""Pre-declared v8 secondary endpoint: continuous third-party braking burden.

Always retain the original collision/brake/TTC primary endpoints. This diagnostic
checks whether admissible *rollout* action pairs become distinguishable when a
continuous metric is used; it is NOT a planning choice or causal human truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .evidence import read_jsonl
from .ranking import audit_rankings
from .science import Protocol, same_initial
from .carla.mechanism_audit import summarize_one

BURDEN_MARGIN_MPS = 0.25  # fixed v8 secondary threshold, not tuned on outcomes


def _sign(d:float, margin:float):
    if d > margin:return 1
    if d < -margin:return -1
    return 0


def audit_secondary(rows:list[dict], *,margin:float=BURDEN_MARGIN_MPS):
    if margin<=0:raise ValueError('Positive prespecified burden margin required')
    ranking=audit_rankings(rows)
    trials={}
    for r in rows:
        if r.get('type')!='trial' or r.get('family') not in ('tm','behavior') or r.get('joint_targets') or r.get('recovery'):
            continue
        key=(str(r.get('anchor')),str(r.get('seed')),r.get('family'),r.get('profile'))
        if key in trials:raise ValueError('Duplicate identity '+str(key))
        trials[key]=r
    out=[]
    for rr in ranking['all_pairs']:
        if not rr['eligible']:continue
        anchor,seed=rr['anchor'],rr['seed']
        a,b=rr['actions']
        byfamily={};cruise_control={};signs={}
        for f in ('tm','behavior'):
            ta=trials[(anchor,seed,f,a)];tb=trials[(anchor,seed,f,b)]
            sa=summarize_one(ta);sb=summarize_one(tb)
            diff=round(sa['burden_b_mps']-sb['burden_b_mps'],6)
            ctrl=trials.get((anchor,seed,f,'cruise'))
            if ctrl:
                sc=summarize_one(ctrl)
                parity,why=same_initial(ta,ctrl,Protocol())
                ctrl_info={'valid':bool(sc['valid']),'same_initial_as_candidate_a':parity,
                           'initial_parity_reason':None if parity else why,
                           'control_burden_b_mps':sc['burden_b_mps'],
                           'control_sustained_brake_b':sc['first_sustained_brake_b_s'] is not None,
                           'candidate_a_extra_burden_vs_control_mps':round(sa['burden_b_mps']-sc['burden_b_mps'],6),
                           'candidate_b_extra_burden_vs_control_mps':round(sb['burden_b_mps']-sc['burden_b_mps'],6)}
            else:ctrl_info={'available':False}
            byfamily[f]={'action_a_burden_b_mps':sa['burden_b_mps'],
                         'action_b_burden_b_mps':sb['burden_b_mps'],
                         'difference_a_minus_b_mps':diff,
                         'action_a_brake_b_time_s':sa['first_sustained_brake_b_s'],
                         'action_b_brake_b_time_s':sb['first_sustained_brake_b_s'],
                         'action_a_lateral_command_s':sa['first_lateral_command_s'],
                         'action_b_lateral_command_s':sb['first_lateral_command_s']}
            signs[f]=_sign(diff,margin);cruise_control[f]=ctrl_info
        out.append({'anchor':anchor,'seed':seed,'actions':[a,b],
                    'burden_signs':signs,
                    'strict_secondary_order_reversal':signs['tm']*signs['behavior']==-1,
                    'burden_tie_or_noncritical':0 in signs.values(),
                    'by_family':byfamily,'cruise_negative_controls':cruise_control,
                    'scope':'EXPLORATORY_V8_SECONDARY_RESPONSE_SENSITIVITY_ONLY'})
    return {'protocol':{'secondary_metric':'follower_b_braking_integral_above_3mps2_after_t1s',
                        'margin_mps':margin,'negative_control':'cruise (not a task-matched alternative)',
                        'primary_endpoints_unchanged':True},
            'summary':{'eligible_primary_gate_pairs':len(out),
                       'strict_secondary_order_reversals':sum(x['strict_secondary_order_reversal'] for x in out),
                       'nonzero_secondary_in_at_least_one_family':sum(any(v!=0 for v in x['burden_signs'].values()) for x in out)},
            'pairs':out,
            'scientific_status':'NOT_RQ1_PROOF_UNLESS_INDEPENDENTLY_REPLICATED_AND_PLANNER_CHOICE_PRELOCKED',
            'limitations':['Burden metric is a simulator control/comfort proxy, not a human risk label',
                           'Even an admissible matched response flip is a simulator outcome ranking, not an ex ante planner error']}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--margin',type=float,default=BURDEN_MARGIN_MPS)
    a=ap.parse_args(argv)
    v=audit_secondary(read_jsonl(a.input),margin=a.margin)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(v['summary'],indent=2))
if __name__=='__main__':main()
