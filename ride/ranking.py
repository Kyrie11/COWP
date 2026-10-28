"""Independent-response *evaluation* ranking sensitivity, not planner performance.

A risk-order reversal here is a SIMULATOR phenomenon obtained after complete
rollouts. It is not evidence a particular online planner would choose either
trajectory, and is not a calibrated causal effect in human driving.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path
from .evidence import read_jsonl
from .science import Protocol, same_initial, same_ego_execution, validate_pair


def risk_key(row, ttc_s=1.5):
    m = row.get('metrics') or {}
    try: ttc=float(m['min_rear_pair_ttc_s']) if m.get('min_rear_pair_ttc_s') is not None else 9999.
    except (ValueError,TypeError):ttc=9999.
    return (int(bool(m.get('non_ego_pair_collision'))),
            int(bool(m.get('hard_brake'))),
            int(ttc < ttc_s))


def sign(a,b):
    return -1 if a < b else (1 if a > b else 0)


def audit_rankings(rows, protocol=None):
    p=protocol or Protocol()
    meta = next((r for r in rows if r.get('type')=='manifest'),None)
    if not meta or meta.get('simulator')!='CARLA':
        raise ValueError('Ranking audit requires CARLA manifest')
    by_context=collections.defaultdict(lambda:collections.defaultdict(dict))
    for r in rows:
        if (r.get('type')!='trial' or r.get('family') not in p.families or
            r.get('joint_targets') or r.get('recovery')):
            continue
        key=(r.get('archetype',meta.get('archetype','convoy')),
             str(r.get('anchor')),str(r.get('seed')))
        f=r['family'];act=r['profile']
        if act in by_context[key][f]:
            raise ValueError('Duplicate ranking identity: '+str((key,f,act)))
        by_context[key][f][act]=r
    outcomes=[]
    for context, fam in sorted(by_context.items()):
        if not all(f in fam for f in p.families):continue
        common=set.intersection(*(set(fam[f]) for f in p.families))
        for a,b in p.contrast_actions:
            if a not in common or b not in common:continue
            reasons=[];metrics={}
            # Ensure BOTH arms in EACH family are valid and same initial states
            # and task matched. Verify cross-family parity, not just within model.
            for f in p.families:
                x,y=fam[f][a],fam[f][b]
                ok, why, _ = validate_pair(x, y, p)
                if not ok:reasons.append(f+':'+why)
                metrics[f]={'a':risk_key(x,p.ttc_s),'b':risk_key(y,p.ttc_s)}
            left,right=p.families[0],p.families[1]
            for action in (a,b):
                ok,why=same_initial(fam[left][action],fam[right][action],p)
                if not ok:reasons.append('cross_family:'+action+':'+why)
                # v5/v6 physically executed CARLA trials must track near-identical
                # ego interventions across response families. Synthetic fixtures
                # and legacy datasets without this recorded protocol are excluded
                # from the additional v6 gate rather than silently retroactively
                # upgrading their evidence.
                if str(meta.get('anchor_protocol_version','')).startswith(('ride_lane_corridor_v5_',
                                                                            'ride_lane_corridor_v6_',
                                                                            'ride_lane_corridor_v8_',
                                                                            'ride_lane_corridor_v9_')):
                    exec_ok,exec_why=same_ego_execution(fam[left][action],fam[right][action])
                    if not exec_ok:reasons.append('cross_family:'+action+':'+exec_why)
            ranks={f:sign(metrics[f]['a'],metrics[f]['b']) for f in p.families}
            eligible=not reasons
            outcomes.append({'archetype':context[0],'anchor':context[1],'seed':context[2],
                             'actions':[a,b],'eligible':eligible,'reasons':reasons,
                             'risk_keys':{f:{k:list(v) for k,v in m.items()} for f,m in metrics.items()},
                             'order_signs':ranks,
                             'strict_reverse':eligible and ranks[left]*ranks[right]==-1,
                             'preference_tie_mismatch':eligible and ranks[left]!=ranks[right] and 0 in ranks.values()})
    valid=[r for r in outcomes if r['eligible']]
    reverse=[r for r in valid if r['strict_reverse']]
    ties=[r for r in valid if r['preference_tie_mismatch']]
    return {'summary':{'candidate_pairs_checked':len(outcomes),
                       'task_matched_cross_model_pairs':len(valid),
                       'primary_pairs_preregistered':[list(a) for a in p.contrast_actions],
                       'strict_risk_order_reversals':len(reverse),
                       'tie_preference_mismatches':len(ties),
                       'posthoc_outcome_ranking_not_actual_planner_choice':True,
                       'third_party_endpoints_lexicographic':['third_party_pair_collision','hard_braking_proxy','short_ttc_proxy'],
                       'human_behavior_calibration_not_established':True},
            'strict_reversals':reverse,'tie_mismatches':ties,'all_pairs':outcomes}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(argv)
    output=audit_rankings(read_jsonl(args.input))
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(output['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
