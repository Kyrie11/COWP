"""Summarize ALL precommitted CARLA cut-in parameter cells; never cherry-pick.

Discovery-only cell statistics. No collision-risk significance tests over seeds
that share the same physical corridor; no translation into human causality.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ride.evidence import read_jsonl
from ride.ranking import audit_rankings
from ride.carla.mechanism_audit import audit as audit_mechanism


def report(root: Path):
    rows=[]
    for path in sorted(root.glob('rear_gap_*/rear_delta_*/carla/cut_in.jsonl')):
        item={'panel':str(path), 'relative_panel':str(path.relative_to(root))}
        try:
            data=read_jsonl(path)
            manifest=data[0]
            rk=audit_rankings(data)['summary']
            me=audit_mechanism(data)['summary']
            item.update({'status':'COMPLETE',
                'rear_gap_m':manifest.get('cutin_rear_gap_m'),
                'rear_speed_delta_mps':manifest.get('rear_speed_delta_mps'),
                'anchors':len(manifest.get('anchors') or []),
                'valid_mechanism_trials':me.get('mechanism_executed'),
                'valid_trials':me.get('valid'),
                'cruise_rear_braking_trials':me.get('cruise_rear_braking_trials'),
                'valid_within_family_pairs':me.get('valid_task_matched_mechanism_pairs'),
                'task_matched_cross_family_pairs':rk.get('task_matched_cross_model_pairs'),
                'strict_ranking_reversals':rk.get('strict_risk_order_reversals'),
                'third_party_collisions':me.get('third_party_collision')})
        except (ValueError,KeyError,OSError) as e:
            item.update({'status':'INVALID_OR_INCOMPLETE','error':str(e),
                'not_a_negative_scientific_result':True})
        rows.append(item)
    return {'cells':rows,'cells_found':len(rows),
            'valid_cells':sum(x['status']=='COMPLETE' for x in rows),
            'limits':['Discovery cells are not independent observations when corridors repeat',
                      'Do not select favorable gaps and report only those on a holdout',
                      'RQ1 strict ranking uses post-hoc simulator outcomes, not actual planner choices',
                      'cruise does not satisfy the same merge task; it is a mechanism control only']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    result=report(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'cells_found':result['cells_found'],'valid_cells':result['valid_cells']},indent=2))
if __name__=='__main__':main()
