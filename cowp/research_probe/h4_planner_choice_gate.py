"""Audit common-interface planner preferences; NEVER invent GameFormer/RA-CBF trials.

Required JSONL per actually executed baseline policy and scenario:
 scenario_id, planner, source_implementation, checkpoint_sha256,
 evaluated_actions (list of IDs), chosen_action, bad_action_id,
 alternative_action_id, bad_progress_m, alternative_progress_m,
 bad_valid, alternative_valid, bad_pair_risk, alternative_pair_risk.

Observed 'would select bad action' is meaningful only if both candidates
were actually available in the same action set with identical infrastructure.
"""
from __future__ import annotations
import argparse,json,collections,math
from pathlib import Path
REQ=('scenario_id','planner','source_implementation','checkpoint_sha256','evaluated_actions',
     'chosen_action','bad_action_id','alternative_action_id','bad_progress_m','alternative_progress_m',
     'bad_valid','alternative_valid','bad_pair_risk','alternative_pair_risk')
def check(rows,tol=3.):
    results=[];errors=[];seen=set()
    for i,r in enumerate(rows):
        missing=[k for k in REQ if k not in r]
        if missing:errors.append({'row':i,'error':'missing '+','.join(missing)});continue
        try:
            sid=r['scenario_id'];planner=r['planner']
            token=(sid,planner);assert token not in seen, 'duplicate scenario/planner';seen.add(token)
            if not isinstance(r['evaluated_actions'],list):raise ValueError('evaluated_actions must be list')
            if len(set(r['evaluated_actions']))!=len(r['evaluated_actions']):raise ValueError('duplicate action in pool')
            bad,good=r['bad_action_id'],r['alternative_action_id']
            both=bad in r['evaluated_actions'] and good in r['evaluated_actions']
            if not str(r['source_implementation']).strip() or not str(r['checkpoint_sha256']).strip():raise ValueError('missing provenance')
            prog=float(r['alternative_progress_m'])-float(r['bad_progress_m'])
            if not math.isfinite(prog):raise ValueError('nonfinite progress')
            task_matched=abs(prog)<=tol
            safety=all(isinstance(r[k],bool) for k in ('bad_valid','alternative_valid')) and r['bad_valid'] and r['alternative_valid']
            risk=float(r['bad_pair_risk'])>float(r['alternative_pair_risk'])
            selected_bad=(r['chosen_action']==bad)
            results.append({'scenario_id':sid,'planner':planner,'choice_evidence_valid':bool(both and task_matched and safety and risk),
                            'selected_bad':selected_bad,'progress_delta_m':prog,'both_actions_available':both,
                            'note':'risk is supplied model metric and not independently validated'})
        except Exception as exc:errors.append({'row':i,'error':str(exc)})
    per=collections.defaultdict(lambda:{'tested':0,'qualified':0,'chose_bad':0})
    for r in results:
        x=per[r['planner']];x['tested']+=1
        if r['choice_evidence_valid']:
            x['qualified']+=1;x['chose_bad']+=r['selected_bad']
    return {'status':'EXTERNAL_PLANNER_TRIAL_AUDIT_NOT_REAL_WORLD_CAUSAL_PROOF',
            'n_submitted':len(rows),'n_checked':len(results),'errors':errors,
            'planner_summaries':dict(per),'rows':results}
def main():
    p=argparse.ArgumentParser();p.add_argument('--jsonl',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();rows=[json.loads(x) for x in Path(a.jsonl).read_text().splitlines() if x.strip()]
    result=check(rows);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
