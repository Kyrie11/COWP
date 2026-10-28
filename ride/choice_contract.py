"""Ex-ante RQ3 selection contract, fail closed on provenance/timing gaps.

This is a validator for exported planner decisions, NOT a planner implementation.
The actual baseline must write its locked proposal before any outcome rollout.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

REQUIRED=('scene_id','initial_state_sha256','candidates','candidate_scores',
          'selected_action','planner','planner_commit','response_model_assumption',
          'checkpoint_sha256','choice_locked_before_rollout','information_cutoff_us')

def validate_record(r):
    missing=[x for x in REQUIRED if x not in r]
    if missing:return False,'MISSING_FIELDS:'+','.join(missing)
    if r['choice_locked_before_rollout'] is not True:return False,'CHOICE_NOT_PRELOCKED'
    c=r['candidates'];scores=r['candidate_scores']
    if not isinstance(c,list) or len(c)<2 or len(set(c))!=len(c):return False,'BAD_CANDIDATE_POOL'
    if r['selected_action'] not in c:return False,'SELECTED_NOT_IN_POOL'
    if not isinstance(scores,dict) or set(scores)!=set(c):return False,'UNMATCHED_CANDIDATE_SCORES'
    try:
        import math
        if not all(math.isfinite(float(x)) for x in scores.values()):return False,'NONFINITE_SCORES'
        if not isinstance(r['information_cutoff_us'],int):return False,'BAD_INFORMATION_CUTOFF'
    except (TypeError,ValueError):return False,'BAD_SCORES'
    for key in ('initial_state_sha256','checkpoint_sha256'):
        if len(str(r[key]))!=64 or any(z not in '0123456789abcdef' for z in str(r[key]).lower()):
            return False,'BAD_DIGEST:'+key
    return True,'LOCK_VALID'

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--choices-jsonl',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    results=[]
    for i,line in enumerate(a.choices_jsonl.read_text().splitlines(),1):
        try:ok,reason=validate_record(json.loads(line))
        except json.JSONDecodeError:ok,reason=False,'INVALID_JSON'
        results.append({'line':i,'eligible':ok,'reason':reason})
    summary={'rows':len(results),'locked_valid':sum(x['eligible'] for x in results),
             'locked_invalid':sum(not x['eligible'] for x in results),
             'scientific_scope':'SCHEMA_VALIDATION_ONLY_NO_PLANNER_ORACLE'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'summary':summary,'rows':results},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not results or any(not x['eligible'] for x in results):raise SystemExit(3)
if __name__=='__main__':main()
