"""Strict scientific go/no-go status. Absence of evidence is not negative evidence."""
import argparse,json
from pathlib import Path


def evaluate(audit,decision=None,ranking=None):
    s=audit['summary']; res={'preliminary_only':True,'human_ground_truth':False}
    if not s['total_trials'] or not s['valid_task_matched_contrasts']:
        res['RQ1']='NOT_TESTED_VALIDLY'
    elif ranking is None:
        res['RQ1']='NOT_TESTED_ACTION_RANKING_REQUIRED'
    elif not ranking['summary'].get('task_matched_cross_model_pairs'):
        res['RQ1']='NOT_TESTED_NO_CROSS_FAMILY_TASK_MATCHED_ACTIONS'
    elif ranking['summary']['strict_risk_order_reversals']:
        res['RQ1']='SIMULATOR_RISK_RANKING_REVERSAL_NOT_ACTUAL_PLANNER_ERROR'
    elif ranking['summary']['tie_preference_mismatches']:
        res['RQ1']='SIMULATOR_RANKING_TIE_DISAGREEMENT_NOT_STRICT_REVERSAL'
    else:res['RQ1']='NO_CROSS_FAMILY_RANKING_REVERSAL_IN_THIS_PANEL'
    if not s['valid_task_matched_contrasts']:
        res['RQ2']='NOT_TESTED_VALIDLY'
    elif s['cross_family_matching_collision_effects']:
        res['RQ2']='SIMULATOR_PAIRED_EFFECT_OBSERVED_AWAITING_INDEPENDENT_REPLICATION'
    else:res['RQ2']='NO_STABLE_PAIRED_THIRD_PARTY_COLLISION_EFFECT_IN_THIS_PANEL'
    if decision is None or not decision.get('summary'):
        res['RQ3']='NOT_TESTED_NO_LOCKED_STRONG_BASELINE_CHOICES'
    elif not decision.get('total_valid',0):
        res['RQ3']='NOT_TESTED_NO_VALID_BASELINE_DECISIONS'
    elif all(str(k).startswith('PROXY_') for k in decision['summary']):
        res['RQ3']='ONLY_PROXY_SCORE_SELECTORS_NOT_STRONG_BASELINE_COMPARISON'
    else:
        res['RQ3']='BASELINE_CHOICES_AUDITED_STILL_REQUIRES_CHECKPOINT_AND_FAIRNESS_REVIEW'
    res['start_new_planner_algorithm']=False
    res['note']='Gates are conservative. Mixed-model CARLA is not a human counterfactual experiment.'
    return res


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--decision')
    p.add_argument('--ranking');p.add_argument('--output',required=True);a=p.parse_args(argv)
    aud=json.loads(Path(a.audit).read_text());dec=json.loads(Path(a.decision).read_text()) if a.decision else None
    ran=json.loads(Path(a.ranking).read_text()) if a.ranking else None
    result=evaluate(aud,dec,ran)
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
