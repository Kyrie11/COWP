"""Pre-outcome candidate selection baselines from EXTERNAL predicted scores.

These are *score-based reference selectors*, NOT official RACP/GameFormer/HYPE.
Predictions must be generated from scene history before the independent rollout;
using real rollout outcomes as input is evaluation leakage.
"""
from __future__ import annotations
import argparse,csv,json,collections,math
from pathlib import Path

POLICIES=('safety_only','soft_courtesy','hard_courtesy')
REQUIRED=('archetype','anchor','seed','action','pred_ego_collision_prob',
          'pred_third_party_collision_prob','pred_burden_cost','pred_progress_m',
          'pred_dynamic_valid','pred_rule_valid')


def as_float(r,k):
    v=float(r[k]);
    if not math.isfinite(v):raise ValueError('Nonfinite prediction: '+k)
    return v


def truth(v):return str(v).lower() in ('1','true','yes')


def choose(records,policy,*,ego_threshold=.1,third_party_threshold=.1,
           burden_threshold=.25,soft_weight=1.0):
    if policy not in POLICIES:raise ValueError('Unknown policy '+policy)
    candidates=[]
    for row in records:
        if not all(k in row and str(row[k]).strip()!='' for k in REQUIRED):
            raise ValueError('Predictor score is incomplete; cannot select actions')
        if not truth(row['pred_dynamic_valid']) or not truth(row['pred_rule_valid']):continue
        ego=as_float(row,'pred_ego_collision_prob')
        third=as_float(row,'pred_third_party_collision_prob')
        burden=as_float(row,'pred_burden_cost')
        progress=as_float(row,'pred_progress_m')
        if not (0<=ego<=1 and 0<=third<=1 and burden>=0):
            raise ValueError('Risk probabilities must be in [0,1], burden >=0')
        if ego>ego_threshold:continue
        if policy=='hard_courtesy' and (third>third_party_threshold or burden>burden_threshold):continue
        utility=-progress
        if policy=='soft_courtesy':utility+=soft_weight*(burden+5*third)
        # Deterministic tie break keeps the same candidate set and reproducibility.
        candidates.append((utility,str(row['action']),row))
    if not candidates:return {'selected_action':None,'status':'NO_PREDICTED_FEASIBLE_ACTION'}
    _,action,row=min(candidates,key=lambda x:(x[0],x[1]))
    return {'selected_action':action,'status':'SELECTED','predicted_progress_m':float(row['pred_progress_m']),
            'surviving_candidates':len(candidates)}


def make_choices(rows,policy,*,reference_families=('tm','behavior'),**kwargs):
    groups=collections.defaultdict(list)
    for r in rows:
        k=(str(r.get('archetype')),str(r.get('anchor')),str(r.get('seed')))
        groups[k].append(r)
    choices=[];failures=[]
    for (archetype,anchor,seed),records in sorted(groups.items()):
        result=choose(records,policy,**kwargs)
        if result['selected_action'] is None:
            failures.append({'archetype':archetype,'anchor':anchor,'seed':seed,**result});continue
        for eval_family in reference_families:
            choices.append({'method_id':'PROXY_'+policy.upper(),
                'archetype':archetype,'anchor':anchor,'seed':seed,
                'response_family':eval_family,'selected_action':result['selected_action'],
                'choice_locked_before_rollout':'true',
                'prediction_only_no_oracle': 'true'})
    return choices,failures


def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True,help='Ex-ante model predictions, NEVER rollout data')
    p.add_argument('--policy',choices=POLICIES,required=True);p.add_argument('--choices',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--ego-risk-threshold',type=float,default=.1)
    p.add_argument('--third-party-risk-threshold',type=float,default=.1)
    p.add_argument('--burden-threshold',type=float,default=.25)
    p.add_argument('--soft-weight',type=float,default=1.)
    a=p.parse_args()
    with open(a.predictions,encoding='utf8',newline='') as f:records=list(csv.DictReader(f))
    if not records:raise ValueError('No ex-ante prediction rows supplied')
    if not 0<=a.ego_risk_threshold<=1 or not 0<=a.third_party_risk_threshold<=1 or a.burden_threshold<0 or a.soft_weight<0:
        raise ValueError('Thresholds not in legal range')
    out,fail=make_choices(records,a.policy,
                          ego_threshold=a.ego_risk_threshold,
                          third_party_threshold=a.third_party_risk_threshold,
                          burden_threshold=a.burden_threshold,
                          soft_weight=a.soft_weight)
    path=Path(a.choices);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Do not overwrite frozen choice file: '+str(path))
    fields=('method_id','archetype','anchor','seed','response_family','selected_action',
            'choice_locked_before_rollout','prediction_only_no_oracle')
    with path.open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(out)
    from .commit import digest
    import datetime
    manifest={'sha256':digest(path),'choices_file':str(path.resolve()),
              'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'policy':a.policy,'hyperparameters':{'ego_risk_threshold':a.ego_risk_threshold,
                  'third_party_risk_threshold':a.third_party_risk_threshold,
                  'burden_threshold':a.burden_threshold,'soft_weight':a.soft_weight},
              'submitted':len(out),'no_feasible_cases':fail,
              'limitations':['Inputs are EXTERNAL predicted estimates not validated ground truth',
                  'These are reference selectors, not official RACP or GameFormer',
                  'Hash does not independently prove pre-outcome timestamp']}
    dst=Path(a.manifest);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise FileExistsError('Do not overwrite commit manifest: '+str(dst))
    dst.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'locked_choices':len(out),'no_feasible':len(fail),'csv':str(path),
                      'manifest':str(dst),'sha256':manifest['sha256']},indent=2))
if __name__=='__main__':main()
