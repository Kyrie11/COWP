"""Validate normalized paired closed-loop trials from distinct RESPONSE model families.

Does NOT run models or certify their independence.  Rows are produced by actual
external policies and must include source provenance.  We refuse to silently
relabel IDM parameter variants as independent models.

Schema per JSONL row:
 scenario_id, family, implementation, checkpoint_sha256, seed,
 bad_action_id, alternative_action_id, pair_object_ids (original WOMD IDs),
 bad_pair_collision, alternative_pair_collision, bad_ego_collision,
 alternative_any_collision, bad_progress_m, alternative_progress_m,
 bad_road_valid, alternative_road_valid, bad_kinematics_valid,
 alternative_kinematics_valid.
"""
from __future__ import annotations
import argparse,json,collections,math
from pathlib import Path

REQUIRED=('scenario_id','family','implementation','checkpoint_sha256','seed','bad_action_id',
'alternative_action_id','pair_object_ids','bad_pair_collision','alternative_pair_collision',
'bad_ego_collision','alternative_any_collision','bad_progress_m','alternative_progress_m',
'bad_road_valid','alternative_road_valid','bad_kinematics_valid','alternative_kinematics_valid')

def key(r):
    pair=r['pair_object_ids']
    if len(pair)!=2 or pair[0]==pair[1]:raise ValueError('pair_object_ids must be two distinct original WOMD IDs')
    return (r['scenario_id'],r['bad_action_id'],r['alternative_action_id'],tuple(sorted(map(int,pair))))

def check(rows,tolerance=3.0,minimum_seeds=5,minimum_family_agreement=.8):
    groups=collections.defaultdict(list);unique=set();invalid=[]
    for j,r in enumerate(rows):
        missing=[x for x in REQUIRED if x not in r]
        if missing:invalid.append({'row':j,'reason':'missing '+','.join(missing)});continue
        try:
            grouping=(key(r),str(r['family']))
            token=(grouping,int(r['seed']))
            if token in unique:raise ValueError('duplicate scene/action/pair/family/seed')
            unique.add(token)
            if not str(r['implementation']).strip() or not str(r['checkpoint_sha256']).strip():
                raise ValueError('provenance missing')
            if not all(isinstance(r[k],bool) for k in ('bad_pair_collision','alternative_pair_collision',
            'bad_ego_collision','alternative_any_collision','bad_road_valid','alternative_road_valid',
            'bad_kinematics_valid','alternative_kinematics_valid')):
                raise ValueError('boolean status must be true/false JSON boolean')
            success=(r['bad_pair_collision'] and not r['alternative_pair_collision']
              and not r['bad_ego_collision'] and not r['alternative_any_collision']
              and r['bad_road_valid'] and r['alternative_road_valid']
              and r['bad_kinematics_valid'] and r['alternative_kinematics_valid']
              and math.isfinite(float(r['bad_progress_m']))
              and math.isfinite(float(r['alternative_progress_m']))
              and abs(float(r['alternative_progress_m'])-float(r['bad_progress_m']))<=tolerance)
            groups[grouping].append((success,r))
        except Exception as exc:invalid.append({'row':j,'reason':str(exc)})
    summaries=collections.defaultdict(list)
    for (trial,family),obs in groups.items():
        success=sum(k for k,_ in obs); n=len(obs)
        implementations=sorted({v['implementation']+'|'+v['checkpoint_sha256'] for _,v in obs})
        summaries[trial].append({'family':family,'seed_count':n,'successful_seeds':success,
            'positive_rate':success/n,'implementations':implementations,
            'qualified':n>=minimum_seeds and success/n>=minimum_family_agreement})
    candidate=[]
    for trial,families in summaries.items():
        qualified=[f for f in families if f['qualified']]
        candidate.append({'scenario_id':trial[0],'bad_action_id':trial[1],
          'alternative_action_id':trial[2],'pair_object_ids':list(trial[3]),
          'families':families,'qualified_family_count':len(qualified),
          'two_distinct_family_screen':len(qualified)>=2})
    candidate.sort(key=lambda x:(-x['qualified_family_count'],x['scenario_id']))
    return {'status':'CROSS_MODEL_SIMULATOR_SCREEN_NOT_CAUSAL_HUMAN_PROOF',
      'n_input':len(rows),'invalid_rows':invalid,'n_valid':sum(len(v) for v in groups.values()),
      'n_pairs':len(candidate),'n_two_family_screens':sum(x['two_distinct_family_screen'] for x in candidate),
      'gate':{'minimum_seeds':minimum_seeds,'minimum_agreement':minimum_family_agreement,
              'progress_tolerance_m':tolerance},
      'limitations':['Implementation/checkpoint provenance is self-reported and must be independently verified',
                     'Two named model families must be architecturally independent, not IDM parameter regimes',
                     'Human potential outcomes remain unobserved',
                     'No joint-feasibility/recourse-completeness proof is implied'],
      'pairs':candidate}

def main():
    p=argparse.ArgumentParser();p.add_argument('--jsonl',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--min-seeds',type=int,default=5)
    p.add_argument('--min-agreement',type=float,default=.8);p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args()
    rows=[json.loads(x) for x in a.jsonl.read_text().splitlines() if x.strip()]
    result=check(rows,a.progress_tolerance,a.min_seeds,a.min_agreement)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
