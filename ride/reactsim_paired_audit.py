"""STRICT normalized ReactSim two-action/two-model audit.

This is NOT an adapter for the authors' raw pickle rollouts. Export one JSON
record per (scene, action, response model) after a verified mapping from their
native output. Never invent an absent agent collision or re-use risk labels.

Required fields (see docs/REACTSIM_EXPORT_SCHEMA_CN.md):
 scene_id, action_id, response_model, initial_state_digest,
 affected_actor_pair_digest, action_trajectory_digest, source_rollout_digest,
 progress_m, ego_collision, third_party_collision,
 hard_brake, rule_valid, mechanism_executed, simulator_completed.

The output is a post-rollout *evaluation model* ranking sensitivity test. It is
not ex-ante planner RQ1 (which additionally requires pre-committed choices).
"""
from __future__ import annotations
import argparse,collections,json,math,statistics
from pathlib import Path

REQ_STRING=('scene_id','action_id','response_model','initial_state_digest',
            'affected_actor_pair_digest','action_trajectory_digest','source_rollout_digest')
REQ_BOOL=('ego_collision','third_party_collision','hard_brake','rule_valid',
          'mechanism_executed','simulator_completed')


def check_record(row):
    for k in REQ_STRING:
        if not isinstance(row.get(k),str) or not row[k].strip():
            raise ValueError('Missing nonempty string: '+k)
    for k in REQ_BOOL:
        if type(row.get(k)) is not bool:
            raise ValueError('Expected literal JSON boolean for '+k)
    x=row.get('progress_m')
    if type(x) not in (int,float) or not math.isfinite(x):
        raise ValueError('Progress must be finite numerical value')
    if not row.get('source_rollout_digest') or len(row['source_rollout_digest'])<12:
        raise ValueError('Source rollout provenance hash too short')


def risk_key(row):
    return (int(row['third_party_collision']),int(row['hard_brake']))


def sign_pair(a,b):
    ka,kb=risk_key(a),risk_key(b)
    return 1 if ka>kb else (-1 if ka<kb else 0)


def analyze(rows,*,action_a,action_b,model_a,model_b,progress_tol=3.0):
    if len({action_a,action_b})!=2 or len({model_a,model_b})!=2 or progress_tol<=0:
        raise ValueError('Need distinct A/B actions and models with positive progress tolerance')
    indexed={};scenes=set()
    for row in rows:
        check_record(row)
        key=(row['scene_id'],row['action_id'],row['response_model'])
        if key in indexed:raise ValueError('Duplicate outcome for '+repr(key))
        indexed[key]=row;scenes.add(row['scene_id'])
    out=[];reasons=collections.Counter()
    for scene in sorted(scenes):
        arms={(act,mod):indexed.get((scene,act,mod))
              for act in (action_a,action_b) for mod in (model_a,model_b)}
        if any(x is None for x in arms.values()):
            reasons['missing_arm']+=1;out.append({'scene_id':scene,'eligible':False,'reason':'missing_arm'});continue
        vals=list(arms.values())
        if not all(x['simulator_completed'] for x in vals):
            reason='incomplete_simulator_rollout'
        elif len({x['initial_state_digest'] for x in vals})!=1:
            reason='initial_state_mismatch'
        elif len({x['affected_actor_pair_digest'] for x in vals})!=1:
            reason='affected_actor_pair_mismatch'
        elif any(len({arms[(act,mod)]['action_trajectory_digest'] for mod in (model_a,model_b)})!=1
                 for act in (action_a,action_b)):
            reason='different_ego_action_across_models'
        elif any(not x['rule_valid'] or not x['mechanism_executed'] or x['ego_collision'] for x in vals):
            reason='unsafe_illegal_or_unexecuted_action'
        elif any(abs(arms[(action_a,mod)]['progress_m']-
                     arms[(action_b,mod)]['progress_m'])>progress_tol for mod in (model_a,model_b)):
            reason='progress_unmatched'
        else:
            reason=None
        if reason:
            reasons[reason]+=1;out.append({'scene_id':scene,'eligible':False,'reason':reason});continue
        signs={mod:sign_pair(arms[(action_a,mod)],arms[(action_b,mod)])
               for mod in (model_a,model_b)}
        reversal=signs[model_a]*signs[model_b]==-1
        out.append({'scene_id':scene,'eligible':True,'risk_signs':signs,
                    'strict_posthoc_response_model_ranking_reversal':reversal,
                    'a_worse_than_b_in': [mod for mod in signs if signs[mod]>0],
                    'ego_progress_gap_m': {mod:abs(arms[(action_a,mod)]['progress_m']-
                                                       arms[(action_b,mod)]['progress_m'])
                                           for mod in (model_a,model_b)},
                    'source_rollout_hashes':[x['source_rollout_digest'] for x in vals]})
    valid=[x for x in out if x.get('eligible')]
    return {'protocol':'normalized_reactsim_posthoc_two_action_v1',
            'summary':{'scenes_seen':len(scenes),'eligible_scenes':len(valid),
                       'strict_posthoc_reversals':sum(x['strict_posthoc_response_model_ranking_reversal'] for x in valid),
                       'invalid_reasons':dict(reasons),
                       'rq1_ex_ante_planner_choice_proven':False,
                       'rq2_human_causal_effect_proven':False,
                       'do_not_treat_reactsim_world_models_as_human_ground_truth':True},
            'action_pair':[action_a,action_b],'model_pair':[model_a,model_b],
            'details':out}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Normalized JSONL exported/verified from native rollout')
    p.add_argument('--output',required=True)
    p.add_argument('--action-a',required=True);p.add_argument('--action-b',required=True)
    p.add_argument('--model-a',required=True);p.add_argument('--model-b',required=True)
    p.add_argument('--progress-tolerance-m',type=float,default=3.)
    a=p.parse_args(argv)
    rows=[json.loads(line) for line in Path(a.input).read_text(encoding='utf8').splitlines() if line.strip()]
    result=analyze(rows,action_a=a.action_a,action_b=a.action_b,
                   model_a=a.model_a,model_b=a.model_b,progress_tol=a.progress_tolerance_m)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError('Refusing to overwrite normalized audit '+str(target))
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
