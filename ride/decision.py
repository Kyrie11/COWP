"""Offline *evaluation* of decisions committed BEFORE independent rollouts.

No implemented baseline here is represented as an official GameFormer, RACP, HYPE,
or RA-CBF checkpoint. External planners must supply locked choices explicitly.
"""
from __future__ import annotations
import argparse,csv,json,collections,statistics
from pathlib import Path


def score(rollout_rows, selected_rows, progress_tolerance_m=3.):
    index={};duplicates=[]
    for r in rollout_rows:
        if r.get('type')!='trial' or r.get('joint_targets') or r.get('recovery'):continue
        key=(r.get('archetype','convoy'),str(r['anchor']),str(r['seed']),r['family'],r['profile'])
        if key in index:duplicates.append(key)
        index[key]=r
    if duplicates:raise ValueError('Repeated rollout identity: '+str(duplicates[:2]))
    results=[]
    seen_choices=set()
    for choice in selected_rows:
        ident=tuple(str(choice.get(k,'')) for k in ('method_id','archetype','anchor','seed','response_family'))
        if ident in seen_choices:raise ValueError('Duplicate baseline decisions for same context: '+str(ident))
        seen_choices.add(ident)
        required=('method_id','archetype','anchor','seed','response_family','selected_action','choice_locked_before_rollout')
        if any(not choice.get(k) for k in required):
            results.append({'choice':choice,'valid':False,'reason':'missing_required_commitment_fields'});continue
        if str(choice['choice_locked_before_rollout']).lower()!='true':
            results.append({'choice':choice,'valid':False,'reason':'not_precommitted'});continue
        key=(choice['archetype'],str(choice['anchor']),str(choice['seed']),choice['response_family'])
        cand={k[4]:v for k,v in index.items() if k[:4]==key}
        selected=choice['selected_action']
        if selected not in cand:
            results.append({'choice':choice,'valid':False,'reason':'missing_selected_rollout'});continue
        x=cand[selected].get('metrics') or {}
        if not x.get('valid'):
            results.append({'choice':choice,'valid':False,'reason':'selected_rollout_invalid'});continue
        if x.get('ego_collision'):
            results.append({'choice':choice,'method_id':choice['method_id'],
                            'valid':False,'reason':'selected_action_ego_collision',
                            'unsafe_selected':True,'ego_collision':True});continue
        eligible=[]
        for name, r in cand.items():
            m=r.get('metrics') or {}
            if (m.get('valid') and not m.get('ego_collision') and
                abs(float(m['ego_progress_m'])-float(x['ego_progress_m']))<=progress_tolerance_m):
                eligible.append((name,m))
        if not eligible:
            results.append({'choice':choice,'valid':False,'reason':'no_matched_alternative'});continue
        # Lexicographic *evaluation endpoint*, not an online baseline score.
        def measured_key(m):return (int(bool(m.get('non_ego_pair_collision'))),
                                    int(bool(m.get('hard_brake'))))
        actual = measured_key(x)
        opt=min(measured_key(m) for _,m in eligible)
        # Explicitly separate collision regret from braking proxy regret.
        results.append({'method_id':choice['method_id'],'archetype':key[0],'anchor':key[1],
                        'seed':key[2],'response_family':key[3],
                        'selected_action':selected,'valid':True,
                        'num_task_matched_candidates':len(eligible),
                        'selected_third_party_collision':actual[0],
                        'third_party_collision_regret':actual[0]-opt[0],
                        'hard_brake_proxy_regret':actual[1]-opt[1] if actual[0]==opt[0] else None,
                        'regret_semantics':'lexicographic; braking regret only comparable at equal collision risk',
                        'post_hoc_best_measured_actions':[name for name,m in eligible if measured_key(m)==opt],
                        'evaluation_oracle_only':True})
    grouped=collections.defaultdict(list)
    for r in results:
        name=r.get('method_id') or r.get('choice',{}).get('method_id') or '__INVALID_METHOD__'
        grouped[name].append(r)
    summary={}
    for method,v in grouped.items():
        good=[r for r in v if r.get('valid')]
        reasons=collections.Counter(r['reason'] for r in v if not r.get('valid'))
        summary[method]={'submitted_decisions':len(v),'valid_decisions':len(good),
                         'unsafe_ego_collision_selected':sum(r.get('unsafe_selected',False) for r in v),
                         'invalid_or_unmatched_reasons':dict(reasons),
                         'collision_regret_mean':statistics.mean(r['third_party_collision_regret'] for r in good) if good else None,
                         'num_collision_regrets':sum(r['third_party_collision_regret']>0 for r in good),
                         'posthoc_oracle_is_not_a_planner':True}
    return {'summary':summary,'total_submitted':len(selected_rows),
            'total_valid':sum(r.get('valid',False) for r in results),
            'total_unsafe_selected':sum(r.get('unsafe_selected',False) for r in results),
            'results':results}


def main(argv=None):
    from .evidence import read_jsonl
    p=argparse.ArgumentParser();p.add_argument('--rollouts',required=True)
    p.add_argument('--choices',required=True,help='CSV with locked planner choices, NOT post-outcome labels')
    p.add_argument('--output',required=True)
    p.add_argument('--commit-manifest',required=True,help='SHA256 manifest frozen before executing outcome rollouts')
    a=p.parse_args(argv)
    from .commit import digest
    committed=json.loads(Path(a.commit_manifest).read_text(encoding='utf8'))
    if digest(a.choices)!=committed.get('sha256'):
        raise RuntimeError('Planner choice CSV changed since the precommitted manifest')
    with open(a.choices,newline='',encoding='utf8') as src:rows=list(csv.DictReader(src))
    result=score(read_jsonl(a.rollouts),rows)
    dst=Path(a.output);dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
