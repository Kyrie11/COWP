"""Decision-relevant *proxy* burden-on/off audit on the SAME predictions.

Risk and dynamic/legal thresholds are held fixed. Only the predicted burden
constraint is toggled. This is not an ablation of a previous neural network,
and it is not independent evidence until locked choices are evaluated against
separately obtained responder outcomes.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math
from pathlib import Path
from .selectors import choose


def compare(rows, *, ego_threshold=.1, third_party_threshold=.1,
            burden_threshold=.25, progress_tolerance_m=3.):
    scenes=collections.defaultdict(list)
    for r in rows:
        if any(not str(r.get(k,'')).strip() for k in ('archetype','anchor','seed','action')):
            raise ValueError('Incomplete prediction scene/action identity')
        scenes[(r['archetype'],str(r['anchor']),str(r['seed']))].append(r)
    results=[]
    for key, candidates in sorted(scenes.items()):
        args={'ego_threshold':ego_threshold,'third_party_threshold':third_party_threshold}
        on=choose(candidates,'hard_courtesy',burden_threshold=burden_threshold,**args)
        off=choose(candidates,'hard_courtesy',burden_threshold=math.inf,**args)
        lookup={r['action']:r for r in candidates}
        selection_on, selection_off=on['selected_action'],off['selected_action']
        difference=None
        if selection_on is not None and selection_off is not None:
            difference=abs(float(lookup[selection_on]['pred_progress_m'])-
                           float(lookup[selection_off]['pred_progress_m']))
        results.append({'archetype':key[0],'anchor':key[1],'seed':key[2],
                        'burden_on':selection_on,'burden_off':selection_off,
                        'both_feasible':selection_on is not None and selection_off is not None,
                        'decision_switch':selection_on!=selection_off,
                        'predicted_progress_gap_m':difference,
                        'task_matched_switch_proxy':bool(selection_on!=selection_off and
                              difference is not None and difference<=progress_tolerance_m),
                        'burden_on_status':on['status'],'burden_off_status':off['status']})
    return {'summary':{'scenes':len(results),
                'decision_switches':sum(r['decision_switch'] for r in results),
                'task_matched_switch_proxies':sum(r['task_matched_switch_proxy'] for r in results),
                'burden_on_no_candidate':sum(r['burden_on'] is None for r in results),
                'burden_off_no_candidate':sum(r['burden_off'] is None for r in results),
                'original_model_burden_ablation_not_implemented':True,
                'independent_outcomes_not_evaluated':True},
            'thresholds':{'ego_prob':ego_threshold,'third_party_prob':third_party_threshold,
                          'burden':burden_threshold,'progress_tolerance_m':progress_tolerance_m},
            'results':results}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ego-risk-threshold',type=float,default=.1)
    p.add_argument('--third-party-risk-threshold',type=float,default=.1)
    p.add_argument('--burden-threshold',type=float,default=.25)
    p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args(argv)
    if not (0<=a.ego_risk_threshold<=1 and 0<=a.third_party_risk_threshold<=1 and
            math.isfinite(a.burden_threshold) and a.burden_threshold>=0 and a.progress_tolerance>0):
        raise ValueError('Invalid burden audit thresholds')
    source=Path(a.predictions)
    with source.open(encoding='utf8',newline='') as fp:rows=list(csv.DictReader(fp))
    if not rows:raise ValueError('Empty ex-ante predictions CSV')
    data=compare(rows,ego_threshold=a.ego_risk_threshold,
                 third_party_threshold=a.third_party_risk_threshold,
                 burden_threshold=a.burden_threshold,
                 progress_tolerance_m=a.progress_tolerance)
    data['prediction_file_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    data['warning']='Predictions must be obtained before reading evaluation rollouts; file hash is not proof of chronology.'
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Refusing to overwrite burden ablation output: '+str(out))
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(data['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
