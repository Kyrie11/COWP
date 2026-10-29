"""Prospective, diagnostic social selectors for shared CARLA ego action pairs.

This is NOT an implementation of RACP, HYPE, GameFormer or any official
planning algorithm. Selectors score SOURCE simulator rollouts, freeze choices
before target exists, then assess target outcomes under the same candidates.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math
from pathlib import Path
from .evidence import read_jsonl
from .science import Protocol, validate_pair, same_initial, same_ego_execution
from .v13_model_transfer import extract_trial_cost, PAIRS

SELECTORS=('PROGRESS_ONLY','C_ONLY','B_ONLY_COURTESY','B_C_COURTESY','WEIGHTED_B_C')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def burden(row,role,threshold=2.0):
    tr=row.get('trace') or []
    if len(tr)<20:raise ValueError('missing trace')
    acc_field=role+'_accel'
    total=0.;steps=0
    for i,t in enumerate(tr):
        if i==0:continue
        dt=float(t['t'])-float(tr[i-1]['t'])
        if dt<=0 or dt>.2:raise ValueError('invalid timebase')
        if float(t['t'])<1.0:continue
        acc=float(t[acc_field]);
        if not math.isfinite(acc):raise ValueError('nonfinite acceleration')
        total+=max(0.,-acc-threshold)*dt;steps+=1
    if steps<20:raise ValueError('too few frames')
    return round(total,6)


def score(row):
    c=extract_trial_cost(row)
    b=burden(row,'follower_b')
    return {'B_burden':b,'C_burden':c['c_burden_mps'],
            'C_primary_risk':c['primary'],'ego_progress_m':c['ego_progress_m']}


def choose(p_a,p_b,selector,budget=0.4,tie=0.05):
    if selector not in SELECTORS:raise ValueError('unknown selector')
    if selector=='PROGRESS_ONLY':v_a=v_b=0.
    elif selector=='C_ONLY':v_a=p_a['C_burden'];v_b=p_b['C_burden']
    elif selector=='WEIGHTED_B_C':
        v_a=p_a['B_burden']+p_a['C_burden'];v_b=p_b['B_burden']+p_b['C_burden']
    elif selector in ('B_ONLY_COURTESY','B_C_COURTESY'):
        role_keys=('B_burden',) if selector=='B_ONLY_COURTESY' else ('B_burden','C_burden')
        qa=all(p_a[x]<=budget for x in role_keys);qb=all(p_b[x]<=budget for x in role_keys)
        if not qa and not qb:return None,'BOTH_FAIL_SPECIFIED_COURTESY_BUDGET_UNKNOWN'
        if qa!=qb:return ('a' if qa else 'b'),'COURTESY_PASS_ONLY_ONE'
        v_a=v_b=0.
    else:raise ValueError(selector)
    # Primary safety takes precedence when C scored, but courtesy still needs
    # explicit B and C cost semantics. A model's own risk is not ground truth.
    if selector in ('C_ONLY','WEIGHTED_B_C') and p_a['C_primary_risk']!=p_b['C_primary_risk']:
        return ('a' if p_a['C_primary_risk']<p_b['C_primary_risk'] else 'b'),'MODEL_PRIMARY_RISK'
    if v_a<v_b-tie:return 'a','ESTIMATED_LOSS'
    if v_b<v_a-tie:return 'b','ESTIMATED_LOSS'
    return ('a' if p_a['ego_progress_m']>=p_b['ego_progress_m'] else 'b'),'PROGRESS_TIEBREAK'


def _trials(rows,family):
    out={}
    for r in rows:
        if r.get('type')!='trial':continue
        if r.get('family')!=family:raise ValueError('unexpected family '+str(r.get('family')))
        key=(str(r['anchor']),str(r['seed']),r['profile'])
        if key in out:raise ValueError('duplicate trial')
        out[key]=r
    return out


def _write_new(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)


def lock(source, target, frozen, choices, *, family, target_family, budget=.4,tie=.05):
    if Path(target).exists():raise ValueError('TARGET_ALREADY_EXISTS: cannot precommit')
    rows=read_jsonl(source);index=_trials(rows,family)
    out=[]
    for anchor,seed in sorted(set((k[0],k[1]) for k in index)):
        for a,b in PAIRS:
            x=index.get((anchor,seed,a));y=index.get((anchor,seed,b))
            base={'anchor':anchor,'seed':seed,'actions':[a,b], 'source_action_valid':False}
            if x is None or y is None:
                base['unknown_reason']='MISSING_SOURCE_ACTION'
            else:
                ok,reason,prog=validate_pair(x,y,Protocol())
                if not ok:base['unknown_reason']='INVALID_SOURCE_PAIR:'+reason
                else:
                    try:
                        sx,sy=score(x),score(y)
                        base.update(source_action_valid=True,source_scores={a:sx,b:sy})
                    except (KeyError,TypeError,ValueError) as ex:
                        base['unknown_reason']='SOURCE_SCORE_UNAVAILABLE:'+str(ex)
            for selector in SELECTORS:
                item=dict(base,selector=selector,choice=None,choice_reason='UNKNOWN')
                if base['source_action_valid']:
                    chosen,why=choose(sx,sy,selector,budget,tie)
                    item['choice']=a if chosen=='a' else b if chosen=='b' else None
                    item['choice_reason']=why
                out.append(item)
    result={'protocol':'V16_PRECOMMITTED_TRANSPARENT_SOCIAL_SELECTORS',
            'not_official_planner_baseline':True,
            'source_family':family,'target_family':target_family,
            'source_sha256':sha(source),'frozen_sha256':sha(frozen),
            'target_expected':str(Path(target).resolve()),
            'budget_C_or_B_mps':budget,'tie_mps':tie,
            'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'timestamp_not_external_notarization':True,'choices':out}
    _write_new(choices,result)
    return {'locked_records':len(out),'precommitted_selected':sum(x['choice'] is not None for x in out)}


def evaluate(source,target,frozen,choices,output,*,family,target_family):
    lockj=json.loads(Path(choices).read_text())
    if lockj['source_sha256']!=sha(source) or lockj['frozen_sha256']!=sha(frozen):
        raise ValueError('SOURCE_OR_FROZEN_CHANGED_AFTER_LOCK')
    if lockj['target_expected']!=str(Path(target).resolve()):raise ValueError('TARGET_PATH_CHANGED')
    if lockj['source_family']!=family or lockj['target_family']!=target_family:
        raise ValueError('RESPONSE_FAMILY_MISMATCH')
    src=_trials(read_jsonl(source),family);tgt=_trials(read_jsonl(target),target_family)
    results=[]
    for row in lockj['choices']:
        q={k:row[k] for k in ('anchor','seed','actions','selector','choice','choice_reason')}
        q['status']='UNKNOWN'
        a,b=row['actions'];keya=(row['anchor'],row['seed'],a);keyb=(row['anchor'],row['seed'],b)
        if row['choice'] is None:q['reason']='SOURCE_SELECTION_UNKNOWN';results.append(q);continue
        if keya not in src or keyb not in src or keya not in tgt or keyb not in tgt:
            q['reason']='MISSING_ARM';results.append(q);continue
        sa,sb=src[keya],src[keyb];ta,tb=tgt[keya],tgt[keyb]
        ok,reason,_=validate_pair(ta,tb,Protocol())
        if not ok:q['reason']='INVALID_TARGET_PAIR:'+reason;results.append(q);continue
        valid=True;errors=[]
        for xs,xt in ((sa,ta),(sb,tb)):
            same,why=same_initial(xs,xt,Protocol());ego,ewhy=same_ego_execution(xs,xt)
            if not same or not ego:valid=False;errors.extend([z for z in (why,ewhy) if z])
        if not valid:q['reason']='SOURCE_TARGET_PARITY:'+','.join(errors);results.append(q);continue
        try:costs={a:score(ta),b:score(tb)}
        except (KeyError,TypeError,ValueError) as ex:
            q['reason']='TARGET_SCORE_UNAVAILABLE:'+str(ex);results.append(q);continue
        chosen=row['choice'];other=b if chosen==a else a
        delta=costs[chosen]['C_burden']-costs[other]['C_burden']
        q.update({'status':'EVALUABLE','reason':'','target_scores':costs,
                  'C_burden_regret_mps':round(max(0,delta),6),
                  'avoidable_C_burden_0_25':bool(delta>=.25),
                  'controlled_response_environment_not_human_ground_truth':True})
        results.append(q)
    out={'protocol':'V16_PRECOMMITTED_TRANSPARENT_SOCIAL_SELECTOR_EVALUATION',
         'target_family':target_family,'target_sha256':sha(target),
         'records':results,'summary':{sel:{'evaluable':sum(r['status']=='EVALUABLE' and r['selector']==sel for r in results),
                                      'avoidable_C_burden_rows':sum(r.get('avoidable_C_burden_0_25',False) and r['selector']==sel for r in results)}
                                      for sel in SELECTORS},
         'limitations':['Simplified transparent source rollout selectors, not original RACP/HYPE/GameFormer.',
                        'B-only and B+C courtesy budgets are author-defined controls, not proven fair parameter optima.',
                        'Primary risk and burden are computed from CARLA simulator outcomes.']}
    _write_new(output,out);return out['summary']


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('lock','evaluate'))
    ap.add_argument('--source',required=True);ap.add_argument('--target',required=True)
    ap.add_argument('--frozen',required=True);ap.add_argument('--choices',required=True)
    ap.add_argument('--source-family',default='mixed_b_tm_c_behavior')
    ap.add_argument('--target-family',default='mixed_b_tm_c_delayed_behavior')
    ap.add_argument('--budget',type=float,default=.4);ap.add_argument('--tie',type=float,default=.05)
    ap.add_argument('--output')
    args=ap.parse_args(argv)
    if not math.isfinite(args.budget) or args.budget<0 or not math.isfinite(args.tie) or args.tie<0:
        ap.error('budget and tie must be finite nonnegative')
    if args.mode=='lock':
        result=lock(args.source,args.target,args.frozen,args.choices,family=args.source_family,
                    target_family=args.target_family,budget=args.budget,tie=args.tie)
    else:
        if not args.output:ap.error('--output required for evaluate')
        result=evaluate(args.source,args.target,args.frozen,args.choices,args.output,
                        family=args.source_family,target_family=args.target_family)
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
