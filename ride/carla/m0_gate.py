"""CARLA M0 diagnostic gate, not a scientific proof of RQ1/RQ2.

Accepts partial JSONL after simulator failure for *diagnosis*, never as
completed scientific data. Incomplete trials and exceptions are all counted.
"""
from __future__ import annotations
import argparse,collections,json,math
from pathlib import Path
from ride.science import Protocol,validate_pair,same_initial,same_ego_execution


def audit_rows(rows):
    p=Protocol()
    manifests=[r for r in rows if r.get('type')=='manifest']
    trials=[r for r in rows if r.get('type')=='trial']
    end=[r for r in rows if r.get('type')=='run_end']
    completed=bool(len(manifests)==1 and len(end)==1 and end[-1].get('completed') and
                   end[-1].get('trial_count')==len(trials))
    counts=collections.Counter(); reasons=collections.Counter(); by={}
    for r in trials:
        family=r.get('family');m=r.get('metrics') or {}
        counts['trials']+=1
        counts['family_'+str(family)]+=1
        if m.get('valid'):counts['valid_'+str(family)]+=1
        if m.get('valid') and m.get('mechanism_executed') and not m.get('ego_collision'):
            counts['ego_safe_mechanism_'+str(family)]+=1
        if m.get('non_ego_pair_collision'):counts['non_ego_pair_collision_'+str(family)]+=1
        if r.get('simulation_exception'):reasons['exception:'+str(r['simulation_exception']).split(':')[0]]+=1
        for reason in m.get('reasons') or []:reasons[str(reason)]+=1
        if family in p.families and not r.get('joint_targets') and not r.get('recovery'):
            key=(r.get('anchor'),r.get('seed'),family,r.get('profile'))
            if key in by:reasons['duplicate_trial_key']+=1
            by[key]=r
    per_anchor=[]; matched=0;eligible_within=0
    for anchor,seed in sorted({(a,s) for a,s,_,_ in by},key=lambda v:str(v)):
        for aa,bb in p.contrast_actions:
            arm={}; has_all=True
            for family in p.families:
                pair=[by.get((anchor,seed,family,action)) for action in (aa,bb)]
                if any(x is None for x in pair):has_all=False;arm[family]={'eligible':False,'reason':'missing_trial'};continue
                ok,why,dp=validate_pair(pair[0],pair[1],p)
                arm[family]={'eligible':ok,'reason':why,'delta_progress_m':dp}
                if ok:eligible_within+=1
            if not has_all:continue
            cross_reason=[]
            for action in (aa,bb):
                left=by[(anchor,seed,p.families[0],action)]
                right=by[(anchor,seed,p.families[1],action)]
                ok,reason=same_initial(left,right,p)
                if not ok:cross_reason.append(action+':'+reason)
                ok,reason=same_ego_execution(left,right)
                if not ok:cross_reason.append(action+':'+reason)
            ok_all=all(x['eligible'] for x in arm.values()) and not cross_reason
            if ok_all:matched+=1
            per_anchor.append({'anchor':anchor,'seed':seed,'actions':[aa,bb],
                'within_family':arm,'cross_family_errors':cross_reason,
                'rq1_pair_eligible':ok_all})
    families_ready=all(counts['ego_safe_mechanism_'+fam]>0 for fam in p.families)
    if not completed:status='INCOMPLETE_PANEL_NOT_SCIENTIFIC_DATA'
    elif any(r.get('simulation_exception') for r in trials):status='ENGINEERING_EXCEPTION_BLOCKER'
    elif not families_ready:status='M0_MECHANISM_OR_VALIDITY_BLOCKER'
    elif matched==0:status='M0_PHYSICS_EXECUTES_BUT_NO_CROSS_FAMILY_TASK_MATCH'
    else:status='M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF'
    return {'status':status,'batch_completed':completed,'count':dict(counts),
      'invalid_reasons':dict(reasons),'valid_within_family_action_pairs':eligible_within,
      'valid_cross_family_task_matched_pairs':matched,'per_anchor_pairs':per_anchor,
      'rq1_established':False,'rq2_established':False,
      'scientific_note':'M0 is an engineering gate, not real-world causal truth or ex ante planner decision validation'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    data=[];broken=0
    if a.input.exists():
        for line in a.input.read_text(encoding='utf8').splitlines():
            if line.strip():
                try:data.append(json.loads(line))
                except (ValueError,TypeError):broken+=1
    result=audit_rows(data)
    result['broken_json_lines']=broken
    if broken:result['status']='INCOMPLETE_OR_BROKEN_JSONL_NOT_SCIENTIFIC_DATA'
    a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','batch_completed','count',
      'invalid_reasons','valid_within_family_action_pairs',
      'valid_cross_family_task_matched_pairs')},ensure_ascii=False,indent=2))
    return 0 if result['status']=='M0_CROSS_FAMILY_PAIRS_AVAILABLE_NOT_RQ1_OR_RQ2_PROOF' else 4

if __name__=='__main__':raise SystemExit(main())
