"""Pair CARLA branches without manufacturing population or human H2 claims."""
from __future__ import annotations
import argparse,json,collections,math
from pathlib import Path
from .protocol import check_contrast


def _same_initial(a,b, tol=1.0,vel_tol=1.0):
    for role in ('ego','follower_a','follower_b'):
        q=a.get('initial_states',{}).get(role)
        r=b.get('initial_states',{}).get(role)
        if not q or not r:return False
        if math.dist(q['xyz'],r['xyz'])>tol or abs(q['speed_mps']-r['speed_mps'])>vel_tol:
            return False
    return True


def analyze(items,progress_tolerance=3.,initial_tolerance=1.):
    manifest=next((x for x in items if x.get('type')=='manifest'),None)
    if manifest is None:raise ValueError('Missing manifest, cannot validate simulator context')
    rows=[x for x in items if x.get('type')=='trial']
    coll=collections.defaultdict(dict);dupes=[]
    for x in rows:
        name=(x['anchor'],x['seed'],x['family'],x['profile'],tuple(x.get('joint_targets') or []),x.get('recovery'))
        if name in coll[x['anchor'],x['seed']]:dupes.append(name)
        coll[x['anchor'],x['seed']][name[2:]]=x
    if dupes:raise ValueError(f'duplicate trials {dupes[:2]}')
    out=[];joint_results=[];recovery_results=[]
    for (anchor,seed),trials in sorted(coll.items()):
        for family in ('tm','behavior'):
            subset={k[1]:v for k,v in trials.items() if k[0]==family and not k[2] and k[3] is None}
            if len(subset)<2:continue
            for bad in subset:
                for alternative in subset:
                    if bad==alternative:continue
                    a,b=subset[bad],subset[alternative]
                    if not _same_initial(a,b,initial_tolerance):
                        out.append({'anchor':anchor,'seed':seed,'family':family,
                            'bad':bad,'alternative':alternative,'eligible':False,
                            'why':'start_state_not_paired'});continue
                    summary=check_contrast(a.get('metrics',{}),b.get('metrics',{}),
                                           progress_tolerance=progress_tolerance)
                    out.append({'anchor':anchor,'seed':seed,'family':family,
                                'bad':bad,'alternative':alternative,**summary})
        # H3 finite backup test: suffix choices must share the exact realized
        # physical state at the common cutoff (all agents, not just ego).
        for family in ('tm','behavior'):
            options=[v for k,v in trials.items() if k[0]==family and k[3] is not None]
            for branch in options:
                original=trials.get((family,branch['profile'],(),None))
                if original is None:continue
                cutoff=branch['recovery_time_s']
                idx=min(len(branch.get('trace',[])),len(original.get('trace',[])),
                        round(cutoff/0.05))-1
                if idx<0:continue
                x=branch['trace'][idx];y=original['trace'][idx]
                parity=(abs(x['ego_target_speed_mps']-y['ego_target_speed_mps'])<1e-6 and
                        all(math.dist(x['xy_by_role'][role],y['xy_by_role'][role])<0.5
                            for role in ('ego','follower_a','follower_b')) and
                        all(abs(x[z]-y[z])<1. for z in
                            ('ego_speed_mps','follower_a_speed_mps','follower_b_speed_mps')))
                m=branch.get('metrics',{});b=original.get('metrics',{})
                backup_safe=bool(m.get('valid') and not m.get('ego_collision') and
                    not m.get('non_ego_collision') and m.get('ego_progress_m',0)>=10.)
                recovery_results.append({'anchor':anchor,'seed':seed,'family':family,
                    'profile':branch['profile'],'recovery':branch['recovery'],
                    'prefix_state_parity':parity,
                    'backup_safe_in_finite_family':backup_safe if parity else None,
                    'original_had_collision':bool(b.get('ego_collision') or b.get('non_ego_collision'))})
        # If the same ego action has a CARLA-physics valid response under
        # a separately controlled joint-search family, this is a bounded
        # constructive witness, not a proof that every human could coordinate.
        search=[v for k,v in trials.items() if k[0]=='joint_witness']
        for q in search:
            m=q.get('metrics',{})
            if not m.get('valid') or m.get('ego_collision') or m.get('non_ego_collision') or m.get('hard_brake'):continue
            for family in ('tm','behavior'):
                bad=trials.get((family,q['profile'],(),None))
                if bad is None or not _same_initial(bad,q,initial_tolerance):continue
                bm=bad.get('metrics',{})
                if not bm.get('valid') or abs(bm['ego_progress_m']-m['ego_progress_m'])>progress_tolerance:continue
                if (abs(bm['follower_a_progress_m']-m['follower_a_progress_m'])>progress_tolerance or
                    abs(bm['follower_b_progress_m']-m['follower_b_progress_m'])>progress_tolerance):continue
                if bm.get('non_ego_pair_collision'):
                    joint_results.append({'anchor':anchor,'seed':seed,'ego_action':q['profile'],
                       'decentralized_family':family,'responder_targets':q['joint_targets'],
                       'status':'CARLA_BOUNDED_JOINT_WITNESS_NOT_HUMAN_PROOF'})
    strict=[x for x in out if x.get('strict_third_party_collision_preventable')]
    burdens=[x for x in out if x.get('hard_brake_preventable_proxy')]
    # Count cross-family agreement ONLY within the same anchor/seed/bad/good,
    # not simply both families having SOME collision in a scenario.
    index=collections.defaultdict(set)
    for x in strict:index[(x['anchor'],x['seed'],x['bad'],x['alternative'])].add(x['family'])
    common=[]
    for k,v in index.items():
        if not {'tm','behavior'}<=v:continue
        tr=coll.get((k[0],k[1]),{})
        tm_bad=tr.get(('tm',k[2],(),None))
        tm_good=tr.get(('tm',k[3],(),None))
        ba_bad=tr.get(('behavior',k[2],(),None))
        ba_good=tr.get(('behavior',k[3],(),None))
        if (tm_bad and tm_good and ba_bad and ba_good and
            _same_initial(tm_bad,ba_bad,initial_tolerance) and
            _same_initial(tm_good,ba_good,initial_tolerance)):
            common.append({'anchor':k[0],'seed':k[1],'bad':k[2],'alternative':k[3]})
    summary={'trials':len(rows),'paired_contrasts':len(out),
        'valid_paired_contrasts':sum(x.get('eligible',False) for x in out),
        'strict_collision_contrasts':len(strict),'hard_brake_proxy_contrasts':len(burdens),
        'cross_family_same_action_strict_cases':len(common),
        'bounded_joint_safe_witnesses':len(joint_results),
        'recovery_trials':len(recovery_results),
        'recovery_prefix_parity_pass':sum(x['prefix_state_parity'] for x in recovery_results),
        'recoveries_with_safe_suffix':sum(x['backup_safe_in_finite_family'] is True for x in recovery_results),
        'human_H2_proven':False,'WOMD_case_reproduced':False,'H4_baseline_run':False,
        'limitations':['Same CARLA simulator/physics across policy families',
          'TM and BehaviorAgent do not establish human counterfactual truth',
          'Straight-lane three-vehicle convoy is an explicitly designed stress case',
          'No same-WOMD geometry transfer / route-matched CARLA replay',
          'Endpoint progress matching excludes unpaired speed profiles',
          'Positive joint witness is existential in finite searched control family',
          'No negative UNSAT conclusion from missing witness']}
    return {'summary':summary,'strict_contrasts':strict,'cross_family_agreement':common,
            'bounded_joint_witnesses':joint_results,'finite_backup_tests':recovery_results,
            'all_contrasts':out}


def main():
    p=argparse.ArgumentParser();p.add_argument('--jsonl',required=True)
    p.add_argument('--out',required=True);p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args();rows=[json.loads(x) for x in Path(a.jsonl).read_text().splitlines() if x.strip()]
    res=analyze(rows,progress_tolerance=a.progress_tolerance)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(res,ensure_ascii=False,indent=2))
    print(json.dumps(res['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
