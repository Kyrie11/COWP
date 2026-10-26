"""Audit CARLA interventions. Keeps failures, paired denominators, and signed effects."""
from __future__ import annotations
import argparse, collections, json, math, random, statistics
from pathlib import Path
from .science import Protocol, assess_pair, same_initial


def read_jsonl(path):
    """Read *completed* outcome panels; reject partial runs as scientific data.

    The manifest-only anchor file is intentionally not a valid outcome panel.
    Combined panels have per-source counts in provenance rather than a run_end.
    """
    rows=[json.loads(ln) for ln in Path(path).read_text(encoding='utf8').splitlines() if ln.strip()]
    manifests=[r for r in rows if r.get('type')=='manifest']
    if len(manifests)!=1 or manifests[0].get('simulator')!='CARLA':
        raise ValueError('Expected exactly one CARLA experiment manifest: '+str(path))
    manifest=manifests[0]
    trials=[r for r in rows if r.get('type')=='trial']
    if manifest.get('archetype')=='MULTI':
        sources=manifest.get('source_manifests')
        if not sources or sum(int(x['count']) for x in sources)!=len(trials):
            raise ValueError('Incomplete combined CARLA panel: '+str(path))
        if any(r.get('archetype') not in {x['archetype'] for x in sources} for r in trials):
            raise ValueError('Combined CARLA panel has unexpected archetype: '+str(path))
        if any(r.get('type')=='run_end' for r in rows):
            raise ValueError('Combined CARLA panel unexpectedly has a run_end')
    else:
        endings=[r for r in rows if r.get('type')=='run_end']
        if (len(endings)!=1 or rows[-1] is not endings[0] or
            endings[0].get('completed') is not True or endings[0].get('trial_count')!=len(trials)):
            raise ValueError('Incomplete CARLA outcome panel (missing/mismatched run_end): '+str(path))
        if any(r.get('archetype')!=manifest.get('archetype') for r in trials):
            raise ValueError('Trial archetype disagrees with CARLA manifest: '+str(path))
    return rows


def cluster_bootstrap_mean(rows, seed=20261026, n=1000):
    """Bootstrap at the *anchor* (physical corridor) level, not at the action level."""
    cluster=collections.defaultdict(list)
    for row in rows: cluster[(row.get('archetype','convoy'),row['anchor'])].append(float(row['signed_collision_effect']))
    if not cluster:return None
    groups=list(cluster.values());rnd=random.Random(seed)
    vals=[]
    for i in range(n):
        picked=[groups[rnd.randrange(len(groups))] for j in groups]
        flat=[v for block in picked for v in block]
        vals.append(statistics.mean(flat))
    vals.sort()
    return {'estimate':statistics.mean([v for block in groups for v in block]),
            'ci95_cluster_bootstrap':[vals[int(.025*n)],vals[min(n-1,int(.975*n))]],
            'bootstrap_clusters':len(groups),'iterations':n,'unit':'signed_collision_difference'}


def audit(rows, protocol=None):
    p = protocol or Protocol()
    manifest = next((r for r in rows if r.get('type')=='manifest'),None)
    if not manifest:raise ValueError('Missing simulator manifest; cannot verify experiment identity')
    if manifest.get('simulator') != 'CARLA': raise ValueError('Expected CARLA trial format')
    trial_rows=[r for r in rows if r.get('type')=='trial']
    trials={}; duplicate=[]
    for r in trial_rows:
        key=(r.get('archetype',manifest.get('archetype','convoy')),r.get('anchor'),r.get('seed'),r.get('family'),r.get('profile'),
             tuple(r.get('joint_targets') or []),r.get('recovery'))
        if key in trials:duplicate.append(list(key))
        trials[key]=r
    if duplicate:raise ValueError('Duplicate trial keys! ' + str(duplicate[:3]))
    pairs=[];missing=[]
    contexts=sorted({(k[0],k[1],k[2]) for k in trials},key=str)
    for archetype,anchor,seed in contexts:
        for f in p.families:
            for a,b in p.contrast_actions:
                ka=(archetype,anchor,seed,f,a,(),None);kb=(archetype,anchor,seed,f,b,(),None)
                if ka not in trials or kb not in trials:
                    missing.append({'archetype':archetype,'anchor':anchor,'seed':seed,'family':f,'actions':[a,b],
                                    'missing':[q for q,k in zip([a,b],[ka,kb]) if k not in trials]})
                    continue
                measured=assess_pair(trials[ka],trials[kb],p)
                pairs.append({'archetype':archetype,'anchor':anchor,'seed':seed,'family':f,'action_a':a,'action_b':b,**measured})
    valid=[q for q in pairs if q['eligible']]
    grp=collections.defaultdict(dict)
    for q in valid:grp[(q['archetype'],q['anchor'],q['seed'],q['action_a'],q['action_b'])][q['family']]=q
    common=[];disagree=[]
    for k,v in grp.items():
        if not all(f in v for f in p.families):continue
        # Compare the same physical initial state across responder families,
        # not merely two within-family pairs with identical scene IDs.
        f0,f1=p.families
        ref=(k[0],k[1],k[2])
        same_a,same_a_reason=same_initial(
            trials[(ref[0],ref[1],ref[2],f0,k[3],(),None)],
            trials[(ref[0],ref[1],ref[2],f1,k[3],(),None)],p)
        same_b,same_b_reason=same_initial(
            trials[(ref[0],ref[1],ref[2],f0,k[4],(),None)],
            trials[(ref[0],ref[1],ref[2],f1,k[4],(),None)],p)
        if not same_a or not same_b:continue
        effects=[v[f]['signed_collision_effect'] for f in p.families]
        item={'archetype':k[0],'anchor':k[1],'seed':k[2],'action_a':k[3],'action_b':k[4],
              'families':list(p.families),'signed_effects':effects,
              'cross_family_initial_parity':True}
        if effects[0]!=0 and len(set(effects))==1:common.append(item)
        if len(set(effects))>1:disagree.append(item)
    invalid=collections.Counter(x.get('reason') or 'other' for x in pairs if not x['eligible'])
    summary={
        'archetypes':sorted(set(k[0] for k in trials)),
        'total_trials':len(trial_rows),'complete_preregistered_contrasts':len(pairs),
        'missing_contrast_arms':len(missing),'valid_task_matched_contrasts':len(valid),
        'signed_effect_a_worse':sum(q['signed_collision_effect']>0 for q in valid),
        'signed_effect_a_better':sum(q['signed_collision_effect']<0 for q in valid),
        'cross_family_matching_collision_effects':len(common),
        'cross_family_response_disagreement':len(disagree),
        'invalid_reasons':dict(invalid),
        'strict_physical_collision_is_not_human_causation':True,
        'calibrated_independent_human_model':False,
        'rq3_strong_baselines_executed':False,
        'model_disagreement_is_not_a_proof_of_decision_error':True,
    }
    return {'manifest':manifest,'protocol':{'progress_m':p.progress_m,'pairs':p.contrast_actions,'families':p.families},
            'summary':summary,'bootstrap':cluster_bootstrap_mean(valid),
            'contrasts':pairs,'missing':missing,'cross_family_same_effect':common,'cross_family_disagreement':disagree}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--progress-tolerance',type=float,default=3.)
    a=p.parse_args(argv)
    if a.progress_tolerance <=0:raise ValueError('progress tolerance must be positive')
    proto=Protocol(progress_m=a.progress_tolerance)
    res=audit(read_jsonl(a.input),proto)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(res['summary'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
