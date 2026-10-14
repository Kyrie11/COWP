"""Re-audit strict H2 and stratify screening endpoints WITHOUT claiming human causality.

Runs on the old matched_dev.json even when the source NPZ cohort is unavailable.
Scenario-level units are never conflated with profile-pair units.  Exact-ID
cross-regime intersections are reported separately from different-model proof.
"""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path

REGIMES=('cautious','assertive','heterogeneous')

def wilson(k,n,z=1.96):
    if not n:return [None,None]
    p=k/n; d=1+z*z/n; mid=(p+z*z/(2*n))/d
    rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,mid-rad),min(1,mid+rad)]

def audit(matched,inventory=None,cohort_manifest=None):
    j=json.loads(Path(matched).read_text())
    allrows=j.get('rows',[]); eligible=[x for x in allrows if x.get('eligible')]
    errors=j.get('errors',[])
    if len(allrows)+len(errors)!=j['summary']['attempted']:
        raise ValueError('Matched dataset attempts not equal completed+errors')
    inv=json.loads(Path(inventory).read_text()) if inventory else None
    man=json.loads(Path(cohort_manifest).read_text()) if cohort_manifest else None
    manifest_by_id={x['scenario_id']:x for x in man['entries']} if man else {}
    n=len(eligible);evidence=[];cross_exact=[];near_union=set();burden_union=set()
    summary_by_regime={r:Counter() for r in REGIMES}
    for x in eligible:
        sid=x['scenario_id'];e={'scenario_id':sid,'partition':'development',
            'strict_regimes':[],'strict_common_pair_regimes':[],'burden_regimes':[],
            'near_pair_regimes':[],'ego_collision_preventable_regimes':[],
            'strict_same_pair_keys':[],'burden_proxy_count':0,'near_pair_proxy_count':0}
        if sid in manifest_by_id:
            e['partition']=manifest_by_id[sid].get('evaluation_partition','unknown')
        per_regime_keys={}
        for reg in REGIMES:
            actions=x['regimes'][reg]['actions']
            strict=x['regimes'][reg]['strict_matches']
            burden=x['regimes'][reg]['matched_burden_proxy_cases']
            near=x['regimes'][reg].get('near_pair_proxy_cases',[])
            if strict:e['strict_regimes'].append(reg);summary_by_regime[reg]['strict_scenes']+=1
            if burden:e['burden_regimes'].append(reg);summary_by_regime[reg]['burden_scenes']+=1;burden_union.add(sid)
            if near:e['near_pair_regimes'].append(reg);summary_by_regime[reg]['near_scenes']+=1;near_union.add(sid)
            e['burden_proxy_count']+=len(burden);e['near_pair_proxy_count']+=len(near)
            # Object IDs in previously generated rows are local; don't silently call them raw WOMD IDs.
            per_regime_keys[reg]={(v['bad_profile'],v['alternative_profile'],tuple(map(tuple,v['pair_ids']))) for v in strict}
            ego_failure=False
            # New descriptive ENDPOINT: directly preventable ego collision with
            # a progress-matched safe alternative within same amplitude/base.
            profiles=x['profile_controls']
            for name,a in actions.items():
                if not a['ego_any_collision']:continue
                for alt,b in actions.items():
                    if alt==name or not name.startswith('b') or not alt.startswith('b'):continue
                    if a['base_acceleration']!=b['base_acceleration']:continue
                    if abs(a['pulse_amplitude'])!=abs(b['pulse_amplitude']):continue
                    if a['pulse_amplitude']!=-b['pulse_amplitude']:continue
                    if b['any_collision']:continue
                    if abs(a['ego_route_tangent_progress_proxy_m']-b['ego_route_tangent_progress_proxy_m'])>3.:continue
                    ego_failure=True
            if ego_failure:
                e['ego_collision_preventable_regimes'].append(reg)
                summary_by_regime[reg]['ego_direct_scenes']+=1
        keys=defaultdict(set)
        for r,rs in per_regime_keys.items():
            for key in rs:keys[key].add(r)
        exact=[(k,sorted(v)) for k,v in keys.items() if len(v)>=2]
        e['strict_same_pair_keys']=[{'bad_profile':k[0],'alternative_profile':k[1],
              'pair_local_indices':[list(pair) for pair in k[2]],'regimes':rs} for k,rs in exact]
        e['strict_common_pair_regimes']=sorted(set().union(*(set(rs) for _,rs in exact))) if exact else []
        if exact:cross_exact.append(sid)
        e['candidate_rank']=(100 if exact else 0)+20*len(e['strict_regimes'])+3*len(e['near_pair_regimes'])+len(e['burden_regimes'])
        evidence.append(e)
    evidence.sort(key=lambda d:(-d['candidate_rank'],d['scenario_id']))
    saved=len(manifest_by_id) if man else None
    full=(inv and inv.get('complete_names') and inv.get('found')==150 and not inv.get('missing_indices'))
    result={'status':'SCREENING_ONLY_NOT_HUMAN_CAUSAL_CLAIM',
      'coverage':{'inventory_150_names_complete':bool(full) if inv else None,
                  'matched_attempted':j['summary']['attempted'],'matched_eligible':n,
                  'saved_cohort':saved,'examined_total':43695 if False else None,
                  'important':'150 files scanned does NOT mean 150 files counterfactually simulated'},
      'strict_h2':{'scene_any_regime':sum(bool(e['strict_regimes']) for e in evidence),
            'same_pair_same_action_repeated_IDM_family_scenes':len(cross_exact),
            'all_three_regimes_exact':sum(bool(x.get('all_three_regimes_shared_matched_pair_keys')) for x in eligible),
            'independent_model_families_run':1,'independent_model_validation':False,
            'real_human_counterfactual':False,'regime_counts':{r:summary_by_regime[r]['strict_scenes'] for r in REGIMES},
            'any_regime_wilson95':wilson(sum(bool(e['strict_regimes']) for e in evidence),n)},
      'other_endpoints':{'burden_any_regime_scenes':len(burden_union),
          'near_pair_any_regime_scenes':len(near_union),
          'ego_direct_preventable':{r:summary_by_regime[r]['ego_direct_scenes'] for r in REGIMES},
          'burden_by_regime':{r:summary_by_regime[r]['burden_scenes'] for r in REGIMES},
          'near_pair_by_regime':{r:summary_by_regime[r]['near_scenes'] for r in REGIMES}},
      'missing_for_H2':['map/route-valid executed ego alternatives','non-ego model calibrated to observed future',
          'at least two structurally independent response models','paired-seed closed-loop intervention',
          'non-ego pair full object-ID parity in Waymax','untouched holdout under frozen protocol',
          'actual competing planners evaluated'],'candidate_scenes':evidence}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matched',required=True,type=Path)
    ap.add_argument('--inventory',type=Path)
    ap.add_argument('--cohort-manifest',type=Path)
    ap.add_argument('--out-dir',required=True,type=Path)
    a=ap.parse_args();o=a.out_dir;o.mkdir(parents=True,exist_ok=True)
    result=audit(a.matched,a.inventory,a.cohort_manifest)
    (o/'h2_evidence_gate.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    with (o/'h2_triage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=('scenario_id','partition','candidate_rank','strict_regimes',
           'strict_common_pair_regimes','burden_regimes','near_pair_regimes','ego_collision_preventable_regimes'))
        w.writeheader()
        for x in result['candidate_scenes']:
            if x['partition']!='development':continue
            w.writerow({k:(','.join(x[k]) if isinstance(x[k],list) else x[k]) for k in w.fieldnames})
    ids=[x['scenario_id'] for x in result['candidate_scenes'] if x['partition']=='development'
         and (x['strict_regimes'] or x['near_pair_regimes'])]
    (o/'h2_priority_dev_ids.txt').write_text('\n'.join(ids)+'\n' if ids else '')
    print(json.dumps({k:v for k,v in result.items() if k!='candidate_scenes'},indent=2))
if __name__=='__main__':main()
