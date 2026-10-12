"""Conservative independent decision for four geometry/response experiments."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def audit(folder):
    folder=Path(folder)
    datasets={k:json.loads((folder/(k+'.json')).read_text()) for k in
      ('primary_circle','alternate_circle','primary_oriented','alternate_oriented')}
    selected={}
    for k,v in datasets.items():
        selected[k]={r['scenario_id']:r for r in v['scenarios'] if r.get('eligible')}
    ids=set.intersection(*(set(x) for x in selected.values()))
    cohort={}
    for sid in sorted(ids):
        rows={k:val[sid] for k,val in selected.items()}
        raw={k:r['pairwise_joint_discordance']>0 for k,r in rows.items()}
        h2={k:r['joint_but_local_conflict']>0 for k,r in rows.items()}
        initial=bool(rows['primary_oriented'].get('initial_overlap_oriented_box_pairs'))
        cohort[sid]={'H1_circle':raw['primary_circle'],'H1_oriented':raw['primary_oriented'],
           'H2_oriented':h2['primary_oriented'],'initial_oriented_overlap':initial,
           'candidate_count':len(rows['primary_oriented']['candidates']),
           'H1_candidate_invariant':rows['primary_oriented'].get('all_candidates_joint_incompatible',False)}
    h1=[sid for sid,r in cohort.items() if r['H1_oriented']]
    h2=[sid for sid,r in cohort.items() if r['H2_oriented']]
    count=lambda pred:sum(pred(r) for r in cohort.values())
    return {'scientific_claim_allowed':False,'shared_n':len(ids),
      'H1_circle_positive':count(lambda r:r['H1_circle']),
      'H1_oriented_positive':len(h1),
      'H1_oriented_without_initial_overlap':count(lambda r:r['H1_oriented'] and not r['initial_oriented_overlap']),
      'H1_oriented_positive_ids':h1,'H2_oriented_positive_ids':h2,
      'H2_oriented_without_initial_overlap':count(lambda r:r['H2_oriented'] and not r['initial_oriented_overlap']),
      'H3_positive':sum(datasets['primary_oriented']['scenarios'][i].get('temporal_collapse_proxy',0)>0 for i in range(len(datasets['primary_oriented']['scenarios']))),
      'caveat':'H1 effectively collapses under more realistic geometry; H2 still uses four fixed longitudinal responses and no ego-action dependence',
      'per_scene':cohort}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); d=audit(a.folder);a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(d,indent=2));print(json.dumps({k:v for k,v in d.items() if k!='per_scene'},indent=2))
