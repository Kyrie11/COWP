"""Paired geometry/response-bank audit; never labels real-world collision causality.

Usage: python -m research_probe.compare_geometry --circle ... --oriented ... --output ...
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path


def _index(d):
    return {str(s['scenario_id']):s for s in d['scenarios'] if s.get('eligible')}

def _positive(s,key):return s.get(key,0)>0

def compare(circle,oriented):
    ca=_index(circle);oa=_index(oriented)
    shared=sorted(set(ca)&set(oa))
    if not shared:raise ValueError('no paired scenario IDs')
    def stats(index,ids):
        rows=[index[i] for i in ids]
        return {'positive_pairwise_joint_discordance':sum(_positive(s,'pairwise_joint_discordance') for s in rows),
                'joint_but_local_conflict':sum(_positive(s,'joint_but_local_conflict') for s in rows),
                'successor_collapse':sum(_positive(s,'temporal_collapse_proxy') for s in rows),
                'avoidable_collapse':sum(_positive(s,'avoidable_collapse_proxy') for s in rows),
                'initial_circle_overlap':sum(bool(s.get('initial_overlap_circle_pairs')) for s in rows),
                'initial_oriented_overlap':sum(bool(s.get('initial_overlap_oriented_box_pairs')) for s in rows),
                'all_candidate_incompatible':sum(bool(s.get('all_candidates_joint_incompatible')) for s in rows),
                'positive_no_initial_oriented_overlap':sum(_positive(s,'pairwise_joint_discordance') and not s.get('initial_overlap_oriented_box_pairs') for s in rows)}
    transitions={'both':[], 'circle_only':[], 'oriented_only':[], 'neither':[]}
    key_diffs=[]
    for i in shared:
        c,o=ca[i],oa[i]
        pc=_positive(c,'pairwise_joint_discordance');po=_positive(o,'pairwise_joint_discordance')
        key=('both' if pc and po else 'circle_only' if pc else 'oriented_only' if po else 'neither')
        transitions[key].append(i)
        cidx={x['candidate_index']:x for x in c['candidates']}
        oidx={x['candidate_index']:x for x in o['candidates']}
        changes=[k for k in cidx.keys()&oidx.keys() if (cidx[k]['pairwise'],cidx[k]['joint'])!=(oidx[k]['pairwise'],oidx[k]['joint'])]
        if changes:key_diffs.append({'scenario_id':i,'changed_candidate_indices':sorted(changes)})
    return {'status':'GEOMETRY_SENSITIVITY_NOT_SCIENTIFIC_GO','shared_eligible_scenes':len(shared),
      'shared_id_sha256':hashlib.sha256('\n'.join(shared).encode()).hexdigest(),
      'circle':stats(ca,shared),'oriented':stats(oa,shared),
      'positive_transitions':transitions,
      'candidate_label_changes':key_diffs,
      'next_action':'For any persistent noninitial positive: inspect oriented boxes, dynamic yaw and response feasibility; validate with Waymax geometry and independently calibrated reactive policies before claims.',
      'not_proven':['real-world event prevalence','ego intervention causality','joint continuous-space infeasibility','recursive feasibility failure','relative performance to published MPC/R3R']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--circle',type=Path,required=True);p.add_argument('--oriented',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    c=json.loads(a.circle.read_text());o=json.loads(a.oriented.read_text())
    if c.get('summary',{}).get('collision_geometry','circle')!='circle':p.error('first input is not circle geometry')
    if o.get('summary',{}).get('collision_geometry')!='oriented-box':p.error('second input is not oriented-box geometry')
    out=compare(c,o);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in out.items() if k not in ('positive_transitions','candidate_label_changes')},indent=2))
if __name__=='__main__':main()
