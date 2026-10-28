"""Prospective CARLA anchor overlap gate; reads frozen manifests, never outcomes.

A projected coordinate radius excludes repeated physical locations but not every
possible shared road segment. The actual route/corridor still needs a full audit.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path


def _man(path:Path):
    with Path(path).open(encoding='utf8') as f:
        data=json.loads(next(f))
    if data.get('type')!='manifest':raise ValueError('Frozen anchor file requires a manifest first row')
    coords=[]
    for a in data.get('anchors') or []:
        xy=a['coords']['ego']
        coords.append((float(xy[0]),float(xy[1])))
    if not coords:raise ValueError('No anchor coordinates in '+str(path))
    return data,coords


def compare(reference:Path,candidate:Path,radius_m:float=25.0):
    if radius_m<=0:raise ValueError('radius_m must be positive')
    mref,ref=_man(reference);mcand,cand=_man(candidate)
    if mref.get('map')!=mcand.get('map'):
        raise ValueError('Map mismatch: no physical correspondence can be asserted')
    overlap=[]
    for i,xy in enumerate(cand):
        closest=min((math.hypot(xy[0]-pt[0],xy[1]-pt[1]),j) for j,pt in enumerate(ref))
        if closest[0] < radius_m:
            overlap.append({'candidate_index':i,'reference_index':closest[1],
                            'distance_m':round(closest[0],3),'candidate_xy':xy})
    return {'reference':str(reference),'candidate':str(candidate),
            'radius_m':radius_m,'reference_anchors':len(ref),'candidate_anchors':len(cand),
            'overlap_count':len(overlap),'overlaps':overlap,
            'position_disjoint_heuristic':not overlap,
            'note':'No outcome labels read; radius only checks positions and cannot certify different corridor topology.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--radius-m',type=float,default=25.)
    p.add_argument('--output',type=Path)
    p.add_argument('--enforce',action='store_true')
    a=p.parse_args(argv)
    v=compare(a.reference,a.candidate,a.radius_m)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v[k] for k in ('reference_anchors','candidate_anchors','overlap_count',
                                      'position_disjoint_heuristic')},indent=2))
    if a.enforce and v['overlap_count']:
        raise SystemExit(3)
if __name__=='__main__':main()
