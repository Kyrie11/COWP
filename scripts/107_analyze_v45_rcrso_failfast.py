from __future__ import annotations
import argparse,json,math
from pathlib import Path

def load(p): return json.loads(Path(p).read_text())
def rows(d): return {str(r['scenario_id']):r for r in d.get('scenario_results',[])}
def ids(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def coll(r):
    v=(r.get('standard_metrics',{}) or {}).get('CollisionRate',0.0)
    return isinstance(v,(int,float)) and math.isfinite(float(v)) and float(v)>0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',nargs='+',required=True); ap.add_argument('--manifests',nargs='+',required=True); ap.add_argument('--total-lost7-manifest',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    if len(a.results)!=len(a.manifests): raise SystemExit('results/manifests length mismatch')
    all_rows={}; seen=[]
    for rp,mp in zip(a.results,a.manifests):
        rr=rows(load(rp)); mm=ids(mp)
        if set(rr)!=set(mm) or len(rr)!=len(mm): raise SystemExit(f'mismatch {rp} {mp}')
        if set(seen)&set(mm): raise SystemExit('batch overlap')
        seen+=mm; all_rows.update(rr)
    target=ids(a.total_lost7_manifest)
    if seen != target[:len(seen)]: raise SystemExit('progressive batches must follow the frozen lost7 order exactly')
    rescued=[sid for sid in seen if not coll(all_rows[sid])]; remaining=len(target)-len(seen); possible=len(rescued)+remaining
    passed=len(rescued)>=2; impossible=possible<2; continue_run=(not passed and not impossible and remaining>0)
    out={'schema_version':'v16.8.45_rcrso_lost7_progressive_v1','evaluated':len(seen),'rescued':len(rescued),'rescued_ids':rescued,'remaining':remaining,'max_possible_rescues':possible,'lost7_gate':{'pass':passed,'threshold':'at least 2/7 newly rescued'},'continue_progressive':continue_run,'mathematically_impossible':impossible,'frozen_order':target}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
    if impossible: raise SystemExit(4)
if __name__=='__main__': main()
