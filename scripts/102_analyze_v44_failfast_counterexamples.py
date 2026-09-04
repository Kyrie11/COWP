from __future__ import annotations
import argparse, json, math
from pathlib import Path


def load(p): return json.loads(Path(p).read_text())
def rows(d): return {str(r['scenario_id']): r for r in d.get('scenario_results', [])}
def coll(r):
    v=(r.get('standard_metrics',{}) or {}).get('CollisionRate',0.0)
    return isinstance(v,(int,float)) and math.isfinite(float(v)) and float(v)>0

def ids(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]

def check_exact(result, manifest):
    rr=rows(load(result)); mm=ids(manifest)
    if len(mm)!=len(set(mm)) or set(rr)!=set(mm):
        raise SystemExit(f'manifest/result mismatch: result={len(rr)} manifest={len(mm)}')
    return rr,mm

def main():
    ap=argparse.ArgumentParser(description='V16.8.44 mathematically valid fail-fast counterexample gates.')
    ap.add_argument('--stage',choices=['lost7','rescue10','induced9'],required=True)
    ap.add_argument('--lost7-result'); ap.add_argument('--lost7-ids')
    ap.add_argument('--retained3-result'); ap.add_argument('--retained3-ids')
    ap.add_argument('--induced9-result'); ap.add_argument('--induced9-ids')
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    out={'schema_version':'v16.8.44_failfast_counterexample_gate_v1','stage':a.stage,
         'classification':'development-only necessary-condition gate; never sufficient for promotion'}
    if a.stage in {'lost7','rescue10'}:
        lost,lost_ids=check_exact(a.lost7_result,a.lost7_ids)
        if len(lost_ids)!=7: raise SystemExit('lost7 must contain exactly 7 IDs')
        rescued=[sid for sid in lost_ids if not coll(lost[sid])]
        out['lost7_new_rescues']=len(rescued); out['lost7_new_rescue_ids']=rescued
        out['lost7_gate']={'pass':len(rescued)>=2,'threshold':'at least 2/7 newly rescued'}
        if a.stage=='lost7':
            out['interpretation']='If <2/7 are rescued, the frozen >=5/10 historical-rescue gate is mathematically impossible even if all three previously retained scenes remain safe.'
        else:
            kept,kept_ids=check_exact(a.retained3_result,a.retained3_ids)
            if len(kept_ids)!=3: raise SystemExit('retained3 must contain exactly 3 IDs')
            preserved=[sid for sid in kept_ids if not coll(kept[sid])]
            total=len(rescued)+len(preserved)
            out['previously_retained3_preserved']=len(preserved); out['previously_retained3_preserved_ids']=preserved
            out['historical_rescue10_retained']=total
            out['rescue10_gate']={'pass':total>=5,'threshold':'at least 5/10 retained'}
            out['interpretation']='Only if >=5/10 historical RVR rescues are collision-free is it rational to test the 9 historical induced counterexamples.'
    else:
        rr,mm=check_exact(a.induced9_result,a.induced9_ids)
        if len(mm)!=9: raise SystemExit('induced9 must contain exactly 9 IDs')
        avoided=[sid for sid in mm if not coll(rr[sid])]
        out['historical_induced9_avoided']=len(avoided); out['historical_induced9_avoided_ids']=avoided
        out['induced9_gate']={'pass':len(avoided)>=7,'threshold':'at least 7/9 avoided'}
        out['interpretation']='If <7/9 are avoided, the frozen counterfactual48 six-item conjunction gate is impossible and remaining29 must not be run.'
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
