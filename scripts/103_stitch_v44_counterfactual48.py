from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def load(p): return json.loads(Path(p).read_text())
def manifest(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def sha(ids): return hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()
def num(x):
    if isinstance(x,bool): return float(x)
    if isinstance(x,(int,float)) and np.isfinite(float(x)): return float(x)
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',nargs=4,required=True)
    ap.add_argument('--counterfactual48-ids',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); payloads=[load(x) for x in a.inputs]; method=str(payloads[0].get('method'))
    if any(str(p.get('method'))!=method for p in payloads): raise SystemExit('method mismatch')
    records=[]
    for p in payloads:
        records.extend(p.get('scenario_results',[]))
    ids=[str(r['scenario_id']) for r in records]; target=manifest(a.counterfactual48_ids)
    if len(ids)!=48 or len(set(ids))!=48 or set(ids)!=set(target):
        raise SystemExit(f'48-ID stitch mismatch rows={len(ids)} unique={len(set(ids))} target={len(target)}')
    by={str(r['scenario_id']):r for r in records}; records=[by[x] for x in target]
    buckets={}
    for r in records:
        for k,v in (r.get('standard_metrics',{}) or {}).items():
            q=num(v)
            if q is not None: buckets.setdefault(k,[]).append(q)
    summary={k:float(np.mean(v)) for k,v in sorted(buckets.items()) if v}
    fb_num=fb_den=0.0
    for r in records:
        d=r.get('diagnostics',{}) or {}; n=float(d.get('steps',0) or 0); fb_num+=n*float(d.get('fallback_step_rate',0) or 0); fb_den+=n
    out={'schema_version':'v16.8.44_counterfactual48_stitch_v1','method':method,'checkpoint':payloads[0].get('checkpoint'),
         'scenario_ids_sha256':sha(target),'num_rollouts':48,'num_shards_merged':sum(int(p.get('num_shards_merged',1) or 1) for p in payloads),
         'input_files':a.inputs,'standard_metric_summary':summary,'ClosedLoopFallbackStepRate':fb_num/fb_den if fb_den else None,
         'scenario_results':records,'stitch_provenance':{'components':['lost7','retained3','induced9','remaining29'],'target_manifest':a.counterfactual48_ids}}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'method':method,'num_rollouts':48,'scenario_ids_sha256':out['scenario_ids_sha256'],'standard_metric_summary':summary},indent=2))
if __name__=='__main__': main()
