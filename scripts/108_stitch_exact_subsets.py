from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def load(p): return json.loads(Path(p).read_text())
def ids(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def sha(x): return hashlib.sha256('\n'.join(x).encode()).hexdigest()
def num(x):
    if isinstance(x,bool): return float(x)
    if isinstance(x,(int,float)) and np.isfinite(float(x)): return float(x)
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--inputs',nargs='+',required=True); ap.add_argument('--target-ids',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    ps=[load(x) for x in a.inputs]; method=str(ps[0].get('method'))
    if any(str(x.get('method'))!=method for x in ps): raise SystemExit('method mismatch')
    rec=[]
    for x in ps: rec.extend(x.get('scenario_results',[]))
    target=ids(a.target_ids); got=[str(r['scenario_id']) for r in rec]
    if len(got)!=len(target) or len(set(got))!=len(got) or set(got)!=set(target): raise SystemExit(f'stitch mismatch got={len(got)} target={len(target)}')
    by={str(r['scenario_id']):r for r in rec}; rec=[by[x] for x in target]
    buckets={}
    for r in rec:
        for k,v in (r.get('standard_metrics',{}) or {}).items():
            q=num(v)
            if q is not None:buckets.setdefault(k,[]).append(q)
    summary={k:float(np.mean(v)) for k,v in buckets.items() if v}
    out={'schema_version':'v16.8.45_exact_subset_stitch_v1','method':method,'checkpoint':ps[0].get('checkpoint'),'rcrso_checkpoint':ps[0].get('rcrso_checkpoint'),'scenario_ids_sha256':sha(target),'num_rollouts':len(rec),'input_files':a.inputs,'standard_metric_summary':summary,'scenario_results':rec}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'method':method,'num_rollouts':len(rec),'scenario_ids_sha256':out['scenario_ids_sha256']},indent=2))
if __name__=='__main__':main()
