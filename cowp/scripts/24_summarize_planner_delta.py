from __future__ import annotations
import argparse, json
from pathlib import Path

KEYS = (
    "CR", "EP", "KinematicsInfeasibilityRate", "OffroadRate", "Offroad", "WrongWayRate",
    "OffRouteRate", "PredFSR_episode", "PredCBS_episode", "PredOPR_min_episode",
    "PredHBCR_episode", "FallbackEpisodeRate",
)

def flatten(x, prefix=""):
    out={}
    if isinstance(x,dict):
        for k,v in x.items(): out.update(flatten(v, f"{prefix}/{k}" if prefix else k))
    elif isinstance(x,(int,float)) and not isinstance(x,bool): out[prefix]=float(x)
    return out

def pick(flat, key):
    exact=[(p,v) for p,v in flat.items() if p.split('/')[-1]==key]
    return exact[0][1] if exact else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',required=True); ap.add_argument('--candidate',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    a=json.load(open(args.reference))
    b=json.load(open(args.candidate))
    fa,fb=flatten(a),flatten(b)
    rows={}
    for k in KEYS:
        av,bv=pick(fa,k),pick(fb,k)
        if av is not None or bv is not None: rows[k]={"reference":av,"candidate":bv,"delta":None if av is None or bv is None else bv-av}
    Path(args.output).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
