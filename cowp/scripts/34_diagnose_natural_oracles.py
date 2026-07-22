from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cowp.data.dataset import COWPNpzDataset


def _indices(n: int, limit: int) -> list[int]:
    return list(range(n)) if limit <= 0 or limit >= n else sorted(set(np.linspace(0, n - 1, limit, dtype=np.int64).tolist()))


def _current(d: dict[str, np.ndarray]) -> np.ndarray | None:
    h = d.get("state/history")
    if h is not None:
        h = np.asarray(h)
        if h.ndim == 3 and h.shape[-1] >= 11:
            return h[:, -1, :11].astype(np.float32)
    def f(name: str):
        x = d.get(f"state/current/{name}")
        return None if x is None else np.asarray(x).reshape(-1).astype(np.float32)
    x, y = f("x"), f("y")
    if x is None or y is None:
        return None
    n = len(x); s = np.zeros((n, 11), np.float32); s[:, 0] = x; s[:, 1] = y
    for c, names in [(3,("length",)),(4,("width",)),(6,("bbox_yaw","heading","yaw")),(7,("velocity_x","vx")),(8,("velocity_y","vy")),(10,("valid",))]:
        for name in names:
            v=f(name)
            if v is not None: s[:,c]=v[:n]; break
    s[:,9]=np.linalg.norm(s[:,7:9],axis=-1)
    return s


def _bank(cur: np.ndarray, H: int, dt: float) -> np.ndarray:
    t=(np.arange(H,dtype=np.float32)+1)*dt
    yaw0=float(cur[6]); speed0=float(max(cur[9],np.linalg.norm(cur[7:9]),0.0))
    accels=(-3.0,-1.5,0.0,1.5,3.0); yaw_rates=(-0.14,0.0,0.14)
    out=[]
    for a in accels:
        for w in yaw_rates:
            speed=np.maximum(0.0,speed0+a*t)
            yaw=yaw0+w*t
            vx=speed*np.cos(yaw); vy=speed*np.sin(yaw)
            x=cur[0]+np.cumsum(vx)*dt; y=cur[1]+np.cumsum(vy)*dt
            tr=np.zeros((H,7),np.float32)
            tr[:,0]=x; tr[:,1]=y; tr[:,2]=yaw; tr[:,3]=vx; tr[:,4]=vy; tr[:,5]=max(float(cur[3]),0.1); tr[:,6]=max(float(cur[4]),0.1)
            out.append(tr)
    return np.stack(out)


def _stats(x: list[float]) -> dict:
    a=np.asarray(x,dtype=np.float64); a=a[np.isfinite(a)]
    if not len(a): return {"count":0,"mean":None,"p50":None,"p90":None}
    return {"count":int(len(a)),"mean":float(a.mean()),"p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90))}


def main() -> None:
    ap=argparse.ArgumentParser(description="Measure simple kinematic oracle coverage of COWP natural-option labels.")
    ap.add_argument("--cache-dir",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--max-scenes",type=int,default=2000); ap.add_argument("--dt",type=float,default=0.1)
    args=ap.parse_args()
    ds=COWPNpzDataset(args.cache_dir); vals=defaultdict(list); counts=defaultdict(int)
    for idx in _indices(len(ds),args.max_scenes):
        d=ds.load(idx); s=_current(d)
        nat=np.asarray(d.get("cowp/natural/traj",[]),np.float32); valid=np.asarray(d.get("cowp/natural/valid",[]),bool)
        source=np.asarray(d.get("cowp/natural/source",np.zeros(valid.shape,np.int64)),np.int64)
        inp=np.asarray(d.get("cowp/critical/input_index",d.get("cowp/critical/track_index",[])),np.int64)
        if s is None or nat.ndim!=4 or valid.shape!=nat.shape[:2]: continue
        H=nat.shape[2]
        for a in range(min(len(inp),nat.shape[0])):
            j=int(inp[a])
            if j<0 or j>=len(s): continue
            bank=_bank(s[j],H,args.dt)
            for m in np.flatnonzero(valid[a]):
                gt=nat[a,m]
                if not np.isfinite(gt[:,:2]).all(): continue
                for sec in (1,3,5,8):
                    h=min(H,max(1,round(sec/args.dt)))
                    ade=np.linalg.norm(bank[:,:h,:2]-gt[None,:h,:2],axis=-1).mean(axis=-1)
                    best=float(ade.min())
                    vals[f"all/{sec}s"].append(best)
                    vals[f"source_{int(source[a,m])}/{sec}s"].append(best)
                counts["natural_roots"]+=1
    report={"cache_dir":str(Path(args.cache_dir)),"sampled_scenes":min(len(ds),args.max_scenes),"counts":dict(counts),"kinematic_bank_minade_m":{k:_stats(v) for k,v in vals.items()}}
    full=report["kinematic_bank_minade_m"].get("all/8s",{})
    report["interpretation"]={
        "oracle_8s_mean_m":full.get("mean"),
        "recommended_model_gate_m":None if full.get("mean") is None else float(full["mean"])+6.0,
        "warning":"A high kinematic oracle does not prove labels are wrong, but >20 m together with a large first-step alignment error strongly suggests track/coordinate misalignment."
    }
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
