from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cowp.core.config import load_config
from cowp.data.dataset import COWPNpzDataset
from cowp.label.safe_responses import expand_root_control_knots
from cowp.models.recourse_set_operator import RCRSOConfig, build_rcrso_features_np
from cowp.waymax_eval.policy_wrapper import (
    _constant_velocity_trajectory_from_state_np,
    _root_conditioned_control_reachable_response_profiles_np,
    _shift_append_terminal_reference_np,
    _verified_root_conditioned_recourse_set_profiles_np,
)


def _scenario_id(path: Path, data: dict[str, np.ndarray]) -> str:
    for key in ("scenario/id", "womd/scenario/id", "scenario_id"):
        if key in data:
            x = np.asarray(data[key]).reshape(-1)
            if x.size:
                v = x[0]
                if isinstance(v, (bytes, np.bytes_)):
                    return v.decode("utf-8")
                return str(v)
    return path.stem


def _load_ids(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            continue
        if p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            vals = obj.get("scenario_ids", obj) if isinstance(obj, dict) else obj
            if isinstance(vals, list):
                out.update(str(x) for x in vals)
        else:
            out.update(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
    return out


def _np_state11(data: dict[str, np.ndarray]) -> tuple[np.ndarray, int, np.ndarray]:
    hist = data.get("state/history")
    if hist is not None:
        h = np.asarray(hist, dtype=np.float32)
        if h.ndim == 4:
            h = h[0]
        cur = h[:, -1]
        state = np.zeros((cur.shape[0], 11), dtype=np.float32)
        state[:, 0:2] = cur[:, 0:2]
        state[:, 3:5] = cur[:, 7:9]
        state[:, 5] = np.linalg.norm(state[:, 3:5], axis=-1)
        state[:, 6] = cur[:, 6]
        state[:, 7:10] = cur[:, 3:6]
        state[:, 10] = cur[:, 10] if cur.shape[1] > 10 else 1.0
    else:
        def arr(name: str, default: float = 0.0) -> np.ndarray:
            x = data.get(f"state/current/{name}")
            if x is None:
                x = data.get(f"womd/state/current/{name}")
            if x is None:
                n = len(np.asarray(data.get("state/current/x", data.get("womd/state/current/x"))).reshape(-1))
                return np.full(n, default, dtype=np.float32)
            return np.asarray(x, dtype=np.float32).reshape(-1)
        x, y = arr("x"), arr("y")
        n = min(x.size, y.size)
        state = np.zeros((n, 11), dtype=np.float32)
        state[:, 0], state[:, 1] = x[:n], y[:n]
        state[:, 3], state[:, 4] = arr("velocity_x")[:n], arr("velocity_y")[:n]
        state[:, 5] = np.linalg.norm(state[:, 3:5], axis=-1)
        yaw = data.get("state/current/bbox_yaw", data.get("state/current/heading"))
        if yaw is not None: state[:, 6] = np.asarray(yaw, dtype=np.float32).reshape(-1)[:n]
        state[:, 7], state[:, 8], state[:, 9] = arr("length",4.8)[:n], arr("width",1.9)[:n], arr("height",1.6)[:n]
        state[:, 10] = arr("valid",1.0)[:n]
    is_sdc = data.get("state/is_sdc", data.get("womd/state/is_sdc"))
    if is_sdc is not None:
        mask = np.asarray(is_sdc).reshape(-1)[: state.shape[0]].astype(bool)
        sdc = int(np.flatnonzero(mask)[0]) if mask.any() else 0
    else:
        sdc = 0
    typ = data.get("state/type", data.get("womd/state/type"))
    object_types = np.asarray(typ, dtype=np.int32).reshape(-1)[: state.shape[0]] if typ is not None else np.zeros(state.shape[0], np.int32)
    if object_types.size < state.shape[0]:
        object_types = np.pad(object_types, (0, state.shape[0]-object_types.size))
    return state, sdc, object_types


def _roadgraph(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    xyz = data.get("roadgraph_samples/xyz", data.get("womd/roadgraph_samples/xyz"))
    if xyz is None:
        x = data.get("roadgraph_samples/x", data.get("womd/roadgraph_samples/x"))
        y = data.get("roadgraph_samples/y", data.get("womd/roadgraph_samples/y"))
        if x is None or y is None:
            return {"xy": np.zeros((0,2),np.float32), "heading": np.zeros(0,np.float32), "valid": np.zeros(0,bool), "types": np.zeros(0,np.int32)}
        xy = np.stack([np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)], axis=-1).astype(np.float32)
    else:
        a=np.asarray(xyz,dtype=np.float32)
        while a.ndim>2: a=a[0]
        xy=a.reshape(-1,a.shape[-1])[:,:2]
    valid=data.get("roadgraph_samples/valid",data.get("womd/roadgraph_samples/valid"))
    v=np.asarray(valid).reshape(-1)[:len(xy)].astype(bool) if valid is not None else np.ones(len(xy),bool)
    typ=data.get("roadgraph_samples/type",data.get("womd/roadgraph_samples/type"))
    t=np.asarray(typ,dtype=np.int32).reshape(-1)[:len(xy)] if typ is not None else np.zeros(len(xy),np.int32)
    d=data.get("roadgraph_samples/dir",data.get("womd/roadgraph_samples/dir"))
    if d is not None:
        dd=np.asarray(d,dtype=np.float32)
        while dd.ndim>2: dd=dd[0]
        heading=np.arctan2(dd[:len(xy),1],dd[:len(xy),0]).astype(np.float32)
    else:
        diff=np.gradient(xy,axis=0) if len(xy)>1 else np.zeros_like(xy)
        heading=np.arctan2(diff[:,1],diff[:,0]).astype(np.float32)
    return {"xy":xy,"heading":heading,"valid":v,"types":t}


def _response_to_normalized_knots(root: np.ndarray, response: np.ndarray, cfg: dict, knot_count: int) -> np.ndarray:
    dt=max(float(cfg.get("time",{}).get("dt",0.1)),1e-6)
    rs=np.linalg.norm(np.asarray(root)[:,3:5],axis=-1)
    qs=np.linalg.norm(np.asarray(response)[:,3:5],axis=-1)
    h=min(len(rs),len(qs))
    dv=(qs[:h]-rs[:h]).astype(np.float32)
    accel=np.diff(np.concatenate([[0.0],dv])).astype(np.float32)/dt
    idx=np.linspace(0,max(h-1,0),knot_count).round().astype(int)
    knots=accel[idx]
    c=cfg.get("candidate",{})
    dec=max(float(c.get("max_decel_mps2",6.0)),1e-6); acc=max(float(c.get("max_accel_mps2",4.0)),1e-6)
    return np.clip(np.where(knots<0,knots/dec,knots/acc),-1,1).astype(np.float32)


def _retained_roots(weights: np.ndarray, valid: np.ndarray, p_min: float, required_mass: float, min_roots: int) -> list[int]:
    w=np.where(np.asarray(valid,bool),np.maximum(np.asarray(weights,float),0.0),0.0)
    keep=w>=p_min
    if not keep.any(): return []
    w=np.where(keep,w,0.0); w=w/max(float(w.sum()),1e-12)
    order=np.argsort(-w,kind="stable")
    out=[]; mass=0.0
    for j in order:
        if w[j]<=0: continue
        out.append(int(j)); mass+=float(w[j])
        if len(out)>=min_roots and mass+1e-9>=required_mass: break
    return out


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--cache-dir",required=True)
    ap.add_argument("--output-root",required=True)
    ap.add_argument("--split",choices=["train","val","heldout"],required=True)
    ap.add_argument("--label-config",default="configs/label_cowp_v16_8.yaml")
    ap.add_argument("--data-config",default="configs/data.yaml")
    ap.add_argument("--eval-config",default="configs/eval_cowp_v16_8.yaml")
    ap.add_argument("--max-scenes",type=int,default=None)
    ap.add_argument("--max-examples-per-scene",type=int,default=64)
    ap.add_argument("--max-positive-controls",type=int,default=32)
    ap.add_argument("--max-negative-controls",type=int,default=32)
    ap.add_argument("--rich-sobol-proposals",type=int,default=32)
    ap.add_argument("--control-knots",type=int,default=8)
    ap.add_argument("--environment-cap",type=int,default=24)
    ap.add_argument("--forbidden-id-file",action="append",default=[])
    ap.add_argument("--num-shards",type=int,default=1)
    ap.add_argument("--shard-index",type=int,default=0)
    args=ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard {args.shard_index}/{args.num_shards}")
    cfg=load_config(args.label_config,args.data_config,args.eval_config)
    forbidden=_load_ids(list(args.forbidden_id_file))
    ds=COWPNpzDataset(args.cache_dir)
    outdir=Path(args.output_root)/args.split; outdir.mkdir(parents=True,exist_ok=True)
    rcrso_cfg=RCRSOConfig(control_knots=int(args.control_knots))
    manifest=[]; counts={"scenes":0,"examples":0,"forbidden_skipped":0,"positive_examples":0,"analytic_nonempty":0,"rich_verified":0}
    try:
        import torch
        sobol=torch.quasirandom.SobolEngine(dimension=int(args.control_knots),scramble=False)
        sobol_bank=(sobol.draw(int(args.rich_sobol_proposals)).cpu().numpy().astype(np.float32)*2.0-1.0) if args.rich_sobol_proposals>0 else np.zeros((0,args.control_knots),np.float32)
    except Exception:
        rng=np.random.default_rng(0); sobol_bank=rng.uniform(-1,1,size=(max(args.rich_sobol_proposals,0),args.control_knots)).astype(np.float32)

    for scene_i,path in enumerate(ds.paths):
        if scene_i % int(args.num_shards) != int(args.shard_index):
            continue
        if args.max_scenes is not None and counts["scenes"]>=args.max_scenes: break
        data=ds.load(scene_i,None); sid=_scenario_id(path,data)
        if sid in forbidden:
            counts["forbidden_skipped"]+=1; continue
        required=("cowp/candidates/trajectory","cowp/candidates/valid","cowp/critical/track_index","cowp/critical/valid","cowp/natural/traj","cowp/natural/weight","cowp/natural/source","cowp/natural/valid","cowp/natural/beta","cowp/response/traj","cowp/response/valid","cowp/response/is_safe","cowp/response/is_low_burden")
        if any(k not in data for k in required): continue
        state,sdc,obj=_np_state11(data); road=_roadgraph(data)
        cand=np.asarray(data["cowp/candidates/trajectory"],np.float32); cvalid=np.asarray(data["cowp/candidates/valid"],bool)
        crit=np.asarray(data["cowp/critical/track_index"],int); critv=np.asarray(data["cowp/critical/valid"],bool)
        nat=np.asarray(data["cowp/natural/traj"],np.float32); nw=np.asarray(data["cowp/natural/weight"],np.float32); ns=np.asarray(data["cowp/natural/source"],int); nv=np.asarray(data["cowp/natural/valid"],bool); beta=np.asarray(data["cowp/natural/beta"],float)
        rt=np.asarray(data["cowp/response/traj"],np.float32); rv=np.asarray(data["cowp/response/valid"],bool); rsafe=np.asarray(data["cowp/response/is_safe"],bool); rlow=np.asarray(data["cowp/response/is_low_burden"],bool)
        rroot=np.asarray(data.get("cowp/response/root_index",np.full(rv.shape,-1)),int)
        rbur=np.asarray(data.get("cowp/response/burden_total",np.zeros(rv.shape)),float)
        pair_rel=np.asarray(data.get("cowp/audit/pair_relevant",np.ones((cand.shape[0],crit.shape[0]),bool)),bool)
        counts["scenes"]+=1; made=0
        for k in range(min(cand.shape[0],cvalid.size)):
            if not cvalid[k]: continue
            ego_cur=cand[k]; ego_shift=_shift_append_terminal_reference_np(ego_cur,float(cfg.get("time",{}).get("dt",0.1)))
            # Stage-0 FullHypothesisRootCoverage is meaningful only when a
            # candidate hypothesis contains *all* retained roots of every included
            # audit-relevant actor.  Never truncate a hypothesis midway merely to
            # hit an example-count budget.  The budget is therefore applied only
            # between complete candidate groups; the first group may exceed it.
            group_contexts=[]
            for a in range(min(crit.size,critv.size,nat.shape[0])):
                j=int(crit[a])
                if not critv[a] or not (0<=j<state.shape[0]) or j==sdc or not pair_rel[k,a]:
                    continue
                for m in _retained_roots(nw[a],nv[a],0.03,0.75,2):
                    group_contexts.append((int(a),int(j),int(m)))
            if not group_contexts:
                continue
            if made>0 and made+len(group_contexts)>int(args.max_examples_per_scene):
                break
            counts["hypothesis_groups"] = int(counts.get("hypothesis_groups",0)) + 1
            for a,j,m in group_contexts:
                    root=nat[a,m]
                    env=[]
                    dists=np.linalg.norm(state[:,:2]-state[j,:2][None],axis=-1); order=np.argsort(dists)
                    for e in order:
                        e=int(e)
                        if e in (sdc,j) or state[e,10]<=0.5: continue
                        curcv=_constant_velocity_trajectory_from_state_np(state,e,len(root),cfg)
                        if curcv is None: continue
                        succ=state.copy(); shiftedcv=_constant_velocity_trajectory_from_state_np(succ,e,len(root),cfg)
                        if shiftedcv is None: continue
                        env.append({"agent_index":e,"object_type":int(obj[e]) if e<obj.size else 0,"trajectory":curcv,"shifted_trajectory":shiftedcv})
                        if len(env)>=args.environment_cap: break
                    pos_knots=[]; pos_burden=[]; pos_source=[]
                    if k<rv.shape[0] and a<rv.shape[1]:
                        for r in range(rv.shape[2]):
                            if not (rv[k,a,r] and rsafe[k,a,r] and rlow[k,a,r]): continue
                            if rroot.shape==rv.shape and int(rroot[k,a,r]) not in (-1,m): continue
                            pos_knots.append(_response_to_normalized_knots(root,rt[k,a,r],cfg,args.control_knots)); pos_burden.append(float(rbur[k,a,r])); pos_source.append(0)
                    analytic,ad=_root_conditioned_control_reachable_response_profiles_np(state,j,int(obj[j]) if j<obj.size else 0,root,float(beta[a]),ego_cur,ego_shift,env,road,cfg,root_ordinal=m,compatibility_cache={})
                    if analytic: counts["analytic_nonempty"]+=1
                    for rec in analytic:
                        pos_knots.append(_response_to_normalized_knots(root,rec["trajectory"],cfg,args.control_knots)); pos_burden.append(float(rec["burden"])); pos_source.append(1)
                    rejected_sobol=[]
                    if sobol_bank.size:
                        rich,rd=_verified_root_conditioned_recourse_set_profiles_np(state,j,int(obj[j]) if j<obj.size else 0,root,float(beta[a]),ego_cur,ego_shift,env,road,cfg,sobol_bank,root_ordinal=m,compatibility_cache={})
                        counts["rich_verified"]+=len(rich)
                        for rec in rich:
                            pos_knots.append(np.asarray(rec["rcrso_control_knots"],np.float32)); pos_burden.append(float(rec["burden"])); pos_source.append(2)
                        outcome_names=list(rd.get("proposal_outcomes", []))
                        reason_code={
                            "no_low_burden_static_control":0,
                            "roadgraph_or_waymax_kinematic_reject":1,
                            "ego_current_reject":2,
                            "ego_shift_reject":3,
                            "environment_current_or_shift_reject":4,
                        }
                        for qi,q in enumerate(sobol_bank):
                            reason=outcome_names[qi] if qi < len(outcome_names) else "unknown"
                            if reason.startswith("root_verified_set_nonempty"):
                                continue
                            rejected_sobol.append((np.asarray(q,np.float32),int(reason_code.get(reason,5))))
                    # Exact dedup in knot space, keep lower burden representative.
                    unique={}
                    for q,b,src in zip(pos_knots,pos_burden,pos_source):
                        key=np.ascontiguousarray(np.round(q,6).astype(np.float32)).tobytes(); old=unique.get(key)
                        if old is None or b<old[1]: unique[key]=(q,b,src)
                    vals=sorted(unique.values(),key=lambda x:(x[1],x[2]))[:args.max_positive_controls]
                    P=args.max_positive_controls; targets=np.zeros((P,args.control_knots),np.float32); tvalid=np.zeros(P,bool); tb=np.zeros(P,np.float32); ts=np.full(P,-1,np.int64)
                    for qi,(q,b,src) in enumerate(vals): targets[qi]=q; tvalid[qi]=True; tb[qi]=b; ts[qi]=src
                    # Keep verifier-rejected Sobol controls as hard negatives.  To
                    # focus supervision near the feasible-set boundary without an
                    # outcome-tuned margin, deterministically retain rejected knots
                    # closest to any verified positive (or smallest norm when this
                    # root has no positive).  These labels train proposal ordering
                    # only and never alter the online hard verifier.
                    N=args.max_negative_controls; neg=np.zeros((N,args.control_knots),np.float32); nvalid=np.zeros(N,bool); nreason=np.full(N,-1,np.int64)
                    if rejected_sobol and N>0:
                        pos_arr=np.stack([x[0] for x in vals],axis=0).astype(np.float32) if vals else np.zeros((0,args.control_knots),np.float32)
                        scored=[]
                        for q,code in rejected_sobol:
                            if len(pos_arr): score=float(np.min(np.mean((pos_arr-q[None,:])**2,axis=1)))
                            else: score=float(np.mean(q*q))
                            scored.append((score,np.ascontiguousarray(np.round(q,6).astype(np.float32)).tobytes(),q,code))
                        seen_neg=set(); kept=[]
                        for _,key,q,code in sorted(scored,key=lambda x:(x[0],x[1])):
                            if key in seen_neg: continue
                            seen_neg.add(key); kept.append((q,code))
                            if len(kept)>=N: break
                        for ni,(q,code) in enumerate(kept): neg[ni]=q; nvalid[ni]=True; nreason[ni]=code
                        counts["hard_negative_examples"] = int(counts.get("hard_negative_examples",0)) + int(nvalid.sum())
                    features=build_rcrso_features_np(root=root,root_mass=float(nw[a,m]),root_source=int(ns[a,m]),blocker_state=state[j],current_ego_trajectory=ego_cur,shifted_ego_trajectory=ego_shift,environment=env,cfg=rcrso_cfg,verifier_cfg=cfg,blocker_object_type=int(obj[j]) if j<obj.size else 0)
                    E=args.environment_cap*2; envtok=np.zeros((E,rcrso_cfg.environment_feature_dim),np.float32); envvalid=np.zeros(E,bool); rawenv=features["environment_tokens"]; nenv=min(E,len(rawenv)); envtok[:nenv]=rawenv[:nenv]; envvalid[:nenv]=True
                    sh=int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8],"little",signed=False) & ((1<<63)-1)
                    hyp=(k*1000000)+(j*1000)+m; hyp_group=k
                    name=f"{sh:016x}_{k:03d}_{j:03d}_{m:02d}.npz"
                    # Keep the exact causal verifier context in the sidecar so Stage-0 can
                    # re-run hard admission for RCRSO proposals instead of using a distance/AUC proxy.
                    env_cur=np.zeros((args.environment_cap,len(root),7),np.float32); env_shift=np.zeros_like(env_cur); env_obj=np.zeros(args.environment_cap,np.int64); env_idx=np.full(args.environment_cap,-1,np.int64)
                    for ei,actor in enumerate(env[:args.environment_cap]):
                        env_cur[ei]=np.asarray(actor["trajectory"],np.float32); env_shift[ei]=np.asarray(actor["shifted_trajectory"],np.float32); env_obj[ei]=int(actor["object_type"]); env_idx[ei]=int(actor["agent_index"])
                    # Store only roadgraph points relevant to this root/ego context. The online
                    # roadgraph predicate itself is unchanged when Stage-0 reconstructs the dict.
                    rg_xy=np.asarray(road.get("xy",np.zeros((0,2),np.float32)),np.float32); rg_valid=np.asarray(road.get("valid",np.zeros(len(rg_xy),bool)),bool); rg_types=np.asarray(road.get("types",np.zeros(len(rg_xy),np.int32)),np.int32); rg_heading=np.asarray(road.get("heading",np.zeros(len(rg_xy),np.float32)),np.float32)
                    if len(rg_xy):
                        allxy=np.concatenate([root[:,:2],ego_cur[:,:2],ego_shift[:,:2]],axis=0); lo=np.nanmin(allxy,axis=0)-20.0; hi=np.nanmax(allxy,axis=0)+20.0; keep_rg=rg_valid & (rg_xy[:,0]>=lo[0]) & (rg_xy[:,0]<=hi[0]) & (rg_xy[:,1]>=lo[1]) & (rg_xy[:,1]<=hi[1]); inds=np.flatnonzero(keep_rg); inds=inds[:4096]
                    else: inds=np.zeros(0,np.int64)
                    R=4096; rxy=np.zeros((R,2),np.float32); rh=np.zeros(R,np.float32); rtpe=np.zeros(R,np.int32); rvld=np.zeros(R,bool); nr=min(R,len(inds));
                    if nr: rxy[:nr]=rg_xy[inds[:nr]]; rh[:nr]=rg_heading[inds[:nr]]; rtpe[:nr]=rg_types[inds[:nr]]; rvld[:nr]=True
                    np.savez_compressed(outdir/name,root_tokens=features["root_tokens"],ego_tokens=features["ego_tokens"],environment_tokens=envtok,environment_valid=envvalid,blocker_state=features["blocker_state"],conflict_features=features["conflict_features"],target_control_knots=targets,target_valid=tvalid,target_burden=tb,target_source=ts,negative_control_knots=neg,negative_valid=nvalid,negative_reason=nreason,root_mass=np.float32(nw[a,m]),root_source=np.int64(ns[a,m]),fixed_verified_nonempty=np.bool_(any(x==0 for x in pos_source)),analytic_verified_nonempty=np.bool_(bool(analytic)),scenario_hash=np.int64(sh),hypothesis_id=np.int64(hyp),hypothesis_group_id=np.int64(hyp_group),candidate_index=np.int64(k),agent_index=np.int64(j),root_index=np.int64(m),root_trajectory=np.asarray(root,np.float32),blocker_state_global=np.asarray(state[j],np.float32),blocker_object_type=np.int64(obj[j] if j<obj.size else 0),beta=np.float32(beta[a]),ego_current=np.asarray(ego_cur,np.float32),ego_shifted=np.asarray(ego_shift,np.float32),environment_current=env_cur,environment_shifted=env_shift,environment_object_type=env_obj,environment_agent_index=env_idx,roadgraph_xy=rxy,roadgraph_heading=rh,roadgraph_types=rtpe,roadgraph_valid=rvld)
                    manifest.append({"file":name,"scenario_id":sid,"candidate_index":k,"agent_index":j,"root_index":m,"verified_targets":int(tvalid.sum()),"fixed_nonempty":bool(any(x==0 for x in pos_source)),"analytic_nonempty":bool(analytic)})
                    counts["examples"]+=1; counts["positive_examples"]+=int(bool(tvalid.any())); made+=1
    suffix = "" if int(args.num_shards) == 1 else f"_s{int(args.shard_index)}of{int(args.num_shards)}"
    (Path(args.output_root)/f"manifest_{args.split}{suffix}.jsonl").write_text("\n".join(json.dumps(x,sort_keys=True) for x in manifest)+("\n" if manifest else ""),encoding="utf-8")
    summary={"version":"V16.8.45","split":args.split,"cache_dir":str(args.cache_dir),"forbidden_id_count":len(forbidden),"num_shards":int(args.num_shards),"shard_index":int(args.shard_index),"counts":counts,"rcrso_config":rcrso_cfg.to_dict(),"contract":{"base_compact5k_modified":False,"lost7_or_counterfactual48_allowed":False,"hard_verifier_semantics":"V42-V44 frozen predicates","rich_proposal_source":"deterministic Sobol knots; proposals admitted only after hard verifier","hard_negative_source":"verifier-rejected Sobol controls retained nearest to verified support"}}
    (Path(args.output_root)/f"summary_{args.split}{suffix}.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
