from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cowp.data.recourse_sidecar import RecourseSidecarDataset
from cowp.geometry.collision import unsafe_between_bool
from cowp.label.safe_responses import build_root_recovery_trajectory_bank, prepare_root_recovery_burden_bank
from cowp.models.recourse_set_operator import RCRSOConfig, RootConditionedRecourseSetTransformer
from cowp.core.constants import PriorityRelation
from cowp.waymax_eval.policy_wrapper import (
    _agent_state_after_future_sample_np,
    _roadgraph_drivable_mask,
    _root_conditioned_control_reachable_response_profiles_np,
    _shift_append_terminal_reference_np,
    _trajectory_waymax_kinematic_safe_np,
    _verified_root_conditioned_recourse_set_profiles_np,
)


def _environment(item: dict) -> list[dict]:
    cur=np.asarray(item["environment_current"],np.float32); sh=np.asarray(item["environment_shifted"],np.float32)
    typ=np.asarray(item["environment_object_type"],int); idx=np.asarray(item["environment_agent_index"],int)
    out=[]
    for i in range(min(len(cur),len(sh),len(typ),len(idx))):
        if int(idx[i])<0: continue
        out.append({"agent_index":int(idx[i]),"object_type":int(typ[i]),"trajectory":cur[i],"shifted_trajectory":sh[i]})
    return out


def _roadgraph(item: dict) -> dict[str,np.ndarray]:
    return {"xy":np.asarray(item["roadgraph_xy"],np.float32),"heading":np.asarray(item["roadgraph_heading"],np.float32),"types":np.asarray(item["roadgraph_types"],np.int32),"valid":np.asarray(item["roadgraph_valid"],bool)}


def _fixed_static_bank_profiles(item: dict, cfg: dict) -> list[dict]:
    """Frozen root-domain profiles before candidate-specific ego/environment checks.

    Online V42--V45 support preparation rejects an actor before any learned
    extension if a retained root has no low-burden, roadgraph-safe and
    current/shift Waymax-kinematic-safe fixed response at all.  Stage-0 must use
    the same callback domain; otherwise RCRSO can appear to improve support for a
    root that the online policy would reject before RCRSO is ever invoked.
    """
    root=np.asarray(item["root_trajectory"],np.float32)
    blocker=np.asarray(item["blocker_state_global"],np.float32)
    object_type=int(np.asarray(item["blocker_object_type"]).item())
    beta=float(np.asarray(item["beta"]).item())
    road=_roadgraph(item); dt=max(float(cfg.get("time",{}).get("dt",0.1)),1.0e-6)
    try:
        bank=build_root_recovery_trajectory_bank(root,cfg)
        burdens=prepare_root_recovery_burden_bank(root,bank,cfg,object_type=object_type,rho=PriorityRelation.AGENT_PRIORITY)
    except Exception:
        return []
    out=[]
    for pi,(tr,burden_entry) in enumerate(zip(bank,burdens)):
        tr=np.asarray(tr,np.float32); burden=float(burden_entry[0])
        if burden>beta+1e-9 or tr.ndim!=2 or tr.shape[0]<=0 or tr.shape[1]<7:
            continue
        if not bool(_roadgraph_drivable_mask(tr,road)):
            continue
        kin_ok,kin=_trajectory_waymax_kinematic_safe_np(blocker,tr,cfg)
        shifted=_shift_append_terminal_reference_np(tr,dt)
        successor=_agent_state_after_future_sample_np(blocker,tr[0])
        shift_ok,shift_kin=_trajectory_waymax_kinematic_safe_np(successor,shifted,cfg)
        if not (kin_ok and shift_ok and bool(_roadgraph_drivable_mask(shifted,road))):
            continue
        out.append({"profile_index":int(pi),"trajectory":tr,"shifted_trajectory":shifted,"burden":burden})
    return out


def _fixed_bank_profiles(item: dict, cfg: dict, *, static_profiles: list[dict] | None = None) -> list[dict]:
    """Frozen profiles surviving the candidate-specific ego/environment screen.

    ``static_profiles`` is an exact work-reuse hook only.  The historical R2
    implementation recomputed the low-burden/roadgraph/kinematic fixed bank
    immediately after computing the identical static bank for
    ``online_static_support_nonempty``.  Reusing that list changes no predicate.
    """
    object_type=int(np.asarray(item["blocker_object_type"]).item())
    ego_cur=np.asarray(item["ego_current"],np.float32); ego_shift=np.asarray(item["ego_shifted"],np.float32)
    env=_environment(item)
    out=[]
    source = _fixed_static_bank_profiles(item,cfg) if static_profiles is None else static_profiles
    for profile in source:
        tr=np.asarray(profile["trajectory"],np.float32); shifted=np.asarray(profile["shifted_trajectory"],np.float32)
        try:
            if unsafe_between_bool(ego_cur,tr,cfg,agent_type=object_type):
                continue
            if unsafe_between_bool(ego_shift,shifted,cfg,agent_type=object_type):
                continue
        except Exception:
            continue
        ok=True
        for actor in env:
            try:
                ac=np.asarray(actor["trajectory"],np.float32); ash=np.asarray(actor["shifted_trajectory"],np.float32); at=int(actor["object_type"])
                if (unsafe_between_bool(tr,ac,cfg,agent_type=at) or unsafe_between_bool(ac,tr,cfg,agent_type=object_type)
                    or unsafe_between_bool(shifted,ash,cfg,agent_type=at) or unsafe_between_bool(ash,shifted,cfg,agent_type=object_type)):
                    ok=False; break
            except Exception:
                ok=False; break
        if ok:
            out.append(profile)
    return out

def _analytic_profiles(item: dict, cfg: dict, *, compatibility_cache: dict | None = None) -> list[dict]:
    blocker=np.asarray(item["blocker_state_global"],np.float32)
    state=blocker[None,:]
    profiles,_=_root_conditioned_control_reachable_response_profiles_np(
        state,0,int(np.asarray(item["blocker_object_type"]).item()),
        np.asarray(item["root_trajectory"],np.float32),float(np.asarray(item["beta"]).item()),
        np.asarray(item["ego_current"],np.float32),np.asarray(item["ego_shifted"],np.float32),
        _environment(item),_roadgraph(item),cfg,profile_index_base=10000,
        root_ordinal=int(np.asarray(item["root_index"]).item()),compatibility_cache={} if compatibility_cache is None else compatibility_cache
    )
    return profiles


def _pair_ok(a:dict,b:dict,cfg:dict)->bool:
    try:
        return not (
            unsafe_between_bool(np.asarray(a["trajectory"],np.float32),np.asarray(b["trajectory"],np.float32),cfg,agent_type=int(b["object_type"]))
            or unsafe_between_bool(np.asarray(b["trajectory"],np.float32),np.asarray(a["trajectory"],np.float32),cfg,agent_type=int(a["object_type"]))
            or unsafe_between_bool(np.asarray(a["shifted_trajectory"],np.float32),np.asarray(b["shifted_trajectory"],np.float32),cfg,agent_type=int(b["object_type"]))
            or unsafe_between_bool(np.asarray(b["shifted_trajectory"],np.float32),np.asarray(a["shifted_trajectory"],np.float32),cfg,agent_type=int(a["object_type"]))
        )
    except Exception:
        return False


def _profile_pair_identity(profile: dict, object_type: int) -> tuple:
    cur=np.ascontiguousarray(np.asarray(profile["trajectory"],np.float32))
    sh=np.ascontiguousarray(np.asarray(profile["shifted_trajectory"],np.float32))
    return (int(object_type),cur.shape,cur.tobytes(),sh.shape,sh.tobytes())

def _csp(nodes:list[dict],cfg:dict, pair_cache:dict | None=None)->bool:
    """Exact historical CSP with optional memoization of exact trajectory pairs."""
    if any(not n["profiles"] for n in nodes): return False
    if pair_cache is None: pair_cache={}
    ordered=sorted(nodes,key=lambda n:(len(n["profiles"]),n["agent_index"],n["root_index"]))
    chosen=[]
    def compatible(node,p,on,op):
        if node["agent_index"]==on["agent_index"]:
            return True
        a=_profile_pair_identity(p,int(node["object_type"])); b=_profile_pair_identity(op,int(on["object_type"]))
        key=(a,b) if a<=b else (b,a)
        if key not in pair_cache:
            pair_cache[key]=bool(_pair_ok({**p,"object_type":node["object_type"]},{**op,"object_type":on["object_type"]},cfg))
        return bool(pair_cache[key])
    def dfs(pos:int)->bool:
        if pos>=len(ordered): return True
        node=ordered[pos]
        for p in node["profiles"]:
            ok=True
            for on,op in chosen:
                if not compatible(node,p,on,op): ok=False; break
            if ok:
                chosen.append((node,p))
                if dfs(pos+1): return True
                chosen.pop()
        return False
    return dfs(0)


def _load_cfg(label_config:str,data_config:str,eval_config:str)->dict:
    # Hard-verifier numerical contracts come from the exact frozen V16.8 config files,
    # never from the learned operator checkpoint.
    from cowp.core.config import load_config
    return load_config(label_config,data_config,eval_config)



_STAGE0_KEYS = (
    "root_tokens", "ego_tokens", "environment_tokens", "environment_valid",
    "blocker_state", "conflict_features", "target_valid", "scenario_hash",
    "hypothesis_group_id", "agent_index", "root_index", "root_trajectory",
    "blocker_state_global", "blocker_object_type", "beta", "ego_current",
    "ego_shifted", "environment_current", "environment_shifted",
    "environment_object_type", "environment_agent_index", "roadgraph_xy",
    "roadgraph_heading", "roadgraph_types", "roadgraph_valid",
)


def _sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _load_stage0_item(path: Path) -> dict:
    # NPZ is lazy per member.  Stage-0 does not need training-only negative/target
    # tensors other than target_valid, so avoid decompressing them all.
    with np.load(path,allow_pickle=False) as z:
        missing=[k for k in _STAGE0_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"Stage-0 sidecar item {path} missing keys: {missing}")
        return {k:z[k] for k in _STAGE0_KEYS}


def _path_key(path: Path) -> tuple[int,int]:
    parts=path.stem.split("_")
    if len(parts)>=2:
        try:
            return int(parts[0],16), int(parts[1])
        except Exception:
            pass
    with np.load(path,allow_pickle=False) as z:
        return int(np.asarray(z["scenario_hash"]).item()), int(np.asarray(z["hypothesis_group_id"]).item())


def _assigned_groups(paths: list[Path], num_shards:int, shard_index:int):
    groups=[]; cur_key=None; cur=[]
    for path in paths:
        sh,gid=_path_key(path)
        if sh % int(num_shards) != int(shard_index):
            continue
        key=(sh,gid)
        if cur_key is not None and key!=cur_key:
            groups.append((cur_key,cur)); cur=[]
        cur_key=key; cur.append(path)
    if cur_key is not None:
        groups.append((cur_key,cur))
    return groups


def _zero_stats(ks:list[int]) -> dict:
    return {
        "examples":0,
        "eligible_examples":0,
        "hypothesis_groups":0,
        "oracle_positive_roots":0,
        "verifier_calls":0,
        "baseline":{
            "fixed_bank":{"root_num":0,"full_num":0,"csp_num":0},
            "v44_analytic_extension":{"root_num":0,"full_num":0,"csp_num":0},
        },
        "rcrso":{
            str(k):{
                "root_num":0,"full_num":0,"csp_num":0,
                "burden_sum":0.0,"burden_count":0,
                "learned_verified_profile_count":0,
                "learned_root_nonempty_num":0,
            } for k in ks
        },
    }


def _profile_fn_analytic(rec:dict)->list[dict]:
    # Historical V44 callback semantics, exactly as V45R2.
    if not bool(rec.get("online_static_support_nonempty",False)):
        return []
    fixed=list(rec["fixed_profiles"])
    return fixed if fixed else list(rec["analytic_profiles"])




def _array_semantic_digest(*arrays: np.ndarray) -> str:
    """SHA256 namespace for exact immutable Stage-0 cache inputs.

    Cross-hypothesis work reuse is allowed only when every candidate-independent
    array that can affect the frozen hard predicates is byte-identical.
    """
    h=hashlib.sha256()
    for raw in arrays:
        a=np.ascontiguousarray(np.asarray(raw))
        h.update(str(a.dtype).encode('ascii')); h.update(repr(a.shape).encode('ascii')); h.update(a.tobytes())
    return h.hexdigest()

def _stage0_static_namespace(item: dict) -> tuple:
    road_sig=_array_semantic_digest(item["roadgraph_xy"],item["roadgraph_heading"],item["roadgraph_types"],item["roadgraph_valid"])
    env_sig=_array_semantic_digest(item["environment_current"],item["environment_shifted"],item["environment_object_type"],item["environment_agent_index"])
    root_sig=_array_semantic_digest(item["root_trajectory"],item["blocker_state_global"])
    return (
        int(np.asarray(item["agent_index"]).item()), int(np.asarray(item["root_index"]).item()),
        int(np.asarray(item["blocker_object_type"]).item()), float(np.asarray(item["beta"]).item()),
        root_sig, road_sig, env_sig,
    )

def _process_item(model, item:dict, cfg:dict, ks:list[int], device:torch.device, scene_cache:dict, timings:dict) -> dict:
    def tt(k): return torch.as_tensor(np.asarray(item[k]),device=device,dtype=torch.float32).unsqueeze(0)
    t=time.perf_counter()
    env_valid=torch.as_tensor(np.asarray(item["environment_valid"]),device=device,dtype=torch.bool).unsqueeze(0)
    with torch.no_grad():
        out=model(root_tokens=tt("root_tokens"),ego_tokens=tt("ego_tokens"),environment_tokens=tt("environment_tokens"),environment_valid=env_valid,blocker_state=tt("blocker_state"),conflict_features=tt("conflict_features"),query_count=max(ks))
    knots=out["control_knots"][0].detach().cpu().numpy().astype(np.float32)
    logits=out["feasible_logits"][0].detach().cpu().numpy().astype(np.float64)
    order=np.argsort(-logits,kind="stable"); knots=knots[order]
    timings["model_forward_s"] += time.perf_counter()-t

    semantic_ns=_stage0_static_namespace(item)
    compat=scene_cache.setdefault(("compatibility",semantic_ns),{})
    t=time.perf_counter()
    agent_state=np.asarray(item["blocker_state_global"],np.float32)[None,:]
    profiles,detail=_verified_root_conditioned_recourse_set_profiles_np(
        agent_state,0,int(np.asarray(item["blocker_object_type"]).item()),
        np.asarray(item["root_trajectory"],np.float32),float(np.asarray(item["beta"]).item()),
        np.asarray(item["ego_current"],np.float32),np.asarray(item["ego_shifted"],np.float32),
        _environment(item),_roadgraph(item),cfg,knots,profile_index_base=20000,
        root_ordinal=int(np.asarray(item["root_index"]).item()),compatibility_cache=compat)
    timings["learned_verify_s"] += time.perf_counter()-t
    by_q={int(p["profile_index"])-20000:p for p in profiles}

    static_key=semantic_ns
    static_cache=scene_cache.setdefault("fixed_static",{})
    t=time.perf_counter()
    fixed_static_profiles=static_cache.get(static_key)
    if fixed_static_profiles is None:
        fixed_static_profiles=_fixed_static_bank_profiles(item,cfg)
        static_cache[static_key]=fixed_static_profiles
    timings["fixed_static_s"] += time.perf_counter()-t

    t=time.perf_counter()
    fixed_profiles=_fixed_bank_profiles(item,cfg,static_profiles=fixed_static_profiles)
    timings["fixed_dynamic_s"] += time.perf_counter()-t

    t=time.perf_counter()
    analytic_profiles=_analytic_profiles(item,cfg,compatibility_cache=compat)
    timings["analytic_s"] += time.perf_counter()-t

    rec={
        "scenario_hash":int(np.asarray(item["scenario_hash"]).item()),
        "hypothesis_group_id":int(np.asarray(item["hypothesis_group_id"]).item()),
        "agent_index":int(np.asarray(item["agent_index"]).item()),
        "root_index":int(np.asarray(item["root_index"]).item()),
        "object_type":int(np.asarray(item["blocker_object_type"]).item()),
        "oracle_positive":bool(np.asarray(item["target_valid"]).any()),
        "online_static_support_nonempty":bool(fixed_static_profiles),
        "fixed_profiles":fixed_profiles,
        "analytic_profiles":analytic_profiles,
        "learned_profiles_by_k":{},
        "online_profiles_by_k":{},
    }
    for k in ks:
        learned=[p for q,p in by_q.items() if q<k]
        online=list(fixed_profiles) + list(learned)
        rec["learned_profiles_by_k"][k]=learned
        rec["online_profiles_by_k"][k]=online
    rec["verifier_calls"]=int(detail.get("profile_evaluations",0))
    return rec


def _accumulate_group(stats:dict, rr:list[dict], ks:list[int], cfg:dict, timings:dict) -> None:
    stats["hypothesis_groups"] += 1
    stats["examples"] += len(rr); stats["eligible_examples"] += len(rr)
    oracle=[r for r in rr if r["oracle_positive"]]
    stats["oracle_positive_roots"] += len(oracle)
    stats["verifier_calls"] += sum(int(r.get("verifier_calls",0)) for r in rr)
    pair_cache={}

    baseline_specs=(
        ("fixed_bank",lambda r:list(r["fixed_profiles"])),
        ("v44_analytic_extension",_profile_fn_analytic),
    )
    t=time.perf_counter()
    for name,fn in baseline_specs:
        profiles=[fn(r) for r in rr]
        stats["baseline"][name]["root_num"] += sum(bool(profiles[i]) for i,r in enumerate(rr) if r["oracle_positive"])
        stats["baseline"][name]["full_num"] += int(all(bool(x) for x in profiles))
        nodes=[{"agent_index":r["agent_index"],"root_index":r["root_index"],"object_type":r["object_type"],"profiles":profiles[i]} for i,r in enumerate(rr)]
        stats["baseline"][name]["csp_num"] += int(_csp(nodes,cfg,pair_cache=pair_cache))
    for k in ks:
        online=[r["online_profiles_by_k"][k] for r in rr]
        learned=[r["learned_profiles_by_k"][k] for r in rr]
        s=stats["rcrso"][str(k)]
        s["root_num"] += sum(bool(online[i]) for i,r in enumerate(rr) if r["oracle_positive"])
        s["full_num"] += int(all(bool(x) for x in online))
        nodes=[{"agent_index":r["agent_index"],"root_index":r["root_index"],"object_type":r["object_type"],"profiles":online[i]} for i,r in enumerate(rr)]
        s["csp_num"] += int(_csp(nodes,cfg,pair_cache=pair_cache))
        for lst in learned:
            s["learned_root_nonempty_num"] += int(bool(lst))
            s["learned_verified_profile_count"] += len(lst)
            for p in lst:
                s["burden_sum"] += float(p["burden"]); s["burden_count"] += 1
    timings["csp_metrics_s"] += time.perf_counter()-t


def _partial_to_summary(partials:list[dict], minimum_lift_pp:float) -> dict:
    if not partials: raise ValueError("no Stage-0 partials")
    ks=partials[0]["k_values"]
    for p in partials:
        if p["k_values"]!=ks: raise ValueError("K mismatch across Stage-0 shards")
    examples=sum(p["stats"]["examples"] for p in partials)
    eligible=sum(p["stats"]["eligible_examples"] for p in partials)
    groups=sum(p["stats"]["hypothesis_groups"] for p in partials)
    oracle=sum(p["stats"]["oracle_positive_roots"] for p in partials)
    verifier_calls=sum(p["stats"]["verifier_calls"] for p in partials)
    baseline={}
    for name in ("fixed_bank","v44_analytic_extension"):
        c={k:sum(p["stats"]["baseline"][name][k] for p in partials) for k in ("root_num","full_num","csp_num")}
        baseline[name]={
            "VerifiedRootRecall":c["root_num"]/max(oracle,1),
            "FullHypothesisRootCoverage":c["full_num"]/max(groups,1),
            "ExactCSPCompletionRate":c["csp_num"]/max(groups,1),
        }
    curves=[]
    for k in ks:
        key=str(k)
        c={name:sum(p["stats"]["rcrso"][key][name] for p in partials) for name in (
            "root_num","full_num","csp_num","burden_sum","burden_count","learned_verified_profile_count","learned_root_nonempty_num")}
        curves.append({
            "K":int(k),
            "VerifiedRootRecall":c["root_num"]/max(oracle,1),
            "FullHypothesisRootCoverage":c["full_num"]/max(groups,1),
            "ExactCSPCompletionRate":c["csp_num"]/max(groups,1),
            "learned_verified_response_burden_mean":c["burden_sum"]/c["burden_count"] if c["burden_count"] else None,
            "learned_verified_profile_count":int(c["learned_verified_profile_count"]),
            "learned_only_root_nonempty_rate":c["learned_root_nonempty_num"]/max(eligible,1),
            "online_extension_semantics":"fixed_plus_rcrso_verified_union_after_nested_v43_failure",
        })
    plateau=max(x["FullHypothesisRootCoverage"] for x in curves); target=0.95*plateau
    selected=min((x for x in curves if x["FullHypothesisRootCoverage"]+1e-12>=target),key=lambda x:x["K"])
    baseline_best=max(baseline["fixed_bank"]["FullHypothesisRootCoverage"],baseline["v44_analytic_extension"]["FullHypothesisRootCoverage"])
    lift=selected["FullHypothesisRootCoverage"]-baseline_best
    gate=lift+1e-12>=float(minimum_lift_pp)/100.0 and selected["VerifiedRootRecall"]>0.0
    runtime={
        "parallel_shards":len(partials),
        "sum_shard_wall_seconds":sum(float(p["timing_seconds"]["wall_seconds"]) for p in partials),
        "max_shard_wall_seconds":max(float(p["timing_seconds"]["wall_seconds"]) for p in partials),
        "phase_seconds_sum":{},
    }
    phase_keys=set().union(*(p["timing_seconds"].keys() for p in partials))
    for phase in sorted(phase_keys):
        if phase=="wall_seconds": continue
        runtime["phase_seconds_sum"][phase]=sum(float(p["timing_seconds"].get(phase,0.0)) for p in partials)
    return {
        "version":"V16.8.45R3-engineering",
        "scientific_method":"V16.8.45 RCRSO unchanged",
        "split":partials[0]["split"],
        "examples":examples,"eligible_examples":eligible,
        "hypothesis_groups_all":groups,"hypothesis_groups":groups,
        "structural_ineligible_hypothesis_groups":0,
        "oracle_positive_roots":oracle,"teacher_verified_positive_roots":oracle,
        "oracle_positive_definition":"known verifier-positive control in the frozen finite sidecar teacher pool; not proof of exhaustive recourse existence",
        "baseline":baseline,"rcrso_curve":curves,
        "plateau_full_hypothesis_root_coverage":plateau,"plateau_95_target":target,
        "selected_k":selected["K"],"selected_metrics":selected,
        "coverage_lift_over_best_frozen_baseline":lift,
        "minimum_required_lift_pp":float(minimum_lift_pp),
        "stage0_support_gate":{"pass":bool(gate),"reason":"full_hypothesis_root_coverage_lift" if gate else "insufficient_verified_support_lift"},
        "verifier_calls":verifier_calls,
        "wall_seconds":runtime["max_shard_wall_seconds"],
        "runtime_observability":runtime,
        "provenance":{
            "checkpoint_sha256":partials[0]["checkpoint_sha256"],
            "sidecar_summary_sha256":partials[0].get("sidecar_summary_sha256"),
            "shard_scenario_hashes":[p["scenario_hashes"] for p in partials],
        },
        "contract":{
            "K_selected_only_on_frozen_validation_sidecar":True,"lost7_used_for_K":False,
            "hard_verifier_replayed":True,"proposal_score_used_only_for_ordering":True,
            "stage0_matches_online_extension_semantics":True,
            "rcrso_verified_profiles_augment_frozen_domains_after_nested_v43_failure":True,
            "empty_frozen_static_root_domains_are_valid_proposal_completeness_targets":True,
            "stage0_panel_role":"outcome-blind label-root support proxy; closed-loop lost7 remains the causal policy gate",
            "parallelization_semantics":"scenario-disjoint exact partition; all roots of each hypothesis group stay on one shard",
            "performance_repairs_change_hard_boolean":False,
        },
    }


def _run_partial(args, model, rcfg, device, ks, cfg) -> dict:
    ds=RecourseSidecarDataset(args.sidecar_root,args.split)
    paths=list(ds.paths)
    groups=_assigned_groups(paths,args.num_shards,args.shard_index)
    total_examples=sum(len(x[1]) for x in groups)
    scenario_hashes=sorted(set(k[0] for k,_ in groups))
    stats=_zero_stats(ks)
    timings={"load_s":0.0,"model_forward_s":0.0,"learned_verify_s":0.0,"fixed_static_s":0.0,"fixed_dynamic_s":0.0,"analytic_s":0.0,"csp_metrics_s":0.0}
    t0=time.perf_counter(); last=t0; done=0; current_scenario=None; scene_cache={}
    max_examples=args.max_examples
    for (group_key,group_paths) in groups:
        if max_examples is not None and done>=max_examples: break
        if max_examples is not None and done>0 and done+len(group_paths)>max_examples: break
        if current_scenario!=group_key[0]:
            current_scenario=group_key[0]; scene_cache={}
        rr=[]
        for path in group_paths:
            t=time.perf_counter(); item=_load_stage0_item(path); timings["load_s"]+=time.perf_counter()-t
            rr.append(_process_item(model,item,cfg,ks,device,scene_cache,timings)); done+=1
        _accumulate_group(stats,rr,ks,cfg,timings)
        now=time.perf_counter()
        if args.progress_every_seconds>0 and now-last>=args.progress_every_seconds:
            elapsed=max(now-t0,1e-9); rate=done/elapsed; remain=max(total_examples-done,0); eta=remain/max(rate,1e-12)
            phase_total=sum(timings.values()); phase_txt=" ".join(f"{(k[:-2] if k.endswith('_s') else k)}={100*v/phase_total:.0f}%" for k,v in timings.items() if phase_total>0 and v>0)
            print(f"[RCRSO-Stage0 {args.split} s{args.shard_index}/{args.num_shards}] examples={done}/{total_examples} groups={stats['hypothesis_groups']} verifier={stats['verifier_calls']} elapsed={elapsed/60:.1f}m rate={rate:.3f}ex/s eta={eta/60:.1f}m timing[{phase_txt}]",file=sys.stderr,flush=True)
            last=now
    wall=time.perf_counter()-t0
    timing_seconds={**timings,"wall_seconds":wall}
    summary_path=Path(args.sidecar_root)/f"summary_{args.split}.json"
    partial={
        "version":"V16.8.45R3-stage0-partial","scientific_method":"V16.8.45 RCRSO unchanged",
        "split":args.split,"num_shards":int(args.num_shards),"shard_index":int(args.shard_index),"k_values":ks,
        "stats":stats,"timing_seconds":timing_seconds,"scenario_hashes":scenario_hashes,
        "checkpoint_sha256":_sha256_file(args.checkpoint),
        "sidecar_summary_sha256":_sha256_file(summary_path) if summary_path.is_file() else None,
        "assigned_examples":total_examples,"processed_examples":done,
        "contract":{"scenario_disjoint_partition":True,"complete_hypothesis_groups_only":True,"hard_verifier_semantics_unchanged":True,"shared_cache_semantics":"exact per-scene semantic memoization only"},
    }
    print(f"[RCRSO-Stage0 {args.split} s{args.shard_index}/{args.num_shards}] DONE examples={done} groups={stats['hypothesis_groups']} wall={wall/60:.1f}m",file=sys.stderr,flush=True)
    return partial


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sidecar-root",required=True); ap.add_argument("--split",default="val",choices=["val","heldout"])
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); ap.add_argument("--selected-checkpoint",default=None)
    ap.add_argument("--k-values",default="2,4,8,16"); ap.add_argument("--device",default="auto"); ap.add_argument("--max-examples",type=int,default=None)
    ap.add_argument("--minimum-full-hypothesis-coverage-lift-pp",type=float,default=3.0)
    ap.add_argument("--label-config",default="configs/label_cowp_v16_8.yaml"); ap.add_argument("--data-config",default="configs/data.yaml"); ap.add_argument("--eval-config",default="configs/eval_cowp_v16_8.yaml")
    ap.add_argument("--num-shards",type=int,default=1); ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--partial-only",action="store_true",help="Write exact raw-count shard summary; final K/gate selection is deferred to the merge step.")
    ap.add_argument("--progress-every-seconds",type=float,default=30.0)
    args=ap.parse_args()
    if args.num_shards<1 or not (0<=args.shard_index<args.num_shards): raise ValueError(f"invalid shard {args.shard_index}/{args.num_shards}")
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    rcfg=RCRSOConfig.from_dict(ckpt.get("rcrso_config",ckpt.get("cfg",{})))
    model=RootConditionedRecourseSetTransformer(rcfg); model.load_state_dict(ckpt["model"],strict=True); model.to(device).eval()
    ks=sorted(set(int(x) for x in args.k_values.split(",") if x.strip() and 1<=int(x)<=rcfg.max_queries))
    if not ks: raise ValueError("No valid K values")
    cfg=_load_cfg(args.label_config,args.data_config,args.eval_config)
    partial=_run_partial(args,model,rcfg,device,ks,cfg)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if args.partial_only or args.num_shards>1:
        Path(args.output).write_text(json.dumps(partial,indent=2,sort_keys=True),encoding="utf-8")
        return
    summary=_partial_to_summary([partial],args.minimum_full_hypothesis_coverage_lift_pp)
    Path(args.output).write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    if not args.selected_checkpoint:
        raise ValueError("--selected-checkpoint is required when finalizing Stage-0")
    selected_ckpt=dict(ckpt); selected_ckpt["selected_k"]=int(summary["selected_k"]); selected_ckpt["stage0_support_audit"]=summary
    selected_ckpt["contract"]={**dict(selected_ckpt.get("contract",{})),"selected_k_status":"frozen_validation_selected","stage0_support_gate_pass":bool(summary["stage0_support_gate"]["pass"])}
    torch.save(selected_ckpt,args.selected_checkpoint)
    print(json.dumps(summary,indent=2,sort_keys=True))
    if not summary["stage0_support_gate"]["pass"]: raise SystemExit(4)


if __name__=="__main__": main()
