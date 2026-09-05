from __future__ import annotations

import argparse
import json
import shutil
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


def _fixed_bank_profiles(item: dict, cfg: dict) -> list[dict]:
    """Frozen profiles surviving the candidate-specific ego/environment screen."""
    object_type=int(np.asarray(item["blocker_object_type"]).item())
    ego_cur=np.asarray(item["ego_current"],np.float32); ego_shift=np.asarray(item["ego_shifted"],np.float32)
    env=_environment(item)
    out=[]
    for profile in _fixed_static_bank_profiles(item,cfg):
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

def _analytic_profiles(item: dict, cfg: dict) -> list[dict]:
    blocker=np.asarray(item["blocker_state_global"],np.float32)
    state=blocker[None,:]
    profiles,_=_root_conditioned_control_reachable_response_profiles_np(
        state,0,int(np.asarray(item["blocker_object_type"]).item()),
        np.asarray(item["root_trajectory"],np.float32),float(np.asarray(item["beta"]).item()),
        np.asarray(item["ego_current"],np.float32),np.asarray(item["ego_shifted"],np.float32),
        _environment(item),_roadgraph(item),cfg,profile_index_base=10000,
        root_ordinal=int(np.asarray(item["root_index"]).item()),compatibility_cache={}
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


def _csp(nodes:list[dict],cfg:dict)->bool:
    if any(not n["profiles"] for n in nodes): return False
    ordered=sorted(nodes,key=lambda n:(len(n["profiles"]),n["agent_index"],n["root_index"]))
    chosen=[]
    def dfs(pos:int)->bool:
        if pos>=len(ordered): return True
        node=ordered[pos]
        for p in node["profiles"]:
            ok=True
            for on,op in chosen:
                if node["agent_index"]==on["agent_index"]: continue
                if not _pair_ok({**p,"object_type":node["object_type"]},{**op,"object_type":on["object_type"]},cfg): ok=False; break
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


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sidecar-root",required=True); ap.add_argument("--split",default="val",choices=["val","heldout"])
    ap.add_argument("--checkpoint",required=True); ap.add_argument("--output",required=True); ap.add_argument("--selected-checkpoint",required=True)
    ap.add_argument("--k-values",default="2,4,8,16"); ap.add_argument("--device",default="auto"); ap.add_argument("--max-examples",type=int,default=None)
    ap.add_argument("--minimum-full-hypothesis-coverage-lift-pp",type=float,default=3.0)
    ap.add_argument("--label-config",default="configs/label_cowp_v16_8.yaml")
    ap.add_argument("--data-config",default="configs/data.yaml")
    ap.add_argument("--eval-config",default="configs/eval_cowp_v16_8.yaml")
    args=ap.parse_args(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device))
    ckpt=torch.load(args.checkpoint,map_location="cpu"); rcfg=RCRSOConfig.from_dict(ckpt.get("rcrso_config",ckpt.get("cfg",{}))); model=RootConditionedRecourseSetTransformer(rcfg); model.load_state_dict(ckpt["model"],strict=True); model.to(device).eval()
    ks=sorted(set(int(x) for x in args.k_values.split(",") if x.strip() and 1<=int(x)<=rcfg.max_queries));
    if not ks: raise ValueError("No valid K values")
    ds=RecourseSidecarDataset(args.sidecar_root,args.split); cfg=_load_cfg(args.label_config,args.data_config,args.eval_config)
    records=[]; t0=time.time(); total_verifier_calls=0
    limit=len(ds) if args.max_examples is None else min(len(ds),args.max_examples)
    with torch.no_grad():
        for i in range(limit):
            item=ds[i]
            def tt(k): return torch.as_tensor(np.asarray(item[k]),device=device,dtype=torch.float32).unsqueeze(0)
            env_valid=torch.as_tensor(np.asarray(item["environment_valid"]),device=device,dtype=torch.bool).unsqueeze(0)
            out=model(root_tokens=tt("root_tokens"),ego_tokens=tt("ego_tokens"),environment_tokens=tt("environment_tokens"),environment_valid=env_valid,blocker_state=tt("blocker_state"),conflict_features=tt("conflict_features"),query_count=max(ks))
            knots=out["control_knots"][0].cpu().numpy().astype(np.float32); logits=out["feasible_logits"][0].cpu().numpy().astype(np.float64); order=np.argsort(-logits,kind="stable"); knots=knots[order]
            # Re-run the frozen hard verifier on every proposal once. Prefix-K metrics then only select subsets of the verified set.
            agent_state=np.asarray(item["blocker_state_global"],np.float32)[None,:]
            profiles,detail=_verified_root_conditioned_recourse_set_profiles_np(agent_state,0,int(np.asarray(item["blocker_object_type"]).item()),np.asarray(item["root_trajectory"],np.float32),float(np.asarray(item["beta"]).item()),np.asarray(item["ego_current"],np.float32),np.asarray(item["ego_shifted"],np.float32),_environment(item),_roadgraph(item),cfg,knots,profile_index_base=20000,root_ordinal=int(np.asarray(item["root_index"]).item()),compatibility_cache={})
            total_verifier_calls+=int(detail.get("profile_evaluations",0))
            by_q={int(p["profile_index"])-20000:p for p in profiles}
            fixed_static_profiles=_fixed_static_bank_profiles(item,cfg)
            fixed_profiles=_fixed_bank_profiles(item,cfg)
            analytic_profiles=_analytic_profiles(item,cfg)
            rec={"scenario_hash":int(np.asarray(item["scenario_hash"]).item()),"hypothesis_group_id":int(np.asarray(item["hypothesis_group_id"]).item()),"agent_index":int(np.asarray(item["agent_index"]).item()),"root_index":int(np.asarray(item["root_index"]).item()),"object_type":int(np.asarray(item["blocker_object_type"]).item()),"oracle_positive":bool(np.asarray(item["target_valid"]).any()),"online_static_support_nonempty":bool(fixed_static_profiles),"fixed_nonempty":bool(fixed_profiles),"analytic_nonempty":bool(analytic_profiles),"fixed_profiles":fixed_profiles,"analytic_profiles":analytic_profiles,"learned_verified_by_k":{},"learned_burden_by_k":{},"learned_profiles_by_k":{},"online_verified_by_k":{},"online_profiles_by_k":{}}
            for k in ks:
                learned=[p for q,p in by_q.items() if q<k]
                # V16.8.45R1 set-operator semantics: after the exact frozen V43
                # path has failed, verified learned proposals augment the frozen
                # candidate-specific response domain.  This can fill an empty root
                # domain *or* add diversity needed by the exact joint CSP.  Every
                # learned profile was independently replayed through the unchanged
                # hard verifier above, so the union changes completeness only.
                online=list(fixed_profiles) + list(learned)
                rec["learned_verified_by_k"][k]=bool(learned)
                rec["learned_burden_by_k"][k]=[float(p["burden"]) for p in learned]
                rec["learned_profiles_by_k"][k]=learned
                rec["online_verified_by_k"][k]=bool(online)
                rec["online_profiles_by_k"][k]=online
            records.append(rec)
    groups_all=defaultdict(list)
    for r in records: groups_all[(r["scenario_hash"],r["hypothesis_group_id"])].append(r)
    # V16.8.45R1 deliberately keeps roots whose frozen static proposal domain is
    # empty: those are proposal-completeness holes that RCRSO is meant to fill.
    # Natural-root count/mass validity is already frozen by sidecar construction;
    # no root/beta/verifier predicate is relaxed here.
    groups={key:rr for key,rr in groups_all.items() if rr}
    eligible_records=list(records)
    oracle=[r for r in eligible_records if r["oracle_positive"]]

    def extension_profiles(r:dict, extension_key:str)->list[dict]:
        # Historical V44 completion lived downstream of frozen support
        # preparation.  A root with no statically feasible frozen profile never
        # reached that callback, so analytic completion must not be credited with
        # repairing such a root in the Stage-0 baseline.
        if not bool(r.get("online_static_support_nonempty", False)):
            return []
        fixed=list(r["fixed_profiles"])
        return fixed if fixed else list(r[extension_key])

    # Baselines are evaluated with their historical online semantics.  V45R1 is
    # intentionally broader only in proposal completeness: its verified learned
    # set may populate a frozen-static proposal hole after frozen V43 has failed.
    baseline={}
    baseline_specs=(
        ("fixed_bank", lambda r:list(r["fixed_profiles"])),
        ("v44_analytic_extension", lambda r:extension_profiles(r,"analytic_profiles")),
    )
    for key,profile_fn in baseline_specs:
        root_recall=sum(bool(profile_fn(r)) for r in oracle)/max(len(oracle),1)
        full=sum(all(bool(profile_fn(x)) for x in rr) for rr in groups.values())/max(len(groups),1)
        csp_ok=0
        for rr in groups.values():
            nodes=[{"agent_index":r["agent_index"],"root_index":r["root_index"],"object_type":r["object_type"],"profiles":profile_fn(r)} for r in rr]
            csp_ok+=int(_csp(nodes,cfg))
        baseline[key]={"VerifiedRootRecall":root_recall,"FullHypothesisRootCoverage":full,"ExactCSPCompletionRate":csp_ok/max(len(groups),1)}
    curves=[]
    for k in ks:
        root_recall=sum(bool(r["online_verified_by_k"][k]) for r in oracle)/max(len(oracle),1)
        full=sum(all(bool(x["online_verified_by_k"][k]) for x in rr) for rr in groups.values())/max(len(groups),1)
        csp_ok=0
        for rr in groups.values():
            nodes=[]
            for r in rr:
                nodes.append({"agent_index":r["agent_index"],"root_index":r["root_index"],"object_type":r["object_type"],"profiles":r["online_profiles_by_k"][k]})
            csp_ok+=int(_csp(nodes,cfg))
        learned_burdens=[b for r in eligible_records for b in r["learned_burden_by_k"][k]]
        curves.append({
            "K":k,
            "VerifiedRootRecall":root_recall,
            "FullHypothesisRootCoverage":full,
            "ExactCSPCompletionRate":csp_ok/max(len(groups),1),
            "learned_verified_response_burden_mean":float(np.mean(learned_burdens)) if learned_burdens else None,
            "learned_verified_profile_count":sum(len(r["learned_profiles_by_k"][k]) for r in eligible_records),
            "learned_only_root_nonempty_rate":sum(bool(r["learned_verified_by_k"][k]) for r in eligible_records)/max(len(eligible_records),1),
            "online_extension_semantics":"fixed_plus_rcrso_verified_union_after_nested_v43_failure",
        })
    plateau=max(x["FullHypothesisRootCoverage"] for x in curves); target=0.95*plateau
    selected=min((x for x in curves if x["FullHypothesisRootCoverage"]+1e-12>=target),key=lambda x:x["K"])
    baseline_best=max(baseline["fixed_bank"]["FullHypothesisRootCoverage"],baseline["v44_analytic_extension"]["FullHypothesisRootCoverage"])
    lift=selected["FullHypothesisRootCoverage"]-baseline_best; gate=lift+1e-12>=args.minimum_full_hypothesis_coverage_lift_pp/100.0 and selected["VerifiedRootRecall"]>0.0
    summary={"version":"V16.8.45R2-engineering","scientific_method":"V16.8.45 RCRSO unchanged","split":args.split,"examples":len(records),"eligible_examples":len(eligible_records),"hypothesis_groups_all":len(groups_all),"hypothesis_groups":len(groups),"structural_ineligible_hypothesis_groups":0,"oracle_positive_roots":len(oracle),"teacher_verified_positive_roots":len(oracle),"oracle_positive_definition":"known verifier-positive control in the frozen finite sidecar teacher pool; not proof of exhaustive recourse existence","baseline":baseline,"rcrso_curve":curves,"plateau_full_hypothesis_root_coverage":plateau,"plateau_95_target":target,"selected_k":selected["K"],"selected_metrics":selected,"coverage_lift_over_best_frozen_baseline":lift,"minimum_required_lift_pp":args.minimum_full_hypothesis_coverage_lift_pp,"stage0_support_gate":{"pass":bool(gate),"reason":"full_hypothesis_root_coverage_lift" if gate else "insufficient_verified_support_lift"},"verifier_calls":total_verifier_calls,"wall_seconds":time.time()-t0,"contract":{"K_selected_only_on_frozen_validation_sidecar":True,"lost7_used_for_K":False,"hard_verifier_replayed":True,"proposal_score_used_only_for_ordering":True,"stage0_matches_online_extension_semantics":True,"rcrso_verified_profiles_augment_frozen_domains_after_nested_v43_failure":True,"empty_frozen_static_root_domains_are_valid_proposal_completeness_targets":True,"stage0_panel_role":"outcome-blind label-root support proxy; closed-loop lost7 remains the causal policy gate"}}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    selected_ckpt=dict(ckpt); selected_ckpt["selected_k"]=int(selected["K"]); selected_ckpt["stage0_support_audit"]=summary; selected_ckpt["contract"]={**dict(selected_ckpt.get("contract",{})),"selected_k_status":"frozen_validation_selected","stage0_support_gate_pass":bool(gate)}; torch.save(selected_ckpt,args.selected_checkpoint)
    print(json.dumps(summary,indent=2,sort_keys=True))
    if not gate: raise SystemExit(4)

if __name__=="__main__": main()
