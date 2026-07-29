#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, post_contact_deployability_score
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _teacher_pcd(d: dict[str, Any], alpha: float, beta: float, top_m: int) -> float:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    res = oc_mero(m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv, use_lcvar=True, use_obs_kernel=True, top_m=top_m)
    opt = best_shared_option_index(res.q, p, gamma=0.0, root_valid=rv, option_valid=ov)
    drs = deployable_recovery_success(m, p, opt, root_valid=rv)
    rd = float(_scalar(d, "r_dep_star", res.r_dep)); ro = float(_scalar(d, "r_orc_star", res.r_orc))
    return float(post_contact_deployability_score(drs, rd, max(0.0, ro-rd)))


def _group_deviation(items: list[dict[str, Any]]) -> list[float]:
    try:
        ref = np.asarray(items[0]["data"]["prefix_states"], dtype=float)[:, :2]
    except Exception:
        return [0.0] * len(items)
    out=[]
    for x in items:
        try:
            xy=np.asarray(x["data"]["prefix_states"], dtype=float)[:, :2]; t=min(len(ref),len(xy))
            out.append(0.0 if t<=0 else float(np.sqrt(np.mean(np.sum((xy[:t]-ref[:t])**2,axis=-1)))/5.0))
        except Exception:
            out.append(0.0)
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description="Calibrate a selection-valid additive lower bound for recovery advantage on the exact actionable candidate set.")
    ap.add_argument("--dataset", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=0.10); ap.add_argument("--required-min-groups", type=int, default=30)
    ap.add_argument("--macro-ids", default="2,3,5,7"); ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0); ap.add_argument("--max-harm", type=float, default=0.70)
    ap.add_argument("--numerical-margin", type=float, default=0.005)
    args=ap.parse_args(); macro_ids={int(x) for x in args.macro_ids.split(',') if x.strip()}
    cfg: dict[str,Any]={}; bundle=load_model_bundle(args.checkpoint,cfg)
    if bundle is None: raise FileNotFoundError(args.checkpoint)
    alpha=float((bundle.cfg.get("ocmero",{}) or {}).get("alpha",0.2)); beta=float((bundle.cfg.get("ocmero",{}) or {}).get("beta",0.2)); top_m=int((bundle.cfg.get("ocmero",{}) or {}).get("top_m",8))
    groups: dict[tuple[str,int],list[dict[str,Any]]]=defaultdict(list)
    paths=iter_sample_paths_many(args.dataset)
    for i,path in enumerate(paths,1):
        split=str(scalar_metadata_for_path(path,"split_id",""))
        if split not in {"calibration","val"}: continue
        d=load_npz(path); pred=predict_sample(d,bundle,cfg)
        if pred.direct_recovery_value is None: raise ValueError("checkpoint has no direct recovery value head")
        row={"data":d,"scene":str(_scalar(d,"scene_id",path.stem)),"time":int(_scalar(d,"time_index",0)),"candidate":int(_scalar(d,"candidate_index",0)),"macro":int(_scalar(d,"prefix_macro_type_id",_scalar(d,"prefix_macro_id",-1))),"nominal":bool(float(_scalar(d,"is_nominal",0))>0.5),"pred":float(pred.direct_recovery_value),"teacher":_teacher_pcd(d,alpha,beta,top_m),"hard":float(_scalar(d,"hard_violation",0.0)),"harm":float(_scalar(d,"harm_proxy",0.0)),"feasible":bool(int(_scalar(d,"feasible",1)))}
        groups[(row["scene"],row["time"])].append(row)
        if i==1 or i%1000==0: print({"event":"v41_calibration_progress","seen":i,"total":len(paths)},flush=True)
    scores=[]; pairs=[]; positive=[]; eligible_counts=[]; skipped={"no_nominal":0,"no_eligible":0}
    for items in groups.values():
        items.sort(key=lambda x:x["candidate"])
        for x,dv in zip(items,_group_deviation(items)): x["deviation"]=dv
        noms=[x for x in items if x["nominal"]]
        if not noms: skipped["no_nominal"]+=1; continue
        nom=noms[0]
        recs=[x for x in items if (not x["nominal"]) and x["macro"] in macro_ids and x["feasible"] and x["hard"]<=args.max_hard and x["harm"]<=args.max_harm and x["deviation"]>=args.min_nominal_deviation]
        if not recs: skipped["no_eligible"]+=1; continue
        gs=[]
        for r in recs:
            ta=r["teacher"]-nom["teacher"]; pa=r["pred"]-nom["pred"]
            gs.append(pa-ta); pairs.append((pa,ta)); positive.append(ta>=0.03)
        scores.append(max(gs)); eligible_counts.append(len(recs))
    warnings=[]; q=float("inf")
    if scores:
        q=max(0.0,float(finite_sample_upper_quantile(np.asarray(scores),args.delta,numerical_margin=args.numerical_margin,strict=True)))
    else: warnings.append("no complete actionable calibration groups")
    if len(scores)<args.required_min_groups: warnings.append(f"num_groups < required_min_groups ({len(scores)} < {args.required_min_groups})")
    pa=np.asarray([x[0] for x in pairs],dtype=float); ta=np.asarray([x[1] for x in pairs],dtype=float)
    lcb=pa-q if np.isfinite(q) else np.full_like(pa,-np.inf)
    pos=ta>=0.03; neg=ta<=0.0; challenge=lcb>=0.02
    result={"method":"actionable_scene_time_simultaneous_additive_conformal_advantage_lcb","dataset":args.dataset,"checkpoint":args.checkpoint,"delta":args.delta,"direct_value_uncertainty_mode":"additive","direct_value_additive_q":q,"num_scene_time_groups":len(groups),"num_calibration_groups":len(scores),"num_candidate_pairs":len(pairs),"eligible_candidates_mean":float(np.mean(eligible_counts)) if eligible_counts else None,"min_nominal_deviation":args.min_nominal_deviation,"max_hard":args.max_hard,"max_harm":args.max_harm,"macro_ids":sorted(macro_ids),"empirical_group_coverage":float(np.mean(np.asarray(scores)<=q)) if scores and np.isfinite(q) else None,"positive_teacher_group_fraction":float(np.mean(positive)) if positive else None,"pair_advantage_mae":float(np.mean(np.abs(pa-ta))) if len(pa) else None,"positive_sign_recall":float(np.mean(challenge[pos])) if np.any(pos) else None,"challenge_precision":float(np.mean(pos[challenge])) if np.any(challenge) else None,"negative_challenge_rate":float(np.mean(challenge[neg])) if np.any(neg) else None,"challenge_rate":float(np.mean(challenge)) if len(challenge) else None,"skipped":skipped,"warnings":warnings}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print(json.dumps(result,ensure_ascii=False))
    return 0 if np.isfinite(q) else 2

if __name__=="__main__": raise SystemExit(main())
