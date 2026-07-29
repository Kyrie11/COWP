from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.data.serialization import load_npz, write_json
from ocrap.evaluation.metrics import (
    best_shared_option_index,
    deployable_recovery_success,
    false_recoverability_admission,
    nominal_utility_preservation,
    post_contact_deployability_score,
    summarize_selection_metrics,
)
from ocrap.external_baselines.data import group_sample_paths, _branch_arrays, _topology_arrays, _history_arrays, _actor_topology_arrays, _map_topology_arrays, use_teacher_branch_context
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.external_baselines.observed_risk import observed_risk_profile
from ocrap.external_baselines.policies import ExternalSelection, select_external_policy
from ocrap.models.data import sample_to_feature


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return float(default)


def _load_checkpoint(checkpoint: str | Path | None, cfg: dict[str, Any]) -> tuple[torch.nn.Module | None, dict[str, Any], torch.device]:
    if not checkpoint:
        return None, cfg, torch.device("cpu")
    path = Path(checkpoint)
    if not path.exists():
        return None, cfg, torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", cfg) or cfg
    merged = dict(ckpt_cfg)
    # Runtime evaluation knobs may override policy thresholds and method lists,
    # but the checkpoint owns model geometry (d_model/layers/heads/max tokens).
    if isinstance(cfg.get("external_baselines", {}), dict):
        eb = dict(merged.get("external_baselines", {}) or {})
        rt = cfg.get("external_baselines", {}) or {}
        for key in ("methods", "policy", "baseline"):
            if key in rt:
                if key == "policy" and isinstance(rt.get(key), dict) and isinstance(eb.get(key), dict):
                    tmp = dict(eb.get(key) or {})
                    tmp.update(rt.get(key) or {})
                    eb[key] = tmp
                else:
                    eb[key] = rt[key]
        # max_candidates is safe to increase/decrease for padding at inference,
        # but not for reconstructing the learned positional parameter.  Keep the
        # checkpoint value when model_state is loaded.
        merged["external_baselines"] = eb
    eb = merged.setdefault("external_baselines", {})
    mcfg = eb.setdefault("model", {})
    if "max_candidates" in ckpt:
        eb["max_candidates"] = int(ckpt.get("max_candidates"))
        mcfg["max_candidates"] = int(ckpt.get("max_candidates"))
    for ck, mk in [
        ("num_roots", "num_roots"), ("num_options", "num_options"), ("root_feature_dim", "root_feature_dim"),
        ("num_topology_agents", "num_topology_agents"), ("topology_feature_dim", "topology_feature_dim"),
        ("actor_topology_feature_dim", "actor_topology_feature_dim"), ("num_topology_map", "num_topology_map"),
        ("map_topology_feature_dim", "map_topology_feature_dim"), ("history_len", "history_len"),
        ("neighbors_to_predict", "neighbors_to_predict"), ("future_len", "future_len"),
    ]:
        if ck in ckpt:
            mcfg[mk] = int(ckpt[ck])
    device_req = str(((merged.get("external_baselines", {}) or {}).get("training", {}) or {}).get("device", (merged.get("training", {}) or {}).get("device", "auto")))
    device = torch.device("cuda" if device_req == "auto" and torch.cuda.is_available() else ("cpu" if device_req == "auto" else device_req))
    model = build_model_from_cfg(int(ckpt["input_dim"]), merged).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, merged, device


def _predict_group(model: torch.nn.Module | None, samples: list[dict[str, Any]], cfg: dict[str, Any], device: torch.device) -> dict[str, np.ndarray] | None:
    if model is None or not samples:
        return None
    max_candidates = int(((cfg.get("external_baselines", {}) or {}).get("max_candidates", len(samples))))
    n = min(len(samples), max_candidates)
    feats = [sample_to_feature(d, cfg) for d in samples[:n]]
    if not feats:
        return None
    D = int(feats[0].shape[0])
    use_branch_context = use_teacher_branch_context(cfg)
    bm0, rf0, _, _, _ = _branch_arrays(samples[0], cfg)
    ego0, neigh0, _, pref0, _ = _history_arrays(samples[0], cfg)
    actor0, _, _ = _actor_topology_arrays(samples[0], cfg)
    map0, _, _ = _map_topology_arrays(samples[0], cfg)
    K, L, Fdim = int(bm0.shape[0]), int(bm0.shape[1]), int(rf0.shape[-1])
    H, A_hist, T = int(ego0.shape[0]), int(neigh0.shape[0]), int(pref0.shape[0])
    A_top, AF = int(actor0.shape[0]), int(actor0.shape[-1])
    M_top, MF = int(map0.shape[0]), int(map0.shape[-1])

    x = np.zeros((1, max_candidates, D), dtype=np.float32)
    mask = np.zeros((1, max_candidates), dtype=bool)
    branch_margins = np.zeros((1, max_candidates, K, L), dtype=np.float32)
    root_features = np.zeros((1, max_candidates, K, Fdim), dtype=np.float32)
    root_probs = np.zeros((1, max_candidates, K), dtype=np.float32)
    root_valid = np.zeros((1, max_candidates, K), dtype=bool)
    option_valid = np.zeros((1, max_candidates, L), dtype=bool)
    ego_history = np.zeros((1, max_candidates, H, 9), dtype=np.float32)
    neighbor_history = np.zeros((1, max_candidates, A_hist, H, 9), dtype=np.float32)
    neighbor_valid = np.zeros((1, max_candidates, A_hist, H), dtype=bool)
    prefix_traj = np.zeros((1, max_candidates, T, 2), dtype=np.float32)
    prefix_valid = np.zeros((1, max_candidates, T), dtype=bool)
    actor_topology_features = np.zeros((1, max_candidates, A_top, AF), dtype=np.float32)
    actor_topology_mask = np.zeros((1, max_candidates, A_top), dtype=bool)
    map_topology_features = np.zeros((1, max_candidates, M_top, MF), dtype=np.float32)
    map_topology_mask = np.zeros((1, max_candidates, M_top), dtype=bool)

    for i, f in enumerate(feats):
        x[0, i] = f
        mask[0, i] = True
        if use_branch_context:
            bm, rf, rp, rv, ov = _branch_arrays(samples[i], cfg)
            branch_margins[0, i] = bm
            root_features[0, i] = rf
            root_probs[0, i] = rp
            root_valid[0, i] = rv
            option_valid[0, i] = ov
        eh, nh, nv, pt, pv = _history_arrays(samples[i], cfg)
        ego_history[0, i] = eh
        neighbor_history[0, i] = nh
        neighbor_valid[0, i] = nv
        prefix_traj[0, i] = pt
        prefix_valid[0, i] = pv
        af, _, am = _actor_topology_arrays(samples[i], cfg)
        mf, _, mm = _map_topology_arrays(samples[i], cfg)
        actor_topology_features[0, i] = af
        actor_topology_mask[0, i] = am
        map_topology_features[0, i] = mf
        map_topology_mask[0, i] = mm
    with torch.no_grad():
        out = model(
            torch.from_numpy(x).to(device),
            torch.from_numpy(mask).to(device),
            branch_margins=(torch.from_numpy(branch_margins).to(device) if use_branch_context else None),
            root_features=(torch.from_numpy(root_features).to(device) if use_branch_context else None),
            root_probs=(torch.from_numpy(root_probs).to(device) if use_branch_context else None),
            root_valid=(torch.from_numpy(root_valid).to(device) if use_branch_context else None),
            option_valid=(torch.from_numpy(option_valid).to(device) if use_branch_context else None),
            ego_history=torch.from_numpy(ego_history).to(device),
            neighbor_history=torch.from_numpy(neighbor_history).to(device),
            neighbor_valid=torch.from_numpy(neighbor_valid).to(device),
            prefix_traj=torch.from_numpy(prefix_traj).to(device),
            prefix_valid=torch.from_numpy(prefix_valid).to(device),
            actor_topology_features=torch.from_numpy(actor_topology_features).to(device),
            actor_topology_mask=torch.from_numpy(actor_topology_mask).to(device),
            map_topology_features=torch.from_numpy(map_topology_features).to(device),
            map_topology_mask=torch.from_numpy(map_topology_mask).to(device),
        )
    result: dict[str, np.ndarray] = {}
    for k, v in out.items():
        if isinstance(v, list):
            continue
        result[k] = v.squeeze(0).detach().cpu().numpy()[:n]
    return result


def _yaw_rate_violation_proxy(d: dict[str, Any], yaw_rate_max: float = 0.6) -> float:
    """Return whether the candidate exceeds the configured yaw-rate limit.

    OC-RAP's ego prefix schema is [x,y,vx,vy,heading,yaw_rate,speed,length,width].
    Prefer the stored yaw-rate channel; fall back to a heading derivative only
    for legacy samples without that channel.
    """
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return 0.0
    if states.shape[1] >= 6:
        rate = np.abs(np.nan_to_num(states[:, 5], nan=0.0))
    elif states.shape[1] >= 5 and states.shape[0] >= 2:
        heading = np.unwrap(states[:, 4])
        dt = float(((d.get("dt", 0.1) if not isinstance(d.get("dt", 0.1), np.ndarray) else np.asarray(d.get("dt", 0.1)).item())))
        rate = np.abs(np.diff(heading)) / max(dt, 1e-3)
    else:
        return 0.0
    return float(np.max(rate) > float(yaw_rate_max)) if rate.size else 0.0


def _contact_extra_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {}
    out: dict[str, float | None] = {}
    out["secondary_collision_rate"] = float(np.mean([r.get("selected_hard_violation", 0.0) > 0.0 or r.get("selected_harm_proxy", 0.0) > 0.05 for r in records]))
    out["stable_stop_success"] = float(np.mean([r.get("drs", 0.0) >= 0.5 and r.get("selected_hard_violation", 0.0) <= 0.0 for r in records]))
    out["max_yaw_rate_violation"] = float(np.mean([r.get("yaw_rate_violation_proxy", 0.0) > 0.0 for r in records]))
    out["route_rejoin_success"] = float(np.mean([r.get("nup", 0.0) >= 0.5 and r.get("selected_harm_proxy", 0.0) <= 0.5 for r in records]))
    out["mean_harm_proxy"] = float(np.mean([r.get("selected_harm_proxy", 0.0) for r in records]))
    out["post_contact_deployability_score"] = float(np.mean([r.get("post_contact_deployability", 0.0) for r in records]))
    return out


def _record_for_selection(method: str, samples: list[dict[str, Any]], sel: ExternalSelection, cfg: dict[str, Any]) -> dict[str, Any]:
    idx = int(np.clip(sel.selected_index, 0, max(len(samples) - 1, 0)))
    chosen = samples[idx]
    utility = np.asarray([_scalar(d, "utility", 0.0) for d in samples], dtype=float)
    r_dep = np.asarray([_scalar(d, "r_dep_star", 0.0) for d in samples], dtype=float)
    r_orc = np.asarray([_scalar(d, "r_orc_star", 0.0) for d in samples], dtype=float)
    odg = float(np.asarray(chosen.get("oracle_gap_star", r_orc[idx] - r_dep[idx])).item()) if samples else 0.0
    selected_option = sel.selected_option
    if selected_option is None:
        selected_option = best_shared_option_index(chosen.get("m_star", np.zeros((0, 0))), chosen.get("root_probs", np.zeros((0,))), gamma=0.0, root_valid=chosen.get("root_valid", None), option_valid=chosen.get("option_valid", None))
    drs = deployable_recovery_success(chosen.get("m_star", np.zeros((0, 0))), chosen.get("root_probs", np.zeros((0,))), int(selected_option), chosen.get("root_valid", None))
    nup = nominal_utility_preservation(utility[0] if utility.size else 0.0, utility[idx] if utility.size else 0.0, sigma_u=float((cfg.get("metrics", {}) or {}).get("sigma_u", 1.0)))
    observed = observed_risk_profile(chosen, cfg)
    return {
        "method": method,
        "fra_cand": false_recoverability_admission(sel.admitted, r_dep),
        "fra_exec": float(r_dep[idx] < 0.0) if r_dep.size else 0.0,
        "drs": float(drs),
        "odg": float(odg),
        "post_contact_deployability": float(post_contact_deployability_score(float(drs), float(r_dep[idx]) if r_dep.size else 0.0, float(odg))),
        "nup": float(nup["bounded_NUP"]),
        "artifact": bool(_scalar(chosen, "i_art_star", 0.0) > 0.5),
        "selected_artifact": bool(_scalar(chosen, "i_art_star", 0.0) > 0.5),
        "selection_reason": sel.reason,
        "selected_index": idx,
        "selected_option": int(selected_option),
        "selected_utility": float(utility[idx]) if utility.size else 0.0,
        "selected_teacher_r_dep": float(r_dep[idx]) if r_dep.size else 0.0,
        "selected_teacher_r_orc": float(r_orc[idx]) if r_orc.size else 0.0,
        "selected_admitted": bool(sel.admitted[idx]) if len(sel.admitted) > idx else False,
        "num_admitted": int(np.asarray(sel.admitted, dtype=bool).sum()),
        "num_admitted_interventions": int(np.asarray(sel.admitted, dtype=bool)[1:].sum()) if len(sel.admitted) > 1 else 0,
        "selected_hard_violation": _scalar(chosen, "hard_violation", 0.0),
        "selected_harm_proxy": _scalar(chosen, "harm_proxy", 0.0),
        "selected_observed_expected_risk": float(observed.expected_loss),
        "selected_observed_cvar_risk": float(observed.cvar_loss),
        "selected_observed_collision_probability": float(observed.collision_probability),
        "selected_observed_min_clearance_m": float(observed.min_clearance),
        "selected_observed_backup_margin_m": float(observed.backup_margin),
        "yaw_rate_violation_proxy": _yaw_rate_violation_proxy(chosen, yaw_rate_max=float(cfg.get("yaw_rate_max_rps", 0.6))),
    }


def _summarize(records: list[dict[str, Any]], method: str, num_groups: int, source: str) -> dict[str, Any]:
    result = summarize_selection_metrics(records)
    if records:
        result.update({
            "intervention_rate": float(np.mean([int(r.get("selected_index", 0)) != 0 for r in records])),
            "selected_admitted_rate": float(np.mean([bool(r.get("selected_admitted", False)) for r in records])),
            "mean_num_admitted": float(np.mean([float(r.get("num_admitted", 0.0)) for r in records])),
            "mean_num_admitted_interventions": float(np.mean([float(r.get("num_admitted_interventions", 0.0)) for r in records])),
            "mean_selected_teacher_R_dep": float(np.mean([r.get("selected_teacher_r_dep", 0.0) for r in records])),
            "mean_selected_teacher_R_orc": float(np.mean([r.get("selected_teacher_r_orc", 0.0) for r in records])),
            "mean_selected_utility": float(np.mean([r.get("selected_utility", 0.0) for r in records])),
            "mean_selected_observed_expected_risk": float(np.mean([r.get("selected_observed_expected_risk", 0.0) for r in records])),
            "mean_selected_observed_cvar_risk": float(np.mean([r.get("selected_observed_cvar_risk", 0.0) for r in records])),
            "mean_selected_observed_collision_probability": float(np.mean([r.get("selected_observed_collision_probability", 0.0) for r in records])),
            "mean_selected_observed_min_clearance_m": float(np.mean([r.get("selected_observed_min_clearance_m", 0.0) for r in records])),
            "mean_selected_observed_backup_margin_m": float(np.mean([r.get("selected_observed_backup_margin_m", 0.0) for r in records])),
            "selection_reason_counts": dict(Counter(str(r.get("selection_reason", "")) for r in records)),
        })
        result.update(_contact_extra_metrics(records))
    result.update({"method": method, "num_scene_time_groups": int(num_groups), "num_records": int(len(records)), "source": source})
    return result


def evaluate_external_baselines(
    dataset: str,
    output: str,
    cfg: dict[str, Any],
    *,
    split: str = "test",
    checkpoint: str | None = None,
    baselines: str | list[str] | None = None,
) -> dict[str, Any]:
    bcfg = cfg.setdefault("external_baselines", {})
    if baselines is None:
        baselines = bcfg.get("methods", [bcfg.get("baseline", "route_bc_lite")])
    if isinstance(baselines, str):
        methods = [m.strip() for m in baselines.split(",") if m.strip()]
    else:
        methods = [str(m).strip() for m in baselines if str(m).strip()]
    if not methods:
        raise ValueError("No external baselines requested")
    model, model_cfg, device = _load_checkpoint(checkpoint, cfg)
    groups = group_sample_paths(dataset, split=split)
    records_by_method: dict[str, list[dict[str, Any]]] = {m: [] for m in methods}
    for gi, paths in enumerate(groups, 1):
        samples = [load_npz(p) for p in paths]
        samples = sorted(samples, key=lambda d: int(np.asarray(d.get("candidate_index", 0)).item()))
        model_outputs = _predict_group(model, samples, model_cfg, device)
        for method in methods:
            sel = select_external_policy(method, samples, model_cfg, model_outputs=model_outputs)
            records_by_method[method].append(_record_for_selection(method, samples, sel, model_cfg))
        if gi == 1 or gi % 500 == 0:
            print({"event": "external_eval_progress", "groups_done": gi, "num_groups": len(groups)}, flush=True)
    learned_methods = {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer", "gameformer", "gameformer_lite", "gameformer_levelk", "betop", "betop_lite", "betopnet", "betopnet_lite"}
    oracle_methods = {"oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"}
    summaries = {}
    for m in methods:
        ml = m.lower()
        if model is not None and ml in learned_methods:
            source = "learned_checkpoint_observation_only_inputs"
        elif ml in oracle_methods:
            source = "teacher_only_oracle_upper_bound"
        else:
            source = "observation_only_rule_or_optimizer"
        summaries[m] = _summarize(records_by_method[m], m, len(groups), source)
    result = {
        "dataset": str(dataset),
        "split": split,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "method_order": methods,
        "methods": summaries,
    }
    if output:
        write_json(result, output)
    return result
