from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.serialization import load_npz, write_json
from ocrap.evaluation.baselines import BASELINES, select_baseline, _bucket_aliases
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation, post_contact_deployability_score, predicted_shared_option_success, summarize_selection_metrics
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample, teacher_prediction_from_sample


def _threshold_lookup(thresholds: dict, delta: float | str | None) -> tuple[float | None, str | None]:
    if not thresholds or delta is None or str(delta) == "":
        return None, None
    candidates = [str(delta)]
    try:
        fd = float(delta)
        candidates.extend([str(fd), f"{fd:g}", f"{fd:.12g}"])
    except Exception:
        fd = None
    for key in candidates:
        if key in thresholds:
            return float(thresholds[key]), key
    if fd is not None:
        for key, val in thresholds.items():
            try:
                if abs(float(key) - fd) <= 1e-12:
                    return float(val), str(key)
            except Exception:
                continue
    return None, None


def _load_gamma(calibration_json: str | Path | None, cfg: dict | None = None) -> float:
    gamma = float(((cfg or {}).get("selection", {}) or {}).get("gamma_rec", 0.0))
    if calibration_json:
        import json

        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        delta = ((cfg or {}).get("evaluation", {}) or {}).get("delta", "")
        found, _key = _threshold_lookup(cal.get("thresholds", {}) or {}, delta)
        if found is not None:
            gamma = float(found)
        else:
            gamma = float(cal.get("gamma_rec", cal.get("gamma", gamma)))
    if not np.isfinite(gamma) and not bool(((cfg or {}).get("evaluation", {}) or {}).get("allow_infinite_gamma", False)):
        raise ValueError(
            "Loaded gamma_rec is not finite. Check calibration.thresholds for the requested evaluation.delta, "
            "increase calibration set size / delta, or pass --set evaluation.allow_infinite_gamma=true for debugging only."
        )
    return gamma


def _load_json_mapping(path: str | Path) -> dict[str, float]:
    import json

    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("gamma_rec_by_bucket"), dict):
        raw = raw["gamma_rec_by_bucket"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
            if np.isfinite(fv):
                out[str(k)] = fv
        except Exception:
            continue
    return out


def _apply_gamma_rec_by_bucket_file(cfg: dict) -> dict:
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    local = dict(cfg)
    new_sel = dict(sel)
    changed = False

    # Backward-compatible scalar recovery threshold map.
    path = sel.get("gamma_rec_by_bucket_file", sel.get("gamma_rec_by_bucket_path", None))
    if path:
        mapping = _load_json_mapping(path)
        if mapping:
            existing = new_sel.get("gamma_rec_by_bucket", {})
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(mapping)
            new_sel["gamma_rec_by_bucket"] = merged
            changed = True

    # v15: allow calibrated auxiliary selector maps to be supplied as JSON files
    # using the same bucket-name format as gamma_rec_by_bucket_file.
    for file_key, target_key in [
        ("option_drs_certificate_threshold_by_bucket_file", "option_drs_certificate_threshold_by_bucket"),
        ("option_drs_certificate_max_gap_by_bucket_file", "option_drs_certificate_max_gap_by_bucket"),
        ("option_drs_certificate_rec_slack_by_bucket_file", "option_drs_certificate_rec_slack_by_bucket"),
    ]:
        path = sel.get(file_key, None)
        if not path:
            continue
        mapping = _load_json_mapping(path)
        if not mapping:
            continue
        existing = new_sel.get(target_key, {})
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(mapping)
        new_sel[target_key] = merged
        changed = True

    if changed:
        local["selection"] = new_sel
        return local
    return cfg
def _strip_version_suffix(name: str) -> str:
    base, sep, version = str(name).rpartition("_v")
    return base if sep and version.isdigit() and base else str(name)


def _gamma_aliases(name: str | None) -> list[str]:
    if not name:
        return []
    raw = str(name)
    aliases = [raw]
    for p in ("test_", "val_", "train_"):
        if raw.startswith(p):
            aliases.append(raw[len(p):])
    aliases.extend([_strip_version_suffix(x) for x in list(aliases)])
    out: list[str] = []
    for x in aliases:
        if x and x not in out:
            out.append(x)
    return out


def _gamma_for_dataset(base_gamma: float, cfg: dict, dataset_label: str | None) -> float:
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    mapping = sel_cfg.get("gamma_rec_by_bucket", {})
    if not isinstance(mapping, dict) or not dataset_label:
        return float(base_gamma)
    for key in _gamma_aliases(dataset_label):
        if key in mapping and mapping[key] not in {None, ""}:
            try:
                val = float(mapping[key])
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    return float(base_gamma)


def _prefix_nominal_deviation_items(items: list[dict]) -> np.ndarray:
    if not items:
        return np.zeros((0,), dtype=np.float32)
    try:
        ref = np.asarray(items[0]["data"]["prefix_states"], dtype=float)[:, :2]
    except Exception:
        return np.zeros((len(items),), dtype=np.float32)
    vals: list[float] = []
    for x in items:
        try:
            xy = np.asarray(x["data"]["prefix_states"], dtype=float)[:, :2]
            T = min(len(ref), len(xy))
            vals.append(0.0 if T <= 0 else float(np.sqrt(np.mean(np.sum((xy[:T] - ref[:T]) ** 2, axis=-1))) / 5.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)


def _method_list(cfg: dict | None) -> list[str]:
    cfg = cfg or {}
    methods = ((cfg.get("evaluation", {}) or {}).get("methods", None) if isinstance(cfg.get("evaluation", {}), dict) else None)
    if not methods:
        return ["ocrap"]
    out = [str(m).lower() for m in methods]
    return [m for m in out if m in BASELINES]


def _records_summary(records: list[dict], split: str, gamma: float, source: str, num_groups: int) -> dict:
    result = summarize_selection_metrics(records)
    if records:
        result["pred_ODG"] = float(np.mean([r["pred_odg"] for r in records]))
        result["mean_selected_pred_R_dep"] = float(np.mean([r.get("pred_r_dep", 0.0) for r in records]))
        result["mean_selected_pred_gap"] = float(np.mean([r.get("pred_gap", 0.0) for r in records]))
        result["mean_selected_pred_DRS_proxy"] = float(np.mean([r.get("pred_drs", 0.0) for r in records]))
        from collections import Counter
        result["selection_reason_counts"] = dict(Counter(str(r.get("selection_reason", "")) for r in records))
        result["mean_selected_teacher_R_dep"] = float(np.mean([r["selected_teacher_r_dep"] for r in records]))
        result["mean_selected_teacher_R_orc"] = float(np.mean([r["selected_teacher_r_orc"] for r in records]))
        result["mean_selected_utility"] = float(np.mean([r["selected_utility"] for r in records]))
        result["intervention_rate"] = float(np.mean([int(r.get("selected_index", 0)) != 0 for r in records]))
        result["selected_admitted_rate"] = float(np.mean([bool(r.get("selected_admitted", False)) for r in records]))
        result["mean_num_admitted"] = float(np.mean([float(r.get("num_admitted", 0.0)) for r in records]))
        result["mean_num_admitted_interventions"] = float(np.mean([float(r.get("num_admitted_interventions", 0.0)) for r in records]))
    result.update({"num_scene_time_groups": int(num_groups), "num_records": int(len(records)), "split": split, "gamma_rec": gamma, "source": source})
    if records:
        gammas = [float(r.get("gamma_rec", gamma)) for r in records if np.isfinite(float(r.get("gamma_rec", gamma)))]
        if gammas and (max(gammas) - min(gammas) > 1.0e-9):
            result["gamma_rec_min"] = float(min(gammas))
            result["gamma_rec_max"] = float(max(gammas))
    return result




def _write_method_tables(result: dict, output: str | Path) -> None:
    """Write slide-friendly method comparison tables next to the JSON output."""
    try:
        out = Path(output)
        methods = result.get("methods", {}) or {}
        order = result.get("method_order", list(methods.keys()))
        cols = [
            "FRA_exec", "FRA_cand", "DRS", "bounded_NUP", "ODG",
            "artifact_selection_rate", "post_contact_deployability", "intervention_rate", "selected_admitted_rate", "mean_num_admitted_interventions", "mean_selected_teacher_R_dep", "mean_selected_utility",
        ]
        pretty = {
            "FRA_exec": "Executed false recovery admission ↓",
            "FRA_cand": "Admitted false-recoverable candidates ↓",
            "DRS": "Deployable recovery success ↑",
            "bounded_NUP": "Nominal utility preservation ↑",
            "ODG": "Selected oracle-to-deployable gap ↓",
            "artifact_selection_rate": "Oracle-artifact selection rate ↓",
            "post_contact_deployability": "Post-contact deployability ↑",
            "intervention_rate": "Intervention rate ↓",
            "selected_admitted_rate": "Selected admitted rate ↑",
            "mean_num_admitted_interventions": "Certified intervention candidates ↑",
            "mean_selected_teacher_R_dep": "Selected deployable recovery score ↑",
            "mean_selected_utility": "Selected utility ↑",
        }
        csv_lines = ["method," + ",".join(cols)]
        md = ["| Method | " + " | ".join(pretty[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for m in order:
            row = methods.get(m, {}) or {}
            vals = []
            for c in cols:
                v = row.get(c, None)
                vals.append("" if v is None else f"{float(v):.4f}")
            csv_lines.append(m + "," + ",".join(vals))
            md.append("| " + m + " | " + " | ".join(vals) + " |")
        out.with_suffix(".methods.csv").write_text("\n".join(csv_lines), encoding="utf-8")
        out.with_suffix(".methods.md").write_text("\n".join(md), encoding="utf-8")
    except Exception:
        return

def _dataset_label_for_path(path: Path) -> str:
    try:
        # samples/foo.npz -> dataset root name; fallback to parent.
        if path.parent.name == "samples":
            return path.parent.parent.name
        return path.parent.name
    except Exception:
        return "dataset"



def _prediction_cfg_for_dataset(cfg: dict, dataset_label: str) -> dict:
    """Supply the active regime to the v44 value expert at prediction time."""
    if not dataset_label:
        return cfg
    local = dict(cfg)
    sel = dict(local.get("selection", {}) or {}) if isinstance(local.get("selection", {}), dict) else {}
    sel["active_bucket_name"] = dataset_label
    local["selection"] = sel
    return local

def _normalise_split_id(value) -> str:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if hasattr(value, "item"):
            value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
        s = str(value).strip()
        # Some manifests/NPZ scalars may stringify bytes as b'test'.
        if (s.startswith("b'") and s.endswith("'")) or (s.startswith('b"') and s.endswith('"')):
            s = s[2:-1]
        return s.strip().lower()
    except Exception:
        return ""


def _path_implied_split(path: Path) -> str:
    label = _dataset_label_for_path(path).lower()
    for prefix in ("train", "val", "test"):
        if label == prefix or label.startswith(prefix + "_") or label.startswith(prefix + "-"):
            return prefix
    return ""


def _split_matches_path(path: Path, split: str) -> bool:
    want = _normalise_split_id(split)
    if not want or want == "all":
        return True
    got = _normalise_split_id(scalar_metadata_for_path(path, "split_id", ""))
    if got == want:
        return True
    # If split_id metadata is missing or stale after renaming/copying dataset
    # roots, use the dataset root name as a conservative fallback.
    if got in {"", "none", "nan"}:
        return _path_implied_split(path) == want
    return False



def _drs_success_gamma_for_dataset(base_gamma: float, cfg: dict, dataset_label: str | None) -> float:
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    # Deployability success should be evaluated at a fixed physical threshold by
    # default.  The calibrated recovery gamma is an admission threshold and can
    # be negative in contact, which makes predicted DRS proxies spuriously high.
    default = sel.get("drs_success_gamma", 0.0)
    for map_key in ("drs_success_gamma_by_bucket", "drs_success_gamma_by_regime"):
        m = sel.get(map_key, None)
        if isinstance(m, dict):
            for key in _bucket_aliases(dataset_label or ""):
                if key in m and m[key] not in {None, ""}:
                    try:
                        return float(m[key])
                    except Exception:
                        pass
    try:
        return float(default)
    except Exception:
        return 0.0


def _validate_eval_selector_config(cfg: dict, methods: list[str]) -> None:
    if "ocrap" not in [str(m).lower() for m in methods]:
        return
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    selector = str(sel.get("ocrap_selector", sel.get("selector", "lcb_constrained"))).lower()
    if bool(eval_cfg.get("require_calibrated_selector", False)) and selector not in {"calibrated", "calibrated_constrained", "soft_constrained", "budgeted_calibrated"}:
        raise ValueError(f"evaluation requires calibrated OC-RAP selector, but selection.ocrap_selector={selector!r}")
    if bool(eval_cfg.get("require_gamma_by_bucket", False)) and not isinstance(sel.get("gamma_rec_by_bucket", None), dict):
        raise ValueError("evaluation requires selection.gamma_rec_by_bucket to be loaded")
    if bool(eval_cfg.get("require_gamma_by_bucket", False)) and not sel.get("gamma_rec_by_bucket", {}):
        raise ValueError("evaluation requires non-empty selection.gamma_rec_by_bucket; check gamma_rec_by_bucket_file path")

def _evaluate_grouped_items(grouped: dict[tuple, list[dict]], methods: list[str], gamma: float, gamma_H: float, gamma_D: float, cfg: dict, split: str, source: str) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    method_records: dict[str, list[dict]] = {m: [] for m in methods}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    use_running_budget = bool(eval_cfg.get("use_running_intervention_budget", False))
    # Stateless offline evaluation hid the selector's exposure-control behavior:
    # the same residual-tail certificate could fire independently on many
    # scene-time groups, making the policy look like frequent braking even when
    # the closed-loop selector was budgeted.  When enabled, mirror the closed-loop
    # runner by passing a running intervention count and cooldown state to the
    # OC-RAP selector.  Counters are separated by method and dataset label so
    # regime-specific budgets remain comparable and deterministic.
    budget_state: dict[tuple[str, str], dict[str, int]] = {
        (str(m).lower(), ""): {"seen": 0, "used": 0, "last": -10**9} for m in methods
    }

    def _group_sort_key(k):
        if isinstance(k, tuple):
            return tuple(str(x) if not isinstance(x, (int, float)) else x for x in k)
        return (str(k),)

    for key in sorted(grouped.keys(), key=_group_sort_key):
        items = grouped[key]
        items.sort(key=lambda x: int(np.asarray(x["data"]["candidate_index"]).item()))
        utility = np.array([float(np.asarray(x["data"].get("utility", 0.0)).item()) for x in items])
        pred_r_dep = np.array([float(x["pred"].r_dep) for x in items])
        pred_r_orc = np.array([float(x["pred"].r_orc) for x in items])
        pred_gap = np.array([float(getattr(x["pred"], "gap", x["pred"].r_orc - x["pred"].r_dep)) for x in items])
        nominal_deviation = _prefix_nominal_deviation_items(items)
        dataset_label = str(items[0].get("dataset_label", "")) if items else ""
        gamma_i = _gamma_for_dataset(gamma, cfg, dataset_label)
        drs_gamma_i = _drs_success_gamma_for_dataset(gamma_i, cfg, dataset_label)
        pred_drs = np.array([predicted_shared_option_success(x["pred"].q, x["pred"].root_probs, gamma=drs_gamma_i, root_valid=x["data"].get("root_valid", None), option_valid=x["data"].get("option_valid", None)) for x in items])
        pred_direct_value = np.array([np.nan if x["pred"].direct_recovery_value is None else float(x["pred"].direct_recovery_value) for x in items])
        pred_direct_rank = np.array([np.nan if x["pred"].direct_recovery_rank is None else float(x["pred"].direct_recovery_rank) for x in items])
        pred_direct_rank = np.where(np.isfinite(pred_direct_rank), pred_direct_rank, pred_direct_value)
        pred_direct_std = np.array([np.nan if x["pred"].direct_recovery_std is None else float(x["pred"].direct_recovery_std) for x in items])
        pred_direct_delta = np.array([np.nan if x["pred"].direct_recovery_delta is None else float(x["pred"].direct_recovery_delta) for x in items])
        pred_direct_delta_std = np.array([np.nan if x["pred"].direct_recovery_delta_std is None else float(x["pred"].direct_recovery_delta_std) for x in items])
        pred_direct_opportunity = np.array([np.nan if x["pred"].direct_recovery_opportunity is None else float(x["pred"].direct_recovery_opportunity) for x in items])
        pred_direct_harm = np.array([np.nan if x["pred"].direct_recovery_harm is None else float(x["pred"].direct_recovery_harm) for x in items])
        opp_logits = np.array([np.nan if x["pred"].direct_recovery_opportunity_logit is None else float(x["pred"].direct_recovery_opportunity_logit) for x in items])
        harm_logits = np.array([np.nan if x["pred"].direct_recovery_harm_logit is None else float(x["pred"].direct_recovery_harm_logit) for x in items])
        nominal_ids = [i for i, x in enumerate(items) if float(np.asarray(x["data"].get("is_nominal", 0.0)).item()) > 0.5]
        sel0=(cfg.get("selection",{}) or {}) if isinstance(cfg.get("selection",{}),dict) else {}
        if nominal_ids:
            ni=nominal_ids[0]; risk_source=str(sel0.get("direct_value_risk_source","heads") or "heads").lower()
            if risk_source=="conformal_delta" and np.isfinite(pred_direct_delta).any():
                dm=np.where(np.isfinite(pred_direct_delta),pred_direct_delta,-np.inf); dm[ni]=0.0
                q=float(sel0.get("direct_value_conformal_overprediction_quantile",sel0.get("direct_value_additive_q",0.0)) or 0.0)
                temp=max(1e-4,float(sel0.get("direct_value_conformal_temperature",0.02) or 0.02))
                pg=float(sel0.get("direct_value_positive_gain",0.015)); ng=float(sel0.get("direct_value_negative_gain",0.010))
                lcb=dm-q
                pred_direct_value=dm.astype(np.float32); pred_direct_std=np.zeros_like(dm,dtype=np.float32)
                pred_direct_opportunity=(1.0/(1.0+np.exp(-np.clip((lcb-pg)/temp,-30,30)))).astype(np.float32)
                pred_direct_harm=(1.0/(1.0+np.exp(-np.clip((-ng-lcb)/temp,-30,30)))).astype(np.float32)
            elif risk_source=="direct_delta" and np.isfinite(pred_direct_delta).any():
                import math
                pred_direct_value = np.where(np.isfinite(pred_direct_delta), pred_direct_delta, -np.inf)
                pred_direct_value[ni] = 0.0
                pred_direct_std = np.where(np.isfinite(pred_direct_delta_std), pred_direct_delta_std, np.inf)
                pred_direct_std[ni] = 0.0
                dm=pred_direct_value; ds=np.maximum(1e-6,pred_direct_std)
                pg=float(sel0.get("direct_value_positive_gain",0.015)); ng=float(sel0.get("direct_value_negative_gain",0.010))
                cdf=np.vectorize(lambda z:0.5*(1.0+math.erf(float(np.clip(z,-12,12))/math.sqrt(2.0))))
                pred_direct_opportunity=cdf((dm-pg)/ds).astype(np.float32); pred_direct_harm=cdf((-ng-dm)/ds).astype(np.float32)
            elif risk_source=="ordinal_evidence" and np.isfinite(opp_logits[ni]) and np.isfinite(harm_logits[ni]):
                od=np.clip(opp_logits-opp_logits[ni],-30,30); hd=np.clip(harm_logits-harm_logits[ni],-30,30)
                pred_direct_opportunity=(1.0/(1.0+np.exp(-od))).astype(np.float32)
                pred_direct_harm=(1.0/(1.0+np.exp(-hd))).astype(np.float32)
                pred_direct_value=(pred_direct_opportunity-pred_direct_harm).astype(np.float32); pred_direct_value[ni]=0.0
                pred_direct_std=np.zeros_like(pred_direct_value,dtype=np.float32)
            elif risk_source=="delta_distribution" and np.isfinite(pred_direct_value[ni]):
                import math
                dm=pred_direct_value-pred_direct_value[ni]; ds=np.sqrt(np.maximum(1e-6,pred_direct_std**2+pred_direct_std[ni]**2))
                pg=float(sel0.get("direct_value_positive_gain",0.015)); ng=float(sel0.get("direct_value_negative_gain",0.010))
                cdf=np.vectorize(lambda z:0.5*(1.0+math.erf(float(np.clip(z,-12,12))/math.sqrt(2.0))))
                pred_direct_opportunity=cdf((dm-pg)/ds).astype(np.float32); pred_direct_harm=cdf((-ng-dm)/ds).astype(np.float32)
            elif np.isfinite(opp_logits[ni]):
                delta = np.clip(opp_logits - opp_logits[ni], -30.0, 30.0)
                pred_direct_opportunity = (1.0 / (1.0 + np.exp(-delta))).astype(np.float32)
                if np.isfinite(harm_logits[ni]):
                    hdelta=np.clip(harm_logits-harm_logits[ni],-30.0,30.0); pred_direct_harm=(1.0/(1.0+np.exp(-hdelta))).astype(np.float32)
        macro_names = [str(np.asarray(x["data"].get("prefix_macro_name", "")).item() if np.asarray(x["data"].get("prefix_macro_name", "")).shape == () else x["data"].get("prefix_macro_name", "")) for x in items]
        teacher_r_dep = np.array([float(np.asarray(x["data"]["r_dep_star"]).item()) for x in items])
        teacher_r_orc = np.array([float(np.asarray(x["data"]["r_orc_star"]).item()) for x in items])
        hard = np.array([float(np.asarray(x["data"].get("hard_violation", 0.0)).item()) for x in items])
        harm = np.array([float(np.asarray(x["data"].get("harm_proxy", 0.0)).item()) for x in items])
        feasible = np.array([bool(int(np.asarray(x["data"].get("feasible", 1)).item())) for x in items])
        base_local_cfg = dict(cfg) if dataset_label else cfg
        if dataset_label:
            local_sel0 = dict(base_local_cfg.get("selection", {}) or {}) if isinstance(base_local_cfg.get("selection", {}), dict) else {}
            local_sel0["active_bucket_name"] = dataset_label
            base_local_cfg["selection"] = local_sel0

        for method in methods:
            method_l = str(method).lower()
            local_cfg = base_local_cfg
            state_key = (method_l, dataset_label or "")
            if use_running_budget and method_l == "ocrap":
                st = budget_state.setdefault(state_key, {"seen": 0, "used": 0, "last": -10**9})
                local_cfg = dict(base_local_cfg)
                local_sel = dict(local_cfg.get("selection", {}) or {}) if isinstance(local_cfg.get("selection", {}), dict) else {}
                local_sel["active_bucket_name"] = dataset_label
                local_sel["intervention_budget_used"] = int(st.get("used", 0))
                local_sel["intervention_budget_steps"] = max(1, int(st.get("seen", 0)) + 1)
                local_sel["steps_since_last_intervention"] = int(st.get("seen", 0)) - int(st.get("last", -10**9))
                local_cfg["selection"] = local_sel
            sel = select_baseline(
                method, utility, pred_r_dep, teacher_r_dep, teacher_r_orc, hard, harm, feasible,
                gamma_i, gamma_H, gamma_D, local_cfg,
                pred_r_orc=pred_r_orc,
                pred_gap=pred_gap,
                nominal_deviation=nominal_deviation,
                pred_drs=pred_drs,
                pred_direct_value=pred_direct_value,
                pred_direct_rank=pred_direct_rank,
                pred_direct_std=pred_direct_std,
                pred_direct_opportunity=pred_direct_opportunity,
                pred_direct_harm=pred_direct_harm,
                candidate_macro_names=macro_names,
            )
            selected_index = int(sel.selected_index)
            chosen = items[selected_index]
            sd = chosen["data"]
            # DRS must evaluate one globally shared recovery action.  When
            # OC-RAP actually switches away from nominal, the option is chosen
            # from the model q-table and then evaluated against teacher margins.
            # If no intervention happened (selected_index == 0), DRS is a
            # nominal-prefix diagnostic rather than a learned recovery-action
            # claim, so use the teacher option just like the nominal baseline.
            q_eval = chosen["pred"].q if (method == "ocrap" and selected_index != 0) else chosen["teacher"].q
            opt_gamma = drs_gamma_i if (method == "ocrap" and selected_index != 0) else 0.0
            selected_option = best_shared_option_index(q_eval, sd["root_probs"], gamma=opt_gamma, root_valid=sd.get("root_valid", None), option_valid=sd.get("option_valid", None))
            drs = deployable_recovery_success(sd["m_star"], sd["root_probs"], int(selected_option), sd.get("root_valid", None))
            odg_val = float(np.asarray(sd.get("oracle_gap_star", teacher_r_orc[selected_index] - teacher_r_dep[selected_index])).item())
            pcds = post_contact_deployability_score(drs, teacher_r_dep[selected_index], odg_val)
            nup = nominal_utility_preservation(utility[0] if len(utility) else 0.0, utility[selected_index], sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
            method_records[method].append({
                "fra_cand": false_recoverability_admission(sel.admitted, teacher_r_dep),
                "fra_exec": float(teacher_r_dep[selected_index] < 0.0),
                "drs": drs,
                "odg": odg_val,
                "pred_odg": float(pred_r_orc[selected_index] - pred_r_dep[selected_index]),
                "pred_r_dep": float(pred_r_dep[selected_index]),
                "pred_gap": float(pred_gap[selected_index]),
                "pred_drs": float(pred_drs[selected_index]),
                "post_contact_deployability": float(pcds),
                "nup": nup["bounded_NUP"],
                "artifact": bool(int(np.asarray(sd.get("i_art_star", 0)).item())),
                "selected_artifact": bool(int(np.asarray(sd.get("i_art_star", 0)).item())),
                "selection_reason": sel.reason,
                "gamma_rec": float(gamma_i),
                "selected_index": selected_index,
                "selected_utility": float(utility[selected_index]),
                "selected_teacher_r_dep": float(teacher_r_dep[selected_index]),
                "selected_teacher_r_orc": float(teacher_r_orc[selected_index]),
                "selected_admitted": bool(sel.admitted[selected_index]) if 0 <= selected_index < len(sel.admitted) else False,
                "num_admitted": int(np.asarray(sel.admitted, dtype=bool).sum()),
                "num_admitted_interventions": int(np.asarray(sel.admitted, dtype=bool)[1:].sum()) if len(sel.admitted) > 1 else 0,
            })
            if use_running_budget and method_l == "ocrap":
                st = budget_state.setdefault(state_key, {"seen": 0, "used": 0, "last": -10**9})
                if selected_index != 0:
                    st["used"] = int(st.get("used", 0)) + 1
                    st["last"] = int(st.get("seen", 0))
                st["seen"] = int(st.get("seen", 0)) + 1

    summaries = {m: _records_summary(rs, split, gamma, source if m == "ocrap" else "dataset_label_baseline", len(grouped)) for m, rs in method_records.items()}
    return summaries, method_records


def evaluate(dataset: str | Path, checkpoint: str | Path | None = None, output: str | Path | None = None, split: str = "test", calibration_json: str | Path | None = None, cfg: dict | None = None) -> dict:
    cfg = _apply_gamma_rec_by_bucket_file(dict(cfg or {}))
    paths = iter_sample_paths_many(dataset)
    print({"event": "evaluate_start", "num_npz_paths": len(paths), "split": split, "dataset": str(dataset)}, flush=True)
    bundle = load_model_bundle(checkpoint, cfg)
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    group_by_dataset = bool(eval_cfg.get("group_by_dataset", True))
    grouped: dict[tuple, list[dict]] = {}
    dataset_grouped: dict[str, dict[tuple, list[dict]]] = {}
    for idx, p in enumerate(paths, 1):
        if idx == 1 or idx % 1000 == 0:
            print({"event": "evaluate_progress", "seen": idx, "groups": len(grouped)}, flush=True)
        if not _split_matches_path(Path(p), split):
            continue
        d = load_npz(p)
        dataset_label = _dataset_label_for_path(Path(p))
        key_base = (str(np.asarray(d["scene_id"]).item()), int(np.asarray(d["time_index"]).item()))
        key = (dataset_label, *key_base) if group_by_dataset else key_base
        pred_cfg = _prediction_cfg_for_dataset(cfg, dataset_label)
        pred = predict_sample(d, bundle, pred_cfg)
        teacher = teacher_prediction_from_sample(d, cfg)
        record = {"path": p, "dataset_label": dataset_label, "data": d, "pred": pred, "teacher": teacher}
        grouped.setdefault(key, []).append(record)
        dataset_grouped.setdefault(dataset_label, {}).setdefault(key_base, []).append(record)

    if not grouped and split and str(split).lower() != "all" and bool(eval_cfg.get("fallback_to_all_if_empty_split", True)) and paths:
        print({"event": "evaluate_empty_split_retry_all", "requested_split": str(split), "num_npz_paths": len(paths)}, flush=True)
        grouped = {}
        dataset_grouped = {}
        for p in paths:
            d = load_npz(p)
            dataset_label = _dataset_label_for_path(Path(p))
            key_base = (str(np.asarray(d["scene_id"]).item()), int(np.asarray(d["time_index"]).item()))
            key = (dataset_label, *key_base) if group_by_dataset else key_base
            pred_cfg = _prediction_cfg_for_dataset(cfg, dataset_label)
            pred = predict_sample(d, bundle, pred_cfg)
            teacher = teacher_prediction_from_sample(d, cfg)
            record = {"path": p, "dataset_label": dataset_label, "data": d, "pred": pred, "teacher": teacher}
            grouped.setdefault(key, []).append(record)
            dataset_grouped.setdefault(dataset_label, {}).setdefault(key_base, []).append(record)
    print({"event": "evaluate_grouping_done", "num_scene_time_groups": len(grouped), "group_by_dataset": group_by_dataset, "dataset_labels": sorted(dataset_grouped)}, flush=True)
    gamma = _load_gamma(calibration_json, cfg)
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    gamma_H = float(sel_cfg.get("gamma_H", 0.0))
    gamma_D = float(sel_cfg.get("gamma_D", 5.0))
    methods = _method_list(cfg)
    _validate_eval_selector_config(cfg, methods)
    source = "model" if bundle is not None else "teacher_fallback"
    summaries, _records = _evaluate_grouped_items(grouped, methods, gamma, gamma_H, gamma_D, cfg, split, source)
    result = summaries.get("ocrap", next(iter(summaries.values()), {}))
    result = dict(result)
    result["methods"] = summaries
    result["method_order"] = methods
    result["group_by_dataset"] = group_by_dataset
    result["gamma_rec_by_bucket"] = (cfg.get("selection", {}) or {}).get("gamma_rec_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    result["selector_config"] = {
        "ocrap_selector": (cfg.get("selection", {}) or {}).get("ocrap_selector", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "drs_success_gamma": (cfg.get("selection", {}) or {}).get("drs_success_gamma", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "safe_force_nominal_when_feasible": (cfg.get("selection", {}) or {}).get("safe_force_nominal_when_feasible", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "safe_force_nominal_mode": (cfg.get("selection", {}) or {}).get("safe_force_nominal_mode", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "safe_force_nominal_when_feasible_by_bucket": (cfg.get("selection", {}) or {}).get("safe_force_nominal_when_feasible_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "safe_force_nominal_mode_by_bucket": (cfg.get("selection", {}) or {}).get("safe_force_nominal_mode_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "stress_preserve_nominal_min_drs_drop_by_bucket": (cfg.get("selection", {}) or {}).get("stress_preserve_nominal_min_drs_drop_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "require_admitted_intervention": (cfg.get("selection", {}) or {}).get("require_admitted_intervention", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "require_admitted_intervention_by_bucket": (cfg.get("selection", {}) or {}).get("require_admitted_intervention_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "require_intervention_evidence": (cfg.get("selection", {}) or {}).get("require_intervention_evidence", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "require_intervention_evidence_by_bucket": (cfg.get("selection", {}) or {}).get("require_intervention_evidence_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "intervention_min_rec_lcb_gain_by_bucket": (cfg.get("selection", {}) or {}).get("intervention_min_rec_lcb_gain_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "intervention_min_drs_gain_by_bucket": (cfg.get("selection", {}) or {}).get("intervention_min_drs_gain_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "intervention_min_gap_reduction_by_bucket": (cfg.get("selection", {}) or {}).get("intervention_min_gap_reduction_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "option_drs_certificate": (cfg.get("selection", {}) or {}).get("option_drs_certificate", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "option_drs_certificate_by_bucket": (cfg.get("selection", {}) or {}).get("option_drs_certificate_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "option_drs_certificate_threshold_by_bucket": (cfg.get("selection", {}) or {}).get("option_drs_certificate_threshold_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "option_drs_certificate_max_gap_by_bucket": (cfg.get("selection", {}) or {}).get("option_drs_certificate_max_gap_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "option_drs_certificate_rec_slack_by_bucket": (cfg.get("selection", {}) or {}).get("option_drs_certificate_rec_slack_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_certificate": (cfg.get("selection", {}) or {}).get("relative_recovery_certificate", None) if isinstance(cfg.get("selection", {}), dict) else None,
        "relative_recovery_certificate_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_certificate_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_min_rec_gain_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_min_rec_gain_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_min_drs_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_min_drs_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_max_gap_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_max_gap_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_min_gap_reduction_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_min_gap_reduction_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_gate_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_gate_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "relative_recovery_use_recovery_pool_by_bucket": (cfg.get("selection", {}) or {}).get("relative_recovery_use_recovery_pool_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "recovery_cert_max_hard_by_bucket": (cfg.get("selection", {}) or {}).get("recovery_cert_max_hard_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
        "recovery_cert_max_harm_by_bucket": (cfg.get("selection", {}) or {}).get("recovery_cert_max_harm_by_bucket", {}) if isinstance(cfg.get("selection", {}), dict) else {},
    }
    result["dataset_group_count"] = {k: len(v) for k, v in sorted(dataset_grouped.items())}
    result["per_dataset"] = {}
    for label, sub_grouped in sorted(dataset_grouped.items()):
        sub_summaries, _ = _evaluate_grouped_items(sub_grouped, methods, gamma, gamma_H, gamma_D, cfg, split, source)
        sub_result = dict(sub_summaries.get("ocrap", next(iter(sub_summaries.values()), {})))
        sub_result["methods"] = sub_summaries
        sub_result["method_order"] = methods
        result["per_dataset"][label] = sub_result
    if output:
        write_json(result, output)
        _write_method_tables(result, output)
    return result
