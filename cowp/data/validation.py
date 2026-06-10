from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cowp.core.constants import MacroType, MechanismToken, NaturalSource, TOKEN_NAMES
from cowp.data.cache_schema import validate_numeric_invariants, validate_schema
from cowp.utils.progress import tqdm_iter


def validate_label_file(path: str | Path, cfg: dict) -> list[str]:
    with np.load(path, allow_pickle=True) as data:
        d = {k: data[k] for k in data.files}
    return validate_schema(d, cfg, strict=True) + validate_numeric_invariants(d, cfg)


def _scalar_string(data: Mapping[str, np.ndarray], key: str, default: str = "") -> str:
    if key not in data:
        return default
    val = np.asarray(data[key])
    if val.shape == ():
        item = val.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    if val.size == 0:
        return default
    item = val.flat[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return str(item)


def _safe_ratio(num: float | int, den: float | int) -> float:
    return float(num) / max(float(den), 1.0)


def _token_name(value: int) -> str:
    try:
        return TOKEN_NAMES.get(MechanismToken(int(value)), str(int(value)))
    except ValueError:
        return str(int(value))


def _macro_name(value: int) -> str:
    try:
        return MacroType(int(value)).name
    except ValueError:
        return str(int(value))


def _natural_source_name(value: int) -> str:
    try:
        return NaturalSource(int(value)).name
    except ValueError:
        return str(int(value))


def _finite_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.percentile(values, 10)), float(np.percentile(values, 50)), float(np.percentile(values, 90))


def summarize_label_file(path: str | Path) -> dict[str, float | int | str | bool]:
    with np.load(path, allow_pickle=True) as data_npz:
        data = {k: data_npz[k] for k in data_npz.files}
    sid = str(data["scenario/id"].item()) if "scenario/id" in data else Path(path).stem
    cand_valid = np.asarray(data["cowp/candidates/valid"], dtype=bool)
    crit_valid = np.asarray(data["cowp/critical/valid"], dtype=bool)
    witness = np.asarray(data["cowp/witness/exists"], dtype=bool)
    pair_mask = cand_valid[:, None] & crit_valid[None, :]
    valid_pair_count = int(np.sum(pair_mask))

    conventional = np.asarray(data["cowp/candidates/conventional_safe"], dtype=bool) & cand_valid
    false_safe = np.asarray(data["cowp/candidates/false_safe"], dtype=bool) & cand_valid
    ncf = np.asarray(data["cowp/candidates/noncoercive_feasible"], dtype=bool) & cand_valid
    positive_mask = witness & pair_mask

    response_valid = np.asarray(data.get("cowp/response/valid", np.zeros((*pair_mask.shape, 0))), dtype=bool)
    response_safe = np.asarray(data.get("cowp/response/is_safe", np.zeros_like(response_valid)), dtype=bool)
    response_low = np.asarray(data.get("cowp/response/is_low_burden", np.zeros_like(response_valid)), dtype=bool)
    has_safe_response = np.any(response_valid & response_safe, axis=-1) if response_valid.ndim == 3 else np.zeros_like(pair_mask)
    has_low_response = np.any(response_valid & response_safe & response_low, axis=-1) if response_valid.ndim == 3 else np.zeros_like(pair_mask)

    nat_valid = np.asarray(data.get("cowp/natural/valid", np.zeros((len(crit_valid), 0))), dtype=bool)
    nat_counts = nat_valid.sum(axis=1)[crit_valid] if nat_valid.ndim == 2 else np.zeros(0, dtype=np.int32)
    nat_weight = np.asarray(data.get("cowp/natural/weight", np.zeros_like(nat_valid, dtype=np.float32)), dtype=np.float32)
    nat_weight_mass = nat_weight[crit_valid].sum(axis=1) if nat_weight.ndim == 2 and np.any(crit_valid) else np.zeros(0, dtype=np.float32)
    beta = np.asarray(data.get("cowp/natural/beta", np.full(len(crit_valid), 0.65)), dtype=np.float32)
    nat_burden = np.asarray(data.get("cowp/natural/burden_neutral", np.zeros_like(nat_weight, dtype=np.float32)), dtype=np.float32)
    nat_low = (nat_burden <= beta[:, None]) & nat_valid if nat_valid.ndim == 2 and nat_burden.ndim == 2 else np.zeros_like(nat_valid, dtype=bool)
    nat_low_counts = nat_low.sum(axis=1)[crit_valid] if nat_low.ndim == 2 else np.zeros(0, dtype=np.int32)
    nat_source = np.asarray(data.get("cowp/natural/source", np.zeros_like(nat_valid, dtype=np.int32)), dtype=np.int32)
    natural_source_tokens: list[str] = []
    if nat_source.ndim == 2:
        for src in nat_source[nat_valid]:
            natural_source_tokens.append(_natural_source_name(int(src)))

    tokens = np.asarray(data["cowp/witness/token"])[positive_mask]
    token_names = [_token_name(int(t)) for t in tokens]

    opr_values = np.asarray(data["cowp/witness/opr"], dtype=np.float32)[pair_mask] if valid_pair_count else np.zeros(0, dtype=np.float32)
    c_i_values = np.asarray(data["cowp/witness/c_i"], dtype=np.float32)[pair_mask] if valid_pair_count else np.zeros(0, dtype=np.float32)
    min_safe_values = np.asarray(data["cowp/witness/min_safe_burden"], dtype=np.float32)[pair_mask] if valid_pair_count else np.zeros(0, dtype=np.float32)
    burden_values = np.asarray(data["cowp/witness/burden_total"], dtype=np.float32)[pair_mask] if valid_pair_count else np.zeros(0, dtype=np.float32)
    conflict_mass_values = np.asarray(data["cowp/witness/natural_conflict_mass"], dtype=np.float32)[pair_mask] if valid_pair_count else np.zeros(0, dtype=np.float32)
    mean_opr, p10_opr, p50_opr, p90_opr = _finite_stats(opr_values)
    mean_burden, p10_burden, p50_burden, p90_burden = _finite_stats(burden_values)
    mean_min_safe, _, _, p90_min_safe = _finite_stats(min_safe_values)
    mean_conflict_mass, _, _, p90_conflict_mass = _finite_stats(conflict_mass_values)
    mean_ci, _, _, p90_ci = _finite_stats(c_i_values)

    macro = np.asarray(data.get("cowp/candidates/macro_type", np.zeros(len(cand_valid))), dtype=np.int32)
    macro_tokens = [_macro_name(int(x)) for x in macro[cand_valid]]
    topology = np.asarray(data.get("cowp/candidates/topology_id", np.arange(len(cand_valid))), dtype=np.int32)
    traj = np.asarray(data.get("cowp/candidates/trajectory", np.zeros((len(cand_valid), 0, 7))), dtype=np.float32)
    endpoints = traj[cand_valid, -1, :2] if traj.ndim == 3 and traj.shape[1] > 0 and np.any(cand_valid) else np.zeros((0, 2), dtype=np.float32)
    endpoint_spread = float(np.mean(np.std(endpoints, axis=0))) if len(endpoints) > 1 else 0.0

    scene_types = _scalar_string(data, "dataset/scene_types", "")
    min_future_dist = float(np.asarray(data.get("dataset/min_future_dist", np.asarray(np.inf))).item())
    interaction_heavy = bool(np.asarray(data.get("dataset/interaction_heavy", np.asarray(True))).item())

    waymax_enabled = bool(np.asarray(data.get("waymax/enabled", np.asarray(False))).item())
    waymax_rollout_valid = np.asarray(data.get("waymax/candidate/rollout_valid", np.zeros_like(cand_valid)), dtype=bool) & cand_valid
    waymax_selected = np.asarray(data.get("waymax/candidate/selected_for_rollout", waymax_rollout_valid), dtype=bool) & cand_valid
    waymax_seconds = np.asarray(data.get("waymax/candidate/rollout_seconds", np.zeros_like(cand_valid, dtype=np.float32)), dtype=np.float32)
    waymax_metric_summary: dict[str, float] = {}
    for key, value in data.items():
        if not str(key).startswith("waymax/metrics/"):
            continue
        vals = np.asarray(value, dtype=np.float32)
        if vals.shape[:1] == cand_valid.shape[:1]:
            mask = waymax_rollout_valid & np.isfinite(vals)
            metric_name = str(key).split("/", 2)[-1].replace("__", "_")
            waymax_metric_summary[f"waymax_metric_mean_{metric_name}"] = float(np.nanmean(vals[mask])) if np.any(mask) else float("nan")

    row_extra = {
        "waymax_enabled": waymax_enabled,
        "waymax_rollout_candidates": int(np.sum(waymax_rollout_valid)),
        "waymax_rollout_candidate_ratio": _safe_ratio(np.sum(waymax_rollout_valid), np.sum(cand_valid)),
        "waymax_selected_candidates": int(np.sum(waymax_selected)),
        "waymax_missing_selected_candidates": int(np.sum(waymax_selected & ~waymax_rollout_valid)),
        "waymax_rollout_seconds_sum": float(np.nansum(waymax_seconds[waymax_rollout_valid])) if np.any(waymax_rollout_valid) else 0.0,
        "waymax_rollout_seconds_mean": float(np.nanmean(waymax_seconds[waymax_rollout_valid])) if np.any(waymax_rollout_valid) else 0.0,
    }
    row_extra.update(waymax_metric_summary)

    row = {
        "scenario_id": sid,
        "interaction_heavy": interaction_heavy,
        "scene_types": scene_types,
        "min_future_dist": min_future_dist,
        "candidate_valid": int(cand_valid.sum()),
        "critical_valid": int(crit_valid.sum()),
        "valid_pairs": valid_pair_count,
        "positive_pairs": int(np.sum(positive_mask)),
        "positive_pair_ratio": _safe_ratio(np.sum(positive_mask), valid_pair_count),
        "conventional_safe_candidates": int(np.sum(conventional)),
        "false_safe_candidates": int(np.sum(false_safe)),
        "ncf_candidates": int(np.sum(ncf)),
        "conventional_safe_candidate_ratio": _safe_ratio(np.sum(conventional), np.sum(cand_valid)),
        "false_safe_candidate_ratio": _safe_ratio(np.sum(false_safe), np.sum(cand_valid)),
        "false_safe_among_conventional_safe": _safe_ratio(np.sum(false_safe), np.sum(conventional)),
        "ncf_candidate_ratio": _safe_ratio(np.sum(ncf), np.sum(cand_valid)),
        "stress_eligible": bool(np.any(false_safe) and np.any(ncf)),
        "response_has_safe_pairs": int(np.sum(has_safe_response & pair_mask)),
        "response_has_low_burden_pairs": int(np.sum(has_low_response & pair_mask)),
        "response_safe_pair_ratio": _safe_ratio(np.sum(has_safe_response & pair_mask), valid_pair_count),
        "response_low_burden_pair_ratio": _safe_ratio(np.sum(has_low_response & pair_mask), valid_pair_count),
        "mean_natural_alternatives": float(np.mean(nat_counts)) if len(nat_counts) else 0.0,
        "min_natural_alternatives": int(np.min(nat_counts)) if len(nat_counts) else 0,
        "mean_low_burden_natural_alternatives": float(np.mean(nat_low_counts)) if len(nat_low_counts) else 0.0,
        "min_low_burden_natural_alternatives": int(np.min(nat_low_counts)) if len(nat_low_counts) else 0,
        "mean_natural_weight_mass": float(np.mean(nat_weight_mass)) if len(nat_weight_mass) else 0.0,
        "mean_opr": mean_opr,
        "p10_opr": p10_opr,
        "p50_opr": p50_opr,
        "p90_opr": p90_opr,
        "max_cbs": float(np.nanmax(burden_values[np.isfinite(burden_values)])) if np.any(np.isfinite(burden_values)) else 0.0,
        "mean_cbs": mean_burden,
        "p10_cbs": p10_burden,
        "p50_cbs": p50_burden,
        "p90_cbs": p90_burden,
        "mean_min_safe_burden": mean_min_safe,
        "p90_min_safe_burden": p90_min_safe,
        "mean_natural_conflict_mass": mean_conflict_mass,
        "p90_natural_conflict_mass": p90_conflict_mass,
        "mean_c_i": mean_ci,
        "p90_c_i": p90_ci,
        "opr_sum": float(np.nansum(opr_values[np.isfinite(opr_values)])),
        "cbs_sum": float(np.nansum(burden_values[np.isfinite(burden_values)])),
        "min_safe_burden_sum": float(np.nansum(min_safe_values[np.isfinite(min_safe_values)])),
        "conflict_mass_sum": float(np.nansum(conflict_mass_values[np.isfinite(conflict_mass_values)])),
        "candidate_macro_types": ",".join(sorted(set(macro_tokens))),
        "candidate_macro_type_count": int(len(set(macro_tokens))),
        "candidate_topology_count": int(len(set(int(x) for x in topology[cand_valid]))) if np.any(cand_valid) else 0,
        "candidate_endpoint_spread_m": endpoint_spread,
        "natural_sources": ",".join(sorted(set(natural_source_tokens))),
        "natural_source_count": int(len(set(natural_source_tokens))),
        "tokens": ",".join(str(int(t)) for t in tokens),
        "mechanism_tokens": ",".join(token_names),
        "mechanism_token_count": int(len(set(token_names))),
    }
    row.update(row_extra)
    return row


def _counter_from_csv_tokens(series: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for tokstr in series.fillna(""):
        for tok in str(tokstr).split(","):
            tok = tok.strip()
            if tok:
                counter[tok] += 1
    return counter


def _sum_col(df: pd.DataFrame, name: str) -> float:
    return float(df[name].sum()) if name in df and not df.empty else 0.0


def _mean_col(df: pd.DataFrame, name: str) -> float:
    return float(df[name].mean()) if name in df and not df.empty else 0.0


def _make_quality_report(stats: dict[str, object], cfg: dict) -> dict[str, object]:
    diag_cfg = cfg.get("diagnostics", {})
    checks: list[dict[str, object]] = []

    def add_check(name: str, value: float | int, ok: bool, target: str, severity: str = "warn") -> None:
        checks.append({"name": name, "value": float(value), "target": target, "ok": bool(ok), "severity": severity})

    add_check("validation_error_files", int(stats.get("validation_error_files", 0)), int(stats.get("validation_error_files", 0)) == 0, "0", "error")
    add_check("mean_candidate_valid", float(stats.get("mean_candidate_valid", 0.0)), float(stats.get("mean_candidate_valid", 0.0)) >= float(diag_cfg.get("min_mean_candidate_valid", 12.0)), f">= {diag_cfg.get('min_mean_candidate_valid', 12.0)}")
    add_check("mean_critical_valid", float(stats.get("mean_critical_valid", 0.0)), float(stats.get("mean_critical_valid", 0.0)) >= float(diag_cfg.get("min_mean_critical_valid", 1.0)), f">= {diag_cfg.get('min_mean_critical_valid', 1.0)}")
    if "max_mean_critical_valid" in diag_cfg:
        add_check("mean_critical_valid_upper", float(stats.get("mean_critical_valid", 0.0)), float(stats.get("mean_critical_valid", 0.0)) <= float(diag_cfg.get("max_mean_critical_valid", 6.5)), f"<= {diag_cfg.get('max_mean_critical_valid', 6.5)}")
    add_check("scenes_with_ncf_candidate_ratio", float(stats.get("scenes_with_ncf_candidate_ratio", 0.0)), float(stats.get("scenes_with_ncf_candidate_ratio", 0.0)) >= float(diag_cfg.get("min_scenes_with_ncf_candidate_ratio", 0.0)), f">= {diag_cfg.get('min_scenes_with_ncf_candidate_ratio', 0.0)}")
    add_check("mean_candidate_endpoint_spread_m", float(stats.get("mean_candidate_endpoint_spread_m", 0.0)), float(stats.get("mean_candidate_endpoint_spread_m", 0.0)) >= float(diag_cfg.get("min_mean_candidate_endpoint_spread_m", 0.0)), f">= {diag_cfg.get('min_mean_candidate_endpoint_spread_m', 0.0)}")
    pos_lo, pos_hi = diag_cfg.get("target_positive_pair_ratio", [0.02, 0.35])
    add_check("positive_pair_ratio", float(stats.get("positive_pair_ratio", 0.0)), float(pos_lo) <= float(stats.get("positive_pair_ratio", 0.0)) <= float(pos_hi), f"{pos_lo} - {pos_hi}")
    fs_lo, fs_hi = diag_cfg.get("target_false_safe_candidate_ratio", [0.05, 0.45])
    add_check("false_safe_candidate_ratio", float(stats.get("false_safe_candidate_ratio", 0.0)), float(fs_lo) <= float(stats.get("false_safe_candidate_ratio", 0.0)) <= float(fs_hi), f"{fs_lo} - {fs_hi}")
    ncf_lo, ncf_hi = diag_cfg.get("target_ncf_candidate_ratio", [0.10, 0.80])
    add_check("ncf_candidate_ratio", float(stats.get("ncf_candidate_ratio", 0.0)), float(ncf_lo) <= float(stats.get("ncf_candidate_ratio", 0.0)) <= float(ncf_hi), f"{ncf_lo} - {ncf_hi}")
    add_check("stress_eligible_scene_ratio", float(stats.get("stress_eligible_scene_ratio", 0.0)), float(stats.get("stress_eligible_scene_ratio", 0.0)) >= float(diag_cfg.get("min_stress_eligible_scene_ratio", 0.01)), f">= {diag_cfg.get('min_stress_eligible_scene_ratio', 0.01)}")
    add_check("response_safe_pair_ratio", float(stats.get("response_safe_pair_ratio", 0.0)), float(stats.get("response_safe_pair_ratio", 0.0)) >= float(diag_cfg.get("min_response_safe_pair_ratio", 0.60)), f">= {diag_cfg.get('min_response_safe_pair_ratio', 0.60)}")
    add_check("mean_natural_alternatives", float(stats.get("mean_natural_alternatives", 0.0)), float(stats.get("mean_natural_alternatives", 0.0)) >= float(diag_cfg.get("min_mean_natural_alternatives", 4.0)), f">= {diag_cfg.get('min_mean_natural_alternatives', 4.0)}")
    token_types = len(stats.get("mechanism_token_counts", {}) or {})
    add_check("mechanism_token_types", token_types, token_types >= int(diag_cfg.get("min_mechanism_token_types", 2)), f">= {diag_cfg.get('min_mechanism_token_types', 2)}")
    if float(stats.get("waymax_enabled_scene_ratio", 0.0)) > 0.0:
        add_check("waymax_missing_selected_candidates", int(stats.get("waymax_missing_selected_candidates", 0)), int(stats.get("waymax_missing_selected_candidates", 0)) == 0, "0", "error")
        add_check("mean_waymax_rollout_candidate_ratio", float(stats.get("mean_waymax_rollout_candidate_ratio", 0.0)), float(stats.get("mean_waymax_rollout_candidate_ratio", 0.0)) >= float(diag_cfg.get("min_waymax_rollout_candidate_ratio", 0.90)), f">= {diag_cfg.get('min_waymax_rollout_candidate_ratio', 0.90)}")

    hard_fail = any((not c["ok"]) and c["severity"] == "error" for c in checks)
    warn_fail = any(not c["ok"] for c in checks)
    assessment = "pass" if not warn_fail else ("fail" if hard_fail else "warn")
    return {
        "assessment": assessment,
        "checks": checks,
        "interpretation": {
            "pass": "Diagnostics are broadly consistent with supporting COWP supervision.",
            "warn": "The dataset can be used for smoke tests, but at least one distributional signal is weak for the paper's stress setting.",
            "fail": "Schema or invariant errors should be fixed before training/evaluation.",
        }[assessment],
    }


def write_visual_diagnostics(labels_dir: str | Path, df: pd.DataFrame, output_dir: str | Path, max_examples: int = 16, progress: bool = True) -> list[dict[str, object]]:
    labels_dir = Path(labels_dir)
    vis_dir = Path(output_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return []
    scored = df.copy()
    scored["_interest_score"] = (
        3.0 * scored["stress_eligible"].astype(float)
        + 2.0 * scored["false_safe_candidate_ratio"].astype(float)
        + 1.5 * scored["positive_pair_ratio"].astype(float)
        + (1.0 - scored["mean_opr"].clip(0.0, 1.0)).astype(float)
        + scored["p90_cbs"].astype(float)
    )
    scored = scored.sort_values("_interest_score", ascending=False).head(max_examples)
    manifest: list[dict[str, object]] = []
    try:
        from cowp.waymax_eval.visualize import plot_witness_scene
    except Exception as exc:  # pragma: no cover - optional matplotlib path
        return [{"error": f"could not import visualization backend: {exc}"}]
    iterator = tqdm_iter(scored.to_dict("records"), enabled=progress, total=len(scored), desc="Write diagnostic visualizations", unit="scene")
    for row in iterator:
        sid = str(row["scenario_id"])
        label_path = labels_dir / f"{sid}.npz"
        if not label_path.exists():
            continue
        with np.load(label_path, allow_pickle=True) as data_npz:
            label = {k: data_npz[k] for k in data_npz.files}
        cand_valid = np.asarray(label["cowp/candidates/valid"], dtype=bool)
        false_safe = np.asarray(label["cowp/candidates/false_safe"], dtype=bool) & cand_valid
        witness = np.asarray(label["cowp/witness/exists"], dtype=bool)
        candidate_pool = np.where(false_safe)[0]
        if candidate_pool.size == 0:
            candidate_pool = np.where(np.any(witness, axis=1) & cand_valid)[0]
        if candidate_pool.size == 0:
            candidate_pool = np.where(cand_valid)[0]
        if candidate_pool.size == 0:
            continue
        k = int(candidate_pool[0])
        out = vis_dir / f"{sid}_cand{k}.png"
        plot_witness_scene(label, k, out)
        manifest.append({
            "scenario_id": sid,
            "candidate_idx": k,
            "path": str(out),
            "positive_pairs": int(row.get("positive_pairs", 0)),
            "false_safe_candidates": int(row.get("false_safe_candidates", 0)),
            "ncf_candidates": int(row.get("ncf_candidates", 0)),
            "mean_opr": float(row.get("mean_opr", 0.0)),
            "p90_cbs": float(row.get("p90_cbs", 0.0)),
        })
    return manifest


def diagnose_dataset(
    labels_dir: str | Path,
    cfg: dict,
    output_dir: str | Path,
    progress: bool = True,
    make_visualizations: bool = False,
    max_visualizations: int = 16,
) -> pd.DataFrame:
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(labels_dir.glob("*.npz"))
    rows = []
    validation_errors: dict[str, list[str]] = {}
    iterator = tqdm_iter(paths, enabled=progress, total=len(paths), desc="Diagnose COWP labels", unit="file")
    for p in iterator:
        errs = validate_label_file(p, cfg)
        if errs:
            validation_errors[p.name] = errs
        rows.append(summarize_label_file(p))
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(errors=len(validation_errors))
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dataset_diagnostics.csv", index=False)
    with (output_dir / "validation_errors.json").open("w", encoding="utf-8") as f:
        json.dump(validation_errors, f, indent=2, ensure_ascii=False)

    token_counter = _counter_from_csv_tokens(df["mechanism_tokens"]) if not df.empty and "mechanism_tokens" in df else Counter()
    macro_counter = _counter_from_csv_tokens(df["candidate_macro_types"]) if not df.empty and "candidate_macro_types" in df else Counter()
    source_counter = _counter_from_csv_tokens(df["natural_sources"]) if not df.empty and "natural_sources" in df else Counter()
    scene_type_counter = _counter_from_csv_tokens(df["scene_types"]) if not df.empty and "scene_types" in df else Counter()
    total_valid_pairs = int(_sum_col(df, "valid_pairs"))
    total_candidate_valid = int(_sum_col(df, "candidate_valid"))
    total_conventional_safe = int(_sum_col(df, "conventional_safe_candidates"))

    stats = {
        "num_scenes": int(len(df)),
        "interaction_heavy_scene_ratio": float(df["interaction_heavy"].mean()) if not df.empty and "interaction_heavy" in df else 0.0,
        "stress_eligible_scenes": int(df["stress_eligible"].sum()) if not df.empty else 0,
        "stress_eligible_scene_ratio": float(df["stress_eligible"].mean()) if not df.empty else 0.0,
        "scenes_with_positive_witness": int((df["positive_pairs"] > 0).sum()) if not df.empty else 0,
        "scenes_with_false_safe_candidate": int((df["false_safe_candidates"] > 0).sum()) if not df.empty else 0,
        "scenes_with_ncf_candidate": int((df["ncf_candidates"] > 0).sum()) if not df.empty else 0,
        "scenes_with_ncf_candidate_ratio": float((df["ncf_candidates"] > 0).mean()) if not df.empty else 0.0,
        "mean_candidate_valid": _mean_col(df, "candidate_valid"),
        "mean_critical_valid": _mean_col(df, "critical_valid"),
        "mean_valid_pairs": _mean_col(df, "valid_pairs"),
        "positive_pair_ratio": _safe_ratio(_sum_col(df, "positive_pairs"), total_valid_pairs),
        "conventional_safe_candidate_ratio": _safe_ratio(_sum_col(df, "conventional_safe_candidates"), total_candidate_valid),
        "false_safe_candidate_ratio": _safe_ratio(_sum_col(df, "false_safe_candidates"), total_candidate_valid),
        "false_safe_among_conventional_safe": _safe_ratio(_sum_col(df, "false_safe_candidates"), total_conventional_safe),
        "ncf_candidate_ratio": _safe_ratio(_sum_col(df, "ncf_candidates"), total_candidate_valid),
        "response_safe_pair_ratio": _safe_ratio(_sum_col(df, "response_has_safe_pairs"), total_valid_pairs),
        "response_low_burden_pair_ratio": _safe_ratio(_sum_col(df, "response_has_low_burden_pairs"), total_valid_pairs),
        "mean_natural_alternatives": _mean_col(df, "mean_natural_alternatives"),
        "mean_low_burden_natural_alternatives": _mean_col(df, "mean_low_burden_natural_alternatives"),
        "mean_natural_weight_mass": _mean_col(df, "mean_natural_weight_mass"),
        "mean_opr": _safe_ratio(_sum_col(df, "opr_sum"), total_valid_pairs),
        "mean_cbs": _safe_ratio(_sum_col(df, "cbs_sum"), total_valid_pairs),
        "mean_min_safe_burden": _safe_ratio(_sum_col(df, "min_safe_burden_sum"), total_valid_pairs),
        "mean_natural_conflict_mass": _safe_ratio(_sum_col(df, "conflict_mass_sum"), total_valid_pairs),
        "p10_opr_scene_mean": _mean_col(df, "p10_opr"),
        "p90_cbs_scene_mean": _mean_col(df, "p90_cbs"),
        "mean_candidate_macro_type_count": _mean_col(df, "candidate_macro_type_count"),
        "mean_candidate_topology_count": _mean_col(df, "candidate_topology_count"),
        "mean_candidate_endpoint_spread_m": _mean_col(df, "candidate_endpoint_spread_m"),
        "mechanism_token_counts": dict(token_counter),
        "candidate_macro_type_scene_counts": dict(macro_counter),
        "natural_source_scene_counts": dict(source_counter),
        "scene_type_counts": dict(scene_type_counter),
        "validation_error_files": int(len(validation_errors)),
        "waymax_enabled_scene_ratio": float(df["waymax_enabled"].mean()) if not df.empty and "waymax_enabled" in df else 0.0,
        "mean_waymax_rollout_candidate_ratio": _mean_col(df, "waymax_rollout_candidate_ratio"),
        "waymax_missing_selected_candidates": int(_sum_col(df, "waymax_missing_selected_candidates")),
        "mean_waymax_rollout_seconds_per_candidate": _safe_ratio(_sum_col(df, "waymax_rollout_seconds_sum"), _sum_col(df, "waymax_rollout_candidates")),
    }
    if not df.empty:
        for col in [c for c in df.columns if c.startswith("waymax_metric_mean_")]:
            stats[col] = _mean_col(df, col)
    quality_report = _make_quality_report(stats, cfg)
    stats["quality_assessment"] = quality_report["assessment"]
    with (output_dir / "dataset_diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with (output_dir / "dataset_quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    if not df.empty:
        interesting = df.assign(
            interest_score=(
                3.0 * df["stress_eligible"].astype(float)
                + 2.0 * df["false_safe_candidate_ratio"].astype(float)
                + 1.5 * df["positive_pair_ratio"].astype(float)
                + (1.0 - df["mean_opr"].clip(0.0, 1.0)).astype(float)
                + df["p90_cbs"].astype(float)
            )
        ).sort_values("interest_score", ascending=False)
        interesting.head(200).to_csv(output_dir / "dataset_diagnostic_interesting_scenes.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "dataset_diagnostic_interesting_scenes.csv", index=False)

    if make_visualizations:
        manifest = write_visual_diagnostics(labels_dir, df, output_dir, max_examples=max_visualizations, progress=progress)
        with (output_dir / "visualization_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    return df
