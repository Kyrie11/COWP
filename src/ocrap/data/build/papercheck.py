from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import load_npz, parse_json_field, write_json
from ocrap.data.validation import missing_fields

from .diagnose import iter_sample_paths


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(x).item())
    except Exception:
        return default


def _offdiag_mean(M: np.ndarray) -> float:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(M.shape[0], dtype=bool)
    return float(M[mask].mean()) if mask.any() else 0.0


def papercheck_dataset(dataset: str | Path, output: str | Path | None = None, max_samples: int | None = None) -> dict:
    paths = iter_sample_paths(dataset, max_samples)
    failures: list[str] = []
    warnings: list[str] = []
    scene_time_candidates: dict[tuple[str, int], set[int]] = defaultdict(set)
    split_by_scene: dict[str, set[str]] = defaultdict(set)
    regime_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    off_y: list[float] = []
    off_c: list[float] = []
    disp: list[float] = []
    artifact = oracle_pos = neg_dep = 0
    odg_vals: list[float] = []
    odg_pos_vals: list[float] = []
    odg_art: list[float] = []
    hidden_emergence = hidden_from_unknown = hidden_invalid = 0
    missing_seen = False
    for p in paths:
        d = load_npz(p)
        miss = missing_fields(set(d.keys()))
        if miss and not missing_seen:
            failures.append(f"missing required fields: {','.join(miss)}")
            missing_seen = True
        scene = str(np.asarray(d.get("scene_id", p.stem)).item())
        time = int(_scalar(d.get("time_index", 0)))
        cand = int(_scalar(d.get("candidate_index", 0)))
        split = str(np.asarray(d.get("split_id", "unknown")).item())
        scene_time_candidates[(scene, time)].add(cand)
        split_by_scene[scene].add(split)
        rp = np.asarray(d.get("root_probs", []), dtype=float).reshape(-1)
        if rp.size == 0 or not np.isclose(rp.sum(), 1.0, atol=1e-3):
            failures.append(f"root_probs not normalized: {p.name}")
        C = np.asarray(d.get("c_star", []), dtype=float)
        Y = np.asarray(d.get("y_obs", []), dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] == 0 or not np.allclose(np.diag(C), 1.0, atol=1e-4):
            failures.append(f"C_star not square / diag not 1: {p.name}")
        if Y.ndim != 2 or Y.shape[0] != Y.shape[1] or not np.allclose(Y, Y.T, atol=1e-4):
            failures.append(f"Y_obs not symmetric: {p.name}")
        if Y.size:
            off_y.append(_offdiag_mean(Y))
        if C.size:
            off_c.append(_offdiag_mean(C))
        if "within_root_obs_dispersion" in d:
            vals = np.asarray(d["within_root_obs_dispersion"], dtype=float).reshape(-1)
            if vals.size:
                disp.append(float(np.mean(vals)))
        sources = [str(x) for x in np.asarray(d.get("future_sources", []), dtype=str).reshape(-1)]
        for s in sources:
            source_counts[s] += 1
        if not {"replay", "reactive", "targeted"}.issubset(set(sources)):
            failures.append(f"no replay/reactive/targeted source: {p.name}")
        metas = parse_json_field(d.get("future_metadata", "[]"), [])
        for m in metas if isinstance(metas, list) else []:
            if m.get("hidden_emergence", False):
                hidden_emergence += 1
                if m.get("from_unknown_mask", False):
                    hidden_from_unknown += 1
                if m.get("hidden_invalid_spawn", False):
                    hidden_invalid += 1
        regimes = parse_json_field(d.get("regime_label", "{}"), {})
        if isinstance(regimes, dict):
            for k, v in regimes.items():
                if v:
                    regime_counts[k] += 1
        r_orc = _scalar(d.get("r_orc_star", 0.0))
        r_dep = _scalar(d.get("r_dep_star", 0.0))
        gap = _scalar(d.get("oracle_gap_star", r_orc - r_dep))
        is_art = bool(int(_scalar(d.get("i_art_star", 0))))
        artifact += int(is_art)
        oracle_pos += int(r_orc >= 0.0)
        neg_dep += int(r_dep < 0.0)
        odg_vals.append(gap)
        odg_pos_vals.append(max(0.0, gap))
        if is_art:
            odg_art.append(gap)
    if any(len(splits) > 1 for splits in split_by_scene.values()):
        failures.append("scenario split leakage")
    num = len(paths)
    artifact_fraction = artifact / max(num, 1)
    oracle_frac = oracle_pos / max(num, 1)
    neg_frac = neg_dep / max(num, 1)
    mean_y = float(np.mean(off_y)) if off_y else 0.0
    mean_c = float(np.mean(off_c)) if off_c else 0.0
    counts = [len(v) for v in scene_time_candidates.values()]
    cand_stats = {"min": int(min(counts)) if counts else 0, "mean": float(np.mean(counts)) if counts else 0.0, "max": int(max(counts)) if counts else 0}
    dataset_name = str(dataset).lower()
    if hidden_emergence > hidden_from_unknown:
        failures.append("hidden_emergence_count > hidden_from_unknown_count")
    if hidden_invalid > 0:
        failures.append("hidden_invalid_spawn_count > 0")
    if num > 0 and (mean_y <= 0.02 or mean_y >= 0.98):
        failures.append("mean_offdiag_y_obs near 0 or near 1 for almost all samples")
    if ("artifact" in dataset_name or "mined" in dataset_name) and artifact_fraction == 0:
        failures.append("artifact_fraction == 0 in artifact/mined dataset")
    if cand_stats["min"] < 2 and num > 0:
        failures.append("candidate_count_per_scene_time.min < 2")
    if neg_frac == 0 and num > 0:
        failures.append("negative_deployable_fraction == 0")
    if artifact_fraction < (0.05 if "train" in dataset_name or "artifact" in dataset_name else 0.01) and num > 0:
        warnings.append("artifact_fraction below recommended threshold")
    if regime_counts.get("occluded", 0) == 0 and num > 0:
        warnings.append("occluded regime count == 0")
    if regime_counts.get("near_contact", 0) == 0 and num > 0:
        warnings.append("near_contact regime count == 0")
    if "post" in dataset_name and regime_counts.get("post_contact", 0) == 0:
        warnings.append("post_contact regime count == 0 when post-contact stress is enabled")
    if (float(np.mean(odg_pos_vals)) if odg_pos_vals else 0.0) <= 1e-6 and num > 0:
        warnings.append("odg_pos_mean <= small_threshold")
    if disp and float(np.mean(disp)) > 2.0:
        warnings.append("within_root_obs_dispersion too high")
    result = {
        "num_samples": num,
        "num_scene_time_groups": len(scene_time_candidates),
        "artifact_fraction": artifact_fraction,
        "oracle_recoverable_fraction": oracle_frac,
        "negative_deployable_fraction": neg_frac,
        "odg_mean": float(np.mean(odg_vals)) if odg_vals else 0.0,
        "odg_pos_mean": float(np.mean(odg_pos_vals)) if odg_pos_vals else 0.0,
        "odg_artifact_mean": float(np.mean(odg_art)) if odg_art else None,
        "regime_counts": dict(regime_counts),
        "future_source_coverage": dict(source_counts),
        "mean_offdiag_y_obs": mean_y,
        "mean_offdiag_c_star": mean_c,
        "hidden_emergence_count": hidden_emergence,
        "hidden_from_unknown_count": hidden_from_unknown,
        "hidden_invalid_spawn_count": hidden_invalid,
        "within_root_obs_dispersion_mean": float(np.mean(disp)) if disp else 0.0,
        "candidate_count_per_scene_time": cand_stats,
        "warnings": sorted(set(warnings)),
        "failures": sorted(set(failures)),
    }
    if output:
        write_json(result, output)
    return result
