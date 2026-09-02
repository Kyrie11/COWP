from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz, parse_json_field, read_json, write_json
from ocrap.data.validation import missing_fields


REQUIRED_FUTURE_SOURCES = {"replay", "reactive", "targeted"}
FINITE_ARRAY_FIELDS = [
    "agent_history",
    "agent_valid",
    "map_polylines",
    "map_valid",
    "dynamic_map",
    "route",
    "bev_occ",
    "ego_state",
    "prefix_states",
    "prefix_controls",
    "prefix_param",
    "future_probs",
    "future_valid",
    "root_probs",
    "root_signature",
    "root_future_signature",
    "root_valid",
    "future_to_root_weight",
    "within_root_obs_dispersion",
    "obs_distance",
    "y_obs",
    "c_star",
    "m_star",
    "option_valid",
    "recovery_params",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
]


def _dataset_specs(dataset: str | Path) -> list[Path]:
    """Parse one or more OC-RAP dataset directories.

    ``diagnose`` historically accepted a single directory.  Training already
    accepts comma-separated shard lists, so diagnostics should mirror that
    behavior.  Commas are used instead of shell globs to avoid accidentally
    mixing unrelated runs.
    """
    if isinstance(dataset, Path):
        return [dataset]
    raw = str(dataset).strip()
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [Path(x) for x in parts]


def _summary_nodes(summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield one dataset summary and any summaries embedded by shard merging."""
    if not isinstance(summary, dict):
        return
    yield summary
    for child in summary.get("shard_summaries", []) or []:
        if isinstance(child, dict):
            yield from _summary_nodes(child)


def _dataset_contract_metadata(dataset: str | Path, cfg: dict | None) -> dict[str, Any]:
    """Recover the build-time regime contract from dataset summaries.

    ``ocrap diagnose`` is normally invoked without repeating every build
    override.  The CLI therefore supplies the global defaults, which used to
    make a clean Safe dataset look like a generic stress dataset.  Builder
    summaries already retain the relevant quality/artifact settings, so merge
    those over the CLI defaults and keep a path-name fallback for legacy data.
    """
    quality = dict(_cfg_get(cfg, ("dataset_quality",), {}) or {})
    artifact = dict(_cfg_get(cfg, ("artifact",), {}) or {})
    generation: dict[str, Any] = {}
    summary_paths: list[str] = []
    for root in _dataset_specs(dataset):
        summary_path = root / "dataset_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = read_json(summary_path)
        except Exception:
            continue
        summary_paths.append(str(summary_path))
        for node in _summary_nodes(summary):
            if isinstance(node.get("dataset_quality"), dict):
                quality.update(node["dataset_quality"])
            if isinstance(node.get("artifact"), dict):
                artifact.update(node["artifact"])
            if isinstance(node.get("generation"), dict):
                generation.update(node["generation"])

    raw_required = quality.get("require_nominal_regimes", [])
    if isinstance(raw_required, str):
        required = {x.strip() for x in raw_required.strip("[]").split(",") if x.strip()}
    elif isinstance(raw_required, (list, tuple, set)):
        required = {str(x).strip() for x in raw_required if str(x).strip()}
    else:
        required = set()
    roots = _dataset_specs(dataset)
    legacy_safe_name = bool(roots) and all("safe" in root.name.lower() for root in roots)
    nominal_regime_dataset = bool(
        quality.get("nominal_regime_dataset", False)
        or (
            "normal" in required
            and not bool(artifact.get("force_mine", False))
            and float(artifact.get("mine_probability", 0.0) or 0.0) <= 0.0
        )
        or legacy_safe_name
    )
    return {
        "dataset_quality": quality,
        "artifact": artifact,
        "generation": generation,
        "summary_paths": summary_paths,
        "nominal_regime_dataset": nominal_regime_dataset,
        "legacy_safe_name_fallback": bool(legacy_safe_name and "normal" not in required),
    }


def iter_sample_paths(dataset: str | Path, max_samples: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in _dataset_specs(dataset):
        if (root / "samples").exists():
            local = sorted((root / "samples").glob("*.npz"))
        else:
            local = sorted(root.glob("*.npz"))
        paths.extend(local)
    # Preserve deterministic order and remove duplicate paths when users pass the
    # same shard twice.
    paths = sorted(dict.fromkeys(paths))
    return paths[:max_samples] if max_samples else paths


def _scalar(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(x).item())
    except Exception:
        return default


def _int_scalar(x: Any, default: int = 0) -> int:
    try:
        return int(round(float(np.asarray(x).item())))
    except Exception:
        return default


def _str_scalar(x: Any, default: str = "") -> str:
    try:
        return str(np.asarray(x).item())
    except Exception:
        return default


def _stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def _counter_dict(counter: Counter, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(k): int(v) for k, v in items}


def _offdiag_values(M: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float64)
    mask = ~np.eye(M.shape[0], dtype=bool)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        if v.size == M.shape[0]:
            mask &= v[:, None] & v[None, :]
    return M[mask]


def _safe_array(d: dict[str, Any], key: str, dtype: Any | None = None) -> np.ndarray | None:
    if key not in d:
        return None
    try:
        arr = np.asarray(d[key], dtype=dtype) if dtype is not None else np.asarray(d[key])
        return arr
    except Exception:
        return None


def _json_field(d: dict[str, Any], key: str, default: Any) -> Any:
    if key not in d:
        return default
    return parse_json_field(d[key], default)


def _check_square_matrix(name: str, M: np.ndarray | None, K: int | None, failures: list[str], p: Path) -> None:
    if M is None:
        return
    if M.ndim != 2 or M.shape[0] != M.shape[1] or (K is not None and M.shape[0] != K):
        failures.append(f"{name} shape mismatch in {p.name}: shape={tuple(M.shape)}, K={K}")


def _best_options(M: np.ndarray, option_valid: np.ndarray | None) -> np.ndarray:
    X = np.asarray(M, dtype=np.float64).copy()
    if X.ndim != 2 or X.shape[1] == 0:
        return np.zeros((0,), dtype=np.int64)
    if option_valid is not None and option_valid.size == X.shape[1]:
        X[:, ~option_valid.astype(bool)] = -np.inf
    return np.argmax(X, axis=1).astype(np.int64)


def _alias_incompatibility(M: np.ndarray, Y: np.ndarray, root_valid: np.ndarray | None, option_valid: np.ndarray | None) -> tuple[int, int, int]:
    if M.ndim != 2 or Y.ndim != 2 or Y.shape[0] != Y.shape[1] or Y.shape[0] != M.shape[0]:
        return 0, 0, 0
    valid = np.ones(M.shape[0], dtype=bool) if root_valid is None or root_valid.size != M.shape[0] else root_valid.astype(bool)
    best = _best_options(M, option_valid)
    alias_pairs = 0
    incompatible = 0
    same = 0
    for i in range(M.shape[0]):
        if not valid[i]:
            continue
        for j in range(i + 1, M.shape[0]):
            if not valid[j] or Y[i, j] < 0.5:
                continue
            alias_pairs += 1
            if best.size == M.shape[0] and best[i] != best[j]:
                incompatible += 1
            else:
                same += 1
    return alias_pairs, incompatible, same


def _cfg_get(cfg: dict | None, path: tuple[str, ...], default: Any) -> Any:
    cur: Any = cfg or {}
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def diagnose_dataset(dataset: str | Path, output: str | Path | None = None, max_samples: int | None = None, cfg: dict | None = None) -> dict:
    paths = iter_sample_paths(dataset, max_samples)
    failures: list[str] = []
    warnings: list[str] = []
    contract_meta = _dataset_contract_metadata(dataset, cfg)
    quality_cfg = contract_meta["dataset_quality"]
    artifact_cfg = contract_meta["artifact"]
    generation_cfg = contract_meta["generation"]
    nominal_regime_dataset = bool(contract_meta["nominal_regime_dataset"])

    split_by_scene: dict[str, set[str]] = defaultdict(set)
    sample_split_counts: Counter[str] = Counter()
    scene_split_counts: dict[str, set[str]] = defaultdict(set)
    candidate_by_scene_time: dict[tuple[str, int], set[int]] = defaultdict(set)
    nominal_by_scene_time: Counter[tuple[str, int]] = Counter()
    macro_counts: Counter[str] = Counter()
    time_reason_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    nominal_regime_counts: Counter[str] = Counter()
    nominal_sample_count = 0
    source_counts: Counter[str] = Counter()
    targeted_type_counts: Counter[str] = Counter()
    hidden_intent_counts: Counter[str] = Counter()
    artifact_pair_branches: dict[str, set[str]] = defaultdict(set)
    root_cluster_meta_counts: Counter[str] = Counter()
    teacher_backend_counts: Counter[str] = Counter()
    margin_override_future_count = 0
    complete_artifact_pair_sample_count = 0
    artifact_by_macro: Counter[str] = Counter()
    nonartifact_by_macro: Counter[str] = Counter()

    missing_field_counter: Counter[str] = Counter()
    missing_field_samples = 0
    finite_failure_samples = 0
    shape_failure_samples = 0
    source_complete_samples = 0
    ocmero_checked = 0
    ocmero_max_abs_error = 0.0

    future_count_vals: list[float] = []
    root_count_vals: list[float] = []
    valid_root_count_vals: list[float] = []
    option_count_vals: list[float] = []
    candidate_utility_vals: list[float] = []
    hard_violation_vals: list[float] = []
    harm_proxy_vals: list[float] = []
    feasible_vals: list[float] = []
    future_entropy_vals: list[float] = []
    root_entropy_vals: list[float] = []
    r_orc_vals: list[float] = []
    r_dep_vals: list[float] = []
    odg_vals: list[float] = []
    odg_pos_vals: list[float] = []
    odg_art_vals: list[float] = []
    off_y_vals: list[float] = []
    off_c_vals: list[float] = []
    obs_distance_vals: list[float] = []
    within_root_disp_vals: list[float] = []
    unknown_ratio_vals: list[float] = []
    alias_pair_vals: list[float] = []
    incompatible_alias_vals: list[float] = []
    same_alias_vals: list[float] = []
    best_option_diversity_vals: list[float] = []

    artifact_count = 0
    oracle_recoverable_count = 0
    negative_deployable_count = 0
    hidden_emergence_count = 0
    hidden_from_unknown_count = 0
    hidden_invalid_spawn_count = 0
    hidden_visible_free_spawn_count = 0
    hidden_start_checked = 0
    hidden_start_violation_count = 0
    plausibility_failed_futures = 0
    plausibility_failure_counts: Counter[str] = Counter()
    waymax_runtime_futures = 0
    total_metadata_futures = 0
    runtime_backend_counts: Counter[str] = Counter()
    synthetic_scene_count = 0

    alpha = float(_cfg_get(cfg, ("ocmero", "alpha"), 0.2))
    beta = float(_cfg_get(cfg, ("ocmero", "beta"), 0.2))
    top_m = int(_cfg_get(cfg, ("ocmero", "top_m"), 8))
    use_lcvar = not bool(_cfg_get(cfg, ("ablation", "without_lower_tail"), False))
    use_obs_kernel = not bool(_cfg_get(cfg, ("ablation", "without_observation_kernel"), False))
    hidden_delay = int(_cfg_get(cfg, ("hidden_emergence_delay_steps",), 2))
    expected_targeted = int(generation_cfg.get("num_targeted_futures", 0) or 0)
    required_sources = {"replay", "reactive"}
    if expected_targeted > 0 and not nominal_regime_dataset:
        required_sources.add("targeted")
    incomplete_source_samples = 0

    for p in paths:
        try:
            d = load_npz(p)
        except Exception as e:
            failures.append(f"cannot load sample {p.name}: {e}")
            continue

        sample_failures_before = len(failures)
        miss = missing_fields(set(d.keys()))
        if miss:
            missing_field_samples += 1
            for k in miss:
                missing_field_counter[k] += 1
            failures.append(f"missing required fields in {p.name}: {','.join(miss[:12])}")

        scene = _str_scalar(d.get("scene_id", p.stem), p.stem)
        if scene.startswith("synthetic"):
            synthetic_scene_count += 1
        time = _int_scalar(d.get("time_index", 0), 0)
        cand = _int_scalar(d.get("candidate_index", 0), 0)
        split = _str_scalar(d.get("split_id", "unknown"), "unknown")
        sample_split_counts[split] += 1
        split_by_scene[scene].add(split)
        scene_split_counts[split].add(scene)
        candidate_by_scene_time[(scene, time)].add(cand)
        is_nominal = bool(_int_scalar(d.get("is_nominal", 0), 0))
        if is_nominal:
            nominal_by_scene_time[(scene, time)] += 1
            nominal_sample_count += 1

        macro = _str_scalar(d.get("prefix_macro_name", "unknown"), "unknown")
        macro_counts[macro] += 1
        candidate_utility_vals.append(_scalar(d.get("utility", 0.0)))
        hard_violation_vals.append(_scalar(d.get("hard_violation", 0.0)))
        harm_proxy_vals.append(_scalar(d.get("harm_proxy", 0.0)))
        feasible_vals.append(float(_int_scalar(d.get("feasible", 0), 0)))

        finite_bad = False
        for key in FINITE_ARRAY_FIELDS:
            arr = _safe_array(d, key)
            if arr is None or arr.dtype.kind not in "biufc":
                continue
            if not np.all(np.isfinite(arr.astype(np.float64, copy=False))):
                finite_bad = True
                failures.append(f"non-finite values in {key}: {p.name}")
                break
        finite_failure_samples += int(finite_bad)

        future_probs = _safe_array(d, "future_probs", float)
        root_probs = _safe_array(d, "root_probs", float)
        root_valid = _safe_array(d, "root_valid", float)
        option_valid = _safe_array(d, "option_valid", float)
        root_assignments = _safe_array(d, "root_assignments", int)
        f2r = _safe_array(d, "future_to_root_weight", float)
        M = _safe_array(d, "m_star", float)
        Y = _safe_array(d, "y_obs", float)
        C = _safe_array(d, "c_star", float)
        Dobs = _safe_array(d, "obs_distance", float)

        F = int(future_probs.size) if future_probs is not None else None
        K = int(root_probs.size) if root_probs is not None else None
        L = int(option_valid.size) if option_valid is not None else None
        if F is not None:
            future_count_vals.append(F)
            s = float(future_probs.sum())
            if not np.isclose(s, 1.0, atol=1e-3):
                failures.append(f"future_probs not normalized in {p.name}: sum={s:.6g}")
            fp = future_probs[future_probs > 0]
            if fp.size:
                future_entropy_vals.append(float(-np.sum(fp * np.log(fp))))
        if K is not None:
            root_count_vals.append(K)
            s = float(root_probs.sum())
            if not np.isclose(s, 1.0, atol=1e-3):
                failures.append(f"root_probs not normalized in {p.name}: sum={s:.6g}")
            rp = root_probs[root_probs > 0]
            if rp.size:
                root_entropy_vals.append(float(-np.sum(rp * np.log(rp))))
        if root_valid is not None:
            rv = root_valid.astype(bool).reshape(-1)
            valid_root_count_vals.append(int(rv.sum()))
        else:
            rv = None
        if L is not None:
            option_count_vals.append(L)

        shape_bad = False
        if F is not None and root_assignments is not None and root_assignments.size != F:
            shape_bad = True
            failures.append(f"root_assignments length mismatch in {p.name}: {root_assignments.size} != {F}")
        if F is not None and root_assignments is not None and K is not None:
            if np.any((root_assignments < 0) | (root_assignments >= K)):
                shape_bad = True
                failures.append(f"root_assignments out of range in {p.name}")
            else:
                implied = np.bincount(root_assignments.reshape(-1), weights=future_probs.reshape(-1) if future_probs is not None else None, minlength=K).astype(float)
                if root_probs is not None and not np.allclose(implied, root_probs, atol=2e-3):
                    warnings.append(f"root_probs differ from future assignment mass in {p.name}")
        if root_valid is not None and K is not None and root_valid.size != K:
            shape_bad = True
            failures.append(f"root_valid length mismatch in {p.name}: {root_valid.size} != {K}")
        if option_valid is not None and M is not None and M.ndim == 2 and option_valid.size != M.shape[1]:
            shape_bad = True
            failures.append(f"option_valid length mismatch in {p.name}: {option_valid.size} != m_star L {M.shape[1]}")
        if M is not None and (M.ndim != 2 or (K is not None and M.shape[0] != K)):
            shape_bad = True
            failures.append(f"m_star shape mismatch in {p.name}: shape={tuple(M.shape)}, K={K}")
        for name, mat in [("y_obs", Y), ("c_star", C), ("obs_distance", Dobs)]:
            before = len(failures)
            _check_square_matrix(name, mat, K, failures, p)
            shape_bad = shape_bad or len(failures) > before
        if f2r is not None and F is not None and K is not None:
            if f2r.shape != (F, K):
                shape_bad = True
                failures.append(f"future_to_root_weight shape mismatch in {p.name}: {tuple(f2r.shape)} != {(F, K)}")
            else:
                col_sums = f2r.sum(axis=0)
                if rv is not None:
                    bad_valid_cols = np.where(rv & (np.abs(col_sums - 1.0) > 1e-3))[0]
                    bad_invalid_cols = np.where((~rv) & (np.abs(col_sums) > 1e-3))[0]
                    if bad_valid_cols.size or bad_invalid_cols.size:
                        warnings.append(f"future_to_root_weight column normalization suspicious in {p.name}")
        shape_failure_samples += int(shape_bad or len(failures) > sample_failures_before)

        if Y is not None and K is not None and Y.ndim == 2 and Y.shape == (K, K):
            if not np.allclose(Y, Y.T, atol=1e-4):
                failures.append(f"Y_obs not symmetric in {p.name}")
            if not np.allclose(np.diag(Y), 1.0, atol=1e-4):
                failures.append(f"Y_obs diagonal not 1 in {p.name}")
            off = _offdiag_values(Y, rv)
            if off.size:
                off_y_vals.extend(off.tolist())
        if C is not None and K is not None and C.ndim == 2 and C.shape == (K, K):
            if np.nanmin(C) < -1e-5 or np.nanmax(C) > 1.0 + 1e-5:
                failures.append(f"C_star outside [0,1] in {p.name}")
            if not np.allclose(C, C.T, atol=1e-4):
                failures.append(f"C_star not symmetric in {p.name}")
            if not np.allclose(np.diag(C), 1.0, atol=1e-4):
                failures.append(f"C_star diagonal not 1 in {p.name}")
            off = _offdiag_values(C, rv)
            if off.size:
                off_c_vals.extend(off.tolist())
        if Dobs is not None and K is not None and Dobs.ndim == 2 and Dobs.shape == (K, K):
            if np.nanmin(Dobs) < -1e-6:
                failures.append(f"obs_distance has negative values in {p.name}")
            off = _offdiag_values(Dobs, rv)
            if off.size:
                obs_distance_vals.extend(off.tolist())

        if M is not None and Y is not None:
            alias_pairs, incompatible_pairs, same_pairs = _alias_incompatibility(M, Y, rv, option_valid.astype(bool) if option_valid is not None else None)
            alias_pair_vals.append(alias_pairs)
            incompatible_alias_vals.append(incompatible_pairs)
            same_alias_vals.append(same_pairs)
            best = _best_options(M, option_valid.astype(bool) if option_valid is not None else None)
            if best.size:
                active = rv if rv is not None and rv.size == best.size else np.ones_like(best, dtype=bool)
                best_option_diversity_vals.append(float(len(set(best[active].tolist()))))

        disp = _safe_array(d, "within_root_obs_dispersion", float)
        if disp is not None and disp.size:
            active = rv if rv is not None and rv.size == disp.size else np.ones(disp.size, dtype=bool)
            within_root_disp_vals.extend(disp.reshape(-1)[active].tolist())

        r_orc = _scalar(d.get("r_orc_star", 0.0))
        r_dep = _scalar(d.get("r_dep_star", 0.0))
        gap = _scalar(d.get("oracle_gap_star", r_orc - r_dep))
        is_art = bool(_int_scalar(d.get("i_art_star", 0), 0))
        r_orc_vals.append(r_orc)
        r_dep_vals.append(r_dep)
        odg_vals.append(gap)
        odg_pos_vals.append(max(0.0, gap))
        artifact_count += int(is_art)
        oracle_recoverable_count += int(r_orc >= 0.0)
        negative_deployable_count += int(r_dep < 0.0)
        if is_art:
            odg_art_vals.append(gap)
            artifact_by_macro[macro] += 1
        else:
            nonartifact_by_macro[macro] += 1

        if M is not None and root_probs is not None and C is not None and M.ndim == 2 and C.ndim == 2 and K is not None and C.shape == (K, K):
            try:
                res = oc_mero(
                    M,
                    root_probs,
                    C,
                    alpha=alpha,
                    beta=beta,
                    option_valid=option_valid.astype(bool) if option_valid is not None else None,
                    root_valid=root_valid.astype(bool) if root_valid is not None else None,
                    use_lcvar=use_lcvar,
                    use_obs_kernel=use_obs_kernel,
                    top_m=top_m,
                )
                ocmero_checked += 1
                ocmero_max_abs_error = max(
                    ocmero_max_abs_error,
                    abs(float(res.r_orc) - r_orc),
                    abs(float(res.r_dep) - r_dep),
                    abs(float(res.gap) - gap),
                )
            except Exception as e:
                warnings.append(f"OC-MERO recompute failed in {p.name}: {e}")

        sources = [str(x) for x in np.asarray(d.get("future_sources", []), dtype=str).reshape(-1)]
        for s in sources:
            source_counts[s] += 1
        if required_sources.issubset(set(sources)):
            source_complete_samples += 1
        else:
            incomplete_source_samples += 1

        metas = _json_field(d, "future_metadata", [])
        if isinstance(metas, list):
            for m in metas:
                if not isinstance(m, dict):
                    continue
                targeted_type = str(m.get("targeted_type", ""))
                if targeted_type:
                    targeted_type_counts[targeted_type] += 1
                if m.get("hidden_intent"):
                    hidden_intent_counts[str(m.get("hidden_intent"))] += 1
                if m.get("hidden_emergence", False):
                    hidden_emergence_count += 1
                    if m.get("from_unknown_mask", False):
                        hidden_from_unknown_count += 1
                    if m.get("hidden_invalid_spawn", False):
                        hidden_invalid_spawn_count += 1
                    if m.get("spawn_in_visible_free", False):
                        hidden_visible_free_spawn_count += 1
                    if "hidden_start_step" in m and "prefix_steps" in m:
                        hidden_start_checked += 1
                        if int(m.get("hidden_start_step", -1)) < int(m.get("prefix_steps", 0)) + hidden_delay:
                            hidden_start_violation_count += 1
                total_metadata_futures += 1
                if m.get("runtime_backend"):
                    runtime_backend_counts[str(m.get("runtime_backend"))] += 1
                if m.get("waymax_teacher_backend"):
                    teacher_backend_counts[str(m.get("waymax_teacher_backend"))] += 1
                if bool(m.get("margin_override_applied", False)):
                    margin_override_future_count += 1
                if bool(m.get("waymax_runtime", False)):
                    waymax_runtime_futures += 1
                if m.get("plausibility_passed") is False:
                    plausibility_failed_futures += 1
                    failures_list = m.get("plausibility_failures", [])
                    if isinstance(failures_list, list):
                        for reason in failures_list:
                            plausibility_failure_counts[str(reason)] += 1
                    else:
                        plausibility_failure_counts[str(failures_list)] += 1
                if m.get("artifact_pair_key") and m.get("artifact_branch"):
                    artifact_pair_branches[str(m.get("artifact_pair_key"))].add(str(m.get("artifact_branch")))

        regimes = _json_field(d, "regime_label", {})
        if isinstance(regimes, dict):
            for k, v in regimes.items():
                if v:
                    regime_counts[str(k)] += 1
                    if is_nominal:
                        nominal_regime_counts[str(k)] += 1

        prefix_diag = _json_field(d, "prefix_diagnostics", {})
        if isinstance(prefix_diag, dict):
            for reason in prefix_diag.get("time_sampling_reasons", []) or []:
                time_reason_counts[str(reason)] += 1

        diagnostics = _json_field(d, "diagnostics", {})
        if isinstance(diagnostics, dict):
            for reason in diagnostics.get("time_sampling_reasons", []) or []:
                time_reason_counts[str(reason)] += 1
            if "unknown_ratio_in_corridor" in diagnostics:
                unknown_ratio_vals.append(float(diagnostics.get("unknown_ratio_in_corridor", 0.0)))
            if bool(diagnostics.get("complete_artifact_pair", False)):
                complete_artifact_pair_sample_count += 1
            rc = diagnostics.get("root_clustering", {})
            if isinstance(rc, dict) and rc.get("scale_source"):
                root_cluster_meta_counts[str(rc.get("scale_source"))] += 1

        modes = [str(x) for x in np.asarray(d.get("recovery_modes", []), dtype=str).reshape(-1)]
        # Count modes with a prefix to keep them distinct from prefix macros.
        for m in modes:
            if m:
                pass

    leakage = sorted([s for s, splits in split_by_scene.items() if len(splits) > 1])
    if leakage:
        failures.append(f"scenario split leakage: {leakage[:8]}")

    candidate_counts = [len(v) for v in candidate_by_scene_time.values()]
    nominal_counts = [int(nominal_by_scene_time.get(k, 0)) for k in candidate_by_scene_time]
    if candidate_counts and min(candidate_counts) < 2:
        failures.append("candidate_count_per_scene_time.min < 2")
    if any(n != 1 for n in nominal_counts):
        failures.append("each scene-time group should contain exactly one nominal candidate")

    complete_artifact_pairs = sum(1 for branches in artifact_pair_branches.values() if {"yield", "accelerate"}.issubset(branches))
    partial_artifact_pairs = sum(1 for branches in artifact_pair_branches.values() if branches and not {"yield", "accelerate"}.issubset(branches))

    num = len(paths)
    artifact_fraction = artifact_count / max(num, 1)
    oracle_recoverable_fraction = oracle_recoverable_count / max(num, 1)
    negative_deployable_fraction = negative_deployable_count / max(num, 1)
    source_complete_fraction = source_complete_samples / max(num, 1)
    waymax_runtime_fraction = waymax_runtime_futures / max(total_metadata_futures, 1)
    alias_total = float(np.sum(alias_pair_vals)) if alias_pair_vals else 0.0
    incompatible_total = float(np.sum(incompatible_alias_vals)) if incompatible_alias_vals else 0.0
    incompatible_alias_fraction = incompatible_total / max(alias_total, 1.0)
    mean_off_y = float(np.mean(off_y_vals)) if off_y_vals else 0.0
    mean_off_c = float(np.mean(off_c_vals)) if off_c_vals else 0.0
    mean_odg_pos = float(np.mean(odg_pos_vals)) if odg_pos_vals else 0.0

    if hidden_emergence_count > hidden_from_unknown_count:
        failures.append("hidden_emergence_count > hidden_from_unknown_count")
    if hidden_invalid_spawn_count > 0:
        failures.append("hidden_invalid_spawn_count > 0")
    if hidden_visible_free_spawn_count > 0:
        failures.append("hidden futures spawned in visible free space")
    if hidden_start_violation_count > 0:
        failures.append("hidden_start_step violates prefix + delay constraint")
    if plausibility_failed_futures > 0:
        warnings.append("some future metadata reports plausibility_passed=false")
    if num > 0 and (mean_off_y <= 0.02 or mean_off_y >= 0.98) and not nominal_regime_dataset:
        failures.append("mean_offdiag_y_obs near 0 or near 1 for almost all samples")
    if negative_deployable_fraction == 0.0 and num > 0:
        (warnings if nominal_regime_dataset else failures).append("negative_deployable_fraction == 0")
    if artifact_fraction == 0.0 and num > 0:
        warnings.append("artifact_fraction == 0; FRA/anti-oracle claims will not be stress-tested")
    if mean_odg_pos <= 1e-6 and num > 0:
        warnings.append("odg_pos_mean <= small_threshold; oracle/deployability gap may be absent")
    if alias_total == 0 and num > 0:
        failures.append("no observation-equivalent root pairs detected")
    if incompatible_total == 0 and artifact_count > 0:
        warnings.append("artifact samples exist but no incompatible alias pairs were detected by m_star argmax")
    if incomplete_source_samples > 0:
        warnings.append(
            f"{incomplete_source_samples} samples miss required future sources: {sorted(required_sources)}"
        )
    if source_complete_fraction < 0.95 and num > 0 and not nominal_regime_dataset and expected_targeted != 0:
        warnings.append("less than 95% of samples contain replay/reactive/targeted futures")
    if complete_artifact_pairs == 0 and hidden_emergence_count > 0:
        warnings.append("no complete hidden yield/accelerate artifact pair found")
    if ocmero_checked > 0 and ocmero_max_abs_error > 1e-3:
        warnings.append(f"stored OC-MERO labels differ from recomputation; max_abs_error={ocmero_max_abs_error:.4g}")
    if sample_split_counts.get("calibration", 0) == 0 and num >= 20:
        warnings.append("calibration split is empty; calibrated gamma_rec cannot be estimated")
    if sample_split_counts.get("test", 0) == 0 and num >= 20:
        warnings.append("test split is empty; final held-out claims cannot be evaluated")
    if nominal_regime_dataset:
        warnings = [w for w in warnings if not (
            w.startswith("artifact_fraction == 0")
            or w.startswith("odg_pos_mean <=")
            or w.startswith("no complete hidden yield/accelerate artifact pair")
            or w.startswith("regime count == 0: oracle_artifact")
        )]
    warn_art_hi = float(quality_cfg.get("warn_if_artifact_fraction_above", 0.80))
    if num > 0 and artifact_fraction > warn_art_hi:
        warnings.append(f"artifact_fraction > {warn_art_hi:.2f}; dataset is stress-only and cannot support primary NUP/calibration claims")
    warn_scene_min = int(quality_cfg.get("warn_if_scene_count_below", 50))
    if num > 0 and len(split_by_scene) < warn_scene_min:
        warnings.append(f"num_scenes < {warn_scene_min}; use only as a smoke/stress dataset, not paper-scale evaluation")
    if len(macro_counts) < 5 and num > 0:
        warnings.append("candidate macro diversity is low; lane_shift/merge/pull_over/stabilize are absent or underrepresented")
    for regime in ["normal", "low_headroom", "occluded", "near_contact", "post_contact", "oracle_artifact"]:
        if regime_counts.get(regime, 0) == 0 and num > 0:
            warnings.append(f"regime count == 0: {regime}")
    if synthetic_scene_count == num and num > 0:
        warnings.append("all inspected samples look synthetic; run on real WOMD before making primary benchmark claims")
    if waymax_runtime_fraction == 0.0 and synthetic_scene_count < num and num > 0:
        warnings.append("no future metadata reports waymax_runtime=true; this is a WOMD-derived surrogate dataset, not a confirmed Waymax closed-loop rollout dataset")

    paper_support = {
        "supports_fra": bool(artifact_fraction > 0.0 and negative_deployable_fraction > 0.0 and source_complete_fraction > 0.0),
        "supports_odg": bool(mean_odg_pos > 1e-6),
        "supports_drs_labels": bool(option_count_vals and root_count_vals and negative_deployable_fraction > 0.0),
        "supports_observation_consistency": bool(alias_total > 0 and 0.02 < mean_off_y < 0.98),
        "supports_alias_incompatibility_cases": bool(incompatible_total > 0),
        "supports_calibration_protocol": bool(sample_split_counts.get("calibration", 0) > 0 and artifact_fraction < 0.95),
        "supports_heldout_test_protocol": bool(sample_split_counts.get("test", 0) > 0 and len(split_by_scene) >= 5),
        "supports_womd_primary_claim": bool(synthetic_scene_count < num and num > 0 and artifact_fraction < 0.80 and sample_split_counts.get("calibration", 0) > 0 and sample_split_counts.get("test", 0) > 0 and len(split_by_scene) >= 50),
        "supports_waymax_runtime_claim": bool(waymax_runtime_fraction >= 0.95 and synthetic_scene_count < num and num > 0),
    }

    result = {
        "dataset": str(dataset),
        "dataset_roots": [str(x) for x in _dataset_specs(dataset)],
        "num_samples": num,
        "num_scenes": len(split_by_scene),
        "num_scene_time_groups": len(candidate_by_scene_time),
        "split_counts": dict(sample_split_counts),
        "scene_counts_by_split": {k: len(v) for k, v in scene_split_counts.items()},
        "future_source_coverage": _counter_dict(source_counts),
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "schema": {
            "missing_field_samples": int(missing_field_samples),
            "missing_field_counts": _counter_dict(missing_field_counter),
            "finite_failure_samples": int(finite_failure_samples),
            "shape_failure_samples": int(shape_failure_samples),
            "ocmero_recomputed_samples": int(ocmero_checked),
            "ocmero_recompute_max_abs_error": float(ocmero_max_abs_error),
        },
        "splits": {
            "sample_counts": dict(sample_split_counts),
            "scene_counts": {k: len(v) for k, v in scene_split_counts.items()},
            "leakage_scenes": leakage[:20],
        },
        "candidate_prefixes": {
            "candidate_count_per_scene_time": _stats(candidate_counts),
            "nominal_count_per_scene_time": _stats(nominal_counts),
            "macro_counts": _counter_dict(macro_counts),
            "artifact_by_macro": _counter_dict(artifact_by_macro),
            "nonartifact_by_macro": _counter_dict(nonartifact_by_macro),
            "feasible_fraction": float(np.mean(feasible_vals)) if feasible_vals else 0.0,
            "utility": _stats(candidate_utility_vals),
            "hard_violation": _stats(hard_violation_vals),
            "harm_proxy": _stats(harm_proxy_vals),
            "time_sampling_reason_counts": _counter_dict(time_reason_counts),
        },
        "future_generation": {
            "future_count": _stats(future_count_vals),
            "future_prior_entropy": _stats(future_entropy_vals),
            "source_counts": _counter_dict(source_counts),
            "required_source_complete_fraction": float(source_complete_fraction),
            "targeted_type_counts": _counter_dict(targeted_type_counts),
            "hidden_intent_counts": _counter_dict(hidden_intent_counts),
            "hidden_emergence_count": int(hidden_emergence_count),
            "hidden_from_unknown_count": int(hidden_from_unknown_count),
            "hidden_invalid_spawn_count": int(hidden_invalid_spawn_count),
            "hidden_visible_free_spawn_count": int(hidden_visible_free_spawn_count),
            "hidden_start_checked": int(hidden_start_checked),
            "hidden_start_violation_count": int(hidden_start_violation_count),
            "plausibility_failed_future_count": int(plausibility_failed_futures),
            "runtime_backend_counts": _counter_dict(runtime_backend_counts),
            "teacher_backend_counts": _counter_dict(teacher_backend_counts),
            "margin_override_future_count": int(margin_override_future_count),
            "complete_artifact_pair_sample_count": int(complete_artifact_pair_sample_count),
            "waymax_runtime_future_count": int(waymax_runtime_futures),
            "waymax_runtime_fraction": float(waymax_runtime_fraction),
            "complete_artifact_pair_count": int(complete_artifact_pairs),
            "partial_artifact_pair_count": int(partial_artifact_pairs),
            "unknown_ratio_in_corridor": _stats(unknown_ratio_vals),
        },
        "roots_and_observation": {
            "root_count": _stats(root_count_vals),
            "valid_root_count": _stats(valid_root_count_vals),
            "root_prior_entropy": _stats(root_entropy_vals),
            "root_clustering_metadata_counts": _counter_dict(root_cluster_meta_counts),
            "mean_offdiag_y_obs": mean_off_y,
            "mean_offdiag_c_star": mean_off_c,
            "offdiag_y_obs": _stats(off_y_vals),
            "offdiag_c_star": _stats(off_c_vals),
            "obs_distance_offdiag": _stats(obs_distance_vals),
            "within_root_obs_dispersion": _stats(within_root_disp_vals),
            "alias_pair_count_per_sample": _stats(alias_pair_vals),
            "incompatible_alias_pair_count_per_sample": _stats(incompatible_alias_vals),
            "same_best_option_alias_pair_count_per_sample": _stats(same_alias_vals),
            "incompatible_alias_pair_fraction": float(incompatible_alias_fraction),
            "best_option_diversity_per_sample": _stats(best_option_diversity_vals),
        },
        "recovery_labels": {
            "option_count": _stats(option_count_vals),
            "artifact_count": int(artifact_count),
            "artifact_fraction": float(artifact_fraction),
            "oracle_recoverable_fraction": float(oracle_recoverable_fraction),
            "negative_deployable_fraction": float(negative_deployable_fraction),
            "r_orc_star": _stats(r_orc_vals),
            "r_dep_star": _stats(r_dep_vals),
            "oracle_gap_star": _stats(odg_vals),
            "oracle_gap_positive_part": _stats(odg_pos_vals),
            "oracle_gap_artifact_samples": _stats(odg_art_vals),
        },
        "regimes": {
            "counts": _counter_dict(regime_counts),
            "sample_fractions": {
                str(k): float(v) / max(num, 1)
                for k, v in sorted(regime_counts.items())
            },
            "nominal_counts": _counter_dict(nominal_regime_counts),
            "nominal_sample_count": int(nominal_sample_count),
            "nominal_fractions": {
                str(k): float(v) / max(nominal_sample_count, 1)
                for k, v in sorted(nominal_regime_counts.items())
            },
            "fra_relevant_regime_count": int(regime_counts.get("oracle_artifact", 0) + regime_counts.get("occluded", 0) + regime_counts.get("low_headroom", 0)),
        },
        "dataset_contract": {
            "nominal_regime_dataset": bool(nominal_regime_dataset),
            "legacy_safe_name_fallback": bool(contract_meta["legacy_safe_name_fallback"]),
            "summary_paths": list(contract_meta["summary_paths"]),
            "require_nominal_regimes": quality_cfg.get("require_nominal_regimes", []),
            "forbid_nominal_regimes": quality_cfg.get("forbid_nominal_regimes", []),
            "forbid_any_regimes": quality_cfg.get("forbid_any_regimes", []),
        },
        "paper_support": paper_support,
    }
    if output:
        write_json(result, output)
    return result
