from __future__ import annotations

import numpy as np

from cowp.core.constants import NaturalSource, PriorityRelation, ObjectType
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.label.burden import adaptive_beta, compute_burden
from cowp.label.priority import determine_priority, priority_preserved
from cowp.label.trajectory_primitives import constant_accel_trajectory, repair_planar_kinematics, resample_logged


def _normalize_weights(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(weights, dtype=np.float32)
    total = float(np.sum(weights[valid]))
    if total > 0:
        out[valid] = weights[valid] / total
    return out


def _traj_distance(a: np.ndarray, b: np.ndarray) -> float:
    T = min(len(a), len(b))
    if T == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:T, :2] - b[:T, :2], axis=-1)))


def _route_geometry_timed_trajectory(
    logged: np.ndarray,
    current: np.ndarray,
    horizon: int,
    dt: float,
    *,
    accel: float = 0.0,
    speed_offset: float = 0.0,
) -> np.ndarray:
    """Retiming proxy that preserves route geometry while removing logged timing.

    Offline COWP labels are allowed to use the logged future as supervision, but
    an ego-neutral root should not simply replay a potentially coerced *timing*
    profile.  Straight constant-acceleration fallbacks also fail systematically
    on curved lanes.  This proxy keeps only the observed path geometry, then
    traverses it with a current-state, constant-acceleration timing law.  It is
    therefore useful as a map-compliant neutral pseudo-target without treating
    observed yielding timing as natural behavior.

    The helper is used only in label construction; inference does not receive
    future logged geometry.
    """
    logged = np.asarray(logged, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    H = int(horizon)
    dt = max(float(dt), 1.0e-3)
    if H <= 0 or logged.ndim != 2 or logged.shape[0] == 0 or logged.shape[1] < 7 or cur.size < 7:
        return constant_accel_trajectory(cur, H, dt, accel=float(accel), speed_offset=float(speed_offset))

    path = np.concatenate([cur[None, :2], logged[:, :2]], axis=0).astype(np.float32)
    finite = np.all(np.isfinite(path), axis=1)
    path = path[finite]
    if len(path) < 2:
        return constant_accel_trajectory(cur, H, dt, accel=float(accel), speed_offset=float(speed_offset))

    # Remove zero-length segments introduced by invalid-future hold padding.
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1.0e-3
    path = path[keep]
    if len(path) < 2:
        return constant_accel_trajectory(cur, H, dt, accel=float(accel), speed_offset=float(speed_offset))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float64)
    if float(arc[-1]) < 1.0e-3:
        return constant_accel_trajectory(cur, H, dt, accel=float(accel), speed_offset=float(speed_offset))

    v = max(float(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5])) + float(speed_offset), 0.0)
    s_query = np.zeros(H, dtype=np.float64)
    speed = np.zeros(H, dtype=np.float64)
    s = 0.0
    for k in range(H):
        v = max(0.0, v + float(accel) * dt)
        s += v * dt
        s_query[k] = s
        speed[k] = v

    length = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    width = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)
    out = np.zeros((H, 7), dtype=np.float32)
    last_dir = path[-1] - path[-2]
    last_norm = max(float(np.linalg.norm(last_dir)), 1.0e-6)
    last_dir = last_dir / last_norm
    for k, sq in enumerate(s_query.tolist()):
        if sq <= float(arc[-1]):
            j = int(np.clip(np.searchsorted(arc, sq, side="right") - 1, 0, len(path) - 2))
            span = max(float(arc[j + 1] - arc[j]), 1.0e-6)
            frac = float(np.clip((sq - float(arc[j])) / span, 0.0, 1.0))
            pos = path[j] + frac * (path[j + 1] - path[j])
            direction = path[j + 1] - path[j]
            norm = max(float(np.linalg.norm(direction)), 1.0e-6)
            direction = direction / norm
        else:
            pos = path[-1] + last_dir * float(sq - float(arc[-1]))
            direction = last_dir
        yaw = float(np.arctan2(direction[1], direction[0]))
        out[k] = [
            float(pos[0]), float(pos[1]), yaw,
            float(direction[0] * speed[k]), float(direction[1] * speed[k]),
            length, width,
        ]
    return repair_planar_kinematics(out, current=cur, dt=dt)


def _ordered_observational_specs(nat_cfg: dict) -> list[tuple[float, float, float]]:
    """Order the observational bank from identity outward.

    v16.8.9 truncated the nested cartesian product at ``max_obs_samples``.
    With the default list order, all eight retained samples used speed_scale
    0.85 and the exact (1, 0, 0) observational root was never generated.  That
    is especially harmful on curved lanes because the only route-following roots
    may be the observational branch.  Sorting by perturbation magnitude keeps
    the same configured candidate family while spending the finite budget on
    the most semantically useful roots first.
    """
    speed_values = [float(x) for x in nat_cfg.get("obs_speed_scale", [0.85, 0.95, 1.0, 1.05, 1.15])]
    shift_values = [float(x) for x in nat_cfg.get("obs_time_shift_s", [-0.5, 0.0, 0.5])]
    lat_values = [float(x) for x in nat_cfg.get("obs_lateral_offset_m", [-0.3, 0.0, 0.3])]
    specs = [(ss, shift_s, lat) for ss in speed_values for shift_s in shift_values for lat in lat_values]
    speed_scale = max([abs(float(np.log(max(x, 1.0e-6)))) for x in speed_values] + [1.0e-6])
    shift_scale = max([abs(x) for x in shift_values] + [1.0e-6])
    lat_scale = max([abs(x) for x in lat_values] + [1.0e-6])

    def rank(x: tuple[float, float, float]) -> tuple[float, float, float, float, float, float, float]:
        ds = abs(float(np.log(max(x[0], 1.0e-6)))) / speed_scale
        dt = abs(x[1]) / shift_scale
        dl = abs(x[2]) / lat_scale
        return (ds + dt + dl, max(ds, dt, dl), ds, dt, dl, x[0], x[1])

    specs.sort(key=rank)
    return specs


def _lane_point_cloud(scene: ScenarioData) -> np.ndarray:
    chunks = [np.asarray(lane.xy, dtype=np.float32) for lane in scene.map_data.lanes.values() if len(lane.xy)]
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0)[:, :2]


def _trajectory_map_compliance(
    tr: np.ndarray, lane_points: np.ndarray, object_type: int, nat_cfg: dict
) -> tuple[bool, float, bool]:
    """Best-effort drivable-corridor check used during label construction.

    It intentionally returns ``verified=False`` when no lane geometry exists, so
    missing map detail is not silently reported as a successful map check.
    """
    if not bool(nat_cfg.get("map_filter_enabled", True)):
        return True, -1.0, False
    if lane_points.size == 0:
        return (not bool(nat_cfg.get("map_filter_require_available", False))), -1.0, False
    stride = max(1, int(nat_cfg.get("map_filter_stride", 4)))
    xy = np.asarray(tr, dtype=np.float32)[::stride, :2]
    if xy.size == 0 or not np.all(np.isfinite(xy)):
        return False, float("inf"), True
    # Lane points in WOMD are dense enough for this sampled point-cloud distance
    # to be a conservative and much faster proxy than per-step polyline search.
    d2 = ((xy[:, None, :] - lane_points[None, :, :]) ** 2).sum(axis=-1)
    d = np.sqrt(np.min(d2, axis=1))
    if int(object_type) == int(ObjectType.VEHICLE):
        threshold = float(nat_cfg.get("map_max_distance_vehicle_m", 5.0))
    else:
        threshold = float(nat_cfg.get("map_max_distance_vru_m", 8.0))
    min_fraction = float(nat_cfg.get("map_min_compliant_fraction", 0.80))
    hard_max = float(nat_cfg.get("map_hard_max_distance_m", 12.0))
    ok = float(np.mean(d <= threshold)) >= min_fraction and float(np.max(d)) <= hard_max
    return bool(ok), float(np.max(d)), True


def _observed_yield_contamination(
    scene: ScenarioData, agent_index: int, logged: np.ndarray, ego_neutral_traj: np.ndarray,
    horizon: int, dt: float, nat_cfg: dict,
) -> float:
    """Estimate whether the logged future is likely an ego-induced yield.

    The score is used only to downweight/drop observational pseudo-targets.  It is
    never an online input.  Evidence combines interaction proximity, speed/progress
    loss relative to a current-state continuation, and clearance gained when ego
    is replaced by the neutral intervention.
    """
    if not bool(nat_cfg.get("obs_decontamination_enabled", True)):
        return 0.0
    current = scene.states[int(agent_index), scene.current_time_index]
    baseline = constant_accel_trajectory(current, horizon, dt, accel=0.0)
    ego_logged = future_states_to_traj7(
        scene.states[scene.sdc_track_index, scene.current_time_index + 1 : scene.current_time_index + 1 + horizon],
        horizon, current_state=scene.states[scene.sdc_track_index, scene.current_time_index],
    )
    T = min(len(logged), len(baseline), len(ego_logged), len(ego_neutral_traj))
    if T <= 0:
        return 0.0
    speed0 = max(float(np.linalg.norm(current[3:5])), 1.0)
    logged_speed = np.linalg.norm(logged[:T, 3:5], axis=-1)
    early = logged_speed[: min(T, max(5, int(round(3.0 / max(dt, 1e-3)))))]
    decel_evidence = np.clip((speed0 - float(np.percentile(early, 20))) / max(speed0, 2.0), 0.0, 1.0)
    base_progress = float(np.linalg.norm(baseline[T - 1, :2] - current[:2]))
    logged_progress = float(np.linalg.norm(logged[T - 1, :2] - current[:2]))
    progress_loss = np.clip((base_progress - logged_progress) / max(base_progress, 5.0), 0.0, 1.0)
    d_logged = np.linalg.norm(logged[:T, :2] - ego_logged[:T, :2], axis=-1)
    d_neutral = np.linalg.norm(logged[:T, :2] - ego_neutral_traj[:T, :2], axis=-1)
    min_logged = float(np.min(d_logged))
    min_neutral = float(np.min(d_neutral))
    radius = max(float(nat_cfg.get("obs_pressure_radius_m", 12.0)), 1e-3)
    proximity = float(np.exp(-0.5 * (min_logged / radius) ** 2))
    relief = float(np.clip((min_neutral - min_logged) / max(float(nat_cfg.get("obs_neutral_relief_scale_m", 6.0)), 1e-3), 0.0, 1.0))
    score = proximity * (0.45 * decel_evidence + 0.35 * progress_loss + 0.20 * relief)
    return float(np.clip(score, 0.0, 1.0))


def generate_natural_alternatives(scene: ScenarioData, critical: dict[str, np.ndarray], ego_neutral_traj: np.ndarray, cfg: dict, ablation: dict | None = None) -> dict[str, np.ndarray]:
    ablation = ablation or {}
    use_obs = bool(ablation.get("use_obs_branch", True))
    use_neu = bool(ablation.get("use_neutral_branch", True))
    use_prio = bool(ablation.get("use_priority_branch", True))
    limits = cfg.get("limits", {})
    nat_cfg = cfg.get("natural", {})
    A = int(limits.get("max_critical_agents", 8))
    M = int(limits.get("max_natural_alternatives", 24))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    traj = np.zeros((A, M, H, 7), dtype=np.float32)
    valid = np.zeros((A, M), dtype=bool)
    source = np.full((A, M), int(NaturalSource.PAD), dtype=np.int32)
    burden_neutral = np.zeros((A, M), dtype=np.float32)
    priority_ok = np.zeros((A, M), dtype=bool)
    weights = np.zeros((A, M), dtype=np.float32)
    beta = np.zeros(A, dtype=np.float32)
    obs_contamination = np.zeros((A, M), dtype=np.float32)
    map_compliant = np.zeros((A, M), dtype=bool)
    map_distance_max = np.full((A, M), -1.0, dtype=np.float32)
    map_verified = np.zeros((A, M), dtype=bool)
    lane_points = _lane_point_cloud(scene)

    cur = scene.current_time_index
    for a in range(A):
        if not critical["valid"][a]:
            continue
        idx = int(critical["track_index"][a])
        object_type = int(scene.object_type[idx])
        fut_states = scene.states[idx, cur + 1 : cur + 1 + H, :]
        fut_mask = fut_states[:, 10] > 0.5
        if np.any(fut_mask):
            logged = future_states_to_traj7(fut_states, H, current_state=scene.states[idx, cur])
        else:
            logged = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=0.0)
        rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))
        if rho == PriorityRelation.UNKNOWN:
            rho = determine_priority(scene, idx, ego_neutral_traj, logged, cfg)
        scene_current = scene.states[:, cur, :] if scene.states.ndim == 3 else None
        beta[a] = adaptive_beta(scene_current, object_type, rho, cfg, use_adaptive=True, ego_index=scene.sdc_track_index)
        candidates: list[tuple[np.ndarray, NaturalSource, float, float]] = []
        obs_contam = _observed_yield_contamination(scene, idx, logged, ego_neutral_traj, H, dt, nat_cfg)
        if use_obs:
            max_obs = int(nat_cfg.get("max_obs_samples", 8))
            for ss, shift_s, lat in _ordered_observational_specs(nat_cfg)[:max_obs]:
                tr = resample_logged(
                    logged, H, time_shift_steps=int(round(float(shift_s) / dt)),
                    speed_scale=float(ss), lateral_offset=float(lat),
                    current=scene.states[idx, cur], dt=dt,
                )
                candidates.append((tr, NaturalSource.OBS, float(nat_cfg.get("source_weight_obs", 1.0)), obs_contam))
        if use_neu:
            count = 0
            for acc in nat_cfg.get("neutral_acc_values_mps2", [-1.0, -0.5, 0.0, 0.5, 1.0]):
                for voff in nat_cfg.get("neutral_target_speed_offsets_mps", [-2.0, 0.0, 2.0]):
                    if count >= int(nat_cfg.get("max_neutral_samples", 8)):
                        break
                    tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=float(acc), speed_offset=float(voff))
                    candidates.append((tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0))
                    count += 1
                if count >= int(nat_cfg.get("max_neutral_samples", 8)):
                    break
        if use_prio:
            count = 0
            for acc in nat_cfg.get("prio_acc_values_mps2", [-0.5, 0.0, 0.5]):
                if count >= int(nat_cfg.get("prio_max_samples", 8)):
                    break
                tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=float(acc))
                candidates.append((tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2)), 0.0))
                count += 1
        # Fallback if ablations remove all branches.
        if not candidates:
            candidates.append((logged, NaturalSource.OBS, 1.0, obs_contam))
        kept = 0
        raw_w = np.zeros(M, dtype=np.float32)
        for tr, src, src_weight, contamination in candidates:
            if kept >= M:
                break
            b_total, _ = compute_burden(tr, ego_neutral_traj, cfg, object_type, natural_ref=logged, rho=rho)
            pr_ok = priority_preserved(tr, logged, rho, cfg)
            dyn_ok = bool(np.all(np.isfinite(tr)))
            map_ok, map_dist, map_was_verified = _trajectory_map_compliance(tr, lane_points, object_type, nat_cfg)
            plausible = b_total <= beta[a] + 0.1
            contamination_ok = not (
                src == NaturalSource.OBS
                and contamination >= float(nat_cfg.get("obs_drop_contamination_above", 0.90))
            )
            keep = dyn_ok and map_ok and plausible and contamination_ok and (pr_ok or rho != PriorityRelation.AGENT_PRIORITY)
            if keep:
                traj[a, kept] = tr
                valid[a, kept] = True
                source[a, kept] = int(src)
                burden_neutral[a, kept] = float(b_total)
                priority_ok[a, kept] = bool(pr_ok)
                obs_contamination[a, kept] = float(contamination)
                map_compliant[a, kept] = bool(map_ok)
                map_distance_max[a, kept] = float(map_dist)
                map_verified[a, kept] = bool(map_was_verified)
                dist = _traj_distance(tr, logged)
                decontam_factor = max(
                    float(nat_cfg.get("obs_weight_floor", 0.05)),
                    1.0 - float(nat_cfg.get("obs_contamination_weight", 0.90)) * float(contamination),
                ) if src == NaturalSource.OBS else 1.0
                raw_w[kept] = float(src_weight) * decontam_factor * np.exp(-dist / max(float(nat_cfg.get("sigma_traj_m", 15.0)), 1e-6)) * np.exp(-b_total / max(float(nat_cfg.get("sigma_b", 0.5)), 1e-6))
                kept += 1
        if kept < int(nat_cfg.get("min_natural_alternatives", 6)):
            # First repair curved-lane support with an ego-neutral timing proxy
            # that follows logged route geometry.  Only the path geometry is
            # reused; candidate-specific logged timing/yielding is discarded.
            # This prevents the old straight-line fallback from being rejected
            # wholesale by the map filter on turns and curved lanes.
            fallback_acc = nat_cfg.get("geometry_neutral_acc_values_mps2", [0.0, -0.5, 0.5, -1.0, 1.0])
            for acc in fallback_acc:
                if kept >= M:
                    break
                tr = _route_geometry_timed_trajectory(
                    logged, scene.states[idx, cur], H, dt, accel=float(acc),
                )
                b_total, _ = compute_burden(tr, ego_neutral_traj, cfg, object_type, natural_ref=logged, rho=rho)
                pr_ok = priority_preserved(tr, logged, rho, cfg)
                map_ok, map_dist, map_was_verified = _trajectory_map_compliance(tr, lane_points, object_type, nat_cfg)
                plausible = b_total <= beta[a] + 0.1
                if not (np.all(np.isfinite(tr)) and map_ok and plausible and (pr_ok or rho != PriorityRelation.AGENT_PRIORITY)):
                    continue
                traj[a, kept] = tr
                valid[a, kept] = True
                source[a, kept] = int(NaturalSource.NEU)
                burden_neutral[a, kept] = float(b_total)
                priority_ok[a, kept] = bool(pr_ok)
                map_compliant[a, kept] = bool(map_ok)
                map_distance_max[a, kept] = float(map_dist)
                map_verified[a, kept] = bool(map_was_verified)
                raw_w[kept] = np.exp(-b_total / max(float(nat_cfg.get("sigma_b", 0.5)), 1e-6))
                kept += 1

        if kept < int(nat_cfg.get("min_natural_alternatives", 6)):
            # Last-resort straight primitives remain useful on map fragments with
            # very short/no usable logged future, but unlike v16.8.9 they must
            # satisfy the same low-burden plausibility contract as primary roots.
            for acc in nat_cfg.get("straight_fallback_acc_values_mps2", [-0.5, 0.0, 0.5, 1.0, -1.0, 1.5, -1.5]):
                if kept >= M:
                    break
                tr = constant_accel_trajectory(scene.states[idx, cur], H, dt, accel=float(acc))
                b_total, _ = compute_burden(tr, ego_neutral_traj, cfg, object_type, natural_ref=logged, rho=rho)
                pr_ok = priority_preserved(tr, logged, rho, cfg)
                map_ok, map_dist, map_was_verified = _trajectory_map_compliance(tr, lane_points, object_type, nat_cfg)
                plausible = b_total <= beta[a] + 0.1
                if not (np.all(np.isfinite(tr)) and map_ok and plausible and (pr_ok or rho != PriorityRelation.AGENT_PRIORITY)):
                    continue
                traj[a, kept] = tr
                valid[a, kept] = True
                source[a, kept] = int(NaturalSource.NEU)
                burden_neutral[a, kept] = float(b_total)
                priority_ok[a, kept] = bool(pr_ok)
                map_compliant[a, kept] = bool(map_ok)
                map_distance_max[a, kept] = float(map_dist)
                map_verified[a, kept] = bool(map_was_verified)
                raw_w[kept] = np.exp(-b_total / max(float(nat_cfg.get("sigma_b", 0.5)), 1e-6))
                kept += 1
        weights[a] = _normalize_weights(raw_w, valid[a])
    return {
        "traj": traj,
        "valid": valid,
        "source": source,
        "burden_neutral": burden_neutral,
        "priority_preserved": priority_ok,
        "weight": weights,
        "beta": beta,
        "obs_contamination": obs_contamination,
        "map_compliant": map_compliant,
        "map_distance_max": map_distance_max,
        "map_verified": map_verified,
    }
