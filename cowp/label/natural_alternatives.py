from __future__ import annotations

import numpy as np

from cowp.core.constants import NaturalSource, PriorityRelation, ObjectType
from cowp.core.types import ScenarioData, future_states_to_traj7
from cowp.geometry.collision import unsafe_between
from cowp.geometry.map_projection import polyline_arc_length, project_state_to_lane
from cowp.label.burden import adaptive_beta, compute_burden
from cowp.label.priority import determine_priority, priority_preserved, priority_preservation_check
from cowp.label.trajectory_primitives import constant_accel_trajectory, repair_planar_kinematics, resample_logged, smooth_stop_trajectory


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


def _traj_distance_to_valid_future(traj: np.ndarray, future_states: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compare a candidate route only at factual WOMD-valid timestamps.

    Counting valid rows and then taking a prefix is incorrect when a track has an
    interior validity gap or appears after the first future sample.  Route choice
    is supervision-only, so use the raw validity mask without converting missing
    rows into hold states.
    """
    tr = np.asarray(traj, dtype=np.float32)
    fut = np.asarray(future_states, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    T = min(len(tr), len(fut), len(mask))
    if T <= 0:
        return float("inf")
    m = mask[:T] & np.all(np.isfinite(fut[:T, :2]), axis=-1) & np.all(np.isfinite(tr[:T, :2]), axis=-1)
    if not np.any(m):
        return float("inf")
    return float(np.mean(np.linalg.norm(tr[:T, :2][m] - fut[:T, :2][m], axis=-1)))


def _ramped_accel_schedule(target_accel: float, horizon: int, dt: float, nat_cfg: dict) -> np.ndarray:
    """Jerk-bounded acceleration schedule for natural/neutral pseudo-roots.

    Natural roots are later filtered by ``priority_preserved`` using a comfort
    jerk bound.  A mathematical constant-acceleration primitive has an impulse at
    activation; in discrete samples that made almost every non-zero fallback fail
    its own priority filter.  Ramp from zero acceleration instead.
    """
    H = max(int(horizon), 0)
    dt = max(float(dt), 1.0e-4)
    jerk = max(float(nat_cfg.get("natural_accel_ramp_jerk_mps3", 1.0)), 1.0e-3)
    target = float(target_accel)
    out = np.zeros(H, dtype=np.float32)
    a = 0.0
    da = jerk * dt
    for k in range(H):
        a += float(np.clip(target - a, -da, da))
        out[k] = a
    return out


def _jerk_bounded_straight_trajectory(
    current: np.ndarray, horizon: int, dt: float, nat_cfg: dict, *, accel: float = 0.0, speed_offset: float = 0.0
) -> np.ndarray:
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    H = int(horizon)
    dt = max(float(dt), 1.0e-4)
    if H <= 0:
        return np.zeros((0, 7), dtype=np.float32)
    heading = float(cur[6] if cur.size > 6 else 0.0)
    direction = np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
    pos = np.asarray(cur[:2], dtype=np.float32).copy()
    v = max(float(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5])) + float(speed_offset), 0.0)
    length = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    width = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)
    schedule = _ramped_accel_schedule(float(accel), H, dt, nat_cfg)
    out = np.zeros((H, 7), dtype=np.float32)
    for k, a in enumerate(schedule.tolist()):
        v = max(0.0, v + float(a) * dt)
        pos = pos + direction * (v * dt)
        out[k] = [pos[0], pos[1], heading, direction[0] * v, direction[1] * v, length, width]
    return repair_planar_kinematics(out, current=cur, dt=dt)


def _route_geometry_timed_trajectory(
    logged: np.ndarray,
    current: np.ndarray,
    horizon: int,
    dt: float,
    *,
    accel: float = 0.0,
    speed_offset: float = 0.0,
    nat_cfg: dict | None = None,
) -> np.ndarray:
    """Retiming proxy that preserves route geometry while removing logged timing.

    Offline COWP labels are allowed to use the logged future as supervision, but
    an ego-neutral root should not simply replay a potentially coerced *timing*
    profile.  Straight constant-acceleration fallbacks also fail systematically
    on curved lanes.  This proxy keeps only the observed path geometry, then
    traverses it with a current-state, jerk-bounded acceleration timing law.  It
    is therefore useful as a map-compliant neutral pseudo-target without treating
    observed yielding timing as natural behavior.

    The helper is used only in label construction; inference does not receive
    future logged geometry.
    """
    logged = np.asarray(logged, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    H = int(horizon)
    dt = max(float(dt), 1.0e-3)
    if H <= 0 or logged.ndim != 2 or logged.shape[0] == 0 or logged.shape[1] < 7 or cur.size < 7:
        return _jerk_bounded_straight_trajectory(
            cur, H, dt, nat_cfg or {}, accel=float(accel), speed_offset=float(speed_offset)
        )

    path = np.concatenate([cur[None, :2], logged[:, :2]], axis=0).astype(np.float32)
    finite = np.all(np.isfinite(path), axis=1)
    path = path[finite]
    if len(path) < 2:
        return _jerk_bounded_straight_trajectory(
            cur, H, dt, nat_cfg or {}, accel=float(accel), speed_offset=float(speed_offset)
        )

    # Remove zero-length segments introduced by invalid-future hold padding.
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1.0e-3
    path = path[keep]
    if len(path) < 2:
        return _jerk_bounded_straight_trajectory(
            cur, H, dt, nat_cfg or {}, accel=float(accel), speed_offset=float(speed_offset)
        )
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float64)
    if float(arc[-1]) < 1.0e-3:
        return constant_accel_trajectory(cur, H, dt, accel=float(accel), speed_offset=float(speed_offset))

    v = max(float(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5])) + float(speed_offset), 0.0)
    s_query = np.zeros(H, dtype=np.float64)
    speed = np.zeros(H, dtype=np.float64)
    s = 0.0
    accel_schedule = _ramped_accel_schedule(float(accel), H, dt, nat_cfg or {})
    for k, a_k in enumerate(accel_schedule.tolist()):
        v = max(0.0, v + float(a_k) * dt)
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




def _polyline_from_s(xy: np.ndarray, start_s: float) -> np.ndarray:
    """Trim a lane centreline at arc-length ``start_s`` and keep continuity."""
    pts = np.asarray(xy, dtype=np.float32)
    if pts.ndim != 2 or len(pts) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    pts = pts[:, :2]
    arc = polyline_arc_length(pts)
    if len(arc) < 2 or float(arc[-1]) <= 1.0e-4:
        return np.zeros((0, 2), dtype=np.float32)
    ss = float(np.clip(start_s, 0.0, float(arc[-1])))
    j = int(np.searchsorted(arc, ss, side="right") - 1)
    j = max(0, min(j, len(pts) - 2))
    denom = max(float(arc[j + 1] - arc[j]), 1.0e-6)
    u = float(np.clip((ss - float(arc[j])) / denom, 0.0, 1.0))
    start = pts[j] + u * (pts[j + 1] - pts[j])
    out = np.concatenate([start[None, :], pts[j + 1 :]], axis=0)
    keep = np.ones(len(out), dtype=bool)
    if len(out) > 1:
        keep[1:] = np.linalg.norm(np.diff(out, axis=0), axis=1) > 1.0e-3
    return out[keep].astype(np.float32)


def _append_lane_polyline(path: np.ndarray, lane_xy: np.ndarray, max_gap_m: float) -> np.ndarray | None:
    nxt = np.asarray(lane_xy, dtype=np.float32)
    if nxt.ndim != 2 or len(nxt) < 2:
        return None
    nxt = nxt[:, :2]
    # Choose the orientation that connects to the current route endpoint. WOMD
    # lane polylines are normally directed, but this guard makes the fallback
    # robust to locally reversed map fragments without changing lane topology.
    d0 = float(np.linalg.norm(path[-1] - nxt[0]))
    d1 = float(np.linalg.norm(path[-1] - nxt[-1]))
    if d1 + 1.0e-4 < d0:
        nxt = nxt[::-1].copy()
        d0 = d1
    if d0 > float(max_gap_m):
        return None
    if d0 < 1.0e-3:
        nxt = nxt[1:]
    if len(nxt) == 0:
        return path
    return np.concatenate([path, nxt], axis=0).astype(np.float32)


def _map_route_polylines(
    scene: ScenarioData,
    current: np.ndarray,
    required_length_m: float,
    nat_cfg: dict,
) -> list[np.ndarray]:
    """Build a small deterministic beam of legal lane-centre continuations.

    This fallback is deliberately map/topology based rather than a straight-line
    extrapolation.  It is used only for offline pseudo-label construction when a
    critical actor has short/curved logged support.  It never enters the online
    model input.
    """
    max_routes = max(int(nat_cfg.get("map_route_max_routes", 3)), 1)
    search_radius = float(nat_cfg.get("map_route_search_radius_m", 8.0))
    max_gap = float(nat_cfg.get("map_route_max_link_gap_m", 8.0))
    max_depth = max(int(nat_cfg.get("map_route_max_depth", 16)), 1)
    proj = project_state_to_lane(np.asarray(current, dtype=np.float32), scene.map_data, search_radius=search_radius)
    if int(proj.lane_id) < 0 or int(proj.lane_id) not in scene.map_data.lanes:
        return []
    start_lane = scene.map_data.lanes[int(proj.lane_id)]
    start_path = _polyline_from_s(start_lane.xy, float(proj.s))
    if len(start_path) < 2:
        return []

    def length(path: np.ndarray) -> float:
        return float(polyline_arc_length(path)[-1]) if len(path) >= 2 else 0.0

    # state = (path, final_lane_id, visited_lane_ids, continuity_cost)
    beam: list[tuple[np.ndarray, int, tuple[int, ...], float]] = [
        (start_path, int(proj.lane_id), (int(proj.lane_id),), 0.0)
    ]
    complete: list[tuple[np.ndarray, float]] = []
    for _depth in range(max_depth):
        nxt_beam: list[tuple[np.ndarray, int, tuple[int, ...], float]] = []
        for path, lane_id, visited, cost in beam:
            if length(path) >= float(required_length_m):
                complete.append((path, cost))
                continue
            lane = scene.map_data.lanes.get(int(lane_id))
            exits = [] if lane is None else [int(x) for x in lane.exit_lanes if int(x) in scene.map_data.lanes and int(x) not in visited]
            if not exits:
                complete.append((path, cost + max(0.0, float(required_length_m) - length(path))))
                continue
            base_dir = path[-1] - path[-2]
            base_h = float(np.arctan2(base_dir[1], base_dir[0]))
            for exit_id in exits:
                exit_lane = scene.map_data.lanes[exit_id]
                appended = _append_lane_polyline(path, exit_lane.xy, max_gap)
                if appended is None or len(appended) < 2:
                    continue
                p0, p1 = appended[len(path) - 1], appended[min(len(path), len(appended) - 1)]
                new_h = float(np.arctan2((p1 - p0)[1], (p1 - p0)[0])) if np.linalg.norm(p1 - p0) > 1.0e-4 else base_h
                dh = float(abs((new_h - base_h + np.pi) % (2.0 * np.pi) - np.pi))
                gap = float(np.linalg.norm(path[-1] - np.asarray(exit_lane.xy, dtype=np.float32)[0, :2]))
                nxt_beam.append((appended, exit_id, visited + (exit_id,), cost + 2.0 * dh + 0.1 * min(gap, max_gap)))
        if not nxt_beam:
            break
        nxt_beam.sort(key=lambda row: (row[3], -length(row[0]), row[1]))
        beam = nxt_beam[: max_routes * 3]
    complete.extend((p, c) for p, _lid, _vis, c in beam)
    # Prefer routes that actually cover the requested distance, then smooth/short
    # continuation cost. Keep geometrically distinct routes only.
    complete.sort(key=lambda row: (length(row[0]) + 1.0e-6 < float(required_length_m), row[1], -length(row[0])))
    out: list[np.ndarray] = []
    for path, _ in complete:
        if any(_traj_distance(
            _timed_polyline_trajectory(path, current, min(20, max(2, int(nat_cfg.get("map_route_compare_steps", 20)))), 0.1, accel=0.0, allow_short=True),
            _timed_polyline_trajectory(old, current, min(20, max(2, int(nat_cfg.get("map_route_compare_steps", 20)))), 0.1, accel=0.0, allow_short=True),
        ) < 0.5 for old in out):
            continue
        out.append(path.astype(np.float32))
        if len(out) >= max_routes:
            break
    return out


def _timed_polyline_trajectory(
    path_xy: np.ndarray,
    current: np.ndarray,
    horizon: int,
    dt: float,
    *,
    accel: float = 0.0,
    speed_offset: float = 0.0,
    allow_short: bool = False,
    nat_cfg: dict | None = None,
) -> np.ndarray | None:
    """Retiming of one map polyline under bounded longitudinal acceleration."""
    path = np.asarray(path_xy, dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32).reshape(-1)
    H = int(horizon)
    dt = max(float(dt), 1.0e-4)
    if path.ndim != 2 or len(path) < 2 or H <= 0:
        return None
    path = path[:, :2]
    arc = polyline_arc_length(path).astype(np.float64)
    if len(arc) < 2 or float(arc[-1]) < 1.0e-3:
        return None
    v = max(float(cur[5] if cur.size > 5 else np.linalg.norm(cur[3:5])) + float(speed_offset), 0.0)
    s_query = np.zeros(H, dtype=np.float64)
    speed = np.zeros(H, dtype=np.float64)
    s = 0.0
    accel_schedule = _ramped_accel_schedule(float(accel), H, dt, nat_cfg or {})
    for t, a_t in enumerate(accel_schedule.tolist()):
        v = max(0.0, v + float(a_t) * dt)
        s += v * dt
        s_query[t] = s
        speed[t] = v
    if (not allow_short) and float(np.max(s_query)) > float(arc[-1]) + 0.25:
        return None
    s_clip = np.minimum(s_query, float(arc[-1]))
    x = np.interp(s_clip, arc, path[:, 0])
    y = np.interp(s_clip, arc, path[:, 1])
    seg = np.diff(path, axis=0)
    seg_h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]).astype(np.float64))
    mid_arc = 0.5 * (arc[:-1] + arc[1:])
    yaw = np.interp(s_clip, mid_arc, seg_h, left=seg_h[0], right=seg_h[-1])
    # If a short comparison trajectory clips at route end, zero the speed there.
    if allow_short:
        speed = np.where(s_query <= float(arc[-1]) + 1.0e-6, speed, 0.0)
    length = float(cur[7] if cur.size > 7 and cur[7] > 0 else 4.8)
    width = float(cur[8] if cur.size > 8 and cur[8] > 0 else 1.9)
    out = np.zeros((H, 7), dtype=np.float32)
    out[:, 0] = x.astype(np.float32)
    out[:, 1] = y.astype(np.float32)
    out[:, 2] = ((yaw + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    out[:, 3] = (np.cos(yaw) * speed).astype(np.float32)
    out[:, 4] = (np.sin(yaw) * speed).astype(np.float32)
    out[:, 5] = length
    out[:, 6] = width
    return out



def _required_route_length(current: np.ndarray, horizon: int, dt: float, nat_cfg: dict, *, accel: float = 0.0, speed_offset: float = 0.0) -> float:
    T = float(horizon) * float(dt)
    v0 = max(float(current[5] if len(current) > 5 else np.linalg.norm(current[3:5])) + float(speed_offset), 0.0)
    return max(
        5.0,
        v0 * T + 0.5 * max(float(accel), 0.0) * T * T + float(nat_cfg.get("map_route_length_margin_m", 8.0)),
    )


def _retime_route_polylines(
    routes: list[np.ndarray],
    current: np.ndarray,
    horizon: int,
    dt: float,
    nat_cfg: dict,
    *,
    accel: float = 0.0,
    speed_offset: float = 0.0,
) -> list[np.ndarray]:
    """Retime a precomputed lane-topology route bank without re-running graph search."""
    out: list[np.ndarray] = []
    for route in routes:
        tr = _timed_polyline_trajectory(
            route,
            current,
            horizon,
            dt,
            accel=float(accel),
            speed_offset=float(speed_offset),
            nat_cfg=nat_cfg,
        )
        if tr is not None and np.all(np.isfinite(tr)):
            out.append(tr)
    return out


def _ordered_neutral_specs(nat_cfg: dict) -> list[tuple[float, float]]:
    """Spend the finite NEU budget on identity/mild interventions before extremes."""
    accs = [float(x) for x in nat_cfg.get("neutral_acc_values_mps2", [-1.0, -0.5, 0.0, 0.5, 1.0])]
    voffs = [float(x) for x in nat_cfg.get("neutral_target_speed_offsets_mps", [-2.0, 0.0, 2.0])]
    a_scale = max([abs(x) for x in accs] + [1.0])
    v_scale = max([abs(x) for x in voffs] + [1.0])
    specs = [(a, v) for a in accs for v in voffs]
    specs.sort(key=lambda z: (abs(z[0]) / a_scale + abs(z[1]) / v_scale, max(abs(z[0]) / a_scale, abs(z[1]) / v_scale), abs(z[0]), abs(z[1]), z[0], z[1]))
    return specs

def _map_route_trajectory_variants(
    scene: ScenarioData,
    current: np.ndarray,
    horizon: int,
    dt: float,
    nat_cfg: dict,
    *,
    accel: float = 0.0,
    speed_offset: float = 0.0,
) -> list[np.ndarray]:
    required = _required_route_length(current, horizon, dt, nat_cfg, accel=float(accel), speed_offset=float(speed_offset))
    routes = _map_route_polylines(scene, current, required, nat_cfg)
    return _retime_route_polylines(routes, current, horizon, dt, nat_cfg, accel=float(accel), speed_offset=float(speed_offset))


def build_pair_specific_ego_neutrals(
    scene: ScenarioData,
    critical: dict[str, np.ndarray],
    fallback_neutral: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Construct one pressure-removing ego intervention per critical agent.

    The manuscript defines Neutralize(S, tau_e) as interaction dependent.  A
    single global stopping trajectory can itself coerce a rear vehicle, while a
    keep trajectory can pressure a crossing/merge agent.  We therefore select a
    neutral intervention from a *fixed, proposal-bank-independent* ego control
    family for each critical actor.  Selection is lexicographic: avoid safety
    pressure first, then minimize the actor burden, then prefer the smallest ego
    control intervention.  Optional planner proposal families cannot change this
    natural-basis label.
    """
    limits = cfg.get("limits", {})
    nat_cfg = cfg.get("natural", {})
    A = int(limits.get("max_critical_agents", len(critical.get("valid", []))))
    H = int(cfg.get("time", {}).get("future_steps", 80))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    out = np.repeat(np.asarray(fallback_neutral, dtype=np.float32)[None, :H], A, axis=0)
    diag: list[dict[str, object]] = []
    cur = int(scene.current_time_index)
    ego_cur = np.asarray(scene.states[scene.sdc_track_index, cur], dtype=np.float32)

    # Fixed neutral family.  Prefer map-centreline timing; use logged *geometry*
    # only as an offline fallback, never logged timing.
    controls = [float(x) for x in nat_cfg.get("pair_neutral_acc_values_mps2", [0.0, -0.75, -1.5, -2.5, 0.75])]
    ego_bank: list[tuple[np.ndarray, float, str]] = []
    max_required = max(_required_route_length(ego_cur, H, dt, nat_cfg, accel=acc) for acc in controls) if controls else 5.0
    ego_routes = _map_route_polylines(scene, ego_cur, max_required, nat_cfg)
    for acc in controls:
        for tr in _retime_route_polylines(ego_routes, ego_cur, H, dt, nat_cfg, accel=acc):
            ego_bank.append((tr, abs(acc), "map_route"))
    ego_fut = scene.states[scene.sdc_track_index, cur + 1 : cur + 1 + H, :]
    if len(ego_fut) and np.any(ego_fut[:, 10] > 0.5):
        ego_logged = future_states_to_traj7(ego_fut, H, current_state=ego_cur)
        for acc in controls:
            tr = _route_geometry_timed_trajectory(ego_logged, ego_cur, H, dt, accel=acc, nat_cfg=nat_cfg)
            ego_bank.append((tr, abs(acc) + 0.05, "logged_geometry"))
    # Straight controls are only a final map-fragment fallback.
    for acc in controls:
        ego_bank.append((_jerk_bounded_straight_trajectory(ego_cur, H, dt, nat_cfg, accel=acc), abs(acc) + 0.25, "straight"))
    ego_bank.append((smooth_stop_trajectory(ego_cur, H, dt, decel=-2.0, creep_speed=0.0), 2.4, "smooth_stop"))
    dedup: list[tuple[np.ndarray, float, str]] = []
    for row in ego_bank:
        if any(_traj_distance(row[0], old[0]) < 0.05 for old in dedup):
            continue
        dedup.append(row)
    ego_bank = dedup or [(np.asarray(fallback_neutral, dtype=np.float32), 0.0, "fallback")]

    for a in range(A):
        row: dict[str, object] = {"slot": int(a), "valid": bool(a < len(critical.get("valid", [])) and critical["valid"][a])}
        if not row["valid"]:
            diag.append(row)
            continue
        idx = int(critical["track_index"][a])
        agent_cur = np.asarray(scene.states[idx, cur], dtype=np.float32)
        fut = scene.states[idx, cur + 1 : cur + 1 + H, :]
        fut_valid = fut[:, 10] > 0.5 if len(fut) else np.zeros(0, dtype=bool)
        logged = future_states_to_traj7(fut, H, current_state=agent_cur) if np.any(fut_valid) else _jerk_bounded_straight_trajectory(agent_cur, H, dt, nat_cfg, accel=0.0)
        map_ref = _map_route_trajectory_variants(scene, agent_cur, H, dt, nat_cfg, accel=0.0)
        if map_ref:
            if np.any(fut_valid):
                agent_ref = min(map_ref, key=lambda tr: _traj_distance_to_valid_future(tr, fut, fut_valid))
            else:
                agent_ref = map_ref[0]
            ref_kind = "map_route"
        elif np.any(fut_valid):
            agent_ref = _route_geometry_timed_trajectory(logged, agent_cur, H, dt, accel=0.0, nat_cfg=nat_cfg)
            ref_kind = "logged_geometry"
        else:
            agent_ref = _jerk_bounded_straight_trajectory(agent_cur, H, dt, nat_cfg, accel=0.0)
            ref_kind = "straight"
        rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))
        best = None
        best_meta = None
        for ego_tr, control_cost, kind in ego_bank:
            unsafe = unsafe_between(ego_tr, agent_ref, cfg, agent_type=int(scene.object_type[idx]))
            b, _ = compute_burden(
                agent_ref, ego_tr, cfg, int(scene.object_type[idx]), natural_ref=agent_ref, rho=rho,
                risk_known_zero=bool(cfg.get("engineering", {}).get("risk_known_zero_fastpath", True)) and not unsafe.unsafe,
            )
            event_count = int(np.asarray(unsafe.event_mask, dtype=bool).sum())
            score = (int(unsafe.unsafe), float(b), float(control_cost), event_count, -float(unsafe.min_distance))
            if best is None or score < best:
                best = score
                best_meta = (ego_tr, kind, b, unsafe.unsafe, unsafe.min_distance)
        assert best_meta is not None
        out[a] = np.asarray(best_meta[0], dtype=np.float32)
        row.update({
            "track_index": idx,
            "future_valid_steps": int(np.sum(fut_valid)),
            "future_valid_fraction": float(np.mean(fut_valid)) if len(fut_valid) else 0.0,
            "agent_reference": ref_kind,
            "neutral_source": str(best_meta[1]),
            "neutral_actor_burden": float(best_meta[2]),
            "neutral_actor_unsafe": bool(best_meta[3]),
            "neutral_min_distance_m": float(best_meta[4]),
        })
        diag.append(row)
    return out.astype(np.float32), diag

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



def _lane_segment_cloud(scene: ScenarioData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous lane-centre geometry for map compliance.

    The v16.8.11 point-cloud proxy measured distance only to sampled lane points.
    A trajectory lying exactly on a long lane segment could therefore be farther
    than 5 m from every *sample point* and be rejected as off-map.  Use exact
    point-to-segment distance while keeping the same physical thresholds.
    """
    starts: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    lens2: list[np.ndarray] = []
    for lane in scene.map_data.lanes.values():
        xy = np.asarray(lane.xy, dtype=np.float32)
        if xy.ndim != 2 or len(xy) < 2:
            continue
        a = xy[:-1, :2]
        v = xy[1:, :2] - a
        l2 = np.sum(v * v, axis=-1)
        keep = np.isfinite(l2) & (l2 > 1.0e-8) & np.all(np.isfinite(a), axis=-1) & np.all(np.isfinite(v), axis=-1)
        if np.any(keep):
            starts.append(a[keep])
            vecs.append(v[keep])
            lens2.append(l2[keep])
    if not starts:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.concatenate(starts, axis=0).astype(np.float32),
        np.concatenate(vecs, axis=0).astype(np.float32),
        np.concatenate(lens2, axis=0).astype(np.float32),
    )


def _trajectory_map_compliance(
    tr: np.ndarray,
    lane_segments: tuple[np.ndarray, np.ndarray, np.ndarray],
    object_type: int,
    nat_cfg: dict,
) -> tuple[bool, float, bool]:
    """Continuous lane-centre corridor check used during label construction."""
    if not bool(nat_cfg.get("map_filter_enabled", True)):
        return True, -1.0, False
    seg_a, seg_v, seg_l2 = lane_segments
    if seg_a.size == 0:
        return (not bool(nat_cfg.get("map_filter_require_available", False))), -1.0, False
    stride = max(1, int(nat_cfg.get("map_filter_stride", 4)))
    xy = np.asarray(tr, dtype=np.float32)[::stride, :2]
    if xy.size == 0 or not np.all(np.isfinite(xy)):
        return False, float("inf"), True

    # Compute exact point-to-polyline-segment distances.  Chunking bounds the
    # temporary [trajectory_points, segments, 2] tensor on map-heavy scenes.
    min_d2 = np.full(len(xy), np.inf, dtype=np.float64)
    chunk = max(int(nat_cfg.get("map_segment_chunk_size", 2048)), 128)
    for lo in range(0, len(seg_a), chunk):
        a = seg_a[lo : lo + chunk]
        v = seg_v[lo : lo + chunk]
        l2 = seg_l2[lo : lo + chunk]
        rel = xy[:, None, :] - a[None, :, :]
        u = np.sum(rel * v[None, :, :], axis=-1) / np.maximum(l2[None, :], 1.0e-8)
        u = np.clip(u, 0.0, 1.0)
        proj = a[None, :, :] + u[..., None] * v[None, :, :]
        d2 = np.sum((xy[:, None, :] - proj) ** 2, axis=-1)
        min_d2 = np.minimum(min_d2, np.min(d2, axis=1))
    d = np.sqrt(np.maximum(min_d2, 0.0))
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


def generate_natural_alternatives(
    scene: ScenarioData,
    critical: dict[str, np.ndarray],
    ego_neutral_traj: np.ndarray,
    cfg: dict,
    ablation: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build typed natural roots with an explicit low-burden support contract.

    ``ego_neutral_traj`` may be one global [H,7] fallback (legacy callers) or a
    pair-specific [A,H,7] intervention bank.  Fresh v16.8.11 label construction
    passes the latter so a neutral intervention for one actor cannot itself
    coerce another actor.
    """
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
    lane_segments = _lane_segment_cloud(scene)
    diagnostics: list[dict[str, object]] = []

    neutral_bank = np.asarray(ego_neutral_traj, dtype=np.float32)

    def neutral_for_slot(a: int) -> np.ndarray:
        if neutral_bank.ndim == 3 and a < neutral_bank.shape[0]:
            return neutral_bank[a, :H]
        return neutral_bank[:H]

    cur = int(scene.current_time_index)
    min_total = max(int(nat_cfg.get("min_natural_alternatives", 6)), 1)
    min_low = max(int(nat_cfg.get("min_low_burden_alternatives", 2)), 1)
    plaus_margin = float(nat_cfg.get("plausibility_beta_margin", 0.10))
    dedup_dist = max(float(nat_cfg.get("root_dedup_mean_distance_m", 0.10)), 0.0)
    obs_min_steps = max(int(nat_cfg.get("obs_min_future_valid_steps", 60)), 0)
    obs_min_frac = float(np.clip(nat_cfg.get("obs_min_future_valid_fraction", 0.70), 0.0, 1.0))

    for a in range(A):
        if not bool(critical["valid"][a]):
            diagnostics.append({"slot": int(a), "valid": False})
            continue
        idx = int(critical["track_index"][a])
        object_type = int(scene.object_type[idx])
        current = np.asarray(scene.states[idx, cur], dtype=np.float32)
        fut_states = np.asarray(scene.states[idx, cur + 1 : cur + 1 + H, :], dtype=np.float32)
        fut_mask = fut_states[:, 10] > 0.5 if len(fut_states) else np.zeros(0, dtype=bool)
        valid_steps = int(np.sum(fut_mask))
        valid_frac = float(np.mean(fut_mask)) if len(fut_mask) else 0.0
        has_future = bool(valid_steps > 0)
        if has_future:
            logged = future_states_to_traj7(fut_states, H, current_state=current)
        else:
            logged = _jerk_bounded_straight_trajectory(current, H, dt, nat_cfg, accel=0.0)

        pair_neutral = neutral_for_slot(a)
        rho = PriorityRelation(int(critical.get("base_priority", np.zeros(A, dtype=np.int32))[a]))

        # Build a timing-neutral route reference even when the full logged future
        # exists.  Raw logged timing remains OBS supervision, but it is not a
        # normative progress/burden baseline for NEU/PRIO roots because it may
        # already contain ego-induced yielding or unrelated interaction effects.
        neutral_accs = [float(x) for x in nat_cfg.get("neutral_acc_values_mps2", [-1.0, -0.5, 0.0, 0.5, 1.0])]
        neutral_voffs = [float(x) for x in nat_cfg.get("neutral_target_speed_offsets_mps", [-2.0, 0.0, 2.0])]
        prio_accs = [float(x) for x in nat_cfg.get("prio_acc_values_mps2", [-0.5, 0.0, 0.5])]
        map_fb_accs = [float(x) for x in nat_cfg.get("map_route_neutral_acc_values_mps2", [0.0, -0.5, 0.5, -1.0, 1.0])]
        all_accs = neutral_accs + prio_accs + map_fb_accs
        max_acc = max(all_accs + [0.0])
        max_voff = max(neutral_voffs + [0.0])
        required = _required_route_length(current, H, dt, nat_cfg, accel=max_acc, speed_offset=max_voff)
        route_polylines = _map_route_polylines(scene, current, required, nat_cfg)

        def route_variants(acc: float = 0.0, speed_offset: float = 0.0) -> list[np.ndarray]:
            return _retime_route_polylines(
                route_polylines, current, H, dt, nat_cfg, accel=float(acc), speed_offset=float(speed_offset)
            )

        map_refs = route_variants(0.0, 0.0)
        if map_refs:
            if has_future:
                natural_ref = min(map_refs, key=lambda tr: _traj_distance_to_valid_future(tr, fut_states, fut_mask))
            else:
                natural_ref = map_refs[0]
            reference_kind = "map_route_neutral_timing"
        elif has_future:
            natural_ref = _route_geometry_timed_trajectory(logged, current, H, dt, accel=0.0, nat_cfg=nat_cfg)
            reference_kind = "logged_geometry_neutral_timing"
        else:
            natural_ref = _jerk_bounded_straight_trajectory(current, H, dt, nat_cfg, accel=0.0)
            reference_kind = "straight_neutral_timing"

        if rho == PriorityRelation.UNKNOWN:
            rho = determine_priority(scene, idx, pair_neutral, natural_ref, cfg)
        scene_current = scene.states[:, cur, :] if scene.states.ndim == 3 else None
        beta[a] = adaptive_beta(scene_current, object_type, rho, cfg, use_adaptive=True, ego_index=scene.sdc_track_index)
        obs_eligible = bool(has_future and valid_steps >= obs_min_steps and valid_frac >= obs_min_frac)
        obs_contam = _observed_yield_contamination(scene, idx, logged, pair_neutral, H, dt, nat_cfg) if has_future else 0.0

        candidates: list[tuple[np.ndarray, NaturalSource, float, float, str]] = []
        if use_obs and obs_eligible:
            max_obs = int(nat_cfg.get("max_obs_samples", 8))
            for ss, shift_s, lat in _ordered_observational_specs(nat_cfg)[:max_obs]:
                tr = resample_logged(
                    logged,
                    H,
                    time_shift_steps=int(round(float(shift_s) / dt)),
                    speed_scale=float(ss),
                    lateral_offset=float(lat),
                    current=current,
                    dt=dt,
                )
                candidates.append((tr, NaturalSource.OBS, float(nat_cfg.get("source_weight_obs", 1.0)), obs_contam, "primary_obs"))
        if use_neu:
            max_neu = int(nat_cfg.get("max_neutral_samples", 8))
            count = 0
            for acc, voff in _ordered_neutral_specs(nat_cfg):
                if count >= max_neu:
                    break
                variants = route_variants(acc, voff)
                if variants:
                    for tr in variants:
                        candidates.append((tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "primary_neu_map_route"))
                        count += 1
                        if count >= max_neu:
                            break
                elif has_future:
                    tr = _route_geometry_timed_trajectory(logged, current, H, dt, accel=float(acc), speed_offset=float(voff), nat_cfg=nat_cfg)
                    candidates.append((tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "primary_neu_logged_geometry"))
                    count += 1
                else:
                    tr = _jerk_bounded_straight_trajectory(current, H, dt, nat_cfg, accel=float(acc), speed_offset=float(voff))
                    candidates.append((tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "primary_neu_straight"))
                    count += 1
        if use_prio:
            max_prio = int(nat_cfg.get("prio_max_samples", 8))
            count = 0
            for acc in prio_accs:
                if count >= max_prio:
                    break
                variants = route_variants(acc, 0.0)
                if variants:
                    for tr in variants:
                        candidates.append((tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2)), 0.0, "primary_prio_map_route"))
                        count += 1
                        if count >= max_prio:
                            break
                elif has_future:
                    tr = _route_geometry_timed_trajectory(logged, current, H, dt, accel=float(acc), nat_cfg=nat_cfg)
                    candidates.append((tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2)), 0.0, "primary_prio_logged_geometry"))
                    count += 1
                else:
                    tr = _jerk_bounded_straight_trajectory(current, H, dt, nat_cfg, accel=float(acc))
                    candidates.append((tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2)), 0.0, "primary_prio_straight"))
                    count += 1
        if not candidates:
            # Preserve legacy ablation behavior: a completely empty branch union
            # gets one observational anchor rather than producing malformed labels.
            candidates.append((logged, NaturalSource.OBS, 1.0, obs_contam, "ablation_anchor"))

        kept = 0
        low_kept = 0
        raw_w = np.zeros(M, dtype=np.float32)
        reject_counts = {k: 0 for k in ("nonfinite", "map", "burden", "contamination", "priority", "duplicate", "capacity")}
        priority_rejection_reasons: dict[str, int] = {}
        map_rejected_min_max_distance = float("inf")
        map_rejected_max_max_distance = 0.0
        best_rejected_burden = float("inf")
        accepted_by_phase: dict[str, int] = {}
        attempted_by_phase: dict[str, int] = {}

        def try_keep(
            tr: np.ndarray,
            src: NaturalSource,
            src_weight: float,
            contamination: float,
            phase: str,
        ) -> bool:
            nonlocal kept, low_kept, map_rejected_min_max_distance, map_rejected_max_max_distance, best_rejected_burden
            attempted_by_phase[phase] = attempted_by_phase.get(phase, 0) + 1
            if kept >= M:
                reject_counts["capacity"] += 1
                return False
            tr = np.asarray(tr, dtype=np.float32)
            finite_ok = bool(tr.shape == (H, 7) and np.all(np.isfinite(tr)))
            if not finite_ok:
                reject_counts["nonfinite"] += 1
                return False
            if dedup_dist > 0.0 and any(_traj_distance(tr, traj[a, m]) < dedup_dist for m in np.where(valid[a])[0]):
                reject_counts["duplicate"] += 1
                return False
            b_total, _ = compute_burden(
                tr,
                pair_neutral,
                cfg,
                object_type,
                natural_ref=natural_ref,
                rho=rho,
            )
            pr_ok, pr_reason = priority_preservation_check(tr, natural_ref, rho, cfg)
            map_ok, map_dist, map_was_verified = _trajectory_map_compliance(tr, lane_segments, object_type, nat_cfg)
            plausible = bool(float(b_total) <= float(beta[a]) + plaus_margin)
            contamination_ok = not (
                src == NaturalSource.OBS
                and float(contamination) >= float(nat_cfg.get("obs_drop_contamination_above", 0.90))
            )
            priority_keep = bool(pr_ok or rho != PriorityRelation.AGENT_PRIORITY)
            if not map_ok:
                reject_counts["map"] += 1
                if np.isfinite(map_dist):
                    map_rejected_min_max_distance = min(map_rejected_min_max_distance, float(map_dist))
                    map_rejected_max_max_distance = max(map_rejected_max_max_distance, float(map_dist))
            if not plausible:
                reject_counts["burden"] += 1
                best_rejected_burden = min(best_rejected_burden, float(b_total))
            if not contamination_ok:
                reject_counts["contamination"] += 1
            if not priority_keep:
                reject_counts["priority"] += 1
                priority_rejection_reasons[pr_reason] = priority_rejection_reasons.get(pr_reason, 0) + 1
            if not (map_ok and plausible and contamination_ok and priority_keep):
                return False
            traj[a, kept] = tr
            valid[a, kept] = True
            source[a, kept] = int(src)
            burden_neutral[a, kept] = float(b_total)
            priority_ok[a, kept] = bool(pr_ok)
            obs_contamination[a, kept] = float(contamination)
            map_compliant[a, kept] = bool(map_ok)
            map_distance_max[a, kept] = float(map_dist)
            map_verified[a, kept] = bool(map_was_verified)
            dist = _traj_distance(tr, natural_ref)
            decontam_factor = (
                max(
                    float(nat_cfg.get("obs_weight_floor", 0.05)),
                    1.0 - float(nat_cfg.get("obs_contamination_weight", 0.90)) * float(contamination),
                )
                if src == NaturalSource.OBS else 1.0
            )
            raw_w[kept] = (
                float(src_weight)
                * decontam_factor
                * np.exp(-dist / max(float(nat_cfg.get("sigma_traj_m", 15.0)), 1.0e-6))
                * np.exp(-float(b_total) / max(float(nat_cfg.get("sigma_b", 0.5)), 1.0e-6))
            )
            if float(b_total) <= float(beta[a]) + 1.0e-8:
                low_kept += 1
            accepted_by_phase[phase] = accepted_by_phase.get(phase, 0) + 1
            kept += 1
            return True

        for tr, src, src_weight, contamination, phase in candidates:
            try_keep(tr, src, src_weight, contamination, phase)

        def support_satisfied() -> bool:
            return kept >= min_total and low_kept >= min_low

        # First fallback: lane-graph centreline roots. This is the main v16.8.11
        # repair for curved lanes and short/invalid future tracks.
        if not support_satisfied():
            if use_neu or (not use_obs and not use_prio):
                for acc in nat_cfg.get("map_route_neutral_acc_values_mps2", [0.0, -0.5, 0.5, -1.0, 1.0]):
                    for tr in route_variants(float(acc), 0.0):
                        try_keep(tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "fallback_map_neu")
                        if support_satisfied() or kept >= M:
                            break
                    if support_satisfied() or kept >= M:
                        break
            if not support_satisfied() and use_prio:
                for acc in nat_cfg.get("map_route_prio_acc_values_mps2", [0.0, 0.5, -0.5]):
                    for tr in route_variants(float(acc), 0.0):
                        try_keep(tr, NaturalSource.PRIO, float(nat_cfg.get("source_weight_prio", 1.2)), 0.0, "fallback_map_prio")
                        if support_satisfied() or kept >= M:
                            break
                    if support_satisfied() or kept >= M:
                        break

        # Second fallback: observed path geometry with fresh timing. Only geometry
        # is reused, so logged yielding timing is not copied into the neutral root.
        if not support_satisfied() and has_future:
            for acc in nat_cfg.get("geometry_neutral_acc_values_mps2", [0.0, -0.5, 0.5, -1.0, 1.0]):
                tr = _route_geometry_timed_trajectory(logged, current, H, dt, accel=float(acc), nat_cfg=nat_cfg)
                try_keep(tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "fallback_logged_geometry")
                if support_satisfied() or kept >= M:
                    break

        # Final conservative fallback for fragmented maps. It still passes the
        # identical map/priority/burden filter and therefore cannot manufacture a
        # support PASS by bypassing the certificate semantics.
        if not support_satisfied():
            for acc in nat_cfg.get("straight_fallback_acc_values_mps2", [-0.5, 0.0, 0.5, 1.0, -1.0, 1.5, -1.5]):
                tr = _jerk_bounded_straight_trajectory(current, H, dt, nat_cfg, accel=float(acc))
                try_keep(tr, NaturalSource.NEU, float(nat_cfg.get("source_weight_neu", 0.8)), 0.0, "fallback_straight")
                if support_satisfied() or kept >= M:
                    break

        weights[a] = _normalize_weights(raw_w, valid[a])
        low_mask = valid[a] & (burden_neutral[a] <= float(beta[a]) + 1.0e-8)
        diagnostics.append({
            "slot": int(a),
            "valid": True,
            "track_index": int(idx),
            "object_type": int(object_type),
            "future_valid_steps": int(valid_steps),
            "future_valid_fraction": float(valid_frac),
            "obs_eligible": bool(obs_eligible),
            "reference_kind": reference_kind,
            "rho": int(rho),
            "beta": float(beta[a]),
            "root_count": int(np.sum(valid[a])),
            "low_burden_root_count": int(np.sum(low_mask)),
            "map_verified_root_count": int(np.sum(map_verified[a] & valid[a])),
            "min_burden": float(np.min(burden_neutral[a, valid[a]])) if np.any(valid[a]) else None,
            "obs_contamination": float(obs_contam),
            "attempted_by_phase": attempted_by_phase,
            "accepted_by_phase": accepted_by_phase,
            "rejection_counts": reject_counts,
            "priority_rejection_reasons": priority_rejection_reasons,
            "map_rejected_min_max_distance_m": None if not np.isfinite(map_rejected_min_max_distance) else float(map_rejected_min_max_distance),
            "map_rejected_max_max_distance_m": float(map_rejected_max_max_distance),
            "best_rejected_burden": None if not np.isfinite(best_rejected_burden) else float(best_rejected_burden),
        })

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
        "_diagnostics": diagnostics,
    }
