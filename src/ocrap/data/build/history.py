from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import RawScenario, SceneHistory
from ocrap.planning.route_lattice import project_to_route
from ocrap.simulation.observation.bev import render_base_occ_mask
from ocrap.utils.geometry import transform_points_to_ego, transform_states_to_ego



def _future_route_proxy(future_ego: np.ndarray, future_valid: np.ndarray, max_points: int) -> np.ndarray:
    if future_ego.size == 0 or future_valid.size == 0:
        pts = np.stack([np.linspace(0, max_points - 1, max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    else:
        valid = future_valid[:, 0].astype(bool) if future_valid.ndim == 2 and future_valid.shape[1] else np.zeros((future_ego.shape[0],), dtype=bool)
        pts = future_ego[valid, 0, :2].astype(np.float32)
        if len(pts) < 2:
            pts = np.stack([np.linspace(0, max_points - 1, max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    if len(pts) < max_points:
        pad = np.repeat(pts[-1:, :], max_points - len(pts), axis=0)
        pts = np.concatenate([pts, pad], axis=0)
    else:
        idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = pts[idx]
    route = np.zeros((max_points, 6), dtype=np.float32)
    route[:, :2] = pts[:, :2]
    d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
    route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return route


def _sanitize_route(route: np.ndarray, future_ego: np.ndarray, future_valid: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    max_points = int(cfg.get("route_points", route.shape[0] if route.size else 80))
    meta = {"route_sanitized": False, "route_projection_distance_m": 0.0}
    if route.size and len(route) >= 2:
        try:
            proj = project_to_route(np.zeros(2, dtype=np.float32), route)
            meta["route_projection_distance_m"] = float(proj.distance)
            length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
            if proj.distance <= float(cfg.get("max_route_projection_distance_m", 8.0)) and length >= float(cfg.get("min_route_length_m", 10.0)):
                return route.astype(np.float32), meta
        except Exception:
            pass
    meta["route_sanitized"] = True
    meta["route_sanitize_reason"] = "route_not_near_ego_or_too_short"
    return _future_route_proxy(future_ego, future_valid, max_points), meta

def ego_from_agent_state(agent_state: np.ndarray) -> np.ndarray:
    return np.array([agent_state[0], agent_state[1], agent_state[3], agent_state[4], agent_state[7], 0.0, math.hypot(agent_state[3], agent_state[4]), agent_state[10], agent_state[11]], dtype=np.float32)


def transform_map_and_route(raw: RawScenario, ego_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_xy = ego_state[:2]
    ego_h = float(ego_state[7])
    maps = raw.map_polylines.copy().astype(np.float32)
    if maps.size:
        maps[..., :2] = transform_points_to_ego(maps[..., :2], ego_xy, ego_h)
    route = raw.route.copy().astype(np.float32)
    if route.size:
        route[..., :2] = transform_points_to_ego(route[..., :2], ego_xy, ego_h)
        if route.shape[-1] > 2:
            route[..., 2] = route[..., 2] - ego_h
    return maps, raw.map_valid.copy(), route


def construct_history(raw: RawScenario, t: int, cfg: dict) -> SceneHistory:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
    total_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
    sdc = int(raw.sdc_track_index)
    order = [sdc] + [i for i in range(raw.agent_states.shape[1]) if i != sdc]
    max_agents = min(int(cfg.get("max_agents", len(order))), len(order))
    order = order[:max_agents]
    t0 = max(0, t - H + 1)
    hist = raw.agent_states[t0 : t + 1, order]
    hist_valid = raw.agent_valid[t0 : t + 1, order]
    if hist.shape[0] < H:
        pad = np.zeros((H - hist.shape[0], hist.shape[1], hist.shape[2]), dtype=np.float32)
        pad_valid = np.zeros((H - hist.shape[0], hist.shape[1]), dtype=bool)
        hist = np.concatenate([pad, hist], axis=0)
        hist_valid = np.concatenate([pad_valid, hist_valid], axis=0)
    tend = min(raw.agent_states.shape[0], t + total_future)
    future = raw.agent_states[t:tend, order]
    future_valid = raw.agent_valid[t:tend, order]
    if future.shape[0] < total_future:
        pad = np.zeros((total_future - future.shape[0], len(order), raw.agent_states.shape[-1]), dtype=np.float32)
        pad_valid = np.zeros((total_future - future.shape[0], len(order)), dtype=bool)
        future = np.concatenate([future, pad], axis=0)
        future_valid = np.concatenate([future_valid, pad_valid], axis=0)
    ego_raw = raw.agent_states[t, sdc]
    hist_e = transform_states_to_ego(hist, ego_raw)
    fut_e = transform_states_to_ego(future, ego_raw)
    maps, map_valid, route = transform_map_and_route(raw, ego_raw)
    route, route_meta = _sanitize_route(route, fut_e, future_valid, cfg)
    dyn = raw.dynamic_map[max(0, t - H + 1) : t + 1]
    if dyn.shape[0] < H:
        pad = np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32)
        dyn = np.concatenate([pad, dyn], axis=0)
    h = SceneHistory(
        scene_id=raw.scenario_id,
        original_scenario_id=str(raw.metadata.get("original_scenario_id", raw.scenario_id)),
        time_index=int(t),
        agent_history=hist_e.astype(np.float32),
        agent_valid=hist_valid.astype(bool),
        map_polylines=maps.astype(np.float32),
        map_valid=map_valid.astype(bool),
        dynamic_map=dyn.astype(np.float32),
        route=route.astype(np.float32),
        occ_mask=np.zeros((int(cfg.get("bev_channels", 7)), 2, 2), dtype=np.float32),
        ego_state=ego_from_agent_state(hist_e[-1, 0]),
        future_agent_states=fut_e.astype(np.float32),
        future_agent_valid=future_valid.astype(bool),
        metadata={
            "speed_limit": float(cfg.get("speed_limit_default", 13.4)),
            "shoulder_available": True,
            "adjacent_available": True,
            "time_sampling_reasons": [],
            "source": raw.metadata.get("source", "unknown"),
            "agent_order": [int(i) for i in order],
            "ego_global_xy": [float(ego_raw[0]), float(ego_raw[1])],
            "ego_global_heading": float(ego_raw[7]),
            "waymax_planning_timestep": int(t),
            "waymax_sdc_original_index": int(sdc),
            # Private runtime-only handles.  These are intentionally not written
            # to NPZ; they are consumed by the Waymax rollout backend before
            # DatasetSample.to_npz_dict serializes json-safe metadata.
            "_waymax_state": raw.metadata.get("_waymax_state"),
            "_waymax_scenario_index": raw.metadata.get("_waymax_scenario_index"),
            "source_scenario_index": raw.metadata.get("_waymax_scenario_index", -1),
            "official_scenario_id": raw.metadata.get("official_scenario_id"),
            "legacy_scenario_id": raw.metadata.get("legacy_scenario_id"),
            "scenario_id_source": raw.metadata.get("scenario_id_source", "unknown"),
            "womd_source_role": raw.metadata.get("womd_source_role", "unknown"),
            "womd_source_pattern": raw.metadata.get("womd_source_pattern", ""),
            "waymax_max_num_objects": raw.metadata.get("waymax_max_num_objects", -1),
            **route_meta,
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h


def construct_history_from_waymax_state(
    state,
    static_template: RawScenario,
    t: int,
    cfg: dict,
    *,
    scenario_id: str | None = None,
    scenario_index: int | None = None,
) -> SceneHistory:
    """Construct a history directly from a Waymax state.

    This compatibility implementation preserves the exact legacy splice
    semantics.  Deployments may replace it with the optimized zero-copy path,
    but keeping the reference implementation here prevents the hot-path caller
    and its regression test from depending on an unavailable symbol.
    """
    from ocrap.data.waymax_loader import raw_scenario_from_waymax_state

    sid = str(scenario_id or static_template.scenario_id)
    idx = int(0 if scenario_index is None else scenario_index)
    raw = raw_scenario_from_waymax_state(
        state,
        sid,
        idx,
        cfg,
        trajectory_mode="closed_loop_splice",
        splice_until=int(t),
        static_template=static_template,
    )
    return construct_history(raw, int(t), cfg)
