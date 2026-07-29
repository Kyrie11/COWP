from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from ocrap.data.schema import F_AGENT, RawScenario
from ocrap.utils.seed import stable_seed


def make_agent_state(x: float, y: float, vx: float, vy: float, heading: float, length: float, width: float, obj_type: int, valid: bool = True, conf: float = 1.0) -> np.ndarray:
    s = np.zeros(F_AGENT, dtype=np.float32)
    s[0], s[1], s[2] = x, y, 0.0
    s[3], s[4] = vx, vy
    s[5], s[6] = 0.0, 0.0
    s[7] = heading
    s[8], s[9] = math.sin(heading), math.cos(heading)
    s[10], s[11], s[12] = length, width, 1.5
    s[13] = float(obj_type)
    s[14] = 1.0 if valid else 0.0
    s[15] = conf if valid else 0.0
    return s


def _make_map(route_x: np.ndarray, max_polylines: int = 16, max_points: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P, Q, F = max_polylines, max_points, 10
    maps = np.zeros((P, Q, F), dtype=np.float32)
    valid = np.zeros((P, Q), dtype=bool)
    ys = [0.0, 3.5, -3.5, 7.0, -7.0]
    for p, y in enumerate(ys[:P]):
        xs = np.linspace(-20, 120, Q, dtype=np.float32)
        maps[p, :, 0] = xs
        maps[p, :, 1] = y
        maps[p, :, 3] = 1.0
        maps[p, :, 5] = 1.0 if abs(y) <= 3.5 else 2.0
        maps[p, :, 6] = 13.4
        maps[p, :, 7] = 1.0 if y == 0.0 else 0.0
        maps[p, :, 9] = 1.0
        valid[p] = True
    # crosswalk polygon-like polyline
    if P > 5:
        p = 5
        pts = np.array([[35, -5], [35, 5], [38, 5], [38, -5]], dtype=np.float32)
        reps = np.resize(pts, (Q, 2))
        maps[p, :, :2] = reps
        maps[p, :, 5] = 4.0
        maps[p, :, 9] = 1.0
        valid[p] = True
    route = np.zeros((len(route_x), 6), dtype=np.float32)
    route[:, 0] = route_x
    route[:, 1] = 0.0
    route[:, 2] = 0.0
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return maps, valid, route


def make_synthetic_scenario(index: int, cfg: dict | None = None, artifact: bool = True) -> RawScenario:
    cfg = cfg or {}
    rng = np.random.default_rng(stable_seed("synthetic", index, artifact))
    T = int(cfg.get("synthetic_T", 80))
    A = int(cfg.get("max_agents", 32))
    A = max(A, 8)
    timestamps = np.arange(T, dtype=np.float32) / float(cfg.get("sample_rate_hz", 10.0))
    states = np.zeros((T, A, F_AGENT), dtype=np.float32)
    valid = np.zeros((T, A), dtype=bool)
    v_ego = 8.0 + 0.2 * (index % 3)
    for t in range(T):
        time = t * 0.1
        x = v_ego * time
        states[t, 0] = make_agent_state(x, 0.0, v_ego, 0.0, 0.0, 4.8, 2.0, 1, True)
        valid[t, 0] = True
        # Vehicle occluder near the route creates a physically grounded unknown shadow.
        occ_x = x + 14.0
        states[t, 1] = make_agent_state(occ_x, -1.2, 0.5, 0.0, 0.0, 5.2, 2.2, 1, True)
        valid[t, 1] = True
        # Leading vehicle and adjacent vehicle create interaction/near-contact signals.
        lead_x = x + 28.0 - 0.03 * t
        states[t, 2] = make_agent_state(lead_x, 0.0, max(3.0, v_ego - 2.5), 0.0, 0.0, 4.8, 2.0, 1, True)
        valid[t, 2] = True
        adj_x = x + 9.0 + 0.4 * math.sin(0.1 * t)
        states[t, 3] = make_agent_state(adj_x, 3.5, v_ego, 0.0, 0.0, 4.8, 2.0, 1, True)
        valid[t, 3] = True
        ped_valid = 30 <= t <= 60
        states[t, 4] = make_agent_state(x + 18.0, -4.0 + 0.08 * (t - 30), 0.0, 0.8, math.pi / 2, 0.7, 0.7, 2, ped_valid)
        valid[t, 4] = ped_valid
    route_x = np.linspace(-20, 140, int(cfg.get("route_points", 80)), dtype=np.float32)
    maps, map_valid, route = _make_map(route_x, int(cfg.get("max_map_polylines", 16)), int(cfg.get("max_polyline_points", 80)))
    B, Fsig = int(cfg.get("max_dynamic_signals", 16)), 6
    dyn = np.zeros((T, B, Fsig), dtype=np.float32)
    # lane_id, state(red/yellow/green numeric), stop_x, stop_y, controlled_lane_id, valid
    dyn[:, 0, 0] = 1
    dyn[:, 0, 1] = 0  # green by default
    dyn[:, 0, 2] = 35
    dyn[:, 0, 3] = 0
    dyn[:, 0, 4] = 1
    dyn[:, 0, 5] = 1
    object_ids = [f"sdc_{index}", f"occluder_{index}", f"lead_{index}", f"adjacent_{index}", f"ped_{index}"] + [f"pad_{i}" for i in range(A - 5)]
    return RawScenario(
        scenario_id=("synthetic_artifact" if artifact else "synthetic") + f"_{index:06d}",
        timestamps=timestamps,
        sdc_track_index=0,
        agent_states=states,
        agent_valid=valid,
        map_polylines=maps,
        map_valid=map_valid,
        route=route,
        dynamic_map=dyn,
        object_ids=object_ids[:A],
        metadata={"source": "synthetic_artifact" if artifact else "synthetic", "artifact_fixture": artifact},
    )


def iter_synthetic_scenarios(num: int, seed: int = 0, cfg: dict | None = None, artifact: bool = True) -> Iterator[RawScenario]:
    for i in range(num):
        yield make_synthetic_scenario(i + seed * 1000, cfg, artifact=artifact)
