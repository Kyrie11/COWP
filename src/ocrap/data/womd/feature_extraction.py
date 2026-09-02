from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import F_AGENT


def state_to_feature(state, object_type: int) -> tuple[np.ndarray, bool]:
    valid = bool(getattr(state, "valid", False))
    x = float(getattr(state, "center_x", 0.0))
    y = float(getattr(state, "center_y", 0.0))
    z = float(getattr(state, "center_z", 0.0))
    vx = float(getattr(state, "velocity_x", 0.0))
    vy = float(getattr(state, "velocity_y", 0.0))
    heading = float(getattr(state, "heading", 0.0))
    length = float(getattr(state, "length", 4.8))
    width = float(getattr(state, "width", 2.0))
    height = float(getattr(state, "height", 1.5))
    feat = np.zeros(F_AGENT, dtype=np.float32)
    feat[:5] = [x, y, z, vx, vy]
    feat[7] = heading
    feat[8] = math.sin(heading)
    feat[9] = math.cos(heading)
    feat[10:14] = [length, width, height, float(object_type)]
    feat[14] = float(valid)
    feat[15] = 1.0 if valid else 0.0
    return feat, valid


def extract_agent_arrays(scenario, indices: list[int], max_agents: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tracks = list(getattr(scenario, "tracks", []))
    T = max((len(getattr(tr, "states", [])) for tr in tracks), default=0)
    states = np.zeros((T, max_agents, F_AGENT), dtype=np.float32)
    valid = np.zeros((T, max_agents), dtype=bool)
    ids: list[str] = []
    for new_i in range(max_agents):
        if new_i >= len(indices):
            ids.append("")
            continue
        old_i = indices[new_i]
        tr = tracks[old_i]
        ids.append(str(getattr(tr, "id", old_i)))
        object_type = int(getattr(tr, "object_type", 1))
        for t, st in enumerate(getattr(tr, "states", [])):
            feat, ok = state_to_feature(st, object_type)
            states[t, new_i] = feat
            valid[t, new_i] = ok
    # acceleration from finite differences for valid adjacent pairs.
    for a in range(max_agents):
        for t in range(1, T):
            if valid[t, a] and valid[t - 1, a]:
                states[t, a, 5] = (states[t, a, 3] - states[t - 1, a, 3]) / 0.1
                states[t, a, 6] = (states[t, a, 4] - states[t - 1, a, 4]) / 0.1
    return states, valid, ids
