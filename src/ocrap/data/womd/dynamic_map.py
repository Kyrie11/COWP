from __future__ import annotations

import numpy as np


def parse_dynamic_map(scenario, T: int, max_signals: int = 16) -> np.ndarray:
    F = 6  # lane_id, state, stop_x, stop_y, controlled_lane_id, valid
    out = np.zeros((T, max_signals, F), dtype=np.float32)
    states = list(getattr(scenario, "dynamic_map_states", []))
    for t, dms in enumerate(states[:T]):
        lane_states = list(getattr(dms, "lane_states", []))
        for b, lane in enumerate(lane_states[:max_signals]):
            out[t, b, 0] = float(getattr(lane, "lane", getattr(lane, "lane_id", 0)))
            out[t, b, 1] = float(getattr(lane, "state", 0))
            stop = getattr(lane, "stop_point", None)
            if stop is not None:
                out[t, b, 2] = float(getattr(stop, "x", 0.0))
                out[t, b, 3] = float(getattr(stop, "y", 0.0))
            out[t, b, 4] = out[t, b, 0]
            out[t, b, 5] = 1.0
    return out
