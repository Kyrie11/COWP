from __future__ import annotations

import numpy as np


def select_sdc_first_indices(scenario, max_agents: int, current_index: int | None = None, radius: float = 80.0) -> list[int]:
    tracks = list(getattr(scenario, "tracks", []))
    if not tracks:
        return []
    sdc = int(getattr(scenario, "sdc_track_index", 0))
    T = max((len(getattr(tr, "states", [])) for tr in tracks), default=0)
    t = current_index if current_index is not None else max(0, min(T - 1, 10))
    sdc = min(max(sdc, 0), len(tracks) - 1)
    def xy_valid(tr, ti):
        states = getattr(tr, "states", [])
        if not states:
            return np.zeros(2), False
        st = states[min(max(ti, 0), len(states) - 1)]
        return np.array([float(getattr(st, "center_x", 0.0)), float(getattr(st, "center_y", 0.0))]), bool(getattr(st, "valid", False))
    ego_xy, _ = xy_valid(tracks[sdc], t)
    scores = []
    for i, tr in enumerate(tracks):
        if i == sdc:
            continue
        min_d = float("inf")
        future_min = float("inf")
        seen_valid = False
        for ti in range(max(0, t - 10), min(T, t + 50)):
            xy, ok = xy_valid(tr, ti)
            if ok:
                seen_valid = True
                d = float(np.linalg.norm(xy - ego_xy))
                min_d = min(min_d, d)
                if ti >= t:
                    future_min = min(future_min, d)
        reachable = future_min < radius * 1.2
        in_radius = min_d < radius
        priority = 0 if in_radius else (1 if reachable else 3)
        if seen_valid:
            scores.append((priority, min_d, future_min, i))
    scores.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    order = [sdc] + [i for *_, i in scores]
    # add remaining tracks by index for padding candidates
    order += [i for i in range(len(tracks)) if i not in set(order)]
    return order[:max_agents]
