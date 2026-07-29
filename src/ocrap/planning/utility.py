from __future__ import annotations

import numpy as np


def nominal_utility(prefix_states: np.ndarray, prefix_controls: np.ndarray, diagnostics: dict, cfg: dict) -> float:
    w = cfg.get("utility_weights", {})
    progress = float(prefix_states[-1, 0] - prefix_states[0, 0])
    comfort = float(np.mean(np.abs(prefix_controls[:, 0])) + 0.3 * np.mean(np.abs(prefix_controls[:, 2])) if len(prefix_controls) else 0.0)
    route_dev = float(diagnostics.get("max_route_deviation", 0.0))
    logdiv = float(diagnostics.get("log_divergence", 0.0))
    offroad = float(diagnostics.get("offroad_hard", 0.0))
    wrong = float(diagnostics.get("wrong_way_hard", 0.0))
    return (
        float(w.get("progress", 1.0)) * progress
        - float(w.get("comfort", 0.05)) * comfort
        - float(w.get("route", 0.5)) * route_dev
        - float(w.get("logdiv", 0.05)) * logdiv
        - float(w.get("offroad", 5.0)) * offroad
        - float(w.get("wrongway", 5.0)) * wrong
    )
