from __future__ import annotations

import numpy as np

from cowp.label.trajectory_primitives import smooth_stop_trajectory


def conservative_fallback(current_state: np.ndarray, cfg: dict) -> np.ndarray:
    horizon = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    decel = float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0))
    return smooth_stop_trajectory(current_state, horizon, dt, decel=decel, creep_speed=0.0)
