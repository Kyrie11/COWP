from __future__ import annotations

import numpy as np

from ocrap.data.schema import RecoveryOption

MODES = [
    "stop", "brake_lane", "lateral_escape", "yield_rejoin", "pull_over", "mitigate_contact", "post_contact_stabilize", "avoid_secondary",
]


def default_recovery_options(num_options: int = 24, shoulder_available: bool = True, adjacent_available: bool = True) -> list[RecoveryOption]:
    opts: list[RecoveryOption] = []
    grids = {
        "stop": [np.array([-3.0, 8.0, 0.0]), np.array([-5.0, 5.0, 0.0]), np.array([-2.0, 15.0, 0.0])],
        "brake_lane": [np.array([-3.0, 1.5, 0.0]), np.array([-5.0, 1.0, 0.0]), np.array([-2.0, 2.5, 0.0])],
        "lateral_escape": [np.array([3.5, 6.0, 1.5]), np.array([-3.5, 5.0, 1.5]), np.array([2.0, 4.0, 1.0])],
        "yield_rejoin": [np.array([-2.0, 8.0, 2.0]), np.array([-3.0, 12.0, 3.0]), np.array([-1.0, 6.0, 1.5])],
        "pull_over": [np.array([-4.0, 25.0, 0.0]), np.array([4.0, 20.0, 0.0]), np.array([-3.0, 15.0, 1.0])],
        "mitigate_contact": [np.array([-4.0, 0.2, 3.0]), np.array([-6.0, -0.2, 2.0]), np.array([-3.0, 0.0, 4.0])],
        "post_contact_stabilize": [np.array([0.8, 1.2, -2.0]), np.array([1.2, 1.8, -3.0]), np.array([0.5, 0.8, -1.0])],
        "avoid_secondary": [np.array([3.5, -3.0, 8.0]), np.array([-3.5, -4.0, 12.0]), np.array([2.0, -2.0, 6.0])],
    }
    oid = 0
    while len(opts) < num_options:
        for mode in MODES:
            p = grids[mode][(oid // len(MODES)) % len(grids[mode])].astype(np.float32)
            valid = True
            if mode == "pull_over" and not shoulder_available:
                valid = False
            if mode in {"lateral_escape", "avoid_secondary"} and not adjacent_available:
                valid = False
            opts.append(RecoveryOption(oid, mode, p, valid))
            oid += 1
            if len(opts) >= num_options:
                break
    return opts


def option_valid_mask(options: list[RecoveryOption]) -> np.ndarray:
    return np.asarray([bool(o.valid) for o in options], dtype=bool)
