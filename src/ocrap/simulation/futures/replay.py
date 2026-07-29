from __future__ import annotations

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, SceneHistory


def copy_future(history: SceneHistory, total_steps: int) -> tuple[np.ndarray, np.ndarray]:
    A = history.agent_history.shape[1]
    F = history.agent_history.shape[2]
    states = np.zeros((total_steps, A, F), dtype=np.float32)
    valid = np.zeros((total_steps, A), dtype=bool)
    avail = min(total_steps, history.future_agent_states.shape[0])
    if avail > 0:
        states[:avail] = history.future_agent_states[:avail]
        valid[:avail] = history.future_agent_valid[:avail].astype(bool)
    # Do not convert invalid tracks into valid objects. Only propagate already valid rows.
    for t in range(avail, total_steps):
        if avail > 0:
            states[t] = states[avail - 1]
            valid[t] = valid[avail - 1]
            dt = 0.1 * (t - avail + 1)
            states[t, valid[t], 0] += states[avail - 1, valid[t], 3] * dt
            states[t, valid[t], 1] += states[avail - 1, valid[t], 4] * dt
        else:
            states[t] = history.agent_history[-1]
            valid[t] = history.agent_valid[-1].astype(bool)
    return states, valid


def inject_ego_prefix(states: np.ndarray, valid: np.ndarray, prefix: CandidatePrefix) -> None:
    T = min(prefix.prefix_states.shape[0], states.shape[0])
    for t in range(T):
        e = prefix.prefix_states[t]
        states[t, 0, 0] = e[0]
        states[t, 0, 1] = e[1]
        states[t, 0, 2] = 0.0
        states[t, 0, 3] = e[2]
        states[t, 0, 4] = e[3]
        states[t, 0, 5] = 0.0
        states[t, 0, 6] = 0.0
        states[t, 0, 7] = e[4]
        states[t, 0, 8] = np.sin(e[4])
        states[t, 0, 9] = np.cos(e[4])
        states[t, 0, 10] = e[7]
        states[t, 0, 11] = e[8]
        states[t, 0, 12] = 1.5
        states[t, 0, 13] = 1.0
        states[t, 0, 14] = 1.0
        states[t, 0, 15] = 1.0
        valid[t, 0] = True


def replay_future(history: SceneHistory, prefix: CandidatePrefix, total_steps: int, prior: float = 0.25) -> CounterfactualFuture:
    states, valid = copy_future(history, total_steps)
    inject_ego_prefix(states, valid, prefix)
    mismatch = False
    T = min(prefix.prefix_states.shape[0], history.future_agent_states.shape[0])
    if T > 0:
        logged = history.future_agent_states[:T, 0, :2]
        mismatch = bool(np.max(np.linalg.norm(logged - prefix.prefix_states[:T, :2], axis=-1)) > 3.0)
    return CounterfactualFuture(0, "replay", prior, states, valid, {"anchor_logged": True, "replay_interaction_mismatch": mismatch, "runtime_backend": "ocrap_surrogate_replay", "waymax_runtime": False})
