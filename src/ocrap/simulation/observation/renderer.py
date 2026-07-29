from __future__ import annotations

import numpy as np

from ocrap.data.schema import CounterfactualFuture, Observation, SceneHistory, CandidatePrefix
from ocrap.utils.geometry import ego_state_to_box

from .bev import render_base_occ_mask
from .visibility import visible_agent_boxes


def render_observation(history: SceneHistory, prefix: CandidatePrefix, future: CounterfactualFuture, cfg: dict) -> Observation:
    T_p = prefix.prefix_states.shape[0]
    idx = min(max(T_p - 1, 0), future.agent_states.shape[0] - 1)
    states = future.agent_states[idx].copy()
    valid = future.agent_valid[idx].copy()
    # Ego state is the physical prefix endpoint; no future branch id or hidden true position is exposed.
    ego = prefix.prefix_states[-1].astype(np.float32)
    states[0, 0] = ego[0]
    states[0, 1] = ego[1]
    states[0, 3] = ego[2]
    states[0, 4] = ego[3]
    states[0, 7] = ego[4]
    states[0, 8] = np.sin(ego[4])
    states[0, 9] = np.cos(ego[4])
    valid[0] = True
    boxes, box_valid, _ = visible_agent_boxes(states, valid, ego, cfg)
    # Re-render visible occupancy from history and prefix endpoint; branch-specific hidden ids are not encoded.
    obs_mask = render_base_occ_mask(history, cfg).copy()
    contact_flag = bool(future.metadata.get("contact_surrogate", False) and prefix.diagnostics.get("prefix_contact", False))
    stability_proxy = np.array([float(future.metadata.get("yaw_rate_impulse", 0.0)), float(future.metadata.get("lateral_velocity_impulse", 0.0)), float(contact_flag)], dtype=np.float32)
    route_visible = history.route[:, :2].astype(np.float32) if history.route.size else np.zeros((0, 2), dtype=np.float32)
    return Observation(ego_state=ego, boxes=boxes, box_valid=box_valid, occ_mask=obs_mask, contact_flag=contact_flag, stability_proxy=stability_proxy, route_visible=route_visible)
