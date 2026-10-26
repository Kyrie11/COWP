from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cowp.external_baselines.rule_based import select_rule_indices
from cowp.waymax_eval.policy_wrapper import (
    _consistent_one_step_target,
    _extract_logged_future_agent_trajs,
    _extract_roadgraph_tokens,
    _wrap_angle,
    build_online_batch,
    extract_agent_history_model_state,
)


def _resolve_execution_trajectory(candidates: np.ndarray, valid: np.ndarray, selected: int, *, fallback_horizon: int, current: np.ndarray) -> tuple[int, np.ndarray, bool, str]:
    cand = np.asarray(candidates, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if 0 <= int(selected) < cand.shape[0] and int(selected) < mask.shape[0] and bool(mask[int(selected)]) and np.isfinite(cand[int(selected)]).all():
        return int(selected), np.nan_to_num(cand[int(selected)], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), False, "none"
    valid_idx = np.flatnonzero(mask[: cand.shape[0]])
    if valid_idx.size > 0:
        idx = int(valid_idx[0])
        return idx, np.nan_to_num(cand[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), True, "selected_invalid_use_first_valid_candidate"
    h = max(int(fallback_horizon), 1)
    out = np.zeros((h, 7), dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)
    out[:, :2] = cur[:2]
    out[:, 2] = cur[6] if cur.shape[0] > 6 else 0.0
    if cur.shape[0] > 5:
        out[:, 3:5] = cur[3:5]
        out[:, 5] = float(np.linalg.norm(cur[3:5]))
    out[:, 6] = 1.0
    return -1, out, True, "no_valid_candidate_emergency_stop"


@dataclass
class RuleBasedWaymaxPolicy:
    method: str
    cfg: dict
    action_mode: str = "delta_xy_yaw"
    require_conventional_safe: bool = True

    def __post_init__(self) -> None:
        self.baseline = str(self.method).lower()
        self._last_diagnostics: dict[str, Any] | None = None
        self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index: int | None = None

    def _trajectory_to_action(self, agent_state: np.ndarray, sdc_index: int, traj: np.ndarray) -> Any:
        try:
            from waymax import datatypes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes is required to convert selected rule-baseline trajectory to a Waymax action.") from exc
        n_agents = agent_state.shape[0]
        data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        if self.action_mode == "absolute_xy_yaw":
            data_dim = max(data_dim, 5)
        data = np.zeros((n_agents, data_dim), dtype=np.float32)
        valid = np.zeros((n_agents, 1), dtype=bool)
        valid[sdc_index, 0] = True
        desired = np.asarray(traj[0], dtype=np.float32)
        target, accel = _consistent_one_step_target(agent_state[sdc_index], desired, self.cfg, self._previous_longitudinal_accel)
        self._previous_longitudinal_accel = float(accel)
        if self.action_mode == "absolute_xy_yaw":
            data[sdc_index, :5] = target
        else:
            dx = float(target[0] - agent_state[sdc_index, 0])
            dy = float(target[1] - agent_state[sdc_index, 1])
            dyaw = float(_wrap_angle(float(target[2] - agent_state[sdc_index, 6])))
            data[sdc_index, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        try:
            import jax.numpy as jnp  # type: ignore
            return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))
        except Exception:
            return datatypes.Action(data=data, valid=valid)

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        if step == 0 or (scenario_index is not None and scenario_index != self._previous_scenario_index):
            self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index = scenario_index
        history, agent_state, sdc_index = extract_agent_history_model_state(state, self.cfg)
        roadgraph = _extract_roadgraph_tokens(state, self.cfg)
        other_future_trajs = _extract_logged_future_agent_trajs(state, sdc_index, self.cfg)
        batch_np = build_online_batch(
            agent_state,
            sdc_index,
            self.cfg,
            history_model_state=history,
            roadgraph=roadgraph,
            other_future_trajs=other_future_trajs,
        )
        selected_arr, accept, scores = select_rule_indices(
            batch_np,
            self.cfg,
            self.baseline,
            require_conventional_safe=self.require_conventional_safe,
        )
        selected = int(selected_arr[0]) if selected_arr.size else -1
        cand = np.asarray(batch_np["cowp/candidates/trajectory"][0], dtype=np.float32)
        valid = np.asarray(batch_np["cowp/candidates/valid"][0], dtype=bool)
        selected, traj, fallback_used, fallback_reason = _resolve_execution_trajectory(
            cand, valid, selected, fallback_horizon=int(self.cfg.get("time", {}).get("future_steps", 80)), current=agent_state[sdc_index]
        )
        conventional = np.asarray(batch_np.get("cowp/candidates/conventional_safe", valid[None])[0], dtype=bool)
        selected_valid = bool(0 <= selected < len(valid) and valid[selected])
        selected_conv = bool(0 <= selected < len(conventional) and conventional[selected])
        self._last_diagnostics = {
            "baseline": self.baseline,
            "selected_idx": selected,
            "valid_candidates": int(valid.sum()),
            "accepted_candidates": int(accept[0].sum()) if accept.ndim == 2 else 0,
            "conventional_candidates": int(conventional.sum()),
            "selected_score": float(scores[0, selected]) if selected >= 0 and scores.ndim == 2 else float("nan"),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "execution_trajectory_source": "candidate",
            "selected_candidate_valid": selected_valid,
            "selected_candidate_conventional_safe": selected_conv,
        }
        return self._trajectory_to_action(agent_state, sdc_index, traj)

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row


def make_rule_waymax_policy(method: str, cfg: dict, *, action_mode: str = "delta_xy_yaw", require_conventional_safe: bool = True):
    return RuleBasedWaymaxPolicy(method=method, cfg=cfg, action_mode=action_mode, require_conventional_safe=require_conventional_safe)
