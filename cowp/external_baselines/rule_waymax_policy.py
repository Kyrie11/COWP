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
        if selected < 0 or selected >= len(cand) or not bool(valid[selected]):
            valid_idx = np.flatnonzero(valid)
            selected = int(valid_idx[0]) if valid_idx.size else 0
        conventional = np.asarray(batch_np.get("cowp/candidates/conventional_safe", valid[None])[0], dtype=bool)
        self._last_diagnostics = {
            "baseline": self.baseline,
            "selected_idx": selected,
            "valid_candidates": int(valid.sum()),
            "accepted_candidates": int(accept[0].sum()) if accept.ndim == 2 else 0,
            "conventional_candidates": int(conventional.sum()),
            "selected_score": float(scores[0, selected]) if selected >= 0 and scores.ndim == 2 else float("nan"),
            "fallback": bool(selected < 0),
        }
        return self._trajectory_to_action(agent_state, sdc_index, cand[selected])

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row


def make_rule_waymax_policy(method: str, cfg: dict, *, action_mode: str = "delta_xy_yaw", require_conventional_safe: bool = True):
    return RuleBasedWaymaxPolicy(method=method, cfg=cfg, action_mode=action_mode, require_conventional_safe=require_conventional_safe)
