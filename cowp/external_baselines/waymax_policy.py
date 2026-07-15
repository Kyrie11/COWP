from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from cowp.external_baselines.adapters import make_external_batch
from cowp.external_baselines.dtpp_cowp import COWPDTPP
from cowp.external_baselines.gameformer_cowp import COWPGameFormer
from cowp.waymax_eval.policy_wrapper import (
    _consistent_one_step_target,
    _extract_logged_future_agent_trajs,
    _extract_roadgraph_tokens,
    _wrap_angle,
    build_online_batch,
    extract_agent_history_model_state,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_external_model_from_checkpoint(checkpoint: str, cfg: dict, device: str = "auto"):
    dev = _device(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    baseline = str(ckpt.get("baseline", args.get("baseline", "gameformer")))
    model_cfg = ckpt.get("cfg", cfg)
    if baseline == "gameformer":
        model = COWPGameFormer(
            modalities=int(args.get("gameformer_modalities", 6)),
            neighbors_to_predict=int(args.get("max_neighbors", 10)),
            future_len=int(args.get("future_len", model_cfg.get("time", {}).get("future_steps", 80))),
            encoder_layers=int(args.get("gameformer_encoder_layers", 6)),
            decoder_levels=int(args.get("gameformer_decoder_levels", 4)),
        )
    elif baseline == "dtpp":
        model = COWPDTPP(
            neighbors=int(args.get("max_neighbors", 10)),
            max_branch=int(args.get("max_candidates", model_cfg.get("limits", {}).get("max_candidates", 30))),
            variable_cost=not bool(args.get("dtpp_fixed_cost", False)),
        )
    else:
        raise ValueError(f"Unsupported external baseline checkpoint: {baseline}")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(dev).eval()
    return model, baseline, model_cfg, args, dev


@dataclass
class ExternalWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    action_mode: str = "delta_xy_yaw"
    require_conventional_safe: bool = False

    def __post_init__(self) -> None:
        self.model, self.baseline, self.model_cfg, self.args, self.dev = build_external_model_from_checkpoint(self.checkpoint, self.cfg, self.device)
        self._last_diagnostics: dict[str, Any] | None = None
        self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index: int | None = None

    def _trajectory_to_action(self, agent_state: np.ndarray, sdc_index: int, traj: np.ndarray) -> Any:
        try:
            from waymax import datatypes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes is required to convert selected external-baseline trajectory to a Waymax action.") from exc
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
        batch_np = build_online_batch(agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph, other_future_trajs=other_future_trajs)
        batch = {k: torch.as_tensor(v, device=self.dev) for k, v in batch_np.items() if isinstance(v, np.ndarray)}
        max_neighbors = int(self.args.get("max_neighbors", 10))
        max_candidates = int(self.args.get("max_candidates", self.cfg.get("limits", {}).get("max_candidates", 30)))
        future_len = int(self.args.get("future_len", self.cfg.get("time", {}).get("future_steps", 80)))
        with torch.inference_mode():
            ext = make_external_batch(batch, self.model_cfg, device=self.dev, max_neighbors=max_neighbors, max_candidates=max_candidates, horizon=future_len)
            if self.baseline == "gameformer":
                scores = self.model.score_candidates(ext.gameformer_inputs, ext.candidates, ext.candidate_valid)[0]
            else:
                scores = self.model.score_candidates(ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, timesteps=future_len)[0]
            mask = ext.candidate_valid[0]
            if self.require_conventional_safe:
                mask = mask & ext.conventional_safe[0]
                if not bool(mask.any()):
                    mask = ext.candidate_valid[0]
            masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
            selected = int(torch.argmax(masked_scores).detach().cpu()) if bool(mask.any()) else -1
        cand = np.asarray(batch_np["cowp/candidates/trajectory"][0], dtype=np.float32)
        valid = np.asarray(batch_np["cowp/candidates/valid"][0], dtype=bool)
        if selected < 0 or selected >= len(cand) or not bool(valid[selected]):
            valid_idx = np.flatnonzero(valid)
            selected = int(valid_idx[0]) if valid_idx.size else 0
        self._last_diagnostics = {
            "baseline": self.baseline,
            "selected_idx": selected,
            "valid_candidates": int(valid.sum()),
            "conventional_candidates": int(np.asarray(batch_np.get("cowp/candidates/conventional_safe", valid[None])[0], dtype=bool).sum()) if "cowp/candidates/conventional_safe" in batch_np else int(valid.sum()),
            "selected_score": float(scores[selected].detach().cpu()) if selected >= 0 else float("nan"),
            "fallback": bool(selected < 0),
        }
        return self._trajectory_to_action(agent_state, sdc_index, cand[selected])

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row


def make_external_waymax_policy(checkpoint: str, cfg: dict, *, device: str = "auto", action_mode: str = "delta_xy_yaw", require_conventional_safe: bool = False):
    return ExternalWaymaxPolicy(checkpoint=checkpoint, cfg=cfg, device=device, action_mode=action_mode, require_conventional_safe=require_conventional_safe)
