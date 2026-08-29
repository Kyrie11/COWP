from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import torch

from cowp.external_baselines.adapters import make_external_batch
from cowp.external_baselines.dtpp_cowp import COWPDTPP
from cowp.external_baselines.gameformer_cowp import COWPGameFormer
from cowp.external_baselines.pluto_cowp import COWPPLUTO
from cowp.external_baselines.plant2_cowp import COWPPlanT2
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


def _get_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    v = args.get(key, default)
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def build_external_model_from_checkpoint(checkpoint: str, cfg: dict, device: str = "auto"):
    dev = _device(device)
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    baseline = str(ckpt.get("baseline", args.get("baseline", "gameformer"))).lower()
    model_cfg = ckpt.get("cfg", cfg)
    future_len = int(args.get("future_len", model_cfg.get("time", {}).get("future_steps", 80)))
    if baseline == "gameformer":
        model = COWPGameFormer(
            modalities=int(args.get("gameformer_modalities", 6)),
            neighbors_to_predict=int(args.get("max_neighbors", 10)),
            future_len=future_len,
            encoder_layers=int(args.get("gameformer_encoder_layers", 6)),
            decoder_levels=int(args.get("gameformer_decoder_levels", 4)),
        )
    elif baseline == "dtpp":
        variable_cost = _get_bool(args, "dtpp_variable_cost", False) and not _get_bool(args, "dtpp_fixed_cost", False)
        model = COWPDTPP(
            neighbors=int(args.get("max_neighbors", 10)),
            max_branch=int(args.get("max_candidates", model_cfg.get("limits", {}).get("max_candidates", 30))),
            variable_cost=variable_cost,
        )
    elif baseline == "pluto":
        model = COWPPLUTO(
            future_len=future_len,
            d_model=int(args.get("pluto_d_model", 128)),
            num_heads=int(args.get("pluto_num_heads", 8)),
            encoder_layers=int(args.get("pluto_encoder_layers", 4)),
            lateral_queries=int(args.get("pluto_lateral_queries", 4)),
            longitudinal_queries=int(args.get("pluto_longitudinal_queries", 6)),
        )
    elif baseline == "plant2":
        model = COWPPlanT2(
            future_len=future_len,
            d_model=int(args.get("plant2_d_model", 128)),
            num_heads=int(args.get("plant2_num_heads", 8)),
            layers=int(args.get("plant2_layers", 4)),
        )
    else:
        raise ValueError(f"Unsupported external baseline checkpoint: {baseline}")
    state = ckpt.get("model", ckpt if isinstance(ckpt, dict) else {})
    model.load_state_dict(state, strict=False)
    model.to(dev).eval()
    return model, baseline, model_cfg, args, dev


def _local_xy_to_global_traj(local_xy: np.ndarray, origin: np.ndarray, yaw0: float, dt: float = 0.1) -> np.ndarray:
    xy = np.asarray(local_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[-1] < 2:
        raise ValueError(f"local_xy must be [T,2], got shape {xy.shape}")
    xy = np.nan_to_num(xy[:, :2], nan=0.0, posinf=0.0, neginf=0.0)
    origin = np.asarray(origin, dtype=np.float32).reshape(2)
    c, s = float(np.cos(float(yaw0))), float(np.sin(float(yaw0)))
    gx = origin[0] + c * xy[:, 0] - s * xy[:, 1]
    gy = origin[1] + s * xy[:, 0] + c * xy[:, 1]
    gxy = np.stack([gx, gy], axis=-1).astype(np.float32)
    prev = np.concatenate([origin.reshape(1, 2), gxy[:-1]], axis=0)
    vel = (gxy - prev) / max(float(dt), 1e-6)
    speed = np.linalg.norm(vel, axis=-1)
    yaw = np.full((gxy.shape[0],), float(yaw0), dtype=np.float32)
    moving = speed > 1.0e-3
    yaw[moving] = np.arctan2(vel[moving, 1], vel[moving, 0]).astype(np.float32)
    out = np.zeros((gxy.shape[0], 7), dtype=np.float32)
    out[:, :2] = gxy
    out[:, 2] = yaw
    out[:, 3:5] = vel.astype(np.float32)
    out[:, 5] = speed.astype(np.float32)
    out[:, 6] = 1.0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)




def _resolve_execution_trajectory(candidates: np.ndarray, valid: np.ndarray, selected: int, *, fallback_horizon: int, current: np.ndarray) -> tuple[int, np.ndarray, bool, str]:
    cand = np.asarray(candidates, dtype=np.float32)
    if cand.ndim == 2:
        cand = cand[None]
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if cand.ndim == 3 and cand.shape[0] and 0 <= int(selected) < cand.shape[0] and int(selected) < mask.shape[0] and bool(mask[int(selected)]) and np.isfinite(cand[int(selected)]).all():
        return int(selected), np.nan_to_num(cand[int(selected)], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), False, "none"
    valid_idx = np.flatnonzero(mask[: cand.shape[0]]) if cand.ndim == 3 else np.asarray([], dtype=np.int64)
    if valid_idx.size > 0:
        idx = int(valid_idx[0])
        return idx, np.nan_to_num(cand[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), True, "selected_invalid_use_first_valid_candidate"
    h = max(int(fallback_horizon), 1)
    out = np.zeros((h, 7), dtype=np.float32)
    cur = np.asarray(current, dtype=np.float32)
    out[:, :2] = cur[:2] if cur.shape[0] >= 2 and np.isfinite(cur[:2]).all() else 0.0
    out[:, 2] = cur[6] if cur.shape[0] > 6 and np.isfinite(cur[6]) else 0.0
    if cur.shape[0] > 8 and np.isfinite(cur[7:9]).all():
        out[:, 3:5] = cur[7:9]
        out[:, 5] = float(np.linalg.norm(cur[7:9]))
    elif cur.shape[0] > 5 and np.isfinite(cur[3:5]).all():
        out[:, 3:5] = cur[3:5]
        out[:, 5] = float(np.linalg.norm(cur[3:5]))
    out[:, 6] = 1.0
    return -1, out, True, "no_valid_candidate_emergency_stop"


def _minimal_external_online_batch(
    history_model_state: np.ndarray,
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray] | None,
    other_future_trajs: np.ndarray | None,
    cfg: dict,
) -> dict[str, Any]:
    """Small test/debug wrapper around the production online batch builder."""
    return build_online_batch(
        agent_state,
        int(sdc_index),
        cfg,
        history_model_state=np.asarray(history_model_state, dtype=np.float32),
        roadgraph=roadgraph,
        other_future_trajs=other_future_trajs,
    )


@dataclass
class ExternalWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    action_mode: str = "delta_xy_yaw"
    require_conventional_safe: bool = False
    execution_mode: str = "auto"
    profile_policy_runtime: bool = False
    profile_policy_runtime_sync: bool = False

    def __post_init__(self) -> None:
        self.model, self.baseline, self.model_cfg, self.args, self.dev = build_external_model_from_checkpoint(self.checkpoint, self.cfg, self.device)
        self._last_diagnostics: dict[str, Any] | None = None
        self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index: int | None = None

    @staticmethod
    def _local_xy_to_global_traj(local_xy: np.ndarray, origin: np.ndarray, yaw0: float, dt: float = 0.1) -> np.ndarray:
        return _local_xy_to_global_traj(local_xy, origin, yaw0, dt)

    def _sync_if_needed(self) -> None:
        if self.profile_policy_runtime_sync and self.dev.type == "cuda":
            try:
                torch.cuda.synchronize(self.dev)
            except Exception:
                pass

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

    def _direct_baseline_requested(self) -> bool:
        mode = str(self.execution_mode or "auto").lower()
        if mode == "candidate_tree":
            return False
        if mode == "direct":
            return self.baseline in {"gameformer", "pluto", "plant2"}
        return self.baseline in {"gameformer", "pluto", "plant2"}

    def _predict_direct_local_xy(self, ext) -> np.ndarray | None:
        with torch.inference_mode():
            if self.baseline == "gameformer":
                outputs = self.model(ext.gameformer_inputs)
                trajs, mode_scores = self.model.final_level(outputs)
                ego_modes = trajs[:, 0, :, :, :2]
                idx = torch.argmax(mode_scores[:, 0, :], dim=-1)
                pred = ego_modes[torch.arange(idx.shape[0], device=idx.device), idx][0]
            elif self.baseline in {"pluto", "plant2"}:
                pred = self.model.predict_trajectory(ext.planner_inputs)[0]
            else:
                return None
        arr = pred.detach().float().cpu().numpy()
        if arr.ndim != 2 or arr.shape[0] == 0 or not np.isfinite(arr[:, :2]).all():
            return None
        return arr[:, :2].astype(np.float32)

    def _score_candidate_index(self, ext) -> tuple[int, torch.Tensor | None, torch.Tensor]:
        with torch.inference_mode():
            if self.baseline == "gameformer":
                scores = self.model.score_candidates(ext.gameformer_inputs, ext.candidates, ext.candidate_valid)[0]
            elif self.baseline == "dtpp":
                scores = self.model.score_candidates(ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, timesteps=int(self.args.get("future_len", self.cfg.get("time", {}).get("future_steps", 80))))[0]
            elif self.baseline in {"pluto", "plant2"}:
                scores = self.model.score_candidates(ext.planner_inputs, ext.candidates, ext.candidate_valid)[0]
            else:
                raise ValueError(f"Unsupported external baseline: {self.baseline}")
            mask = ext.candidate_valid[0]
            if self.require_conventional_safe and mask.numel() > 0:
                conv = mask & ext.conventional_safe[0]
                if bool(conv.any()):
                    mask = conv
            masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9)) if mask.numel() else scores
            selected = int(torch.argmax(masked_scores).detach().cpu()) if bool(mask.any()) else -1
        return selected, scores, mask

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        t_policy0 = time.perf_counter()
        if step == 0 or (scenario_index is not None and scenario_index != self._previous_scenario_index):
            self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index = scenario_index
        t0 = time.perf_counter()
        history, agent_state, sdc_index = extract_agent_history_model_state(state, self.cfg)
        roadgraph = _extract_roadgraph_tokens(state, self.cfg)
        other_future_trajs = _extract_logged_future_agent_trajs(state, sdc_index, self.cfg)
        batch_np = build_online_batch(agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph, other_future_trajs=other_future_trajs)
        t_state = time.perf_counter() - t0
        t0 = time.perf_counter()
        batch = {k: torch.as_tensor(v, device=self.dev) for k, v in batch_np.items() if isinstance(v, np.ndarray) and v.dtype.kind in "fbiu"}
        max_neighbors = int(self.args.get("max_neighbors", 10))
        max_candidates = int(self.args.get("max_candidates", self.cfg.get("limits", {}).get("max_candidates", 30)))
        future_len = int(self.args.get("future_len", self.cfg.get("time", {}).get("future_steps", 80)))
        direct = self._direct_baseline_requested()
        t_h2d = time.perf_counter() - t0
        t0 = time.perf_counter()
        ext = make_external_batch(
            batch,
            self.model_cfg,
            device=self.dev,
            max_neighbors=max_neighbors,
            max_candidates=max_candidates,
            horizon=future_len,
            baseline=self.baseline,
            require_candidates=not direct,
            require_future=False,
        )
        t_adapter = time.perf_counter() - t0
        t0 = time.perf_counter()
        selected = -1
        selected_score = float("nan")
        mask_np = np.asarray([], dtype=bool)
        source = "direct" if direct else "candidate"
        fallback_used = False
        fallback_reason = "none"
        if direct:
            self._sync_if_needed()
            local_xy = self._predict_direct_local_xy(ext)
            self._sync_if_needed()
            if local_xy is not None:
                traj = _local_xy_to_global_traj(local_xy, ext.origin[0].detach().cpu().numpy(), float(ext.yaw0[0].detach().cpu()), float(self.cfg.get("time", {}).get("dt", 0.1)))
            else:
                source = "candidate"
                fallback_used = True
                fallback_reason = "direct_prediction_invalid_use_candidate"
        if not direct or source == "candidate":
            self._sync_if_needed()
            selected, scores, mask = self._score_candidate_index(ext)
            self._sync_if_needed()
            mask_np = mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(mask) else np.asarray([], dtype=bool)
            cand_world = np.asarray(batch_np.get("cowp/candidates/trajectory", np.zeros((1, 0, future_len, 7), dtype=np.float32))[0], dtype=np.float32)
            valid_world = np.asarray(batch_np.get("cowp/candidates/valid", np.zeros((1, 0), dtype=bool))[0], dtype=bool)
            candidate_finite = np.isfinite(cand_world.reshape(cand_world.shape[0], -1)).all(axis=-1) if cand_world.size else np.zeros((0,), dtype=bool)
            adapter_valid = ext.candidate_valid[0].detach().cpu().numpy().astype(bool) if ext.candidate_valid.numel() else np.zeros_like(valid_world, dtype=bool)
            if adapter_valid.shape[0] < valid_world.shape[0]:
                adapter_valid = np.pad(adapter_valid, (0, valid_world.shape[0] - adapter_valid.shape[0]), constant_values=False)
            valid = valid_world[: candidate_finite.shape[0]]
            valid = valid & candidate_finite & adapter_valid[: valid.shape[0]]
            selected, traj, used_resolver_fallback, resolver_reason = _resolve_execution_trajectory(
                cand_world, valid, selected, fallback_horizon=future_len, current=agent_state[sdc_index]
            )
            fallback_used = bool(fallback_used or used_resolver_fallback)
            if used_resolver_fallback:
                fallback_reason = resolver_reason
            if scores is not None and selected >= 0 and selected < int(scores.shape[0]):
                selected_score = float(scores[selected].detach().cpu())
        t_model = time.perf_counter() - t0
        t0 = time.perf_counter()
        action = self._trajectory_to_action(agent_state, sdc_index, traj)
        t_action = time.perf_counter() - t0
        valid = np.asarray(batch_np.get("cowp/candidates/valid", np.zeros((1, 0), dtype=bool))[0], dtype=bool)
        conventional = np.asarray(batch_np.get("cowp/candidates/conventional_safe", valid[None])[0], dtype=bool) if valid.size else np.asarray([], dtype=bool)
        selected_valid = bool(0 <= selected < len(valid) and valid[selected]) if source != "direct" else True
        selected_conv = bool(0 <= selected < len(conventional) and conventional[selected]) if source != "direct" else True
        direct_rate = 1.0 if source == "direct" else 0.0
        self._last_diagnostics = {
            "baseline": self.baseline,
            "selected_idx": int(selected),
            "valid_candidates": int(valid.sum()) if valid.size else int(ext.candidate_valid[0].sum().detach().cpu()) if ext.candidate_valid.numel() else 0,
            "accepted_candidates": int(mask_np.sum()) if mask_np.size else int(valid.sum()) if valid.size else 0,
            "conventional_candidates": int(conventional.sum()) if conventional.size else int(valid.sum()) if valid.size else 0,
            "selected_score": selected_score,
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "execution_trajectory_source": source,
            "direct_execution": bool(source == "direct"),
            "direct_execution_step_rate": direct_rate,
            "selected_candidate_valid": selected_valid,
            "selected_candidate_conventional_safe": selected_conv,
            "runtime_state_extract_map_s": float(t_state),
            "runtime_candidate_build_cpu_s": float(t_adapter),
            "runtime_h2d_s": float(t_h2d),
            "runtime_model_forward_s": float(t_model),
            "runtime_selection_s": float(t_model),
            "runtime_action_projection_s": float(t_action),
            "runtime_policy_total_s": float(time.perf_counter() - t_policy0),
        }
        return action

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row


def make_external_waymax_policy(
    checkpoint: str,
    cfg: dict,
    *,
    device: str = "auto",
    action_mode: str = "delta_xy_yaw",
    require_conventional_safe: bool = False,
    execution_mode: str = "auto",
    profile_policy_runtime: bool = False,
    profile_policy_runtime_sync: bool = False,
):
    return ExternalWaymaxPolicy(
        checkpoint=checkpoint,
        cfg=cfg,
        device=device,
        action_mode=action_mode,
        require_conventional_safe=require_conventional_safe,
        execution_mode=execution_mode,
        profile_policy_runtime=profile_policy_runtime,
        profile_policy_runtime_sync=profile_policy_runtime_sync,
    )
