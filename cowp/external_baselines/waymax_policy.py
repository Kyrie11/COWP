from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from cowp.external_baselines.adapters import make_external_batch
from cowp.external_baselines.dtpp_cowp import COWPDTPP
from cowp.external_baselines.gameformer_cowp import COWPGameFormer
from cowp.external_baselines.pluto_cowp import COWPPLUTO
from cowp.external_baselines.plant2_cowp import COWPPlanT2
from cowp.external_baselines.training_contract import EXTERNAL_TRAINING_CONTRACT_VERSION
from cowp.waymax_eval.policy_wrapper import (
    _consistent_one_step_target,
    _resolve_execution_trajectory,
    _extract_roadgraph_tokens,
    _extract_sdc_path_tokens,
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
    try:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    baseline = str(ckpt.get("baseline", args.get("baseline", "gameformer")))
    contract = ckpt.get("training_contract_version") if isinstance(ckpt, dict) else None
    allow_legacy = str(__import__("os").environ.get("ALLOW_LEGACY_EXTERNAL_CHECKPOINT", "0")).strip().lower() in {"1", "true", "yes"}
    if contract != EXTERNAL_TRAINING_CONTRACT_VERSION and not allow_legacy:
        raise RuntimeError(
            f"External baseline checkpoint contract={contract!r} is incompatible with "
            f"required contract={EXTERNAL_TRAINING_CONTRACT_VERSION!r}: {checkpoint}. "
            "Rerun RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all so the checkpoint is overwritten at the same path. "
            "Set ALLOW_LEGACY_EXTERNAL_CHECKPOINT=1 only for explicit historical auditing, not publication evaluation."
        )
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
        # New checkpoints record the explicit opt-in.  For pre-V3 checkpoints,
        # preserve their historical --dtpp-fixed-cost semantics so loading is
        # backward compatible and never silently changes a stored experiment.
        if "dtpp_variable_cost" in args:
            variable_cost = bool(args.get("dtpp_variable_cost", False))
        elif "dtpp_fixed_cost" in args:
            variable_cost = not bool(args.get("dtpp_fixed_cost", False))
        else:
            variable_cost = False
        model = COWPDTPP(
            neighbors=int(args.get("max_neighbors", 10)),
            max_branch=int(args.get("max_candidates", model_cfg.get("limits", {}).get("max_candidates", 30))),
            variable_cost=variable_cost,
        )
    elif baseline == "pluto":
        model = COWPPLUTO(
            future_len=int(args.get("future_len", model_cfg.get("time", {}).get("future_steps", 80))),
            d_model=int(args.get("vector_d_model", 128)), num_heads=int(args.get("vector_heads", 8)),
            encoder_layers=int(args.get("vector_layers", 4)),
            lateral_queries=int(args.get("pluto_lateral_queries", 4)),
            longitudinal_queries=int(args.get("pluto_longitudinal_queries", 6)),
            dropout=float(args.get("vector_dropout", 0.1)),
        )
    elif baseline == "plant2":
        model = COWPPlanT2(
            future_len=int(args.get("future_len", model_cfg.get("time", {}).get("future_steps", 80))),
            d_model=int(args.get("vector_d_model", 128)), num_heads=int(args.get("vector_heads", 8)),
            layers=int(args.get("vector_layers", 4)), dropout=float(args.get("vector_dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported external baseline checkpoint: {baseline}")
    # A baseline benchmark must never run with silently missing/random/corrupt weights.
    state_dict = ckpt["model"]
    for name, tensor in state_dict.items():
        if torch.is_tensor(tensor) and tensor.numel() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"Checkpoint contains non-finite tensor {name}: {checkpoint}")
    model.load_state_dict(state_dict, strict=True)
    model.to(dev).eval()
    return model, baseline, model_cfg, args, dev


def _minimal_external_online_batch(
    history: np.ndarray,
    agent_state: np.ndarray,
    sdc_index: int,
    roadgraph: dict[str, np.ndarray],
    sdc_paths: dict[str, np.ndarray] | None,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Build only causal observation tensors required by direct external planners.

    This deliberately does *not* generate the COWP proposal bank, critical-agent
    ranking, witness/conflict tokens, or any logged-future tensors.
    """
    max_agents = int(cfg.get("limits", {}).get("max_agents", cfg.get("model", {}).get("max_agents", 128)))
    n = min(int(agent_state.shape[0]), max_agents)
    mask = np.zeros(max_agents, dtype=bool)
    mask[:n] = agent_state[:n, 10] > 0.5
    if 0 <= int(sdc_index) < max_agents:
        mask[int(sdc_index)] = True
    is_sdc = np.zeros(max_agents, dtype=bool)
    if 0 <= int(sdc_index) < max_agents:
        is_sdc[int(sdc_index)] = True
    batch: dict[str, np.ndarray] = {
        "state/history": np.asarray(history, dtype=np.float32)[None],
        "state/agent_valid": mask[None],
        "state/is_sdc": is_sdc[None],
    }
    xy = np.asarray(roadgraph.get("xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    valid = np.asarray(roadgraph.get("valid", np.zeros(len(xy), dtype=bool)), dtype=bool)
    if len(xy):
        xyz = np.zeros((len(xy), 3), dtype=np.float32)
        xyz[:, :2] = xy
        batch["roadgraph_samples/xyz"] = xyz[None]
        batch["roadgraph_samples/valid"] = valid[None]
    if sdc_paths is not None:
        pxyz = np.asarray(sdc_paths.get("xyz", np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
        pvalid = np.asarray(sdc_paths.get("valid", np.zeros(pxyz.shape[:2], dtype=bool)), dtype=bool)
        pon = np.asarray(sdc_paths.get("on_route", np.zeros((pxyz.shape[0], 1), dtype=bool)), dtype=bool)
        if pxyz.ndim == 3 and pxyz.shape[0] > 0:
            batch["path_samples/xyz"] = pxyz[None]
            batch["path_samples/valid"] = pvalid[None]
            batch["path_samples/on_route"] = pon[None]
    return batch


@dataclass
class ExternalWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    action_mode: str = "delta_xy_yaw"
    require_conventional_safe: bool = False
    execution_mode: str = "auto"
    profile_timing: bool = False

    def __post_init__(self) -> None:
        self.model, self.baseline, self.model_cfg, self.args, self.dev = build_external_model_from_checkpoint(self.checkpoint, self.cfg, self.device)
        self._last_diagnostics: dict[str, Any] | None = None
        self._previous_longitudinal_accel = 0.0
        self._previous_scenario_index: int | None = None
        # Roadgraph and WOMD 1.3.1 SDC paths are scenario-static.  Cache their
        # host representations so a closed-loop rollout does not copy tens of
        # thousands of JAX values to NumPy again at every 100 ms step.
        self._cached_roadgraph: dict[str, np.ndarray] | None = None
        self._cached_sdc_paths: dict[str, np.ndarray] | None = None
        if self.execution_mode not in {"auto", "direct", "candidate"}:
            raise ValueError(f"Unsupported external execution_mode={self.execution_mode!r}")

    @staticmethod
    def _local_xy_to_global_traj(local_xy: np.ndarray, origin: np.ndarray, yaw0: float, dt: float = 0.1) -> np.ndarray:
        """Convert an ego-frame waypoint sequence to the global 7-D trajectory contract."""
        xy = np.asarray(local_xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[-1] != 2:
            raise ValueError(f"Expected local_xy [T,2], got {xy.shape}")
        if not np.isfinite(xy).all():
            raise FloatingPointError("External planner produced non-finite local waypoints")
        c, s = float(np.cos(yaw0)), float(np.sin(yaw0))
        gx = origin[0] + c * xy[:, 0] - s * xy[:, 1]
        gy = origin[1] + s * xy[:, 0] + c * xy[:, 1]
        gxy = np.stack([gx, gy], axis=-1).astype(np.float32)
        prev = np.concatenate([np.asarray(origin, dtype=np.float32)[None, :2], gxy[:-1]], axis=0)
        vel = (gxy - prev) / max(float(dt), 1.0e-6)
        speed = np.linalg.norm(vel, axis=-1)
        yaw = np.full((gxy.shape[0],), float(yaw0), dtype=np.float32)
        moving = speed > 1.0e-3
        yaw[moving] = np.arctan2(vel[moving, 1], vel[moving, 0]).astype(np.float32)
        # Hold the last meaningful heading through predicted stops.
        for t in range(1, len(yaw)):
            if not moving[t]:
                yaw[t] = yaw[t - 1]
        out = np.zeros((gxy.shape[0], 7), dtype=np.float32)
        out[:, :2] = gxy
        out[:, 2] = yaw
        out[:, 3:5] = vel
        if not np.isfinite(out).all():
            raise FloatingPointError("Globalized external-planner trajectory became non-finite")
        return out

    def _direct_local_xy(self, ext) -> tuple[np.ndarray | None, float | None, str]:
        """Return the source planner's own direct ego trajectory when it has one."""
        if self.baseline == "gameformer":
            outputs = self.model(ext.gameformer_inputs)
            trajs, scores = self.model.final_level(outputs)
            row = scores[0, 0].float()
            finite = torch.isfinite(row)
            if not bool(finite.any()):
                return None, None, "no_finite_mode_score"
            safe_row = torch.where(finite, row, torch.full_like(row, -torch.inf))
            mode = int(torch.argmax(safe_row).detach().cpu())
            xy_t = trajs[0, 0, mode, :, :2].float()
            if not bool(torch.isfinite(xy_t).all()):
                return None, float(row[mode].detach().cpu()), "nonfinite_mode_trajectory"
            xy = xy_t.detach().cpu().numpy()
            return xy, float(row[mode].detach().cpu()), f"mode_{mode}"
        if self.baseline == "pluto":
            outputs = self.model(ext.planner_inputs)
            row = outputs["scores"][0].float()
            finite = torch.isfinite(row)
            if not bool(finite.any()):
                return None, None, "no_finite_mode_score"
            safe_row = torch.where(finite, row, torch.full_like(row, -torch.inf))
            mode = int(torch.argmax(safe_row).detach().cpu())
            xy_t = outputs["trajectories"][0, mode].float()
            if not bool(torch.isfinite(xy_t).all()):
                return None, float(row[mode].detach().cpu()), "nonfinite_mode_trajectory"
            xy = xy_t.detach().cpu().numpy()
            return xy, float(row[mode].detach().cpu()), f"mode_{mode}"
        if self.baseline == "plant2":
            outputs = self.model(ext.planner_inputs)
            xy_t = outputs["trajectory"][0].float()
            if not bool(torch.isfinite(xy_t).all()):
                return None, None, "nonfinite_autoregressive_waypoints"
            xy = xy_t.detach().cpu().numpy()
            return xy, None, "autoregressive_waypoints"
        return None, None, "candidate_tree_required"

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
        if not np.isfinite(desired).all():
            raise FloatingPointError("Selected external-baseline trajectory begins with NaN/Inf")
        target, accel = _consistent_one_step_target(agent_state[sdc_index], desired, self.cfg, self._previous_longitudinal_accel)
        if not np.isfinite(target).all() or not np.isfinite(accel):
            raise FloatingPointError("External-baseline one-step target became NaN/Inf")
        self._previous_longitudinal_accel = float(accel)
        if self.action_mode == "absolute_xy_yaw":
            data[sdc_index, :5] = target
        else:
            dx = float(target[0] - agent_state[sdc_index, 0])
            dy = float(target[1] - agent_state[sdc_index, 1])
            dyaw = float(_wrap_angle(float(target[2] - agent_state[sdc_index, 6])))
            data[sdc_index, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        if not np.isfinite(data).all():
            raise FloatingPointError("Refusing to hide a non-finite Waymax action with nan_to_num")
        data = data.astype(np.float32, copy=False)
        try:
            import jax.numpy as jnp  # type: ignore
            return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))
        except Exception:
            return datatypes.Action(data=data, valid=valid)

    def _timing_sync(self) -> None:
        if self.profile_timing and self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        if self.profile_timing:
            self._timing_sync()
            _t_total0 = time.perf_counter()
        new_scenario = bool(step == 0 or (scenario_index is not None and scenario_index != self._previous_scenario_index))
        if new_scenario:
            self._previous_longitudinal_accel = 0.0
            self._cached_roadgraph = None
            self._cached_sdc_paths = None
        self._previous_scenario_index = scenario_index
        history, agent_state, sdc_index = extract_agent_history_model_state(state, self.cfg)
        if self.profile_timing:
            _t_after_obs = time.perf_counter()
            _t_map0 = _t_after_obs
        if self._cached_roadgraph is None:
            self._cached_roadgraph = _extract_roadgraph_tokens(state, self.cfg)
        roadgraph = self._cached_roadgraph
        if self.profile_timing:
            _t_after_map = time.perf_counter()
        max_neighbors = int(self.args.get("max_neighbors", 10))
        max_candidates = int(self.args.get("max_candidates", self.cfg.get("limits", {}).get("max_candidates", 30)))
        future_len = int(self.args.get("future_len", self.cfg.get("time", {}).get("future_steps", 80)))
        resolved_mode = self.execution_mode
        if resolved_mode == "auto":
            resolved_mode = "direct" if self.baseline in {"gameformer", "pluto", "plant2"} else "candidate"

        # Direct trajectory planners do not need COWP proposal generation.  Use
        # WOMD 1.3.1 SDC paths directly for route conditioning instead.
        if resolved_mode == "direct":
            if self._cached_sdc_paths is None:
                self._cached_sdc_paths = _extract_sdc_path_tokens(state, self.cfg)
            sdc_paths = self._cached_sdc_paths
            if self.profile_timing:
                _t_direct0 = time.perf_counter()
                self._timing_sync()
            batch_np = _minimal_external_online_batch(history, agent_state, sdc_index, roadgraph, sdc_paths, self.cfg)
            batch = {k: torch.as_tensor(v) for k, v in batch_np.items() if isinstance(v, np.ndarray)}
            with torch.inference_mode():
                ext = make_external_batch(
                    batch, self.model_cfg, device=self.dev, max_neighbors=max_neighbors,
                    max_candidates=max_candidates, horizon=future_len, baseline=self.baseline,
                    require_candidates=False, require_future=False,
                )
                if bool(ext.sdc_current_valid[0]):
                    direct_xy, direct_score, direct_source = self._direct_local_xy(ext)
                else:
                    direct_xy, direct_score, direct_source = None, None, "invalid_sdc_observation"
            if self.profile_timing:
                self._timing_sync()
                _t_after_direct = time.perf_counter()
            if direct_xy is not None:
                try:
                    traj = self._local_xy_to_global_traj(
                        direct_xy,
                        ext.origin[0].detach().float().cpu().numpy(),
                        float(ext.yaw0[0].detach().cpu()),
                        dt=float(self.cfg.get("time", {}).get("dt", 0.1)),
                    )
                except (FloatingPointError, ValueError, OverflowError):
                    # A finite tensor can still overflow while globalizing very
                    # large waypoints.  Fall through to the already-causal
                    # candidate/emergency path instead of aborting the rollout.
                    direct_xy = None
                    direct_score = None
                    direct_source = "invalid_globalized_direct_trajectory"
            if direct_xy is not None:
                self._last_diagnostics = {
                    "baseline": self.baseline,
                    "execution_mode": "direct",
                    "direct_source": direct_source,
                    "selected_idx": -1,
                    "valid_candidates": 0,
                    "conventional_candidates": 0,
                    "selected_score": direct_score,
                    "fallback": False,
                    "cowp_candidate_filter_applied": False,
                    "uses_waymax_sdc_paths": bool("path_samples/xyz" in batch_np),
                    "optimized_direct_observation_path": True,
                    "scenario_static_map_cache": True,
                    "causal_no_logged_future": True,
                }
                if self.profile_timing:
                    self._last_diagnostics.update({
                        "timing_ms/observation": 1000.0 * (_t_after_obs - _t_total0),
                        "timing_ms/static_map_host": 1000.0 * (_t_after_map - _t_map0),
                        "timing_ms/adapter_and_model": 1000.0 * (_t_after_direct - _t_direct0),
                        "timing_ms/total_before_action": 1000.0 * (time.perf_counter() - _t_total0),
                    })
                return self._trajectory_to_action(agent_state, sdc_index, traj)
            # DTPP normally uses candidate mode directly.  A direct planner also
            # arrives here when its native trajectory is unusable; preserve the
            # same causal proposal/emergency execution fallback.
            resolved_mode = "candidate"

        # Closed-loop planners must not read the future portion of Waymax's
        # logged trajectory.  ``other_future_trajs=None`` makes the shared
        # proposal generator use causal current-state/history extrapolation.
        if self.profile_timing:
            _t_prop0 = time.perf_counter()
        batch_np = build_online_batch(
            agent_state, sdc_index, self.cfg, history_model_state=history, roadgraph=roadgraph,
            other_future_trajs=None, compute_rule_risk=False,
            include_interaction_tokens=False,
        )
        if self.profile_timing:
            _t_after_prop = time.perf_counter()
            self._timing_sync()
            _t_model0 = time.perf_counter()
        batch = {k: torch.as_tensor(v) for k, v in batch_np.items() if isinstance(v, np.ndarray)}
        with torch.inference_mode():
            ext = make_external_batch(
                batch, self.model_cfg, device=self.dev, max_neighbors=max_neighbors, max_candidates=max_candidates,
                horizon=future_len, baseline=self.baseline, require_candidates=True, require_future=False,
            )
            if self.baseline == "gameformer":
                scores = self.model.score_candidates(ext.gameformer_inputs, ext.candidates, ext.candidate_valid)[0]
            elif self.baseline == "dtpp":
                scores = self.model.score_candidates(ext.dtpp_inputs, ext.dtpp_candidate_tree, ext.candidate_valid, timesteps=future_len)[0]
            elif self.baseline in {"pluto", "plant2"}:
                scores = self.model.score_candidates(ext.planner_inputs, ext.candidates, ext.candidate_valid)[0]
            else:
                raise ValueError(self.baseline)
            mask = ext.candidate_valid[0]
            finite_scores = torch.isfinite(scores)
            mask = mask & finite_scores
            if self.require_conventional_safe:
                mask2 = mask & ext.conventional_safe[0]
                if bool(mask2.any()):
                    mask = mask2
            masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
            selected = int(torch.argmax(masked_scores).detach().cpu()) if bool(mask.any()) else -1
        if self.profile_timing:
            self._timing_sync()
            _t_after_model = time.perf_counter()

        cand = np.asarray(batch_np["cowp/candidates/trajectory"][0], dtype=np.float32)
        valid = np.asarray(batch_np["cowp/candidates/valid"][0], dtype=bool)
        candidate_finite = np.isfinite(cand).all(axis=(1, 2))
        # Keep the adapter's validity contract (including SDC-current validity)
        # in the execution path.  Falling back from a bad model score must not
        # resurrect a proposal that the adapter already invalidated.
        adapter_valid = ext.candidate_valid[0].detach().cpu().numpy().astype(bool, copy=False)
        valid = valid & candidate_finite & adapter_valid
        conv_np = np.asarray(batch_np.get("cowp/candidates/conventional_safe", valid[None])[0], dtype=bool) & candidate_finite
        fallback_reason = None
        if selected < 0 or selected >= len(cand) or not bool(valid[selected]):
            valid_idx = np.flatnonzero(valid)
            selected = int(valid_idx[0]) if valid_idx.size else -1
            fallback_reason = "no_finite_candidate_score" if valid_idx.size else "no_valid_candidate"
        has_valid_execution = bool(selected >= 0 and selected < len(cand) and valid[selected])
        execution_traj, emergency_action_used, execution_source = _resolve_execution_trajectory(
            cand, selected, has_valid_execution, np.asarray(agent_state[sdc_index], dtype=np.float32), self.cfg
        )
        selected_score_value = float(scores[selected].detach().cpu()) if has_valid_execution else float("nan")
        if not np.isfinite(selected_score_value):
            selected_score_value = None
        finite_valid_candidates = int((ext.candidate_valid[0] & torch.isfinite(scores)).sum().detach().cpu())
        self._last_diagnostics = {
            "baseline": self.baseline,
            "execution_mode": "candidate",
            "selected_idx": selected,
            "valid_candidates": int(valid.sum()),
            "conventional_candidates": int(conv_np.sum()),
            "selected_score": selected_score_value,
            "fallback": bool(fallback_reason is not None),
            "fallback_reason": fallback_reason,
            "finite_scored_candidates": finite_valid_candidates,
            "cowp_candidate_filter_applied": bool(self.require_conventional_safe),
            "optimized_candidate_observation_path": True,
            "scenario_static_map_cache": True,
            "causal_no_logged_future": True,
            "emergency_action_used": bool(emergency_action_used),
            "execution_trajectory_source": execution_source,
        }
        if self.profile_timing:
            self._last_diagnostics.update({
                "timing_ms/observation": 1000.0 * (_t_after_obs - _t_total0),
                "timing_ms/static_map_host": 1000.0 * (_t_after_map - _t_map0),
                "timing_ms/proposal_generation": 1000.0 * (_t_after_prop - _t_prop0),
                "timing_ms/adapter_and_model": 1000.0 * (_t_after_model - _t_model0),
                "timing_ms/total_before_action": 1000.0 * (time.perf_counter() - _t_total0),
            })
        return self._trajectory_to_action(agent_state, sdc_index, execution_traj)

    def consume_diagnostics(self) -> dict[str, Any] | None:
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row


def make_external_waymax_policy(
    checkpoint: str, cfg: dict, *, device: str = "auto", action_mode: str = "delta_xy_yaw",
    require_conventional_safe: bool = False, execution_mode: str = "auto", profile_timing: bool = False,
):
    return ExternalWaymaxPolicy(
        checkpoint=checkpoint, cfg=cfg, device=device, action_mode=action_mode,
        require_conventional_safe=require_conventional_safe, execution_mode=execution_mode, profile_timing=profile_timing,
    )
