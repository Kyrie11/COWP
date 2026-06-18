from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cowp.core.constants import MacroType
from cowp.label.trajectory_primitives import constant_accel_trajectory, smooth_stop_trajectory


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        x = jax.device_get(x)
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception as exc:
        raise TypeError(f"Cannot convert object of type {type(x)!r} to numpy array") from exc


def _get_field(obj: Any, names: tuple[str, ...]) -> Any | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return None


def extract_current_agent_state(state: Any) -> tuple[np.ndarray, int]:
    """Best-effort extraction of [N,11] current states from a Waymax SimulatorState."""
    traj = _get_field(state, ("sim_trajectory", "trajectory", "log_trajectory"))
    timestep = _get_field(state, ("timestep", "time_index", "current_timestep"))
    t = int(_to_numpy(timestep).reshape(-1)[0]) if timestep is not None else 0
    if traj is None:
        raise ValueError("SimulatorState has no sim_trajectory/trajectory/log_trajectory attribute.")

    x = _to_numpy(_get_field(traj, ("x", "center_x")))
    y = _to_numpy(_get_field(traj, ("y", "center_y")))
    yaw = _to_numpy(_get_field(traj, ("yaw", "heading", "bbox_yaw"))) if _get_field(traj, ("yaw", "heading", "bbox_yaw")) is not None else np.zeros_like(x)
    vx = _to_numpy(_get_field(traj, ("vel_x", "velocity_x", "vx"))) if _get_field(traj, ("vel_x", "velocity_x", "vx")) is not None else np.zeros_like(x)
    vy = _to_numpy(_get_field(traj, ("vel_y", "velocity_y", "vy"))) if _get_field(traj, ("vel_y", "velocity_y", "vy")) is not None else np.zeros_like(x)
    length = _to_numpy(_get_field(traj, ("length",))) if _get_field(traj, ("length",)) is not None else np.full_like(x, 4.8)
    width = _to_numpy(_get_field(traj, ("width",))) if _get_field(traj, ("width",)) is not None else np.full_like(x, 1.9)
    height = _to_numpy(_get_field(traj, ("height",))) if _get_field(traj, ("height",)) is not None else np.full_like(x, 1.6)
    valid = _to_numpy(_get_field(traj, ("valid",))) if _get_field(traj, ("valid",)) is not None else np.ones_like(x, dtype=bool)

    # Common Waymax shapes are [num_objects, num_timesteps] or batched variants.
    while x.ndim > 2:
        x = x[0]
        y = y[0]
        yaw = yaw[0]
        vx = vx[0]
        vy = vy[0]
        length = length[0]
        width = width[0]
        height = height[0]
        valid = valid[0]
    if x.ndim == 2:
        t = min(max(t, 0), x.shape[1] - 1)
        cols = [arr[:, t] for arr in (x, y, yaw, vx, vy, length, width, height, valid)]
    else:
        cols = [x, y, yaw, vx, vy, length, width, height, valid]
    x_c, y_c, yaw_c, vx_c, vy_c, l_c, w_c, h_c, v_c = cols
    N = int(np.asarray(x_c).shape[0])
    out = np.zeros((N, 11), dtype=np.float32)
    out[:, 0] = x_c
    out[:, 1] = y_c
    out[:, 3] = vx_c
    out[:, 4] = vy_c
    out[:, 5] = np.linalg.norm(out[:, 3:5], axis=-1)
    out[:, 6] = yaw_c
    out[:, 7] = np.where(np.asarray(l_c) > 0, l_c, 4.8)
    out[:, 8] = np.where(np.asarray(w_c) > 0, w_c, 1.9)
    out[:, 9] = np.where(np.asarray(h_c) > 0, h_c, 1.6)
    out[:, 10] = np.asarray(v_c).astype(bool).astype(np.float32)

    is_sdc = _get_field(state, ("is_sdc", "sdc_mask"))
    if is_sdc is None:
        meta = _get_field(state, ("object_metadata", "metadata"))
        is_sdc = _get_field(meta, ("is_sdc",)) if meta is not None else None
    if is_sdc is not None:
        sdc_arr = _to_numpy(is_sdc)
        while sdc_arr.ndim > 1:
            sdc_arr = sdc_arr[0]
        sdc_idx = int(np.argmax(sdc_arr.astype(float)))
    else:
        sdc_idx = 0
    return out, sdc_idx


def _online_candidates(current: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    limits = cfg.get("limits", {})
    K = int(limits.get("max_candidates", 64))
    H = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    dt = float(cfg.get("time", {}).get("dt", 0.1))
    acc_values = [0.0, -0.8, -1.5, -2.5, 0.5, 1.0]
    traj = np.zeros((K, H, 7), dtype=np.float32)
    valid = np.zeros(K, dtype=bool)
    macro = np.full(K, int(MacroType.PAD), dtype=np.int64)
    utility = np.zeros(K, dtype=np.float32)
    i = 0
    for a in acc_values:
        if i >= K:
            break
        traj[i] = constant_accel_trajectory(current, H, dt, accel=a)
        valid[i] = True
        macro[i] = int(MacroType.KEEP_LANE if a >= -0.1 else MacroType.YIELD)
        utility[i] = 0.05 * abs(float(a)) - 0.01 * np.linalg.norm(traj[i, -1, :2] - traj[i, 0, :2])
        i += 1
    if i < K:
        traj[i] = smooth_stop_trajectory(current, H, dt, decel=float(cfg.get("planning", {}).get("fallback_decel_mps2", -2.0)))
        valid[i] = True
        macro[i] = int(MacroType.STOP_BEFORE_CONFLICT)
        utility[i] = 1.0
    return traj, valid, macro, utility


def _critical_nearest(agent_state: np.ndarray, sdc_index: int, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    A = int(cfg.get("limits", {}).get("max_critical_agents", 8))
    valid_agents = agent_state[:, 10] > 0.5
    ego_xy = agent_state[sdc_index, :2]
    dist = np.linalg.norm(agent_state[:, :2] - ego_xy[None], axis=-1)
    order = [i for i in np.argsort(dist).tolist() if i != sdc_index and valid_agents[i]]
    idx = np.full(A, 0, dtype=np.int64)
    mask = np.zeros(A, dtype=bool)
    for a, j in enumerate(order[:A]):
        idx[a] = int(j)
        mask[a] = True
    return idx, mask


def build_online_batch(agent_state: np.ndarray, sdc_index: int, cfg: dict) -> dict[str, Any]:
    K = int(cfg.get("limits", {}).get("max_candidates", 64))
    A = int(cfg.get("limits", {}).get("max_critical_agents", 8))
    M = int(cfg.get("limits", {}).get("max_natural_alternatives", 24))
    R = int(cfg.get("limits", {}).get("max_safe_responses", 32))
    H = int(cfg.get("time", {}).get("future_steps", cfg.get("eval", {}).get("rollout_horizon_steps", 80)))
    C = int(cfg.get("limits", {}).get("max_conflict_regions", 64))
    d_state = int(cfg.get("model", cfg).get("d_state", 11))
    max_agents = int(cfg.get("limits", {}).get("max_agents", cfg.get("model", cfg).get("max_agents", 128)))
    hist = np.zeros((max_agents, 1, d_state), dtype=np.float32)
    n = min(max_agents, agent_state.shape[0])
    # Convert ScenarioData state [x,y,z,vx,vy,speed,heading,l,w,h,valid] to model
    # history [x,y,z,l,w,h,heading,vx,vy,speed,valid].
    hist[:n, 0, 0:3] = agent_state[:n, 0:3]
    hist[:n, 0, 3:6] = agent_state[:n, 7:10]
    hist[:n, 0, 6] = agent_state[:n, 6]
    hist[:n, 0, 7:9] = agent_state[:n, 3:5]
    hist[:n, 0, 9] = agent_state[:n, 5]
    hist[:n, 0, 10] = agent_state[:n, 10]
    agent_mask = np.zeros(max_agents, dtype=bool)
    agent_mask[:n] = agent_state[:n, 10] > 0.5
    agent_mask[min(sdc_index, max_agents - 1)] = True
    cand_traj, cand_valid, macro, utility = _online_candidates(agent_state[sdc_index], cfg)
    crit_idx, crit_valid = _critical_nearest(agent_state, sdc_index, cfg)
    batch = {
        "state/history": hist[None],
        "state/agent_valid": agent_mask[None],
        "cowp/candidates/trajectory": cand_traj[None],
        "cowp/candidates/valid": cand_valid[None],
        "cowp/candidates/macro_type": macro[None],
        "cowp/candidates/ego_utility_prior": utility[None],
        "cowp/candidates/conventional_safe": cand_valid[None],
        "cowp/critical/track_index": crit_idx[None],
        "cowp/critical/valid": crit_valid[None],
        "cowp/natural/traj": np.zeros((1, A, M, H, 7), dtype=np.float32),
        "cowp/natural/valid": np.zeros((1, A, M), dtype=bool),
        "cowp/natural/weight": np.zeros((1, A, M), dtype=np.float32),
        "cowp/natural/source": np.zeros((1, A, M), dtype=np.int64),
        "cowp/natural/priority_preserved": np.zeros((1, A, M), dtype=bool),
        "cowp/response/valid": np.zeros((1, K, A, R), dtype=bool),
        "cowp/response/is_safe": np.zeros((1, K, A, R), dtype=bool),
        "cowp/response/is_low_burden": np.zeros((1, K, A, R), dtype=bool),
        "cowp/response/burden_total": np.zeros((1, K, A, R), dtype=np.float32),
        "cowp/response/burden_components": np.zeros((1, K, A, R, 6), dtype=np.float32),
        "cowp/witness/exists": np.zeros((1, K, A), dtype=bool),
        "cowp/witness/token": np.zeros((1, K, A), dtype=np.int64),
        "cowp/witness/burden_total": np.zeros((1, K, A), dtype=np.float32),
        "cowp/witness/burden_components": np.zeros((1, K, A, 6), dtype=np.float32),
        "cowp/witness/opr": np.ones((1, K, A), dtype=np.float32),
        "cowp/witness/c_i": np.zeros((1, K, A), dtype=np.float32),
        "cowp/witness/conflict_interval": np.zeros((1, K, A, 2), dtype=np.int64),
        "map/conflict_regions": np.zeros((1, C, 8), dtype=np.float32),
        "map/conflict_region_valid": np.zeros((1, C), dtype=bool),
    }
    return batch


@dataclass
class COWPWaymaxPolicy:
    checkpoint: str
    cfg: dict
    device: str = "auto"
    witness_threshold: float = 0.5
    action_mode: str = "delta_xy_yaw"

    def __post_init__(self) -> None:
        import torch

        from cowp.models.cowp_model import COWPModel

        dev = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available() else ("cpu" if self.device == "auto" else self.device))
        ckpt = torch.load(self.checkpoint, map_location=dev)
        model_cfg = ckpt.get("cfg", self.cfg)
        self.model = COWPModel(model_cfg).to(dev)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.torch = torch
        self.dev = dev
        self._last_diagnostics: dict[str, Any] | None = None
        self._diagnostics_log: list[dict[str, Any]] = []

    def _trajectory_to_action(self, state: Any, agent_state: np.ndarray, sdc_index: int, traj: np.ndarray) -> Any:
        try:
            from waymax import datatypes  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("waymax.datatypes is required to convert a selected COWP trajectory to a Waymax action.") from exc
        N = agent_state.shape[0]
        data_dim = int(self.cfg.get("waymax", {}).get("action_dim", 3))
        data = np.zeros((N, data_dim), dtype=np.float32)
        valid = np.zeros((N,), dtype=bool)
        valid[sdc_index] = True
        next_pose = traj[min(1, len(traj) - 1)]
        dx = float(next_pose[0] - agent_state[sdc_index, 0])
        dy = float(next_pose[1] - agent_state[sdc_index, 1])
        dyaw = float(next_pose[2] - agent_state[sdc_index, 6])
        if self.action_mode == "absolute_xy_yaw":
            data[sdc_index, : min(data_dim, 3)] = np.asarray([next_pose[0], next_pose[1], next_pose[2]], dtype=np.float32)[:data_dim]
        else:
            data[sdc_index, : min(data_dim, 3)] = np.asarray([dx, dy, dyaw], dtype=np.float32)[:data_dim]
        try:
            import jax.numpy as jnp  # type: ignore

            return datatypes.Action(data=jnp.asarray(data), valid=jnp.asarray(valid))
        except Exception:
            return datatypes.Action(data=data, valid=valid)

    def __call__(self, state: Any, *, step: int | None = None, scenario_index: int | None = None) -> Any:
        agent_state, sdc_index = extract_current_agent_state(state)
        batch_np = build_online_batch(agent_state, sdc_index, self.cfg)
        batch = {k: self.torch.as_tensor(v, device=self.dev) for k, v in batch_np.items()}
        with self.torch.no_grad():
            pred = self.model(batch)
            scores = pred["planner_score"][0]
            cand_valid = batch["cowp/candidates/valid"][0].bool()
            witness = self.torch.sigmoid(pred["witness"]["exist_logits"])[0]
            opr = pred["witness"]["opr"][0]
            burden = pred["witness"].get("burden_total")
            c_i = pred["witness"].get("c_i")
            crit_mask = batch["cowp/critical/valid"][0].bool()
            witness = self.torch.where(crit_mask[None, :], witness, self.torch.zeros_like(witness))
            opr = self.torch.where(crit_mask[None, :], opr, self.torch.ones_like(opr))
            accepted = cand_valid & ~(witness >= float(self.witness_threshold)).any(dim=-1)
            accepted = accepted & (opr.min(dim=-1).values >= float(self.cfg.get("planning", {}).get("alpha_opr_infer", 0.35)))
            mask = accepted if accepted.any() else cand_valid
            selected = int(self.torch.argmin(self.torch.where(mask, scores, self.torch.full_like(scores, float("inf")))).item())
            selected_witness = witness[selected]
            selected_opr = opr[selected]
            diag = {
                "scenario_index": int(scenario_index) if scenario_index is not None else -1,
                "step": int(step) if step is not None else -1,
                "selected_candidate": int(selected),
                "accepted_candidates": int(accepted.sum().detach().cpu().item()),
                "fallback_used": bool(not accepted.any().detach().cpu().item()),
                "max_witness_prob": float(selected_witness.max().detach().cpu().item()) if selected_witness.numel() else 0.0,
                "min_opr": float(selected_opr.min().detach().cpu().item()) if selected_opr.numel() else 1.0,
                "mean_opr": float(selected_opr.mean().detach().cpu().item()) if selected_opr.numel() else 1.0,
                "score": float(scores[selected].detach().cpu().item()),
                "witness_threshold": float(self.witness_threshold),
                "alpha_opr": float(self.cfg.get("planning", {}).get("alpha_opr_infer", 0.35)),
            }
            if burden is not None:
                bsel = burden[0, selected]
                diag["max_predicted_burden"] = float(bsel[crit_mask].max().detach().cpu().item()) if bool(crit_mask.any().detach().cpu().item()) else 0.0
            if c_i is not None:
                csel = c_i[0, selected]
                diag["max_predicted_c_i"] = float(csel[crit_mask].max().detach().cpu().item()) if bool(crit_mask.any().detach().cpu().item()) else 0.0
        self._last_diagnostics = diag
        self._diagnostics_log.append(diag)
        traj = batch_np["cowp/candidates/trajectory"][0, selected]
        return self._trajectory_to_action(state, agent_state, sdc_index, traj)

    def consume_diagnostics(self) -> dict[str, Any] | None:
        """Return and clear the most recent online COWP policy diagnostic row."""
        row = self._last_diagnostics
        self._last_diagnostics = None
        return row

    def diagnostics_log(self) -> list[dict[str, Any]]:
        return list(self._diagnostics_log)


def make_cowp_policy(checkpoint: str, cfg: dict, *, device: str = "auto", witness_threshold: float = 0.5, action_mode: str = "delta_xy_yaw") -> COWPWaymaxPolicy:
    return COWPWaymaxPolicy(checkpoint=checkpoint, cfg=cfg, device=device, witness_threshold=witness_threshold, action_mode=action_mode)
