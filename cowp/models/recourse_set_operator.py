from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RCRSOConfig:
    """Architecture contract for V16.8.45 RCRSO.

    The operator proposes a *finite set* of same-root control residuals.  It is
    deliberately not a certificate: every proposal must pass the unchanged
    analytic verifier before it can enter the joint CSP.
    """

    d_model: int = 128
    nhead: int = 4
    encoder_layers: int = 2
    dropout: float = 0.0
    max_queries: int = 16
    control_knots: int = 8
    root_feature_dim: int = 9
    ego_feature_dim: int = 9
    environment_feature_dim: int = 18
    blocker_feature_dim: int = 13
    conflict_feature_dim: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RCRSOConfig":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: data[k] for k in data if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TokenEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, layers: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input(x)
        padding = None if valid is None else ~valid.bool()
        return self.encoder(h, src_key_padding_mask=padding)


class RootConditionedRecourseSetTransformer(nn.Module):
    """Set-valued recourse proposal operator used by V16.8.45.

    Inputs are blocker/root-local.  Environment actors are encoded as a set, so
    their ordering cannot affect the output.  Learned query slots emit bounded
    longitudinal residual knots.  Feasibility/burden heads are *auxiliary only*;
    the online hard verifier ignores them for admission.
    """

    def __init__(self, cfg: RCRSOConfig | dict[str, Any] | None = None):
        super().__init__()
        self.cfg = cfg if isinstance(cfg, RCRSOConfig) else RCRSOConfig.from_dict(cfg)
        c = self.cfg
        self.root_encoder = _TokenEncoder(c.root_feature_dim, c.d_model, c.nhead, c.encoder_layers, c.dropout)
        self.ego_encoder = _TokenEncoder(c.ego_feature_dim, c.d_model, c.nhead, 1, c.dropout)
        self.env_encoder = _TokenEncoder(c.environment_feature_dim, c.d_model, c.nhead, 1, c.dropout)
        self.blocker = nn.Sequential(nn.Linear(c.blocker_feature_dim, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))
        self.conflict = nn.Sequential(nn.Linear(c.conflict_feature_dim, c.d_model), nn.GELU(), nn.LayerNorm(c.d_model))
        self.ego_cross = nn.MultiheadAttention(c.d_model, c.nhead, dropout=c.dropout, batch_first=True)
        self.env_cross = nn.MultiheadAttention(c.d_model, c.nhead, dropout=c.dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(c.max_queries, c.d_model) * 0.02)
        self.query_cross = nn.MultiheadAttention(c.d_model, c.nhead, dropout=c.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(c.d_model)
        self.control_head = nn.Sequential(nn.Linear(c.d_model, c.d_model), nn.GELU(), nn.Linear(c.d_model, c.control_knots))
        self.feasible_head = nn.Linear(c.d_model, 1)
        self.burden_head = nn.Sequential(nn.Linear(c.d_model, c.d_model // 2), nn.GELU(), nn.Linear(c.d_model // 2, 1))

    @staticmethod
    def _masked_mean(x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if valid is None:
            return x.mean(dim=1)
        w = valid.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        *,
        root_tokens: torch.Tensor,
        ego_tokens: torch.Tensor,
        environment_tokens: torch.Tensor,
        blocker_state: torch.Tensor,
        conflict_features: torch.Tensor,
        root_valid: torch.Tensor | None = None,
        ego_valid: torch.Tensor | None = None,
        environment_valid: torch.Tensor | None = None,
        query_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        c = self.cfg
        k = int(c.max_queries if query_count is None else query_count)
        if not 1 <= k <= c.max_queries:
            raise ValueError(f"query_count must be in [1,{c.max_queries}], got {k}")
        root_h = self.root_encoder(root_tokens, root_valid)
        ego_h = self.ego_encoder(ego_tokens, ego_valid)
        env_h = self.env_encoder(environment_tokens, environment_valid)

        root_ctx = self._masked_mean(root_h, root_valid).unsqueeze(1)
        ego_ctx, _ = self.ego_cross(root_ctx, ego_h, ego_h, key_padding_mask=None if ego_valid is None else ~ego_valid.bool())
        env_ctx, _ = self.env_cross(root_ctx, env_h, env_h, key_padding_mask=None if environment_valid is None else ~environment_valid.bool())
        global_ctx = root_ctx.squeeze(1) + ego_ctx.squeeze(1) + env_ctx.squeeze(1)
        global_ctx = global_ctx + self.blocker(blocker_state) + self.conflict(conflict_features)

        q = self.query[:k].unsqueeze(0).expand(root_tokens.shape[0], -1, -1)
        memory = torch.cat([root_h, ego_h, env_h, global_ctx.unsqueeze(1)], dim=1)
        q2, _ = self.query_cross(q, memory, memory)
        q = self.query_norm(q + q2 + global_ctx.unsqueeze(1))
        knots = torch.tanh(self.control_head(q))
        feasible_logits = self.feasible_head(q).squeeze(-1)
        burden = torch.nn.functional.softplus(self.burden_head(q).squeeze(-1))
        return {
            "control_knots": knots,
            "feasible_logits": feasible_logits,
            "burden_prediction": burden,
        }


def _local_xy(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    p = np.asarray(xy, dtype=np.float32) - np.asarray(origin, dtype=np.float32)[None, :]
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    x = c * p[:, 0] + s * p[:, 1]
    y = -s * p[:, 0] + c * p[:, 1]
    return np.stack([x, y], axis=-1).astype(np.float32)


def _local_velocity(v: np.ndarray, yaw: float) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], axis=-1).astype(np.float32)


def build_rcrso_features_np(
    *,
    root: np.ndarray,
    root_mass: float,
    root_source: int,
    blocker_state: np.ndarray,
    current_ego_trajectory: np.ndarray,
    shifted_ego_trajectory: np.ndarray,
    environment: list[dict[str, Any]],
    cfg: RCRSOConfig | dict[str, Any] | None = None,
    verifier_cfg: dict[str, Any] | None = None,
    blocker_object_type: int = 0,
) -> dict[str, np.ndarray]:
    """Create deterministic blocker/root-local RCRSO features.

    This function uses no logged future.  It accepts only the causal trajectories
    already available to the V42--V44 verifier.
    """
    c = cfg if isinstance(cfg, RCRSOConfig) else RCRSOConfig.from_dict(cfg)
    root = np.asarray(root, dtype=np.float32)
    bs = np.asarray(blocker_state, dtype=np.float32).reshape(-1)
    cur = np.asarray(current_ego_trajectory, dtype=np.float32)
    sh = np.asarray(shifted_ego_trajectory, dtype=np.float32)
    origin = root[0, :2].astype(np.float32)
    yaw0 = float(root[0, 2])

    root_xy = _local_xy(root[:, :2], origin, yaw0)
    root_v = _local_velocity(root[:, 3:5], yaw0)
    root_yaw = ((root[:, 2] - yaw0 + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
    t = np.linspace(0.0, 1.0, len(root), dtype=np.float32)
    root_tokens = np.stack([
        root_xy[:, 0], root_xy[:, 1], np.sin(root_yaw), np.cos(root_yaw),
        root_v[:, 0], root_v[:, 1], t, np.full_like(t, float(root_mass)),
        np.full_like(t, float(root_source) / 4.0),
    ], axis=-1)

    def ego_tokens(tr: np.ndarray, shifted_flag: float) -> np.ndarray:
        xy = _local_xy(tr[:, :2], origin, yaw0)
        vv = _local_velocity(tr[:, 3:5], yaw0)
        yy = ((tr[:, 2] - yaw0 + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)
        tt = np.linspace(0.0, 1.0, len(tr), dtype=np.float32)
        return np.stack([xy[:, 0], xy[:, 1], np.sin(yy), np.cos(yy), vv[:, 0], vv[:, 1], tt, np.full_like(tt, shifted_flag), np.ones_like(tt)], axis=-1)
    ego = np.concatenate([ego_tokens(cur, 0.0), ego_tokens(sh, 1.0)], axis=0).astype(np.float32)

    env_rows: list[np.ndarray] = []
    for actor in environment:
        a = np.asarray(actor["trajectory"], dtype=np.float32)
        b = np.asarray(actor["shifted_trajectory"], dtype=np.float32)
        for tr, shifted_flag in ((a, 0.0), (b, 1.0)):
            xy = _local_xy(tr[:, :2], origin, yaw0)
            vv = _local_velocity(tr[:, 3:5], yaw0)
            dist = np.linalg.norm(xy, axis=-1)
            j = int(np.argmin(dist)) if len(dist) else 0
            row = np.array([
                xy[0, 0], xy[0, 1], vv[0, 0], vv[0, 1],
                xy[j, 0], xy[j, 1], float(dist[j]), float(j / max(len(tr) - 1, 1)),
                xy[-1, 0], xy[-1, 1], vv[-1, 0], vv[-1, 1],
                float(actor.get("object_type", 0)) / 10.0, shifted_flag,
                float(np.min(dist) if len(dist) else 0.0), float(np.mean(dist) if len(dist) else 0.0),
                float(np.std(dist) if len(dist) else 0.0), 1.0,
            ], dtype=np.float32)
            env_rows.append(row)
    if not env_rows:
        env_rows = [np.zeros(c.environment_feature_dim, dtype=np.float32)]
    env = np.stack(env_rows, axis=0).astype(np.float32)

    # Blocker state: current global state expressed in the root frame plus stable
    # scalar geometry/type slots.  Agent state in COWP is [x,y,yaw,vx,vy,...].
    bxy = _local_xy(bs[None, :2], origin, yaw0)[0] if bs.size >= 2 else np.zeros(2, np.float32)
    bv = _local_velocity(bs[None, 3:5], yaw0)[0] if bs.size >= 5 else np.zeros(2, np.float32)
    byaw = float(((float(bs[2]) - yaw0 + np.pi) % (2 * np.pi) - np.pi)) if bs.size >= 3 else 0.0
    blocker = np.zeros(c.blocker_feature_dim, dtype=np.float32)
    vals = [bxy[0], bxy[1], np.sin(byaw), np.cos(byaw), bv[0], bv[1]]
    if bs.size > 9:
        vals += [float(bs[7]), float(bs[8]), float(bs[9]), float(bs[10])]
    root_speed = np.linalg.norm(root[:, 3:5], axis=-1) if root.shape[1] >= 5 else np.zeros(len(root), dtype=np.float32)
    root_initial_accel = float((root_speed[1] - root_speed[0]) / max(0.1, 1.0e-6)) if len(root_speed) >= 2 else 0.0
    if verifier_cfg is not None:
        root_initial_accel = float((root_speed[1] - root_speed[0]) / max(float(verifier_cfg.get("time", {}).get("dt", 0.1)), 1.0e-6)) if len(root_speed) >= 2 else 0.0
    vals += [float(root_mass), float(root_source) / 4.0, root_initial_accel]
    blocker[: min(len(vals), blocker.size)] = np.asarray(vals[: blocker.size], dtype=np.float32)

    # Conflict-event encoding is descriptive only.  When the frozen verifier
    # configuration is available, use the *same causal unsafe-event predicate*
    # as V42--V44 to encode first/last unsafe support.  These features guide the
    # proposer but never certify a response.  If the predicate cannot be
    # evaluated, fail closed to a zero event encoding rather than reading logged
    # future or inventing a learned collision label.
    conflict = np.zeros(c.conflict_feature_dim, dtype=np.float32)

    def event_interval(left: np.ndarray, right: np.ndarray, right_type: int) -> tuple[float, float, float]:
        if verifier_cfg is None:
            return 0.0, 0.0, 0.0
        try:
            from cowp.geometry.collision import unsafe_between
            res = unsafe_between(np.asarray(left, np.float32), np.asarray(right, np.float32), verifier_cfg, agent_type=int(right_type))
            mask = np.asarray(res.event_mask, dtype=bool).reshape(-1)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return 0.0, 0.0, 0.0
            den = float(max(mask.size - 1, 1))
            return 1.0, float(idx[0] / den), float(idx[-1] / den)
        except Exception:
            return 0.0, 0.0, 0.0

    ego_cur = event_interval(cur, root, int(blocker_object_type))
    ego_shift = event_interval(sh, root, int(blocker_object_type))
    env_events: list[int] = []
    env_horizon = max(len(root), 1)
    if verifier_cfg is not None:
        try:
            from cowp.geometry.collision import unsafe_between
            for actor in environment:
                actor_type = int(actor.get("object_type", 0))
                for left, right, right_type in (
                    (root, np.asarray(actor["trajectory"], np.float32), actor_type),
                    (np.asarray(actor["trajectory"], np.float32), root, int(blocker_object_type)),
                ):
                    try:
                        res = unsafe_between(left, right, verifier_cfg, agent_type=right_type)
                        env_events.extend(np.flatnonzero(np.asarray(res.event_mask, dtype=bool)).tolist())
                    except Exception:
                        pass
                shifted_root = np.concatenate([root[1:], root[-1:]], axis=0) if len(root) else root
                for left, right, right_type in (
                    (shifted_root, np.asarray(actor["shifted_trajectory"], np.float32), actor_type),
                    (np.asarray(actor["shifted_trajectory"], np.float32), shifted_root, int(blocker_object_type)),
                ):
                    try:
                        res = unsafe_between(left, right, verifier_cfg, agent_type=right_type)
                        env_events.extend(np.flatnonzero(np.asarray(res.event_mask, dtype=bool)).tolist())
                    except Exception:
                        pass
        except Exception:
            env_events = []
    if env_events:
        den = float(max(env_horizon - 1, 1))
        env_int = (1.0, float(min(env_events) / den), float(max(env_events) / den))
    else:
        env_int = (0.0, 0.0, 0.0)
    conflict[:] = np.asarray([
        ego_cur[0], ego_cur[1], ego_cur[2],
        ego_shift[0], ego_shift[1], ego_shift[2],
        env_int[0], env_int[1], env_int[2], float(root_mass),
    ], dtype=np.float32)

    return {
        "root_tokens": root_tokens.astype(np.float32),
        "ego_tokens": ego.astype(np.float32),
        "environment_tokens": env.astype(np.float32),
        "blocker_state": blocker.astype(np.float32),
        "conflict_features": conflict.astype(np.float32),
    }
