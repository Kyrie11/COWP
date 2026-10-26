from __future__ import annotations

import math
from typing import Mapping

import torch


def _first(batch: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = batch.get(name)
        if value is not None and torch.is_tensor(value):
            return value
    return None


def _reshape_current(x: torch.Tensor, *, max_agents: int | None = None) -> torch.Tensor:
    """Return [B, N] current-state tensor from WOMD flattened or shaped input."""
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        if max_agents is None:
            return x.unsqueeze(0)
        return x.reshape(1, max_agents)
    raise ValueError(f"Cannot interpret current WOMD tensor shape {tuple(x.shape)}")


def _reshape_temporal(x: torch.Tensor, *, max_agents: int | None = None, steps: int | None = None) -> torch.Tensor:
    """Return [B, N, T] temporal tensor from WOMD flattened or shaped input."""
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        b, flat = x.shape
        if max_agents is not None and flat % max_agents == 0:
            return x.reshape(b, max_agents, flat // max_agents)
        if steps is not None and flat % steps == 0:
            return x.reshape(b, flat // steps, steps)
        side = int(round(math.sqrt(flat)))
        if side * side == flat:
            return x.reshape(b, side, side)
    if x.ndim == 1:
        if max_agents is not None and x.numel() % max_agents == 0:
            return x.reshape(1, max_agents, x.numel() // max_agents)
        if steps is not None and x.numel() % steps == 0:
            return x.reshape(1, x.numel() // steps, steps)
    raise ValueError(f"Cannot interpret temporal WOMD tensor shape {tuple(x.shape)}")


def _past(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor | None:
    return _first(batch, (f"womd/state/past/{name}", f"state/past/{name}"))


def _current(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor | None:
    return _first(batch, (f"womd/state/current/{name}", f"state/current/{name}"))


def _all(batch: Mapping[str, torch.Tensor], name: str) -> torch.Tensor | None:
    return _first(batch, (f"womd/state/{name}", f"state/{name}"))


def has_womd_state(batch: Mapping[str, torch.Tensor]) -> bool:
    return _past(batch, "x") is not None and _current(batch, "x") is not None


def build_agent_history_from_womd(
    batch: Mapping[str, torch.Tensor],
    *,
    max_agents: int | None = None,
    history_steps: int = 11,
    d_state: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize WOMD tf.Example features into the model's [B,N,Th,D] state/history.

    The tensor cache stores raw WOMD tf.Example arrays under ``womd/state/...``.
    Depending on TensorFlow decoding, each feature may arrive either already shaped
    as [B,N,T] / [B,N] or flattened as [B,N*T] / [B,N].  This adapter keeps the
    model from silently falling back to label-only features when a real tensor
    cache is available.
    """
    x_past = _past(batch, "x")
    x_cur = _current(batch, "x")
    if x_past is None or x_cur is None:
        raise KeyError("WOMD state features are missing: expected womd/state/past/x and womd/state/current/x")

    # WOMD past has 10 steps and current has 1 step, yielding the 11-step history
    # used by the paper/model.  Use configured values only as reshape hints.
    past_steps_hint = max(int(history_steps) - 1, 1)
    xp = _reshape_temporal(x_past.float(), max_agents=max_agents, steps=past_steps_hint)
    xc = _reshape_current(x_cur.float(), max_agents=max_agents)
    B, N, Tp = xp.shape
    if xc.shape[0] != B:
        raise ValueError(f"Batch mismatch between WOMD past {tuple(xp.shape)} and current {tuple(xc.shape)}")
    if xc.shape[1] != N:
        # Prefer the current tensor for N when max_agents was not known; otherwise
        # fail loudly instead of mixing agents.
        if max_agents is None and xp.numel() % xc.shape[1] == 0:
            xp = x_past.float().reshape(B, xc.shape[1], -1)
            B, N, Tp = xp.shape
        else:
            raise ValueError(f"Agent count mismatch between WOMD past {tuple(xp.shape)} and current {tuple(xc.shape)}")

    device = xp.device
    out = torch.zeros(B, N, Tp + 1, d_state, device=device, dtype=torch.float32)

    def fill_temporal(dst_dim: int, names: tuple[str, ...], *, current_names: tuple[str, ...] | None = None) -> None:
        p = None
        for name in names:
            p = _past(batch, name)
            if p is not None:
                break
        c = None
        for name in current_names or names:
            c = _current(batch, name)
            if c is not None:
                break
        if p is not None:
            out[:, :, :Tp, dst_dim] = _reshape_temporal(p.float(), max_agents=N, steps=Tp)
        if c is not None:
            out[:, :, Tp, dst_dim] = _reshape_current(c.float(), max_agents=N)

    fill_temporal(0, ("x",))
    fill_temporal(1, ("y",))
    fill_temporal(2, ("z",))
    fill_temporal(3, ("length",), current_names=("length",))
    fill_temporal(4, ("width",), current_names=("width",))
    fill_temporal(5, ("height",), current_names=("height",))
    fill_temporal(6, ("bbox_yaw", "heading", "yaw"), current_names=("bbox_yaw", "heading", "yaw"))
    fill_temporal(7, ("velocity_x", "vx"), current_names=("velocity_x", "vx"))
    fill_temporal(8, ("velocity_y", "vy"), current_names=("velocity_y", "vy"))
    out[:, :, :, 9] = torch.linalg.norm(out[:, :, :, 7:9], dim=-1)

    valid_p = _past(batch, "valid")
    valid_c = _current(batch, "valid")
    valid = torch.zeros(B, N, Tp + 1, device=device, dtype=torch.bool)
    if valid_p is not None:
        valid[:, :, :Tp] = _reshape_temporal(valid_p.float(), max_agents=N, steps=Tp) > 0.5
    else:
        valid[:, :, :Tp] = torch.isfinite(out[:, :, :Tp, 0])
    if valid_c is not None:
        valid[:, :, Tp] = _reshape_current(valid_c.float(), max_agents=N) > 0.5
    else:
        valid[:, :, Tp] = torch.isfinite(out[:, :, Tp, 0])
    out[:, :, :, 10] = valid.float()

    # Agent-level mask means "observed at least once", with current-valid preferred.
    if valid_c is not None:
        agent_mask = _reshape_current(valid_c.float(), max_agents=N) > 0.5
    else:
        agent_mask = valid.any(dim=-1)

    # Force the SDC slot to remain visible if WOMD provides it, otherwise keep slot 0.
    is_sdc = _all(batch, "is_sdc")
    if is_sdc is not None:
        sdc = _reshape_current(is_sdc.float(), max_agents=N) > 0.5
        agent_mask = agent_mask | sdc
    elif agent_mask.shape[1] > 0:
        agent_mask[:, 0] = True

    return out, agent_mask
