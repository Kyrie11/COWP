from __future__ import annotations

import torch


def _first_tensor(batch: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = batch.get(name)
        if value is not None and torch.is_tensor(value):
            return value
    return None


def infer_sdc_index(
    batch: dict[str, torch.Tensor], agent_history: torch.Tensor, *, require_explicit: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer the SDC index and one-hot ego mask for every scene.

    WOMD global coordinates have an arbitrary per-scenario origin.  The planner
    therefore needs an explicit ego identity before it can normalize the scene.
    Online batches expose ``state/is_sdc``; cached WOMD tensors usually expose
    ``womd/state/is_sdc``.  We retain index zero only as a legacy fallback.
    """
    B, N = agent_history.shape[:2]
    is_sdc = _first_tensor(
        batch,
        (
            "state/is_sdc",
            "womd/state/is_sdc",
            "state/current/is_sdc",
            "womd/state/current/is_sdc",
        ),
    )
    if is_sdc is None:
        if require_explicit:
            raise RuntimeError(
                "Causal coordinate violation: state/is_sdc is missing. Refusing "
                "to assume agent row 0 is ego in a reported run."
            )
        idx = torch.zeros(B, device=agent_history.device, dtype=torch.long)
    else:
        mask = is_sdc.bool()
        while mask.ndim > 2:
            mask = mask.any(dim=-1)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != B:
            mask = mask.reshape(B, -1)
        if mask.shape[1] < N:
            pad = torch.zeros(B, N - mask.shape[1], device=mask.device, dtype=torch.bool)
            mask = torch.cat([mask, pad], dim=1)
        mask = mask[:, :N]
        has_sdc = mask.any(dim=1)
        if require_explicit and not bool(has_sdc.all().item()):
            missing = torch.where(~has_sdc)[0].detach().cpu().tolist()
            raise RuntimeError(
                f"Causal coordinate violation: no valid SDC marker for batch rows {missing}."
            )
        idx = mask.float().argmax(dim=1).long()
        idx = torch.where(has_sdc, idx, torch.zeros_like(idx))
    ego_mask = torch.zeros(B, N, device=agent_history.device, dtype=torch.bool)
    ego_mask.scatter_(1, idx[:, None].clamp(0, max(N - 1, 0)), True)
    return idx, ego_mask


def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _rotate_xy(xy: torch.Tensor, cos_yaw: torch.Tensor, sin_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate global vectors by -ego_yaw with broadcasted [B] cos/sin."""
    while cos_yaw.ndim < xy.ndim - 1:
        cos_yaw = cos_yaw.unsqueeze(-1)
        sin_yaw = sin_yaw.unsqueeze(-1)
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack([cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y], dim=-1)


def ego_centric_inputs(
    agent_history: torch.Tensor,
    candidate_traj: torch.Tensor | None,
    conflict_regions: torch.Tensor | None,
    sdc_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return ego-centric copies used only by neural encoders.

    Absolute labels remain untouched and decoder trajectories are still anchored
    in the original WOMD frame.  This removes arbitrary map-origin translation
    and rotation from the representation while preserving all target semantics.
    """
    hist = torch.nan_to_num(agent_history.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
    B, N = hist.shape[:2]
    current = hist[:, :, -1] if hist.ndim == 4 else hist
    gather = sdc_index.clamp(0, max(N - 1, 0))[:, None, None].expand(B, 1, current.shape[-1])
    ego = torch.gather(current, 1, gather).squeeze(1)
    origin = ego[..., 0:2]
    ego_yaw = ego[..., 6] if ego.shape[-1] > 6 else torch.zeros(B, device=hist.device, dtype=hist.dtype)
    c = torch.cos(ego_yaw)
    s = torch.sin(ego_yaw)

    # Agent history format: [x,y,z,length,width,height,heading,vx,vy,speed,valid].
    hist[..., 0:2] = _rotate_xy(hist[..., 0:2] - origin[:, None, None, :] if hist.ndim == 4 else hist[..., 0:2] - origin[:, None, :], c, s)
    if hist.shape[-1] > 6:
        yaw_shape = [B] + [1] * (hist.ndim - 2)
        hist[..., 6] = _wrap_angle_torch(hist[..., 6] - ego_yaw.view(*yaw_shape))
    if hist.shape[-1] > 8:
        hist[..., 7:9] = _rotate_xy(hist[..., 7:9], c, s)

    cand = None
    if candidate_traj is not None:
        cand = torch.nan_to_num(candidate_traj.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
        cand[..., 0:2] = _rotate_xy(cand[..., 0:2] - origin[:, None, None, :], c, s)
        cand[..., 2] = _wrap_angle_torch(cand[..., 2] - ego_yaw[:, None, None])
        if cand.shape[-1] > 4:
            cand[..., 3:5] = _rotate_xy(cand[..., 3:5], c, s)

    conflict = None
    if conflict_regions is not None:
        conflict = torch.nan_to_num(conflict_regions.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
        if conflict.shape[-1] >= 3:
            conflict[..., 1:3] = _rotate_xy(conflict[..., 1:3] - origin[:, None, :], c, s)
    return hist, cand, conflict
