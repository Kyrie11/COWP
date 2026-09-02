from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from cowp.data.dataset import COWPNpzDataset, _key_allowed, collate_torch
from cowp.data.womd_features import build_agent_history_from_womd


EVAL_LABEL_KEYS = {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
    "cowp/candidates/macro_type",
    "cowp/candidates/ego_utility_prior",
    "cowp/candidates/is_neutral",
    "cowp/candidates/is_logged",
    "cowp/candidates/conventional_safe",
    "cowp/candidates/false_safe",
    "cowp/candidates/noncoercive_feasible",
    "cowp/critical/valid",
    "cowp/natural/beta",
    "cowp/witness/exists",
    "cowp/witness/token",
    "cowp/witness/burden_total",
    "cowp/witness/min_safe_burden",
    "cowp/witness/opr",
    "cowp/witness/conflict_interval",
    "waymax/candidate_rollout_valid",
    "waymax/candidate_collision",
    "waymax/candidate_offroad",
    "waymax/candidate_log_divergence",
}


EXTERNAL_WANTED_PREFIXES = {
    "state/",
    "womd/state/",
    "roadgraph_samples/",
    "womd/roadgraph_samples/",
    "map/",
    "scenario/",
    "dataset/",
    "cowp/candidates/",
    "cowp/critical/",
    "cowp/witness/",
    "cowp/natural/beta",
    "waymax/candidate_rollout_valid",
    "waymax/candidate_collision",
    "waymax/candidate_offroad",
    "waymax/candidate_log_divergence",
}


class ExternalCOWPDataset(Dataset):
    """COWP npz dataset for GameFormer/DTPP adapters.

    The normal staged COWP dataset avoids loading dense raw WOMD future/map
    tensors.  GameFormer/DTPP need logged ego/neighbor futures and optionally
    vectorized map tokens, so this dataset uses a custom key whitelist while
    preserving the same NPZ loading/canonicalization path as COWP.
    """

    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz", include_waymax_outcomes: bool = True):
        self.base = COWPNpzDataset(cache_dir, pattern)
        self._wanted = set(EXTERNAL_WANTED_PREFIXES)
        if not include_waymax_outcomes:
            self._wanted = {x for x in self._wanted if not x.startswith("waymax/")}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = self.base.load(idx, self._wanted)
        out: dict[str, torch.Tensor] = {}
        for k, v in d.items():
            if not _key_allowed(k, self._wanted):
                continue
            arr = np.asarray(v)
            if arr.dtype.kind in "fiu" or arr.dtype == np.bool_:
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                out[k] = torch.from_numpy(arr)
        return out


def first_tensor(batch: Mapping[str, torch.Tensor], names: Iterable[str]) -> torch.Tensor | None:
    for name in names:
        x = batch.get(name)
        if torch.is_tensor(x):
            return x
    return None


def _reshape_current(x: torch.Tensor, *, max_agents: int | None = None) -> torch.Tensor:
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        if max_agents is None:
            return x.unsqueeze(0)
        return x.reshape(1, max_agents)
    raise ValueError(f"Cannot interpret current tensor shape {tuple(x.shape)}")


def _reshape_temporal(x: torch.Tensor, *, max_agents: int | None = None, steps: int | None = None) -> torch.Tensor:
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        b, flat = x.shape
        if max_agents is not None and flat % max_agents == 0:
            return x.reshape(b, max_agents, flat // max_agents)
        if steps is not None and flat % steps == 0:
            return x.reshape(b, flat // steps, steps)
    if x.ndim == 1:
        if max_agents is not None and x.numel() % max_agents == 0:
            return x.reshape(1, max_agents, x.numel() // max_agents)
        if steps is not None and x.numel() % steps == 0:
            return x.reshape(1, x.numel() // steps, steps)
    raise ValueError(f"Cannot interpret temporal tensor shape {tuple(x.shape)}")


def future_tensor(batch: Mapping[str, torch.Tensor], name: str, *, max_agents: int, steps: int | None = None) -> torch.Tensor | None:
    x = first_tensor(batch, (f"state/future/{name}", f"womd/state/future/{name}"))
    if x is None:
        return None
    return _reshape_temporal(x.float(), max_agents=max_agents, steps=steps)


def current_tensor(batch: Mapping[str, torch.Tensor], name: str, *, max_agents: int) -> torch.Tensor | None:
    x = first_tensor(batch, (f"state/current/{name}", f"womd/state/current/{name}"))
    if x is None:
        return None
    return _reshape_current(x.float(), max_agents=max_agents)


def all_tensor(batch: Mapping[str, torch.Tensor], name: str, *, max_agents: int) -> torch.Tensor | None:
    x = first_tensor(batch, (f"state/{name}", f"womd/state/{name}"))
    if x is None:
        return None
    if x.ndim <= 2:
        return _reshape_current(x.float(), max_agents=max_agents)
    return x.float()


def sdc_indices(batch: Mapping[str, torch.Tensor], max_agents: int, batch_size: int, device: torch.device) -> torch.Tensor:
    is_sdc = all_tensor(batch, "is_sdc", max_agents=max_agents)
    if is_sdc is not None and is_sdc.ndim >= 2:
        return torch.argmax(is_sdc.float().to(device), dim=1).long()
    return torch.zeros(batch_size, device=device, dtype=torch.long)


def gather_rows(x: torch.Tensor, row_idx: torch.Tensor) -> torch.Tensor:
    """Gather rows from x [B,N,...] by row_idx [B,M] -> [B,M,...]."""
    B, N = x.shape[:2]
    idx = row_idx.clamp(0, max(N - 1, 0)).long()
    view = idx.reshape(B, -1, *([1] * (x.ndim - 2))).expand(B, idx.shape[1], *x.shape[2:])
    return torch.gather(x, 1, view)


def _agent_type(batch: Mapping[str, torch.Tensor], max_agents: int, device: torch.device) -> torch.Tensor | None:
    typ = all_tensor(batch, "type", max_agents=max_agents)
    if typ is None:
        typ = current_tensor(batch, "type", max_agents=max_agents)
    return typ.to(device) if typ is not None else None


def _select_neighbors(agent_history: torch.Tensor, agent_mask: torch.Tensor, sdc: torch.Tensor, max_neighbors: int) -> torch.Tensor:
    B, N = agent_history.shape[:2]
    cur = agent_history[:, :, -1]
    ego_xy = cur[torch.arange(B, device=cur.device), sdc.clamp(0, N - 1), :2]
    dist = torch.linalg.norm(cur[:, :, :2] - ego_xy[:, None, :], dim=-1)
    valid = agent_mask.bool() & (cur[:, :, 10] > 0.5)
    valid[torch.arange(B, device=cur.device), sdc.clamp(0, N - 1)] = False
    dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
    order = torch.argsort(dist, dim=1)
    if max_neighbors <= order.shape[1]:
        return order[:, :max_neighbors]
    pad = torch.zeros(B, max_neighbors - order.shape[1], device=order.device, dtype=order.dtype)
    return torch.cat([order, pad], dim=1)




def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _batch_view_1d(x: torch.Tensor, target_ndim_minus_last: int) -> torch.Tensor:
    return x.reshape(x.shape[0], *([1] * max(target_ndim_minus_last - 1, 0)))


def _xy_to_ego_frame(xy: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    """Rotate/translate global xy coordinates into each batch ego frame."""
    origin_v = origin.reshape(origin.shape[0], *([1] * (xy.ndim - 2)), 2).to(device=xy.device, dtype=xy.dtype)
    yaw_v = _batch_view_1d(yaw0.to(device=xy.device, dtype=xy.dtype), xy.ndim - 1)
    c = torch.cos(yaw_v)
    s = torch.sin(yaw_v)
    rel = xy - origin_v
    return torch.stack((c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]), dim=-1)


def _rotate_global_to_local(vec: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    """Rotate global 2-D vectors into each batch ego frame without translation."""
    yaw_v = _batch_view_1d(yaw0.to(device=vec.device, dtype=vec.dtype), vec.ndim - 1)
    c = torch.cos(yaw_v)
    s = torch.sin(yaw_v)
    return torch.stack((c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]), dim=-1)


def _history_to_ego_frame(hist: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    out = hist.clone()
    out[..., 0:2] = _xy_to_ego_frame(out[..., 0:2], origin, yaw0)
    if out.shape[-1] > 6:
        yaw_v = _batch_view_1d(yaw0.to(device=out.device, dtype=out.dtype), out[..., 6].ndim)
        out[..., 6] = _wrap_angle_torch(out[..., 6] - yaw_v)
    if out.shape[-1] > 8:
        out[..., 7:9] = _rotate_global_to_local(out[..., 7:9], yaw0)
        if out.shape[-1] > 9:
            out[..., 9] = torch.linalg.norm(out[..., 7:9], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _candidate_to_ego_frame(candidates: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    out = candidates.clone()
    if out.numel() == 0:
        return out
    out[..., 0:2] = _xy_to_ego_frame(out[..., 0:2], origin, yaw0)
    if out.shape[-1] > 2:
        yaw_v = _batch_view_1d(yaw0.to(device=out.device, dtype=out.dtype), out[..., 2].ndim)
        out[..., 2] = _wrap_angle_torch(out[..., 2] - yaw_v)
    if out.shape[-1] > 4:
        out[..., 3:5] = _rotate_global_to_local(out[..., 3:5], yaw0)
        if out.shape[-1] > 5:
            out[..., 5] = torch.linalg.norm(out[..., 3:5], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def history_to_gameformer_state(hist: torch.Tensor, types: torch.Tensor | None = None) -> torch.Tensor:
    """Map COWP history [x,y,z,l,w,h,yaw,vx,vy,speed,valid] to GameFormer [x,y,yaw,vx,vy,w,l,h,type]."""
    B, N, T, _ = hist.shape
    out = torch.zeros(B, N, T, 9, device=hist.device, dtype=hist.dtype)
    out[..., 0] = hist[..., 0]
    out[..., 1] = hist[..., 1]
    out[..., 2] = hist[..., 6]
    out[..., 3] = hist[..., 7]
    out[..., 4] = hist[..., 8]
    out[..., 5] = torch.clamp(hist[..., 4], min=0.1)  # width
    out[..., 6] = torch.clamp(hist[..., 3], min=0.1)  # length
    out[..., 7] = torch.clamp(hist[..., 5], min=0.1)  # height
    if types is None:
        out[..., 8] = 1.0
    else:
        out[..., 8] = torch.clamp(types[:, :, None].float(), min=0, max=3)
    out = torch.where(hist[..., 10:11] > 0.5, out, torch.zeros_like(out))
    return out


def history_to_dtpp_ego(hist_ego: torch.Tensor) -> torch.Tensor:
    """DTPP ego past: [x,y,yaw,vx,vy,speed,valid]."""
    out = torch.zeros(*hist_ego.shape[:-1], 7, device=hist_ego.device, dtype=hist_ego.dtype)
    out[..., 0] = hist_ego[..., 0]
    out[..., 1] = hist_ego[..., 1]
    out[..., 2] = hist_ego[..., 6]
    out[..., 3] = hist_ego[..., 7]
    out[..., 4] = hist_ego[..., 8]
    out[..., 5] = hist_ego[..., 9]
    out[..., 6] = hist_ego[..., 10]
    return torch.where(hist_ego[..., 10:11] > 0.5, out, torch.zeros_like(out))


def history_to_dtpp_neighbors(hist: torch.Tensor, types: torch.Tensor | None = None) -> torch.Tensor:
    """DTPP neighbor past: 11 dims with current-state attributes in dims 6:10 for cost model."""
    out = torch.zeros(*hist.shape[:-1], 11, device=hist.device, dtype=hist.dtype)
    out[..., 0] = hist[..., 0]
    out[..., 1] = hist[..., 1]
    out[..., 2] = hist[..., 6]
    out[..., 3] = hist[..., 7]
    out[..., 4] = hist[..., 8]
    out[..., 5] = hist[..., 9]
    out[..., 6] = torch.clamp(hist[..., 3], min=0.1)
    out[..., 7] = torch.clamp(hist[..., 4], min=0.1)
    out[..., 8] = torch.clamp(hist[..., 5], min=0.1)
    if types is None:
        out[..., 9] = 1.0
    else:
        out[..., 9] = torch.clamp(types[:, :, None].float(), min=0, max=3)
    out[..., 10] = hist[..., 10]
    return torch.where(hist[..., 10:11] > 0.5, out, torch.zeros_like(out))


def _future_xy_for_selected(batch: Mapping[str, torch.Tensor], selected_idx: torch.Tensor, max_agents: int, horizon: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fx = future_tensor(batch, "x", max_agents=max_agents, steps=horizon)
    fy = future_tensor(batch, "y", max_agents=max_agents, steps=horizon)
    fv = future_tensor(batch, "valid", max_agents=max_agents, steps=horizon)
    B, M = selected_idx.shape
    if fx is None or fy is None:
        return torch.zeros(B, M, horizon, 2, device=device), torch.zeros(B, M, horizon, device=device, dtype=torch.bool)
    fx = fx.to(device)
    fy = fy.to(device)
    T = min(horizon, fx.shape[-1], fy.shape[-1])
    xy = torch.zeros(B, max_agents, horizon, 2, device=device, dtype=torch.float32)
    xy[:, :, :T, 0] = fx[:, :, :T]
    xy[:, :, :T, 1] = fy[:, :, :T]
    if fv is not None:
        valid = torch.zeros(B, max_agents, horizon, device=device, dtype=torch.bool)
        valid[:, :, :T] = fv.to(device)[:, :, :T] > 0.5
    else:
        valid = torch.isfinite(xy).all(dim=-1)
    return gather_rows(xy, selected_idx), gather_rows(valid.float().unsqueeze(-1), selected_idx).squeeze(-1).bool()



def candidate_geometry_finite(candidates: torch.Tensor) -> torch.Tensor:
    """Return [B,K] finite-geometry mask for candidate trajectories.

    Keep this as a single-dimension reduction for compatibility with older
    torch builds whose ``Tensor.all`` does not accept tuple dims.
    """
    if candidates.ndim < 3:
        return torch.isfinite(candidates)
    flat = torch.isfinite(candidates).reshape(candidates.shape[0], candidates.shape[1], -1)
    return flat.all(dim=-1)


def _named_tensor(batch: Mapping[str, torch.Tensor], base: str) -> torch.Tensor | None:
    return first_tensor(batch, (base, f"womd/{base}"))


def _normalise_batched(x: torch.Tensor, *, trailing: int | None = None) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if trailing is not None and x.ndim == 2 and trailing > 1:
        x = x.reshape(x.shape[0], -1, trailing)
    return x


def _roadgraph_arrays(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    origin: torch.Tensor | None = None,
    yaw0: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    xyz = _named_tensor(batch, "roadgraph_samples/xyz")
    if xyz is not None:
        xyz = _normalise_batched(xyz.float().to(device), trailing=3)
        xy_raw = xyz[..., :2].reshape(xyz.shape[0], -1, 2)
    else:
        x = _named_tensor(batch, "roadgraph_samples/x")
        y = _named_tensor(batch, "roadgraph_samples/y")
        if x is None or y is None:
            return None, None, None, None, None
        x = _normalise_batched(x.float().to(device))
        y = _normalise_batched(y.float().to(device))
        xy_raw = torch.stack([x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)], dim=-1)
    B, P = xy_raw.shape[:2]
    valid_t = _named_tensor(batch, "roadgraph_samples/valid")
    if valid_t is not None:
        valid = _normalise_batched(valid_t.bool().to(device)).reshape(B, -1)[:, :P]
    else:
        valid = torch.ones(B, P, device=device, dtype=torch.bool)
    finite = torch.isfinite(xy_raw).reshape(B, P, -1).all(dim=-1)
    valid = valid & finite
    xy = torch.nan_to_num(xy_raw, nan=0.0, posinf=0.0, neginf=0.0)

    ids_t = _named_tensor(batch, "roadgraph_samples/id")
    types_t = _named_tensor(batch, "roadgraph_samples/type")
    ids = _normalise_batched(ids_t.to(device)).reshape(B, -1)[:, :P].long() if ids_t is not None else None
    types = _normalise_batched(types_t.to(device)).reshape(B, -1)[:, :P].long() if types_t is not None else None
    dir_t = _named_tensor(batch, "roadgraph_samples/dir")
    dir_xy = None
    if dir_t is not None:
        d = _normalise_batched(dir_t.float().to(device), trailing=3)
        if d.shape[-1] >= 2:
            dir_xy = torch.nan_to_num(d[..., :2].reshape(B, -1, 2)[:, :P], nan=0.0, posinf=0.0, neginf=0.0)

    if origin is not None and yaw0 is not None:
        xy = _xy_to_ego_frame(xy, origin, yaw0)
        if dir_xy is not None:
            dir_xy = _rotate_global_to_local(dir_xy, yaw0)
    xy = torch.where(valid[..., None], xy, torch.zeros_like(xy))
    if dir_xy is not None:
        dir_xy = torch.where(valid[..., None], dir_xy, torch.zeros_like(dir_xy))
    return xy, valid, ids, types, dir_xy


def _roadgraph_xy(batch: Mapping[str, torch.Tensor], device: torch.device, origin: torch.Tensor | None = None, yaw0: torch.Tensor | None = None) -> torch.Tensor | None:
    xy, valid, _ids, _types, _dir = _roadgraph_arrays(batch, device, origin=origin, yaw0=yaw0)
    if xy is None or valid is None:
        return None
    return torch.where(valid[..., None], xy, torch.zeros_like(xy))


def _first_present_count(*xs: torch.Tensor | None) -> int:
    for x in xs:
        if x is not None:
            return int(x.shape[1])
    return 0


def external_map_topology_report(batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
    xyz = first_tensor(batch, ("roadgraph_samples/xyz", "womd/roadgraph_samples/xyz"))
    x = first_tensor(batch, ("roadgraph_samples/x", "womd/roadgraph_samples/x"))
    y = first_tensor(batch, ("roadgraph_samples/y", "womd/roadgraph_samples/y"))
    ids = first_tensor(batch, ("roadgraph_samples/id", "womd/roadgraph_samples/id"))
    typ = first_tensor(batch, ("roadgraph_samples/type", "womd/roadgraph_samples/type"))
    direction = first_tensor(batch, ("roadgraph_samples/dir", "womd/roadgraph_samples/dir"))
    valid = first_tensor(batch, ("roadgraph_samples/valid", "womd/roadgraph_samples/valid"))
    if xyz is not None:
        points = int(xyz.reshape(xyz.shape[0], -1, xyz.shape[-1]).shape[1]) if xyz.ndim >= 2 else int(xyz.numel() // 3)
        has_xy = True
    elif x is not None and y is not None:
        points = int(x.reshape(x.shape[0], -1).shape[1]) if x.ndim >= 2 else int(x.numel())
        has_xy = True
    else:
        points = 0
        has_xy = False
    counts = []
    for t in (ids, typ, valid):
        if t is not None:
            counts.append(int(t.reshape(t.shape[0], -1).shape[1]) if t.ndim >= 2 else int(t.numel()))
    if direction is not None:
        counts.append(int(direction.reshape(direction.shape[0], -1, direction.shape[-1]).shape[1]) if direction.ndim >= 2 else int(direction.numel() // 3))
    aligned = bool(has_xy and ids is not None and typ is not None and direction is not None and valid is not None and all(c == points for c in counts))
    return {
        "has_xy": bool(has_xy),
        "has_id": ids is not None,
        "has_type": typ is not None,
        "has_dir": direction is not None,
        "has_valid": valid is not None,
        "aligned": aligned,
        "points": int(points),
    }


_LANE_CENTER_TYPES = {1, 2, 3}
_CROSSWALK_TYPES = {18}


def _feature_groups_for_scene(
    xy: torch.Tensor,
    valid: torch.Tensor,
    ids: torch.Tensor | None,
    types: torch.Tensor | None,
    dir_xy: torch.Tensor | None,
    allowed_types: set[int],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if ids is None or types is None or ids.numel() != valid.numel() or types.numel() != valid.numel():
        return []
    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seen: list[int] = []
    valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
    for idx_t in valid_idx.detach().cpu().tolist():
        fid = int(ids[idx_t].detach().cpu())
        if fid in seen:
            continue
        seen.append(fid)
        mask = valid & (ids == fid)
        if not bool(mask.any()):
            continue
        typ_vals = types[mask]
        typ = int(typ_vals[0].detach().cpu()) if typ_vals.numel() else -1
        if typ not in allowed_types:
            continue
        pts = xy[mask]
        if pts.shape[0] == 0:
            continue
        if dir_xy is not None and dir_xy.shape[0] == xy.shape[0]:
            dirs = dir_xy[mask]
        else:
            dirs = torch.zeros_like(pts)
            if pts.shape[0] > 1:
                dirs[1:] = pts[1:] - pts[:-1]
                dirs[0] = dirs[1]
        out.append((pts, dirs, torch.full((pts.shape[0],), typ, device=xy.device, dtype=torch.long)))
    return out


def _heading_from_dirs_or_points(pts: torch.Tensor, dirs: torch.Tensor | None = None) -> torch.Tensor:
    if dirs is not None and dirs.numel() and bool((torch.linalg.norm(dirs, dim=-1) > 1.0e-6).any()):
        d = dirs
    else:
        d = torch.zeros_like(pts)
        if pts.shape[0] > 1:
            d[1:] = pts[1:] - pts[:-1]
            d[0] = d[1]
    return torch.atan2(d[..., 1], d[..., 0])


def _copy_lane_feature(dst: torch.Tensor, valid_dst: torch.Tensor, pts: torch.Tensor, dirs: torch.Tensor | None, max_points: int, *, gameformer: bool, src_valid: torch.Tensor | None = None) -> None:
    P = min(int(max_points), int(pts.shape[0]))
    if P <= 0:
        return
    p = pts[:P]
    h = _heading_from_dirs_or_points(p, dirs[:P] if dirs is not None else None)
    dst[:P, 0:2] = p
    dst[:P, 2] = h
    if gameformer and dst.shape[-1] >= 16:
        dst[:P, 10] = 1.0
        dst[:P, 14] = 1.0
    if src_valid is None:
        valid_dst[:P] = True
    else:
        valid_dst[:P] = src_valid[:P].bool()


def _copy_cross_feature(dst: torch.Tensor, valid_dst: torch.Tensor, pts: torch.Tensor, max_points: int, src_valid: torch.Tensor | None = None) -> None:
    P = min(int(max_points), int(pts.shape[0]))
    if P <= 0:
        return
    dst[:P, 0:2] = pts[:P]
    if dst.shape[-1] > 2:
        dst[:P, 2] = 1.0
    if src_valid is None:
        valid_dst[:P] = True
    else:
        valid_dst[:P] = src_valid[:P].bool()


def _sort_features_by_anchor(features: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], anchor: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if not features:
        return []
    scored = []
    for feat in features:
        pts = feat[0]
        d = torch.linalg.norm(pts[:, :2] - anchor[None, :2], dim=-1).min()
        scored.append((float(d.detach().cpu()), feat))
    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored]


def build_gameformer_map(
    batch: Mapping[str, torch.Tensor],
    num_agents_to_predict: int,
    device: torch.device,
    n_lanes: int = 6,
    lane_points: int = 100,
    n_crosswalks: int = 4,
    *,
    origin: torch.Tensor | None = None,
    yaw0: torch.Tensor | None = None,
    agent_xy: torch.Tensor | None = None,
    agent_valid: torch.Tensor | None = None,
    return_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
    A = int(num_agents_to_predict)
    lanes = torch.zeros(B, A, n_lanes, lane_points, 16, device=device)
    cross = torch.zeros(B, A, n_crosswalks, lane_points, 3, device=device)
    lane_valid = torch.zeros(B, A, n_lanes, lane_points, device=device, dtype=torch.bool)
    cross_valid = torch.zeros(B, A, n_crosswalks, lane_points, device=device, dtype=torch.bool)
    xy, valid, ids, types, dirs = _roadgraph_arrays(batch, device, origin=origin, yaw0=yaw0)
    if xy is None or valid is None:
        return (lanes, cross, lane_valid, cross_valid) if return_valid else (lanes, cross)
    if agent_xy is None:
        agent_xy = torch.zeros(B, A, 2, device=device)
    else:
        agent_xy = agent_xy.to(device)
        if origin is not None and yaw0 is not None:
            # Caller may pass global anchors; if they already passed local anchors
            # with origin/yaw omitted, this is a no-op path.
            agent_xy = _xy_to_ego_frame(agent_xy, origin, yaw0)
    if agent_valid is None:
        agent_valid = torch.ones(B, A, device=device, dtype=torch.bool)
    else:
        agent_valid = agent_valid.bool().to(device)
    for b in range(B):
        lane_feats = _feature_groups_for_scene(xy[b], valid[b], ids[b] if ids is not None else None, types[b] if types is not None else None, dirs[b] if dirs is not None else None, _LANE_CENTER_TYPES)
        cross_feats = _feature_groups_for_scene(xy[b], valid[b], ids[b] if ids is not None else None, types[b] if types is not None else None, dirs[b] if dirs is not None else None, _CROSSWALK_TYPES)
        if not lane_feats:
            pts = xy[b]
            vv = valid[b]
            if pts.numel() > 0 and bool(vv.any()):
                d = torch.zeros_like(pts)
                if pts.shape[0] > 1:
                    d[1:] = pts[1:] - pts[:-1]
                    d[0] = d[1]
                lane_feats = [(pts, d, torch.ones(pts.shape[0], device=device, dtype=torch.long), vv)]
        for a in range(A):
            if a >= agent_valid.shape[1] or not bool(agent_valid[b, a]):
                continue
            anchor = agent_xy[b, min(a, agent_xy.shape[1] - 1)]
            for j, feat in enumerate(_sort_features_by_anchor(lane_feats, anchor)[:n_lanes]):
                _copy_lane_feature(lanes[b, a, j], lane_valid[b, a, j], feat[0], feat[1], lane_points, gameformer=True, src_valid=feat[3] if len(feat) > 3 else None)
            for j, feat in enumerate(_sort_features_by_anchor(cross_feats, anchor)[:n_crosswalks]):
                _copy_cross_feature(cross[b, a, j], cross_valid[b, a, j], feat[0], lane_points, src_valid=feat[3] if len(feat) > 3 else None)
    return (lanes, cross, lane_valid, cross_valid) if return_valid else (lanes, cross)


def build_dtpp_map(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    n_lanes: int = 50,
    lane_points: int = 50,
    n_crosswalks: int = 20,
    cross_points: int = 30,
    *,
    origin: torch.Tensor | None = None,
    yaw0: torch.Tensor | None = None,
    return_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
    lanes = torch.zeros(B, n_lanes, lane_points, 7, device=device)
    cross = torch.zeros(B, n_crosswalks, cross_points, 3, device=device)
    lane_valid = torch.zeros(B, n_lanes, lane_points, device=device, dtype=torch.bool)
    cross_valid = torch.zeros(B, n_crosswalks, cross_points, device=device, dtype=torch.bool)
    xy, valid, ids, types, dirs = _roadgraph_arrays(batch, device, origin=origin, yaw0=yaw0)
    if xy is None or valid is None:
        return (lanes, cross, lane_valid, cross_valid) if return_valid else (lanes, cross)
    for b in range(B):
        lane_feats = _feature_groups_for_scene(xy[b], valid[b], ids[b] if ids is not None else None, types[b] if types is not None else None, dirs[b] if dirs is not None else None, _LANE_CENTER_TYPES)
        cross_feats = _feature_groups_for_scene(xy[b], valid[b], ids[b] if ids is not None else None, types[b] if types is not None else None, dirs[b] if dirs is not None else None, _CROSSWALK_TYPES)
        if not lane_feats:
            pts = xy[b]
            vv = valid[b]
            if pts.numel() > 0 and bool(vv.any()):
                total = min(n_lanes * lane_points, int(pts.shape[0]))
                pts2 = pts[:total]
                vv2 = vv[:total]
                d = torch.zeros_like(pts2)
                if pts2.shape[0] > 1:
                    d[1:] = pts2[1:] - pts2[:-1]
                    d[0] = d[1]
                for j in range(min(n_lanes, (total + lane_points - 1) // lane_points)):
                    lo = j * lane_points
                    hi = min(lo + lane_points, total)
                    if bool(vv2[lo:hi].any()):
                        _copy_lane_feature(lanes[b, j], lane_valid[b, j], pts2[lo:hi], d[lo:hi], lane_points, gameformer=False, src_valid=vv2[lo:hi])
        else:
            anchor = torch.zeros(2, device=device)
            for j, feat in enumerate(_sort_features_by_anchor(lane_feats, anchor)[:n_lanes]):
                _copy_lane_feature(lanes[b, j], lane_valid[b, j], feat[0], feat[1], lane_points, gameformer=False, src_valid=feat[3] if len(feat) > 3 else None)
        for j, feat in enumerate(cross_feats[:n_crosswalks]):
            _copy_cross_feature(cross[b, j], cross_valid[b, j], feat[0], cross_points, src_valid=feat[3] if len(feat) > 3 else None)
    return (lanes, cross, lane_valid, cross_valid) if return_valid else (lanes, cross)


def candidates_to_dtpp_tree(candidates: torch.Tensor) -> torch.Tensor:
    """COWP candidate [B,K,T,7] -> DTPP ego tree [B,K,T,6]."""
    B, K, T = candidates.shape[:3]
    out = torch.zeros(B, K, T, 6, device=candidates.device, dtype=candidates.dtype)
    if K == 0 or T == 0:
        return out
    out[..., 0] = candidates[..., 0]
    out[..., 1] = candidates[..., 1]
    out[..., 2] = candidates[..., 2] if candidates.shape[-1] > 2 else 0.0
    vx = candidates[..., 3] if candidates.shape[-1] > 3 else torch.zeros_like(out[..., 0])
    vy = candidates[..., 4] if candidates.shape[-1] > 4 else torch.zeros_like(out[..., 0])
    speed = torch.linalg.norm(torch.stack([vx, vy], dim=-1), dim=-1)
    out[..., 3] = speed
    dt = 0.1
    accel = torch.zeros_like(speed)
    if T > 1:
        accel[..., 1:] = (speed[..., 1:] - speed[..., :-1]) / dt
    out[..., 4] = accel
    yaw = out[..., 2]
    dyaw = torch.zeros_like(yaw)
    if T > 1:
        dyaw[..., 1:] = torch.atan2(torch.sin(yaw[..., 1:] - yaw[..., :-1]), torch.cos(yaw[..., 1:] - yaw[..., :-1]))
    ds = torch.clamp(speed * dt, min=1e-3)
    out[..., 5] = dyaw / ds
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _build_route(batch: Mapping[str, torch.Tensor], device: torch.device, B: int, *, route_points: int = 80, origin: torch.Tensor | None = None, yaw0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    route = torch.zeros(B, route_points, 4, device=device)
    route_valid = torch.zeros(B, route_points, device=device, dtype=torch.bool)
    xyz = _named_tensor(batch, "path_samples/xyz")
    valid_t = _named_tensor(batch, "path_samples/valid")
    on_route = _named_tensor(batch, "path_samples/on_route")
    if xyz is not None:
        arr = xyz.float().to(device)
        if arr.ndim == 3:
            arr = arr.unsqueeze(0)
        # [B,P,Q,3]
        if arr.ndim == 4:
            Pn, Q = arr.shape[1], arr.shape[2]
            v = valid_t.bool().to(device) if valid_t is not None else torch.isfinite(arr[..., :2]).reshape(B, Pn, Q, -1).all(dim=-1)
            if v.ndim == 3:
                pass
            elif v.ndim == 2:
                v = v.reshape(B, Pn, Q)
            on = on_route.bool().to(device) if on_route is not None else torch.ones(B, Pn, 1, device=device, dtype=torch.bool)
            if on.ndim == 2:
                on = on.unsqueeze(-1)
            for b in range(min(B, arr.shape[0])):
                choices = torch.nonzero(on[b].reshape(-1), as_tuple=False).flatten()
                pidx = int(choices[0].detach().cpu()) if choices.numel() else 0
                pts = arr[b, pidx, :, :2]
                vv = v[b, pidx] & torch.isfinite(pts).reshape(pts.shape[0], -1).all(dim=-1)
                if origin is not None and yaw0 is not None:
                    pts = _xy_to_ego_frame(pts.unsqueeze(0), origin[b:b+1], yaw0[b:b+1]).squeeze(0)
                m = min(route_points, int(pts.shape[0]))
                route[b, :m, :2] = torch.nan_to_num(pts[:m], nan=0.0, posinf=0.0, neginf=0.0)
                route[b, :m, 3] = vv[:m].float()
                route_valid[b, :m] = vv[:m]
            return route, route_valid
    rg = _roadgraph_xy(batch, device, origin=origin, yaw0=yaw0)
    if rg is not None:
        valid_rg = torch.linalg.norm(rg, dim=-1).isfinite()
        for b in range(B):
            pts = rg[b]
            vv = valid_rg[b]
            m = min(route_points, int(pts.shape[0]))
            route[b, :m, :2] = pts[:m]
            route[b, :m, 3] = vv[:m].float()
            route_valid[b, :m] = vv[:m]
    return route, route_valid


@dataclass
class ExternalBatch:
    gameformer_inputs: dict[str, torch.Tensor]
    dtpp_inputs: dict[str, torch.Tensor]
    planner_inputs: dict[str, torch.Tensor]
    ego_future_xy: torch.Tensor
    ego_future_valid: torch.Tensor
    neighbors_future_xy: torch.Tensor
    neighbors_future_valid: torch.Tensor
    candidates: torch.Tensor
    candidate_valid: torch.Tensor
    conventional_safe: torch.Tensor
    dtpp_candidate_tree: torch.Tensor
    neighbor_indices: torch.Tensor
    sdc_indices: torch.Tensor
    origin: torch.Tensor
    yaw0: torch.Tensor
    sdc_current_valid: torch.Tensor


def make_external_batch(
    batch: Mapping[str, torch.Tensor],
    cfg: dict,
    *,
    device: torch.device,
    max_neighbors: int = 10,
    max_candidates: int = 30,
    horizon: int | None = None,
    baseline: str | None = None,
    require_candidates: bool = True,
    require_future: bool = True,
) -> ExternalBatch:
    # Build canonical COWP history [B,N,Th,11].
    hist = first_tensor(batch, ("state/history", "womd/state/history"))
    if hist is not None:
        agent_history = hist.float().to(device)
        mask_t = first_tensor(batch, ("state/agent_valid", "womd/state/agent_valid", "state/current/valid", "womd/state/current/valid"))
        agent_mask = mask_t.bool().to(device) if mask_t is not None and mask_t.shape[:2] == agent_history.shape[:2] else (agent_history[..., -1, 10] > 0.5)
    else:
        agent_history, agent_mask = build_agent_history_from_womd(batch, max_agents=int(cfg.get("limits", {}).get("max_agents", cfg.get("model", {}).get("max_agents", 128))), history_steps=int(cfg.get("model", {}).get("history_steps", 11)), d_state=11)
        agent_history = agent_history.to(device)
        agent_mask = agent_mask.to(device)
    B, N, Th, _ = agent_history.shape
    horizon = int(horizon or cfg.get("time", {}).get("future_steps", 80))
    sdc = sdc_indices(batch, N, B, device)
    neighbors_idx = _select_neighbors(torch.nan_to_num(agent_history, nan=0.0, posinf=0.0, neginf=0.0), agent_mask, sdc, max_neighbors)
    ego_idx = sdc[:, None]
    all_pred_idx = torch.cat([ego_idx, neighbors_idx], dim=1)
    rows = torch.arange(B, device=device)
    sdc_safe = sdc.clamp(0, N - 1)
    sdc_current = agent_history[rows, sdc_safe, -1]
    sdc_current_valid = (sdc_current[..., 10] > 0.5) & torch.isfinite(sdc_current).all(dim=-1)
    origin = torch.where(sdc_current_valid[:, None], torch.nan_to_num(sdc_current[:, :2], nan=0.0, posinf=0.0, neginf=0.0), torch.zeros(B, 2, device=device))
    yaw0 = torch.where(sdc_current_valid, torch.nan_to_num(sdc_current[:, 6], nan=0.0, posinf=0.0, neginf=0.0), torch.zeros(B, device=device))

    typ = _agent_type(batch, N, device)
    selected_types = gather_rows(typ.unsqueeze(-1), all_pred_idx).squeeze(-1) if typ is not None else None
    selected_hist_raw = gather_rows(agent_history, all_pred_idx)
    selected_hist = _history_to_ego_frame(torch.nan_to_num(selected_hist_raw, nan=0.0, posinf=0.0, neginf=0.0), origin, yaw0)
    selected_hist = torch.where(sdc_current_valid[:, None, None, None], selected_hist, torch.zeros_like(selected_hist))
    actor_valid = (selected_hist_raw[..., 10] > 0.5) & torch.isfinite(selected_hist_raw).reshape(B, selected_hist_raw.shape[1], selected_hist_raw.shape[2], -1).all(dim=-1)
    actor_valid = actor_valid & sdc_current_valid[:, None, None]
    gf_state = history_to_gameformer_state(selected_hist, selected_types)
    gf_state = torch.where(actor_valid[..., None], gf_state, torch.zeros_like(gf_state))
    gf_ego = gf_state[:, 0]
    gf_neighbors = gf_state[:, 1:]
    agent_cur_xy = selected_hist_raw[:, :, -1, :2]
    gf_lanes, gf_cross, gf_lane_valid, gf_cross_valid = build_gameformer_map(
        batch, max_neighbors + 1, device, origin=origin, yaw0=yaw0, agent_xy=agent_cur_xy, agent_valid=actor_valid.any(dim=-1), return_valid=True
    )

    ego_hist = selected_hist[:, 0]
    neigh_hist = selected_hist[:, 1:]
    neigh_types = gather_rows(typ.unsqueeze(-1), neighbors_idx).squeeze(-1) if typ is not None else None
    dtpp_ego = history_to_dtpp_ego(ego_hist)
    dtpp_neighbors = history_to_dtpp_neighbors(neigh_hist, neigh_types)
    dtpp_lanes, dtpp_cross, dtpp_lane_valid, dtpp_cross_valid = build_dtpp_map(batch, device, origin=origin, yaw0=yaw0, return_valid=True)
    route, route_valid = _build_route(batch, device, B, route_points=max(horizon, 1), origin=origin, yaw0=yaw0)

    if require_future:
        future_xy, future_valid = _future_xy_for_selected(batch, all_pred_idx, N, horizon, device)
        future_xy = _xy_to_ego_frame(torch.nan_to_num(future_xy, nan=0.0, posinf=0.0, neginf=0.0), origin, yaw0)
        future_valid = future_valid & sdc_current_valid[:, None, None]
        ego_future_xy = future_xy[:, 0]
        ego_future_valid = future_valid[:, 0]
        neighbors_future_xy = future_xy[:, 1:]
        neighbors_future_valid = future_valid[:, 1:]
    else:
        ego_future_xy = torch.zeros(B, 0, 2, device=device)
        ego_future_valid = torch.zeros(B, 0, device=device, dtype=torch.bool)
        neighbors_future_xy = torch.zeros(B, max_neighbors, 0, 2, device=device)
        neighbors_future_valid = torch.zeros(B, max_neighbors, 0, device=device, dtype=torch.bool)

    cand_src = batch.get("cowp/candidates/trajectory")
    cand_valid_src = batch.get("cowp/candidates/valid")
    if cand_src is None or cand_valid_src is None:
        if require_candidates:
            raise KeyError("External candidate/tree evaluation requires cowp/candidates/trajectory and cowp/candidates/valid")
        cand = torch.zeros(B, 0, horizon, 7, device=device)
        cand_valid = torch.zeros(B, 0, device=device, dtype=torch.bool)
    else:
        cand_src_device = cand_src.float().to(device)
        raw_geom_finite = candidate_geometry_finite(cand_src_device)
        cand = _candidate_to_ego_frame(torch.nan_to_num(cand_src_device, nan=0.0, posinf=0.0, neginf=0.0), origin, yaw0)
        cand_valid = cand_valid_src.bool().to(device)
        if cand.shape[1] > max_candidates:
            cand = cand[:, :max_candidates]
            cand_valid = cand_valid[:, :max_candidates]
        elif cand.shape[1] < max_candidates and require_candidates:
            pad_k = max_candidates - cand.shape[1]
            cand = torch.cat([cand, torch.zeros(B, pad_k, cand.shape[2], cand.shape[3], device=device, dtype=cand.dtype)], dim=1)
            cand_valid = torch.cat([cand_valid, torch.zeros(B, pad_k, device=device, dtype=torch.bool)], dim=1)
        if cand.shape[2] > horizon:
            cand = cand[:, :, :horizon]
        elif cand.shape[2] < horizon:
            pad_t = horizon - cand.shape[2]
            cand = torch.cat([cand, torch.zeros(B, cand.shape[1], pad_t, cand.shape[3], device=device, dtype=cand.dtype)], dim=2)
        adapter_geom_finite = candidate_geometry_finite(cand)
        raw_geom_finite = raw_geom_finite[:, : cand.shape[1]] if raw_geom_finite.shape[1] >= cand.shape[1] else torch.cat([raw_geom_finite, torch.zeros(B, cand.shape[1] - raw_geom_finite.shape[1], device=device, dtype=torch.bool)], dim=1)
        geom_finite = raw_geom_finite & adapter_geom_finite
        cand_valid = cand_valid[:, : cand.shape[1]] & geom_finite & sdc_current_valid[:, None]
        cand = torch.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0)
    conventional = batch.get("cowp/candidates/conventional_safe")
    if conventional is not None and cand.shape[1] > 0:
        conventional = conventional.bool().to(device)
        if conventional.shape[1] >= cand_valid.shape[1]:
            conventional = conventional[:, : cand_valid.shape[1]]
        else:
            conventional = torch.cat([conventional, torch.zeros(B, cand_valid.shape[1] - conventional.shape[1], device=device, dtype=torch.bool)], dim=1)
        conventional = conventional & cand_valid
    else:
        conventional = cand_valid.clone()
    dtpp_tree = candidates_to_dtpp_tree(cand)
    planner_inputs = {
        "agents": gf_state,
        "agent_valid": actor_valid,
        "map_lanes": dtpp_lanes,
        "map_lanes_valid": dtpp_lane_valid,
        "route": route,
        "neighbors_future_xy": neighbors_future_xy,
        "neighbors_future_valid": neighbors_future_valid,
    }
    gameformer_inputs = {
        "ego_state": gf_ego,
        "neighbors_state": gf_neighbors,
        "actors_valid": actor_valid,
        "map_lanes": gf_lanes,
        "map_crosswalks": gf_cross,
        "map_lanes_valid": gf_lane_valid,
        "map_crosswalks_valid": gf_cross_valid,
    }
    dtpp_inputs = {
        "ego_agent_past": dtpp_ego,
        "neighbor_agents_past": dtpp_neighbors,
        "map_lanes": dtpp_lanes,
        "map_crosswalks": dtpp_cross,
        "map_lanes_valid": dtpp_lane_valid,
        "map_crosswalks_valid": dtpp_cross_valid,
    }
    return ExternalBatch(
        gameformer_inputs=gameformer_inputs,
        dtpp_inputs=dtpp_inputs,
        planner_inputs=planner_inputs,
        ego_future_xy=ego_future_xy,
        ego_future_valid=ego_future_valid,
        neighbors_future_xy=neighbors_future_xy,
        neighbors_future_valid=neighbors_future_valid,
        candidates=cand,
        candidate_valid=cand_valid,
        conventional_safe=conventional,
        dtpp_candidate_tree=dtpp_tree,
        neighbor_indices=neighbors_idx,
        sdc_indices=sdc,
        origin=origin,
        yaw0=yaw0,
        sdc_current_valid=sdc_current_valid,
    )


def best_candidate_to_logged_ego(candidates: torch.Tensor, candidate_valid: torch.Tensor, ego_future_xy: torch.Tensor, ego_future_valid: torch.Tensor) -> torch.Tensor:
    if candidates.shape[1] == 0:
        return torch.zeros(candidates.shape[0], dtype=torch.long, device=candidates.device)
    valid_t = ego_future_valid[:, None, :].float()
    target = torch.where(ego_future_valid[:, None, :, None].bool(), ego_future_xy[:, None], candidates[:, :, :, :2].detach())
    diff = (candidates[..., :2] - target) * valid_t[..., None]
    denom = valid_t.sum(dim=-1).clamp_min(1.0)
    ade = torch.linalg.norm(diff, dim=-1).sum(dim=-1) / denom
    ade = torch.where(candidate_valid, ade, torch.full_like(ade, 1e6))
    return torch.argmin(ade, dim=1)


def label_from_batch_item(batch: Mapping[str, torch.Tensor], i: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k in EVAL_LABEL_KEYS:
        v = batch.get(k)
        if torch.is_tensor(v):
            try:
                out[k] = v[i].detach().cpu().numpy()
            except Exception:
                pass
    return out
