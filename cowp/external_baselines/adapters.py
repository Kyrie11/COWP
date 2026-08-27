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
    "path_samples/",
    "womd/path_samples/",
    "traffic_light_state/",
    "womd/traffic_light_state/",
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


# Training does not need COWP witness/transport tensors for direct planners.
# Keeping the whitelist narrow materially reduces NPZ decompression and IPC cost.
EXTERNAL_TRAIN_COMMON_PREFIXES = {
    "state/",
    "womd/state/",
    "roadgraph_samples/",
    "womd/roadgraph_samples/",
    "path_samples/",
    "womd/path_samples/",
}

EXTERNAL_TRAIN_DTPP_PREFIXES = EXTERNAL_TRAIN_COMMON_PREFIXES | {
    "cowp/candidates/trajectory",
    "cowp/candidates/valid",
}


def candidate_geometry_finite(candidates: torch.Tensor) -> torch.Tensor:
    """Return a [..candidate-prefix..] finite mask without tuple-dim reductions.

    Some deployed COWP environments use a PyTorch build whose ``Tensor.all``
    accepts only a single integer ``dim``.  In particular,
    a multi-axis ``Tensor.all`` reduction raises ``TypeError`` before DTPP reaches its
    forward pass.  Flattening the trajectory/state tail and reducing one axis
    is equivalent and works across older and newer supported PyTorch builds.
    """
    if candidates.ndim < 2:
        raise ValueError(f"candidate tensor must have at least 2 dims, got {tuple(candidates.shape)}")
    finite = torch.isfinite(candidates)
    return finite.reshape(*finite.shape[:-2], -1).all(dim=-1)


def external_wanted_prefixes(*, baseline: str | None = None, purpose: str = "audit", include_waymax_outcomes: bool = True) -> set[str]:
    """Return the smallest key set needed by an external-baseline phase.

    ``purpose='train'`` intentionally excludes COWP mechanism labels.  DTPP keeps
    only the proposal trajectory/valid tensors because its current WOMD adapter is
    a proposal-tree scorer.  ``purpose='audit'`` preserves the old broad contract.
    """
    purpose = str(purpose).lower()
    baseline = str(baseline).lower() if baseline is not None else None
    if purpose == "train":
        wanted = set(EXTERNAL_TRAIN_DTPP_PREFIXES if baseline == "dtpp" else EXTERNAL_TRAIN_COMMON_PREFIXES)
    else:
        wanted = set(EXTERNAL_WANTED_PREFIXES)
    if not include_waymax_outcomes:
        wanted = {x for x in wanted if not x.startswith("waymax/")}
    return wanted


class ExternalCOWPDataset(Dataset):
    """COWP npz dataset for GameFormer/DTPP adapters.

    The normal staged COWP dataset avoids loading dense raw WOMD future/map
    tensors.  GameFormer/DTPP need logged ego/neighbor futures and optionally
    vectorized map tokens, so this dataset uses a custom key whitelist while
    preserving the same NPZ loading/canonicalization path as COWP.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        pattern: str = "*.npz",
        include_waymax_outcomes: bool = True,
        *,
        baseline: str | None = None,
        purpose: str = "audit",
    ):
        self.base = COWPNpzDataset(cache_dir, pattern)
        self.baseline = str(baseline).lower() if baseline is not None else None
        self.purpose = str(purpose).lower()
        self._wanted = external_wanted_prefixes(
            baseline=self.baseline, purpose=self.purpose, include_waymax_outcomes=include_waymax_outcomes
        )

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


def external_map_topology_report(sample: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Summarize whether one cache sample preserves WOMD vector-map topology.

    The V6 learned-map adapters require the original WOMD roadgraph feature
    identity/type/direction fields.  V5 could silently accept a cache containing
    only coordinates and then reconstruct fake polylines by point position.  A
    lightweight report lets the train/eval entry points fail *before* a long run
    if a production cache would fall back to that legacy behavior.

    This function intentionally validates representation only; numerical map
    validity remains the responsibility of :func:`_roadgraph_feature_bundle`.
    """

    def pick(*names: str) -> torch.Tensor | None:
        for name in names:
            value = sample.get(name)
            if torch.is_tensor(value):
                return value
        return None

    xyz = pick("roadgraph_samples/xyz", "womd/roadgraph_samples/xyz")
    x = pick("roadgraph_samples/x", "womd/roadgraph_samples/x")
    y = pick("roadgraph_samples/y", "womd/roadgraph_samples/y")
    ids = pick("roadgraph_samples/id", "womd/roadgraph_samples/id")
    types = pick("roadgraph_samples/type", "womd/roadgraph_samples/type")
    direction = pick("roadgraph_samples/dir", "womd/roadgraph_samples/dir")
    valid = pick("roadgraph_samples/valid", "womd/roadgraph_samples/valid")

    def scalar_points(value: torch.Tensor | None) -> int:
        return int(value.numel()) if value is not None else 0

    def vector_points(value: torch.Tensor | None, width: int) -> int:
        if value is None:
            return 0
        if value.ndim >= 2 and int(value.shape[-1]) == int(width):
            return int(value.numel() // width)
        return int(value.numel() // width) if value.numel() % width == 0 else 0

    xyz_points = vector_points(xyz, 3)
    if xyz_points <= 0 and x is not None and y is not None:
        xyz_points = min(scalar_points(x), scalar_points(y))
    id_points = scalar_points(ids)
    type_points = scalar_points(types)
    valid_points = scalar_points(valid)
    dir_points = vector_points(direction, 3)
    point_count = max(xyz_points, id_points, type_points, valid_points, dir_points)

    aligned = bool(
        point_count > 0
        and xyz_points == point_count
        and id_points == point_count
        and type_points == point_count
        and valid_points == point_count
        and dir_points == point_count
    )
    return {
        "has_xy": bool(xyz_points > 0),
        "has_id": ids is not None,
        "has_type": types is not None,
        "has_dir": direction is not None,
        "has_valid": valid is not None,
        "aligned": aligned,
        "points": int(point_count),
        "xyz_points": int(xyz_points),
        "id_points": int(id_points),
        "type_points": int(type_points),
        "dir_points": int(dir_points),
        "valid_points": int(valid_points),
    }


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
        scores = torch.nan_to_num(is_sdc.float().to(device), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.argmax(scores, dim=1).long()
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
    return torch.nan_to_num(typ.to(device), nan=0.0, posinf=0.0, neginf=0.0) if typ is not None else None


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


def _roadgraph_xy(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    origin: torch.Tensor | None = None,
    yaw0: torch.Tensor | None = None,
    *,
    max_points: int | None = None,
) -> torch.Tensor | None:
    xyz = first_tensor(batch, ("roadgraph_samples/xyz", "womd/roadgraph_samples/xyz"))
    if xyz is not None:
        arr = xyz.float()
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        arr = arr[..., :2].reshape(arr.shape[0], -1, 2)
        if max_points is not None and arr.shape[1] > int(max_points):
            arr = arr[:, : int(max_points)]
        arr = arr.to(device, non_blocking=True)
    else:
        x = first_tensor(batch, ("roadgraph_samples/x", "womd/roadgraph_samples/x"))
        y = first_tensor(batch, ("roadgraph_samples/y", "womd/roadgraph_samples/y"))
        if x is None or y is None:
            return None
        x = x.float()
        y = y.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if max_points is not None and x.shape[1] > int(max_points):
            x = x[:, : int(max_points)]
            y = y[:, : int(max_points)]
        arr = torch.stack([x, y], dim=-1).to(device, non_blocking=True)

    # Preserve padded/invalid roadgraph tokens.  Translating a zero-padding point
    # by -ego_origin turns it into a huge fake map point and was another source of
    # unstable external-baseline inputs.
    valid_t = first_tensor(batch, ("roadgraph_samples/valid", "womd/roadgraph_samples/valid"))
    finite_geometry = torch.isfinite(arr).all(dim=-1)
    if valid_t is not None:
        valid = valid_t.bool()
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        valid = valid.reshape(valid.shape[0], -1)[:, : arr.shape[1]].to(device, non_blocking=True)
        valid = valid & finite_geometry
    else:
        valid = finite_geometry & (torch.linalg.norm(torch.nan_to_num(arr), dim=-1) > 1.0e-6)
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if origin is not None and yaw0 is not None:
        arr = _xy_to_ego_frame(arr, origin, yaw0)
    arr = torch.where(valid[..., None], arr, torch.zeros_like(arr))
    return arr



def build_route_centerline(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    origin: torch.Tensor,
    yaw0: torch.Tensor,
    points: int = 160,
) -> torch.Tensor:
    """Return ego-frame route samples [B,points,4]=(x,y,heading,valid).

    WOMD 1.3.1 stores 45 SDC paths with 800 points/path when
    ``include_sdc_paths=True``.  We prefer a valid ``on_route`` path.  Older
    caches without SDC paths fall back to nearby forward roadgraph samples.
    """
    B = origin.shape[0]
    out = torch.zeros(B, points, 4, device=device, dtype=torch.float32)
    xyz = first_tensor(batch, ("path_samples/xyz", "womd/path_samples/xyz"))
    valid_t = first_tensor(batch, ("path_samples/valid", "womd/path_samples/valid"))
    on_route_t = first_tensor(batch, ("path_samples/on_route", "womd/path_samples/on_route"))
    if xyz is not None and valid_t is not None:
        xyz = xyz.float().to(device)
        valid = valid_t.bool().to(device)
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        # Canonicalize to [B,P,Q,*].
        if xyz.ndim >= 4:
            paths_xy = xyz[..., :2]
            paths_valid = valid.reshape(B, xyz.shape[1], -1)[..., : xyz.shape[2]]
        else:
            flat_xy = xyz.reshape(B, -1, 3)[..., :2] if xyz.shape[-1] != 2 else xyz.reshape(B, -1, 2)
            flat_valid = valid.reshape(B, -1)
            if on_route_t is not None:
                onr = on_route_t.to(device).reshape(B, -1)
                n_paths = max(int(onr.shape[1]), 1)
            else:
                n_paths = 45 if flat_xy.shape[1] % 45 == 0 else 1
                onr = torch.zeros(B, n_paths, device=device)
            q = max(flat_xy.shape[1] // n_paths, 1)
            paths_xy = flat_xy[:, : n_paths * q].reshape(B, n_paths, q, 2)
            paths_valid = flat_valid[:, : n_paths * q].reshape(B, n_paths, q)
        # Source validity is necessary but not sufficient: a marked-valid NaN/Inf
        # path point must be excluded before nearest-point and heading operations.
        paths_valid = paths_valid & torch.isfinite(paths_xy).all(dim=-1)
        paths_xy = torch.nan_to_num(paths_xy, nan=0.0, posinf=0.0, neginf=0.0)
        P = paths_xy.shape[1]
        if on_route_t is not None:
            onr = on_route_t.to(device).reshape(B, -1)[:, :P]
        else:
            onr = torch.zeros(B, P, device=device)
        for b in range(B):
            usable = paths_valid[b].sum(dim=-1)
            preference = (onr[b] > 0.5).float() * 1e6 + usable.float()
            pi = int(preference.argmax().item()) if P else 0
            mask = paths_valid[b, pi]
            pts = paths_xy[b, pi][mask]
            if pts.numel() == 0:
                continue
            pts = _xy_to_ego_frame(pts[None], origin[b:b+1], yaw0[b:b+1])[0]
            # Keep the path segment nearest the ego, then uniformly sample ahead.
            start = int(torch.linalg.norm(pts, dim=-1).argmin().item())
            pts = pts[start:]
            if pts.shape[0] > points:
                idx = torch.linspace(0, pts.shape[0] - 1, points, device=device).round().long()
                pts = pts[idx]
            n = min(points, pts.shape[0])
            out[b, :n, :2] = pts[:n]
            if n > 1:
                d = torch.zeros(n, 2, device=device)
                d[:-1] = pts[1:n] - pts[:n-1]
                d[-1] = d[-2]
                out[b, :n, 2] = torch.atan2(d[:, 1], d[:, 0])
            out[b, :n, 3] = 1.0
    missing = out[..., 3].sum(dim=-1) < 2
    if bool(missing.any()):
        rg = _roadgraph_xy(batch, device, origin=origin, yaw0=yaw0)
        if rg is not None:
            for b in torch.where(missing)[0].tolist():
                pts = rg[b]
                finite = torch.isfinite(pts).all(dim=-1) & (torch.linalg.norm(pts, dim=-1) > 1e-4)
                pts = pts[finite]
                if pts.numel() == 0:
                    continue
                # Approximate a route from roadgraph samples in front of the ego.
                pts = pts[(pts[:, 0] > -5.0) & (pts[:, 0] < 150.0) & (pts[:, 1].abs() < 15.0)]
                if pts.numel() == 0:
                    continue
                order = torch.argsort(pts[:, 0] + 0.05 * pts[:, 1].abs())
                pts = pts[order][:points]
                n = pts.shape[0]
                out[b, :n, :2] = pts
                if n > 1:
                    d = torch.zeros(n, 2, device=device)
                    d[:-1] = pts[1:] - pts[:-1]
                    d[-1] = d[-2]
                    out[b, :n, 2] = torch.atan2(d[:, 1], d[:, 0])
                out[b, :n, 3] = 1.0
    return out

def _reshape_roadgraph_scalar(x: torch.Tensor | None, batch_size: int, points: int, device: torch.device) -> torch.Tensor | None:
    if x is None:
        return None
    y = x
    if y.ndim == 0:
        return None
    if y.ndim == 1:
        if batch_size != 1:
            return None
        y = y.unsqueeze(0)
    if y.shape[0] != batch_size:
        if batch_size == 1:
            y = y.reshape(1, -1)
        else:
            return None
    y = y.reshape(batch_size, -1)[:, :points]
    return y.to(device, non_blocking=True)


def _reshape_roadgraph_vector(x: torch.Tensor | None, batch_size: int, points: int, device: torch.device) -> torch.Tensor | None:
    """Canonicalize roadgraph xyz/dir to [B,P,3] without guessing feature topology."""
    if x is None:
        return None
    y = x.float()
    if y.ndim == 1:
        if batch_size != 1 or y.numel() % 3:
            return None
        y = y.reshape(1, -1, 3)
    elif y.ndim == 2:
        if y.shape[0] == batch_size and y.shape[1] % 3 == 0:
            y = y.reshape(batch_size, -1, 3)
        elif batch_size == 1 and y.shape[-1] >= 3:
            y = y.unsqueeze(0)
        else:
            return None
    else:
        if y.shape[0] != batch_size:
            return None
        if y.shape[-1] < 3:
            flat = y.reshape(batch_size, -1)
            if flat.shape[1] % 3:
                return None
            y = flat.reshape(batch_size, -1, 3)
        else:
            y = y.reshape(batch_size, -1, y.shape[-1])[..., :3]
    return y[:, :points].to(device, non_blocking=True)


def _roadgraph_valid_mask(batch: Mapping[str, torch.Tensor], device: torch.device, max_points: int) -> torch.Tensor | None:
    """Return source validity AND *raw* finite geometry.

    Do not derive finiteness from ``_roadgraph_xy``: that helper deliberately
    ``nan_to_num`` sanitizes coordinates before transforming them, so checking it
    afterwards would incorrectly resurrect a declared-valid NaN point.
    """
    valid_t = first_tensor(batch, ("roadgraph_samples/valid", "womd/roadgraph_samples/valid"))
    xyz_t = first_tensor(batch, ("roadgraph_samples/xyz", "womd/roadgraph_samples/xyz"))

    # Infer batch size from whichever roadgraph tensor is present.
    src = xyz_t if xyz_t is not None else valid_t
    if src is None:
        x_t = first_tensor(batch, ("roadgraph_samples/x", "womd/roadgraph_samples/x"))
        src = x_t
    if src is None:
        return None
    B = int(src.shape[0]) if src.ndim > 1 else 1

    finite = None
    if xyz_t is not None:
        raw_xyz = _reshape_roadgraph_vector(xyz_t, B, int(max_points), device)
        if raw_xyz is not None:
            finite = torch.isfinite(raw_xyz[..., :2]).all(dim=-1)
    else:
        x_t = first_tensor(batch, ("roadgraph_samples/x", "womd/roadgraph_samples/x"))
        y_t = first_tensor(batch, ("roadgraph_samples/y", "womd/roadgraph_samples/y"))
        if x_t is not None and y_t is not None:
            x = _reshape_roadgraph_scalar(x_t, B, int(max_points), device)
            y = _reshape_roadgraph_scalar(y_t, B, int(max_points), device)
            if x is not None and y is not None:
                n = min(x.shape[1], y.shape[1], int(max_points))
                finite = torch.isfinite(x[:, :n]) & torch.isfinite(y[:, :n])
    if finite is None:
        return None

    if valid_t is None:
        return finite
    valid = _reshape_roadgraph_scalar(valid_t, B, finite.shape[1], device)
    if valid is None:
        return finite
    n = min(finite.shape[1], valid.shape[1])
    return valid[:, :n].bool() & finite[:, :n]


def _roadgraph_feature_bundle(
    batch: Mapping[str, torch.Tensor], device: torch.device, *,
    origin: torch.Tensor | None, yaw0: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return local roadgraph (xy, valid, id, type, heading) tensors.

    WOMD explicitly provides ``roadgraph_samples/id`` as the vector-map feature id
    and ``roadgraph_samples/dir`` as the direction to the next sample.  V5 ignored
    both and sliced the flat roadgraph stream by point count, which can splice
    unrelated lane centers/road lines/crosswalks into one fake polyline.  Learned
    vector-map encoders are especially sensitive to that topology error.
    """
    xy = _roadgraph_xy(batch, device, origin=origin, yaw0=yaw0, max_points=None)
    if xy is None:
        return None, None, None, None, None
    B, P = xy.shape[:2]
    valid = _roadgraph_valid_mask(batch, device, P)
    if valid is None:
        valid = torch.isfinite(xy).all(dim=-1)
    if valid.shape[1] < P:
        pad = torch.zeros(B, P - valid.shape[1], dtype=torch.bool, device=device)
        valid = torch.cat([valid, pad], dim=1)
    else:
        valid = valid[:, :P]

    ids = _reshape_roadgraph_scalar(
        first_tensor(batch, ("roadgraph_samples/id", "womd/roadgraph_samples/id")), B, P, device
    )
    types = _reshape_roadgraph_scalar(
        first_tensor(batch, ("roadgraph_samples/type", "womd/roadgraph_samples/type")), B, P, device
    )
    if ids is not None:
        ids = ids.long()
    if types is not None:
        types = types.long()

    direction = _reshape_roadgraph_vector(
        first_tensor(batch, ("roadgraph_samples/dir", "womd/roadgraph_samples/dir")), B, P, device
    )
    heading = None
    if direction is not None:
        dxy = direction[..., :2]
        finite_dir = torch.isfinite(dxy).all(dim=-1)
        dxy = torch.nan_to_num(dxy, nan=0.0, posinf=0.0, neginf=0.0)
        if yaw0 is not None:
            dxy = _rotate_global_to_local(dxy, yaw0)
        heading = torch.atan2(dxy[..., 1], dxy[..., 0])
        heading = torch.where(finite_dir & valid, heading, torch.zeros_like(heading))
    return xy, valid, ids, types, heading


def _polyline_heading(xy: torch.Tensor, source_heading: torch.Tensor | None = None) -> torch.Tensor:
    if source_heading is not None:
        return source_heading
    h = torch.zeros(xy.shape[0], device=xy.device, dtype=xy.dtype)
    if xy.shape[0] > 1:
        d = xy[1:] - xy[:-1]
        hd = torch.atan2(d[:, 1], d[:, 0])
        h[:-1] = hd
        h[-1] = hd[-1]
    return h


def _sample_polyline_indices(indices: torch.Tensor, max_points: int) -> torch.Tensor:
    """Keep source order and uniformly subsample a long WOMD map feature."""
    n = int(indices.numel())
    if n <= max_points:
        return indices
    pos = torch.linspace(0, n - 1, max_points, device=indices.device).round().long()
    return indices[pos]


def _nearest_feature_ids(
    xy: torch.Tensor,
    valid: torch.Tensor,
    ids: torch.Tensor,
    types: torch.Tensor | None,
    anchors: torch.Tensor,
    *,
    allowed_types: tuple[int, ...] | None,
    max_features: int,
    radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank vector-map feature ids by minimum point-to-anchor distance.

    Returns (feature_ids [A,K], feature_dist2 [A,K], selected_point_mask_source).
    ``scatter_reduce_`` avoids a Python loop over hundreds of roadgraph ids.
    """
    mask = valid.clone()
    if types is not None and allowed_types is not None:
        type_mask = torch.zeros_like(mask)
        for typ in allowed_types:
            type_mask |= types == int(typ)
        mask &= type_mask
    point_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if point_idx.numel() == 0 or anchors.numel() == 0:
        return (
            torch.empty(anchors.shape[0], 0, dtype=ids.dtype, device=ids.device),
            torch.empty(anchors.shape[0], 0, dtype=xy.dtype, device=xy.device),
            point_idx,
        )
    point_ids = ids[point_idx]
    unique_ids, inv = torch.unique(point_ids, sorted=False, return_inverse=True)
    pts = xy[point_idx]
    dist2 = ((pts[:, None, :] - anchors[None, :, :]) ** 2).sum(dim=-1)
    feature_dist2 = torch.full(
        (unique_ids.shape[0], anchors.shape[0]), float("inf"), device=xy.device, dtype=xy.dtype
    )
    # scatter_reduce_ exists in every PyTorch version supported by this project;
    # it computes the min point distance for each map feature in one operation.
    feature_dist2.scatter_reduce_(
        0, inv[:, None].expand(-1, anchors.shape[0]), dist2, reduce="amin", include_self=True
    )
    k = min(int(max_features), int(unique_ids.shape[0]))
    if k <= 0:
        return (
            torch.empty(anchors.shape[0], 0, dtype=ids.dtype, device=ids.device),
            torch.empty(anchors.shape[0], 0, dtype=xy.dtype, device=xy.device),
            point_idx,
        )
    vals, feat_idx = torch.topk(feature_dist2.transpose(0, 1), k=k, largest=False, dim=1)
    selected_ids = unique_ids[feat_idx]
    radius2 = float(radius_m) * float(radius_m)
    selected_ids = torch.where(vals <= radius2, selected_ids, torch.full_like(selected_ids, torch.iinfo(selected_ids.dtype).min))
    return selected_ids, vals, point_idx


def _legacy_gameformer_flat_map(
    batch: Mapping[str, torch.Tensor], device: torch.device, *, B: int, num_agents_to_predict: int,
    n_lanes: int, lane_points: int, origin: torch.Tensor | None, yaw0: torch.Tensor | None,
    lanes: torch.Tensor, lanes_valid: torch.Tensor,
) -> None:
    """Compatibility fallback for synthetic/legacy caches lacking roadgraph ids."""
    rg = _roadgraph_xy(batch, device, origin=origin, yaw0=yaw0, max_points=lane_points)
    rg_valid = _roadgraph_valid_mask(batch, device, lane_points)
    if rg is None or rg.shape[1] == 0:
        return
    P = min(lane_points, rg.shape[1])
    xy = rg[:, :P]
    valid = rg_valid[:, :P] if rg_valid is not None else torch.isfinite(xy).all(dim=-1)
    for b in range(B):
        h = _polyline_heading(xy[b])
        lanes[b, :, 0, :P, 0:2] = xy[b][None]
        lanes[b, :, 0, :P, 2] = h[None]
        lanes_valid[b, :, 0, :P] = valid[b][None]


def build_gameformer_map(
    batch: Mapping[str, torch.Tensor], num_agents_to_predict: int, device: torch.device,
    n_lanes: int = 6, lane_points: int = 100, n_crosswalks: int = 4, *,
    origin: torch.Tensor | None = None, yaw0: torch.Tensor | None = None,
    agent_xy: torch.Tensor | None = None, agent_valid: torch.Tensor | None = None,
    return_valid: bool = False,
):
    """Build source-structured GameFormer map features from WOMD roadgraph ids.

    The public GameFormer preprocessor selects nearby lane polylines separately
    for each predicted actor.  WOMD tf.Example already exposes per-point feature
    ids/types/directions, so retain those polylines instead of treating the flat
    point array as a single lane.  Coordinates remain in the current SDC frame;
    ``agent_xy`` is used only to choose each actor's nearby features.
    """
    B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
    lanes = torch.zeros(B, num_agents_to_predict, n_lanes, lane_points, 16, device=device)
    cross = torch.zeros(B, num_agents_to_predict, n_crosswalks, lane_points, 3, device=device)
    lanes_valid = torch.zeros(B, num_agents_to_predict, n_lanes, lane_points, device=device, dtype=torch.bool)
    cross_valid = torch.zeros(B, num_agents_to_predict, n_crosswalks, lane_points, device=device, dtype=torch.bool)

    xy, rg_valid, ids, types, source_heading = _roadgraph_feature_bundle(
        batch, device, origin=origin, yaw0=yaw0
    )
    # Exact feature ids are required to reconstruct topology.  Legacy/synthetic
    # caches without ids retain the old shape-preserving fallback for compatibility.
    if xy is None or rg_valid is None:
        if return_valid:
            return lanes, cross, lanes_valid, cross_valid
        return lanes, cross
    if ids is None:
        _legacy_gameformer_flat_map(
            batch, device, B=B, num_agents_to_predict=num_agents_to_predict,
            n_lanes=n_lanes, lane_points=lane_points, origin=origin, yaw0=yaw0,
            lanes=lanes, lanes_valid=lanes_valid,
        )
        if return_valid:
            return lanes, cross, lanes_valid, cross_valid
        return lanes, cross

    if agent_xy is None:
        anchors = torch.zeros(B, num_agents_to_predict, 2, device=device, dtype=xy.dtype)
    else:
        anchors = torch.nan_to_num(agent_xy.to(device=device, dtype=xy.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        if anchors.shape[1] < num_agents_to_predict:
            anchors = torch.cat([anchors, torch.zeros(B, num_agents_to_predict - anchors.shape[1], 2, device=device, dtype=xy.dtype)], dim=1)
        anchors = anchors[:, :num_agents_to_predict]
    if agent_valid is None:
        anchor_valid = torch.ones(B, num_agents_to_predict, device=device, dtype=torch.bool)
    else:
        anchor_valid = agent_valid.to(device).bool()
        if anchor_valid.ndim > 2:
            anchor_valid = anchor_valid[..., -1]
        if anchor_valid.shape[1] < num_agents_to_predict:
            anchor_valid = torch.cat([anchor_valid, torch.zeros(B, num_agents_to_predict - anchor_valid.shape[1], device=device, dtype=torch.bool)], dim=1)
        anchor_valid = anchor_valid[:, :num_agents_to_predict]

    invalid_id = torch.iinfo(ids.dtype).min
    for b in range(B):
        lane_ids, _, _ = _nearest_feature_ids(
            xy[b], rg_valid[b], ids[b], None if types is None else types[b], anchors[b],
            allowed_types=(1, 2, 3) if types is not None else None,
            max_features=n_lanes, radius_m=200.0,
        )
        cross_ids, _, _ = _nearest_feature_ids(
            xy[b], rg_valid[b], ids[b], None if types is None else types[b], anchors[b],
            allowed_types=(18,) if types is not None else None,
            max_features=n_crosswalks, radius_m=200.0,
        )
        for a in range(num_agents_to_predict):
            if not bool(anchor_valid[b, a]):
                continue
            for slot in range(lane_ids.shape[1]):
                fid = lane_ids[a, slot]
                if bool(fid == invalid_id):
                    continue
                idx = torch.nonzero(rg_valid[b] & (ids[b] == fid), as_tuple=False).flatten()
                if types is not None:
                    idx = idx[(types[b, idx] >= 1) & (types[b, idx] <= 3)]
                if idx.numel() == 0:
                    continue
                idx = _sample_polyline_indices(idx, lane_points)
                pts = xy[b, idx]
                # Keep a bounded local context.  One nearby point should not drag
                # an extremely long feature hundreds of metres through the MLP.
                keep = torch.linalg.norm(pts - anchors[b, a], dim=-1) <= 250.0
                idx = idx[keep]
                pts = pts[keep]
                if idx.numel() == 0:
                    continue
                n = int(idx.numel())
                hsrc = None if source_heading is None else source_heading[b, idx]
                h = _polyline_heading(pts, hsrc)
                lanes[b, a, slot, :n, 0:2] = pts
                lanes[b, a, slot, :n, 2] = h
                if types is not None:
                    lanes[b, a, slot, :n, 10] = types[b, idx].clamp(0, 3).to(lanes.dtype)
                lanes_valid[b, a, slot, :n] = True
            for slot in range(cross_ids.shape[1]):
                fid = cross_ids[a, slot]
                if bool(fid == invalid_id):
                    continue
                idx = torch.nonzero(rg_valid[b] & (ids[b] == fid), as_tuple=False).flatten()
                if types is not None:
                    idx = idx[types[b, idx] == 18]
                if idx.numel() == 0:
                    continue
                idx = _sample_polyline_indices(idx, lane_points)
                pts = xy[b, idx]
                keep = torch.linalg.norm(pts - anchors[b, a], dim=-1) <= 250.0
                idx = idx[keep]
                pts = pts[keep]
                if idx.numel() == 0:
                    continue
                n = int(idx.numel())
                hsrc = None if source_heading is None else source_heading[b, idx]
                cross[b, a, slot, :n, 0:2] = pts
                cross[b, a, slot, :n, 2] = _polyline_heading(pts, hsrc)
                cross_valid[b, a, slot, :n] = True
    if return_valid:
        return lanes, cross, lanes_valid, cross_valid
    return lanes, cross


def _legacy_dtpp_flat_map(
    batch: Mapping[str, torch.Tensor], device: torch.device, *, B: int, n_lanes: int,
    lane_points: int, origin: torch.Tensor | None, yaw0: torch.Tensor | None,
    lanes: torch.Tensor, lanes_valid: torch.Tensor,
) -> None:
    rg = _roadgraph_xy(batch, device, origin=origin, yaw0=yaw0, max_points=n_lanes * lane_points)
    rg_valid = _roadgraph_valid_mask(batch, device, n_lanes * lane_points)
    if rg is None or rg.shape[1] == 0:
        return
    total = min(n_lanes * lane_points, rg.shape[1])
    xy = rg[:, :total]
    valid = rg_valid[:, :total] if rg_valid is not None else torch.isfinite(xy).all(dim=-1)
    for b in range(B):
        for lane in range(n_lanes):
            lo = lane * lane_points
            hi = min(lo + lane_points, total)
            if lo >= hi:
                break
            pts = xy[b, lo:hi]
            n = int(pts.shape[0])
            lanes[b, lane, :n, :2] = pts
            lanes[b, lane, :n, 2] = _polyline_heading(pts)
            lanes_valid[b, lane, :n] = valid[b, lo:hi]


def build_dtpp_map(
    batch: Mapping[str, torch.Tensor], device: torch.device, n_lanes: int = 50,
    lane_points: int = 50, n_crosswalks: int = 20, cross_points: int = 30, *,
    origin: torch.Tensor | None = None, yaw0: torch.Tensor | None = None,
    return_valid: bool = False,
):
    """Build DTPP vector-set inputs from intact nearby WOMD map features.

    Public DTPP extracts nearby LANE/ROUTE_LANES/CROSSWALK elements within an
    80 m ego radius and then resamples each element.  WOMD has no nuPlan route
    lane layer in roadgraph_samples, so this adaptation uses the nearest 50 lane
    center feature ids and keeps crosswalk polygons separate; it never chunks the
    flat roadgraph stream into fake 50-point lanes.
    """
    B = next(v for v in batch.values() if torch.is_tensor(v)).shape[0]
    lanes = torch.zeros(B, n_lanes, lane_points, 7, device=device)
    cross = torch.zeros(B, n_crosswalks, cross_points, 3, device=device)
    lanes_valid = torch.zeros(B, n_lanes, lane_points, device=device, dtype=torch.bool)
    cross_valid = torch.zeros(B, n_crosswalks, cross_points, device=device, dtype=torch.bool)
    xy, rg_valid, ids, types, source_heading = _roadgraph_feature_bundle(
        batch, device, origin=origin, yaw0=yaw0
    )
    if xy is None or rg_valid is None:
        if return_valid:
            return lanes, cross, lanes_valid, cross_valid
        return lanes, cross
    if ids is None:
        _legacy_dtpp_flat_map(
            batch, device, B=B, n_lanes=n_lanes, lane_points=lane_points,
            origin=origin, yaw0=yaw0, lanes=lanes, lanes_valid=lanes_valid,
        )
        if return_valid:
            return lanes, cross, lanes_valid, cross_valid
        return lanes, cross

    invalid_id = torch.iinfo(ids.dtype).min
    anchor = torch.zeros(1, 2, device=device, dtype=xy.dtype)
    for b in range(B):
        lane_ids, _, _ = _nearest_feature_ids(
            xy[b], rg_valid[b], ids[b], None if types is None else types[b], anchor,
            allowed_types=(1, 2, 3) if types is not None else None,
            max_features=n_lanes, radius_m=80.0,
        )
        cross_ids, _, _ = _nearest_feature_ids(
            xy[b], rg_valid[b], ids[b], None if types is None else types[b], anchor,
            allowed_types=(18,) if types is not None else None,
            max_features=n_crosswalks, radius_m=80.0,
        )
        for slot in range(lane_ids.shape[1]):
            fid = lane_ids[0, slot]
            if bool(fid == invalid_id):
                continue
            idx = torch.nonzero(rg_valid[b] & (ids[b] == fid), as_tuple=False).flatten()
            if types is not None:
                idx = idx[(types[b, idx] >= 1) & (types[b, idx] <= 3)]
            if idx.numel() == 0:
                continue
            idx = _sample_polyline_indices(idx, lane_points)
            pts = xy[b, idx]
            keep = torch.linalg.norm(pts, dim=-1) <= 100.0
            idx = idx[keep]
            pts = pts[keep]
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            hsrc = None if source_heading is None else source_heading[b, idx]
            lanes[b, slot, :n, :2] = pts
            lanes[b, slot, :n, 2] = _polyline_heading(pts, hsrc)
            # DTPP lane channels 3:7 are the four traffic-light one-hot values.
            # WOMD traffic-light ids can be added when a lane-id association is
            # retained in the cache; zero here represents unknown/no light.
            lanes_valid[b, slot, :n] = True
        for slot in range(cross_ids.shape[1]):
            fid = cross_ids[0, slot]
            if bool(fid == invalid_id):
                continue
            idx = torch.nonzero(rg_valid[b] & (ids[b] == fid), as_tuple=False).flatten()
            if types is not None:
                idx = idx[types[b, idx] == 18]
            if idx.numel() == 0:
                continue
            idx = _sample_polyline_indices(idx, cross_points)
            pts = xy[b, idx]
            keep = torch.linalg.norm(pts, dim=-1) <= 100.0
            idx = idx[keep]
            pts = pts[keep]
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            hsrc = None if source_heading is None else source_heading[b, idx]
            cross[b, slot, :n, :2] = pts
            cross[b, slot, :n, 2] = _polyline_heading(pts, hsrc)
            cross_valid[b, slot, :n] = True
    if return_valid:
        return lanes, cross, lanes_valid, cross_valid
    return lanes, cross

def candidates_to_dtpp_tree(candidates: torch.Tensor) -> torch.Tensor:
    """COWP candidate [B,K,T,7] -> DTPP ego tree [B,K,T,6]."""
    B, K, T, _ = candidates.shape
    out = torch.zeros(B, K, T, 6, device=candidates.device, dtype=candidates.dtype)
    out[..., 0] = candidates[..., 0]
    out[..., 1] = candidates[..., 1]
    out[..., 2] = candidates[..., 2]
    vx = candidates[..., 3]
    vy = candidates[..., 4]
    speed = torch.linalg.norm(torch.stack([vx, vy], dim=-1), dim=-1)
    out[..., 3] = speed
    dt = 0.1
    accel = torch.zeros_like(speed)
    accel[..., 1:] = (speed[..., 1:] - speed[..., :-1]) / dt
    out[..., 4] = accel
    yaw = candidates[..., 2]
    dyaw = torch.zeros_like(yaw)
    dyaw[..., 1:] = torch.atan2(torch.sin(yaw[..., 1:] - yaw[..., :-1]), torch.cos(yaw[..., 1:] - yaw[..., :-1]))
    ds = torch.clamp(speed * dt, min=1e-3)
    out[..., 5] = dyaw / ds
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)



def _wrap_angle_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _rotate_global_to_local(xy: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    """Rotate vectors [...,2] by -yaw0 for each batch item."""
    c = torch.cos(yaw0)
    s = torch.sin(yaw0)
    while c.ndim < xy.ndim - 1:
        c = c.unsqueeze(-1)
        s = s.unsqueeze(-1)
    x = xy[..., 0]
    y = xy[..., 1]
    return torch.stack((c * x + s * y, -s * x + c * y), dim=-1)


def _history_to_ego_frame(hist: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    out = hist.clone()
    delta = out[..., :2] - origin[:, None, None, :]
    out[..., :2] = _rotate_global_to_local(delta, yaw0)
    out[..., 6] = _wrap_angle_tensor(out[..., 6] - yaw0[:, None, None])
    out[..., 7:9] = _rotate_global_to_local(out[..., 7:9], yaw0)
    return out


def _xy_to_ego_frame(xy: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    delta = xy - origin.reshape(origin.shape[0], *([1] * (xy.ndim - 2)), 2)
    return _rotate_global_to_local(delta, yaw0)


def _candidate_to_ego_frame(traj: torch.Tensor, origin: torch.Tensor, yaw0: torch.Tensor) -> torch.Tensor:
    out = traj.clone()
    out[..., :2] = _xy_to_ego_frame(out[..., :2], origin, yaw0)
    if out.shape[-1] > 2:
        out[..., 2] = _wrap_angle_tensor(out[..., 2] - yaw0[:, None, None])
    if out.shape[-1] > 4:
        out[..., 3:5] = _rotate_global_to_local(out[..., 3:5], yaw0)
    return out

@dataclass
class ExternalBatch:
    gameformer_inputs: dict[str, torch.Tensor]
    dtpp_inputs: dict[str, torch.Tensor]
    ego_future_xy: torch.Tensor
    ego_future_valid: torch.Tensor
    neighbors_future_xy: torch.Tensor
    neighbors_future_valid: torch.Tensor
    candidates: torch.Tensor
    candidate_valid: torch.Tensor
    conventional_safe: torch.Tensor
    dtpp_candidate_tree: torch.Tensor
    planner_inputs: dict[str, torch.Tensor]
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
        source_agent_mask = mask_t.bool().to(device) if mask_t is not None and mask_t.shape[:2] == agent_history.shape[:2] else None
    else:
        agent_history, source_agent_mask = build_agent_history_from_womd(batch, max_agents=int(cfg.get("limits", {}).get("max_agents", cfg.get("model", {}).get("max_agents", 128))), history_steps=int(cfg.get("model", {}).get("history_steps", 11)), d_state=11)
        agent_history = agent_history.to(device)
        source_agent_mask = source_agent_mask.to(device)
    # Enforce the core WOMD state invariant before *any* coordinate transform:
    # a timestep is usable only when declared valid AND all state fields used by
    # the adapters are finite.  Invalid values are zero-filled only after their
    # validity bit has been cleared so they cannot masquerade as real states.
    state_finite = torch.isfinite(agent_history[..., :10]).all(dim=-1)
    step_valid = (agent_history[..., 10] > 0.5) & state_finite
    agent_history = torch.nan_to_num(agent_history, nan=0.0, posinf=0.0, neginf=0.0)
    agent_history[..., 10] = step_valid.to(agent_history.dtype)
    current_valid = step_valid[..., -1]
    agent_mask = current_valid if source_agent_mask is None else (source_agent_mask.bool() & current_valid)
    B, N, Th, _ = agent_history.shape
    horizon = int(horizon or cfg.get("time", {}).get("future_steps", 80))
    sdc = sdc_indices(batch, N, B, device)
    neighbors_idx = _select_neighbors(agent_history, agent_mask, sdc, max_neighbors)
    ego_idx = sdc[:, None]
    all_pred_idx = torch.cat([ego_idx, neighbors_idx], dim=1)

    # External planners are trained/evaluated in the current SDC frame.  The
    # previous adapter fed raw WOMD world coordinates into sinusoidal/MLP encoders
    # and GMM regression, causing huge offsets and FP16 overflow.
    rows = torch.arange(B, device=device)
    sdc_safe = sdc.clamp(0, N - 1)
    origin = agent_history[rows, sdc_safe, -1, :2].clone()
    yaw0 = agent_history[rows, sdc_safe, -1, 6].clone()
    sdc_current_valid = current_valid[rows, sdc_safe]

    baseline_l = str(baseline).lower() if baseline is not None else None
    need_gf = baseline_l in {None, "gameformer"}
    need_dtpp = baseline_l in {None, "dtpp"}
    need_planner = baseline_l in {None, "pluto", "plant2"}

    typ = _agent_type(batch, N, device)
    # Build only the representation family consumed by this baseline.  The old
    # adapter eagerly constructed both GameFormer-style and DTPP-style histories
    # for every method, even though only one was used.
    need_gf_state = bool(need_gf or need_planner)
    if need_gf_state:
        selected_types = gather_rows(typ.unsqueeze(-1), all_pred_idx).squeeze(-1) if typ is not None else None
        selected_hist = _history_to_ego_frame(gather_rows(agent_history, all_pred_idx), origin, yaw0)
        gf_state = history_to_gameformer_state(selected_hist, selected_types)
        gf_ego = gf_state[:, 0]
        gf_neighbors = gf_state[:, 1:]
    else:
        selected_hist = torch.empty(B, 0, Th, agent_history.shape[-1], device=device)
        gf_state = torch.empty(B, 0, Th, 11, device=device)
        gf_ego = torch.empty(B, Th, 11, device=device)
        gf_neighbors = torch.empty(B, 0, Th, 11, device=device)
    if need_gf:
        gf_lanes, gf_cross, gf_lanes_valid, gf_cross_valid = build_gameformer_map(
            batch, max_neighbors + 1, device, origin=origin, yaw0=yaw0,
            agent_xy=selected_hist[..., -1, :2],
            agent_valid=(selected_hist[..., -1, 10] > 0.5),
            return_valid=True,
        )
    else:
        gf_lanes = torch.empty(B, 0, 0, 0, 16, device=device)
        gf_cross = torch.empty(B, 0, 0, 0, 3, device=device)
        gf_lanes_valid = torch.empty(B, 0, 0, 0, device=device, dtype=torch.bool)
        gf_cross_valid = torch.empty(B, 0, 0, 0, device=device, dtype=torch.bool)

    if need_dtpp:
        ego_hist = _history_to_ego_frame(gather_rows(agent_history, ego_idx), origin, yaw0).squeeze(1)
        neigh_hist = _history_to_ego_frame(gather_rows(agent_history, neighbors_idx), origin, yaw0)
        neigh_types = gather_rows(typ.unsqueeze(-1), neighbors_idx).squeeze(-1) if typ is not None else None
        dtpp_ego = history_to_dtpp_ego(ego_hist)
        dtpp_neighbors = history_to_dtpp_neighbors(neigh_hist, neigh_types)
    else:
        dtpp_ego = torch.empty(B, Th, 7, device=device)
        dtpp_neighbors = torch.empty(B, max_neighbors, Th, 11, device=device)
    if need_dtpp or need_planner:
        dtpp_lanes, dtpp_cross, dtpp_lanes_valid, dtpp_cross_valid = build_dtpp_map(
            batch, device, origin=origin, yaw0=yaw0, return_valid=True
        )
    else:
        dtpp_lanes = torch.empty(B, 0, 0, 7, device=device)
        dtpp_cross = torch.empty(B, 0, 0, 3, device=device)
        dtpp_lanes_valid = torch.empty(B, 0, 0, device=device, dtype=torch.bool)
        dtpp_cross_valid = torch.empty(B, 0, 0, device=device, dtype=torch.bool)
    route = build_route_centerline(batch, device, origin=origin, yaw0=yaw0) if need_planner else torch.empty(B, 0, 4, device=device)

    if require_future:
        future_xy, future_valid = _future_xy_for_selected(batch, all_pred_idx, N, horizon, device)
        future_xy = _xy_to_ego_frame(future_xy, origin, yaw0)
        future_finite = torch.isfinite(future_xy).all(dim=-1)
        future_valid = future_valid & future_finite & sdc_current_valid[:, None, None]
        future_xy = torch.nan_to_num(future_xy, nan=0.0, posinf=0.0, neginf=0.0)
        ego_future_xy = future_xy[:, 0]
        ego_future_valid = future_valid[:, 0]
        neighbors_future_xy = future_xy[:, 1:]
        neighbors_future_valid = future_valid[:, 1:]
    else:
        ego_future_xy = torch.empty(B, 0, 2, device=device)
        ego_future_valid = torch.empty(B, 0, dtype=torch.bool, device=device)
        neighbors_future_xy = torch.empty(B, max_neighbors, 0, 2, device=device)
        neighbors_future_valid = torch.empty(B, max_neighbors, 0, dtype=torch.bool, device=device)

    if require_candidates:
        if "cowp/candidates/trajectory" not in batch or "cowp/candidates/valid" not in batch:
            raise KeyError("require_candidates=True but COWP candidate tensors are absent")
        cand = _candidate_to_ego_frame(batch["cowp/candidates/trajectory"].float().to(device, non_blocking=True), origin, yaw0)
        cand_valid = batch["cowp/candidates/valid"].bool().to(device, non_blocking=True)
        # One malformed proposal must never inject NaN/Inf into a whole-batch
        # transformer/loss.  Keep the proposal bank shape stable and invalidate
        # only the offending branch.
        cand_valid = cand_valid & candidate_geometry_finite(cand) & sdc_current_valid[:, None]
        cand = torch.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        cand = torch.empty(B, 0, horizon, 7, device=device)
        cand_valid = torch.empty(B, 0, dtype=torch.bool, device=device)
    if require_candidates:
        if cand.shape[1] > max_candidates:
            cand = cand[:, :max_candidates]
            cand_valid = cand_valid[:, :max_candidates]
        elif cand.shape[1] < max_candidates:
            pad_k = max_candidates - cand.shape[1]
            cand = torch.cat([cand, torch.zeros(B, pad_k, cand.shape[2], cand.shape[3], device=device, dtype=cand.dtype)], dim=1)
            cand_valid = torch.cat([cand_valid, torch.zeros(B, pad_k, device=device, dtype=torch.bool)], dim=1)
        if cand.shape[2] > horizon:
            cand = cand[:, :, :horizon]
        elif cand.shape[2] < horizon:
            pad_t = horizon - cand.shape[2]
            cand = torch.cat([cand, torch.zeros(B, cand.shape[1], pad_t, cand.shape[3], device=device, dtype=cand.dtype)], dim=2)
        conventional = batch.get("cowp/candidates/conventional_safe")
        if conventional is not None:
            conventional = conventional.bool().to(device, non_blocking=True)
            conventional = conventional[:, : cand_valid.shape[1]] if conventional.shape[1] >= cand_valid.shape[1] else torch.cat([conventional, torch.zeros(B, cand_valid.shape[1] - conventional.shape[1], device=device, dtype=torch.bool)], dim=1)
            conventional = conventional & cand_valid
        else:
            conventional = cand_valid
        dtpp_tree = candidates_to_dtpp_tree(cand)
    else:
        conventional = cand_valid
        dtpp_tree = torch.empty(B, 0, horizon, 6, device=device)

    return ExternalBatch(
        gameformer_inputs=({
            "ego_state": gf_ego, "neighbors_state": gf_neighbors,
            "actors_valid": (selected_hist[..., 10] > 0.5),
            "map_lanes": gf_lanes, "map_crosswalks": gf_cross,
            "map_lanes_valid": gf_lanes_valid, "map_crosswalks_valid": gf_cross_valid,
        } if need_gf else {}),
        dtpp_inputs=({
            "ego_agent_past": dtpp_ego, "neighbor_agents_past": dtpp_neighbors,
            "map_lanes": dtpp_lanes, "map_crosswalks": dtpp_cross,
            "map_lanes_valid": dtpp_lanes_valid, "map_crosswalks_valid": dtpp_cross_valid,
        } if need_dtpp else {}),
        ego_future_xy=ego_future_xy,
        ego_future_valid=ego_future_valid,
        neighbors_future_xy=neighbors_future_xy,
        neighbors_future_valid=neighbors_future_valid,
        candidates=cand,
        candidate_valid=cand_valid,
        conventional_safe=conventional,
        dtpp_candidate_tree=dtpp_tree,
        planner_inputs=({
            "agents": gf_state,
            "agent_valid": (selected_hist[..., 10] > 0.5),
            "map_lanes": dtpp_lanes,
            "map_lanes_valid": dtpp_lanes_valid,
            "route": route,
            "neighbors_future_xy": neighbors_future_xy,
            "neighbors_future_valid": neighbors_future_valid,
        } if need_planner else {}),
        neighbor_indices=neighbors_idx,
        sdc_indices=sdc,
        origin=origin,
        yaw0=yaw0,
        sdc_current_valid=sdc_current_valid,
    )


def best_candidate_to_logged_ego(candidates: torch.Tensor, candidate_valid: torch.Tensor, ego_future_xy: torch.Tensor, ego_future_valid: torch.Tensor) -> torch.Tensor:
    valid_b = ego_future_valid[:, None, :, None].bool()
    safe_target = torch.where(valid_b, ego_future_xy[:, None], candidates[..., :2].detach())
    diff = candidates[..., :2] - safe_target
    valid_t = valid_b.float()
    denom = valid_t.sum(dim=-1).sum(dim=-1).clamp_min(1.0)
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
