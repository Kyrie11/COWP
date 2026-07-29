from __future__ import annotations

import csv
import os
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz


RECOVERY_MODE_VOCAB = [
    "stop",
    "brake_lane",
    "lateral_escape",
    "yield_rejoin",
    "pull_over",
    "mitigate_contact",
    "post_contact_stabilize",
    "avoid_secondary",
]
RECOVERY_MODE_TO_ID = {name: i for i, name in enumerate(RECOVERY_MODE_VOCAB)}
BUCKET_TO_ID = {"safe": 0, "near_contact": 1, "contact": 2, "other": 3}
OPTION_PARAM_DIM = 3
OPTION_FEATURE_DIM = len(RECOVERY_MODE_VOCAB) + OPTION_PARAM_DIM + 2


def _path_key(path: str | Path) -> str:
    """Stable absolute key for matching sample paths to manifest rows."""
    return os.path.abspath(os.fspath(path))


def _dataset_root_for_sample(path: str | Path) -> Path:
    """Return the OC-RAP dataset root for either root/samples/x.npz or root/x.npz."""
    p = Path(path)
    return p.parent.parent if p.parent.name == "samples" else p.parent


@lru_cache(maxsize=128)
def _manifest_metadata_map(dataset_root_key: str) -> dict[str, dict[str, str]]:
    """Load lightweight per-sample metadata from dataset_root/manifest.csv.

    Training only needs metadata such as split_id and i_art_star for pre-flight
    filtering/sampling.  Reading those fields from the manifest avoids opening
    every compressed .npz before the first epoch.
    """
    root = Path(dataset_root_key)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    try:
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_path = str(row.get("path", "")).strip()
                if not raw_path:
                    continue
                sample_path = Path(raw_path)
                if not sample_path.is_absolute():
                    sample_path = root / sample_path
                out[_path_key(sample_path)] = {str(k): "" if v is None else str(v) for k, v in row.items()}
    except Exception:
        return {}
    return out


def manifest_metadata_for_path(path: str | Path) -> dict[str, str] | None:
    """Return manifest metadata for a sample path when available."""
    root = _dataset_root_for_sample(path)
    table = _manifest_metadata_map(_path_key(root))
    return table.get(_path_key(path)) if table else None


def npz_scalar(path: str | Path, key: str, default: Any = None) -> Any:
    """Read a single scalar-like value from an .npz without materializing arrays.

    This is intentionally different from load_npz(), which expands every array in
    the archive.  It keeps split filtering and sampler construction cheap even
    for large compressed samples containing BEV/map/history tensors.
    """
    try:
        with np.load(path, allow_pickle=True) as z:
            if key not in z.files:
                return default
            arr = np.asarray(z[key])
            if arr.shape == ():
                return arr.item()
            if arr.size == 1:
                return arr.reshape(-1)[0].item()
            return arr.tolist()
    except Exception:
        return default


def scalar_metadata_for_path(path: str | Path, key: str, default: Any = None) -> Any:
    """Read metadata from manifest.csv first, falling back to one-key NPZ read."""
    row = manifest_metadata_for_path(path)
    if row is not None and key in row and row.get(key, "") != "":
        return row.get(key, default)
    return npz_scalar(path, key, default)


def iter_sample_paths_many(dataset: str | Path, max_samples: int | None = None) -> list[Path]:
    """Return sample paths from one or more OC-RAP dataset directories.

    The CLI still exposes a single --dataset argument for backwards
    compatibility.  To train on multiple sharded builds, pass a comma-separated
    list, e.g. /path/w0,/path/w1,/path/w2,/path/w3.
    """
    if isinstance(dataset, Path):
        specs = [dataset]
    else:
        raw = str(dataset)
        sep_specs: list[str] = []
        for chunk in raw.split(','):
            chunk = chunk.strip()
            if chunk:
                sep_specs.append(chunk)
        # Also accept the platform path separator when it is unambiguous.
        if len(sep_specs) == 1 and os.pathsep in sep_specs[0] and not Path(sep_specs[0]).exists():
            sep_specs = [x.strip() for x in sep_specs[0].split(os.pathsep) if x.strip()]
        specs = [Path(x) for x in sep_specs]
    paths: list[Path] = []
    for spec in specs:
        paths.extend(iter_sample_paths(spec))
    paths = sorted(dict.fromkeys(paths))
    return paths[:max_samples] if max_samples else paths


def _arr(d: dict[str, Any], key: str, default_shape: tuple[int, ...] = (0,)) -> np.ndarray:
    if key not in d:
        return np.zeros(default_shape, dtype=np.float32)
    try:
        return np.asarray(d[key], dtype=np.float32)
    except Exception:
        return np.zeros(default_shape, dtype=np.float32)


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return default


def _int_scalar(d: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(round(float(np.asarray(d.get(key, default)).item())))
    except Exception:
        return default


def _pad_flat(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float32)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    out[: min(n, v.size)] = v[:n]
    return out


def _finite_stats(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([v.mean(), v.std(), v.min(), np.quantile(v, 0.25), np.quantile(v, 0.75), v.max()], dtype=np.float32)


def _one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=np.float32)
    if 0 <= int(index) < int(width):
        out[int(index)] = 1.0
    return out


def _string_array(d: dict[str, Any], key: str) -> list[str]:
    if key not in d:
        return []
    arr = np.asarray(d[key])
    out: list[str] = []
    for v in arr.reshape(-1).tolist():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def option_features_from_sample(d: dict[str, Any], cfg: dict | None = None) -> np.ndarray:
    """Encode recovery option identity and parameters for the margin decoder.

    The paper's margin head is conditioned on the recovery option ``g_l``.  Older
    code used only learned option-index embeddings, which makes two runs with the
    same option count but different option modes/parameters indistinguishable to
    the model.  This compact feature keeps the model architecture independent of
    Python dataclasses while preserving the semantic option conditioning saved in
    each ``.npz`` sample.
    """
    del cfg  # reserved for future feature scaling knobs
    params = _arr(d, "recovery_params")
    if params.ndim == 1:
        params = params.reshape(-1, OPTION_PARAM_DIM) if params.size else np.zeros((0, OPTION_PARAM_DIM), dtype=np.float32)
    if params.ndim != 2:
        params = np.zeros((0, OPTION_PARAM_DIM), dtype=np.float32)
    L = int(params.shape[0]) if params.size else int(np.asarray(d.get("m_star", np.zeros((1, 0)))).shape[-1])
    out = np.zeros((L, OPTION_FEATURE_DIM), dtype=np.float32)
    modes = _string_array(d, "recovery_modes")
    valid = np.asarray(d.get("option_valid", np.ones((L,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    for l in range(L):
        mode = modes[l] if l < len(modes) else ""
        mid = RECOVERY_MODE_TO_ID.get(mode, -1)
        if mid >= 0:
            out[l, mid] = 1.0
        if l < params.shape[0]:
            p = np.asarray(params[l], dtype=np.float32).reshape(-1)
            out[l, len(RECOVERY_MODE_VOCAB) : len(RECOVERY_MODE_VOCAB) + OPTION_PARAM_DIM] = np.pad(p[:OPTION_PARAM_DIM], (0, max(0, OPTION_PARAM_DIM - min(OPTION_PARAM_DIM, p.size))))[:OPTION_PARAM_DIM]
        out[l, -2] = float(valid[l] > 0.5) if l < valid.size else 1.0
        out[l, -1] = float(l) / max(float(max(L - 1, 1)), 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _agent_features(d: dict[str, Any], max_agents: int) -> np.ndarray:
    hist = _arr(d, "agent_history")
    valid = _arr(d, "agent_valid")
    ego = _arr(d, "ego_state", (9,)).reshape(-1)
    if hist.ndim != 3 or valid.ndim < 2 or hist.shape[1] == 0:
        return np.zeros(8 + max_agents * 10, dtype=np.float32)
    last = hist[-1]
    vmask = valid[-1].astype(bool) if valid.shape[0] else np.zeros((last.shape[0],), dtype=bool)
    ego_xy = ego[:2] if ego.size >= 2 else np.zeros(2, dtype=np.float32)
    rows = []
    dists = []
    speeds = []
    # Agent feature layout follows ocrap.data.schema.AGENT_FEATURES.
    for a in range(1, min(last.shape[0], len(vmask))):
        if not vmask[a]:
            continue
        s = last[a]
        if s.size < 16:
            continue
        dx = float(s[0] - ego_xy[0])
        dy = float(s[1] - ego_xy[1])
        dist = float(np.hypot(dx, dy))
        speed = float(np.hypot(s[3], s[4]))
        dists.append(dist)
        speeds.append(speed)
        rows.append((dist, np.array([
            dx / 80.0,
            dy / 80.0,
            float(s[3]) / 20.0,
            float(s[4]) / 20.0,
            speed / 20.0,
            float(s[8]) if s.size > 8 else 0.0,
            float(s[9]) if s.size > 9 else 1.0,
            float(s[10]) / 10.0 if s.size > 10 else 0.0,
            float(s[11]) / 5.0 if s.size > 11 else 0.0,
            float(s[13]) / 10.0 if s.size > 13 else 0.0,
        ], dtype=np.float32)))
    rows.sort(key=lambda x: x[0])
    packed = np.zeros((max_agents, 10), dtype=np.float32)
    for i, (_, feat) in enumerate(rows[:max_agents]):
        packed[i] = feat
    d_arr = np.asarray(dists, dtype=np.float32)
    s_arr = np.asarray(speeds, dtype=np.float32)
    summary = np.array([
        float(len(rows)) / max(float(max_agents), 1.0),
        float(d_arr.min() / 80.0) if d_arr.size else 0.0,
        float(d_arr.mean() / 80.0) if d_arr.size else 0.0,
        float(d_arr.std() / 80.0) if d_arr.size else 0.0,
        float((d_arr < 8.0).mean()) if d_arr.size else 0.0,
        float((d_arr < 20.0).mean()) if d_arr.size else 0.0,
        float(s_arr.mean() / 20.0) if s_arr.size else 0.0,
        float(s_arr.max() / 20.0) if s_arr.size else 0.0,
    ], dtype=np.float32)
    return np.concatenate([summary, packed.reshape(-1)], axis=0)


def _feature_layout_values(cfg: dict | None = None) -> dict[str, int]:
    cfg = cfg or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    return {
        "max_agents": int(model_cfg.get("feature_max_agents", 32)),
        "num_macros": int(model_cfg.get("num_macros", 16)),
        "prefix_flat": int(model_cfg.get("feature_prefix_flat_dim", 80)),
        "control_flat": int(model_cfg.get("feature_control_flat_dim", 40)),
        "route_flat": int(model_cfg.get("feature_route_flat_dim", 64)),
        "map_flat": int(model_cfg.get("feature_map_flat_dim", 64)),
        "dyn_flat": int(model_cfg.get("feature_dynamic_map_flat_dim", 32)),
    }


def _candidate_feature_parts(d: dict[str, Any], cfg: dict | None = None) -> list[np.ndarray]:
    """Candidate-dependent prefix features in the exact historical layout."""
    cfg = cfg or {}
    layout = _feature_layout_values(cfg)
    ego = _pad_flat(_arr(d, "ego_state", (9,)), 9)
    prefix_param = _pad_flat(_arr(d, "prefix_param", (5,)), int(cfg.get("prefix_param_dim", 5)))
    macro_type_id = _int_scalar(d, "prefix_macro_type_id", _int_scalar(d, "prefix_macro_id", 0))
    macro = _one_hot(macro_type_id, layout["num_macros"])
    scalar_feat = np.array([
        _scalar(d, "utility") / 20.0,
        _scalar(d, "hard_violation") / 5.0,
        _scalar(d, "harm_proxy") / 5.0,
        _scalar(d, "feasible"),
        _scalar(d, "is_nominal"),
        _scalar(d, "time_index") / 1000.0,
    ], dtype=np.float32)
    return [
        ego,
        prefix_param,
        macro,
        scalar_feat,
        _pad_flat(_arr(d, "prefix_states"), layout["prefix_flat"]),
        _pad_flat(_arr(d, "prefix_controls"), layout["control_flat"]),
    ]


def _shared_scene_feature_parts(d: dict[str, Any], cfg: dict | None = None) -> list[np.ndarray]:
    """History/map/BEV features shared by all candidates of one replan."""
    cfg = cfg or {}
    layout = _feature_layout_values(cfg)
    bev = _arr(d, "bev_occ")
    route = _arr(d, "route")
    maps = _arr(d, "map_polylines")
    dyn = _arr(d, "dynamic_map")

    bev_stats: list[np.ndarray] = []
    if bev.ndim >= 3:
        ch = bev.reshape(bev.shape[0], -1)
        bev_stats.append(ch.mean(axis=1))
        bev_stats.append(ch.std(axis=1))
    else:
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
    bev_feat = _pad_flat(np.concatenate(bev_stats, axis=0), 2 * int(cfg.get("bev_channels", 7)))

    return [
        _agent_features(d, layout["max_agents"]),
        bev_feat,
        _finite_stats(route),
        _pad_flat(route[..., :2] if route.ndim >= 2 else route, layout["route_flat"]),
        _finite_stats(maps),
        _pad_flat(maps[..., :2] if maps.ndim >= 3 else maps, layout["map_flat"]),
        _finite_stats(dyn),
        _pad_flat(dyn, layout["dyn_flat"]),
    ]


def sample_to_feature(d: dict[str, Any], cfg: dict | None = None) -> np.ndarray:
    """Convert one sample to the flat model feature without changing layout."""
    parts = [*_candidate_feature_parts(d, cfg), *_shared_scene_feature_parts(d, cfg)]
    x = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def samples_to_feature_matrix(
    ds: list[dict[str, Any]],
    cfg: dict | None = None,
    *,
    shared_scene: bool = False,
) -> np.ndarray:
    """Vectorize feature extraction and optionally reuse scene-static work.

    All candidates created at one closed-loop replan share history, map, route,
    BEV and dynamic-map tensors.  The previous implementation recomputed agent
    sorting, BEV statistics, map statistics and flattening once per candidate.
    ``shared_scene=True`` computes those exact arrays once and concatenates them
    with each candidate's prefix-dependent features.  The resulting rows are
    numerically identical to repeated :func:`sample_to_feature` calls.
    """
    if not ds:
        return np.zeros((0, 0), dtype=np.float32)
    if not shared_scene:
        return np.stack([sample_to_feature(d, cfg) for d in ds], axis=0)
    shared = _shared_scene_feature_parts(ds[0], cfg)
    rows = []
    for d in ds:
        x = np.concatenate([*_candidate_feature_parts(d, cfg), *shared], axis=0).astype(np.float32)
        rows.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    return np.stack(rows, axis=0)


def _target_dim_from_cfg(cfg: dict, key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _model_target_dim(cfg: dict, key: str, default: int) -> int:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    try:
        return int(model_cfg.get(key, default))
    except Exception:
        return int(default)


def _fix_1d(x: np.ndarray, n: int, *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((int(n),), fill, dtype=dtype)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=dtype).reshape(-1)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _fix_bool_1d(x: np.ndarray, n: int, *, fill: bool = False) -> np.ndarray:
    out = np.full((int(n),), bool(fill), dtype=bool)
    if n <= 0:
        return out
    v = np.asarray(x).reshape(-1).astype(bool)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return out


def _fix_2d(x: np.ndarray, shape: tuple[int, int], *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    rows, cols = int(shape[0]), int(shape[1])
    out = np.full((rows, cols), fill, dtype=dtype)
    if rows <= 0 or cols <= 0:
        return out
    v = np.asarray(x, dtype=dtype)
    if v.ndim == 1:
        if rows == 1:
            v = v.reshape(1, -1)
        else:
            v = v.reshape(-1, cols) if v.size and v.size % cols == 0 else v.reshape(1, -1)
    if v.ndim != 2:
        return out
    r = min(rows, v.shape[0])
    c = min(cols, v.shape[1])
    if r > 0 and c > 0:
        out[:r, :c] = v[:r, :c]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _fix_square(x: np.ndarray, n: int, *, fill_offdiag: float = 0.0, diag: float = 1.0, dtype=np.float32) -> np.ndarray:
    out = np.full((int(n), int(n)), fill_offdiag, dtype=dtype)
    if n > 0:
        np.fill_diagonal(out, diag)
    v = np.asarray(x, dtype=dtype)
    if v.ndim == 2:
        r = min(int(n), v.shape[0])
        c = min(int(n), v.shape[1])
        if r > 0 and c > 0:
            out[:r, :c] = v[:r, :c]
            np.fill_diagonal(out, diag)
    return np.nan_to_num(out, nan=fill_offdiag, posinf=fill_offdiag, neginf=fill_offdiag)




def bucket_id_for_path(path: str | Path) -> int:
    """Coarse regime id inferred from the current dataset root path.

    This is used only for regime-aware training losses and diagnostics.  The
    selector must still make decisions from model predictions and candidate
    metadata; it should not require teacher labels at test time.
    """
    try:
        name = str(_dataset_root_for_sample(path).name).lower()
    except Exception:
        name = str(path).lower()
    if "near" in name:
        return int(BUCKET_TO_ID["near_contact"])
    if "contact" in name:
        return int(BUCKET_TO_ID["contact"])
    if "safe" in name or "normal" in name or "background" in name:
        return int(BUCKET_TO_ID["safe"])
    return int(BUCKET_TO_ID["other"])


def stable_scene_hash(scene_id: object) -> int:
    """Stable non-negative 31-bit hash for grouping scene-time candidates."""
    b = str(scene_id).encode("utf-8", errors="ignore")
    return int(zlib.adler32(b) & 0x7FFFFFFF)

def _geometry_from_sample(d: dict[str, Any]) -> tuple[int, int, int, int]:
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=np.float32)
    K = int(m.shape[0]) if m.ndim >= 1 else 0
    L = int(m.shape[1]) if m.ndim >= 2 else 0
    rs = np.asarray(d.get("root_signature", np.zeros((K, 0))), dtype=np.float32)
    fs = np.asarray(d.get("root_future_signature", np.zeros((K, 0))), dtype=np.float32)
    return K, L, int(rs.shape[-1]) if rs.ndim >= 2 else 0, int(fs.shape[-1]) if fs.ndim >= 2 else 0


def fix_sample_geometry(d: dict[str, Any], *, num_roots: int, num_options: int, d_signature: int = 0, d_future_signature: int = 0) -> dict[str, np.ndarray]:
    """Return training/evaluation arrays padded to checkpoint geometry.

    The four intended OC-RAP dataset families are intentionally heterogeneous:
    proof-artifact sets often use fewer roots/options than natural/strict/post
    sets.  A single neural model still needs fixed tensor shapes, so padded roots
    and options are marked invalid and excluded by masks, losses, OC-MERO, and
    metrics.
    """
    K = int(num_roots)
    L = int(num_options)
    root_probs = _fix_1d(np.asarray(d.get("root_probs", []), dtype=np.float32), K, fill=0.0)
    root_valid = _fix_bool_1d(np.asarray(d.get("root_valid", root_probs > 0), dtype=bool), K, fill=False)
    if root_valid.any():
        root_probs = np.where(root_valid, np.clip(root_probs, 0.0, None), 0.0).astype(np.float32)
        s = float(root_probs.sum())
        if s > 1e-8:
            root_probs = (root_probs / s).astype(np.float32)
    option_valid = _fix_bool_1d(np.asarray(d.get("option_valid", np.ones((L,), dtype=bool)), dtype=bool), L, fill=False)
    return {
        "root_probs": root_probs,
        "m_star": _fix_2d(np.asarray(d.get("m_star", []), dtype=np.float32), (K, L), fill=-1e9),
        "c_star": _fix_square(np.asarray(d.get("c_star", []), dtype=np.float32), K, fill_offdiag=0.0, diag=1.0),
        "y_obs": _fix_square(np.asarray(d.get("y_obs", []), dtype=np.float32), K, fill_offdiag=0.0, diag=1.0),
        "option_valid": option_valid,
        "root_valid": root_valid,
        "root_signature": _fix_2d(np.asarray(d.get("root_signature", []), dtype=np.float32), (K, int(d_signature)), fill=0.0),
        "root_future_signature": _fix_2d(np.asarray(d.get("root_future_signature", []), dtype=np.float32), (K, int(d_future_signature)), fill=0.0),
        "option_features": _fix_2d(option_features_from_sample(d), (L, OPTION_FEATURE_DIM), fill=0.0),
    }


class OCRAPSampleDataset(Dataset):
    def __init__(self, paths: list[Path], cfg: dict | None = None):
        self.paths = list(paths)
        self.cfg = cfg or {}
        if not self.paths:
            raise ValueError("OCRAPSampleDataset requires at least one sample path")
        first = load_npz(self.paths[0])
        first_K, first_L, first_sig, first_fsig = _geometry_from_sample(first)
        # Default to the paper/build geometry in config, but never shrink below
        # the first sample.  This makes the README's mixed proof/natural/stress
        # training command work without requiring per-run shape bookkeeping.
        self.num_roots = max(_target_dim_from_cfg(self.cfg, "num_roots", first_K or 1), first_K or 1)
        self.num_options = max(_target_dim_from_cfg(self.cfg, "num_recovery_options", first_L or 1), first_L or 1)
        self.d_signature = max(_model_target_dim(self.cfg, "d_signature", first_sig), first_sig)
        self.d_future_signature = max(_model_target_dim(self.cfg, "d_future_signature", first_fsig), first_fsig)
        self.feature_dim = int(sample_to_feature(first, self.cfg).shape[0])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = load_npz(self.paths[idx])
        x = sample_to_feature(d, self.cfg)
        fixed = fix_sample_geometry(
            d,
            num_roots=self.num_roots,
            num_options=self.num_options,
            d_signature=self.d_signature,
            d_future_signature=self.d_future_signature,
        )
        out = {
            "x": torch.from_numpy(x),
            "root_probs": torch.from_numpy(fixed["root_probs"]),
            "m_star": torch.from_numpy(fixed["m_star"]),
            "c_star": torch.from_numpy(fixed["c_star"]),
            "y_obs": torch.from_numpy(fixed["y_obs"]),
            "option_valid": torch.from_numpy(fixed["option_valid"]),
            "root_valid": torch.from_numpy(fixed["root_valid"]),
            "r_dep_star": torch.tensor(float(np.asarray(d["r_dep_star"]).item()), dtype=torch.float32),
            "r_orc_star": torch.tensor(float(np.asarray(d["r_orc_star"]).item()), dtype=torch.float32),
            "i_art_star": torch.tensor(float(np.asarray(d["i_art_star"]).item()), dtype=torch.float32),
            "utility": torch.tensor(float(np.asarray(d.get("utility", 0.0)).item()), dtype=torch.float32),
            "hard_violation": torch.tensor(float(np.asarray(d.get("hard_violation", 0.0)).item()), dtype=torch.float32),
            "harm_proxy": torch.tensor(float(np.asarray(d.get("harm_proxy", 0.0)).item()), dtype=torch.float32),
            "feasible": torch.tensor(float(np.asarray(d.get("feasible", 1.0)).item()), dtype=torch.float32),
            "scene_hash": torch.tensor(stable_scene_hash(d.get("scene_id", "")), dtype=torch.long),
            "time_index": torch.tensor(int(np.asarray(d.get("time_index", 0)).item()), dtype=torch.long),
            "candidate_index": torch.tensor(int(np.asarray(d.get("candidate_index", 0)).item()), dtype=torch.long),
            # Semantic macro id used by macro-conditioned recovery losses.
            # Older datasets only stored prefix_macro_id, so keep the same
            # fallback convention as sample_to_feature().
            "prefix_macro_type_id": torch.tensor(
                int(np.asarray(d.get("prefix_macro_type_id", d.get("prefix_macro_id", 0))).item()),
                dtype=torch.long,
            ),
            "is_nominal": torch.tensor(float(np.asarray(d.get("is_nominal", 0)).item()), dtype=torch.float32),
            "bucket_id": torch.tensor(bucket_id_for_path(self.paths[idx]), dtype=torch.long),
            "root_signature": torch.from_numpy(fixed["root_signature"]),
            "root_future_signature": torch.from_numpy(fixed["root_future_signature"]),
            "option_features": torch.from_numpy(fixed["option_features"]),
        }
        return out


def split_paths_by_npz_split(paths: list[Path], split: str | set[str]) -> list[Path]:
    splits = {split} if isinstance(split, str) else set(split)
    if "all" in splits:
        return list(paths)
    keep: list[Path] = []
    for p in paths:
        sid = str(scalar_metadata_for_path(p, "split_id", ""))
        if sid in splits:
            keep.append(p)
    return keep
