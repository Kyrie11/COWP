from __future__ import annotations

import csv
import os
import zlib
import hashlib
import json
import time
import fcntl
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz_selected
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.simulation.teacher.controllers import rollout_recovery_controller
from ocrap.v48_74_signed_viability import (
    V48_74_FEATURE_DIM as DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_DIM,
    V48_74_SCHEMA as DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    enabled as _v48_74_signed_viability_enabled,
    signed_viability_diagnostics as _v48_74_signed_viability_diagnostics,
)


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

# v48.45.6 engineering-only I/O fast path.  These are exactly the NPZ members
# consumed by OCRAPSampleDataset/sample_to_feature/fix_sample_geometry and the
# training labels returned by __getitem__.  Keeping this list explicit makes the
# optimization auditable: it changes no tensor value, sample order, loss, or model
# input; it only avoids decompressing unrelated archive members.
MODEL_SAMPLE_NPZ_KEYS: frozenset[str] = frozenset({
    "agent_history", "agent_valid", "ego_state", "bev_occ", "route",
    "map_polylines", "dynamic_map", "prefix_param", "prefix_states",
    "prefix_controls", "prefix_macro_type_id", "prefix_macro_id",
    "utility", "hard_violation", "harm_proxy", "feasible",
    "is_nominal", "scene_id", "time_index", "candidate_index",
    "root_probs", "m_star", "c_star", "y_obs", "root_valid",
    "option_valid", "root_signature", "root_future_signature",
    "recovery_params", "recovery_modes", "r_dep_star", "r_orc_star",
    "i_art_star",
})

# v48.56 engineering-only teacher-index fast path.  The teacher index never
# constructs model features; it only consumes OC-MERO matrices/probabilities,
# cached recovery scalars and grouping metadata.  Keeping this subset separate
# avoids decompressing BEV/map/history/prefix/signature tensors for every
# calibration sample.  This changes no teacher value or row ordering.
TEACHER_PCD_NPZ_KEYS: frozenset[str] = frozenset({
    "m_star", "root_probs", "c_star", "root_valid", "option_valid",
    "r_dep_star", "r_orc_star", "hard_violation", "harm_proxy",
    "scene_id", "time_index", "candidate_index", "prefix_macro_type_id",
    "prefix_macro_id", "is_nominal",
})


NOMINAL_DEVIATION_NPZ_KEYS: frozenset[str] = frozenset({
    "scene_id", "time_index", "is_nominal", "prefix_states",
})


# v48.16 ANCHOR: protocol-role aliases. Dedicated calibration intentionally
# renames split_id to make train/dev/certificate roles explicit. All readers
# must agree on these aliases instead of silently discarding every sample.
SPLIT_ROLE_ALIASES: dict[str, frozenset[str]] = {
    "train": frozenset({"train", "evidence_adapt_train"}),
    "val": frozenset({"val", "evidence_adapt_dev"}),
    "calibration": frozenset({"calibration", "certificate_pool"}),
    "certificate_pool": frozenset({"certificate_pool"}),
    "evidence_adapt_train": frozenset({"evidence_adapt_train"}),
    "evidence_adapt_dev": frozenset({"evidence_adapt_dev"}),
}


def expand_split_roles(splits: str | set[str] | tuple[str, ...] | list[str]) -> set[str]:
    """Expand semantic split roles into concrete manifest split_id values."""
    raw = {splits} if isinstance(splits, str) else {str(x) for x in splits}
    out: set[str] = set()
    for item in raw:
        item = str(item).strip()
        if not item:
            continue
        out.update(SPLIT_ROLE_ALIASES.get(item, frozenset({item})))
    return out


def split_id_matches(split_id: object, allowed: str | set[str] | tuple[str, ...] | list[str]) -> bool:
    """Return whether a manifest/NPZ split id belongs to an allowed role."""
    return str(split_id).strip() in expand_split_roles(allowed)


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




# v48.60.1 engineering correction.  CPHR must use the complete executable
# prefix, not the historically truncated 80-D encoder block.  Keep the feature
# schema explicit so stale v48.60.0 checkpoints/caches cannot be reused silently.
DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA = 2
DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_DIM = 6


def direct_absolute_physical_headroom_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None
) -> np.ndarray:
    """Compute the six CPHR coordinates from full observable sample tensors.

    This is deliberately a side-channel *to the CPHR source only*.  The frozen
    Stage-I encoder continues to consume the exact historical flat feature row.
    Inputs are restricted to decision-time observation plus the executable
    candidate prefix: ``ego_state``, ``agent_history[-1]``, ``agent_valid[-1]``,
    ``prefix_states`` and ``prefix_controls``.  No future/teacher tensor is read.

    Feature order is unchanged from v48.60.0:
      0 signed minimum-clearance reserve
      1 signed terminal-clearance reserve
      2 clearance recovery gain
      3 stopping reserve
      4 control-envelope reserve
      5 stability reserve
    """
    cfg = cfg or {}
    states = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=np.float32)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 4))), dtype=np.float32)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=np.float32).reshape(-1)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=np.float32)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)

    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 9:
        raise ValueError(
            f"CPHR requires full prefix_states[T,>=9], got shape={getattr(states, 'shape', None)}"
        )
    if ego.size < 9:
        raise ValueError(f"CPHR requires ego_state[>=9], got shape={ego.shape}")
    if controls.ndim != 2:
        controls = np.zeros((0, 4), dtype=np.float32)
    if controls.shape[1] < 4 and controls.shape[0] > 0:
        raise ValueError(f"CPHR requires prefix_controls[T,>=4], got shape={controls.shape}")

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    max_agents = int(model_cfg.get("feature_max_agents", 32))
    ego_xy = ego[:2].astype(np.float64, copy=False)
    ego_len = max(abs(float(ego[7])), 1.0e-3)
    ego_wid = max(abs(float(ego[8])), 1.0e-3)
    ego_rad = 0.5 * float(np.hypot(ego_len, ego_wid))

    # Match the encoder's observable-neighbour contract: current valid non-ego
    # agents, nearest-first, capped at feature_max_agents.  Unlike v48.60.0 we do
    # not round-trip through normalized packed tokens, so no prefix information
    # is lost before the CPHR calculation.
    agents: list[tuple[float, np.ndarray]] = []
    if hist.ndim == 3 and hist.shape[0] > 0 and hist.shape[1] > 1 and valid.ndim >= 2 and valid.shape[0] > 0:
        last = hist[-1]
        vmask = valid[-1].reshape(-1)
        for aidx in range(1, min(last.shape[0], vmask.size)):
            if not bool(vmask[aidx]):
                continue
            row = np.asarray(last[aidx], dtype=np.float64).reshape(-1)
            if row.size < 12 or not np.isfinite(row[:12]).all():
                continue
            rel = row[:2] - ego_xy
            agents.append((float(np.linalg.norm(rel)), row))
    agents.sort(key=lambda z: z[0])
    agents = agents[:max_agents]

    T = int(states.shape[0])
    sample_rate = float(cfg.get("sample_rate_hz", 10.0) or 10.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"invalid sample_rate_hz for CPHR: {sample_rate}")
    dt = 1.0 / sample_rate
    # prefix_generation._rollout stores the first executable state after one
    # integration interval, so its physical timestamp is dt rather than 0.
    times = (np.arange(T, dtype=np.float64) + 1.0) * dt
    prefix_xy_rel = states[:, :2].astype(np.float64) - ego_xy[None, :]

    if agents:
        rel0 = np.stack([row[:2] - ego_xy for _, row in agents], axis=0)
        vel = np.stack([row[3:5] for _, row in agents], axis=0)
        alen = np.asarray([max(abs(float(row[10])), 1.0e-3) for _, row in agents], dtype=np.float64)
        awid = np.asarray([max(abs(float(row[11])), 1.0e-3) for _, row in agents], dtype=np.float64)
        arad = 0.5 * np.hypot(alen, awid)
        agent_future = rel0[None, :, :] + times[:, None, None] * vel[None, :, :]
        delta = prefix_xy_rel[:, None, :] - agent_future
        signed_clearance = np.linalg.norm(delta, axis=-1) - ego_rad - arad[None, :]
        c_min = float(np.min(signed_clearance))
        c_terminal = float(np.min(signed_clearance[-1]))
        c0 = float(np.min(np.linalg.norm(rel0, axis=-1) - ego_rad - arad))
    else:
        c_min = c_terminal = c0 = 40.0

    speed = np.abs(states[:, 6].astype(np.float64))
    max_speed = float(np.max(speed))
    terminal_speed = float(speed[-1])
    d_safe0 = float(cfg.get("d_safe0_m", 1.0))
    headway = float(cfg.get("safe_time_headway_s", 0.5))
    scales = cfg.get("margin_scales", {}) if isinstance(cfg.get("margin_scales", {}), dict) else {}
    distance_scale = max(float(scales.get("distance", 2.0)), 1.0e-6)
    stop_scale = max(float(scales.get("stop", 5.0)), 1.0e-6)
    d_safe = d_safe0 + headway * max_speed

    h_min_clear = np.tanh((c_min - d_safe) / distance_scale)
    h_terminal_clear = np.tanh((c_terminal - d_safe) / distance_scale)
    h_clear_gain = np.tanh((c_terminal - c0) / distance_scale)

    limits = cfg.get("control_limits", {}) if isinstance(cfg.get("control_limits", {}), dict) else {}
    a_max = float(limits.get("a_max", 3.0))
    a_min = float(limits.get("a_min", -6.0))
    delta_max = float(limits.get("delta_max", 0.55))
    jerk_max = float(limits.get("j_max", 6.0))
    steer_rate_max = float(limits.get("steer_rate_max", 0.5))
    stop_decel = max(abs(a_min), 1.0e-3)
    stop_required = terminal_speed * terminal_speed / (2.0 * stop_decel) + d_safe0
    h_stop = np.tanh((c_terminal - stop_required) / stop_scale)

    if controls.shape[0] > 0:
        c = controls[:, :4].astype(np.float64)
        accel_scale = max(float(scales.get("accel", 1.0)), 1.0e-6)
        decel_scale = max(float(scales.get("decel", 1.0)), 1.0e-6)
        steer_scale = max(float(scales.get("steer", 0.1)), 1.0e-6)
        jerk_scale = max(float(scales.get("jerk", 2.0)), 1.0e-6)
        rate_scale = max(float(scales.get("steer_rate", 0.1)), 1.0e-6)
        ctrl_terms = np.asarray([
            (a_max - float(np.max(np.maximum(c[:, 0], 0.0)))) / accel_scale,
            (abs(a_min) - float(np.max(np.maximum(-c[:, 0], 0.0)))) / decel_scale,
            (delta_max - float(np.max(np.abs(c[:, 1])))) / steer_scale,
            (jerk_max - float(np.max(np.abs(c[:, 2])))) / jerk_scale,
            (steer_rate_max - float(np.max(np.abs(c[:, 3])))) / rate_scale,
        ], dtype=np.float64)
    else:
        # No control action consumes no envelope; this matches the old padded-zero
        # semantics while avoiding a fabricated extra timestep.
        ctrl_terms = np.asarray([
            a_max / max(float(scales.get("accel", 1.0)), 1.0e-6),
            abs(a_min) / max(float(scales.get("decel", 1.0)), 1.0e-6),
            delta_max / max(float(scales.get("steer", 0.1)), 1.0e-6),
            jerk_max / max(float(scales.get("jerk", 2.0)), 1.0e-6),
            steer_rate_max / max(float(scales.get("steer_rate", 0.1)), 1.0e-6),
        ], dtype=np.float64)
    h_control = np.tanh(float(np.min(ctrl_terms)))

    yaw_rate_max = float(cfg.get("yaw_rate_max_rps", 0.6))
    yaw_scale = max(float(scales.get("yaw", 0.2)), 1.0e-6)
    yaw_rate = float(np.max(np.abs(states[:, 5].astype(np.float64))))
    h_stability = np.tanh((yaw_rate_max - yaw_rate) / yaw_scale)

    out = np.asarray([
        h_min_clear,
        h_terminal_clear,
        h_clear_gain,
        h_stop,
        h_control,
        h_stability,
    ], dtype=np.float32)
    if out.shape != (DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_DIM,) or not np.isfinite(out).all():
        raise ValueError(f"invalid CPHR feature vector: {out}")
    return out




# v48.61 ERWF (Executable Recovery Witness Field).  Unlike CPHR's one vector
# per candidate, ERWF builds one signed observable continuation-witness vector
# per *recovery option*.  The side channel therefore has shape [L, 6] and is
# consumed only by the Stage-II absolute source.  Stage-I features remain byte-
# semantically unchanged.
DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA = 1
DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_DIM = 6


def _decode_recovery_mode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    return str(value)


def direct_executable_recovery_witness_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None, *, num_options: int | None = None,
    include_recovery_stability_tail: bool = False,
    include_semantic_alignment_tail: bool = False,
    include_active_constraint_tail: bool = False,
    project_control_envelope: bool = False,
    projection_fidelity_weighting: bool = False,
    robust_occupancy_envelope: bool = False,
    soft_occupancy_disagreement: bool = False,
    boundary_localized_occupancy_trust: bool = False,
    history_occupancy_reachability: bool = False,
    interaction_box_support: bool = False,
    interaction_hull_support: bool = False,
    interaction_anchor_support: bool = False,
    interaction_response_support: bool = False,
) -> np.ndarray:
    """Return the v48.61 option-resolved executable recovery witness field.

    For every valid recovery option ``g_l`` this function rolls the existing
    deterministic recovery controller *after the candidate prefix*, starting at
    the true terminal prefix state.  Surrounding agents are extrapolated only
    from the current observable history using constant velocity.  No
    counterfactual future, root identity, teacher margin/component, regime ID or
    held-out label is read.

    Each option receives six signed, tanh-bounded coordinates, all with the same
    physical zero semantics as CPHR but evaluated over the recovery
    continuation rather than over the candidate prefix itself:

      0 minimum clearance reserve over the recovery continuation
      1 terminal clearance reserve after the recovery continuation
      2 clearance recovery gain relative to candidate-prefix terminal clearance
      3 terminal stopping reserve
      4 recovery-controller control-envelope reserve
      5 recovery-controller stability reserve

    The field is padded to ``num_options`` with zeros; padded/invalid options are
    separately masked by ``option_valid`` and never enter OC-MERO.  This makes
    ERWF candidate x option structured while preserving one regime-agnostic
    shared physical field and the frozen Stage-I representation.
    """
    cfg = cfg or {}
    if include_active_constraint_tail and not (include_recovery_stability_tail and include_semantic_alignment_tail):
        raise ValueError(
            "active-constraint tail requires the full v48.64 semantic witness prefix (12 coordinates)"
        )
    states = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=np.float32)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 4))), dtype=np.float32)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=np.float32).reshape(-1)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=np.float32)
    valid_hist = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    modes = np.asarray(d.get("recovery_modes", []), dtype=object).reshape(-1)
    params = np.asarray(d.get("recovery_params", np.zeros((0, 3))), dtype=np.float32)
    option_valid_raw = np.asarray(d.get("option_valid", np.ones((len(modes),), dtype=bool)), dtype=bool).reshape(-1)

    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 9:
        raise ValueError(
            f"ERWF requires full prefix_states[T,>=9], got shape={getattr(states, 'shape', None)}"
        )
    if ego.size < 9:
        raise ValueError(f"ERWF requires ego_state[>=9], got shape={ego.shape}")
    if controls.ndim != 2:
        controls = np.zeros((0, 4), dtype=np.float32)
    if controls.shape[0] > 0 and controls.shape[1] < 4:
        raise ValueError(f"ERWF requires prefix_controls[T,>=4], got shape={controls.shape}")
    if params.ndim != 2:
        raise ValueError(f"ERWF requires recovery_params[L,P], got shape={params.shape}")
    raw_L = int(max(len(modes), params.shape[0], option_valid_raw.size))
    if raw_L <= 0:
        raise ValueError("ERWF requires recovery_modes/recovery_params/option_valid")
    L = int(num_options if num_options is not None else raw_L)
    if L < raw_L:
        raise ValueError(f"ERWF num_options={L} cannot shrink raw recovery option count={raw_L}")

    sample_rate = float(cfg.get("sample_rate_hz", 10.0) or 10.0)
    recovery_horizon_s = float(cfg.get("recovery_horizon_s", 4.0) or 4.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"invalid sample_rate_hz for ERWF: {sample_rate}")
    if not np.isfinite(recovery_horizon_s) or recovery_horizon_s <= 0.0:
        raise ValueError(f"invalid recovery_horizon_s for ERWF: {recovery_horizon_s}")
    dt = 1.0 / sample_rate
    horizon_steps = max(2, int(round(recovery_horizon_s * sample_rate)))

    prefix = CandidatePrefix(
        macro_id=int(np.asarray(d.get("prefix_macro_id", 0)).reshape(-1)[0]),
        macro_name=str(np.asarray(d.get("prefix_macro_name", "candidate")).reshape(-1)[0]),
        params=np.asarray(d.get("prefix_param", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
        prefix_states=states,
        prefix_controls=controls,
        utility=float(np.asarray(d.get("utility", 0.0)).reshape(-1)[0]),
        feasible=bool(float(np.asarray(d.get("feasible", 1.0)).reshape(-1)[0]) > 0.5),
        hard_violation=float(np.asarray(d.get("hard_violation", 0.0)).reshape(-1)[0]),
        harm_proxy=float(np.asarray(d.get("harm_proxy", 0.0)).reshape(-1)[0]),
    )

    # Observable agents: same nearest-current-agent contract as CPHR.  Use a
    # circle support approximation for efficient vectorized distance evolution;
    # the scientific intervention in v48.61 is continuation/option resolution,
    # not a new collision-geometry estimator.
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    max_agents = int(model_cfg.get("feature_max_agents", 32))
    ego_xy = ego[:2].astype(np.float64, copy=False)
    ego_len = max(abs(float(ego[7])), 1.0e-3)
    ego_wid = max(abs(float(ego[8])), 1.0e-3)
    ego_rad = 0.5 * float(np.hypot(ego_len, ego_wid))
    agents: list[tuple[float, int, np.ndarray]] = []
    if hist.ndim == 3 and hist.shape[0] > 0 and hist.shape[1] > 1 and valid_hist.ndim >= 2 and valid_hist.shape[0] > 0:
        last = hist[-1]
        vmask = valid_hist[-1].reshape(-1)
        for aidx in range(1, min(last.shape[0], vmask.size)):
            if not bool(vmask[aidx]):
                continue
            row = np.asarray(last[aidx], dtype=np.float64).reshape(-1)
            if row.size < 12 or not np.isfinite(row[:12]).all():
                continue
            agents.append((float(np.linalg.norm(row[:2] - ego_xy)), aidx, row))
    agents.sort(key=lambda z: z[0])
    agents = agents[:max_agents]

    if agents:
        rel0 = np.stack([row[:2] - ego_xy for _, _, row in agents], axis=0)
        vel = np.stack([row[3:5] for _, _, row in agents], axis=0)
        # Current observed acceleration is part of AGENT_FEATURES (ax, ay) and
        # is therefore deployable.  v48.68 optionally treats CV and a bounded
        # constant-acceleration continuation as an observation-consistent
        # occupancy set; no teacher future or hidden branch metadata is used.
        acc = np.stack([row[5:7] for _, _, row in agents], axis=0)
        alen = np.asarray([max(abs(float(row[10])), 1.0e-3) for _, _, row in agents], dtype=np.float64)
        awid = np.asarray([max(abs(float(row[11])), 1.0e-3) for _, _, row in agents], dtype=np.float64)
        arad = 0.5 * np.hypot(alen, awid)

        # v48.71 OC-BORW: construct a set-valued, observation-only acceleration
        # envelope from the complete valid history of each currently observed
        # agent.  The axis-aligned acceleration box contains every observed
        # acceleration sample and zero acceleration (the historical CV model).
        # We propagate its circumscribed L2 ball analytically, so the resulting
        # occupancy tube is a conservative reachable set without a new learned
        # predictor, hidden future, regime id, threshold, or tuned horizon.
        hist_acc_center_rows: list[np.ndarray] = []
        hist_acc_radius_rows: list[float] = []
        hist_acc_halfwidth_rows: list[np.ndarray] = []
        hist_acc_samples_rows: list[np.ndarray] = []
        hist_jerk_samples_rows: list[np.ndarray] = []
        for _, aidx, row in agents:
            valid_idx = np.zeros((0,), dtype=np.int64)
            if hist.ndim == 3 and valid_hist.ndim >= 2 and aidx < hist.shape[1] and aidx < valid_hist.shape[1]:
                mask = np.asarray(valid_hist[:, aidx], dtype=bool).reshape(-1)
                ah_all = np.asarray(hist[:, aidx, 5:7], dtype=np.float64)
                mask = mask[: ah_all.shape[0]] & np.isfinite(ah_all).all(axis=1)
                valid_idx = np.flatnonzero(mask)
                ah = ah_all[mask]
            else:
                ah = np.zeros((0, 2), dtype=np.float64)
            if ah.size == 0:
                ah = np.asarray(row[5:7], dtype=np.float64).reshape(1, 2)
                valid_idx = np.asarray([0], dtype=np.int64)
            # v48.73 OC-IRRW: retain temporal adjacency instead of treating the
            # history acceleration set as if any old vector could appear at t=0.
            # Jerk is estimated only between consecutive *valid observations*,
            # dividing by their true sample-index gap.  Zero is always included,
            # so directional response support is non-negative and can express
            # persistence without inventing an inward response.
            if ah.shape[0] >= 2 and valid_idx.size == ah.shape[0]:
                gaps = np.maximum(np.diff(valid_idx).astype(np.float64) * dt, dt)
                jerk = np.diff(ah, axis=0) / gaps[:, None]
                jerk = jerk[np.isfinite(jerk).all(axis=1)]
            else:
                jerk = np.zeros((0, 2), dtype=np.float64)
            jerk = np.concatenate([np.zeros((1, 2), dtype=np.float64), jerk], axis=0)
            # Include zero acceleration because CV remains the signed reference
            # model; the ambiguity set therefore represents persistence/decay
            # uncertainty around the observable acceleration history.
            ah = np.concatenate([np.zeros((1, 2), dtype=np.float64), ah], axis=0)
            amin = np.min(ah, axis=0)
            amax = np.max(ah, axis=0)
            center = 0.5 * (amin + amax)
            halfwidth = 0.5 * (amax - amin)
            radius = float(np.linalg.norm(halfwidth))
            hist_acc_center_rows.append(center)
            hist_acc_radius_rows.append(radius)
            hist_acc_halfwidth_rows.append(halfwidth)
            hist_acc_samples_rows.append(ah)
            hist_jerk_samples_rows.append(jerk)
        hist_acc_center = np.stack(hist_acc_center_rows, axis=0)
        hist_acc_radius = np.asarray(hist_acc_radius_rows, dtype=np.float64)
        hist_acc_halfwidth = np.stack(hist_acc_halfwidth_rows, axis=0)
        # Schema-8/9 diagnostics consume both the directional component-box
        # and empirical-joint-hull support.  Pad the per-agent observation
        # histories once so all agent/time support values can be evaluated in
        # one ordered NumPy kernel.  Padding with zero is exact because zero is
        # already an explicit member of every empirical acceleration set.
        max_hist_acc_samples = max((row.shape[0] for row in hist_acc_samples_rows), default=1)
        hist_acc_samples = np.zeros(
            (len(hist_acc_samples_rows), max_hist_acc_samples, 2), dtype=np.float64
        )
        for agent_idx, row in enumerate(hist_acc_samples_rows):
            hist_acc_samples[agent_idx, : row.shape[0], :] = row
        max_hist_jerk_samples = max((row.shape[0] for row in hist_jerk_samples_rows), default=1)
        hist_jerk_samples = np.zeros(
            (len(hist_jerk_samples_rows), max_hist_jerk_samples, 2), dtype=np.float64
        )
        for agent_idx, row in enumerate(hist_jerk_samples_rows):
            hist_jerk_samples[agent_idx, : row.shape[0], :] = row
    else:
        rel0 = np.zeros((0, 2), dtype=np.float64)
        vel = np.zeros((0, 2), dtype=np.float64)
        acc = np.zeros((0, 2), dtype=np.float64)
        arad = np.zeros((0,), dtype=np.float64)
        hist_acc_center = np.zeros((0, 2), dtype=np.float64)
        hist_acc_radius = np.zeros((0,), dtype=np.float64)
        hist_acc_halfwidth = np.zeros((0, 2), dtype=np.float64)
        hist_acc_samples_rows = []
        hist_acc_samples = np.zeros((0, 1, 2), dtype=np.float64)
        hist_jerk_samples_rows = []
        hist_jerk_samples = np.zeros((0, 1, 2), dtype=np.float64)

    # Acceleration is held only for the already-configured prefix horizon and
    # then propagated with the resulting velocity.  This avoids an unbounded
    # four-second constant-acceleration extrapolation while introducing no new
    # tuned horizon.  When robust_occupancy_envelope=False the helper is exactly
    # the historical CV forecast.
    accel_hold_s = max(float(cfg.get("prefix_horizon_s", 1.0) or 1.0), dt)

    def _agent_future(times: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        tt = np.asarray(times, dtype=np.float64).reshape(-1)
        cv = rel0[None, :, :] + tt[:, None, None] * vel[None, :, :]
        if not (robust_occupancy_envelope or soft_occupancy_disagreement or boundary_localized_occupancy_trust or history_occupancy_reachability or interaction_box_support or interaction_hull_support or interaction_anchor_support or interaction_response_support) or not agents:
            return cv, None
        hold = np.minimum(tt, accel_hold_s)
        accel_disp = (tt * hold - 0.5 * hold * hold)[:, None, None] * acc[None, :, :]
        ca = cv + accel_disp
        return cv, ca

    def _signed_clearance_components(ego_xy_rel_t: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        cv, ca = _agent_future(times)
        delta_cv = ego_xy_rel_t[:, None, :] - cv
        clear_cv = np.linalg.norm(delta_cv, axis=-1) - ego_rad - arad[None, :]
        if ca is None:
            return clear_cv, None
        delta_ca = ego_xy_rel_t[:, None, :] - ca
        clear_ca = np.linalg.norm(delta_ca, axis=-1) - ego_rad - arad[None, :]
        return clear_cv, clear_ca

    def _history_tube_clearance(ego_xy_rel_t: np.ndarray, times: np.ndarray) -> np.ndarray | None:
        if not (history_occupancy_reachability or interaction_box_support or interaction_hull_support or interaction_anchor_support or interaction_response_support) or not agents:
            return None
        tt = np.asarray(times, dtype=np.float64).reshape(-1)
        hold = np.minimum(tt, accel_hold_s)
        coeff = (tt * hold - 0.5 * hold * hold)[:, None]
        center = (
            rel0[None, :, :]
            + tt[:, None, None] * vel[None, :, :]
            + coeff[:, :, None] * hist_acc_center[None, :, :]
        )
        radius = coeff * hist_acc_radius[None, :]
        delta = ego_xy_rel_t[:, None, :] - center
        return np.linalg.norm(delta, axis=-1) - ego_rad - arad[None, :] - radius

    def _interaction_support_clearances(
        ego_xy_rel_t: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return component-box and empirical-hull directional clearances.

        V48.71 converted the componentwise acceleration box into an isotropic
        circumscribed ball.  That is conservative but direction-blind: lateral
        acceleration spread is charged even when it cannot move the agent toward
        the candidate recovery.  V48.72 evaluates support along the
        candidate-specific CV line of sight.  The box support is analytic; the
        empirical convex-hull support is the maximum projection of the observed
        joint acceleration samples (plus zero).

        Both diagnostics are always emitted by schema 8, irrespective of which
        one the model consumes.  Computing their common CV geometry once is an
        execution-exact engineering optimization and lets the two causal arms
        reuse one persistent tensor cache.
        """
        if not (
            interaction_box_support
            or interaction_hull_support
            or interaction_anchor_support
            or interaction_response_support
        ) or not agents:
            return None, None
        tt = np.asarray(times, dtype=np.float64).reshape(-1)
        hold = np.minimum(tt, accel_hold_s)
        coeff = (tt * hold - 0.5 * hold * hold)[:, None]
        cv = rel0[None, :, :] + tt[:, None, None] * vel[None, :, :]
        rel = ego_xy_rel_t[:, None, :] - cv
        dist = np.linalg.norm(rel, axis=-1)
        n = rel / np.maximum(dist[..., None], 1.0e-9)

        support_box = (
            np.sum(n * hist_acc_center[None, :, :], axis=-1)
            + np.sum(np.abs(n) * hist_acc_halfwidth[None, :, :], axis=-1)
        )
        support_box = np.maximum(support_box, 0.0)

        # [time, agent, sample].  Padded zero samples are semantically neutral
        # because zero is already part of every hull, and preserve the old
        # per-agent max-projection result exactly after float32 feature storage.
        sample_projection = np.einsum(
            "taj,asj->tas", n, hist_acc_samples, optimize=True
        )
        support_hull = np.max(sample_projection, axis=-1)

        base = dist - ego_rad - arad[None, :]
        return base - coeff * support_box, base - coeff * support_hull

    def _interaction_response_clearances(
        ego_xy_rel_t: np.ndarray, times: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return v48.73 anchored-ramp and observed-jerk response clearances.

        V48.72's empirical acceleration hull is candidate-oriented but
        temporally static: any stale history acceleration can be charged from
        the first future instant and held for the entire existing prefix
        horizon. V48.73 preserves the joint empirical hull and line-of-sight
        support while anchoring the future response at the latest observation.

        N73 uses a parameter-free linear ramp from current acceleration to the
        empirical-hull directional support over the existing prefix horizon.
        O73/Main replaces that artificial ramp rate with the maximum observed
        directional jerk, capped at the same hull support. Both diagnostics are
        emitted together so the causal arms share one tensor cache and common
        geometry. Zero padding is exact because zero is explicitly in every
        acceleration and jerk ambiguity set.
        """
        if not (interaction_anchor_support or interaction_response_support) or not agents:
            return None, None
        tt = np.asarray(times, dtype=np.float64).reshape(-1)
        h = np.minimum(tt, accel_hold_s)
        cv = rel0[None, :, :] + tt[:, None, None] * vel[None, :, :]
        rel = ego_xy_rel_t[:, None, :] - cv
        dist = np.linalg.norm(rel, axis=-1)
        n = rel / np.maximum(dist[..., None], 1.0e-9)

        sigma_a = np.max(
            np.einsum("taj,asj->tas", n, hist_acc_samples, optimize=True),
            axis=-1,
        )
        sigma_a = np.maximum(sigma_a, 0.0)
        a0 = np.einsum("taj,aj->ta", n, acc, optimize=True)
        sigma_a = np.maximum(sigma_a, a0)

        H = max(float(accel_hold_s), 1.0e-9)
        c0 = tt * h - 0.5 * h * h
        c1 = 0.5 * tt * h * h - (h * h * h) / 3.0
        anchor_disp = a0 * c0[:, None] + ((sigma_a - a0) / H) * c1[:, None]

        sigma_j = np.max(
            np.einsum("taj,asj->tas", n, hist_jerk_samples, optimize=True),
            axis=-1,
        )
        sigma_j = np.maximum(sigma_j, 0.0)
        gap = np.maximum(sigma_a - a0, 0.0)
        hit = np.full_like(gap, np.inf)
        moving = sigma_j > 1.0e-12
        hit[moving] = gap[moving] / sigma_j[moving]
        q = np.minimum(h[:, None], hit)
        tt2 = tt[:, None]
        hh = h[:, None]
        first = (
            a0 * (tt2 * q - 0.5 * q * q)
            + sigma_j * (0.5 * tt2 * q * q - (q * q * q) / 3.0)
        )
        second = sigma_a * (tt2 * (hh - q) - 0.5 * (hh * hh - q * q))
        response_disp = first + second

        base = dist - ego_rad - arad[None, :]
        return base - anchor_disp, base - response_disp

    def _signed_clearance(ego_xy_rel_t: np.ndarray, times: np.ndarray) -> np.ndarray:
        clear_cv, clear_ca = _signed_clearance_components(ego_xy_rel_t, times)
        if robust_occupancy_envelope and clear_ca is not None:
            return np.minimum(clear_cv, clear_ca)
        return clear_cv

    prefix_duration = float(states.shape[0]) * dt
    terminal_xy_rel = states[-1, :2].astype(np.float64) - ego_xy
    if agents:
        prefix_terminal_clear = float(_signed_clearance(
            terminal_xy_rel.reshape(1, 2), np.asarray([prefix_duration], dtype=np.float64)
        ).min())
    else:
        prefix_terminal_clear = 40.0

    # v48.64 observable active-set evidence.  This is deliberately derived
    # from the executable candidate prefix and current observation only; no
    # regime label or teacher-future contact flag is exposed.  A stability
    # constraint is considered physically active when the prefix itself has
    # already entered contact/near-overlap or is dynamically unstable.
    if agents:
        prefix_times = (np.arange(states.shape[0], dtype=np.float64) + 1.0) * dt
        prefix_xy_rel = states[:, :2].astype(np.float64) - ego_xy[None, :]
        prefix_signed_clearance = _signed_clearance(prefix_xy_rel, prefix_times)
        prefix_min_clear = float(np.min(prefix_signed_clearance))
    else:
        prefix_min_clear = 40.0

    scales = cfg.get("margin_scales", {}) if isinstance(cfg.get("margin_scales", {}), dict) else {}
    distance_scale = max(float(scales.get("distance", 2.0)), 1.0e-6)
    stop_scale = max(float(scales.get("stop", 5.0)), 1.0e-6)
    accel_scale = max(float(scales.get("accel", 1.0)), 1.0e-6)
    decel_scale = max(float(scales.get("decel", 1.0)), 1.0e-6)
    steer_scale = max(float(scales.get("steer", 0.1)), 1.0e-6)
    jerk_scale = max(float(scales.get("jerk", 2.0)), 1.0e-6)
    rate_scale = max(float(scales.get("steer_rate", 0.1)), 1.0e-6)
    yaw_scale = max(float(scales.get("yaw", 0.2)), 1.0e-6)
    route_scale = max(float(scales.get("route", 1.0)), 1.0e-6)
    route_dev_max = float(cfg.get("route_dev_max_m", 2.5))
    d_safe0 = float(cfg.get("d_safe0_m", 1.0))
    headway = float(cfg.get("safe_time_headway_s", 0.5))
    limits = cfg.get("control_limits", {}) if isinstance(cfg.get("control_limits", {}), dict) else {}
    a_max = float(limits.get("a_max", 3.0))
    a_min = float(limits.get("a_min", -6.0))
    delta_max = float(limits.get("delta_max", 0.55))
    jerk_max = float(limits.get("j_max", 6.0))
    steer_rate_max = float(limits.get("steer_rate_max", 0.5))
    yaw_rate_max = float(cfg.get("yaw_rate_max_rps", 0.6))
    stop_decel = max(abs(a_min), 1.0e-3)

    feature_dim = (
        22 if (interaction_anchor_support or interaction_response_support) else
        (20 if (interaction_box_support or interaction_hull_support or interaction_anchor_support or interaction_response_support) else
        (18 if (boundary_localized_occupancy_trust or history_occupancy_reachability) else
        (15 if soft_occupancy_disagreement else
        (14 if include_active_constraint_tail else
        (12 if include_semantic_alignment_tail else
         (10 if include_recovery_stability_tail else DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_DIM))))))
    )
    field = np.zeros((L, feature_dim), dtype=np.float32)
    for l in range(raw_L):
        valid_l = bool(option_valid_raw[l]) if l < option_valid_raw.size else True
        if not valid_l:
            continue
        if l >= len(modes) or l >= params.shape[0]:
            raise ValueError(f"ERWF option {l} missing mode/params")
        mode = _decode_recovery_mode(modes[l])
        option = RecoveryOption(
            option_id=l,
            mode=mode,
            params=np.asarray(params[l], dtype=np.float32).reshape(-1),
            valid=True,
        )
        raw_rec_states = raw_rec_controls = None
        if project_control_envelope and projection_fidelity_weighting:
            raw_rec_states, raw_rec_controls, _raw_diag = rollout_recovery_controller(
                prefix, option, horizon_steps, cfg, project_control_envelope=False,
            )
        rec_states, rec_controls, _diag = rollout_recovery_controller(
            prefix, option, horizon_steps, cfg,
            project_control_envelope=bool(project_control_envelope),
        )
        rec_states = np.asarray(rec_states, dtype=np.float64)
        rec_controls = np.asarray(rec_controls, dtype=np.float64)
        if rec_states.ndim != 2 or rec_states.shape[0] < 1 or rec_states.shape[1] < 9:
            raise ValueError(f"ERWF recovery rollout invalid for option={l}: shape={rec_states.shape}")

        max_speed = float(np.max(np.abs(rec_states[:, 6])))
        terminal_speed = float(abs(rec_states[-1, 6]))
        d_safe = d_safe0 + headway * max_speed
        signed_viability_debt1 = 0.0
        signed_viability_debt2 = 0.0
        if agents:
            # rec_states[0] is exactly the candidate terminal state at
            # prefix_duration; later rows are separated by dt.
            times = prefix_duration + np.arange(rec_states.shape[0], dtype=np.float64) * dt
            rec_xy_rel = rec_states[:, :2] - ego_xy[None, :]
            clear_cv, clear_ca = _signed_clearance_components(rec_xy_rel, times)
            signed_clearance = (
                np.minimum(clear_cv, clear_ca)
                if robust_occupancy_envelope and clear_ca is not None
                else clear_cv
            )
            if _v48_74_signed_viability_enabled() and (interaction_anchor_support or interaction_response_support):
                # V48.74 OC-SVBW consumes the *same actuator-projected recovery*
                # and historical observation-only CV agent continuation already
                # used by the accepted physical witness.  Time is placed on the
                # final axis; the non-time agent axis is reduced by worst-case
                # debt.  Pair radii provide a physical, parameter-free length
                # normalization and introduce no learned/swept hyperparameter.
                pair_scale = np.maximum(ego_rad + arad, 1.0e-8)
                svbw = _v48_74_signed_viability_diagnostics(
                    np.asarray(signed_clearance, dtype=np.float64).T,
                    times,
                    clearance_scale=pair_scale,
                    reduce_axes=(0,),
                )
                signed_viability_debt1 = float(np.asarray(svbw.first_order_debt).reshape(()))
                signed_viability_debt2 = float(np.asarray(svbw.second_order_debt).reshape(()))
                if not (np.isfinite(signed_viability_debt1) and np.isfinite(signed_viability_debt2)):
                    raise ValueError(
                        f"non-finite V48.74 signed-viability debt for option={l}: "
                        f"d1={signed_viability_debt1}, d2={signed_viability_debt2}"
                    )
            c_min = float(np.min(signed_clearance))
            c_terminal = float(np.min(signed_clearance[-1]))
            # v48.70 diagnostic retained for exact historical comparison.
            # It measures raw point-prediction disagreement irrespective of the
            # physical safety boundary and is intentionally *not* the v48.71 Main
            # trust variable.
            if clear_ca is not None:
                occupancy_optimism_gap = float(
                    np.max(np.maximum(clear_cv - clear_ca, 0.0)) / distance_scale
                )
                current_boundary_deficit = float(
                    np.max(np.maximum(d_safe - clear_ca, 0.0)) / distance_scale
                )
            else:
                occupancy_optimism_gap = 0.0
                current_boundary_deficit = 0.0

            clear_hist_tube = _history_tube_clearance(rec_xy_rel, times)
            if clear_hist_tube is not None:
                history_optimism_gap = float(
                    np.max(np.maximum(clear_cv - clear_hist_tube, 0.0)) / distance_scale
                )
                history_boundary_deficit = float(
                    np.max(np.maximum(d_safe - clear_hist_tube, 0.0)) / distance_scale
                )
            else:
                history_optimism_gap = 0.0
                history_boundary_deficit = 0.0

            clear_interaction_box, clear_interaction_hull = (
                _interaction_support_clearances(rec_xy_rel, times)
            )
            interaction_box_optimism = (
                float(np.max(np.maximum(clear_cv - clear_interaction_box, 0.0)) / distance_scale)
                if clear_interaction_box is not None else 0.0
            )
            interaction_hull_optimism = (
                float(np.max(np.maximum(clear_cv - clear_interaction_hull, 0.0)) / distance_scale)
                if clear_interaction_hull is not None else 0.0
            )
            clear_interaction_anchor, clear_interaction_response = (
                _interaction_response_clearances(rec_xy_rel, times)
            )
            interaction_anchor_optimism = (
                float(np.max(np.maximum(clear_cv - clear_interaction_anchor, 0.0)) / distance_scale)
                if clear_interaction_anchor is not None else 0.0
            )
            interaction_response_optimism = (
                float(np.max(np.maximum(clear_cv - clear_interaction_response, 0.0)) / distance_scale)
                if clear_interaction_response is not None else 0.0
            )
        else:
            c_min = c_terminal = 40.0
            occupancy_optimism_gap = 0.0
            current_boundary_deficit = 0.0
            history_optimism_gap = 0.0
            history_boundary_deficit = 0.0
            interaction_box_optimism = 0.0
            interaction_hull_optimism = 0.0
            interaction_anchor_optimism = 0.0
            interaction_response_optimism = 0.0

        h_min_clear = np.tanh((c_min - d_safe) / distance_scale)
        h_terminal_clear = np.tanh((c_terminal - d_safe) / distance_scale)
        h_clear_gain = np.tanh((c_terminal - prefix_terminal_clear) / distance_scale)
        stop_required = terminal_speed * terminal_speed / (2.0 * stop_decel) + d_safe0
        h_stop = np.tanh((c_terminal - stop_required) / stop_scale)

        control_source = (
            np.asarray(raw_rec_controls, dtype=np.float64)
            if (project_control_envelope and projection_fidelity_weighting and raw_rec_controls is not None)
            else rec_controls
        )
        if control_source.ndim == 2 and control_source.shape[0] > 0 and control_source.shape[1] >= 4:
            rc = control_source[:, :4]
            ctrl_terms = np.asarray([
                (a_max - float(np.max(np.maximum(rc[:, 0], 0.0)))) / accel_scale,
                (abs(a_min) - float(np.max(np.maximum(-rc[:, 0], 0.0)))) / decel_scale,
                (delta_max - float(np.max(np.abs(rc[:, 1])))) / steer_scale,
                (jerk_max - float(np.max(np.abs(rc[:, 2])))) / jerk_scale,
                (steer_rate_max - float(np.max(np.abs(rc[:, 3])))) / rate_scale,
            ], dtype=np.float64)
            h_control = np.tanh(float(np.min(ctrl_terms)))
        else:
            h_control = np.float64(1.0)
        yaw = float(np.max(np.abs(rec_states[:, 5])))
        h_stability = np.tanh((yaw_rate_max - yaw) / yaw_scale)
        values = [
            h_min_clear,
            h_terminal_clear,
            h_clear_gain,
            h_stop,
            h_control,
            h_stability,
        ]
        if include_recovery_stability_tail:
            terminal_yaw = float(abs(rec_states[-1, 5]))
            initial_yaw = float(abs(rec_states[0, 5]))
            h_terminal_stability = np.tanh((yaw_rate_max - terminal_yaw) / yaw_scale)
            h_stability_gain = np.tanh((initial_yaw - terminal_yaw) / yaw_scale)
            # Path-preservation witnesses prevent a terminal recovery from hiding
            # a worse secondary excursion during the recovery continuation.
            h_clearance_floor_gain = np.tanh((c_min - prefix_terminal_clear) / distance_scale)
            h_stability_floor_gain = np.tanh((initial_yaw - yaw) / yaw_scale)
            values.extend([
                h_terminal_stability, h_stability_gain,
                h_clearance_floor_gain, h_stability_floor_gain,
            ])
        if include_semantic_alignment_tail:
            # Constraint-native stopping semantics: stopping reserve is path
            # capacity, not radial terminal clearance.  We use the executable
            # recovery path and the same observable-agent CV prediction already
            # used by the continuation witness.  If no predicted conflict occurs,
            # the public default available distance supplies the open-path cap.
            if agents:
                step_delta = np.diff(rec_states[:, :2], axis=0)
                step_len = np.linalg.norm(step_delta, axis=-1)
                path_s = np.concatenate([[0.0], np.cumsum(step_len)])
                min_clear_t = np.min(signed_clearance, axis=1)
                conflict = np.flatnonzero(min_clear_t <= 0.0)
                if conflict.size:
                    s_available = float(path_s[int(conflict[0])])
                else:
                    s_available = float(cfg.get("default_available_distance_m", 60.0))
            else:
                s_available = float(cfg.get("default_available_distance_m", 60.0))
            # Match the structural teacher's option-native stopping demand:
            # the first option parameter controls the stopping/braking scale.
            # This is only activated later for stopping-semantic options.
            option_decel = max(2.0 * abs(float(option.params[0])) if option.params.size else 4.0, 1.0)
            teacher_style_stop_required = float(rec_states[0, 6] ** 2 / option_decel)
            h_path_stop = np.tanh((s_available - teacher_style_stop_required) / stop_scale)
            prefix_unstable = float(np.max(np.abs(states[:, 5]))) > yaw_rate_max
            stability_active = bool(
                mode == "post_contact_stabilize" or prefix_min_clear <= 0.0 or prefix_unstable
            )
            values.extend([h_path_stop, 1.0 if stability_active else 0.0])
        if include_active_constraint_tail:
            # v48.66 observable active-constraint coverage repair.
            #
            # Route is fully observation-certifiable and uses the same local
            # executable-recovery coordinate as the structural teacher.  It is
            # inactive for post-contact stabilization exactly as in the teacher
            # active mask.  No hidden route_blocked flag is read.
            route_active = mode != "post_contact_stabilize"
            d_route = float(np.max(np.abs(rec_states[:, 1]))) if rec_states.size else 0.0
            h_route = (
                np.tanh((route_dev_max - d_route) / route_scale)
                if route_active else np.float64(1.0)
            )

            # Persistent re-entry is the observation-only counterpart of the
            # secondary-collision requirement.  The existing v48.62 finite-time
            # clearance logic correctly permits recovery from an already
            # violated initial state, but it may still accept a later re-contact
            # if that excursion is shallower than the initial penetration.  Once
            # an observed-contact recovery re-enters the positive safe-clearance
            # set, require that it remains there for the rest of the executable
            # continuation.  This uses only the same current-observation CV
            # occupancy forecast already used by the clearance witness.
            reentry_active = bool(
                prefix_min_clear <= 0.0 or mode in {"post_contact_stabilize", "avoid_secondary"}
            )
            if not reentry_active:
                h_reentry = np.float64(1.0)
            elif not agents:
                h_reentry = np.float64(1.0)
            else:
                min_clear_t = np.min(signed_clearance, axis=1)
                reserve_t = min_clear_t - d_safe
                reentered = np.flatnonzero(reserve_t >= 0.0)
                if reentered.size:
                    first_reentry = int(reentered[0])
                    persistent_reserve = float(np.min(reserve_t[first_reentry:]))
                else:
                    # No finite-time re-entry: preserve signed boundary rather
                    # than inventing a free veto or a tuned threshold.
                    persistent_reserve = float(reserve_t[-1])
                h_reentry = np.tanh(persistent_reserve / distance_scale)
            values.extend([h_route, h_reentry])
        if interaction_anchor_support or interaction_response_support:
            # Coordinates 0--19 stay execution-exact to v48.72/v48.73.
            # With the V48.74 switch enabled, coordinates 20/21 are the raw
            # normalized first/high-order finite-time signed-viability debts.
            # With the switch disabled, preserve historical v48.73 schema-9
            # tanh(anchor/jerk optimism) values bitwise.
            tail_20_21 = (
                [float(signed_viability_debt1), float(signed_viability_debt2)]
                if _v48_74_signed_viability_enabled()
                else [
                    float(np.tanh(interaction_anchor_optimism)),
                    float(np.tanh(interaction_response_optimism)),
                ]
            )
            values.extend([
                float(np.tanh(occupancy_optimism_gap)),
                float(np.tanh(current_boundary_deficit)),
                float(np.tanh(history_optimism_gap)),
                float(np.tanh(history_boundary_deficit)),
                float(np.tanh(interaction_box_optimism)),
                float(np.tanh(interaction_hull_optimism)),
                *tail_20_21,
            ])
        elif interaction_box_support or interaction_hull_support:
            # v48.72 schema-8: coordinates 0--17 are execution-exact v48.71
            # diagnostics.  18 is component-box directional support optimism;
            # 19 is empirical-convex-hull directional support optimism.
            values.extend([
                float(np.tanh(occupancy_optimism_gap)),
                float(np.tanh(current_boundary_deficit)),
                float(np.tanh(history_optimism_gap)),
                float(np.tanh(history_boundary_deficit)),
                float(np.tanh(interaction_box_optimism)),
                float(np.tanh(interaction_hull_optimism)),
            ])
        elif boundary_localized_occupancy_trust or history_occupancy_reachability:
            # v48.71 schema-7 diagnostics.  Coordinate 14 preserves v48.70 raw
            # current-CA optimism for direct ablation.  Coordinate 15 is the
            # current-CA *boundary deficit*.  Coordinates 16/17 are the analogous
            # history-reachability-tube optimism and boundary deficit.  All are
            # non-negative confidence diagnostics; the signed CV certificate in
            # coordinates 0--13 remains unchanged.
            values.extend([
                float(np.tanh(occupancy_optimism_gap)),
                float(np.tanh(current_boundary_deficit)),
                float(np.tanh(history_optimism_gap)),
                float(np.tanh(history_boundary_deficit)),
            ])
        elif soft_occupancy_disagreement:
            # Strictly non-negative bounded v48.70 diagnostic.
            values.append(float(np.tanh(occupancy_optimism_gap)))
        field[l] = np.asarray(values, dtype=np.float32)

    if field.shape != (L, feature_dim) or not np.isfinite(field).all():
        raise ValueError(f"invalid ERWF feature field shape/value: shape={field.shape}")
    return field


# v48.62 OC-CWRF (Observation-Consistent Common-Witness Recovery Field).
# The first six coordinates are execution-identical to ERWF.  Two recovery-tail
# stability coordinates make post-contact recovery non-compensatory without
# treating the already-violated initial state as permanently infeasible.
DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA = 1
DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_DIM = 10


def direct_common_recovery_witness_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None, *, num_options: int | None = None
) -> np.ndarray:
    """Return the v48.62 option-resolved finite-time recovery witness field.

    Feature order:
      0 continuation minimum-clearance reserve
      1 continuation terminal-clearance reserve
      2 clearance recovery gain from candidate terminal
      3 terminal stopping reserve
      4 controller control-envelope reserve
      5 minimum stability reserve over the continuation
      6 terminal stability reserve
      7 stability recovery gain (initial |yaw| - terminal |yaw|)
      8 clearance floor gain (minimum continuation clearance - initial clearance)
      9 stability floor gain (initial |yaw| - maximum continuation |yaw|)

    No regime ID, latent-root identity, teacher component/future or held-out label
    is read.  This is the same deterministic executable continuation used by
    ERWF, augmented only with finite-time recovery semantics for stability.
    """
    out = direct_executable_recovery_witness_features_from_sample(
        d, cfg, num_options=num_options, include_recovery_stability_tail=True
    )
    if out.ndim != 2 or out.shape[1] != DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_DIM:
        raise ValueError(f"invalid OC-CWRF feature field shape: {out.shape}")
    return out


# v48.64 OC-SARW (Observation-Consistent Semantics-Aligned Recovery Witness).
# The first ten coordinates are execution-identical to v48.62/v48.63.  The
# final two coordinates repair the two constraint-semantics mismatches exposed
# by the v48.63 quantifier-coverage diagnostic: path-capacity stopping reserve
# and an observable stability active-set indicator.
DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA = 1
DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_DIM = 12

# v48.66 extends the v48.64/v48.65 side channel without changing its first
# twelve coordinates.  Schema 2 appends two observation-only active-constraint
# coordinates: route consistency and post-contact persistent re-entry.
DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA = 2
DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_DIM = 14

# v48.67 keeps the 14-dimensional active-constraint field but changes how the
# executable recovery trace may be realized: schema 3 explicitly records the
# actuator-projected witness contract so persistent caches/checkpoints cannot
# silently mix projected and historical unprojected traces.
DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA = 3
DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_DIM = 14

# v48.68 keeps the 14-D layout but changes two observation-only semantics:
# robust CV/CA occupancy and soft projection-fidelity weighting.  A distinct
# schema prevents persistent caches/checkpoints from mixing those values with
# v48.67 even though the tensor shape is unchanged.
DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA = 4
DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_DIM = 14

# v48.69 keeps the exact same 14-D observable side-channel as v48.68 T; only
# the model-side interpretation of raw projection severity changes by
# observation-derived recovery demand.  A distinct schema prevents checkpoint
# or persistent-cache mixing despite byte-identical feature tensors.
DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA = 5
DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_DIM = 14

# v48.70 appends one observation-only *soft* occupancy-disagreement coordinate.
# The first 14 coordinates are byte-semantically v48.68-T/v48.69-D; coordinate
# 14 records normalized CV optimism relative to the already-defined bounded
# current-acceleration counterfactual.  It is confidence only, never a hard
# physical barrier.
DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA = 6
DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_DIM = 15

# v48.71 OC-BORW replaces raw point-disagreement trust with a boundary-localized
# observation-history reachability construction.  The first 14 coordinates stay
# byte-semantically identical to v48.68-T; coordinate 14 is the v48.70 current-CA
# raw optimism diagnostic, 15 is current-CA boundary deficit, 16 is history-tube
# raw optimism, and 17 is history-tube boundary deficit.
DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA = 7
DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_DIM = 18

# v48.72 OC-IORW appends two interaction-oriented reachability diagnostics.
# The signed CV certificate and all first 18 coordinates remain execution-exact
# to v48.71 schema 7.  Coordinate 18 is support-function erosion for the full
# componentwise history box; coordinate 19 uses the tighter empirical convex
# hull conv({0,a_tau}), excluding unobserved Cartesian corner combinations.
DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA = 8
DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_DIM = 20

# v48.73 OC-IRRW retains every v48.72 coordinate and appends two temporal
# interaction-response diagnostics.  Coordinate 20 anchors the empirical hull to
# current acceleration with a parameter-free ramp over the existing prefix hold;
# coordinate 21 additionally restricts evolution by observed history jerk.
DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA = 9
DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_DIM = 22


def direct_semantic_recovery_witness_features_from_sample(
    d: dict[str, Any], cfg: dict | None = None, *, num_options: int | None = None
) -> np.ndarray:
    """Return the v48.64 semantics-aligned option-resolved witness field.

    Coordinates 0--9 are exactly OC-CWRF/OC-QARW.  Coordinate 10 is a
    path-capacity stopping reserve computed from the executable continuation
    and current observable agents.  Coordinate 11 is an observable active-set
    indicator for stability (prefix contact/instability or explicit
    post-contact-stabilize semantics).  Neither coordinate reads a regime id,
    teacher future/component margin, held-out label, or latent-root identity.
    """
    cfg = cfg or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    control_projection = bool(
        model_cfg.get("direct_recovery_semantic_witness_control_projection", False)
    )
    boundary_transport = bool(
        model_cfg.get("direct_recovery_semantic_witness_boundary_transport", False)
    )
    projection_fidelity = bool(
        model_cfg.get("direct_recovery_semantic_witness_projection_fidelity_weighting", False)
    )
    demand_normalized_fidelity = bool(
        model_cfg.get("direct_recovery_semantic_witness_demand_normalized_fidelity", False)
    )
    robust_occupancy = bool(
        model_cfg.get("direct_recovery_semantic_witness_robust_occupancy", False)
    )
    soft_occupancy_disagreement = bool(
        model_cfg.get("direct_recovery_semantic_witness_soft_occupancy_disagreement", False)
    )
    boundary_localized_occupancy_trust = bool(
        model_cfg.get("direct_recovery_semantic_witness_boundary_localized_occupancy_trust", False)
    )
    history_occupancy_reachability = bool(
        model_cfg.get("direct_recovery_semantic_witness_history_occupancy_reachability", False)
    )
    interaction_box_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_box_support", False)
    )
    interaction_hull_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_hull_support", False)
    )
    interaction_anchor_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_anchor_support", False)
    )
    interaction_response_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_response_support", False)
    )
    active_constraint_tail = bool(
        model_cfg.get("direct_recovery_semantic_witness_route_alignment", False)
        or model_cfg.get("direct_recovery_semantic_witness_reentry_alignment", False)
        or control_projection or boundary_transport or projection_fidelity
        or demand_normalized_fidelity or robust_occupancy or soft_occupancy_disagreement
        or boundary_localized_occupancy_trust or history_occupancy_reachability
        or interaction_box_support or interaction_hull_support
        or interaction_anchor_support or interaction_response_support
    )
    out = direct_executable_recovery_witness_features_from_sample(
        d, cfg, num_options=num_options, include_recovery_stability_tail=True,
        include_semantic_alignment_tail=True,
        include_active_constraint_tail=active_constraint_tail,
        project_control_envelope=control_projection,
        projection_fidelity_weighting=projection_fidelity,
        robust_occupancy_envelope=robust_occupancy,
        soft_occupancy_disagreement=soft_occupancy_disagreement,
        boundary_localized_occupancy_trust=boundary_localized_occupancy_trust,
        history_occupancy_reachability=history_occupancy_reachability,
        interaction_box_support=interaction_box_support,
        interaction_hull_support=interaction_hull_support,
        interaction_anchor_support=interaction_anchor_support,
        interaction_response_support=interaction_response_support,
    )
    expected_dim = (
        DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_DIM
        if (interaction_anchor_support or interaction_response_support) else
        (DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_DIM
        if (interaction_box_support or interaction_hull_support) else
        (DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_DIM
        if (boundary_localized_occupancy_trust or history_occupancy_reachability) else
        (DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_DIM
        if soft_occupancy_disagreement else
        (DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_DIM
         if active_constraint_tail else DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_DIM))))
    )
    if out.ndim != 2 or out.shape[1] != expected_dim:
        raise ValueError(f"invalid OC-SARW feature field shape: {out.shape}")
    return out


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





@lru_cache(maxsize=8)
def _load_absolute_truth_index(path_str: str) -> dict[str, dict[str, Any]]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"absolute feasibility truth index not found: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid truth-index JSONL at {path}:{line_no}: {exc}") from exc
            sample_path = str(row.get("sample_path", "")).strip()
            if not sample_path:
                raise ValueError(f"missing sample_path in truth-index row {path}:{line_no}")
            key = str(Path(sample_path).expanduser().resolve())
            if key in out:
                raise ValueError(f"duplicate sample_path in truth index: {key}")
            out[key] = row
    return out


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


def _nominal_deviation_by_path(paths: list[Path]) -> list[float]:
    """Compute the certificate's candidate-vs-nominal prefix deviation.

    v48.31 keeps validation/checkpoint selection on exactly the same eligible
    population as policy calibration.  The value is precomputed once because a
    normal Dataset item has no access to the other candidates in its scene-time
    group.
    """

    records: list[tuple[tuple[int, str, int], bool, np.ndarray | None]] = []
    grouped: dict[tuple[int, str, int], list[int]] = {}
    for index, path in enumerate(paths):
        d = load_npz_selected(path, NOMINAL_DEVIATION_NPZ_KEYS)
        scene = str(np.asarray(d.get("scene_id", path.stem)).item())
        time_index = int(np.asarray(d.get("time_index", 0)).item())
        key = (bucket_id_for_path(path), scene, time_index)
        is_nominal = bool(float(np.asarray(d.get("is_nominal", 0.0)).item()) > 0.5)
        try:
            prefix = np.asarray(d.get("prefix_states"), dtype=np.float64)[:, :2]
        except Exception:
            prefix = None
        records.append((key, is_nominal, prefix))
        grouped.setdefault(key, []).append(index)
    out = [0.0] * len(paths)
    for indices in grouped.values():
        nominal_index = next((i for i in indices if records[i][1]), None)
        if nominal_index is None:
            continue
        reference = records[nominal_index][2]
        if reference is None:
            continue
        for i in indices:
            prefix = records[i][2]
            if prefix is None:
                continue
            length = min(len(reference), len(prefix))
            if length <= 0:
                continue
            out[i] = float(
                np.sqrt(np.mean(np.sum((prefix[:length] - reference[:length]) ** 2, axis=-1)))
                / 5.0
            )
    return out


_PERSISTENT_TENSOR_CACHE_SCHEMA = 7

def _load_persistent_tensor_cache_payload(cache_path: Path) -> dict[str, Any] | None:
    """Load an immutable decoded-tensor cache with mmap when supported.

    Multiple v48.46 arms/variants may read the same cache concurrently.  mmap
    keeps tensor storages file-backed so the OS page cache can be shared across
    processes instead of materialising another full host-RAM copy per trainer.
    The payload contains tensors plus primitive provenance only, so
    ``weights_only=True`` is sufficient on modern PyTorch.  Older PyTorch
    versions fall back to the historical loader.  A malformed/truncated cache
    is treated as a cache miss and rebuilt under the existing flock rather than
    becoming an experiment-level RC=30.
    """
    attempts = (
        {"weights_only": True, "mmap": True},
        {"weights_only": True},
        {"weights_only": False},
    )
    for extra in attempts:
        try:
            payload = torch.load(cache_path, map_location="cpu", **extra)
            return payload if isinstance(payload, dict) else None
        except TypeError:
            # ``mmap``/``weights_only`` are version-dependent keyword args.
            continue
        except Exception:
            # A second loader mode may still work (for example mmap unsupported
            # by the filesystem).  If every mode fails the caller rebuilds.
            continue
    return None

def _dataset_manifest_fingerprint(paths: list[Path]) -> list[dict[str, Any]]:
    """Stable provenance for a decoded-tensor cache without opening every NPZ."""
    roots = sorted({_dataset_root_for_sample(p).resolve() for p in paths}, key=lambda x: str(x))
    rows: list[dict[str, Any]] = []
    for root in roots:
        manifest = root / "manifest.csv"
        if manifest.is_file():
            raw = manifest.read_bytes()
            rows.append({
                "root": str(root),
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "manifest_size": len(raw),
            })
        else:
            # Fail-safe fallback for non-manifest datasets used by older tests.
            member = [p for p in paths if _dataset_root_for_sample(p).resolve() == root]
            rows.append({
                "root": str(root),
                "members": [(str(p.resolve()), int(p.stat().st_size), int(p.stat().st_mtime_ns)) for p in member],
            })
    return rows

def _persistent_tensor_cache_key(
    paths: list[Path], cfg: dict, *, num_roots: int, num_options: int,
    d_signature: int, d_future_signature: int, feature_dim: int,
) -> str:
    training = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    cphr_features_enabled = bool(model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False))
    erwf_features_enabled = bool(model_cfg.get("direct_recovery_absolute_executable_witness_correction", False))
    common_witness_features_enabled = bool(
        model_cfg.get("direct_recovery_absolute_common_witness_correction", False)
        or model_cfg.get("direct_recovery_absolute_quantifier_witness_correction", False)
    )
    semantic_witness_features_enabled = bool(
        model_cfg.get("direct_recovery_absolute_semantic_witness_correction", False)
    )
    semantic_route_alignment = bool(
        model_cfg.get("direct_recovery_semantic_witness_route_alignment", False)
    )
    semantic_reentry_alignment = bool(
        model_cfg.get("direct_recovery_semantic_witness_reentry_alignment", False)
    )
    semantic_control_projection = bool(
        model_cfg.get("direct_recovery_semantic_witness_control_projection", False)
    )
    semantic_boundary_transport = bool(
        model_cfg.get("direct_recovery_semantic_witness_boundary_transport", False)
    )
    semantic_projection_fidelity = bool(
        model_cfg.get("direct_recovery_semantic_witness_projection_fidelity_weighting", False)
    )
    semantic_demand_normalized_fidelity = bool(
        model_cfg.get("direct_recovery_semantic_witness_demand_normalized_fidelity", False)
    )
    semantic_robust_occupancy = bool(
        model_cfg.get("direct_recovery_semantic_witness_robust_occupancy", False)
    )
    semantic_soft_occupancy_disagreement = bool(
        model_cfg.get("direct_recovery_semantic_witness_soft_occupancy_disagreement", False)
    )
    semantic_boundary_localized_occupancy_trust = bool(
        model_cfg.get("direct_recovery_semantic_witness_boundary_localized_occupancy_trust", False)
    )
    semantic_history_occupancy_reachability = bool(
        model_cfg.get("direct_recovery_semantic_witness_history_occupancy_reachability", False)
    )
    semantic_interaction_box_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_box_support", False)
    )
    semantic_interaction_hull_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_hull_support", False)
    )
    semantic_interaction_anchor_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_anchor_support", False)
    )
    semantic_interaction_response_support = bool(
        model_cfg.get("direct_recovery_semantic_witness_interaction_response_support", False)
    )
    semantic_feature_schema = (
        DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and _v48_74_signed_viability_enabled()
        and (semantic_interaction_anchor_support or semantic_interaction_response_support)
        else (DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and (semantic_interaction_anchor_support or semantic_interaction_response_support)
        else (DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and (semantic_interaction_box_support or semantic_interaction_hull_support)
        else (DIRECT_BOUNDARY_OCCUPANCY_REACHABILITY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and (semantic_boundary_localized_occupancy_trust or semantic_history_occupancy_reachability)
        else (DIRECT_OCCUPANCY_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and semantic_soft_occupancy_disagreement
        else (DIRECT_DEMAND_TEMPERED_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and semantic_demand_normalized_fidelity
        else (DIRECT_ROBUST_TRUST_RECOVERY_WITNESS_FEATURE_SCHEMA
        if semantic_witness_features_enabled and (semantic_projection_fidelity or semantic_robust_occupancy)
        else (DIRECT_PROJECTED_BOUNDARY_RECOVERY_WITNESS_FEATURE_SCHEMA
              if semantic_witness_features_enabled and (semantic_control_projection or semantic_boundary_transport)
              else (DIRECT_ACTIVE_CONSTRAINT_RECOVERY_WITNESS_FEATURE_SCHEMA
                    if semantic_witness_features_enabled and (semantic_route_alignment or semantic_reentry_alignment)
                    else (DIRECT_SEMANTIC_RECOVERY_WITNESS_FEATURE_SCHEMA if semantic_witness_features_enabled else 0)))))))))
    )
    # Schema 8 materializes both directional-box and empirical-hull
    # diagnostics in every sample.  The box/hull booleans only select the model
    # coordinate and therefore must not split the expensive tensor cache.
    # Keep this narrowly scoped: schema 7 H/J/K genuinely construct different
    # tensors. Schema 8 materializes both box/hull diagnostics; schema 9
    # materializes both anchored-ramp and observed-jerk diagnostics. Their arm
    # booleans only select the model coordinate and must not split the expensive
    # decoded tensor cache.
    cache_interaction_box_support = semantic_interaction_box_support
    cache_interaction_hull_support = semantic_interaction_hull_support
    cache_interaction_anchor_support = semantic_interaction_anchor_support
    cache_interaction_response_support = semantic_interaction_response_support
    if semantic_feature_schema == DIRECT_INTERACTION_ORIENTED_RECOVERY_WITNESS_FEATURE_SCHEMA:
        cache_interaction_box_support = True
        cache_interaction_hull_support = True
    elif semantic_feature_schema in {
        DIRECT_INTERACTION_RESPONSE_RECOVERY_WITNESS_FEATURE_SCHEMA,
        DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA,
    }:
        cache_interaction_box_support = True
        cache_interaction_hull_support = True
        cache_interaction_anchor_support = True
        cache_interaction_response_support = True
    payload = {
        "schema": _PERSISTENT_TENSOR_CACHE_SCHEMA,
        "manifests": _dataset_manifest_fingerprint(paths),
        "path_count": len(paths),
        "paths": [str(p.resolve()) for p in paths],
        "geometry": {
            "num_roots": int(num_roots), "num_options": int(num_options),
            "d_signature": int(d_signature), "d_future_signature": int(d_future_signature),
            "feature_dim": int(feature_dim),
        },
        # Only tensor-construction settings belong in this key.  Model-head,
        # optimizer, ROCT and option-execution settings do not change any
        # OCRAPSampleDataset tensor and previously forced redundant 90--250 s
        # decompression passes between witness/factor stages and ablation arms.
        "feature_layout": _feature_layout_values(cfg),
        "prefix_param_dim": int(cfg.get("prefix_param_dim", 5)),
        "bev_channels": int(cfg.get("bev_channels", 7)),
        "exact_eligibility": bool(training.get("direct_policy_metric_exact_eligibility", False)),
        "cphr_full_prefix_features": cphr_features_enabled,
        "cphr_feature_schema": (DIRECT_ABSOLUTE_PHYSICAL_HEADROOM_FEATURE_SCHEMA if cphr_features_enabled else 0),
        "erwf_option_resolved_features": erwf_features_enabled,
        "erwf_feature_schema": (DIRECT_EXECUTABLE_RECOVERY_WITNESS_FEATURE_SCHEMA if erwf_features_enabled else 0),
        "erwf_recovery_horizon_s": (float(cfg.get("recovery_horizon_s", 4.0)) if erwf_features_enabled else 0.0),
        "erwf_sample_rate_hz": (float(cfg.get("sample_rate_hz", 10.0)) if erwf_features_enabled else 0.0),
        "common_witness_option_resolved_features": common_witness_features_enabled,
        "common_witness_feature_schema": (DIRECT_COMMON_RECOVERY_WITNESS_FEATURE_SCHEMA if common_witness_features_enabled else 0),
        "common_witness_recovery_horizon_s": (float(cfg.get("recovery_horizon_s", 4.0)) if common_witness_features_enabled else 0.0),
        "common_witness_sample_rate_hz": (float(cfg.get("sample_rate_hz", 10.0)) if common_witness_features_enabled else 0.0),
        "semantic_witness_option_resolved_features": semantic_witness_features_enabled,
        "semantic_witness_feature_schema": semantic_feature_schema,
        "semantic_witness_v48_74_signed_viability": bool(
            semantic_witness_features_enabled and _v48_74_signed_viability_enabled()
        ),
        "semantic_witness_route_alignment": semantic_route_alignment,
        "semantic_witness_reentry_alignment": semantic_reentry_alignment,
        "semantic_witness_control_projection": semantic_control_projection,
        "semantic_witness_boundary_transport": semantic_boundary_transport,
        "semantic_witness_projection_fidelity_weighting": semantic_projection_fidelity,
        "semantic_witness_robust_occupancy": semantic_robust_occupancy,
        "semantic_witness_soft_occupancy_disagreement": semantic_soft_occupancy_disagreement,
        "semantic_witness_boundary_localized_occupancy_trust": semantic_boundary_localized_occupancy_trust,
        "semantic_witness_history_occupancy_reachability": semantic_history_occupancy_reachability,
        "semantic_witness_interaction_box_support": cache_interaction_box_support,
        "semantic_witness_interaction_hull_support": cache_interaction_hull_support,
        "semantic_witness_interaction_anchor_support": cache_interaction_anchor_support,
        "semantic_witness_interaction_response_support": cache_interaction_response_support,
        "semantic_witness_recovery_horizon_s": (float(cfg.get("recovery_horizon_s", 4.0)) if semantic_witness_features_enabled else 0.0),
        "semantic_witness_sample_rate_hz": (float(cfg.get("sample_rate_hz", 10.0)) if semantic_witness_features_enabled else 0.0),
        "npz_keys": sorted(MODEL_SAMPLE_NPZ_KEYS),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _stack_tensor_items(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        return {}
    keys = tuple(items[0].keys())
    if any(tuple(x.keys()) != keys for x in items):
        raise ValueError("persistent tensor cache item-key mismatch")
    return {k: torch.stack([x[k] for x in items], dim=0).contiguous() for k in keys}


def _build_items_ordered(build_item, num_items: int, workers: int) -> list[dict[str, torch.Tensor]]:
    """Build cache items concurrently without changing dataset order.

    ``Executor.map`` preserves input order.  The default remains one worker so
    historical workflows retain their resource profile; V48.72 explicitly opts
    into parallel decode/feature construction.
    """
    count = int(num_items)
    worker_count = max(1, int(workers))
    if count <= 0:
        return []
    if worker_count == 1:
        return [build_item(i) for i in range(count)]
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ocrap-cache") as pool:
        return list(pool.map(build_item, range(count)))


class OCRAPSampleDataset(Dataset):
    def __init__(self, paths: list[Path], cfg: dict | None = None):
        self.paths = list(paths)
        self.cfg = cfg or {}
        if not self.paths:
            raise ValueError("OCRAPSampleDataset requires at least one sample path")
        first = load_npz_selected(self.paths[0], MODEL_SAMPLE_NPZ_KEYS)
        first_K, first_L, first_sig, first_fsig = _geometry_from_sample(first)
        # Default to the paper/build geometry in config, but never shrink below
        # the first sample.  This makes the README's mixed proof/natural/stress
        # training command work without requiring per-run shape bookkeeping.
        self.num_roots = max(_target_dim_from_cfg(self.cfg, "num_roots", first_K or 1), first_K or 1)
        self.num_options = max(_target_dim_from_cfg(self.cfg, "num_recovery_options", first_L or 1), first_L or 1)
        self.d_signature = max(_model_target_dim(self.cfg, "d_signature", first_sig), first_sig)
        self.d_future_signature = max(_model_target_dim(self.cfg, "d_future_signature", first_fsig), first_fsig)
        self.feature_dim = int(sample_to_feature(first, self.cfg).shape[0])
        training_cfg = self.cfg.get("training", {}) if isinstance(self.cfg.get("training", {}), dict) else {}
        truth_policy = str(training_cfg.get("direct_value_absolute_feasibility_truth_contract", "legacy_full")).strip().lower()
        self.absolute_truth_contract_event: dict[str, Any] = {"policy": truth_policy, "enabled": False}
        self._absolute_truth_records: list[dict[str, Any]] | None = None
        if truth_policy in {"censor_structural_tail", "structural_interval_bounds", "switch_inverse_interval_bounds"}:
            truth_index_raw = str(training_cfg.get("direct_value_absolute_feasibility_truth_index", "") or "").strip()
            if not truth_index_raw:
                raise ValueError(f"{truth_policy} requires training.direct_value_absolute_feasibility_truth_index")
            truth_index_path = Path(truth_index_raw).expanduser().resolve()
            index = _load_absolute_truth_index(str(truth_index_path))
            records: list[dict[str, Any]] = []
            missing: list[str] = []
            invalid: list[str] = []
            for sample_path in self.paths:
                key = str(sample_path.resolve())
                rec = index.get(key)
                if rec is None:
                    missing.append(key)
                    continue
                if not bool(rec.get("valid", False)):
                    invalid.append(key)
                records.append(rec)
            if missing or invalid or len(records) != len(self.paths):
                raise ValueError(
                    "absolute truth index fail-closed: "
                    f"missing={len(missing)} invalid={len(invalid)} indexed={len(records)} expected={len(self.paths)}; "
                    f"examples_missing={missing[:3]} examples_invalid={invalid[:3]}"
                )
            self._absolute_truth_records = records
            physical = sum(bool(r.get("physical_identifiable", r.get("exact_physical", False))) for r in records)
            informative = sum(bool(r.get("informative", True)) for r in records)
            self.absolute_truth_contract_event = {
                "policy": truth_policy, "enabled": True, "index": str(truth_index_path),
                "rows": len(records), "physical_identifiable_rows": int(physical),
                "structurally_exposed_rows": int(len(records) - physical),
                "physical_identifiable_fraction": float(physical / max(len(records), 1)),
                "informative_interval_rows": int(informative),
                "informative_interval_fraction": float(informative / max(len(records), 1)),
                "max_r_dep_abs_error": float(max((float(r.get("r_dep_abs_error", 0.0)) for r in records), default=0.0)),
            }
        self.nominal_deviation = (
            _nominal_deviation_by_path(self.paths)
            if bool(training_cfg.get("direct_policy_metric_exact_eligibility", False))
            else [0.0] * len(self.paths)
        )
        # v48.45.6 engineering-only fast path.  A decoded model item is only a
        # few KB (flat feature + recovery labels), whereas the source NPZ can
        # contain large compressed map/BEV/debug arrays.  Caching the final CPU
        # tensors once removes repeated ZIP decompression across 8--20 epochs.
        # It is opt-in so non-v48 workflows keep their historical memory profile.
        self.persistent_tensor_cache = bool(training_cfg.get("persistent_tensor_cache", False))
        self.persistent_tensor_cache_build_workers = max(
            1, int(training_cfg.get("persistent_tensor_cache_build_workers", 1) or 1)
        )
        cache_dir_raw = str(training_cfg.get("persistent_tensor_cache_dir", "") or "").strip()
        self.persistent_tensor_cache_dir = Path(cache_dir_raw).expanduser() if cache_dir_raw else None
        self.cache_samples_in_memory = bool(training_cfg.get("cache_samples_in_memory", False)) or self.persistent_tensor_cache
        self._item_cache: list[dict[str, torch.Tensor]] | None = None
        self._stacked_item_cache: dict[str, torch.Tensor] | None = None
        self.tensor_cache_event: dict[str, Any] = {"enabled": self.persistent_tensor_cache, "hit": False}
        if self.persistent_tensor_cache:
            if self.persistent_tensor_cache_dir is None:
                raise ValueError("training.persistent_tensor_cache=true requires persistent_tensor_cache_dir")
            self._load_or_build_persistent_tensor_cache()
        elif self.cache_samples_in_memory:
            self._item_cache = [self._build_item(i) for i in range(len(self.paths))]

    def _load_or_build_persistent_tensor_cache(self) -> None:
        assert self.persistent_tensor_cache_dir is not None
        cache_dir = self.persistent_tensor_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _persistent_tensor_cache_key(
            self.paths, self.cfg, num_roots=self.num_roots, num_options=self.num_options,
            d_signature=self.d_signature, d_future_signature=self.d_future_signature,
            feature_dim=self.feature_dim,
        )
        cache_path = cache_dir / f"ocrap_tensor_items_{key}.pt"
        lock_path = cache_dir / f"ocrap_tensor_items_{key}.lock"
        t0 = time.perf_counter()
        lock_wait_start = time.perf_counter()
        with lock_path.open("a+b") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            lock_wait_seconds = float(time.perf_counter() - lock_wait_start)
            try:
                if cache_path.is_file():
                    payload = _load_persistent_tensor_cache_payload(cache_path)
                    if (
                        isinstance(payload, dict)
                        and int(payload.get("schema", -1)) == _PERSISTENT_TENSOR_CACHE_SCHEMA
                        and payload.get("key") == key
                        and int(payload.get("num_items", -1)) == len(self.paths)
                        and isinstance(payload.get("tensors"), dict)
                    ):
                        tensors = payload["tensors"]
                        if tensors and all(int(v.shape[0]) == len(self.paths) for v in tensors.values()):
                            self._stacked_item_cache = tensors
                            self.tensor_cache_event = {
                                "enabled": True, "hit": True, "key": key, "path": str(cache_path),
                                "seconds": float(time.perf_counter() - t0),
                                "lock_wait_seconds": lock_wait_seconds,
                                "build_seconds": 0.0,
                                "build_workers": self.persistent_tensor_cache_build_workers,
                            }
                            return
                    # Corrupt/stale same-name artifact should never be trusted.
                    cache_path.unlink(missing_ok=True)
                build_start = time.perf_counter()
                items = _build_items_ordered(
                    self._build_item, len(self.paths), self.persistent_tensor_cache_build_workers
                )
                tensors = _stack_tensor_items(items)
                build_seconds = float(time.perf_counter() - build_start)
                payload = {
                    "schema": _PERSISTENT_TENSOR_CACHE_SCHEMA, "key": key,
                    "num_items": len(self.paths), "tensors": tensors,
                }
                tmp = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
                torch.save(payload, tmp)
                os.replace(tmp, cache_path)
                self._stacked_item_cache = tensors
                self.tensor_cache_event = {
                    "enabled": True, "hit": False, "key": key, "path": str(cache_path),
                    "seconds": float(time.perf_counter() - t0),
                    "lock_wait_seconds": lock_wait_seconds,
                    "build_seconds": build_seconds,
                    "build_workers": self.persistent_tensor_cache_build_workers,
                }
            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

    def cached_tensor_bytes(self) -> int:
        if self._stacked_item_cache is not None:
            return int(sum(v.numel() * v.element_size() for v in self._stacked_item_cache.values()))
        if self._item_cache is not None:
            return int(sum(t.numel() * t.element_size() for item in self._item_cache for t in item.values()))
        return 0

    def __len__(self) -> int:
        return len(self.paths)

    def _build_item(self, idx: int) -> dict[str, torch.Tensor]:
        d = load_npz_selected(self.paths[idx], MODEL_SAMPLE_NPZ_KEYS)
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
            "nominal_deviation": torch.tensor(float(self.nominal_deviation[idx]), dtype=torch.float32),
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
        model_cfg = self.cfg.get("model", {}) if isinstance(self.cfg.get("model", {}), dict) else {}
        if bool(model_cfg.get("direct_recovery_absolute_physical_headroom_correction", False)):
            out["direct_absolute_physical_headroom_features"] = torch.from_numpy(
                direct_absolute_physical_headroom_features_from_sample(d, self.cfg)
            )
        if bool(model_cfg.get("direct_recovery_absolute_executable_witness_correction", False)):
            out["direct_absolute_executable_witness_features"] = torch.from_numpy(
                direct_executable_recovery_witness_features_from_sample(
                    d, self.cfg, num_options=self.num_options
                )
            )
        if bool(
            model_cfg.get("direct_recovery_absolute_common_witness_correction", False)
            or model_cfg.get("direct_recovery_absolute_quantifier_witness_correction", False)
        ):
            out["direct_absolute_common_witness_features"] = torch.from_numpy(
                direct_common_recovery_witness_features_from_sample(
                    d, self.cfg, num_options=self.num_options
                )
            )
        if bool(model_cfg.get("direct_recovery_absolute_semantic_witness_correction", False)):
            out["direct_absolute_semantic_witness_features"] = torch.from_numpy(
                direct_semantic_recovery_witness_features_from_sample(
                    d, self.cfg, num_options=self.num_options
                )
            )
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._stacked_item_cache is not None:
            out = {k: v[idx] for k, v in self._stacked_item_cache.items()}
        elif self._item_cache is not None:
            out = dict(self._item_cache[idx])
        else:
            out = self._build_item(idx)
        if self._absolute_truth_records is not None:
            rec = self._absolute_truth_records[idx]
            out["absolute_truth_physical_identifiable"] = torch.tensor(
                float(bool(rec.get("physical_identifiable", False))), dtype=torch.float32
            )
            out["absolute_truth_structural_exposure"] = torch.tensor(
                float(rec.get("structural_exposure_mass", 0.0)), dtype=torch.float32
            )
            out["absolute_truth_physical_lower"] = torch.tensor(
                float(rec.get("physical_lower", rec.get("r_dep_stored", -1.0e6))), dtype=torch.float32
            )
            out["absolute_truth_physical_upper"] = torch.tensor(
                float(rec.get("physical_upper", rec.get("r_dep_stored", 1.0e6))), dtype=torch.float32
            )
            out["absolute_truth_interval_informative"] = torch.tensor(
                float(bool(rec.get("informative", rec.get("physical_identifiable", False)))), dtype=torch.float32
            )
        return out


def split_paths_by_npz_split(paths: list[Path], split: str | set[str]) -> list[Path]:
    requested = {split} if isinstance(split, str) else set(split)
    if "all" in requested:
        return list(paths)
    allowed = expand_split_roles(requested)
    keep: list[Path] = []
    for p in paths:
        sid = str(scalar_metadata_for_path(p, "split_id", ""))
        if sid in allowed:
            keep.append(p)
    return keep
