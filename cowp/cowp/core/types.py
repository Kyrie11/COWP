from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Lane:
    lane_id: int
    polyline: np.ndarray
    speed_limit_mps: float = 13.9
    turn_direction: int = 0
    entry_lanes: tuple[int, ...] = ()
    exit_lanes: tuple[int, ...] = ()
    left_neighbors: tuple[int, ...] = ()
    right_neighbors: tuple[int, ...] = ()
    controlled_by_stop: bool = False
    controlled_by_signal: bool = False
    lane_type: int = 0

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.polyline, dtype=np.float32)[..., :2]


@dataclass
class MapData:
    lanes: dict[int, Lane] = field(default_factory=dict)
    road_lines: dict[int, np.ndarray] = field(default_factory=dict)
    road_edges: dict[int, np.ndarray] = field(default_factory=dict)
    crosswalks: dict[int, np.ndarray] = field(default_factory=dict)
    speed_bumps: dict[int, np.ndarray] = field(default_factory=dict)
    stop_signs: dict[int, dict[str, Any]] = field(default_factory=dict)
    dynamic_signals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ScenarioData:
    scenario_id: str
    timestamps: np.ndarray
    current_time_index: int
    states: np.ndarray  # [N,T,11] x,y,z,vx,vy,speed,heading,l,w,h,valid
    object_type: np.ndarray
    track_id: np.ndarray
    sdc_track_index: int
    objects_of_interest: np.ndarray
    tracks_to_predict: np.ndarray
    map_data: MapData
    raw: Any = None

    @property
    def num_agents(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[1])

    @property
    def future_slice(self) -> slice:
        return slice(self.current_time_index + 1, None)

    @property
    def history_slice(self) -> slice:
        return slice(0, self.current_time_index + 1)

    @property
    def ego_current(self) -> np.ndarray:
        return self.states[self.sdc_track_index, self.current_time_index]

    @property
    def ego_future(self) -> np.ndarray:
        return self.states[self.sdc_track_index, self.future_slice]


def ensure_trajectory_7(traj: np.ndarray, default_length: float = 4.8, default_width: float = 1.9) -> np.ndarray:
    """Return [T,7] x,y,heading,vx,vy,length,width from either [T,7] or [T,>=11]."""
    arr = np.asarray(traj, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"trajectory must be [T,D], got shape {arr.shape}")
    if arr.shape[1] == 7:
        return arr.astype(np.float32)
    if arr.shape[1] >= 11:
        out = np.zeros((arr.shape[0], 7), dtype=np.float32)
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        out[:, 2] = arr[:, 6]
        out[:, 3] = arr[:, 3]
        out[:, 4] = arr[:, 4]
        out[:, 5] = np.where(arr[:, 7] > 0, arr[:, 7], default_length)
        out[:, 6] = np.where(arr[:, 8] > 0, arr[:, 8], default_width)
        return out
    if arr.shape[1] >= 5:
        out = np.zeros((arr.shape[0], 7), dtype=np.float32)
        out[:, :5] = arr[:, :5]
        out[:, 5] = default_length
        out[:, 6] = default_width
        return out
    raise ValueError(f"trajectory must have at least 5 columns, got {arr.shape}")


def future_states_to_traj7(future_states: np.ndarray, horizon: int, current_state: np.ndarray | None = None) -> np.ndarray:
    """Convert WOMD future states to a fixed-horizon [T,7] trajectory.

    WOMD tracks may contain invalid future rows.  The previous implementation
    often passed the full [T,11] state block directly to ``ensure_trajectory_7``;
    invalid rows are usually zero-filled, so downstream collision, burden and
    critical-agent logic could see artificial jumps to the origin.  This helper
    keeps time alignment, replaces invalid rows by the nearest valid state, and
    pads/truncates to ``horizon``.
    """
    horizon = int(horizon)
    if horizon <= 0:
        return np.zeros((0, 7), dtype=np.float32)

    arr = np.asarray(future_states, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        if current_state is not None:
            base = ensure_trajectory_7(np.asarray(current_state, dtype=np.float32)[None, :])[0]
        else:
            base = np.zeros(7, dtype=np.float32)
            base[5:7] = (4.8, 1.9)
        return np.repeat(base[None, :], horizon, axis=0).astype(np.float32)

    traj = ensure_trajectory_7(arr)
    valid = arr[:, 10] > 0.5 if arr.shape[1] >= 11 else np.ones(arr.shape[0], dtype=bool)
    if not np.any(valid):
        if current_state is not None:
            base = ensure_trajectory_7(np.asarray(current_state, dtype=np.float32)[None, :])[0]
        else:
            base = traj[0]
        out = np.repeat(base[None, :], min(horizon, max(len(traj), 1)), axis=0)
    else:
        out = traj.copy()
        first_valid = int(np.where(valid)[0][0])
        out[:first_valid] = out[first_valid]
        last = out[first_valid].copy()
        for t in range(first_valid, len(out)):
            if valid[t]:
                last = out[t].copy()
            else:
                out[t] = last

    if len(out) < horizon:
        out = np.concatenate([out, np.repeat(out[-1:len(out)], horizon - len(out), axis=0)], axis=0)
    return out[:horizon].astype(np.float32)


def pad_array(arr: np.ndarray, shape: tuple[int, ...], pad_value: float | int | bool = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr)
    out = np.full(shape, pad_value, dtype=arr.dtype if arr.size else np.asarray(pad_value).dtype)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, shape))
    out[slices] = arr[slices]
    mask_shape = shape[:-1] if arr.ndim == len(shape) and len(shape) > 1 else shape
    mask = np.zeros(mask_shape, dtype=bool)
    valid_slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape[: len(mask_shape)], mask_shape))
    mask[valid_slices] = True
    return out, mask
