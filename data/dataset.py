from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _restore_key(key: str) -> str:
    return key.replace("__", "/")


def _canonicalize_state_aliases(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normalize WOMD tf.Example state keys to the model-facing ``state/*`` namespace.

    Tensor caches produced by this project store raw tf.Example tensors as
    ``womd/state/...`` while some older/debug caches store already-shaped tensors
    as ``state/...``. Mixing those files in one DataLoader batch is dangerous if
    collation intersects keys: the effective state input can disappear. We
    canonicalize to ``state/...`` immediately after NPZ loading so equivalent
    caches expose the same keys without duplicating tensors.
    """
    for key in list(data.keys()):
        if not key.startswith("womd/state/"):
            continue
        canon = "state/" + key[len("womd/state/") :]
        if canon not in data:
            data[canon] = data[key]
        del data[key]
    return data


def _has_model_state(data: dict[str, np.ndarray]) -> bool:
    """Whether a sample contains enough encoder input for the COWP graph."""
    if "state/history" in data or "state/all" in data:
        return True
    return _first_array(data, ("state/past/x",)) is not None and _first_array(data, ("state/current/x",)) is not None


def _missing_required_for_stage(data: dict[str, np.ndarray], stage: str | None) -> list[str]:
    """Return human-readable missing fields for a training stage.

    Invalid/partial cache files should be skipped before batching. Otherwise one
    incomplete item can remove required tensors from the whole batch and cause a
    late ``KeyError`` inside the compiled model.
    """
    stage = stage or "all"
    missing: list[str] = []
    if not _has_model_state(data):
        missing.append("state/history or state/all or state/{past,current}/x")
    for key in ("cowp/critical/track_index", "cowp/critical/valid"):
        if key not in data:
            missing.append(key)
    if stage in ("representation", "natural", "all"):
        for key in (
            "cowp/natural/traj",
            "cowp/natural/valid",
            "cowp/natural/weight",
            "cowp/natural/source",
            "cowp/natural/priority_preserved",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("response", "witness", "planner", "planner_eval", "all"):
        for key in ("cowp/candidates/trajectory", "cowp/candidates/macro_type", "cowp/candidates/valid"):
            if key not in data:
                missing.append(key)
    if stage in ("response", "all"):
        for key in (
            "cowp/response/valid",
            "cowp/response/is_safe",
            "cowp/response/is_low_burden",
            "cowp/response/burden_total",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("witness", "planner", "planner_eval", "all"):
        for key in (
            "cowp/witness/exists",
            "cowp/witness/token",
            "cowp/witness/burden_total",
            "cowp/witness/conflict_interval",
            "cowp/witness/opr",
            "cowp/witness/c_i",
        ):
            if key not in data:
                missing.append(key)
    if stage in ("planner", "planner_eval", "all"):
        for key in ("cowp/candidates/noncoercive_feasible", "cowp/candidates/false_safe"):
            if key not in data:
                missing.append(key)
    return missing


def _first_array(data: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        arr = data.get(name)
        if arr is not None:
            return np.asarray(arr)
    return None


def _infer_agent_count(data: dict[str, np.ndarray]) -> int | None:
    """Infer the number of model-visible WOMD agents in one cache item.

    Tensor-cache files may store temporal WOMD fields flattened as [N*T].  Agent
    count must therefore be inferred primarily from current/id fields ([N]), not
    from past arrays ([N*T]).
    """
    hist = _first_array(data, ("state/history", "womd/state/history"))
    if hist is not None and hist.ndim >= 2:
        return int(hist.shape[0])

    for key in (
        "state/current/x",
        "womd/state/current/x",
        "state/current/valid",
        "womd/state/current/valid",
        "state/id",
        "womd/state/id",
        "state/agent_valid",
        "womd/state/agent_valid",
        "state/is_sdc",
        "womd/state/is_sdc",
    ):
        arr = data.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 0:
            continue
        return int(arr.shape[0])

    # Last-resort temporal-only inference.  WOMD past has 10 steps; if the flat
    # size is divisible by 10, recover N instead of returning N*T.
    for key in ("state/past/x", "womd/state/past/x"):
        arr = data.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim >= 2:
            return int(arr.shape[0])
        if arr.ndim == 1 and arr.size % 10 == 0:
            return int(arr.size // 10)
        if arr.ndim == 1:
            return int(arr.size)
    return None


def _infer_agent_visible_mask(data: dict[str, np.ndarray], num_agents: int | None = None) -> np.ndarray | None:
    """Infer which agent rows are visible to the model encoder.

    This mirrors ``build_agent_history_from_womd`` as closely as possible without
    importing torch: current-valid is preferred; otherwise any known agent row is
    considered visible.  The SDC row is always kept visible.
    """
    if num_agents is None:
        num_agents = _infer_agent_count(data)
    if num_agents is None or num_agents <= 0:
        return None
    visible = np.ones(int(num_agents), dtype=bool)
    cur_valid = _first_array(data, ("state/current/valid", "womd/state/current/valid", "state/agent_valid", "womd/state/agent_valid"))
    if cur_valid is not None and cur_valid.size >= num_agents:
        visible = np.asarray(cur_valid).reshape(-1)[:num_agents].astype(float) > 0.5
    is_sdc = _first_array(data, ("state/is_sdc", "womd/state/is_sdc"))
    if is_sdc is not None and is_sdc.size >= num_agents:
        visible = visible | (np.asarray(is_sdc).reshape(-1)[:num_agents].astype(float) > 0.5)
    elif num_agents > 0:
        visible[0] = True
    return visible


def _critical_gather_index(data: dict[str, np.ndarray]) -> tuple[str, np.ndarray] | tuple[None, None]:
    """Return the agent-row index used by the model.

    ``cowp/critical/track_index`` is the original Scenario proto track index.
    In real WOMD tf.Example tensors the model-visible agent row should be found
    by matching the object id.  New tensor caches therefore store
    ``cowp/critical/input_index`` and the model/dataset should prefer it.
    """
    if "cowp/critical/input_index" in data:
        return "cowp/critical/input_index", np.asarray(data["cowp/critical/input_index"]).astype(np.int64, copy=False)
    if "cowp/critical/track_index" in data:
        return "cowp/critical/track_index", np.asarray(data["cowp/critical/track_index"]).astype(np.int64, copy=False)
    return None, None


def align_critical_agents_to_womd_input(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map Scenario critical tracks to WOMD tf.Example/model input rows.

    Label construction is performed from Scenario protos, where critical agents
    are naturally indexed by Scenario track index.  The training model reads raw
    WOMD tf.Example state tensors whose first dimension is an input row and may
    be capped/reordered.  Matching by object id avoids supervising/gathering the
    wrong agent.  This function is intentionally conservative: if object ids are
    unavailable it falls back to the legacy track-index convention and runtime
    masks still prevent crashes.
    """
    track_index = data.get("cowp/critical/track_index")
    if track_index is None:
        return data
    track_index = np.asarray(track_index).astype(np.int64, copy=False).reshape(-1)
    # Preserve an authoritative cache-time WOMD/model-row mapping when present.
    # Replacing it with Scenario track_index merely because object ids are absent
    # can silently point legacy/import paths at the wrong actor.
    existing_input = data.get("cowp/critical/input_index")
    if existing_input is not None and np.asarray(existing_input).size == track_index.size:
        input_index = np.asarray(existing_input).astype(np.int64, copy=True).reshape(-1)
    else:
        input_index = np.array(track_index, dtype=np.int64, copy=True)
    mapped_by_id = np.zeros_like(track_index, dtype=bool)

    track_id = data.get("cowp/critical/track_id")
    state_id = _first_array(data, ("state/id", "womd/state/id"))
    n = _infer_agent_count(data)
    if track_id is not None and state_id is not None:
        ids = np.asarray(state_id).reshape(-1)
        if n is not None:
            ids = ids[: int(n)]
        id_to_row: dict[int, int] = {}
        visible = _infer_agent_visible_mask(data, len(ids))
        for row, tid in enumerate(ids):
            try:
                tid_int = int(tid)
            except Exception:
                continue
            # Ignore repeated padded ids when the corresponding row is invalid.
            if visible is not None and row < len(visible) and not bool(visible[row]) and tid_int in id_to_row:
                continue
            if tid_int not in id_to_row:
                id_to_row[tid_int] = row
        tids = np.asarray(track_id).reshape(-1)
        if tids.size == track_index.size:
            input_index = np.full_like(track_index, -1, dtype=np.int64)
            for a, tid in enumerate(tids):
                try:
                    row = id_to_row.get(int(tid), -1)
                except Exception:
                    row = -1
                input_index[a] = int(row)
                mapped_by_id[a] = row >= 0
            data["cowp/critical/track_index_original"] = np.array(track_index, dtype=np.int64, copy=True)
            data["cowp/critical/mapped_by_id"] = mapped_by_id.astype(bool)

    data["cowp/critical/input_index"] = input_index.astype(np.int64)
    return data


def mask_out_of_range_critical_agents(data: dict[str, np.ndarray], num_agents: int | None = None) -> dict[str, np.ndarray]:
    """Mask critical slots whose model input row is invisible.

    This prevents both hard crashes (e.g. gather index 149 for a 128-agent tensor)
    and silent wrong supervision.  New caches should first call
    ``align_critical_agents_to_womd_input`` so the check is performed on
    ``cowp/critical/input_index`` rather than the original Scenario track index.
    """
    if num_agents is None:
        num_agents = _infer_agent_count(data)
    idx_key, idx = _critical_gather_index(data)
    if num_agents is None or idx is None:
        return data

    idx = np.asarray(idx).astype(np.int64, copy=False)
    if idx.ndim != 1:
        return data
    in_range = (idx >= 0) & (idx < int(num_agents))
    agent_visible = _infer_agent_visible_mask(data, num_agents)
    visible = in_range.copy()
    if agent_visible is not None and len(agent_visible) > 0:
        safe_idx = np.clip(idx, 0, len(agent_visible) - 1)
        visible = visible & agent_visible[safe_idx]
    old_valid = np.asarray(data.get("cowp/critical/valid", np.ones_like(visible, dtype=bool))).astype(bool, copy=False)
    new_valid = old_valid & visible
    data["cowp/critical/input_visible"] = visible.astype(bool)
    if np.array_equal(new_valid, old_valid):
        return data

    data["cowp/critical/valid"] = new_valid.astype(bool)
    bad = ~new_valid
    if not bad.any():
        return data

    # Natural labels are [A,...].
    for key in list(data.keys()):
        arr0 = np.asarray(data[key])
        if key.startswith("cowp/natural/") and arr0.ndim >= 1 and arr0.shape[0] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/valid") or key.endswith("/priority_preserved"):
                arr[bad] = False
            else:
                arr[bad] = 0
            data[key] = arr

    # Response / witness pair labels are [K,A,...].
    for key in list(data.keys()):
        arr0 = np.asarray(data[key])
        if key.startswith("cowp/response/") and arr0.ndim >= 2 and arr0.shape[1] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/valid") or key.endswith("/is_safe") or key.endswith("/is_low_burden"):
                arr[:, bad] = False
            else:
                arr[:, bad] = 0
            data[key] = arr
        elif key.startswith("cowp/witness/") and arr0.ndim >= 2 and arr0.shape[1] == len(idx):
            arr = np.array(arr0, copy=True)
            if key.endswith("/exists"):
                arr[:, bad] = False
            elif key.endswith("/opr"):
                arr[:, bad] = 1.0
            else:
                arr[:, bad] = 0
            data[key] = arr
        elif key in {"cowp/witness/critical_agent_track_index"} and arr0.ndim == 1 and arr0.shape[0] == len(idx):
            arr = np.array(arr0, copy=True)
            arr[bad] = -1
            data[key] = arr
    return data


def _wanted_keys_for_stage(
    stage: str | None,
    *,
    include_response_traj: bool = True,
    include_response_components: bool = True,
    include_waymax_outcomes: bool = False,
) -> set[str] | None:
    if stage is None or stage == "all":
        return None
    always_prefixes = (
        "womd/state/",
        "state/",
        "cowp/critical/",
    )
    wanted: set[str] = set()
    # prefixes are represented by ending slash markers in this helper
    for p in always_prefixes:
        wanted.add(p)
    # The model only consumes these two map tensors.  Loading broad ``map/`` or
    # ``waymax/`` prefixes in Stage A can read large arrays that the loss never
    # uses, slowing every batch without changing the objective.
    wanted.update({"map/conflict_regions", "map/conflict_region_valid"})
    if stage in ("representation", "natural"):
        wanted.add("cowp/natural/")
    elif stage == "response":
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/response/valid",
            "cowp/response/is_safe",
            "cowp/response/is_low_burden",
            "cowp/response/burden_total",
        })
        if include_response_components:
            wanted.add("cowp/response/burden_components")
        if include_response_traj:
            wanted.add("cowp/response/traj")
    elif stage == "witness":
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/witness/",
        })
    elif stage in ("planner", "planner_eval"):
        # Training may use optional Waymax candidate-outcome labels, but broad
        # waymax/* loading can pull large tensors into every batch.  Load only
        # the three scalar candidate outcome arrays used by planner_outcome_loss,
        # and only when explicitly requested.
        if stage == "planner" and include_waymax_outcomes:
            wanted.update({
                "waymax/candidate_rollout_valid",
                "waymax/candidate_collision",
                "waymax/candidate_offroad",
                "waymax/candidate_log_divergence",
            })
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/candidates/conventional_safe",
            "cowp/candidates/false_safe",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/is_logged",
            "cowp/natural/beta",
            "cowp/witness/",
        })
    return wanted


def _key_allowed(key: str, wanted: set[str] | None) -> bool:
    if wanted is None:
        return True
    for item in wanted:
        if item.endswith("/") and key.startswith(item):
            return True
        if key == item:
            return True
    return False


class COWPNpzDataset:
    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz"):
        self.cache_dir = Path(cache_dir)
        self.paths = sorted(self.cache_dir.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No cache npz files found in {self.cache_dir} matching {pattern}")

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, idx: int, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
        with np.load(self.paths[idx], allow_pickle=True) as data:
            out: dict[str, np.ndarray] = {}
            for raw_key in data.files:
                key = _restore_key(raw_key)
                if _key_allowed(key, wanted):
                    out[key] = data[raw_key]
        _canonicalize_state_aliases(out)
        align_critical_agents_to_womd_input(out)
        return mask_out_of_range_critical_agents(out)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.load(idx, None)


class TorchCOWPDataset:
    def __init__(
        self,
        cache_dir: str | Path,
        pattern: str = "*.npz",
        stage: str | None = None,
        *,
        skip_invalid: bool = True,
        include_response_traj: bool = True,
        include_response_components: bool = True,
        include_waymax_outcomes: bool = False,
    ):
        self.base = COWPNpzDataset(cache_dir, pattern)
        self.stage = stage
        self._wanted = _wanted_keys_for_stage(
            stage,
            include_response_traj=include_response_traj,
            include_response_components=include_response_components,
            include_waymax_outcomes=include_waymax_outcomes,
        )
        self.skip_invalid = bool(skip_invalid)

    def __len__(self) -> int:
        return len(self.base)

    def _load_valid_np(self, idx: int) -> dict[str, np.ndarray]:
        last_missing: list[str] = []
        last_path: Path | None = None
        n = len(self.base)
        for off in range(n if self.skip_invalid else 1):
            j = (int(idx) + off) % n
            d = self.base.load(j, self._wanted)
            missing = _missing_required_for_stage(d, self.stage)
            if not missing:
                return d
            last_missing = missing
            last_path = self.base.paths[j]
            if not self.skip_invalid:
                break
        raise KeyError(f"No valid COWP sample found for stage={self.stage!r}; last_path={last_path}; missing={last_missing}")

    def __getitem__(self, idx: int):
        import torch

        d = self._load_valid_np(idx)
        out = {}
        for k, v in d.items():
            if not _key_allowed(k, self._wanted):
                continue
            arr = np.asarray(v)
            if arr.dtype.kind in "fiu" or arr.dtype == np.bool_:
                # np.load may return non-writable views; copying avoids rare pin-memory
                # and from_numpy warnings while keeping per-stage I/O bounded.
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                elif not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                out[k] = torch.from_numpy(arr)
        return out


def collate_torch(batch):
    import torch

    batch = [x for x in batch if isinstance(x, dict) and x]
    if not batch:
        return {}
    out = {}
    # Dataset-level validation makes stage-required tensors present for every
    # item. Intersecting keys is therefore safe and avoids fabricating labels.
    keys = set.intersection(*(set(x.keys()) for x in batch))
    optional_prefixes = ("scenario/", "dataset/", "womd/scenario/", "womd/roadgraph/", "roadgraph/")
    for k in keys:
        vals = [x[k] for x in batch]
        try:
            out[k] = torch.stack(vals, dim=0)
        except Exception as exc:
            if k.startswith(optional_prefixes):
                continue
            shapes = [tuple(v.shape) for v in vals if torch.is_tensor(v)]
            raise RuntimeError(f"Failed to collate tensor key {k!r}; shapes={shapes}") from exc
    return out
