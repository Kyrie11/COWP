from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _restore_key(key: str) -> str:
    return key.replace("__", "/")


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


def _wanted_keys_for_stage(stage: str | None) -> set[str] | None:
    if stage is None or stage == "all":
        return None
    always_prefixes = (
        "womd/state/",
        "state/",
        "map/",
        "cowp/critical/",
        "waymax/",
    )
    wanted: set[str] = set()
    # prefixes are represented by ending slash markers in this helper
    for p in always_prefixes:
        wanted.add(p)
    if stage in ("representation", "natural"):
        wanted.add("cowp/natural/")
    elif stage == "response":
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/response/",
        })
    elif stage == "witness":
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/witness/",
        })
    elif stage == "planner":
        wanted.update({
            "cowp/candidates/trajectory",
            "cowp/candidates/macro_type",
            "cowp/candidates/valid",
            "cowp/candidates/conventional_safe",
            "cowp/candidates/false_safe",
            "cowp/candidates/noncoercive_feasible",
            "cowp/candidates/ego_utility_prior",
            "cowp/candidates/is_logged",
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

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        with np.load(self.paths[idx], allow_pickle=True) as data:
            out = {_restore_key(k): data[k] for k in data.files}
        align_critical_agents_to_womd_input(out)
        return mask_out_of_range_critical_agents(out)


class TorchCOWPDataset:
    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz", stage: str | None = None):
        self.base = COWPNpzDataset(cache_dir, pattern)
        self.stage = stage
        self._wanted = _wanted_keys_for_stage(stage)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        import torch

        d = self.base[idx]
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

    out = {}
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
