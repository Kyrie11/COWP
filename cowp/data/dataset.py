from __future__ import annotations

from pathlib import Path

import numpy as np


def _restore_key(key: str) -> str:
    return key.replace("__", "/")


def _infer_agent_count(data: dict[str, np.ndarray]) -> int | None:
    """Infer the number of model-visible WOMD agents in one cache item.

    Proto labels may contain original Scenario track indices, while WOMD
    tf.Example tensors expose a fixed padded agent dimension.  Training must not
    gather an agent embedding for a track index that is not present in those
    tensors.
    """
    for key in (
        "state/history",
        "womd/state/history",
        "state/current/x",
        "womd/state/current/x",
        "state/id",
        "womd/state/id",
        "state/past/x",
        "womd/state/past/x",
    ):
        arr = data.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 0:
            continue
        if key.endswith("history") and arr.ndim >= 2:
            return int(arr.shape[0])
        # Per-sample WOMD fields are normally [N], [N,1] or [N,T].
        return int(arr.shape[0])
    return None


def mask_out_of_range_critical_agents(data: dict[str, np.ndarray], num_agents: int | None = None) -> dict[str, np.ndarray]:
    """Mask critical slots whose Scenario track index is invisible to model input.

    This fixes crashes such as ``index 149 is out of bounds for dimension 1 with
    size 128`` and, more importantly, prevents losses from supervising a
    candidate-agent pair whose agent state was not loaded from the tensor cache.
    The original label tensors are not mutated by callers unless they pass a dict
    they intend to modify.
    """
    if num_agents is None:
        num_agents = _infer_agent_count(data)
    if num_agents is None or "cowp/critical/track_index" not in data:
        return data

    idx = np.asarray(data["cowp/critical/track_index"]).astype(np.int64, copy=False)
    if idx.ndim != 1:
        return data
    visible = (idx >= 0) & (idx < int(num_agents))
    old_valid = np.asarray(data.get("cowp/critical/valid", np.ones_like(visible, dtype=bool))).astype(bool, copy=False)
    new_valid = old_valid & visible
    if np.array_equal(new_valid, old_valid):
        data["cowp/critical/input_visible"] = visible.astype(bool)
        return data

    data["cowp/critical/input_visible"] = visible.astype(bool)
    data["cowp/critical/valid"] = new_valid.astype(bool)
    bad = ~new_valid
    if not bad.any():
        return data

    # Natural labels are [A,...].
    for key in list(data.keys()):
        if key.startswith("cowp/natural/") and np.asarray(data[key]).ndim >= 1 and np.asarray(data[key]).shape[0] == len(idx):
            arr = np.array(data[key], copy=True)
            if key.endswith("/valid") or key.endswith("/priority_preserved"):
                arr[bad] = False
            elif key.endswith("/weight") or key.endswith("/burden_neutral") or key.endswith("/beta"):
                arr[bad] = 0
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
        return mask_out_of_range_critical_agents(out)


class TorchCOWPDataset:
    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz"):
        self.base = COWPNpzDataset(cache_dir, pattern)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        import torch

        d = self.base[idx]
        out = {}
        for k, v in d.items():
            arr = np.asarray(v)
            if arr.dtype.kind in "fiu" or arr.dtype == np.bool_:
                out[k] = torch.from_numpy(arr.astype(np.float32) if arr.dtype == np.float64 else arr)
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
