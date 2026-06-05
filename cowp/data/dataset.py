from __future__ import annotations

from pathlib import Path

import numpy as np


def _restore_key(key: str) -> str:
    return key.replace("__", "/")


class COWPNpzDataset:
    def __init__(self, cache_dir: str | Path, pattern: str = "*.npz"):
        self.cache_dir = Path(cache_dir)
        self.paths = sorted(self.cache_dir.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No cache npz files found in {self.cache_dir} matching {pattern}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = np.load(self.paths[idx], allow_pickle=True)
        return {_restore_key(k): data[k] for k in data.files}


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
    for k in keys:
        vals = [x[k] for x in batch]
        try:
            out[k] = torch.stack(vals, dim=0)
        except Exception:
            pass
    return out
