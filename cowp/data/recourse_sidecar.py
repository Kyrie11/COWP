from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RecourseSidecarDataset:
    """Small independent NPZ sidecar used only by the V16.8.45 recourse operator.

    It never modifies the compact-5k base tensor cache.  Every item is one
    ``(scenario, ego-hypothesis, exact actor, natural root)`` context and contains
    an outcome-independent verified proposal pool plus causal context features.
    """

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = str(split)
        self.paths = sorted((self.root / self.split).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No RCRSO sidecar files in {self.root / self.split}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        with np.load(self.paths[int(idx)], allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def collate_recourse_sidecar(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not batch:
        return {}
    out: dict[str, Any] = {}
    scalar_keys = {
        "root_mass", "root_source", "fixed_verified_nonempty", "analytic_verified_nonempty",
        "scenario_hash", "hypothesis_id", "hypothesis_group_id", "candidate_index", "agent_index", "root_index",
    }
    stack_keys = {
        "root_tokens", "ego_tokens", "environment_tokens", "environment_valid",
        "blocker_state", "conflict_features", "target_control_knots", "target_valid",
        "target_burden", "target_source", "negative_control_knots", "negative_valid", "negative_reason",
    }
    for key in stack_keys | scalar_keys:
        if key not in batch[0]:
            continue
        vals = [np.asarray(x[key]) for x in batch]
        arr = np.stack(vals, axis=0)
        if arr.dtype.kind in "iu":
            out[key] = torch.from_numpy(arr.astype(np.int64, copy=False))
        elif arr.dtype == np.bool_:
            out[key] = torch.from_numpy(arr.astype(np.bool_, copy=False))
        else:
            out[key] = torch.from_numpy(arr.astype(np.float32, copy=False))
    return out
