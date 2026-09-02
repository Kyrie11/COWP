"""External baseline adapters for OC-RAP.

Keep package import lightweight.  Provenance/index/audit utilities import
``ocrap.external_baselines.provenance`` and must not pull in PyTorch/NumPy just
because the package ``__init__`` is executed.  The heavy train/evaluate modules
are therefore imported lazily only when their public callables are requested.
"""

from __future__ import annotations

from typing import Any

__all__ = ["train_external_baseline", "evaluate_external_baselines"]


def __getattr__(name: str) -> Any:
    if name == "train_external_baseline":
        from .train import train_external_baseline

        return train_external_baseline
    if name == "evaluate_external_baselines":
        from .evaluate import evaluate_external_baselines

        return evaluate_external_baselines
    raise AttributeError(name)
