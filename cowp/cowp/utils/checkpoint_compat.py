from __future__ import annotations

from collections.abc import Mapping

import torch


# v16.8.1 grows the final root-transport primitive from four rows
# (conflict, retention, uncertainty, recovery) to five rows by adding b*.
# Copying the shared leading rows preserves an old mechanism checkpoint while
# leaving only the new burden row at its current initialization.
_ROW_GROWTH_SUFFIXES = (
    "set_transport.mode_out.1.weight",
    "set_transport.mode_out.1.bias",
)


def strip_compiled_prefix(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state)
    if out and all(str(k).startswith("_orig_mod.") for k in out):
        return {str(k)[len("_orig_mod."):]: v for k, v in out.items()}
    return out


def compatible_state_dict(
    model_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Return exact-shape weights plus explicitly supported row-growth migrations.

    PyTorch ``strict=False`` still raises on tensor shape mismatch.  This helper
    keeps ordinary incompatible tensors out, and handles only the audited
    v16.8 -> v16.8.1 SetTransport output-row extension.  It intentionally does
    not perform generic slicing, which could silently corrupt unrelated layers.
    """
    compatible: dict[str, torch.Tensor] = {}
    migrated: list[str] = []
    ignored: list[str] = []
    for key, value in candidate_state.items():
        target = model_state.get(key)
        if target is None:
            ignored.append(key)
            continue
        if tuple(target.shape) == tuple(value.shape):
            compatible[key] = value
            continue
        allow_row_growth = any(str(key).endswith(suffix) for suffix in _ROW_GROWTH_SUFFIXES)
        if (
            allow_row_growth
            and target.ndim == value.ndim
            and target.ndim >= 1
            and tuple(target.shape[1:]) == tuple(value.shape[1:])
            and 0 < int(value.shape[0]) < int(target.shape[0])
        ):
            merged = target.detach().clone()
            merged[: int(value.shape[0])].copy_(value.to(device=merged.device, dtype=merged.dtype))
            compatible[key] = merged
            migrated.append(key)
            continue
        ignored.append(key)
    return compatible, migrated, ignored
