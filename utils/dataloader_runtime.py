from __future__ import annotations

"""Runtime hardening for PyTorch multi-process DataLoaders.

The project returns dictionaries containing many independent tensors per sample.
Under Linux, PyTorch's default ``file_descriptor`` sharing strategy transfers a
file descriptor for every shared CPU storage.  DDP multiplies that pressure by
one DataLoader worker pool per rank.  When the process approaches its open-file
limit, Python's multiprocessing receiver can fail while unpickling a batch with
``RuntimeError: received 0 items of ancdata``.

Switching to ``file_system`` changes only how CPU tensor storage is transported
between DataLoader workers and the training process.  It does not alter sample
order, labels, model execution, losses, gradients, or checkpoint semantics.
"""

import os
import shutil
from pathlib import Path
from typing import Any


def _open_fd_count() -> int | None:
    proc_fd = Path("/proc/self/fd")
    try:
        return len(list(proc_fd.iterdir()))
    except Exception:
        return None


def _nofile_limit() -> tuple[int | None, int | None]:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return int(soft), int(hard)
    except Exception:
        return None, None


def _shared_memory_bytes() -> tuple[int | None, int | None]:
    try:
        usage = shutil.disk_usage("/dev/shm")
        return int(usage.free), int(usage.total)
    except Exception:
        return None, None


def configure_dataloader_runtime(requested: str | None = None) -> dict[str, Any]:
    """Configure and describe PyTorch CPU-tensor sharing for DataLoader workers.

    Resolution order:

    1. explicit ``requested`` argument;
    2. ``COWP_TORCH_SHARING_STRATEGY`` environment variable;
    3. ``auto``.

    On Linux/Unix, ``auto`` prefers ``file_system`` when available because this
    repository emits many tensor storages per batch and commonly runs one worker
    pool per DDP rank.  Set the environment variable or CLI option to
    ``file_descriptor`` to restore the PyTorch default for a host with unusually
    constrained ``/dev/shm``.
    """

    import torch.multiprocessing as torch_mp

    raw = requested if requested is not None else os.environ.get("COWP_TORCH_SHARING_STRATEGY", "auto")
    choice = str(raw or "auto").strip().lower().replace("-", "_")
    if choice in {"default", "current"}:
        choice = "current"

    available = sorted(str(x) for x in torch_mp.get_all_sharing_strategies())
    before = str(torch_mp.get_sharing_strategy())

    if choice == "auto":
        if os.name == "posix" and "file_system" in available:
            selected = "file_system"
        else:
            selected = before
    elif choice == "current":
        selected = before
    elif choice in available:
        selected = choice
    else:
        raise ValueError(
            f"Unsupported torch sharing strategy {raw!r}; available={available}, "
            "special values=['auto', 'current']"
        )

    if selected != before:
        torch_mp.set_sharing_strategy(selected)
    after = str(torch_mp.get_sharing_strategy())

    soft, hard = _nofile_limit()
    shm_free, shm_total = _shared_memory_bytes()
    return {
        "requested": choice,
        "selected": after,
        "available": available,
        "open_fds": _open_fd_count(),
        "nofile_soft": soft,
        "nofile_hard": hard,
        "shm_free_bytes": shm_free,
        "shm_total_bytes": shm_total,
    }
