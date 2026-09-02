from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*items: Any) -> int:
    """Return a process-independent uint32 seed for arbitrary values."""
    s = "|".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest()[:8], 16)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    import random
    random.seed(int(seed))
    try:
        import numpy as np
        np.random.seed(int(seed) % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass
