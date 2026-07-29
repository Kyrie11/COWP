from __future__ import annotations

import numpy as np


def normalize_future_priors(futures: list) -> None:
    priors = np.asarray([max(0.0, float(f.prior)) for f in futures], dtype=np.float64)
    if priors.sum() <= 1e-8:
        priors[:] = 1.0 / max(len(priors), 1)
    else:
        priors /= priors.sum()
    for f, p in zip(futures, priors):
        f.prior = float(p)
