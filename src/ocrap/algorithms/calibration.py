from __future__ import annotations

import numpy as np

from .lcv import finite_sample_upper_quantile


def calibrate_threshold(pred_r_dep: np.ndarray, teacher_r_dep: np.ndarray, delta: float, numerical_margin: float = 0.0, strict: bool = True, required_min: int = 100) -> tuple[float, list[str]]:
    pred = np.asarray(pred_r_dep, dtype=np.float64).reshape(-1)
    teacher = np.asarray(teacher_r_dep, dtype=np.float64).reshape(-1)
    neg = pred[teacher < 0]
    warnings: list[str] = []
    if neg.size < required_min:
        warnings.append(f"num_negative < required_min_for_delta ({neg.size} < {required_min})")
    gamma = finite_sample_upper_quantile(neg, delta, numerical_margin=numerical_margin, strict=strict)
    return gamma, warnings
